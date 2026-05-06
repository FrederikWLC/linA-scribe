import copy
import cv2
import numpy as np
from config import config
from data.split import get_support_data
from data.split import get_support_data
from fatesam2d_api.predictor import prepare_inference_state, run_from_inference_state
from fatesam2d_api.tensor_handling import images_to_tensor, labels_to_tensor
from fatesam2d_api.sam2.build_sam import build_sam2_video_predictor, build_sam2_video_predictor_fate, device_setup
from scribe.auto_prompts import auto_points
from scribe.base import PointScribe
from scribe.baselines.gaussian import build_gaussian
from scribe.baselines.sam import AUTOPOINT_SPECS
from scribe.binary_mask import BinaryMask
from scribe.prompts import PointPrompt, PointPromptList
from scribe.tunable import BilateralTunable, TunableConfiguration

class FATESAM2DConfiguration(TunableConfiguration):
    def __init__(self, support_data_root=config.DATA_DIR, top_n_supports=3, use_autopoints=False, is_blank=False):
        
        self.support_data_root = support_data_root
        self.top_n_supports = top_n_supports
        self.use_autopoints = use_autopoints
        self.is_blank = is_blank

        self.checkpoint_path = config.FATESAM_CHECKPOINT_PATH
        self.config_file = config.FATESAM_CONFIG

        name = "FATESAM2D+pts" if use_autopoints else "FATESAM2DBlank" if is_blank else "FATESAM2D"

        short_name = "FATE+p" if use_autopoints else "FATEbl" if  is_blank else "FATE"

        hyperparameter_specs = AUTOPOINT_SPECS if use_autopoints else []

        super().__init__(
            name=name,
            short_name=short_name,
            hyperparameter_specs=hyperparameter_specs
        )

    def get_support_data(self):
        support_images, support_labels, _ = get_support_data(data_root=self.support_data_root)
        if self.is_blank: # We give blank supports to disable the few-shot effect.
            rng = np.random.default_rng(42)
            support_labels = [
                np.where(
                    rng.random(label.shape) < 0.05,
                    0,
                    255,
                ).astype(np.uint8)
                for label in support_labels
            ]
        return support_images, support_labels
class FATESAM2D(PointScribe):
    """2D pseudo-sequence adaptation of FATE-SAM.

    The query image is frame 0 and the selected support images are appended as
    pseudo-video frames. Support masks are injected on those support frames, and
    optional point prompts can be injected on the query frame.
    """

    def __init__(self, configuration: TunableConfiguration,
                 **kwargs):
        super().__init__(configuration=configuration, **kwargs)
        support_images, support_labels = configuration.get_support_data()
        self.support_images = images_to_tensor(support_images, image_size=1024)
        self.support_labels = labels_to_tensor(support_labels, image_size=1024)
        self.top_n_supports = int(configuration.top_n_supports)
        self.inference_state = None
        self.similarity_results = None
        self._output_hw = None

        self.sam2_predictor = build_sam2_video_predictor(
            config_file=configuration.config_file,
            ckpt_path=str(configuration.checkpoint_path),
            device=device_setup(),
        )

        self.sam2_predictor_fate = build_sam2_video_predictor_fate(
            config_file=configuration.config_file,
            ckpt_path=str(configuration.checkpoint_path),
            device=device_setup(),
        )


    def hasImage(self) -> bool:
        return self.inference_state is not None

    def setImage(self, image):
        image = np.asarray(image)
        self.inference_state, self.similarity_results = prepare_inference_state(
            sam2_predictor=self.sam2_predictor,
            query_image=image,
            support_images=self.support_images,
            support_labels=self.support_labels,
            top_n=self.top_n_supports,
        )
        self._output_hw = image.shape[:2]
        return self

    def decode_mask(self, prompts: tuple[list[PointPrompt], PointPromptList]) -> BinaryMask:
        if not self.hasImage():
            raise RuntimeError("No image is set. Call setImage(image) before decode_mask(...).")
        _, point_prompt_list = prompts if prompts else (None, None)
        points, labels = point_prompt_list.to_arrays() if point_prompt_list else (None, None)
        frame_pred = run_from_inference_state(
            sam2_predictor_fate=self.sam2_predictor_fate,
            inference_state=copy.deepcopy(self.inference_state),
            similarity_results=copy.deepcopy(self.similarity_results),
            points=points,
            labels=labels,
            prompt_input_hw=self._output_hw,
        )
        return self._merge_frame_prediction(frame_pred)

    def segment(self, image=None, prompts=None) -> BinaryMask:
        if image is not None:
            self.setImage(image)
        elif not self.hasImage():
            raise RuntimeError("No image provided and no image is set. Call setImage(image) first.")
        return self.decode_mask(prompts=prompts)
    

    def _merge_frame_prediction(self, frame_pred) -> BinaryMask:
        "Merges the predicted masks (for each object) of the frame into one"
        height, width = self._output_hw
        if not frame_pred:
            return BinaryMask.from_bool(np.zeros((height, width), dtype=np.uint8))

        merged = np.zeros_like(next(iter(frame_pred.values())).squeeze(), dtype=np.uint8)
        for obj_id, obj_mask in frame_pred.items():
            merged[np.squeeze(obj_mask) > 0] = np.uint8(obj_id)

        if tuple(merged.shape[:2]) != tuple(self._output_hw):
            merged = cv2.resize(merged, (width, height), interpolation=cv2.INTER_NEAREST)

        return BinaryMask.from_bool(merged > 0)


class FATESAM2DAutoPoint(FATESAM2D,BilateralTunable):
    """FATESAM2D with automatic point prompts."""

    def autoprompt(self, image: np.ndarray) -> list[PointPrompt]:
        d_bilateral = int(self.configuration.get_value("d_bilateral"))
        sigma_bilateral = int(self.configuration.get_value("sigma_bilateral"))
        C = int(self.configuration.get_value("C"))
        d_gaussian = int(self.configuration.get_value("d_gaussian"))
        n_fgd_points = int(self.configuration.get_value("n_fgd_points"))
        n_bgd_points = int(self.configuration.get_value("n_bgd_points"))
        d_gap_erosion = int(self.configuration.get_value("d_gap_erosion"))

        thresh = build_gaussian().set_hyperparameters(
            d_bilateral=d_bilateral,
            sigma_bilateral=sigma_bilateral,
            C=C,
            d_gaussian=d_gaussian
        ).predict(image)
        
        points = auto_points(thresh, n_fgd_points, n_bgd_points, d_gap_erosion)
        return None, points # only points, no box prompts
    
    # no filter
    def preprocess(self, image):
        return image

def build_from_fatesam_configuration(configuration: TunableConfiguration) -> FATESAM2D:
    if configuration.use_autopoints:
        return FATESAM2DAutoPoint(configuration=configuration)
    else:
        return FATESAM2D(configuration=configuration)
    
def build_fatesam2d(support_data_root: str = config.DATA_DIR, top_n_supports: int = 3, use_autopoints: bool = False, is_blank: bool = False) -> FATESAM2D:
    configuration = FATESAM2DConfiguration(
        support_data_root=support_data_root,
        top_n_supports=top_n_supports,
        use_autopoints=use_autopoints,
        is_blank=is_blank
    )
    return build_from_fatesam_configuration(configuration)

def get_default_fatesam2d_configuration(support_data_root: str = config.DATA_DIR) -> FATESAM2DConfiguration:
    return FATESAM2DConfiguration(support_data_root=support_data_root, top_n_supports=3, use_autopoints=False, is_blank=False)

def get_all_fatesam2d_configurations(data_root: str = config.DATA_DIR) -> list[FATESAM2DConfiguration]:
    return [
        FATESAM2DConfiguration(support_data_root=data_root, top_n_supports=3, use_autopoints=False, is_blank=False),
        FATESAM2DConfiguration(support_data_root=data_root, top_n_supports=3, use_autopoints=True, is_blank=False),
        FATESAM2DConfiguration(support_data_root=data_root, top_n_supports=3, use_autopoints=False, is_blank=True)
    ]

def get_all_tunable_fatesam2d_configurations(data_root: str = config.DATA_DIR) -> list[FATESAM2DConfiguration]:
    return [conf for conf in get_all_fatesam2d_configurations(data_root=data_root) if conf.is_tunable()]
