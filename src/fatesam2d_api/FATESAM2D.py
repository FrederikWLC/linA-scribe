import copy
import cv2
import numpy as np
from config import config
from fatesam2d_api.configuration import FATESAM2DConfiguration
from fatesam2d_api.predictor import prepare_inference_state, run_from_inference_state
from fatesam2d_api.tensor_handling import images_to_tensor, labels_to_tensor
from fatesam2d_api.sam2.build_sam import build_sam2_video_predictor, build_sam2_video_predictor_fate, device_setup
from scribe.auto_prompts import auto_points
from scribe.baselines.gaussian import build_gaussian
from scribe.binary_mask import BinaryMask
from scribe.prompts import PointPrompt
from scribe.tunable import BilateralTunable, TunableConfiguration
from scribe.baselines.sam import SAMCore


class FATESAM2D(SAMCore):
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
        
    def decode_mask_single(self, box=None, points=None, labels=None) -> BinaryMask:
        frame_pred = run_from_inference_state(
            sam2_predictor_fate=self.sam2_predictor_fate,
            inference_state=copy.deepcopy(self.inference_state), # we copy the inference state to avoid in-place modifications that could affect future predictions with the same image
            similarity_results=copy.deepcopy(self.similarity_results), # same goes for the similarity results
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
