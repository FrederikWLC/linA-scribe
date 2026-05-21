import cv2
from optuna import Trial
from config import config
from scribe.auto_prompts import auto_points
from scribe.base import PointScribe
from scribe.tunable import BILATERAL_SPECS, BilateralTunable, HyperparameterSpec, TunableConfiguration
import numpy as np
from scribe.prompts import BoxPrompt, PointPrompt, PointPromptList, Prompt, build_box_point_prompts
from scribe.prompts import BoxPointPrompt
from scribe.binary_mask import BinaryMask
from scribe.baselines.gaussian import GAUSSIAN_SPECS, Gaussian, build_gaussian
from config import config

# Model presets (for factory initialization of main stream SAM variants)
SAM_PRESETS = {
    "SAM2": {
        "CHECKPOINT_PATH": config.SAM2_CHECKPOINT_PATH,
        "CONFIG": config.SAM2_CONFIG,
        "ABBREVIATION": "SAM2",
        "ABBREVIATION_SHORT": "S2"
    },
    "SAM": {
        "VIT_TYPE": config.SAM_VIT_TYPE, 
        "CHECKPOINT_PATH": config.SAM_CHECKPOINT_PATH,
        "ABBREVIATION": "SAM",
        "ABBREVIATION_SHORT": "S"},
    "MobileSAM": {
        "VIT_TYPE": config.MOBILESAM_VIT_TYPE,
        "CHECKPOINT_PATH": config.MOBILESAM_CHECKPOINT_PATH,
        "ABBREVIATION": "mSAM",
        "ABBREVIATION_SHORT": "M"
    }
}

AUTOPOINT_SPECS = GAUSSIAN_SPECS + [
    HyperparameterSpec("n_fgd_points", default=1000, suggest=lambda trial:
        trial.suggest_int("n_fgd_points", 1, 2000)),
    HyperparameterSpec("n_bgd_points", default=1000, suggest=lambda trial:
        trial.suggest_int("n_bgd_points", 1, 2000)),
    HyperparameterSpec("d_gap_erosion", default=3, suggest=lambda trial:
        trial.suggest_categorical("d_gap_erosion", [i * 2 + 1 for i in range(1, 11)]))
]

class SAMConfiguration(TunableConfiguration):
    def __init__(self, sam_type, use_bilateral_filter, use_autopoints):
        self.sam_type = sam_type
        self.checkpoint_path = SAM_PRESETS[sam_type]["CHECKPOINT_PATH"]
        self.vit_type = SAM_PRESETS[sam_type].get("VIT_TYPE", None) # only for SAM and MobileSAM, not for SAM2
        self.cfg = SAM_PRESETS[sam_type].get("CONFIG", None) # only for SAM2, not for SAM and MobileSAM

        self.use_bilateral_filter = use_bilateral_filter
        self.use_autopoints = use_autopoints

        self.abbreviation = (SAM_PRESETS[sam_type]["ABBREVIATION"] + "+pts") if use_autopoints else SAM_PRESETS[sam_type]["ABBREVIATION"]
        self.abbreviation_short = (SAM_PRESETS[sam_type]["ABBREVIATION_SHORT"] + "+p") if use_autopoints else SAM_PRESETS[sam_type]["ABBREVIATION_SHORT"]

        self.filter_str = "bilateral-filter" if use_bilateral_filter else "no-filter"
        self.name = f"{self.abbreviation}-{self.filter_str}"
        self.short_name = f"{self.abbreviation_short}-{self.filter_str[0]}"

        specs = AUTOPOINT_SPECS if use_autopoints else BILATERAL_SPECS if use_bilateral_filter else []
        
        super().__init__(name=self.name, short_name=self.short_name, hyperparameter_specs=specs)


    def get_sam_predictor(self):
        import torch 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.sam_type == "SAM2":
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_model = build_sam2(
                self.cfg,
                self.checkpoint_path,
                device=device)
            sam2_model.to(device)
            return SAM2ImagePredictor(sam2_model)
        elif self.sam_type == "SAM": # default to original sam
            from segment_anything import sam_model_registry, SamPredictor
            sam_model = sam_model_registry[self.vit_type](
                checkpoint=str(self.checkpoint_path)
            )
            sam_model.to(device)
            return SamPredictor(sam_model)
        elif self.sam_type == "MobileSAM":
            from mobile_sam import sam_model_registry, SamPredictor
            mobilesam_model = sam_model_registry[self.vit_type](
                checkpoint=str(self.checkpoint_path)
            )
            mobilesam_model.to(device)
            return SamPredictor(mobilesam_model)
    
# Basic mask decoding and prompt logic
class SAMCore(PointScribe):

    # decode the mask
    def decode_mask(self, prompts: tuple[list[BoxPrompt], PointPromptList] | None = None) -> list[BinaryMask]:
        box_prompts, point_prompt_list = prompts if prompts else ([], PointPromptList([]))
        if box_prompts: # interactive style multi-run decoding per box prompt
            box_point_prompts = build_box_point_prompts(box_prompts, point_prompt_list)
            mask = self.decode_mask_per_box_with_points(box_point_prompts)
        else: # non-interactive style single-run decoding without box prompts
            points, labels = point_prompt_list.to_arrays()
            mask = self.decode_mask_single(points=points, labels=labels)
        return mask
    
    # this is for the interactive SAM variant that makes use of several box prompts possible
    def decode_mask_per_box_with_points(self, box_point_prompts: list[BoxPointPrompt]) -> BinaryMask:
        masks = []
        for box_point_prompt in box_point_prompts:
            box, points, labels = box_point_prompt.to_arrays()
            mask = self.decode_mask_single(box=box, points=points, labels=labels)
            masks.append(mask)
        return BinaryMask.from_union(*masks)
    
    # decode the masks with single run, given optional point prompts (and up to one box prompt)
    def decode_mask_single(self,
        box: np.ndarray = None,
        points: np.ndarray = None,
        labels: np.ndarray  = None) -> list[BinaryMask]:
        pass

class SAM(SAMCore, BilateralTunable):

    def __init__(self, configuration: SAMConfiguration,
                   **kwargs):
        super().__init__(configuration,**kwargs) # allow for extensions to call
        self.predictor = configuration.get_sam_predictor()
        self._image_set = False
    
    # decode the masks with single run, given optional point prompts (and up to one box prompt)
    def decode_mask_single(
        self,
        box: np.ndarray = None,
        points: np.ndarray = None,
        labels: np.ndarray  = None
    ) -> list[BinaryMask]:
        # now, running the mask decoder...
        # only the fourth unambigous output token is processed and returned as mask
        mask, _score, _logit = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=False
            )
        return BinaryMask(np.squeeze(mask))
    
    # prompts given are tuple of (box prompts, point prompts) as that's the easiest
    def segment(self, image: np.ndarray, prompts: tuple[list[BoxPrompt], PointPromptList] | None = None) -> BinaryMask:
        if image is not None:
            # runs the ViT for the image and saves the image embedding
            self.set_image(image,False) # already preprocessed in predict() 
        elif not self.has_image():
            raise RuntimeError("No image provided and no image is set. Call set_image(image) first.")
        return self.decode_mask(prompts)
    
    def set_image(self, image: np.ndarray, preprocess: bool = True) -> None:
        image = self.preprocess(image) if preprocess else image # to not preprocess twice if called from segment()
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # convert to rgb for sam
        self.predictor.set_image(image)
        self._image_set = True

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if self.configuration.use_bilateral_filter:
            return super().preprocess(image)
        else:
            return image
    
    def has_image(self) -> bool:
        return self._image_set
    
class SAMAutopoint(SAM):
    """SAM with automatic point prompts."""

    def autoprompt(self, image: np.ndarray) -> tuple[None, PointPromptList]:

        thresh = build_gaussian().set_hyperparameters_from(**self.hyperparameter_values).predict(image)

        n_fgd_points = int(self.configuration.get_value("n_fgd_points"))
        n_bgd_points = int(self.configuration.get_value("n_bgd_points"))
        d_gap_erosion = int(self.configuration.get_value("d_gap_erosion"))

        prompts = None, auto_points(thresh, n_fgd_points, n_bgd_points, d_gap_erosion) # no box prompts, only point prompts from autoseeding
        return prompts


# ============================================================================
# The 12 Different SAM variants used
# Combinations of:
# - SAM2 (2.1 Hiera Large) vs. SAM (VitH) vs. MobileSAMv2 (TinyVit)
# - Bilateral filter vs no filter
# - Autoprompts vs no autoprompts (i.e. manual point prompts)
# ============================================================================


def build_from_sam_configuration(sam_configuration: SAMConfiguration):
    if sam_configuration.use_autopoints:
        return SAMAutopoint(configuration=sam_configuration)
    return SAM(configuration=sam_configuration)

def build_sam_variant(sam_type, use_bilateral_filter, use_autopoints):
    configuration = SAMConfiguration(sam_type, use_bilateral_filter, use_autopoints)
    return build_from_sam_configuration(configuration)

def get_all_sam_configurations():
    configurations = []
    for sam_type in SAM_PRESETS.keys(): # 3 *
        for use_bilateral_filter in [False, True]: # 2 *
            for use_autopoints in [False, True]: # 2 = 12 total configurations
                configuration = SAMConfiguration(sam_type, use_bilateral_filter, use_autopoints)
                configurations.append(configuration)
    return configurations

def get_all_tunable_sam_configurations():
    configurations = get_all_sam_configurations()
    tunable_configurations = [config for config in configurations if config.is_tunable()]
    return tunable_configurations

# ======================================================================
# The Best SAM variant configuration found in the RQ1 evaluation
# ======================================================================

def get_best_sam_configuration():
    return SAMConfiguration("SAM", use_bilateral_filter=False, use_autopoints=True)
