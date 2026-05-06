import cv2
import numpy as np
from optuna import Trial
from scribe.base import BrushScribe
from scribe.tunable import BilateralTunable, HyperparameterSpec, TunableConfiguration
from scribe.auto_prompts import auto_brush
from scribe.binary_mask import BinaryMask
from scribe.prompts import BrushPrompt
from scribe.baselines.gaussian import GAUSSIAN_SPECS, Gaussian, build_gaussian

GRABCUT_SPECS = GAUSSIAN_SPECS + [
    HyperparameterSpec("d_prb_erosion", default=3, suggest=lambda trial: trial.suggest_categorical("d_prb_erosion", [i * 2 + 1 for i in range(1,11)])), # odd integers from 3 to 21
    HyperparameterSpec("iters", default=1, suggest=lambda trial: trial.suggest_int("iters", 1, 15))
]

class GrabCutConfiguration(TunableConfiguration):
    def __init__(self):
        super().__init__(
            name="GrabCutAutoBrush",
            short_name="GC+brush",
            hyperparameter_specs=GRABCUT_SPECS
        )

# implementation of GrabCut with automatic brush mask given as prompt
class GrabCutAutoBrush(BrushScribe, BilateralTunable):

    configuration: GrabCutConfiguration

    def segment(self, image: np.ndarray, brushmask=None) -> BinaryMask:
        if len(image.shape) == 2:  # convert to bgr
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if brushmask is None:
            brushmask = np.zeros(image.shape[:2], np.uint8)

        # 65 is the number of GMM components used internally by GrabCut, must be 65 for OpenCV's implementation
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        try:
            iters = int(self.configuration.get_value("iters"))
            brushmask, bgdModel, fgdModel = cv2.grabCut(image,brushmask,None,bgdModel,fgdModel,iters,cv2.GC_INIT_WITH_MASK)
            # end result is union of sure foreground and probable foreground
            is_fgd = (brushmask == cv2.GC_FGD) | (brushmask == cv2.GC_PR_FGD)
        except cv2.error: # if not fgd or bgd pixels provided
            is_fgd = np.zeros(image.shape[:2], dtype=bool)

        return BinaryMask.from_bool(is_fgd)
    
    def autoprompt(self, image: np.ndarray) -> list[BrushPrompt]:

        thresh = build_gaussian().set_hyperparameters_from(**self.hyperparameter_values).predict(image)
        d_prb_erosion = int(self.configuration.get_value("d_prb_erosion"))
        brush_mask = auto_brush(
                                thresh,
                                d_prb_erosion=d_prb_erosion)
        return brush_mask
    
def build_grabcut():
    return GrabCutAutoBrush(GrabCutConfiguration())
