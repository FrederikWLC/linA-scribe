import cv2
import numpy as np
from optuna import Trial
from scribe.base import BrushScribe
from scribe.tunable import BilateralTunable
from evaluation.utils.auto_prompts import auto_brush
from scribe.binary_mask import BinaryMask
from scribe.prompts import BrushPrompt
from evaluation.baselines.gaussian import Gaussian

# implementation of GrabCut with automatic brush mask given as prompt
class GrabCutAutoBrush(BrushScribe, BilateralTunable):
    def __init__(self, iters=1, d_bilateral=15, sigma_bilateral=75, C=5, d_gaussian=21, d_prb_erosion=12):
        self.iters = int(iters)
        self.C = int(C)
        self.d_gaussian = int(d_gaussian)
        self.d_prb_erosion = int(d_prb_erosion)

        super().__init__(d_bilateral=d_bilateral, sigma_bilateral=sigma_bilateral)

    def segment(self, image: np.ndarray, brushmask=None) -> BinaryMask:
        if len(image.shape) == 2:  # convert to bgr
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if brushmask is None:
            brushmask = np.zeros(image.shape[:2], np.uint8)

        # 65 is the number of GMM components used internally by GrabCut, must be 65 for OpenCV's implementation
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        try:
            iters = int(self.iters)
            brushmask, bgdModel, fgdModel = cv2.grabCut(image,brushmask,None,bgdModel,fgdModel,iters,cv2.GC_INIT_WITH_MASK)
            # end result is union of sure foreground and probable foreground
            is_fgd = (brushmask == cv2.GC_FGD) | (brushmask == cv2.GC_PR_FGD)
        except cv2.error: # if not fgd or bgd pixels provided
            is_fgd = np.zeros(image.shape[:2], dtype=bool)

        return BinaryMask.from_bool(is_fgd)
    
    def autoprompt(self, image: np.ndarray) -> list[BrushPrompt]:
        d_bilateral = int(self.d_bilateral)
        sigma_bilateral = int(self.sigma_bilateral)
        C = int(self.C)
        d_gaussian = int(self.d_gaussian)
        # constrain bgd gaussian to have bigger kernel size than fgd gaussian,
        # to ensure that sure bgd thresh is more loose than sure fgd
        # this will lead to bigger sure bgd regions, and thus less non-sure-bgd noise, 
        # which seems to be beneficial for performance of GrabCut (see tuning-GC+brush-trials-prev.csv)
        d_prb_erosion = int(self.d_prb_erosion)
        
        thresh = Gaussian(
            C=C,
            d_gaussian=d_gaussian,
            d_bilateral=d_bilateral,
            sigma_bilateral=sigma_bilateral
        ).predict(image)

        brush_mask = auto_brush(
                                thresh,
                                d_prb_erosion=d_prb_erosion)
        return brush_mask

    
    @property
    def hyperparameters(self) -> dict:
        return super().hyperparameters | {
                # GrabCut iters
                "iters":int(self.iters),

                # General Gaussian hyperparameters
                "C":int(self.C),
                
                # Specific Gaussian hyperparameters for probable foreground and sure foreground (used for autoseeding of brushes)
                "d_gaussian":int(self.d_gaussian),
                
                # Erosion kernel size for erosion of fgd/bgd, used for determining size and placement of prb bgd regions
                "d_prb_erosion":int(self.d_prb_erosion)
                }
    
    def hyperparameter_ranges(self,trial: Trial) -> dict:
        return super().hyperparameter_ranges(trial) | {
                # GrabCut iters
                "iters":trial.suggest_int("iters", 1, 15),
                                
                # General Gaussian hyperparameters
                "C":trial.suggest_int("C", 0, 10),
                
                # Specific Gaussian hyperparameters for probable foreground and sure foreground (used for autoseeding of brushes)
                "d_gaussian":trial.suggest_categorical("d_gaussian", [i * 2 + 1 for i in range(1,20)]), # odd integers from 1 to 41
                
                # Erosion kernel size for erosion of fgd/bgd, used for determining size and placement of prb bgd regions
                "d_prb_erosion":trial.suggest_categorical("d_prb_erosion", [i * 2 + 1 for i in range(1,11)]) # odd integers from 3 to 21
                }
    
    @property
    def name(self):
        return "GC+brush"
    
