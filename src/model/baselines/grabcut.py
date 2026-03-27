import cv2
import numpy as np
from optuna import Trial
from model.scribe import SeedableScribe, Tunable
from utils.auto_prompts import auto_brushes
from utils.binary_mask import BinaryMask
from utils.seeds import BrushSeed
from model.baselines.gaussian import Gaussian

# implementation of GrabCut
class GrabCut(SeedableScribe,Tunable):
    def __init__(self, iters=1):
        self.iters = iters

    def segment(self, image: np.ndarray, brushseeds=None) -> BinaryMask:
        if len(image.shape) == 2:  # convert to bgr
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = np.zeros(image.shape[:2],np.uint8)
        
        mask[True] = cv2.GC_PR_BGD # default is probable background
        
        if brushseeds:
            for brush in brushseeds:
                if len(brush.pixels) > 0:
                    xs, ys = zip(*brush.pixels)
                    mask[ys, xs] = brush.label

        # 65 is the number of GMM components used internally by GrabCut, must be 65 for OpenCV's implementation
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        try:
            mask, bgdModel, fgdModel = cv2.grabCut(image,mask,None,bgdModel,fgdModel,self.iters,cv2.GC_INIT_WITH_MASK)
        except cv2.error: # if not fgd or bgd pixels provided
           pass

        # end result is union of sure foreground and probable foreground
        is_fgd = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        return BinaryMask.from_bool(is_fgd)
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        d_bilateral = self.d_bilateral
        sigma_color = self.sigma
        sigma_space = self.sigma
        return cv2.bilateralFilter(image, d=d_bilateral, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    @property
    def hyperparameters(self) -> dict:
        return {
                "iters":self.iters,
                "d_bilateral":self.d_bilateral,
                "sigma":self.sigma
                }
    
    def hyperparameter_ranges(self,trial: Trial) -> dict:
        return {
                "iters":trial.suggest_int("iters", 1, 15),
                "d_bilateral":trial.suggest_int("d_bilateral", 1, 15),
                "sigma":trial.suggest_int("sigma", 0, 150)
                }

# implementation of GrabCut with automatic brushes given as seeds
class GrabCutAutoBrush(GrabCut):
    def __init__(self, iters=10, d_bilateral=15, sigma=75, C=5, d_prb_gaussian=19, extra_d_sure_gaussian=15, d_sure_bgd_erosion=3, d_prb_fgd_erosion=3, d_sure_fgd_erosion=3):
        super().__init__(iters)
        self.d_bilateral = d_bilateral
        self.sigma = sigma
        self.C = C
        self.d_prb_gaussian = d_prb_gaussian
        self.extra_d_sure_gaussian = extra_d_sure_gaussian
        self.d_sure_bgd_erosion = d_sure_bgd_erosion
        self.d_prb_fgd_erosion = d_prb_fgd_erosion
        self.d_sure_fgd_erosion = d_sure_fgd_erosion

    def autoseed(self, image: np.ndarray) -> list[BrushSeed]:
        d_bilateral = self.d_bilateral
        sigma = self.sigma
        C = self.C
        d_prb_gaussian = self.d_prb_gaussian
        d_sure_gaussian = self.extra_d_sure_gaussian + d_prb_gaussian
        d_sure_bgd_erosion = self.d_sure_bgd_erosion
        d_prb_fgd_erosion = self.d_prb_fgd_erosion
        d_sure_fgd_erosion = self.d_sure_fgd_erosion

        prb_fgd_thresh = Gaussian(C,d_prb_gaussian, d_bilateral,sigma).predict(image)
        sure_fgd_thresh = Gaussian(C,d_sure_gaussian, d_bilateral,sigma).predict(image)
        brushes = auto_brushes(prb_fgd_thresh,sure_fgd_thresh,d_sure_bgd_erosion,d_prb_fgd_erosion,d_sure_fgd_erosion)
        return brushes

    @property
    def name(self):
        return "GC+brush"
    
    @property
    def hyperparameters(self) -> dict:
        return super().hyperparameters | {
                # General Gaussian hyperparameters
                "C":self.C,
                # Specific Gaussian hyperparameters for probable foreground and sure foreground (used for autoseeding of brushes)
                "extra_d_sure_gaussian":self.extra_d_sure_gaussian,
                "d_prb_gaussian":self.d_prb_gaussian,

                # Kernel sizes for erosion of different mask regions
                "d_sure_bgd_erosion":self.d_sure_bgd_erosion,
                "d_prb_fgd_erosion":self.d_prb_fgd_erosion,
                "d_sure_fgd_erosion":self.d_sure_fgd_erosion
                }
    
    def hyperparameter_ranges(self,trial: Trial) -> dict:
        return super().hyperparameter_ranges(trial) | {
                # General Gaussian hyperparameters
                "C":trial.suggest_int("C", 0, 10),
                # Specific Gaussian hyperparameters for probable foreground and sure foreground (used for autoseeding of brushes)
                "extra_d_sure_gaussian":trial.suggest_categorical("extra_d_sure_gaussian", [i * 2 for i in range(1,11)]), # even integers from 0 to 20
                "d_prb_gaussian":trial.suggest_categorical("d_prb_gaussian", [i * 2 + 1 for i in range(1,11)]), # odd integers from 3 to 21

                # Kernel sizes for erosion of different mask regions
                "d_sure_bgd_erosion":trial.suggest_categorical("d_sure_bgd_erosion", [i * 2 + 1 for i in range(1,11)]), # odd integers from 3 to 21
                "d_prb_fgd_erosion":trial.suggest_categorical("d_prb_fgd_erosion", [i * 2 + 1 for i in range(1,11)]), # odd integers from 3 to 21
                "d_sure_fgd_erosion":trial.suggest_categorical("d_sure_fgd_erosion", [i * 2 + 1 for i in range(1,11)]) # odd integers from 3 to 21
                }
