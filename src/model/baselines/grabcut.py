import cv2
import numpy as np
from optuna import Trial
from model.scribe import BrushScribe, Tunable
from utils.auto_prompts import auto_brush
from utils.binary_mask import BinaryMask
from utils.seeds import BrushSeed
from model.baselines.gaussian import Gaussian

# implementation of GrabCut with automatic brush mask given as seed
class GrabCutAutoBrush(BrushScribe,Tunable):
    def __init__(self, iters=1, d_bilateral=15, sigma=75, C=5, d_gaussian=19, d_erosion=12):
        self.iters = int(iters)
        self.d_bilateral = int(d_bilateral)
        self.sigma = int(sigma)
        self.C = int(C)
        self.d_gaussian = int(d_gaussian)
        self.d_erosion = int(d_erosion)

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
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        d_bilateral = int(self.d_bilateral)
        sigma_color = int(self.sigma)
        sigma_space = int(self.sigma)
        return cv2.bilateralFilter(image, d=d_bilateral, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    def autoseed(self, image: np.ndarray) -> list[BrushSeed]:
        d_bilateral = int(self.d_bilateral)
        sigma = int(self.sigma)
        C = int(self.C)
        d_gaussian = int(self.d_gaussian)
        d_erosion = int(self.d_erosion)

        thresh = Gaussian(
            C=C,
            d_gaussian=d_gaussian,
            d_bilateral=d_bilateral,
            sigma=sigma
            ).predict(image)
        
        brush_mask = auto_brush(
                                thresh=thresh,
                                d_erosion=d_erosion)
        return brush_mask

    
    @property
    def hyperparameters(self) -> dict:
        return {
                # GrabCut iters
                "iters":int(self.iters),

                # Bilateral filter hyperparameters
                "d_bilateral":int(self.d_bilateral),
                "sigma":int(self.sigma),

                # General Gaussian hyperparameters
                "C":int(self.C),
                
                "d_gaussian":int(self.d_gaussian),
                
                # Erosion kernel size for erosion of fgd/bgd, used for determining size and placement of prb bgd regions
                "d_erosion":int(self.d_erosion)
                }
    
    def hyperparameter_ranges(self,trial: Trial) -> dict:
        return {
                # GrabCut iters
                "iters":trial.suggest_int("iters", 1, 15),
                
                # Bilateral filter hyperparameters
                "d_bilateral":trial.suggest_int("d_bilateral", 3, 15),
                "sigma":trial.suggest_int("sigma", 0, 100),
                
                # General Gaussian hyperparameters
                "C":trial.suggest_int("C", 0, 10),
                
                # Specific Gaussian hyperparameters
                "d_gaussian":trial.suggest_categorical("d_gaussian", [i * 2 + 1 for i in range(1,11)]), # odd integers from 3 to 21
                
                # Erosion kernel size for erosion of fgd/bgd, used for determining size and placement of prb bgd regions
                "d_erosion":trial.suggest_categorical("d_erosion", [i * 2 + 1 for i in range(1,11)]) # odd integers from 3 to 21
                }
    
    @property
    def name(self):
        return "GC+brush"
    
