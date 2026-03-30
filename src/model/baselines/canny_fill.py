from optuna import Trial
from model.scribe import Scribe, Tunable
import cv2
import numpy as np
from utils.binary_mask import BinaryMask


class CannyFill(Scribe,Tunable):
    def __init__(self, d=15,sigma=0.1):
        self.d = d
        self.sigma = sigma


    def segment(self, image: np.ndarray) -> BinaryMask:

        sigma = float(self.sigma)
        d = int(self.d)

        # auto canny thresholds
        v = np.median(image)
        low = int(max(0, (1 - sigma) * v))
        high = int(min(255, (1 + sigma) * v))

        # canny edge detection
        edges = cv2.Canny(image, low, high)

        # filling up the edges via morphological closing (dilation followed by erosion)
        kernel = np.ones((d, d), np.uint8)
        filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return BinaryMask(filled)
    
    def preprocess(self, image):
        return image
        
    @property
    def name(self):
        return "CannyFill"
    
    @property
    def hyperparameters(self):
        return {
            # Kernel size for morphological closing (filling)
            "d":int(self.d),
            # Sigma for auto Canny thresholding (determines how much the thresholds deviate from the median pixel intensity)
            "sigma":float(self.sigma)
            }

    def hyperparameter_ranges(self,trial: Trial):
        return {
            "d":trial.suggest_categorical("d", [i * 2 + 1 for i in range(1,16)]), # odd integers from 3 to 31
            "sigma":trial.suggest_float("sigma", 0, 1)
            }