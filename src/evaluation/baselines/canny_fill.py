from optuna import Trial
from scribe.base import BilateralTunable, Scribe
import cv2
import numpy as np
from scribe.binary_mask import BinaryMask


class CannyFill(Scribe, BilateralTunable):
    def __init__(self, d_closing=15, sigma_canny=0.1, d_bilateral=15, sigma_bilateral=75):
        self.d_closing = int(d_closing)
        self.sigma_canny = float(sigma_canny)
        super().__init__(d_bilateral=d_bilateral, sigma_bilateral=sigma_bilateral)


    def segment(self, image: np.ndarray) -> BinaryMask:

        sigma = float(self.sigma_canny)
        d = int(self.d_closing)

        # auto canny thresholds
        v = np.median(image)
        low = int(max(0, (1 - sigma) * v))
        high = int(min(255, (1 + sigma) * v))

        # canny edge detection
        edges = cv2.Canny(image, low, high)

        # filling up the edges via morphological closing (dilation followed by erosion)
        # using elliptical kernel to avoid blocky look
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(d, d))
        filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return BinaryMask(filled)
    
    @property
    def name(self):
        return "CannyFill"
    
    @property
    def hyperparameters(self):
        return super().hyperparameters | {
            # Kernel size for morphological closing (filling)
            "d_closing":int(self.d_closing),
            # Sigma for auto Canny thresholding (determines how much the thresholds deviate from the median pixel intensity)
            "sigma_canny":float(self.sigma_canny)
            }

    def hyperparameter_ranges(self,trial: Trial):
        return super().hyperparameter_ranges(trial) | {
            "d_closing":trial.suggest_categorical("d_closing", [i * 2 + 1 for i in range(1,16)]), # odd integers from 3 to 31
            "sigma_canny":trial.suggest_float("sigma_canny", 0, 1)
            }
