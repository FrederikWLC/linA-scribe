from model.scribe import BilateralTunable, Scribe
import numpy as np
import cv2
from utils.binary_mask import BinaryMask
from optuna import Trial

# implementation of the Gaussian threshold
class Gaussian(Scribe, BilateralTunable):
    def __init__(self, C=5, d_gaussian=19, d_bilateral=15, sigma_bilateral=75):
        self.d_gaussian = int(d_gaussian)
        self.C = int(C)
        super().__init__(d_bilateral=d_bilateral, sigma_bilateral=sigma_bilateral)

    def segment(self, image: np.ndarray) -> BinaryMask:
        kernel_size = int(self.d_gaussian) # size of kernel (neighborhood)
        C = int(self.C) # constant to be subtracted for local threshold computation
        # 255 is the value put 
        # for the brightests pixels
        # that pass the threshold...
        # in our case, this is the white background
        thresholded = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kernel_size, C)
        # it gives the thresholded image
        # as 255 (white, background)
        # and black (0, ink)

        # Convert and return image as binary mask
        # having 1 as foreground (ink), and 0 as background
        return BinaryMask.from_image(thresholded)

    @property
    def hyperparameters(self):
        return super().hyperparameters | {
            "C":int(self.C),
            "d_gaussian":int(self.d_gaussian),
        }
    
    def hyperparameter_ranges(self,trial: Trial) -> dict:
        return super().hyperparameter_ranges(trial) | {
            "C":trial.suggest_int("C", 0, 10),
            "d_gaussian":trial.suggest_categorical("d_gaussian", [i * 2 + 1 for i in range(1,21)]), # odd integers from 3 to 41
        }