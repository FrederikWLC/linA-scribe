from scribe.base import Scribe
from scribe.tunable import BILATERAL_SPECS, BilateralTunable, HyperparameterSpec, TunableConfiguration
import numpy as np
import cv2
from scribe.binary_mask import BinaryMask
from optuna import Trial

GAUSSIAN_SPECS = BILATERAL_SPECS + [
    HyperparameterSpec("C", default=9, suggest=lambda trial: trial.suggest_int("C", 0, 10)),
    HyperparameterSpec("d_gaussian", default=39, suggest=lambda trial: trial.suggest_categorical("d_gaussian", [i * 2 + 1 for i in range(1,21)])), # odd integers from 3 to 41
]   
                       
class GaussianConfiguration(TunableConfiguration):
    def __init__(self):
        super().__init__(
            name="Gaussian",
            short_name="Gaus",
            hyperparameter_specs=GAUSSIAN_SPECS
        )

# implementation of the Gaussian threshold
class Gaussian(Scribe, BilateralTunable):
    
    configuration: GaussianConfiguration

    def segment(self, image: np.ndarray) -> BinaryMask:
        kernel_size = int(self.configuration.get_value("d_gaussian")) # size of kernel (neighborhood)
        C = int(self.configuration.get_value("C")) # constant to be subtracted for local threshold computation
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

def build_gaussian():
    return Gaussian(GaussianConfiguration())