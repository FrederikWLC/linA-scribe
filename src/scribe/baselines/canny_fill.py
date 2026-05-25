from optuna import Trial
from scribe.base import Scribe
from scribe.tunable import BilateralTunable, HyperparameterSpec, TunableConfiguration, BILATERAL_SPECS
import cv2
import numpy as np
from scribe.binary_mask import BinaryMask

CANNY_SPECS = BILATERAL_SPECS + [
    HyperparameterSpec("d_closing", default=19, suggest=lambda trial: trial.suggest_categorical("d_closing", [i * 2 + 1 for i in range(1,16)])), # odd integers from 3 to 31
    HyperparameterSpec("sigma_canny", default=0.869782773928433, suggest=lambda trial: trial.suggest_float("sigma_canny", 0, 1))
]
class CannyConfiguration(TunableConfiguration):
    def __init__(self):
        super().__init__(
            name="CannyFill",
            short_name="CannyFill",
            hyperparameter_specs=CANNY_SPECS
        )

class CannyFill(Scribe, BilateralTunable):

    configuration: CannyConfiguration

    def segment(self, image: np.ndarray) -> BinaryMask:

        sigma = float(self.configuration.get_value("sigma_canny"))
        d = int(self.configuration.get_value("d_closing"))

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
    
def build_cannyfill():
    return CannyFill(CannyConfiguration())