from model.scribe import Scribe
import cv2
import numpy as np
from utils.binary_mask import BinaryMask

# implementation of the 
# Canny edge detection + filling method
from model.scribe import Scribe
import cv2
import numpy as np
from utils.binary_mask import BinaryMask


class CannyFill(Scribe):
    def __init__(self, sigma=0.1, kernel_size=15):
        self.sigma = sigma
        self.kernel_size = kernel_size

    def segment(self, image: np.ndarray) -> BinaryMask:

        # auto canny thresholds
        v = np.median(image)
        low = int(max(0, (1 - self.sigma) * v))
        high = int(min(255, (1 + self.sigma) * v))

        # canny edge detection
        edges = cv2.Canny(image, low, high)

        # filling up the edges via morphological closing (dilation followed by erosion)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return BinaryMask(filled)
    
    def preprocess(self, image):
        return image
        
    @property
    def name(self):
        return "CannyFill"