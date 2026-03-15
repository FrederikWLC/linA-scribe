from model.scribe import Scribe
import numpy as np
import cv2
from utils.binary_mask import BinaryMask

# implementation of the Gaussian threshold
class Gaussian(Scribe):
    def __init__(self):
        pass

    def segment(self, image: np.ndarray) -> BinaryMask:
        kernel_size = 19 # size of kernel (neighborhood)
        C = 5 # constant to be subtracted for local threshold computation
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

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # preprocess with bilateral filter
        d = 15 # kernel size
        sigma_color = 75 # intensity similarity smoothing influence (the bigger the difference the less, the bigger the value the more smoothing of contrasting pixels)
        sigma_space = 75 # distance-based smoothing influence (the further the less, the bigger the value the more uniform)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        