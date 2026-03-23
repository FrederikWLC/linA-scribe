from model.scribe import Scribe
import numpy as np
import cv2
from utils.binary_mask import BinaryMask

# implementation of the Gaussian threshold
class Gaussian(Scribe):
    def __init__(self, d_bilateral=15, sigma_color=75, sigma_space=75, d_gaussian=19, C=5):
        self.d_bilateral = d_bilateral
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.d_gaussian = d_gaussian
        self.C = C

    def segment(self, image: np.ndarray) -> BinaryMask:
        kernel_size = self.d_gaussian # size of kernel (neighborhood)
        C = self.C # constant to be subtracted for local threshold computation
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
        d = self.d_bilateral # kernel size

        # spatial standard deviation : intensity-similarity smoothing index
        sigma_color = self.sigma_color #   (the bigger the difference the smaller the value, the bigger the value the more smoothing of contrasting pixels)
        
        # intensity/range standard deviation : distance-based smoothing index
        # the further away a pixel is from kernel center the less influence, the bigger the value the more uniform weighting of pixels)
        sigma_space = self.sigma_space
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        