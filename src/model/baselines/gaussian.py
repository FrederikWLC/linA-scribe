from model.scribe import Scribe, Tunable
import numpy as np
import cv2
from utils.binary_mask import BinaryMask
from optuna import Trial

# implementation of the Gaussian threshold
class Gaussian(Scribe,Tunable):
    def __init__(self, C=5, d_gaussian=19, d_bilateral=15, sigma=75):
        self.d_bilateral = d_bilateral
        self.sigma = sigma
        self.d_gaussian = d_gaussian
        self.C = C

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

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # preprocess with bilateral filter
        d = int(self.d_bilateral) # kernel size

        # spatial standard deviation : intensity-similarity smoothing index
        sigma_color = int(self.sigma) #   (the bigger the difference the smaller the value, the bigger the value the more smoothing of contrasting pixels)
        
        # intensity/range standard deviation : distance-based smoothing index
        # the further away a pixel is from kernel center the less influence, the bigger the value the more uniform weighting of pixels)
        sigma_space = int(self.sigma)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        
    @property
    def hyperparameters(self):
        return {
            "C":int(self.C),
            "d_gaussian":int(self.d_gaussian),
            "d_bilateral":int(self.d_bilateral),
            "sigma":int(self.sigma),
        }
    
    def hyperparameter_ranges(self,trial: Trial) -> dict:
        return {
            "C":trial.suggest_int("C", 0, 10),
            "d_gaussian":trial.suggest_categorical("d_gaussian", [i * 2 + 1 for i in range(1,16)]), # odd integers from 3 to 31
            "d_bilateral":trial.suggest_categorical("d_bilateral", [i * 2 + 1 for i in range(1,16)]), # odd integers from 3 to 31
            "sigma":trial.suggest_int("sigma", 0, 150)
        }