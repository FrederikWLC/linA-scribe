from model.scribe import Scribe
from utils.image import Image
import numpy as np
import cv2

class Gaussian(Scribe):
    def __init__(self):
        pass

    def segment(self, image: Image | np.ndarray):
        gaussian = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 5)
        return gaussian
    
    def preprocess(self, image):
        return cv2.bilateralFilter(image, 15, 75, 75)
        