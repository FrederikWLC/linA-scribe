from model.scribe import Scribe
from utils.image import Image
import numpy as np
import cv2

class BilateralGaussian(Scribe):
    def __init__(self):
        pass

    def scribe(self, image: Image | np.ndarray):
        
        image = Image(image)
        bilateral = cv2.bilateralFilter(image, 15, 75, 75)
        gaussian = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 5)
        return gaussian
        