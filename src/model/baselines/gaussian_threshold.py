from model.scribe import Scribe
from utils.image import Image
import numpy as np
import cv2

class GaussianThreshold(Scribe):
    def __init__(self):
        pass

    def scribe(self, image: Image | np.ndarray):
        
        image = Image(image)
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
