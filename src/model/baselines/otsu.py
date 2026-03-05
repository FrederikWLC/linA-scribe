import cv2

from model.scribe import Scribe

class Otsu(Scribe):
    def __init__(self):
        pass

    def segment(self, image):
        
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary
    
    def preprocess(self, image):
        return cv2.bilateralFilter(image, 15, 75, 75)