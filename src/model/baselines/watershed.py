import cv2
import numpy as np
from model.scribe import Scribe

class Watershed(Scribe):
    def __init__(self):
        pass

    def segment(self, image):

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Create markers manually
        markers = np.zeros(image.shape, dtype=np.int32)

        # Example: simple gradient-based seeds
        gradient = cv2.Laplacian(image, cv2.CV_8U)

        # Foreground seeds = low gradient areas (inside regions)
        markers[gradient < 10] = 2

        # Background seed (optional)
        markers[gradient > 30] = 1

        # Watershed requires 3-channel image
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        markers = cv2.watershed(bgr, markers)
        mask = ((markers > 1).astype(np.uint8) * 255)   # treat all non-background regions as foreground
        ink = cv2.bitwise_not(mask)
        kernel = np.ones((2, 2), np.uint8)
        ink = cv2.dilate(ink, kernel, iterations=3)  
        mask = cv2.bitwise_not(ink)  # back to original convention
        return mask
    
    def preprocess(self, image):
        return cv2.medianBlur(image, 3)