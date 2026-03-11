from model.scribe import Scribe
import cv2
import numpy as np

class CannyFill(Scribe):
    def __init__(self, sigma=0.33, kernel_size=5):
        self.sigma = sigma
        self.kernel_size = kernel_size

    def _auto_canny(self, gray):
        v = np.median(gray)
        lower = int(max(0, (1.0 - self.sigma) * v)) # lower threshold
        upper = int(min(255, (1.0 + self.sigma) * v)) # upper threshold
        return lower, upper

    def segment(self, image : np.ndarray):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        low, high = self._auto_canny(image)

        edges = cv2.Canny(image, low, high)

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        filled = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.bitwise_not(filled)
    
    def preprocess(self, image: np.ndarray):
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)