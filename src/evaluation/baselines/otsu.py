import cv2
from scribe.base import BilateralTunable, Scribe
from scribe.binary_mask import BinaryMask

class Otsu(BilateralTunable,Scribe):
    def __init__(self, d_bilateral=15, sigma_bilateral=75):
        super().__init__(d_bilateral=d_bilateral, sigma_bilateral=sigma_bilateral)

    def segment(self, image):
        
        # simultaneously computes
        # the optimal Otsu threshold T
        # and the thresholded image...
        # 0 is a threshold value not even used by the method (overwritten by Otsu's threshold)
        # 255 is the value put 
        # for the brightests pixels
        # that pass the threshold...
        # in our case, this is the white background
        _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # it gives the thresholded image
        # as 255 (white, background)
        # and black (0, ink)

        # Convert and return image as binary mask
        # having 1 as foreground (ink), and 0 as background
        return BinaryMask.from_image(thresholded)
