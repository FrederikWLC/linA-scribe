import cv2
from utils import Image
import numpy as np
from model.evaluator import Evaluator


class Scribe(Evaluator): # the abstract base class for all "scribe" models.

    def scribe(self, image: Image | np.ndarray) -> np.ndarray:
        return self.segment(self.preprocess(image))

    def segment(self, image: Image | np.ndarray) -> np.ndarray: # raw segmentation
        raise NotImplementedError("Subclasses must implement the segment method.")
    
    def preprocess(self, image: Image | np.ndarray) -> np.ndarray: # must be implemented by subclass, default is no preprocessing
        return image


        