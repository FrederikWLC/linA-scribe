from abc import ABC, abstractmethod
import numpy as np
from utils.seeds import Seed
from utils.binary_mask import BinaryMask

class BaseScribe(ABC):

    # method where the preprocessing happens (returns an image, not a binary mask)
    def preprocess(self, image: np.ndarray) -> np.ndarray: # must be implemented by subclass, default is no preprocessing
        return image

    # name of the model, used for display and evaluation purposes
    @property
    def name(self) -> str:
        return type(self).__name__
    
# the abstract base class for all "scribe" models.
class Scribe(BaseScribe):

    # main method consisting of preprocessing followed by segmenation (returns a binary mask)
    def predict(self, image: np.ndarray) -> BinaryMask:
        preprocessed = self.preprocess(image)
        return self.segment(preprocessed)

    # method where the segmentation happens (returns a binary mask)
    @abstractmethod
    def segment(self, image: np.ndarray) -> BinaryMask: # raw segmentation
        pass

# the abstract base class for scribe models that can take seeds (i.e. prompts, e.g. points or boxes)
class SeedableScribe(BaseScribe): 

    # main method consisting of preprocessing followed by seed parsing and segmenation (returns a binary mask)
    def predict(self, image: np.ndarray, seeds : list[Seed] | None = None, autoseed : bool = True) -> BinaryMask:
        preprocessed = self.preprocess(image)
        if seeds is None and autoseed: # if no seeds are provided but autoseeding is enabled, autoseeding happens
            seeds = self.autoseed(image) # autoseeding is supposed to be done on the raw image, not the preprocessed one
        return self.segment(preprocessed, seeds)
    
    # method where the autoseeding happens
    def autoseed(self, image: np.ndarray) -> list[Seed] | None:
        return None

    # method where the segmentation happens (returns a binary mask)
    @abstractmethod
    def segment(self, image: np.ndarray, seeds: list[Seed] | None = None) -> BinaryMask:
        pass

def predict(model: BaseScribe, image, seeds: Seed | None = None) -> BinaryMask:
    if isinstance(model, SeedableScribe):
        return model.predict(image,seeds=seeds)
    return model.predict(image)

def predict_batch(model: BaseScribe, images: list[np.ndarray], seeds=None) -> list[BinaryMask]:
    return [predict(model,img,seeds) for img in images]