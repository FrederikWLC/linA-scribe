import numpy as np
from model.evaluator import Evaluator
from utils.seeds import Seed


class Scribe(Evaluator): # the abstract base class for all "scribe" models.

    def scribe(self, image: np.ndarray) -> np.ndarray:
        return self.segment(self.preprocess(image))

    def segment(self, image: np.ndarray) -> np.ndarray: # raw segmentation
        raise NotImplementedError("Subclasses must implement the segment method.")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray: # must be implemented by subclass, default is no preprocessing
        return image
    
    @property
    def name(self):
        return self.__class__.__name__


class SeedableScribe(Scribe): # the abstract base class for scribe models that require seeds (e.g. points or boxes)

    def scribe(self, image: np.ndarray, seeds : list[Seed] = None, autoseed : bool = False) -> np.ndarray:
        preprocessed = self.preprocess(image)
        seeds = seeds if not autoseed else self.autoseed(preprocessed)
        return self.segment(preprocessed, seeds)
    
    def segment(self, image: np.ndarray, seeds: list[Seed] = None) -> np.ndarray: # raw segmentation
        raise NotImplementedError("Subclasses must implement the segment method.")

    def autoseed(self, image: np.ndarray) -> list[Seed]: # autoseeding
        raise NotImplementedError("Subclasses must implement the autoseed method.")


