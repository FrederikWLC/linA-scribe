from abc import ABC, abstractmethod
import cv2
import numpy as np
from utils.seeds import BoxSeed, BrushSeed, PointSeed, Seed
from utils.binary_mask import BinaryMask
from optuna import Trial

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
    def predict(self, image: np.ndarray, seed = None, autoseed : bool = True) -> BinaryMask:
        if seed is None and autoseed: # if no seed is provided but autoseeding is enabled, autoseeding happens
            seed = self.autoseed(image) # autoseeding is supposed to be done on the raw image, not the preprocessed one
        preprocessed = self.preprocess(image)
        return self.segment(preprocessed, seed)
    
    # method where the autoseeding happens
    def autoseed(self, image: np.ndarray) -> list[Seed] | None:
        return None
    
    @abstractmethod
    def draw_seed(self, image: np.ndarray, seed) -> np.ndarray:
        pass
    
    # method where the segmentation happens (returns a binary mask)
    @abstractmethod
    def segment(self, image: np.ndarray, seeds: list[Seed] | None = None) -> BinaryMask:
        pass

class PointScribe(SeedableScribe):
    def draw_seed(self, image: np.ndarray, point_seeds: list[PointSeed]) -> np.ndarray:
        drawn = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for seed in point_seeds:
            # green circles for fgd points, red circles for bgd points
            color = (0, 255, 0) if seed.label == 1 else (0, 0, 255)
            cv2.circle(drawn, (seed.x, seed.y), 5, color, -1)
        return drawn

class BrushScribe(SeedableScribe):
    def draw_seed(self, image: np.ndarray, brush_mask: np.ndarray) -> np.ndarray:
        drawn = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        brush_color_from_label = {
            cv2.GC_BGD: (0, 0, 255), # red for sure background
            cv2.GC_PR_BGD: (255, 255, 255), # white for probable background
            cv2.GC_PR_FGD: (0, 0, 0), # black for probable foreground
            cv2.GC_FGD: (0, 255, 0) # green for sure foreground
        }
        brush_alpha_from_label = {
            cv2.GC_BGD: 0.4, # transparent overlay for sure background
            cv2.GC_FGD: 0.4, # transparent overlay for sure foreground
            cv2.GC_PR_BGD: 0.4, # transparent overlay for probable background
            cv2.GC_PR_FGD: 0.4 # transparent overlay for probable foreground
        }
        for label in [cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD]:
            label_mask = brush_mask == label
            ys, xs = np.where(label_mask)
            color = np.array(brush_color_from_label[label])
            alpha = brush_alpha_from_label[label]
            drawn[ys, xs] = (
                (1 - alpha) * drawn[ys, xs] + alpha * color
                ).astype(np.uint8)
        return drawn
    
class Tunable(ABC):

    @property
    @abstractmethod
    def hyperparameters(self) -> dict:
        pass
    
    @abstractmethod
    def hyperparameter_ranges(self,trial: Trial) -> dict:
        pass
    
    def set_hyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.hyperparameters:
                setattr(self, key, value)


def predict(model: BaseScribe, image, seed = None) -> BinaryMask:
    if isinstance(model, SeedableScribe):
        return model.predict(image,seed=seed)
    return model.predict(image)

def predict_batch(model: BaseScribe, images: list[np.ndarray], seed=None) -> list[BinaryMask]:
    return [predict(model,img,seed) for img in images]