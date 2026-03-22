from abc import ABC, abstractmethod
import cv2
import numpy as np
from utils.seeds import BoxSeed, BrushSeed, PointSeed, Seed
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

    def __init__(self, display_seeds: bool = False):
        self.display_seeds = display_seeds

    # main method consisting of preprocessing followed by seed parsing and segmenation (returns a binary mask)
    def predict(self, image: np.ndarray, seeds : list[Seed] | None = None, autoseed : bool = True) -> BinaryMask:
        if seeds is None and autoseed: # if no seeds are provided but autoseeding is enabled, autoseeding happens
            seeds = self.autoseed(image) # autoseeding is supposed to be done on the raw image, not the preprocessed one
        if self.display_seeds and seeds: # for visualization of the seeds, can be turned off if not wanted or if too slow
            seed_overlay = self.draw_seeds(image, seeds) if seeds else image
            cv2.imshow("seeds", seed_overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        preprocessed = self.preprocess(image)
        return self.segment(preprocessed, seeds)
    
    def draw_seeds(self, image: np.ndarray, seeds: list[Seed]) -> np.ndarray:
        drawn = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        for seed in seeds:
            if isinstance(seed, BoxSeed): # green boxes for box seeds
                cv2.rectangle(drawn, (seed.x1, seed.y1), (seed.x2, seed.y2), (0, 255, 0), 2)
            elif isinstance(seed, PointSeed): # green circles for fgd points, red circles for bgd points
                color = (0, 255, 0) if seed.label == 1 else (0, 0, 255)
                cv2.circle(drawn, (seed.x, seed.y), 5, color, -1)
            elif isinstance(seed, BrushSeed): # green pixels for fgd, red pixels for bgd
                brush_alpha = 0.3
                pixels = np.array(seed.pixels)
                xs = pixels[:, 0]
                ys = pixels[:, 1]
                mask = (xs >= 0) & (ys >= 0) & (xs < drawn.shape[1]) & (ys < drawn.shape[0])
                color = np.array((0, 255, 0) if seed.label == cv2.GC_FGD else (0, 0, 255), dtype=np.float32)
                drawn[ys[mask], xs[mask]] = (
                    (1 - brush_alpha) * drawn[ys[mask], xs[mask]] + brush_alpha * color
                ).astype(np.uint8)
        return drawn
    
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