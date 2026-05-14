from abc import ABC, abstractmethod
import cv2
import numpy as np
from scribe.binary_mask import BinaryMask
from scribe.prompts import PointPrompt, PointPromptList, Prompt
from config import config
class ModelConfiguration:
    def __init__(self, name: str = None, short_name: str = None):
        self.name = name
        self.short_name = short_name    

class BaseScribe(ABC):

    def __init__(self, configuration: ModelConfiguration, **kwargs):
        self.configuration = configuration

    # method where the preprocessing happens (returns an image, not a binary mask)
    def preprocess(self, image: np.ndarray) -> np.ndarray:  # must be implemented by subclass, default is no preprocessing
        return image
    
    @property
    def name(self):
        return self.configuration.name
    
    @property
    def short_name(self):
        return self.configuration.short_name



# the abstract base class for all "scribe" models.
class Scribe(BaseScribe):

    # main method consisting of preprocessing followed by segmenation (returns a binary mask)
    def predict(self, image: np.ndarray) -> BinaryMask:
        preprocessed = self.preprocess(image)
        return self.segment(preprocessed)

    # method where the segmentation happens (returns a binary mask)
    @abstractmethod
    def segment(self, image: np.ndarray) -> BinaryMask:  # raw segmentation
        pass


# the abstract base class for scribe models that can take prompts (e.g. points or boxes)
class SeedableScribe(BaseScribe):

    # main method consisting of preprocessing followed by prompt parsing and segmenation (returns a binary mask)
    def predict(self, image: np.ndarray = None, prompts=None, autoprompt: bool = True) -> BinaryMask:
        if prompts is None and autoprompt:  # if no prompt is provided but autoprompting is enabled, autoprompting happens
            prompts = self.autoprompt(image)  # autoprompting is supposed to be done on the raw image, not the preprocessed one
        preprocessed = self.preprocess(image) if image is not None else None # allow for possibility of prompt-only prediction (for future extensions where image can be set prior to prediction)
        return self.segment(preprocessed, prompts)

    # method where the autoprompting happens
    def autoprompt(self, image: np.ndarray) -> list[Prompt] | None:
        return None

    @abstractmethod
    def draw_prompt(self, image: np.ndarray, prompts) -> np.ndarray:
        pass

    # method where the segmentation happens (returns a binary mask)
    @abstractmethod
    def segment(self, image: np.ndarray, prompts: list[Prompt] | None = None) -> BinaryMask:
        pass


class PointScribe(SeedableScribe):
    def draw_prompt(self, image: np.ndarray, point_prompt_list: PointPromptList) -> np.ndarray:
        drawn = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for prompt in point_prompt_list.point_prompts:
            # green circles for fgd points, red circles for bgd points
            color = (0, 255, 0) if prompt.label == 1 else (0, 0, 255)
            cv2.circle(drawn, (prompt.x, prompt.y), 5, color, -1)
        return drawn


def predict(model: BaseScribe, image, prompt=None) -> BinaryMask:
    if isinstance(model, SeedableScribe):
        return model.predict(image, prompts=prompt)
    return model.predict(image)


def predict_batch(model: BaseScribe, images: list[np.ndarray], prompt=None) -> list[BinaryMask]:
    return [predict(model, img, prompt=prompt) for img in images]
