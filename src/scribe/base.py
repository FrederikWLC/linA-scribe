from abc import ABC, abstractmethod
import cv2
import numpy as np
from optuna import Trial

from scribe.binary_mask import BinaryMask
from scribe.prompts import PointPrompt, Prompt

class Named:

    # name of the model, used for display and evaluation purposes
    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def short_name(self) -> str:
        return self.name

class BaseScribe(ABC, Named):

    # method where the preprocessing happens (returns an image, not a binary mask)
    def preprocess(self, image: np.ndarray) -> np.ndarray:  # must be implemented by subclass, default is no preprocessing
        return image



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
        preprocessed = self.preprocess(image) if image is not None else None
        return self.segment(preprocessed, prompts)

    # method where the autoprompting happens
    def autoprompt(self, image: np.ndarray) -> list[Prompt] | None:
        return None

    @abstractmethod
    def draw_prompt(self, image: np.ndarray, prompt) -> np.ndarray:
        pass

    # method where the segmentation happens (returns a binary mask)
    @abstractmethod
    def segment(self, image: np.ndarray, prompts: list[Prompt] | None = None) -> BinaryMask:
        pass


class PointScribe(SeedableScribe):
    def draw_prompt(self, image: np.ndarray, point_prompts: list[PointPrompt]) -> np.ndarray:
        drawn = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for prompt in point_prompts:
            # green circles for fgd points, red circles for bgd points
            color = (0, 255, 0) if prompt.label == 1 else (0, 0, 255)
            cv2.circle(drawn, (prompt.x, prompt.y), 5, color, -1)
        return drawn


class BrushScribe(SeedableScribe):
    def draw_prompt(self, image: np.ndarray, brush_mask: np.ndarray) -> np.ndarray:
        drawn = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        brush_color_from_label = {
            cv2.GC_BGD: (0, 0, 255),  # red for sure background
            cv2.GC_PR_BGD: (255, 255, 255),  # white for probable background
            cv2.GC_PR_FGD: (0, 0, 0),  # black for probable foreground
            cv2.GC_FGD: (0, 255, 0),  # green for sure foreground
        }
        brush_alpha_from_label = {
            cv2.GC_BGD: 0.4,  # transparent overlay for sure background
            cv2.GC_FGD: 0.4,  # transparent overlay for sure foreground
            cv2.GC_PR_BGD: 0.4,  # transparent overlay for probable background
            cv2.GC_PR_FGD: 0.4,  # transparent overlay for probable foreground
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
    def hyperparameters(self) -> dict:
        return {}

    def hyperparameter_ranges(self, trial: Trial) -> dict:
        return {}

    def set_hyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.hyperparameters:
                setattr(self, key, value)


class BilateralTunable(BaseScribe, Tunable, ABC):

    def __init__(self, d_bilateral: int = 15, sigma_bilateral: int = 75, **kwargs):
        self.d_bilateral = int(d_bilateral)
        self.sigma_bilateral = int(sigma_bilateral)
        super().__init__(**kwargs)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        d = int(self.d_bilateral)
        sigma_color = int(self.sigma_bilateral)
        sigma_space = int(self.sigma_bilateral)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    @property
    def hyperparameters(self) -> dict:
        return super().hyperparameters | {
            "d_bilateral": int(self.d_bilateral),
            "sigma_bilateral": int(self.sigma_bilateral),
        }

    def hyperparameter_ranges(self, trial: Trial, sigma_max: int = 150) -> dict:
        return super().hyperparameter_ranges(trial) | {
            "d_bilateral": trial.suggest_int("d_bilateral", 3, 31),
            "sigma_bilateral": trial.suggest_int("sigma_bilateral", 0, sigma_max),
        }


def predict(model: BaseScribe, image, prompt=None) -> BinaryMask:
    if isinstance(model, SeedableScribe):
        return model.predict(image, prompts=prompt)
    return model.predict(image)


def predict_batch(model: BaseScribe, images: list[np.ndarray], prompt=None) -> list[BinaryMask]:
    return [predict(model, img, prompt=prompt) for img in images]
