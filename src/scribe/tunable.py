from abc import ABC

import cv2
import numpy as np
from optuna import Trial

from scribe.base import BaseScribe


class Tunable(ABC):

    @property
    def hyperparameters(self) -> dict:
        return {}

    @classmethod
    def hyperparameter_ranges(cls, trial: Trial) -> dict:
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

    @classmethod
    def hyperparameter_ranges(cls, trial: Trial, sigma_max: int = 150) -> dict:
        return super().hyperparameter_ranges(trial) | {
            "d_bilateral": trial.suggest_int("d_bilateral", 3, 31),
            "sigma_bilateral": trial.suggest_int("sigma_bilateral", 0, sigma_max),
        }
