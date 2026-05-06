from abc import ABC

import cv2
import numpy as np
from optuna import Trial
from scribe.base import BaseScribe, ModelConfiguration
from config import config

class HyperparameterSpec:
    def __init__(self, name,default,suggest):
        self.name = name
        self.default = default
        self.suggest = suggest

class TunableConfiguration(ModelConfiguration):
    def __init__(self, name, short_name, hyperparameter_specs: list[HyperparameterSpec]):
        super().__init__(name, short_name)
        self.hyperparameter_specs = hyperparameter_specs
        self.hyperparameter_values = {spec.name: spec.default for spec in hyperparameter_specs}
        self.hyperparameters = [spec.name for spec in hyperparameter_specs]

    def get_value(self, name):
        return self.hyperparameter_values.get(name, None)

    def hyperparameter_ranges(self, trial: Trial) -> dict:
        return {
            spec.name: spec.suggest(trial)
            for spec in self.hyperparameter_specs
        }
    
    def set_hyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.hyperparameter_values:
                self.hyperparameter_values[key] = value

    # Variant that does not error if kwargs contains keys that are not hyperparameters of this model; it just ignores them
    def set_hyperparameters_from(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.hyperparameter_values:
                self.hyperparameter_values[key] = value

    def is_tunable(self):
        return len(self.hyperparameter_specs) > 0
    
class Tunable(BaseScribe, ABC):

    configuration: TunableConfiguration
    
    def set_hyperparameters(self, **kwargs):
        self.configuration.set_hyperparameters(**kwargs)
        return self # for chaining pattern: scribe.set_hyperparameters(...).predict(image)
    
    def set_hyperparameters_from(self, **kwargs):
        self.configuration.set_hyperparameters_from(**kwargs)
        return self # for chaining pattern: scribe.set_hyperparameters_from(...).predict(image)

    def hyperparameter_ranges(self, trial: Trial) -> dict:
        return self.configuration.hyperparameter_ranges(trial)
    
    def is_tunable(self):
        return self.configuration.is_tunable()
    
    @property
    def hyperparameters(self) -> list[str]:
        return self.configuration.hyperparameters

    @property
    def hyperparameter_values(self) -> dict:
        return self.configuration.hyperparameter_values
    
    
    
class BilateralTunable(Tunable, ABC):

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        d = int(self.configuration.get_value("d_bilateral"))
        sigma_color = int(self.configuration.get_value("sigma_bilateral"))
        sigma_space = int(self.configuration.get_value("sigma_bilateral"))
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

BILATERAL_SPECS = [
    HyperparameterSpec("d_bilateral", default=15, suggest=lambda trial: trial.suggest_int("d_bilateral", 3, 31)),
    HyperparameterSpec("sigma_bilateral", default=75, suggest=lambda trial: trial.suggest_int("sigma_bilateral", 0, 150)),
]
