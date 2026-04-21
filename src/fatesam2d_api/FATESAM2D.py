from __future__ import annotations

import copy

import cv2
import numpy as np

from fatesam2d_api.dataset_loader import images_to_tensor, labels_to_tensor
from fatesam2d_api.predictor import prepare_inference_state, run_from_inference_state
from scribe.auto_prompts import auto_points
from scribe.base import PointScribe
from scribe.baselines.gaussian import Gaussian
from scribe.binary_mask import BinaryMask
from scribe.prompts import PointPrompt
from scribe.tunable import BilateralTunable
from optuna import Trial


class FATESAM2D(PointScribe):
    """2D pseudo-sequence adaptation of FATE-SAM.

    The query image is frame 0 and the selected support images are appended as
    pseudo-video frames. Support masks are injected on those support frames, and
    optional point prompts can be injected on the query frame.
    """

    NAME = "FATESAM2D"
    SHORT_NAME = "FATE"

    def __init__(self, support_images, support_labels, top_n_supports: int = 3):
        self.support_images = images_to_tensor(support_images, image_size=1024)
        self.support_labels = labels_to_tensor(support_labels, image_size=1024)
        self.top_n_supports = int(top_n_supports)
        self.inference_state = None
        self.similarity_results = None
        self._output_hw = None

    def hasImage(self) -> bool:
        return self.inference_state is not None

    def setImage(self, image):
        image = np.asarray(image)
        self.inference_state, self.similarity_results = prepare_inference_state(
            query_image=image,
            support_images=self.support_images,
            support_labels=self.support_labels,
            top_n=self.top_n_supports,
        )
        self._output_hw = image.shape[:2]
        return self

    def decode_mask(self, prompts=None) -> BinaryMask:
        if not self.hasImage():
            raise RuntimeError("No image is set. Call setImage(image) before decode_mask(...).")

        frame_pred = run_from_inference_state(
            copy.deepcopy(self.inference_state),
            copy.deepcopy(self.similarity_results),
            prompts=prompts,
            prompt_input_hw=self._output_hw,
        )
        return self._merge_frame_prediction(frame_pred)

    def segment(self, image=None, prompts=None) -> BinaryMask:
        if image is not None:
            self.setImage(image)
        elif not self.hasImage():
            raise RuntimeError("No image provided and no image is set. Call setImage(image) first.")
        return self.decode_mask(prompts=prompts)

    def _merge_frame_prediction(self, frame_pred) -> BinaryMask:
        height, width = self._output_hw
        if not frame_pred:
            return BinaryMask.from_bool(np.zeros((height, width), dtype=np.uint8))

        merged = np.zeros_like(next(iter(frame_pred.values())).squeeze(), dtype=np.uint8)
        for obj_id, obj_mask in frame_pred.items():
            merged[np.squeeze(obj_mask) > 0] = np.uint8(obj_id)

        if tuple(merged.shape[:2]) != tuple(self._output_hw):
            merged = cv2.resize(merged, (width, height), interpolation=cv2.INTER_NEAREST)

        return BinaryMask.from_bool(merged > 0)

class FATESAM2DAutoPoint(FATESAM2D,BilateralTunable):
    """FATESAM2D with automatic point prompts."""

    NAME = "FATESAM2D+pts"
    SHORT_NAME = "FATE+pts"

    def __init__(self, support_images, support_labels, top_n_supports: int = 3,
                 d_bilateral=15, sigma_bilateral=75, C=5, d_gaussian=19, 
                 n_fgd_points=1000, n_bgd_points=1000, d_gap_erosion=3):
        FATESAM2D.__init__(
            self,
            support_images=support_images,
            support_labels=support_labels,
            top_n_supports=top_n_supports,
        )
        BilateralTunable.__init__(
            self,
            d_bilateral=d_bilateral,
            sigma_bilateral=sigma_bilateral,
        )
        
        self.C = int(C)
        self.d_gaussian = int(d_gaussian)
        self.n_fgd_points = int(n_fgd_points)
        self.n_bgd_points = int(n_bgd_points)
        self.d_gap_erosion = int(d_gap_erosion)

    def autoprompt(self, image: np.ndarray) -> list[PointPrompt]:
        d_bilateral = int(self.d_bilateral)
        sigma_bilateral = int(self.sigma_bilateral)
        C = int(self.C)
        d_gaussian = int(self.d_gaussian)
        n_fgd_points = int(self.n_fgd_points)
        n_bgd_points = int(self.n_bgd_points)
        d_gap_erosion = int(self.d_gap_erosion)

        thresh = Gaussian(C, d_gaussian, d_bilateral, sigma_bilateral=sigma_bilateral).predict(image)
        points = auto_points(thresh, n_fgd_points, n_bgd_points, d_gap_erosion)
        return points
    
    # no filter
    def preprocess(self, image):
        return image

    @property
    def hyperparameters(self) -> dict:
        return super().hyperparameters | {
            "C": int(self.C),
            "d_gaussian": int(self.d_gaussian),
            "n_fgd_points": int(self.n_fgd_points),
            "n_bgd_points": int(self.n_bgd_points),
            "d_gap_erosion": int(self.d_gap_erosion),
        }


    @classmethod
    def hyperparameter_ranges(cls, trial: Trial) -> dict:
        return super().hyperparameter_ranges(trial) | {
            "C": trial.suggest_int("C", 0, 10),
            "d_gaussian": trial.suggest_categorical("d_gaussian", [i * 2 + 1 for i in range(1, 16)]),
            "n_fgd_points": trial.suggest_int("n_fgd_points", 1, 2000),
            "n_bgd_points": trial.suggest_int("n_bgd_points", 1, 2000),
            "d_gap_erosion": trial.suggest_categorical("d_gap_erosion", [i * 2 + 1 for i in range(1, 11)]),
        }
