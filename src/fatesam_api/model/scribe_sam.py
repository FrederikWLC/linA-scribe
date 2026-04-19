# Source: custom wrapper for this repository.
# Purpose: provide in-program FATE-SAM usage with a Scribe-style interface.

import cv2
import numpy as np
from fatesam_api.model.dataset_loader import images_to_tensor, labels_to_tensor
from fatesam_api.model.predictor import (
    prepare_inference_state,
    run_from_inference_state,
)
from scribe.base import PointScribe
from scribe.binary_mask import BinaryMask

class ScribeSAM(PointScribe):
    def __init__(
        self,
        support_images,
        support_labels
        ):
        self.support_images = images_to_tensor(support_images, image_size=1024)
        self.support_labels = labels_to_tensor(support_labels, image_size=1024)
        self.inference_state = None
        self.similarity_results = None
        
    def hasImage(self):
        return self.inference_state is not None

    def setImage(self, image):
        self.inference_state, self.similarity_results = prepare_inference_state(
            query_image=image,
            support_images=self.support_images,
            support_labels=self.support_labels
        )
        self._output_hw = image.shape[:2]
        return self
    
    def decode_mask(self, prompts=None) -> BinaryMask:
        if not self.hasImage():
            raise RuntimeError("No image is set. Call setImage(image) before decode_mask(...).")
        
        frame_pred = run_from_inference_state(
            self.inference_state,
            self.similarity_results,
            prompts=prompts
        )

        if not frame_pred:
            height, width = self._output_hw
            return BinaryMask.from_bool(np.zeros((height, width), dtype=np.uint8))

        merged = np.zeros_like(next(iter(frame_pred.values())).squeeze(), dtype=np.uint8)
        for obj_id, obj_mask in frame_pred.items():
            merged[np.squeeze(obj_mask) > 0] = np.uint8(obj_id)

        if tuple(merged.shape[:2]) != tuple(self._output_hw):
            merged = cv2.resize(merged, (self._output_hw[1], self._output_hw[0]), interpolation=cv2.INTER_NEAREST)

        return BinaryMask.from_bool(merged > 0)

    def segment(self, image=None, prompts=None) -> BinaryMask:
        if image is not None:
            self.setImage(image)
        elif not self.hasImage():
            raise RuntimeError("No image provided and no image is set. Call setImage(image) first.")
        return self.decode_mask(prompts=prompts)
