import cv2
import numpy as np
from pathlib import Path
from utils.image import Image

class Mask:
    def __init__(
        self,
        mask: np.ndarray,
        score: float,
        logit,
        source_image: Image | np.ndarray
    ):
        self.mask = mask
        self.score = score
        self.logit = logit
        self.source_image = Image(source_image)

    def to_overlay(self) -> Image:
        return Image((
            self.source_image * self.mask[..., None]
        ).astype("uint8"))

    def to_image(self) -> Image:
        image = (self.mask * 255).astype("uint8")
        print(f"Mask to image: {image.shape}, {image.dtype}")
        return Image(image)