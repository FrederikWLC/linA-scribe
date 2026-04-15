# Source: adapted from upstream FATE-SAM dataset loader.
# Upstream file: https://github.com/I3Tlab/FATE-SAM/blob/main/notebooks/dataset_loader.py
# Adaptation here: changed 3D NIfTI volume loading to 2D JPG/PNG raw+label pair loading.

import os
from pathlib import Path

import cv2
import numpy as np
import torch


class ImageLabelLoader:
    """Original-style support loader adapted to 2D JPG/PNG pairs."""

    IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, images_path, labels_path=None, image_size=1024):
        self.images_path = Path(images_path)
        self.labels_path = Path(labels_path) if labels_path is not None else None
        self.image_size = image_size

    def _find_label_path(self, image_stem: str) -> Path:
        search_root = self.labels_path if self.labels_path is not None else self.images_path

        candidates = [search_root / f"{image_stem}{suf}" for suf in self.IMAGE_SUFFIXES]
        candidates += [search_root / f"{image_stem.replace('_img', '_label')}{suf}" for suf in self.IMAGE_SUFFIXES]

        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Label not found for support image stem '{image_stem}'")

    def _load_image_tensor(self, image_path: Path) -> torch.Tensor:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Failed to read support image: {image_path}")
        resized = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        return torch.from_numpy(rgb).permute(2, 0, 1)

    def _load_label(self, label_path: Path, target_hw: tuple[int, int]) -> np.ndarray:
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Failed to read support label: {label_path}")
        if label.shape != target_hw:
            label = cv2.resize(label, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)
        if np.unique(label).size <= 2:
            label = (label > 0).astype(np.int32)
        return label

    def load_data(self):
        data = []
        image_paths = sorted(
            [p for p in self.images_path.iterdir() if p.is_file() and p.suffix.lower() in self.IMAGE_SUFFIXES]
        )
        for image_path in image_paths:
            label_path = self._find_label_path(image_path.stem)

            raw_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if raw_image is None:
                continue

            data.append(
                {
                    "image": self._load_image_tensor(image_path).unsqueeze(0),
                    "image_folder": str(image_path.parent),
                    "label": [self._load_label(label_path, raw_image.shape)],
                    "volume_idx": image_path.stem,
                }
            )
        return data