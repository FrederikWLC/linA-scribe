# Source: custom wrapper for this repository.
# Purpose: provide in-program FATE-SAM usage with a Scribe-style interface.

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.model.scribe_sam.dataset_loader import ImageLabelLoader
from src.model.scribe_sam.sam2.build_sam import sam2_predictor, sam2_predictor_fate
from src.model.scribe_sam.predictor import (
    compute_features,
    find_top_similar_images_embed,
    as_gray_image,
    load_image,
    load_support_data_from_arrays,
    load_support_data_from_loader,
    run_inference_single_volume,
)
from scribe.base import PointScribe
from scribe.binary_mask import BinaryMask


class ScribeSAM(PointScribe):
    def __init__(
        self,
        support_images=None,
        support_labels=None,
        support_images_path=None,
        support_labels_path=None,
        num_classes=0,
        top_n=3,
        image_size=1024,
    ):
        self.num_classes = int(num_classes)
        self.top_n = int(top_n)
        self.image_size = int(image_size)

        if support_images is not None and support_labels is not None:
            self.support_images, self.support_labels = load_support_data_from_arrays(
                support_images,
                support_labels,
                image_size=self.image_size,
            )
        else:
            if support_images_path is None:
                raise ValueError(
                    "Provide either support_images/support_labels arrays or support_images_path/support_labels_path."
                )
            loader = ImageLabelLoader(support_images_path, support_labels_path, image_size=self.image_size)
            self.support_images, self.support_labels = load_support_data_from_loader(loader)

        self._feature_predictor = sam2_predictor()
        self._fate_predictor = sam2_predictor_fate()
        self._support_features = None

        self._cached_query_folder = None
        self._cached_similarity_results = None
        self._cached_output_hw = None

    def _write_temp_query_image_dir(self, image: np.ndarray) -> str:
        tmp_dir = tempfile.mkdtemp(prefix="fate_sam_query_")
        image_path = Path(tmp_dir) / "00000.jpg"
        gray = as_gray_image(np.asarray(image))
        cv2.imwrite(str(image_path), gray)
        return tmp_dir

    def _ensure_support_features(self, folder_path: str):
        if self._support_features is None:
            self._support_features = compute_features(
                folder_path=folder_path,
                images=self.support_images,
                predictor=self._feature_predictor,
                batch_size=1,
            )

    def clear_image(self):
        if self._cached_query_folder is not None:
            shutil.rmtree(self._cached_query_folder, ignore_errors=True)
        self._cached_query_folder = None
        self._cached_similarity_results = None
        self._cached_output_hw = None

    def setImage(self, image: np.ndarray):
        self.clear_image()
        self._cached_output_hw = np.asarray(image).shape[:2]
        self._cached_query_folder = self._write_temp_query_image_dir(image)

        query_image = load_image(self._cached_query_folder)
        self._ensure_support_features(self._cached_query_folder)

        query_feature = compute_features(
            folder_path=self._cached_query_folder,
            images=query_image,
            predictor=self._feature_predictor,
            batch_size=1,
        )

        self._cached_similarity_results = find_top_similar_images_embed(
            self.support_images,
            self._support_features,
            self.support_labels,
            query_feature,
            top_n=self.top_n,
        )
        return self

    def decode_mask(self, prompts=None) -> BinaryMask:
        if self._cached_query_folder is None or self._cached_similarity_results is None:
            raise RuntimeError("No image is set. Call setImage(image) before decode_mask(...).")

        _dice_df, seg_predictions = run_inference_single_volume(
            image_folder=self._cached_query_folder,
            label=None,
            similarity_results=self._cached_similarity_results,
            predictor=self._fate_predictor,
            num_classes=self.num_classes,
            prompts=prompts,
        )

        if not seg_predictions:
            h, w = self._cached_output_hw if self._cached_output_hw is not None else (1024, 1024)
            return BinaryMask.from_bool(np.zeros((h, w), dtype=np.uint8))

        frame_idx = min(seg_predictions.keys())
        frame_pred = seg_predictions[frame_idx]
        merged = np.zeros_like(next(iter(frame_pred.values())).squeeze(), dtype=np.uint8)
        for obj_id, obj_mask in frame_pred.items():
            merged[np.squeeze(obj_mask) > 0] = np.uint8(obj_id)

        if self._cached_output_hw is not None and merged.shape != self._cached_output_hw:
            merged = cv2.resize(
                merged,
                (self._cached_output_hw[1], self._cached_output_hw[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        return BinaryMask.from_bool(merged > 0)

    def segment(self, image: np.ndarray, prompts=None) -> BinaryMask:
        if image is not None:
            self.setImage(image)
        elif self._cached_query_folder is None:
            raise RuntimeError("No image provided and no cached image available. Provide image or call setImage(image).")
        return self.decode_mask(prompts=prompts)
