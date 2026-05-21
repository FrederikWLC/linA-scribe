from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import sys

import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from data.split import get_support_data
from scribe.base import BaseScribe, ModelConfiguration

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from matcher.GFSAM import build_model
import numpy as np
from scribe.binary_mask import BinaryMask
from config import config

class GFSAMConfiguration(ModelConfiguration):
    def __init__(
        self,
        support_data_root: str = config.DATA_DIR,
        dinov2_weights: str | Path = "/root/models/dinov2_vitl14_pretrain.pth",
        sam_weights: str | Path = "/root/models/sam_vit_h_4b8939.pth",
        dinov2_size: str = "vit_large",
        sam_size: str = "vit_h",
        img_size: int = 1024,
        device=None
    ):
        self.support_data_root = support_data_root
        self.dinov2_weights = dinov2_weights
        self.sam_weights = sam_weights
        self.dinov2_size = dinov2_size
        self.sam_size = sam_size
        self.img_size = img_size
        self.device = device

        super().__init__(
            name="GFSAM",
            short_name="GFSAM"
        )

    def get_support_data(self):
        support_images, support_labels, _ = get_support_data(data_root=self.support_data_root, binarized=True)
        return support_images, support_labels

def build_from_gfsam_configuration(configuration: GFSAMConfiguration) -> GFSAM:
    return GFSAM(configuration)

def get_gfsam_configuration():
    return GFSAMConfiguration(
        support_data_root=config.DATA_DIR,
        dinov2_weights="/root/models/dinov2_vitl14_pretrain.pth",
        sam_weights="/root/models/sam_vit_h_4b8939.pth",
        img_size=1024,
        dinov2_size="vit_large",
        sam_size="vit_h",
        device=None,
)

def build_gfsam() -> GFSAM:
    return build_from_gfsam_configuration(get_gfsam_configuration())

class GFSAM(BaseScribe):
    """GF-SAM wrapper with automatic single-support selection.

    A support set is supplied at construction time. For each query image, the
    nearest support image is selected with DINOv2 pooled-feature L1 distance,
    then GF-SAM predicts from that selected support image and mask.
    """

    def __init__(
        self,
        configuration: GFSAMConfiguration
    ):
        self.dinov2_weights = str(configuration.dinov2_weights)
        self.sam_weights = str(configuration.sam_weights)
        self.img_size = int(configuration.img_size)
        self.dinov2_size = configuration.dinov2_size
        self.sam_size = configuration.sam_size
        self.device = configuration.device

        self.model = None
        self.transform = None
        self.query_tensor = None
        self.output_hw = None
        self.selected_support_index = None
        self.selected_support_score = None
        self.support_embeddings = None

        support_images, support_labels = configuration.get_support_data()
        if not support_images:
            raise RuntimeError(
                f"No GFSAM support images found in {configuration.support_data_root}. "
                "Expected images under raw/easy, raw/medium, or raw/hard."
            )
        self.support_images = [to_rgb_image(image) for image in support_images]
        self.support_labels = [BinaryMask.from_image(label) for label in support_labels]
        self.support_embeddings = None
        self.device = self.device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = build_model(
            Namespace(
                img_size=self.img_size,
                dinov2_size=self.dinov2_size,
                sam_size=self.sam_size,
                dinov2_weights=self.dinov2_weights,
                sam_weights=self.sam_weights,
                device=self.device,
            )
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
    def hasImage(self) -> bool:
        return self.query_tensor is not None

    def setImage(self, image):
        image = to_rgb_image(image)
        self.output_hw = image.shape[:2]
        self.query_tensor = self._to_model_tensor(image)
        self.selected_support_index, self.selected_support_score = self._select_support(image)
        return self

    @torch.inference_mode()
    def decode_mask(self, prompts=None) -> BinaryMask:
        if prompts is not None:
            raise ValueError("GFSAM does not accept point prompts.")
        if not self.hasImage():
            raise RuntimeError("No image is set. Call setImage(image) before decode_mask().")

        support_image = self.support_images[self.selected_support_index]
        support_tensor = self._to_model_tensor(support_image)
        support_mask = torch.tensor(self.support_labels[self.selected_support_index])
        support_mask = F.interpolate(
            support_mask.unsqueeze(0).unsqueeze(0).float(),
            support_tensor.size()[-2:],
            mode="nearest",
        )

        self.model.clear()
        self.model.set_reference(
            support_tensor.unsqueeze(0).unsqueeze(0).to(self.device),
            support_mask.to(self.device),
        )
        self.model.set_target(self.query_tensor.unsqueeze(0).to(self.device))
        pred_mask, _ = self.model.predict()

        mask = pred_mask.squeeze().detach().cpu().numpy() > 0.5
        return BinaryMask.from_bool(self._resize_mask(mask))

    def segment(self, image=None, prompts=None) -> BinaryMask:
        if image is not None:
            self.setImage(image)
        elif not self.hasImage():
            raise RuntimeError("No image provided and no image is set. Call setImage(image) first.")
        return self.decode_mask(prompts=prompts)

    def predict(self, image) -> BinaryMask:
        self.setImage(image)
        return self.decode_mask()

    def _select_support(self, image) -> tuple[int, float]:

        support_embeddings = self._embed_images(self.support_images)
        query_embedding = self._embed_images([image])
        distances = torch.abs(support_embeddings - query_embedding).sum(dim=1)
        index = int(torch.argmin(distances).item())
        return index, float(distances[index].item())

    @torch.inference_mode()
    def _embed_images(self, images, batch_size: int = 8):
        embeddings = []
        for start in range(0, len(images), batch_size):
            batch = torch.stack(
                [
                    self.model.encoder_transform(
                        self._resize_image(image)
                    )
                    for image in images[start:start + batch_size]
                ]
            )
            feats = self.model.encoder.forward_features(batch.to(self.device))["x_prenorm"][:, 1:]
            embeddings.append(F.normalize(feats.mean(dim=1), dim=-1, p=2).detach().cpu())
        return torch.cat(embeddings, dim=0)

    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.output_hw is None or tuple(mask.shape[:2]) == tuple(self.output_hw):
            return mask

        height, width = self.output_hw
        return np.asarray(
            cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        ) > 0

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

    def _to_model_tensor(self, image: np.ndarray) -> torch.Tensor:
        return self.transform(self._resize_image(image))

def to_rgb_image(image) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
