from __future__ import annotations

import logging
from pathlib import Path
import time

import numpy as np

from scribe.base import Named
from scribe.binary_mask import BinaryMask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GFSAM(Named):
    def __init__(
        self,
        dinov2_weights: str | Path,
        sam_weights: str | Path,
        img_size: int = 1024,
        dinov2_size: str = "vit_large",
        sam_size: str = "vit_h",
        device=None,
        support_images=None,
        support_labels=None,
    ):
        self.dinov2_weights = str(dinov2_weights)
        self.sam_weights = str(sam_weights)
        self.img_size = img_size
        self.dinov2_size = dinov2_size
        self.sam_size = sam_size
        self.device = device
        self.model = None
        self.transform = None
        self.support_images = None
        self.support_labels = None
        self._support_embeddings = None
        self.query_image = None
        self.query_tensor = None
        self._output_hw = None
        self.selected_support_index = None
        self.selected_support_score = None
        logger.info(
            "GFSAM.__init__: img_size=%s dinov2_size=%s sam_size=%s support_count=%s",
            self.img_size,
            self.dinov2_size,
            self.sam_size,
            0 if support_images is None else len(support_images),
        )
        self._set_supports(support_images, support_labels)
        self._ensure_model()
        
    def _ensure_model(self):
        if self.model is not None:
            return

        t0 = time.perf_counter()

        import sys
        from argparse import Namespace
        from pathlib import Path

        import torch
        import torchvision.transforms as transforms

        gf_sam_root = Path(__file__).resolve().parent / "vendor" / "gf_sam"
        if str(gf_sam_root) not in sys.path:
            sys.path.insert(0, str(gf_sam_root))

        from matcher.GFSAM import build_model

        device = self.device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(
            "GFSAM._ensure_model: building model on device=%s gf_sam_root=%s",
            device,
            gf_sam_root,
        )
        logger.info(
            "GFSAM._ensure_model: dinov2_weights=%s sam_weights=%s",
            self.dinov2_weights,
            self.sam_weights,
        )
        args = Namespace(
            img_size=self.img_size,
            dinov2_size=self.dinov2_size,
            sam_size=self.sam_size,
            dinov2_weights=self.dinov2_weights,
            sam_weights=self.sam_weights,
            device=device,
        )
        self.model = build_model(args)
        self.transform = transforms.Compose([
            transforms.Resize(size=(self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        self.device = device
        logger.info("GFSAM._ensure_model: model ready in %.3fs", time.perf_counter() - t0)

    def _as_rgb_image(self, image):
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.fromarray(np.asarray(image)).convert("RGB")

    def _as_mask_array(self, mask) -> np.ndarray:
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        if mask_array.max(initial=0) > 1:
            return BinaryMask.from_image(mask_array).astype(np.uint8)
        return (mask_array > 0).astype(np.uint8)

    def _set_supports(self, support_images, support_labels):
        if support_images is None or support_labels is None:
            raise ValueError("Both support_images and support_labels are required.")
        if len(support_images) != len(support_labels):
            raise ValueError("support_images and support_labels must have the same length.")
        if len(support_images) == 0:
            raise ValueError("At least one support image is required.")

        self.support_images = [np.asarray(image) for image in support_images]
        self.support_labels = [self._as_mask_array(label) for label in support_labels]
        self._support_embeddings = None
        self.selected_support_index = None
        self.selected_support_score = None
        logger.info("GFSAM._set_supports: loaded %d support images", len(self.support_images))
        if self.support_images:
            logger.info(
                "GFSAM._set_supports: first support image shape=%s first label shape=%s",
                self.support_images[0].shape,
                self.support_labels[0].shape,
            )
        return self

    def hasImage(self):
        return self.query_image is not None

    def _embed_images(self, images, batch_size: int = 8):
        self._ensure_model()

        import torch
        import torch.nn.functional as F

        t0 = time.perf_counter()
        logger.info("GFSAM._embed_images: embedding %d images with batch_size=%d", len(images), batch_size)
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch_images = images[start:start + batch_size]
                # GF-SAM prediction sees square-resized images. Do the same for
                # support selection so DINO tensors can be stacked.
                batch = torch.stack([
                    self.model.encoder_transform(self._as_rgb_image(image).resize((self.img_size, self.img_size)))
                    for image in batch_images
                ])
                logger.debug("GFSAM._embed_images: batch tensor shape %s", tuple(batch.shape))
                feats = self.model.encoder.forward_features(batch.to(self.device))["x_prenorm"][:, 1:]
                pooled = F.normalize(feats.mean(dim=1), dim=-1, p=2)
                embeddings.append(pooled.detach().cpu())
                logger.info(
                    "GFSAM._embed_images: embedded %d/%d images",
                    min(start + batch_size, len(images)),
                    len(images),
                )
        result = torch.cat(embeddings, dim=0)
        logger.info(
            "GFSAM._embed_images: done, embedding shape %s in %.3fs",
            tuple(result.shape),
            time.perf_counter() - t0,
        )
        return result

    def _ensure_support_embeddings(self):
        if self.support_images is None:
            raise RuntimeError("No supports are set. Call set_supports(...) before setImage(...).")
        if self._support_embeddings is None:
            logger.info(
                "GFSAM._ensure_support_embeddings: computing embeddings for %d supports",
                len(self.support_images),
            )
            self._support_embeddings = self._embed_images(self.support_images)
        else:
            logger.info(
                "GFSAM._ensure_support_embeddings: using cached support embeddings with shape %s",
                tuple(self._support_embeddings.shape),
            )
        return self._support_embeddings

    def _select_support(self, image):
        import torch

        t0 = time.perf_counter()
        support_embeddings = self._ensure_support_embeddings()
        query_embedding = self._embed_images([image])
        distances = torch.abs(support_embeddings - query_embedding).sum(dim=1)
        index = int(torch.argmin(distances).item())
        score = float(distances[index].item())
        logger.info(
            "GFSAM._select_support: selected support index=%d score=%.6f from %d supports in %.3fs",
            index,
            score,
            len(support_embeddings),
            time.perf_counter() - t0,
        )
        logger.debug("GFSAM._select_support: all distances=%s", distances.tolist())
        return index, score

    def setImage(self, image):
        self._ensure_model()

        t0 = time.perf_counter()
        query = self._as_rgb_image(image)
        self.query_image = query
        self.query_tensor = self.transform(query)
        self._output_hw = np.asarray(image).shape[:2]
        logger.info(
            "GFSAM.setImage: query shape=%s transformed tensor shape=%s output_hw=%s",
            np.asarray(image).shape,
            tuple(self.query_tensor.shape),
            self._output_hw,
        )
        self.selected_support_index, self.selected_support_score = self._select_support(image)
        logger.info(
            "GFSAM.setImage: image set with selected_support_index=%s selected_support_score=%.6f in %.3fs",
            self.selected_support_index,
            self.selected_support_score,
            time.perf_counter() - t0,
        )
        return self

    set_supports = _set_supports

    def _resize_mask_to_output(self, mask: np.ndarray) -> np.ndarray:
        if self._output_hw is None or tuple(mask.shape[:2]) == tuple(self._output_hw):
            return mask

        from PIL import Image

        height, width = self._output_hw
        logger.info("GFSAM._resize_mask_to_output: resizing mask from %s to %s", mask.shape[:2], self._output_hw)
        resized = Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(resized) > 0

    def _predict_with_support(self, image, support_image, support_mask) -> BinaryMask:
        self._ensure_model()

        import torch
        import torch.nn.functional as F

        query = self._as_rgb_image(image)
        support = self._as_rgb_image(support_image)
        support_mask_array = self._as_mask_array(support_mask)

        query_tensor = self.transform(query)
        support_tensor = self.transform(support)
        support_mask_tensor = torch.tensor(support_mask_array)
        support_mask_tensor = F.interpolate(
            support_mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            support_tensor.size()[-2:],
            mode="nearest",
        )

        logger.info(
            "GFSAM._predict_with_support: query_tensor=%s support_tensor=%s support_mask_tensor=%s",
            tuple(query_tensor.shape),
            tuple(support_tensor.shape),
            tuple(support_mask_tensor.shape),
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            self.model.clear()
            self.model.set_reference(
                support_tensor.unsqueeze(0).unsqueeze(0).to(self.device),
                support_mask_tensor.to(self.device),
            )
            self.model.set_target(query_tensor.unsqueeze(0).to(self.device))
            pred_mask, _ = self.model.predict()
        logger.info("GFSAM._predict_with_support: model.predict done in %.3fs", time.perf_counter() - t0)

        mask = pred_mask.squeeze().detach().cpu().numpy() > 0.5
        logger.info("GFSAM._predict_with_support: raw prediction mask shape=%s foreground_pixels=%d", mask.shape, int(mask.sum()))
        output_hw = np.asarray(image).shape[:2]
        if tuple(mask.shape[:2]) != tuple(output_hw):
            from PIL import Image

            height, width = output_hw
            logger.info("GFSAM._predict_with_support: resizing output mask from %s to %s", mask.shape[:2], output_hw)
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST)) > 0
        return BinaryMask.from_bool(mask)

    def decode_mask(self, prompts=None) -> BinaryMask:
        if prompts is not None:
            raise ValueError("GFSAM does not accept point prompts. Call decode_mask() without prompts.")
        if not self.hasImage():
            raise RuntimeError("No image is set. Call setImage(image) before decode_mask().")
        if self.selected_support_index is None:
            raise RuntimeError("No support was selected. Call setImage(image) after set_supports(...).")

        self._ensure_model()

        import torch
        import torch.nn.functional as F

        support = self._as_rgb_image(self.support_images[self.selected_support_index])
        support_tensor = self.transform(support)
        support_mask_tensor = torch.tensor(self.support_labels[self.selected_support_index])
        support_mask_tensor = F.interpolate(
            support_mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            support_tensor.size()[-2:],
            mode="nearest",
        )

        logger.info(
            "GFSAM.decode_mask: selected_support_index=%d support_score=%.6f query_tensor=%s support_tensor=%s support_mask_tensor=%s",
            self.selected_support_index,
            self.selected_support_score,
            tuple(self.query_tensor.shape),
            tuple(support_tensor.shape),
            tuple(support_mask_tensor.shape),
        )
        t0 = time.perf_counter()
        with torch.no_grad():
            self.model.clear()
            self.model.set_reference(
                support_tensor.unsqueeze(0).unsqueeze(0).to(self.device),
                support_mask_tensor.to(self.device),
            )
            self.model.set_target(self.query_tensor.unsqueeze(0).to(self.device))
            pred_mask, _ = self.model.predict()
        logger.info("GFSAM.decode_mask: model.predict done in %.3fs", time.perf_counter() - t0)

        mask = pred_mask.squeeze().detach().cpu().numpy() > 0.5
        logger.info("GFSAM.decode_mask: raw prediction mask shape=%s foreground_pixels=%d", mask.shape, int(mask.sum()))
        resized = self._resize_mask_to_output(mask)
        logger.info("GFSAM.decode_mask: final mask shape=%s foreground_pixels=%d", resized.shape, int(resized.sum()))
        return BinaryMask.from_bool(resized)

    def segment(self, image=None, prompts=None) -> BinaryMask:
        if image is not None:
            self.setImage(image)
        elif not self.hasImage():
            raise RuntimeError("No image provided and no image is set. Call setImage(image) first.")
        return self.decode_mask(prompts=prompts)

    def predict(self, image, support_image=None, support_mask=None) -> BinaryMask:
        if support_image is not None or support_mask is not None:
            if support_image is None or support_mask is None:
                raise ValueError("support_image and support_mask must be provided together.")
            logger.info("GFSAM.predict: running explicit support prediction")
            return self._predict_with_support(image, support_image, support_mask)

        logger.info("GFSAM.predict: running stateful setImage + decode_mask")
        self.setImage(image)
        return self.decode_mask()
