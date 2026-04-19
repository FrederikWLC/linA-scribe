from __future__ import annotations

from pathlib import Path

import numpy as np

from scribe.base import Named
from scribe.binary_mask import BinaryMask


class GFSAM(Named):
    def __init__(
        self,
        dinov2_weights: str | Path,
        sam_weights: str | Path,
        img_size: int = 1024,
        dinov2_size: str = "vit_large",
        sam_size: str = "vit_h",
        device=None,
    ):
        self.dinov2_weights = str(dinov2_weights)
        self.sam_weights = str(sam_weights)
        self.img_size = img_size
        self.dinov2_size = dinov2_size
        self.sam_size = sam_size
        self.device = device
        self.model = None
        self.transform = None

    def _ensure_model(self):
        if self.model is not None:
            return

        import sys
        from argparse import Namespace
        from pathlib import Path

        import torch
        import torchvision.transforms as transforms

        gf_sam_root = Path(__file__).resolve().parents[1] / "gf_sam"
        if str(gf_sam_root) not in sys.path:
            sys.path.insert(0, str(gf_sam_root))

        from matcher.GFSAM import build_model

        device = self.device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    def predict(self, image, support_image, support_mask) -> BinaryMask:
        self._ensure_model()

        import torch
        import torch.nn.functional as F
        from PIL import Image

        query = Image.fromarray(np.asarray(image)).convert("RGB")
        support = Image.fromarray(np.asarray(support_image)).convert("RGB")
        support_mask_array = np.asarray(support_mask)

        query_tensor = self.transform(query)
        support_tensor = self.transform(support)
        support_mask_tensor = torch.tensor(support_mask_array)
        support_mask_tensor = F.interpolate(
            support_mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            support_tensor.size()[-2:],
            mode="nearest",
        )

        with torch.no_grad():
            self.model.clear()
            self.model.set_reference(
                support_tensor.unsqueeze(0).unsqueeze(0).to(self.device),
                support_mask_tensor.to(self.device),
            )
            self.model.set_target(query_tensor.unsqueeze(0).to(self.device))
            pred_mask, _ = self.model.predict()

        mask = pred_mask.squeeze().detach().cpu().numpy() > 0.5
        return BinaryMask.from_bool(mask)
