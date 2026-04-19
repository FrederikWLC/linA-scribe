from __future__ import annotations

import os
import sys
import threading
import logging
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image


def _ensure_workspace_src_on_path() -> None:
    candidates = [Path("/workspace/src")]

    resolved_file = Path(__file__).resolve()
    for parent in resolved_file.parents:
        if (parent / "fatesam_api").exists() and (parent / "scribe").exists():
            candidates.append(parent)

    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _normalize_modal_token_env() -> None:
    if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
        return

    api_token = os.getenv("MODAL_API_TOKEN", "").strip()
    if not api_token:
        return

    for separator in (":", ",", " "):
        if separator in api_token:
            token_id, token_secret = [part.strip() for part in api_token.split(separator, 1)]
            if token_id and token_secret:
                os.environ.setdefault("MODAL_TOKEN_ID", token_id)
                os.environ.setdefault("MODAL_TOKEN_SECRET", token_secret)
            return


_ensure_workspace_src_on_path()
_normalize_modal_token_env()

from scribe.prompts import PointPrompt  # noqa: E402

logger = logging.getLogger(__name__)


class ScribeSAMService:
    def __init__(self) -> None:
        self.model = None
        self._lock = threading.Lock()
        self._image_hw: tuple[int, int] | None = None

    def _get_model(self):
        if self.model is None:
            from fatesam_api.model.modal_scribe_sam import ModalFATESAM2D

            self.model = ModalFATESAM2D()
        return self.model

    @property
    def image_hw(self) -> tuple[int, int] | None:
        return self._image_hw

    async def set_image_from_upload(self, file: UploadFile) -> dict[str, object]:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Image file is empty")

        try:
            with Image.open(BytesIO(content)) as img:
                image = np.asarray(img.convert("L"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

        with self._lock:
            try:
                model = self._get_model()
                result = model.setImage(image)
            except Exception as exc:
                logger.exception("ModalFATESAM2D setImage failed")
                raise HTTPException(status_code=502, detail=f"ModalFATESAM2D setImage failed: {exc}") from exc

            self._image_hw = tuple(int(value) for value in image.shape[:2])

        return {
            "status": "ok",
            "width": self._image_hw[1],
            "height": self._image_hw[0],
            "model": self._get_model().name,
            "set_image": result,
        }

    def predict_mask_png(
        self,
        xs: list[float],
        ys: list[float],
        labels: list[int],
        coordinate_space: str,
    ) -> bytes:
        if self._image_hw is None:
            raise HTTPException(status_code=409, detail="No image is set. Call set-image before predict.")

        prompts = self._make_point_prompts(xs, ys, labels, coordinate_space)

        with self._lock:
            try:
                mask = self._get_model().decode_mask_image(prompts=prompts)
            except Exception as exc:
                logger.exception("ModalFATESAM2D prediction failed")
                raise HTTPException(status_code=502, detail=f"ModalFATESAM2D prediction failed: {exc}") from exc

        mask_image = Image.fromarray(np.asarray(mask), mode="L")
        output = BytesIO()
        mask_image.save(output, format="PNG")
        return output.getvalue()

    def _make_point_prompts(
        self,
        xs: list[float],
        ys: list[float],
        labels: list[int],
        coordinate_space: str,
    ) -> list[PointPrompt]:
        if not (len(xs) == len(ys) == len(labels)):
            raise HTTPException(status_code=400, detail="x, y, and labels must have the same length")

        if coordinate_space not in {"percent", "pixel"}:
            raise HTTPException(status_code=400, detail="coordinate_space must be 'percent' or 'pixel'")

        height, width = self._image_hw or (0, 0)
        prompts = []
        for x, y, label in zip(xs, ys, labels):
            if label not in {0, 1}:
                raise HTTPException(status_code=400, detail="Point labels must be 1 for foreground or 0 for background")

            if coordinate_space == "percent":
                point_x = round((float(x) / 100) * max(width - 1, 0))
                point_y = round((float(y) / 100) * max(height - 1, 0))
            else:
                point_x = round(float(x))
                point_y = round(float(y))

            prompts.append(PointPrompt(x=point_x, y=point_y, label=int(label)))

        return prompts


scribe_sam_service = ScribeSAMService()
