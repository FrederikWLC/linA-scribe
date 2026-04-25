from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image

def _ensure_workspace_src_on_path() -> None:
    candidates = [Path("/workspace/src")]

    resolved_file = Path(__file__).resolve()
    for parent in resolved_file.parents:
        if (parent / "fatesam2d_api").exists() and (parent / "scribe").exists():
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

MODEL_OPTIONS: dict[str, dict[str, object]] = {
    "modal-mobilesam": {
        "label": "BestMobileSAMv2Implementation",
        "import_path": "sam_api.modal_sam",
        "class_name": "ModalBestMobileSAMv2Implementation",
        "requires_set_image": True,
        "accepts_prompts": True,
    },
    "gaussian": {
        "label": "Gaussian",
        "import_path": "scribe.baselines.gaussian",
        "class_name": "Gaussian",
        "requires_set_image": False,
        "accepts_prompts": False,
    },
    "grabcut-auto-brush": {
        "label": "GC+brush",
        "import_path": "scribe.baselines.grabcut",
        "class_name": "GrabCutAutoBrush",
        "requires_set_image": False,
        "accepts_prompts": False,
    },
}

DEFAULT_MODEL_KEY = "modal-mobilesam"
MODAL_MODEL_KEYS = tuple(key for key in MODEL_OPTIONS if key.startswith("modal-"))


def _model_metadata(model_key: str) -> dict[str, object]:
    metadata = MODEL_OPTIONS.get(model_key)
    if metadata is None:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_key}")
    return metadata


def _load_model_factory(metadata: dict[str, object]) -> Callable[[], object]:
    module = importlib.import_module(str(metadata["import_path"]))
    return getattr(module, str(metadata["class_name"]))


def _read_model_label(model: object, fallback: str) -> str:
    return str(getattr(model, "name", fallback))


def _json_safe(value: object) -> object:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _extract_autoseed_prompts(set_image_result: object) -> list[object]:
    if not isinstance(set_image_result, dict):
        return []

    prompts = set_image_result.get("autoseed_prompts") or set_image_result.get("autoprompts") or []
    return prompts if isinstance(prompts, list) else []


async def _read_upload_as_grayscale(file: UploadFile) -> np.ndarray:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty")

    try:
        with Image.open(BytesIO(content)) as img:
            return np.asarray(img.convert("L"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc


def _mask_to_png(mask: object) -> bytes:
    image = mask.to_image() if hasattr(mask, "to_image") else np.asarray(mask)
    mask_image = Image.fromarray(np.asarray(image).astype(np.uint8), mode="L")
    output = BytesIO()
    mask_image.save(output, format="PNG")
    return output.getvalue()


@dataclass
class UserScribeState:
    models: dict[str, object] = field(default_factory=dict)
    image_hw: tuple[int, int] | None = None
    image: np.ndarray | None = None
    image_model_key: str | None = None


class ScribeSAMService:
    def __init__(self) -> None:
        self._states: dict[str, UserScribeState] = {}
        self._lock = threading.Lock()

    @property
    def model_options(self) -> list[dict[str, object]]:
        return [{"key": key, **metadata} for key, metadata in MODEL_OPTIONS.items()]

    def _get_state(self, username: str) -> UserScribeState:
        if username not in self._states:
            self._states[username] = UserScribeState()
        return self._states[username]

    def _get_model(self, state: UserScribeState, model_key: str) -> object:
        metadata = _model_metadata(model_key)
        if model_key not in state.models:
            state.models[model_key] = _load_model_factory(metadata)()
        return state.models[model_key]

    def warmup_user_models(self, username: str, model_keys: tuple[str, ...] = MODAL_MODEL_KEYS) -> dict[str, object]:
        results: dict[str, object] = {}
        with self._lock:
            state = self._get_state(username)

        for model_key in model_keys:
            try:
                with self._lock:
                    existing_model = state.models.get(model_key)

                if existing_model is None:
                    metadata = _model_metadata(model_key)
                    model = _load_model_factory(metadata)()
                    with self._lock:
                        state.models.setdefault(model_key, model)
                        model = state.models[model_key]
                else:
                    model = existing_model

                results[model_key] = {
                    "status": "ok",
                    "model": _read_model_label(model, str(_model_metadata(model_key)["label"])),
                }
            except Exception as exc:
                logger.exception("%s warmup failed for %s", model_key, username)
                results[model_key] = {"status": "error", "detail": str(exc)}

        return {"username": username, "models": results}

    async def set_image_from_upload(self, username: str, file: UploadFile, model_key: str = DEFAULT_MODEL_KEY) -> dict[str, object]:
        metadata = _model_metadata(model_key)
        if not metadata["requires_set_image"]:
            raise HTTPException(status_code=400, detail=f"{metadata['label']} does not use set-image")

        image = await _read_upload_as_grayscale(file)
        with self._lock:
            state = self._get_state(username)
            try:
                model = self._get_model(state, model_key)
                result = model.setImage(image) if hasattr(model, "setImage") else {"status": "cached"}
            except Exception as exc:
                logger.exception("%s setImage failed", metadata["label"])
                raise HTTPException(status_code=502, detail=f"{metadata['label']} setImage failed: {exc}") from exc

            state.image = image
            state.image_hw = tuple(int(value) for value in image.shape[:2])
            state.image_model_key = model_key

        return {
            "status": "ok",
            "width": state.image_hw[1],
            "height": state.image_hw[0],
            "model": _read_model_label(self._get_model(state, model_key), str(metadata["label"])),
            "model_key": model_key,
            "autoseed_prompts": _json_safe(_extract_autoseed_prompts(result)),
            "set_image": _json_safe(result),
        }

    def predict_mask_png(
        self,
        username: str,
        xs: list[float],
        ys: list[float],
        labels: list[int],
        coordinate_space: str,
        model_key: str = DEFAULT_MODEL_KEY,
    ) -> bytes:
        metadata = _model_metadata(model_key)
        if not metadata["requires_set_image"]:
            raise HTTPException(status_code=400, detail=f"{metadata['label']} should use upload prediction")

        with self._lock:
            state = self._get_state(username)

        if state.image_hw is None or state.image_model_key != model_key:
            raise HTTPException(status_code=409, detail="No image is set. Call set-image before predict.")

        prompts = None
        if metadata["accepts_prompts"] and xs:
            prompts = self._make_point_prompts(xs, ys, labels, coordinate_space, state.image_hw)

        with self._lock:
            try:
                state = self._get_state(username)
                model = self._get_model(state, model_key)
                if state.image is not None and getattr(model, "prefers_predict_with_image", False):
                    mask = model.predict(state.image, prompts=prompts)
                elif hasattr(model, "decode_mask_image"):
                    mask = model.decode_mask_image(prompts=prompts)
                elif hasattr(model, "decode_mask"):
                    mask = model.decode_mask(prompts=prompts) if metadata["accepts_prompts"] else model.decode_mask()
                elif state.image is not None:
                    mask = model.predict(state.image, prompts=prompts, autoprompt=False)
                else:
                    raise RuntimeError("No cached image is available")
            except Exception as exc:
                logger.exception("%s prediction failed", metadata["label"])
                raise HTTPException(status_code=502, detail=f"{metadata['label']} prediction failed: {exc}") from exc

        return _mask_to_png(mask)

    async def predict_upload_mask_png(
        self,
        username: str,
        file: UploadFile,
        xs: list[float],
        ys: list[float],
        labels: list[int],
        coordinate_space: str,
        model_key: str,
    ) -> bytes:
        metadata = _model_metadata(model_key)
        image = await _read_upload_as_grayscale(file)
        image_hw = tuple(int(value) for value in image.shape[:2])
        prompts = None
        if metadata["accepts_prompts"] and xs:
            prompts = self._make_point_prompts(xs, ys, labels, coordinate_space, image_hw)

        with self._lock:
            state = self._get_state(username)
            try:
                model = self._get_model(state, model_key)
                if metadata["accepts_prompts"]:
                    mask = model.predict(image, prompts=prompts)
                else:
                    mask = model.predict(image)
            except Exception as exc:
                logger.exception("%s upload prediction failed", metadata["label"])
                raise HTTPException(status_code=502, detail=f"{metadata['label']} prediction failed: {exc}") from exc

        return _mask_to_png(mask)

    def _make_point_prompts(
        self,
        xs: list[float],
        ys: list[float],
        labels: list[int],
        coordinate_space: str,
        image_hw: tuple[int, int] | None = None,
    ) -> list[PointPrompt]:
        if not (len(xs) == len(ys) == len(labels)):
            raise HTTPException(status_code=400, detail="x, y, and labels must have the same length")

        if coordinate_space not in {"percent", "pixel"}:
            raise HTTPException(status_code=400, detail="coordinate_space must be 'percent' or 'pixel'")

        height, width = image_hw or (0, 0)
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
