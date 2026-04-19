from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import modal
import numpy as np

from scribe.base import Named
from scribe.binary_mask import BinaryMask


APP_NAME = os.getenv("GF_SAM_MODAL_APP_NAME", "GFSAM")
GF_SAM_REMOTE_ROOT = Path("/root/gf_sam")
MODEL_DIR = Path("/root/models")
DINO_WEIGHTS = MODEL_DIR / "dinov2_vitl14_pretrain.pth"
SAM_WEIGHTS = MODEL_DIR / "sam_vit_h_4b8939.pth"
DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

LOCAL_SRC_ROOT = Path(__file__).resolve().parents[1]
LOCAL_GF_SAM_ROOT = LOCAL_SRC_ROOT / "gf_sam"
LOCAL_GF_SAM_API_ROOT = LOCAL_SRC_ROOT / "gf_sam_api"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["git", "wget", "libgl1", "libglib2.0-0"])
    .pip_install(
        [
            "torch==1.13.1",
            "torchvision==0.14.1",
            "future==0.18.2",
            "gradio==3.32.0",
            "gradio-client==0.2.5",
            "matplotlib==3.7.5",
            "torchmetrics==0.11.0",
            "torchshow==0.5.0",
            "opencv-python-headless==4.6.0.66",
            "timm==0.6.12",
            "POT==0.9.0",
            "omegaconf",
            "iopath",
            "numpy==1.24.4",
            "tqdm==4.64.1",
            "scipy",
            "pillow",
        ]
    )
    .add_local_dir(local_path=str(LOCAL_GF_SAM_ROOT), remote_path=str(GF_SAM_REMOTE_ROOT))
    .add_local_dir(local_path=str(LOCAL_GF_SAM_API_ROOT), remote_path="/root/gf_sam_api")
)

app = modal.App(APP_NAME)

_modal_app_started = False
_modal_app_thread: threading.Thread | None = None
_modal_app_start_event = threading.Event()
_modal_app_start_error: Exception | None = None


async def _run_modal_app_background() -> None:
    global _modal_app_start_error
    try:
        async with app.run():
            _modal_app_start_event.set()
            await asyncio.Event().wait()
    except BaseException as exc:
        _modal_app_start_error = exc
        _modal_app_start_event.set()


def _modal_app_thread_target() -> None:
    try:
        asyncio.run(_run_modal_app_background())
    except BaseException as exc:
        global _modal_app_start_error
        _modal_app_start_error = exc
        _modal_app_start_event.set()


def _ensure_modal_app_started() -> None:
    global _modal_app_started, _modal_app_thread, _modal_app_start_error
    if _modal_app_started:
        return

    if _modal_app_thread is None or not _modal_app_thread.is_alive():
        _modal_app_start_event.clear()
        _modal_app_start_error = None
        _modal_app_thread = threading.Thread(target=_modal_app_thread_target, daemon=True)
        _modal_app_thread.start()

    if not _modal_app_start_event.wait(timeout=60):
        raise RuntimeError("Timed out waiting for Modal GFSAM app to start.")
    if _modal_app_start_error is not None:
        if isinstance(_modal_app_start_error, modal.exception.InvalidError) and "already running" in str(_modal_app_start_error):
            _modal_app_started = True
            return
        raise RuntimeError("Failed to start Modal GFSAM app") from _modal_app_start_error

    _modal_app_started = True


def _ensure_weights() -> tuple[str, str]:
    import subprocess

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not DINO_WEIGHTS.exists():
        subprocess.run(["wget", "-O", str(DINO_WEIGHTS), DINO_URL], check=True)
    if not SAM_WEIGHTS.exists():
        subprocess.run(["wget", "-O", str(SAM_WEIGHTS), SAM_URL], check=True)
    return str(DINO_WEIGHTS), str(SAM_WEIGHTS)


@app.cls(image=image, gpu="T4", timeout=30 * 60)
class GFSAMInterface:
    @modal.enter()
    def setup(self) -> None:
        import sys

        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        if str(GF_SAM_REMOTE_ROOT) not in sys.path:
            sys.path.insert(0, str(GF_SAM_REMOTE_ROOT))

        from gf_sam_api.gf_sam import GFSAM

        dinov2_weights, sam_weights = _ensure_weights()
        self.model = GFSAM(
            dinov2_weights=dinov2_weights,
            sam_weights=sam_weights,
        )
        self.model._ensure_model()

    def _get_model(self):
        if getattr(self, "model", None) is None:
            self.setup()
        return self.model

    @modal.method()
    def predict_mask_image(self, image, support_image, support_mask):
        mask = self._get_model().predict(
            image=np.asarray(image),
            support_image=np.asarray(support_image),
            support_mask=np.asarray(support_mask),
        )
        return mask.to_image()


class ModalGFSAM(Named):
    def __init__(self):
        _ensure_modal_app_started()
        self.interface = GFSAMInterface()

    def predict(self, image, support_image, support_mask) -> BinaryMask:
        mask_image = self.interface.predict_mask_image.remote(
            image=image,
            support_image=support_image,
            support_mask=support_mask,
        )
        return BinaryMask.from_image(np.asarray(mask_image))
