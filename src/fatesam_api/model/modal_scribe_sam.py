from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path

import modal
import numpy as np
from config import config
from scribe.base import Named

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
        raise RuntimeError("Timed out waiting for Modal app to start.")
    if _modal_app_start_error is not None:
        if isinstance(_modal_app_start_error, modal.exception.InvalidError) and "already running" in str(_modal_app_start_error):
            _modal_app_started = True
            return
        raise RuntimeError("Failed to start Modal app") from _modal_app_start_error

    _modal_app_started = True

APP_NAME = os.getenv("MODAL_APP_NAME", "FATESAM2D")
CHECKPOINT_DIR = Path("/root/checkpoints")
CHECKPOINT_FILE = config.SAM2_CHECKPOINT
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
CONFIG_FILE = config.SAM2_CONFIG

DATA_REMOTE_ROOT = Path("/root/data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["wget", "libgl1", "libglib2.0-0"])
    .pip_install(
        [
            "numpy",
            "opencv-python-headless",
            "optuna",
            "pandas",
            "scipy",
            "tqdm",
            "hydra-core",
            "omegaconf",
            "huggingface_hub",
            "pillow",
            "pyyaml",
            "torch",
            "torchvision",
        ]
    )
    .add_local_file(local_path=str(config.CONFIG_LOCAL_PATH), remote_path="/root/config.py")
    .add_local_file(local_path=str(config.DATA_SPLIT_LOCAL_PATH), remote_path="/root/data/split.py")
    .add_local_file(local_path=str(config.SAM2_CHECKPOINT_PATH), remote_path=f"/root/checkpoints/{config.SAM2_CHECKPOINT}")
    .add_local_file(local_path=str(config.SAM2_CONFIG_PATH), remote_path=f"/root/configs/{config.SAM2_CONFIG}")
    .add_local_dir(local_path=str(config.FATESAM_LOCAL_ROOT), remote_path="/root/fatesam-api")
    .add_local_dir(local_path=str(config.SCRIBE_LOCAL_ROOT), remote_path="/root/scribe")
    .add_local_dir(local_path=str(config.GT_LOCAL_ROOT), remote_path="/root/data/ground_truth/registered")
    .add_local_dir(local_path=str(config.RAW_LOCAL_ROOT), remote_path="/root/data/raw")
)

app = modal.App(APP_NAME)


def _ensure_checkpoint_file() -> str:
    import subprocess

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = f"{CHECKPOINT_DIR}/{CHECKPOINT_FILE}"
    if not os.path.exists(path):
        subprocess.run(["wget", "-O", path, CHECKPOINT_URL], check=True)
    return path


def _setup_runtime_env() -> None:
    if "/root" not in os.sys.path:
        os.sys.path.insert(0, "/root")
    if "/root/fatesam-api" not in os.sys.path:
        os.sys.path.insert(0, "/root/fatesam-api")
    os.environ.setdefault("SAM2_CHECKPOINT", CHECKPOINT_FILE)
    os.environ.setdefault("SAM2_CONFIG", CONFIG_FILE)


@app.cls(image=image, gpu="T4", timeout=30 * 60)
class FATESAM2DInterface:
    @modal.enter()
    def setup(self) -> None:
        from data.split import get_training_data
        from fatesam_api.model.scribe_sam import FATESAM2D

        _ensure_checkpoint_file()
        _setup_runtime_env()
        support_images, support_labels, _ = get_training_data(seed=42, data_root=DATA_REMOTE_ROOT)
        self.model = FATESAM2D(
            support_images=support_images,
            support_labels=support_labels,
        )
    def _get_model(self):
        if getattr(self, "model", None) is None:
            self.setup()
        return self.model

    @modal.method()
    def ping(self) -> str:
        return f"ready: {_ensure_checkpoint_file()}"

    @modal.method()
    def smoke(self) -> dict[str, str]:
        from fatesam_api.model.scribe_sam import FATESAM2D
        _ = FATESAM2D(
            support_images=[],
            support_labels=[]
        )
        return {"status": "ok", "message": "FATESAM2D predictor loaded on Modal GPU"}

    @modal.method()
    def setImage(self, image) -> dict[str, object]:
        model = self._get_model()
        model.setImage(np.asarray(image))
        return {
            "status": "ok",
            "output_hw": [] if getattr(model, "_output_hw", None) is None else [int(x) for x in model._output_hw],
        }

    @modal.method()
    def decode_mask(self, prompts=None):
        model = self._get_model()
        return model.decode_mask(prompts=prompts)

    @modal.method()
    def decode_mask_image(self, prompts=None):
        model = self._get_model()
        return model.decode_mask(prompts=prompts).to_image()

    @modal.method()
    def predict(self, image, prompts=None):
        model = self._get_model()
        return model.segment(np.asarray(image), prompts=prompts)

    def segment(self, image, prompts=None):
        model = self._get_model()
        return model.segment(np.asarray(image), prompts=prompts)

class ModalFATESAM2D(Named):
    def __init__(
        self
    ):
        _ensure_modal_app_started()
        self.interface = FATESAM2DInterface()

    def predict(self, image, prompts=None):
        return self.interface.predict.remote(image=image, prompts=prompts)

    def setImage(self, image) -> dict[str, object]:
        return self.interface.setImage.remote(image=image)

    def decode_mask(self, prompts=None):
        return self.interface.decode_mask.remote(prompts=prompts)

    def decode_mask_image(self, prompts=None):
        return self.interface.decode_mask_image.remote(prompts=prompts)


ScribeSAMInterface = FATESAM2DInterface
ModalScribeSAM = ModalFATESAM2D

