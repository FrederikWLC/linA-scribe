from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from pathlib import Path

import modal
import numpy as np
from optuna import Trial

from config import config
from data.split import get_training_data
from fatesam2d_api.FATESAM2D import FATESAM2D, FATESAM2DAutoPoint
from scribe.base import Named


APP_NAME = os.getenv("MODAL_APP_NAME", "FATESAM2D")
CHECKPOINT_DIR = Path("/root/checkpoints")
CHECKPOINT_FILE = config.SAM2_CHECKPOINT
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
CONFIG_FILE = config.SAM2_CONFIG
DATA_REMOTE_ROOT = Path("/root/data")
FATESAM_REMOTE_ROOT = "/root/fatesam_api"

_modal_app_started = False
_modal_app_thread: threading.Thread | None = None
_modal_app_start_event = threading.Event()
_modal_app_start_error: BaseException | None = None


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
    .add_local_dir(local_path=str(config.FATESAM_LOCAL_ROOT), remote_path=FATESAM_REMOTE_ROOT)
    .add_local_dir(local_path=str(config.SCRIBE_LOCAL_ROOT), remote_path="/root/scribe")
    .add_local_dir(local_path=str(config.GT_LOCAL_ROOT), remote_path="/root/data/ground_truth/registered")
    .add_local_dir(local_path=str(config.RAW_LOCAL_ROOT), remote_path="/root/data/raw")
)

app = modal.App(APP_NAME)


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


def _ensure_modal_app_started(timeout_seconds: int = 60) -> None:
    global _modal_app_started, _modal_app_thread
    if _modal_app_started:
        return

    if _modal_app_thread is None or not _modal_app_thread.is_alive():
        _modal_app_start_event.clear()
        _modal_app_thread = threading.Thread(target=_modal_app_thread_target, daemon=True)
        _modal_app_thread.start()

    if not _modal_app_start_event.wait(timeout=timeout_seconds):
        raise RuntimeError("Timed out waiting for Modal FATESAM2D app to start.")
    if _modal_app_start_error is not None:
        if isinstance(_modal_app_start_error, modal.exception.InvalidError) and "already running" in str(_modal_app_start_error):
            _modal_app_started = True
            return
        raise RuntimeError("Failed to start Modal FATESAM2D app") from _modal_app_start_error

    _modal_app_started = True


def _ensure_checkpoint_file() -> str:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_FILE
    if not checkpoint_path.exists():
        subprocess.run(["wget", "-O", str(checkpoint_path), CHECKPOINT_URL], check=True)
    return str(checkpoint_path)


def _setup_runtime_env() -> None:
    if "/root" not in os.sys.path:
        os.sys.path.insert(0, "/root")
    if FATESAM_REMOTE_ROOT not in os.sys.path:
        os.sys.path.insert(0, FATESAM_REMOTE_ROOT)
    os.environ.setdefault("SAM2_CHECKPOINT", CHECKPOINT_FILE)
    os.environ.setdefault("SAM2_CONFIG", CONFIG_FILE)


@app.cls(image=image, gpu="T4", timeout=30 * 60)
class FATESAM2DInterface:
    def _setup_model(self) -> None:
        _setup_runtime_env()
        _ensure_checkpoint_file()
        support_images, support_labels, _ = get_training_data(seed=42, data_root=DATA_REMOTE_ROOT)
        self.model = FATESAM2D(support_images=support_images, support_labels=support_labels)

    @modal.enter()
    def setup(self) -> None:
        self._setup_model()

    def _get_model(self) -> FATESAM2D:
        if getattr(self, "model", None) is None:
            self._setup_model()
        return self.model

    @modal.method()
    def ping(self) -> str:
        return f"ready: {_ensure_checkpoint_file()}"

    @modal.method()
    def setImage(self, image) -> dict[str, object]:
        model = self._get_model()
        model.setImage(np.asarray(image))
        output_hw = getattr(model, "_output_hw", None)
        return {
            "status": "ok",
            "output_hw": [] if output_hw is None else [int(x) for x in output_hw],
        }

    @modal.method()
    def decode_mask(self, prompts=None):
        return self._get_model().decode_mask(prompts=prompts)

    @modal.method()
    def decode_mask_image(self, prompts=None):
        return self._get_model().decode_mask(prompts=prompts).to_image()

    @modal.method()
    def predict(self, image, prompts=None):
        return self._get_model().predict(np.asarray(image), prompts=prompts, autoprompt=False)

@app.cls(image=image, gpu="T4", timeout=30 * 60)
class FATESAM2DAutoPointInterface:

    def _setup_model(self) -> None:
        _setup_runtime_env()
        _ensure_checkpoint_file()
        support_images, support_labels, _ = get_training_data(seed=42, data_root=DATA_REMOTE_ROOT)
        self.model = FATESAM2DAutoPoint(support_images=support_images, support_labels=support_labels)

    @modal.enter()
    def setup(self) -> None:
        self._setup_model()
        
    def _get_model(self) -> FATESAM2DAutoPoint:
        if getattr(self, "model", None) is None:
            self._setup_model()
        return self.model

    @modal.method()
    def ping(self) -> str:
        return f"ready: {_ensure_checkpoint_file()}"

    @modal.method()
    def setImage(self, image) -> dict[str, object]:
        model = self._get_model()
        model.setImage(np.asarray(image))
        output_hw = getattr(model, "_output_hw", None)
        return {
            "status": "ok",
            "output_hw": [] if output_hw is None else [int(x) for x in output_hw],
        }

    @modal.method()
    def decode_mask(self, prompts=None):
        return self._get_model().decode_mask(prompts=prompts)

    @modal.method()
    def decode_mask_image(self, prompts=None):
        return self._get_model().decode_mask(prompts=prompts).to_image()

    @modal.method()
    def predict(self, image, prompts=None):
        return self._get_model().predict(np.asarray(image), prompts=prompts, autoprompt=True)

    @modal.method()
    def set_hyperparameters(self, **kwargs) -> None:
        self._get_model().set_hyperparameters(**kwargs)

    @modal.method()
    def hyperparameters(self) -> dict:
        return self._get_model().hyperparameters
    
    @modal.method()
    def hyperparameter_ranges(self, trial: Trial) -> dict:
        return FATESAM2DAutoPoint.hyperparameter_ranges(trial)

class ModalFATESAM2D(Named):
    NAME = FATESAM2D.NAME
    SHORT_NAME = FATESAM2D.SHORT_NAME

    def __init__(self):
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


class ModalFATESAM2DAutoPoint(ModalFATESAM2D):
    NAME = FATESAM2DAutoPoint.NAME
    SHORT_NAME = FATESAM2DAutoPoint.SHORT_NAME

    def __init__(self):
        _ensure_modal_app_started()
        self.interface = FATESAM2DAutoPointInterface()

    @property
    def hyperparameters(self) -> dict:
        return self.interface.hyperparameters.remote()
    
    @classmethod
    def hyperparameter_ranges(cls, trial: Trial) -> dict:
        return FATESAM2DAutoPoint.hyperparameter_ranges(trial)
    
    def set_hyperparameters(self, **kwargs):
        self.interface.set_hyperparameters.remote(**kwargs)
