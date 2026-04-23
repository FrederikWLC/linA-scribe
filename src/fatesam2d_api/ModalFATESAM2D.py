from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from pathlib import Path

import modal

from config import config


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


class FATESAM2DInterfaceTemplate:
    def _setup_model(self) -> None:
        import numpy as np  # noqa: F401
        from data.split import get_support_data
        from fatesam2d_api.FATESAM2D import FATESAM2D

        _setup_runtime_env()
        _ensure_checkpoint_file()
        support_images, support_labels, _ = get_support_data(data_root=DATA_REMOTE_ROOT)
        self.model = FATESAM2D(support_images=support_images, support_labels=support_labels)

    @modal.enter()
    def setup(self) -> None:
        self._setup_model()

    def _get_model(self):
        if getattr(self, "model", None) is None:
            self._setup_model()
        return self.model

    @modal.method()
    def ping(self) -> str:
        return f"ready: {_ensure_checkpoint_file()}"

    @modal.method()
    def setImage(self, image) -> dict[str, object]:
        import numpy as np

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
        import numpy as np

        return self._get_model().predict(np.asarray(image), prompts=prompts, autoprompt=False)


@app.cls(image=image, gpu="A10", timeout=30 * 60)
class FATESAM2DInterface(FATESAM2DInterfaceTemplate):
    pass

@app.cls(image=image, gpu="A10", timeout=30 * 60)
class FATESAM2DBlankInterface(FATESAM2DInterfaceTemplate):
    def _setup_model(self) -> None:
        from data.split import get_support_data
        from fatesam2d_api.FATESAM2D import FATESAM2DBlank

        _setup_runtime_env()
        _ensure_checkpoint_file()
        support_images, support_labels, _ = get_support_data(data_root=DATA_REMOTE_ROOT)
        print(f"ModalFATESAM2DBlank: loaded support_images={len(support_images)}, support_labels={len(support_labels)}")
        if not support_images or not support_labels:
            raise RuntimeError("ModalFATESAM2DBlank failed to load support data from /root/data")
        self.model = FATESAM2DBlank(support_images=support_images, support_labels=support_labels)

@app.cls(image=image, gpu="A10", timeout=30 * 60)
class FATESAM2DAutoPointInterface(FATESAM2DInterfaceTemplate):

    def _setup_model(self) -> None:
        from data.split import get_support_data
        from fatesam2d_api.FATESAM2D import FATESAM2DAutoPoint

        _setup_runtime_env()
        _ensure_checkpoint_file()
        support_images, support_labels, _ = get_support_data(data_root=DATA_REMOTE_ROOT)
        print(f"ModalFATESAM2DAutoPoint: loaded support_images={len(support_images)}, support_labels={len(support_labels)}")
        if not support_images or not support_labels:
            raise RuntimeError("ModalFATESAM2DAutoPoint failed to load support data from /root/data")
        self.model = FATESAM2DAutoPoint(support_images=support_images, support_labels=support_labels)

    @modal.method()
    def set_hyperparameters(self, **kwargs) -> None:
        self._get_model().set_hyperparameters(**kwargs)

    @modal.method()
    def hyperparameters(self) -> dict:
        return self._get_model().hyperparameters

    @modal.method()
    def hyperparameter_ranges(self, trial) -> dict:
        from fatesam2d_api.FATESAM2D import FATESAM2DAutoPoint

        return FATESAM2DAutoPoint.hyperparameter_ranges(trial)


class ModalFATESAM2D:
    NAME = "FATESAM2D"
    SHORT_NAME = "FATE"

    def __init__(self):
        _ensure_modal_app_started()
        self.interface = FATESAM2DInterface()

    @property
    def name(self) -> str:
        return self.NAME

    def predict(self, image, prompts=None):
        return self.interface.predict.remote(image=image, prompts=prompts)

    def setImage(self, image) -> dict[str, object]:
        return self.interface.setImage.remote(image=image)

    def decode_mask(self, prompts=None):
        return self.interface.decode_mask.remote(prompts=prompts)

    def decode_mask_image(self, prompts=None):
        return self.interface.decode_mask_image.remote(prompts=prompts)


class ModalFATESAM2DBlank(ModalFATESAM2D):
    NAME = "FATESAM2DBlank"
    SHORT_NAME = "FATEBlank"

    def __init__(self):
        _ensure_modal_app_started()
        self.interface = FATESAM2DBlankInterface()


class ModalFATESAM2DAutoPoint(ModalFATESAM2D):
    NAME = "FATESAM2D+pts"
    SHORT_NAME = "FATE+pts"

    def __init__(self):
        _ensure_modal_app_started()
        self.interface = FATESAM2DAutoPointInterface()

    @property
    def hyperparameters(self) -> dict:
        return self.interface.hyperparameters.remote()

    @classmethod
    def hyperparameter_ranges(cls, trial) -> dict:
        return {
            "d_bilateral": trial.suggest_int("d_bilateral", 3, 31),
            "sigma_bilateral": trial.suggest_int("sigma_bilateral", 0, 150),
            "C": trial.suggest_int("C", 0, 10),
            "d_gaussian": trial.suggest_categorical("d_gaussian", [i * 2 + 1 for i in range(1, 16)]),
            "n_fgd_points": trial.suggest_int("n_fgd_points", 1, 2000),
            "n_bgd_points": trial.suggest_int("n_bgd_points", 1, 2000),
            "d_gap_erosion": trial.suggest_categorical("d_gap_erosion", [i * 2 + 1 for i in range(1, 11)]),
        }

    def set_hyperparameters(self, **kwargs):
        self.interface.set_hyperparameters.remote(**kwargs)
