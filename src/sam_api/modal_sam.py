from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path, PurePosixPath

import modal


APP_NAME = os.getenv("MOBILE_SAM_MODAL_APP_NAME", "MobileSAM")
START_TIMEOUT_SECONDS = int(os.getenv("MOBILE_SAM_MODAL_START_TIMEOUT_SECONDS", "900"))

REMOTE_SRC_ROOT = "/root/src"
REMOTE_SCRIBE_ROOT = f"{REMOTE_SRC_ROOT}/scribe"
REMOTE_CONFIG_PATH = f"{REMOTE_SRC_ROOT}/config.py"

LOCAL_SRC_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SCRIBE_ROOT = LOCAL_SRC_ROOT / "scribe"

for candidate in (LOCAL_SRC_ROOT, Path(REMOTE_SRC_ROOT)):
    if (candidate / "config.py").exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from config import config  # noqa: E402

REMOTE_CHECKPOINT_PATH = PurePosixPath(REMOTE_SRC_ROOT) / "checkpoints" / config.MOBILESAM_CHECKPOINT

_modal_app_started = False
_modal_app_thread: threading.Thread | None = None
_modal_app_start_event = threading.Event()
_modal_app_start_error: BaseException | None = None


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "libgl1", "libglib2.0-0"])
    .pip_install(
        [
            "numpy==1.26.4",
            "opencv-python-headless==4.11.0.86",
            "optuna==4.8.0",
            "pillow==11.2.1",
            "timm==0.9.16",
            "torch==2.2.2",
            "torchvision==0.17.2",
            "git+https://github.com/ChaoningZhang/MobileSAM.git",
        ]
    )
    .add_local_file(local_path=str(config.CONFIG_LOCAL_PATH), remote_path=REMOTE_CONFIG_PATH)
    .add_local_file(local_path=str(config.MOBILESAM_CHECKPOINT_PATH), remote_path=str(REMOTE_CHECKPOINT_PATH))
    .add_local_dir(local_path=str(LOCAL_SCRIBE_ROOT), remote_path=REMOTE_SCRIBE_ROOT)
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


def _ensure_modal_app_started() -> None:
    global _modal_app_started, _modal_app_thread
    if _modal_app_started:
        return

    if _modal_app_thread is None or not _modal_app_thread.is_alive():
        _modal_app_start_event.clear()
        _modal_app_thread = threading.Thread(target=_modal_app_thread_target, daemon=True)
        _modal_app_thread.start()

    if not _modal_app_start_event.wait(timeout=START_TIMEOUT_SECONDS):
        raise RuntimeError("Timed out waiting for Modal MobileSAM app to start.")
    if _modal_app_start_error is not None:
        if isinstance(_modal_app_start_error, modal.exception.InvalidError) and "already running" in str(_modal_app_start_error):
            _modal_app_started = True
            return
        raise RuntimeError("Failed to start Modal MobileSAM app") from _modal_app_start_error

    _modal_app_started = True


def _setup_runtime_env() -> None:
    import sys

    for path in (REMOTE_SRC_ROOT, REMOTE_SCRIBE_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)

    os.environ.setdefault("SAM_BACKEND", "mobile")
    os.environ.setdefault("SAM_MODEL_TYPE", config.MOBILESAM_MODEL_TYPE)
    os.environ.setdefault("SAM_CHECKPOINT", config.MOBILESAM_CHECKPOINT)


def _serialize_point_prompts(prompts) -> list[dict[str, int]]:
    return [
        {
            "x": int(prompt.x),
            "y": int(prompt.y),
            "label": int(prompt.label),
        }
        for prompt in (prompts or [])
    ]


@app.cls(image=image, gpu="T4", timeout=30 * 60)
class BestMobileSAMv2Interface:
    @modal.enter()
    def setup(self) -> None:
        _setup_runtime_env()

        from scribe.baselines.sam import BestMobileSAMv2Implementation

        self.model = BestMobileSAMv2Implementation()
        self.image = None

    def get_model(self):
        if getattr(self, "model", None) is None:
            self.setup()
        return self.model

    @modal.method()
    def setImage(self, image) -> dict[str, object]:
        import numpy as np

        model = self.get_model()
        self.image = np.asarray(image)
        autoseed_prompts = _serialize_point_prompts(model.autoprompt(self.image))
        return {
            "status": "ok",
            "output_hw": [int(x) for x in self.image.shape[:2]],
            "autoseed_prompts": autoseed_prompts,
        }

    @modal.method()
    def autoseed(self, image=None) -> list[dict[str, int]]:
        import numpy as np

        if image is None:
            if self.image is None:
                raise RuntimeError("No image is set. Call setImage(image) before autoseed().")
            image = self.image

        return _serialize_point_prompts(self.get_model().autoprompt(np.asarray(image)))

    @modal.method()
    def decode_mask_image(self, prompts=None):
        if self.image is None:
            raise RuntimeError("No image is set. Call setImage(image) before decode_mask_image(...).")
        return self._predict_image(self.image, prompts=prompts, autoprompt=False)

    @modal.method()
    def predict(self, image, prompts=None):
        import numpy as np

        return self._predict_image(np.asarray(image), prompts=prompts, autoprompt=not prompts)

    def _predict_image(self, image, prompts=None, autoprompt=False):
        model = self.get_model()
        normalized_prompts = None if not prompts else prompts
        mask = model.predict(
            image,
            prompts=normalized_prompts,
            autoprompt=bool(autoprompt),
        )
        return mask.to_image()

    @modal.method()
    def mask_stats(self, prompts=None) -> dict[str, object]:
        import numpy as np

        if self.image is None:
            raise RuntimeError("No image is set. Call setImage(image) before mask_stats(...).")

        image = self._predict_image(self.image, prompts=prompts, autoprompt=False)
        foreground = image == 0
        return {
            "shape": [int(x) for x in image.shape[:2]],
            "foreground_pixels": int(np.count_nonzero(foreground)),
            "total_pixels": int(foreground.size),
            "foreground_ratio": float(np.mean(foreground)),
            "unique_values": [int(x) for x in np.unique(image)],
        }

    @modal.method()
    def smoke(self) -> dict[str, str]:
        self.get_model()
        return {"status": "ok", "message": "BestMobileSAMv2Implementation loaded on Modal"}


class ModalBestMobileSAMv2Implementation:
    NAME = "BestMobileSAMv2Implementation"
    SHORT_NAME = "mSAM+pts"
    prefers_predict_with_image = True

    def __init__(self):
        _ensure_modal_app_started()
        self.interface = BestMobileSAMv2Interface()

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def short_name(self) -> str:
        return self.SHORT_NAME

    def setImage(self, image):
        return self.interface.setImage.remote(image=image)

    def autoseed(self, image=None):
        return self.interface.autoseed.remote(image=image)

    def decode_mask(self, prompts=None):
        from scribe.binary_mask import BinaryMask

        return BinaryMask.from_image(self.interface.decode_mask_image.remote(prompts=prompts))

    def decode_mask_image(self, prompts=None):
        return self.interface.decode_mask_image.remote(prompts=prompts)

    def mask_stats(self, prompts=None):
        return self.interface.mask_stats.remote(prompts=prompts)

    def predict(self, image, prompts=None):
        from scribe.binary_mask import BinaryMask

        return BinaryMask.from_image(self.interface.predict.remote(image=image, prompts=prompts))
