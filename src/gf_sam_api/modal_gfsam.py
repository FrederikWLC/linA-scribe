from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import time
from pathlib import Path

import modal
import numpy as np

from scribe.base import Named
from scribe.binary_mask import BinaryMask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


APP_NAME = os.getenv("GF_SAM_MODAL_APP_NAME", "GFSAM")
MODAL_START_TIMEOUT_SECONDS = int(os.getenv("GF_SAM_MODAL_START_TIMEOUT_SECONDS", "900"))
GF_SAM_REMOTE_ROOT = "/root/gf_sam"
DATA_REMOTE_ROOT = Path("/root/data")
MODEL_DIR = Path("/root/models")
DINO_WEIGHTS = MODEL_DIR / "dinov2_vitl14_pretrain.pth"
SAM_WEIGHTS = MODEL_DIR / "sam_vit_h_4b8939.pth"
DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

LOCAL_SRC_ROOT = Path(__file__).resolve().parents[1]
LOCAL_GF_SAM_ROOT = LOCAL_SRC_ROOT / "gf_sam_api" / "vendor" / "gf_sam"
LOCAL_GF_SAM_API_ROOT = LOCAL_SRC_ROOT / "gf_sam_api"
LOCAL_SCRIBE_ROOT = LOCAL_SRC_ROOT / "scribe"
LOCAL_DATA_SPLIT_PATH = LOCAL_SRC_ROOT / "data" / "split.py"
LOCAL_RAW_ROOT = LOCAL_SRC_ROOT / "data" / "raw"
LOCAL_GT_ROOT = LOCAL_SRC_ROOT / "data" / "ground_truth" / "registered"

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
    .add_local_dir(local_path=str(LOCAL_SCRIBE_ROOT), remote_path="/root/scribe")
    .add_local_file(local_path=str(LOCAL_DATA_SPLIT_PATH), remote_path="/root/data/split.py")
    .add_local_dir(local_path=str(LOCAL_GT_ROOT), remote_path="/root/data/ground_truth/registered")
    .add_local_dir(local_path=str(LOCAL_RAW_ROOT), remote_path="/root/data/raw")
)

app = modal.App(APP_NAME)

_modal_app_started = False
_modal_app_thread: threading.Thread | None = None
_modal_app_start_event = threading.Event()
_modal_app_start_error: Exception | None = None


def _array_to_npy_bytes(array) -> bytes:
    t0 = time.perf_counter()
    arr = np.asarray(array)
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    payload = buffer.getvalue()
    logger.debug(
        "modal_gfsam._array_to_npy_bytes: shape=%s dtype=%s bytes=%d in %.3fs",
        arr.shape,
        arr.dtype,
        len(payload),
        time.perf_counter() - t0,
    )
    return payload


def _npy_bytes_to_array(payload: bytes) -> np.ndarray:
    t0 = time.perf_counter()
    arr = np.load(io.BytesIO(payload), allow_pickle=False)
    logger.debug(
        "modal_gfsam._npy_bytes_to_array: shape=%s dtype=%s bytes=%d in %.3fs",
        arr.shape,
        arr.dtype,
        len(payload),
        time.perf_counter() - t0,
    )
    return arr


async def _run_modal_app_background() -> None:
    global _modal_app_start_error
    try:
        logger.info("modal_gfsam: starting Modal app %s", APP_NAME)
        async with app.run():
            logger.info("modal_gfsam: Modal app %s started", APP_NAME)
            _modal_app_start_event.set()
            await asyncio.Event().wait()
    except BaseException as exc:
        logger.exception("modal_gfsam: Modal app %s failed during startup/run", APP_NAME)
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
        logger.info("modal_gfsam._ensure_modal_app_started: Modal app already started")
        return

    if _modal_app_thread is None or not _modal_app_thread.is_alive():
        logger.info(
            "modal_gfsam._ensure_modal_app_started: starting background app thread with timeout=%ss",
            MODAL_START_TIMEOUT_SECONDS,
        )
        _modal_app_start_event.clear()
        _modal_app_start_error = None
        _modal_app_thread = threading.Thread(target=_modal_app_thread_target, daemon=True)
        _modal_app_thread.start()

    if not _modal_app_start_event.wait(timeout=MODAL_START_TIMEOUT_SECONDS):
        raise RuntimeError(
            "Timed out waiting for Modal GFSAM app to start "
            f"after {MODAL_START_TIMEOUT_SECONDS} seconds."
        )
    if _modal_app_start_error is not None:
        if isinstance(_modal_app_start_error, modal.exception.InvalidError) and "already running" in str(_modal_app_start_error):
            _modal_app_started = True
            logger.info("modal_gfsam._ensure_modal_app_started: Modal app was already running")
            return
        raise RuntimeError("Failed to start Modal GFSAM app") from _modal_app_start_error

    _modal_app_started = True
    logger.info("modal_gfsam._ensure_modal_app_started: Modal app startup complete")


def _ensure_weights() -> tuple[str, str]:
    import subprocess

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not DINO_WEIGHTS.exists():
        logger.info("modal_gfsam._ensure_weights: downloading DINO weights to %s", DINO_WEIGHTS)
        subprocess.run(["wget", "-O", str(DINO_WEIGHTS), DINO_URL], check=True)
    else:
        logger.info("modal_gfsam._ensure_weights: DINO weights already present at %s", DINO_WEIGHTS)
    if not SAM_WEIGHTS.exists():
        logger.info("modal_gfsam._ensure_weights: downloading SAM weights to %s", SAM_WEIGHTS)
        subprocess.run(["wget", "-O", str(SAM_WEIGHTS), SAM_URL], check=True)
    else:
        logger.info("modal_gfsam._ensure_weights: SAM weights already present at %s", SAM_WEIGHTS)
    return str(DINO_WEIGHTS), str(SAM_WEIGHTS)


@app.cls(image=image, gpu="T4", timeout=30 * 60)
class GFSAMInterface:
    @modal.enter()
    def setup(self) -> None:
        t0 = time.perf_counter()
        logger.info("GFSAMInterface.setup: starting remote setup")
        import sys

        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        if str(GF_SAM_REMOTE_ROOT) not in sys.path:
            sys.path.insert(0, str(GF_SAM_REMOTE_ROOT))
        if "/root/scribe" not in sys.path:
            sys.path.insert(0, "/root/scribe")

        from data.split import get_training_data
        from gf_sam_api.gf_sam import GFSAM

        logger.info("GFSAMInterface.setup: loading training data from %s", DATA_REMOTE_ROOT)
        support_images, support_labels, _ = get_training_data(seed=42, data_root=DATA_REMOTE_ROOT)
        logger.info(
            "GFSAMInterface.setup: loaded %d support images and %d labels",
            len(support_images),
            len(support_labels),
        )
        dinov2_weights, sam_weights = _ensure_weights()
        self.model = GFSAM(
            support_images=support_images,
            support_labels=support_labels,
            dinov2_weights=dinov2_weights,
            sam_weights=sam_weights,
        )
        logger.info("GFSAMInterface.setup: remote setup complete in %.3fs", time.perf_counter() - t0)

    def _get_model(self):
        if getattr(self, "model", None) is None:
            logger.info("GFSAMInterface._get_model: model missing, running setup")
            self.setup()
        return self.model

    @modal.method()
    def setImage(self, image):
        t0 = time.perf_counter()
        arr = _npy_bytes_to_array(image)
        logger.info("GFSAMInterface.setImage: received image shape=%s dtype=%s", arr.shape, arr.dtype)
        model = self._get_model().setImage(arr)
        logger.info(
            "GFSAMInterface.setImage: selected support index=%s score=%s in %.3fs",
            model.selected_support_index,
            model.selected_support_score,
            time.perf_counter() - t0,
        )
        return {
            "selected_support_index": model.selected_support_index,
            "selected_support_score": model.selected_support_score,
        }

    @modal.method()
    def decode_mask_image(self):
        t0 = time.perf_counter()
        logger.info("GFSAMInterface.decode_mask_image: decoding mask")
        mask = self._get_model().decode_mask()
        mask_image = mask.to_image()
        logger.info(
            "GFSAMInterface.decode_mask_image: decoded mask image shape=%s in %.3fs",
            mask_image.shape,
            time.perf_counter() - t0,
        )
        return _array_to_npy_bytes(mask_image)

    @modal.method()
    def predict_mask_image(self, image, support_image, support_mask):
        t0 = time.perf_counter()
        image_arr = _npy_bytes_to_array(image)
        support_image_arr = _npy_bytes_to_array(support_image)
        support_mask_arr = _npy_bytes_to_array(support_mask)
        logger.info(
            "GFSAMInterface.predict_mask_image: image=%s support_image=%s support_mask=%s",
            image_arr.shape,
            support_image_arr.shape,
            support_mask_arr.shape,
        )
        mask = self._get_model().predict(
            image=image_arr,
            support_image=support_image_arr,
            support_mask=support_mask_arr,
        )
        mask_image = mask.to_image()
        logger.info(
            "GFSAMInterface.predict_mask_image: prediction complete mask_image=%s in %.3fs",
            mask_image.shape,
            time.perf_counter() - t0,
        )
        return _array_to_npy_bytes(mask_image)
    
    @modal.method()
    def smoke(self) -> dict[str, str]:
        t0 = time.perf_counter()
        logger.info("GFSAMInterface.smoke: starting smoke test")
        from data.split import get_training_data
        from gf_sam_api.gf_sam import GFSAM

        support_images, support_labels, _ = get_training_data(seed=42, data_root=DATA_REMOTE_ROOT)
        logger.info("GFSAMInterface.smoke: loaded %d support images", len(support_images))
        dinov2_weights, sam_weights = _ensure_weights()
        self.model = GFSAM(
            support_images=support_images,
            support_labels=support_labels,
            dinov2_weights=dinov2_weights,
            sam_weights=sam_weights,
        )
        logger.info("GFSAMInterface.smoke: smoke test complete in %.3fs", time.perf_counter() - t0)
        return {"status": "ok", "message": "GFSAM predictor loaded on Modal GPU"}



class ModalGFSAM(Named):
    def __init__(self):
        logger.info("ModalGFSAM.__init__: ensuring Modal app is started")
        _ensure_modal_app_started()
        self.interface = GFSAMInterface()
        logger.info("ModalGFSAM.__init__: interface ready")

    def setImage(self, image):
        logger.info("ModalGFSAM.setImage: sending image shape=%s", np.asarray(image).shape)
        self.image_metadata = self.interface.setImage.remote(image=_array_to_npy_bytes(image))
        logger.info("ModalGFSAM.setImage: remote metadata=%s", self.image_metadata)
        return self

    def decode_mask(self) -> BinaryMask:
        t0 = time.perf_counter()
        logger.info("ModalGFSAM.decode_mask: requesting remote mask")
        mask_image = _npy_bytes_to_array(self.interface.decode_mask_image.remote())
        logger.info("ModalGFSAM.decode_mask: received mask image shape=%s in %.3fs", mask_image.shape, time.perf_counter() - t0)
        return BinaryMask.from_image(mask_image)

    def predict(self, image, support_image=None, support_mask=None) -> BinaryMask:
        if support_image is None and support_mask is None:
            logger.info("ModalGFSAM.predict: running stateful setImage + decode_mask")
            self.setImage(image)
            return self.decode_mask()
        if support_image is None or support_mask is None:
            raise ValueError("support_image and support_mask must be provided together.")

        t0 = time.perf_counter()
        logger.info(
            "ModalGFSAM.predict: sending explicit support prediction image=%s support_image=%s support_mask=%s",
            np.asarray(image).shape,
            np.asarray(support_image).shape,
            np.asarray(support_mask).shape,
        )
        mask_image = self.interface.predict_mask_image.remote(
            image=_array_to_npy_bytes(image),
            support_image=_array_to_npy_bytes(support_image),
            support_mask=_array_to_npy_bytes(support_mask),
        )
        mask_arr = _npy_bytes_to_array(mask_image)
        logger.info("ModalGFSAM.predict: received explicit mask image shape=%s in %.3fs", mask_arr.shape, time.perf_counter() - t0)
        return BinaryMask.from_image(mask_arr)
