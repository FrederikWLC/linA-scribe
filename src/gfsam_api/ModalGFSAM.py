import asyncio
import os
import subprocess
import sys
import threading
from pathlib import Path
import modal

if "/root" not in sys.path:
    sys.path.insert(0, "/root")

from scribe.base import BaseScribe
from scribe.binary_mask import BinaryMask
from gfsam_api.GFSAM import GFSAMConfiguration

APP_NAME = os.getenv("GF_SAM_MODAL_APP_NAME", "GFSAM")
START_TIMEOUT_SECONDS = int(os.getenv("GF_SAM_MODAL_START_TIMEOUT_SECONDS", "900"))
GFSAM_REMOTE_ROOT = "/root/gfsam_api"
DATA_REMOTE_ROOT = "/root/data"
MODEL_DIR = Path("/root/models")
DINO_WEIGHTS = MODEL_DIR / "dinov2_vitl14_pretrain.pth"
SAM_WEIGHTS = MODEL_DIR / "sam_vit_h_4b8939.pth"
DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

LOCAL_SRC_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG_PATH = LOCAL_SRC_ROOT / "config.py"
LOCAL_GF_SAM_API_ROOT = LOCAL_SRC_ROOT / "gfsam_api"
LOCAL_SCRIBE_ROOT = LOCAL_SRC_ROOT / "scribe"
LOCAL_DATA_SPLIT_PATH = LOCAL_SRC_ROOT / "data" / "split.py"
LOCAL_RAW_ROOT = LOCAL_SRC_ROOT / "data" / "raw"
LOCAL_GT_ROOT = LOCAL_SRC_ROOT / "data" / "ground_truth" / "registered"

_modal_app_started = False
_modal_app_thread: threading.Thread | None = None
_modal_app_start_event = threading.Event()
_modal_app_start_error: BaseException | None = None


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
            "numpy",
            "tqdm==4.64.1",
            "scipy",
            "pillow",
        ]
    )
    .env({"PYTHONPATH": "/root"})
    .add_local_file(local_path=str(LOCAL_CONFIG_PATH), remote_path="/root/config.py")
    .add_local_dir(local_path=str(LOCAL_GF_SAM_API_ROOT), remote_path=GFSAM_REMOTE_ROOT)
    .add_local_dir(local_path=str(LOCAL_SCRIBE_ROOT), remote_path="/root/scribe")
    .add_local_file(local_path=str(LOCAL_DATA_SPLIT_PATH), remote_path="/root/data/split.py")
    .add_local_dir(local_path=str(LOCAL_GT_ROOT), remote_path="/root/data/ground_truth/registered")
    .add_local_dir(local_path=str(LOCAL_RAW_ROOT), remote_path="/root/data/raw")
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
        raise RuntimeError("Timed out waiting for Modal GFSAM app to start.")
    if _modal_app_start_error is not None:
        if isinstance(_modal_app_start_error, modal.exception.InvalidError) and "already running" in str(_modal_app_start_error):
            _modal_app_started = True
            return
        raise RuntimeError("Failed to start Modal GFSAM app") from _modal_app_start_error

    _modal_app_started = True


def _ensure_weights() -> tuple[str, str]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not DINO_WEIGHTS.exists():
        subprocess.run(["wget", "-O", str(DINO_WEIGHTS), DINO_URL], check=True)
    if not SAM_WEIGHTS.exists():
        subprocess.run(["wget", "-O", str(SAM_WEIGHTS), SAM_URL], check=True)
    return str(DINO_WEIGHTS), str(SAM_WEIGHTS)


@app.cls(image=image, gpu="T4", timeout=30 * 60)
class GFSAMInterface:

    support_data_root: str = modal.parameter()
    dinov2_size: str = modal.parameter()
    sam_size: str = modal.parameter()
    img_size: int = modal.parameter()

    @modal.enter()
    def setup(self) -> None:
        for path in ("/root", GFSAM_REMOTE_ROOT, "/root/scribe"):
            if path not in sys.path:
                sys.path.insert(0, path)

        from gfsam_api.GFSAM import build_from_gfsam_configuration, GFSAMConfiguration

        dinov2_weights, sam_weights = _ensure_weights()

        self.model = build_from_gfsam_configuration(
            GFSAMConfiguration(
                support_data_root=self.support_data_root,
                dinov2_weights=dinov2_weights,
                sam_weights=sam_weights,
                dinov2_size=self.dinov2_size,
                sam_size=self.sam_size,
                img_size=self.img_size,
            )
        )

    def get_model(self):
        if getattr(self, "model", None) is None:
            self.setup()
        return self.model

    @modal.method()
    def setImage(self, image) -> dict[str, object]:
        model = self.get_model().setImage(image)
        return {
            "selected_support_index": model.selected_support_index,
            "selected_support_score": model.selected_support_score,
        }

    @modal.method()
    def decode_mask_image(self):
        return self.get_model().decode_mask().to_image()

    @modal.method()
    def predict(self, image):
        return self.get_model().predict(image).to_image()

    @modal.method()
    def smoke(self) -> dict[str, str]:
        self.get_model()
        return {"status": "ok", "message": "GFSAM predictor loaded on Modal GPU"}


class ModalGFSAM(BaseScribe):

    def __init__(self, configuration: GFSAMConfiguration):
        super().__init__(configuration)
        _ensure_modal_app_started()
        support_data_root = configuration.support_data_root
        img_size = configuration.img_size
        dinov2_size = configuration.dinov2_size
        sam_size = configuration.sam_size
        self.interface = GFSAMInterface(support_data_root=str(support_data_root), img_size=img_size, dinov2_size=dinov2_size, sam_size=sam_size)


    def setImage(self, image):
        self.interface.setImage.remote(image=image)
        return self

    def decode_mask(self):
        return BinaryMask.from_image(self.interface.decode_mask_image.remote())

    def predict(self, image):
        return BinaryMask.from_image(self.interface.predict.remote(image=image))

def build_modal_gfsam() -> ModalGFSAM:
    return ModalGFSAM(GFSAMConfiguration(support_data_root=DATA_REMOTE_ROOT))
