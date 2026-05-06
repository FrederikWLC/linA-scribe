import asyncio
import os
import sys
import threading
from pathlib import Path
from scribe.prompts import PointPromptList
from scribe.baselines.sam import SAMConfiguration, get_all_sam_configurations, get_all_tunable_sam_configurations, get_best_sam_configuration
import modal
from scribe.tunable import Tunable


APP_NAME = os.getenv("MODAL_APP_NAME", "SAM")
GPU = os.getenv("MODAL_GPU", "T4")
START_TIMEOUT_SECONDS = int(os.getenv("SAM_MODAL_START_TIMEOUT_SECONDS", "900"))

REMOTE_SRC_ROOT = "/root/src"
LOCAL_SRC_ROOT = Path(__file__).resolve().parents[1]

if str(LOCAL_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC_ROOT))

CHECKPOINT_LOCAL_DIR = LOCAL_SRC_ROOT / "checkpoints"

_modal_app_started = False
_modal_app_thread: threading.Thread | None = None
_modal_app_start_event = threading.Event()
_modal_app_start_error: BaseException | None = None



def _validate_checkpoint(checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint is missing: {path}")
    if path.stat().st_size <= 0:
        raise FileNotFoundError(f"Checkpoint is empty: {path}")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["git", "libgl1", "libglib2.0-0"])
    .pip_install(
        [
            "numpy==1.26.4",
            "opencv-python-headless==4.11.0.86",
            "optuna==4.8.0",
            "hydra-core==1.3.2",
            "omegaconf==2.3.0",
            "pillow==11.2.1",
            "timm==0.9.16",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "git+https://github.com/facebookresearch/segment-anything.git",
            "git+https://github.com/ChaoningZhang/MobileSAM.git",
        ]
    )
    .run_commands("SAM2_BUILD_CUDA=0 pip install --no-cache-dir git+https://github.com/facebookresearch/sam2.git")
    .env({"PYTHONPATH": f"{REMOTE_SRC_ROOT}"})
    .add_local_dir(str(LOCAL_SRC_ROOT), f"{REMOTE_SRC_ROOT}")
    .add_local_dir(str(CHECKPOINT_LOCAL_DIR), f"{REMOTE_SRC_ROOT}/checkpoints")
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
        raise RuntimeError("Timed out waiting for Modal SAM app to start.")
    if _modal_app_start_error is not None:
        if isinstance(_modal_app_start_error, modal.exception.InvalidError) and "already running" in str(_modal_app_start_error):
            _modal_app_started = True
            return
        raise RuntimeError("Failed to start Modal SAM app") from _modal_app_start_error

    _modal_app_started = True


@app.cls(image=image, gpu=GPU, timeout=START_TIMEOUT_SECONDS)
class SAMInterface:
    sam_type: str = modal.parameter()
    use_bilateral_filter: bool = modal.parameter()
    use_autopoints: bool = modal.parameter()

    @modal.enter()
    def setup(self) -> None:
        for path in (REMOTE_SRC_ROOT, f"{REMOTE_SRC_ROOT}/scribe"):
            if path not in sys.path:
                sys.path.insert(0, path)
        self.setup_model()

    def setup_model(self) -> None:
        from scribe.baselines.sam import SAMConfiguration, build_from_sam_configuration
        sam_configuration = SAMConfiguration(
            sam_type=self.sam_type,
            use_bilateral_filter=self.use_bilateral_filter,
            use_autopoints=self.use_autopoints,
        )
        _validate_checkpoint(sam_configuration.checkpoint_path)
        self._model = build_from_sam_configuration(sam_configuration)

    def model(self):
        return self._model
    
    @modal.method()
    def get_model(self):
        return self._model

    @modal.method()
    def setImage(self, image):
        model = self.model()
        model.set_image(image)

    @modal.method()
    def autoprompt(self, image=None) -> tuple[None,PointPromptList]:
        if image is None:
            raise RuntimeError("No image is given. Call autoprompt with image.")
        return self.model().autoprompt(image)

    @modal.method()
    def decode_mask(self, prompts=None):
        if not self.get_model().has_image():
            raise RuntimeError("No image is set. Call setImage(image) before decode_mask(...).")
        return self.model().decode_mask(prompts)

    @modal.method()
    def predict(self, image, prompts=None):
        mask = self.model().predict(
            image,
            prompts=prompts
        )
        return mask
    
    @modal.method()
    def hyperparameter_values(self) -> dict:
        return self.model().hyperparameter_values

    @modal.method()
    def set_hyperparameters(self, hyperparameters: dict) -> dict:
        model = self.model()
        model.set_hyperparameters(**hyperparameters)

    @modal.method()
    def smoke(self) -> dict[str, str]:
        return {"status": "ok", "message": f"{self.model().name} loaded on Modal"}


class ModalSAM(Tunable):

    def __init__(
        self,
        configuration: SAMConfiguration
    ):
        super().__init__(configuration=configuration)

        _ensure_modal_app_started()
        self.interface = SAMInterface(
            sam_type=str(configuration.sam_type),
            use_bilateral_filter=bool(configuration.use_bilateral_filter),
            use_autopoints=bool(configuration.use_autopoints),
        )

    def setImage(self, image):
        self.interface.setImage.remote(image=image)
        return self

    def autoprompt(self, image=None):
        return self.interface.autoprompt.remote(image=image)

    def decode_mask(self, prompts=None):
        return self.interface.decode_mask.remote(prompts=prompts)

    def predict(self, image, prompts=None):
        return self.interface.predict.remote(image=image, prompts=prompts)
    
    @property
    def hyperparameter_values(self) -> dict:
        return self.interface.hyperparameter_values.remote()

    def set_hyperparameters(self, **kwargs) -> dict:
        self.interface.set_hyperparameters.remote(hyperparameters=kwargs)
        return self

    
def build_all_modal_sam_variants():
    configurations = get_all_sam_configurations()
    variants = [ModalSAM(configuration=conf) for conf in configurations]
    return variants

def build_all_tunable_modal_sam_variants():
    configurations = get_all_tunable_sam_configurations()
    variants = [ModalSAM(configuration=conf) for conf in configurations]
    return variants

def build_best_modal_sam_variant():
    configuration = get_best_sam_configuration()
    variant = ModalSAM(configuration=configuration)
    return variant
