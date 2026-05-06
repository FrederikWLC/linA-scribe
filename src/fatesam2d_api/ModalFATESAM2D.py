import asyncio
import os
import threading
from pathlib import Path, PurePosixPath

import modal
from scribe.tunable import Tunable

from config import config
from fatesam2d_api.FATESAM2D import FATESAM2DConfiguration, get_all_fatesam2d_configurations, get_all_tunable_fatesam2d_configurations, get_default_fatesam2d_configuration


APP_NAME = os.getenv("MODAL_APP_NAME", "FATESAM2D")
START_TIMEOUT_SECONDS = int(os.getenv("FATESAM_MODAL_START_TIMEOUT_SECONDS", "900"))
REMOTE_ROOT = PurePosixPath("/root")

_modal_app_started = False
_modal_app_thread: threading.Thread | None = None
_modal_app_start_event = threading.Event()
_modal_app_start_error: BaseException | None = None


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libgl1", "libglib2.0-0"])
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
    .env({"PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'fatesam2d_api'}"})
    .add_local_file(local_path=str(config.CONFIG_LOCAL_PATH), remote_path=str(REMOTE_ROOT / "config.py"))
    .add_local_file(local_path=str(config.DATA_SPLIT_LOCAL_PATH), remote_path=str(REMOTE_ROOT / "data/split.py"))
    .add_local_file(
        local_path=str(config.FATESAM_CHECKPOINT_PATH),
        remote_path=str(REMOTE_ROOT / "checkpoints" / config.FATESAM_CHECKPOINT),
    )
    .add_local_dir(local_path=str(config.FATESAM_LOCAL_ROOT), remote_path=str(REMOTE_ROOT / "fatesam2d_api"))
    .add_local_dir(local_path=str(config.SCRIBE_LOCAL_ROOT), remote_path=str(REMOTE_ROOT / "scribe"))
    .add_local_dir(local_path=str(config.GT_LOCAL_ROOT), remote_path=str(REMOTE_ROOT / "data/ground_truth/registered"))
    .add_local_dir(local_path=str(config.RAW_LOCAL_ROOT), remote_path=str(REMOTE_ROOT / "data/raw"))
)

app = modal.App(APP_NAME)


async def _run_modal_app_background() -> None:
    global _modal_app_start_error
    try:
        async with app.run():
            _modal_app_start_event.set()
            await asyncio.Event().wait()
    except BaseException as exc:
        if isinstance(exc, modal.exception.InvalidError) and "already running" in str(exc):
            _modal_app_start_event.set()
            return
        _modal_app_start_error = exc
        _modal_app_start_event.set()


def _modal_app_thread_target() -> None:
    try:
        asyncio.run(_run_modal_app_background())
    except BaseException as exc:
        global _modal_app_start_error
        _modal_app_start_error = exc
        _modal_app_start_event.set()


def _ensure_modal_app_started(timeout_seconds: int = START_TIMEOUT_SECONDS) -> None:
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


def _validate_checkpoint_file(checkpoint_path: str | Path) -> str:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint is missing: {checkpoint_path}")
    if checkpoint_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Checkpoint is empty: {checkpoint_path}")
    return str(checkpoint_path)


def _setup_runtime_env(configuration: FATESAM2DConfiguration) -> None:
    for path in (config.ROOT_DIR, config.FATESAM_LOCAL_ROOT):
        path = str(path)
        if path not in os.sys.path:
            os.sys.path.insert(0, path)
    os.environ.setdefault("SAM2_CHECKPOINT", Path(configuration.checkpoint_path).name)
    os.environ.setdefault("SAM2_CONFIG", str(configuration.config_file))


@app.cls(image=image, gpu="T4", timeout=30 * 60)
class FATESAM2DInterface:

    support_data_root: str = modal.parameter(default=str(REMOTE_ROOT / "data"))
    is_blank: bool = modal.parameter(default=False)
    use_autopoints: bool = modal.parameter(default=False)
    top_n_supports: int = modal.parameter(default=3)

    def _setup_model(self) -> None:
        from fatesam2d_api.FATESAM2D import FATESAM2DConfiguration, build_from_fatesam_configuration

        configuration = FATESAM2DConfiguration(
            support_data_root=self.support_data_root,
            is_blank=self.is_blank,
            use_autopoints=self.use_autopoints,
            top_n_supports=self.top_n_supports
        )
        _setup_runtime_env(configuration)
        _validate_checkpoint_file(configuration.checkpoint_path)
        self.model = build_from_fatesam_configuration(configuration)

    @modal.enter()
    def setup(self) -> None:
        self._setup_model()

    def _get_model(self):
        if getattr(self, "model", None) is None:
            self._setup_model()
        return self.model

    @modal.method()
    def ping(self) -> str:
        configuration = FATESAM2DConfiguration(
            support_data_root=self.support_data_root,
            is_blank=self.is_blank,
            use_autopoints=self.use_autopoints,
            top_n_supports=self.top_n_supports
        )
        return f"ready: {_validate_checkpoint_file(configuration.checkpoint_path)}"

    @modal.method()
    def setImage(self, image) -> dict[str, object]:
        model = self._get_model()
        model.setImage(image)

    @modal.method()
    def decode_mask(self, prompts=None):
        return self._get_model().decode_mask(prompts=prompts)

    @modal.method()
    def decode_mask_image(self, prompts=None):
        return self._get_model().decode_mask(prompts=prompts).to_image()

    @modal.method()
    def predict(self, image, prompts=None):
        return self._get_model().predict(image, prompts=prompts)
    
    @modal.method()
    def hyperparameter_values(self) -> dict:
        return self._get_model().hyperparameter_values
    
    @modal.method()
    def set_hyperparameters(self, **kwargs):
        self._get_model().set_hyperparameters(**kwargs)
        

class ModalFATESAM2D(Tunable):

    def __init__(self,configuration: FATESAM2DConfiguration):
        super().__init__(configuration=configuration)
        _ensure_modal_app_started()
        self.interface = FATESAM2DInterface(
            support_data_root=str(configuration.support_data_root),
            is_blank=configuration.is_blank,
            use_autopoints=configuration.use_autopoints,
            top_n_supports=configuration.top_n_supports
        )

    @property
    def hyperparameter_values(self) -> dict:
        return self.interface.hyperparameter_values.remote()
        
    def set_hyperparameters(self, **kwargs):
        self.interface.set_hyperparameters.remote(**kwargs)
        return self
    
    def setImage(self, image) -> dict[str, object]:
        self.interface.setImage.remote(image=image)
        return self

    def predict(self, image, prompts=None):
        return self.interface.predict.remote(image=image, prompts=prompts)

    def decode_mask(self, prompts=None):
        return self.interface.decode_mask.remote(prompts=prompts)

    def decode_mask_image(self, prompts=None):
        return self.interface.decode_mask_image.remote(prompts=prompts)

   
def build_all_modal_fatesam2d_variants():
    configurations = get_all_fatesam2d_configurations(data_root=str(REMOTE_ROOT / "data"))
    variants = [ModalFATESAM2D(configuration=conf) for conf in configurations]
    return variants

def build_all_tunable_modal_fatesam2d_variants():
    configurations = get_all_tunable_fatesam2d_configurations(data_root=str(REMOTE_ROOT / "data"))
    variants = [ModalFATESAM2D(configuration=conf) for conf in configurations]
    return variants

def build_default_modal_fatesam2d():
    configuration = get_default_fatesam2d_configuration(support_data_root=str(REMOTE_ROOT / "data"))
    variant = ModalFATESAM2D(configuration=configuration)
    return variant
