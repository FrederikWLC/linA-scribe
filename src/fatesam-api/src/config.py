import os
from pathlib import Path
from types import SimpleNamespace


class Config:
    ROOT_DIR = Path(__file__).resolve().parent

    SAM2_BACKEND = "sam2"
    SAM2_MODEL_TYPE = "vit_t"
    SAM2_CONFIG = os.getenv("SAM2_CONFIG", "sam2_hiera_t.yaml")
    SAM2_CONFIG_PATH = ROOT_DIR / "configs" / SAM2_CONFIG
    SAM2_CHECKPOINT = os.getenv("SAM2_CHECKPOINT", "sam2_hiera_tiny.pt")
    SAM2_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / SAM2_CHECKPOINT


config = SimpleNamespace(**{k: v for k, v in vars(Config).items() if not k.startswith("_")})
