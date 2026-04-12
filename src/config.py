import os
from pathlib import Path
from types import SimpleNamespace

class Config:
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "data"
    
    MOBILE_SAM_BACKEND = "mobile"
    MOBILESAM_MODEL_TYPE = "vit_t"
    MOBILESAM_CHECKPOINT = "mobile_sam.pt"
    MOBILESAM_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / MOBILESAM_CHECKPOINT

    SAM2_BACKEND = "sam2"
    SAM2_MODEL_TYPE = "vit_t"
    SAM2_CHECKPOINT = "sam2_hiera_tiny.pt"
    SAM2_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / SAM2_CHECKPOINT
    SAM2_CONFIG = "sam2_hiera_tiny.yaml"
    SAM2_CONFIG_PATH = ROOT_DIR / "configs" / SAM2_CONFIG


    # Default to MobileSAM for all SAM-related baselines and comparisons
    # If in the future to test on other SAM backends, it's possible
    SAM_BACKEND = os.getenv("SAM_BACKEND", MOBILE_SAM_BACKEND)
    SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", MOBILESAM_MODEL_TYPE)
    SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", MOBILESAM_CHECKPOINT)
    SAM_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / SAM_CHECKPOINT

config = SimpleNamespace(**{k: v for k, v in vars(Config).items() if not k.startswith("_")})