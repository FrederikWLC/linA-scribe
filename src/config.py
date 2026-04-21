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

    # Default to MobileSAM for all SAM-related baselines and comparisons
    # If in the future to test on other SAM backends, it's possible
    SAM_BACKEND = os.getenv("SAM_BACKEND", MOBILE_SAM_BACKEND)
    SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", MOBILESAM_MODEL_TYPE)
    SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", MOBILESAM_CHECKPOINT)
    SAM_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / SAM_CHECKPOINT

    SAM2_BACKEND = "sam2"
    SAM2_MODEL_TYPE = "vit_t"
    SAM2_CONFIG = os.getenv("SAM2_CONFIG", "sam2_hiera_t.yaml")
    SAM2_CONFIG_PATH = ROOT_DIR / "configs" / SAM2_CONFIG
    SAM2_CHECKPOINT = os.getenv("SAM2_CHECKPOINT", "sam2_hiera_tiny.pt")
    SAM2_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / SAM2_CHECKPOINT

    # Path definitions for project structure
    FATESAM_LOCAL_ROOT = ROOT_DIR / "fatesam2d_api"
    SCRIBE_LOCAL_ROOT = ROOT_DIR / "scribe"
    CONFIG_LOCAL_PATH = ROOT_DIR / "config.py"
    RAW_LOCAL_ROOT = DATA_DIR / "raw"
    GT_LOCAL_ROOT = DATA_DIR / "ground_truth" / "registered"
    DATA_SPLIT_LOCAL_PATH = DATA_DIR / "split.py"

config = SimpleNamespace(**{k: v for k, v in vars(Config).items() if not k.startswith("_")})
