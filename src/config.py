import os
from pathlib import Path
from types import SimpleNamespace

class Config:
    
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "data"
    
    MOBILE_SAM_BACKEND = "mobile_sam"
    MOBILESAM_VIT_TYPE = "vit_t"
    MOBILESAM_CHECKPOINT = "mobile_sam.pt"
    MOBILESAM_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / MOBILESAM_CHECKPOINT

    # Default to MobileSAM for all SAM-related baselines and comparisons
    # If in the future to test on other SAM backends, it's possible
    SAM_VIT_TYPE = "vit_h"
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    SAM_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / SAM_CHECKPOINT

    SAM2_CONFIG = os.getenv("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_l.yaml")
    SAM2_CHECKPOINT = os.getenv("SAM2_CHECKPOINT", "sam2.1_hiera_large.pt")
    SAM2_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / SAM2_CHECKPOINT

    # Path definitions for project structure
    FATESAM_LOCAL_ROOT = ROOT_DIR / "fatesam2d_api"
    SCRIBE_LOCAL_ROOT = ROOT_DIR / "scribe"
    CONFIG_LOCAL_PATH = ROOT_DIR / "config.py"
    RAW_LOCAL_ROOT = DATA_DIR / "raw"
    GT_LOCAL_ROOT = DATA_DIR / "ground_truth" / "registered"
    GT0_LOCAL_ROOT = DATA_DIR / "ground_truth" / "registered0"
    GT2_LOCAL_ROOT = DATA_DIR / "ground_truth" / "registered2"
    DATA_SPLIT_LOCAL_PATH = DATA_DIR / "split.py"

    FATESAM_CHECKPOINT = os.getenv("FATESAM_CHECKPOINT", "sam2_hiera_large.pt")
    FATESAM_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / FATESAM_CHECKPOINT
    FATESAM_CONFIGS = FATESAM_LOCAL_ROOT / "configs"
    FATESAM_CONFIG = FATESAM_CONFIGS / "sam2/sam2_hiera_l.yaml"


config = SimpleNamespace(**{k: v for k, v in vars(Config).items() if not k.startswith("_")})
