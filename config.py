import os
from pathlib import Path
from types import SimpleNamespace

class Config:
    SAM_BACKEND = os.getenv("SAM_BACKEND", "mobile")
    SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_t")
    SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "mobile_sam.pt")
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "data"
    SAM_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / SAM_CHECKPOINT

config = SimpleNamespace(**{k: v for k, v in vars(Config).items() if not k.startswith("_")})