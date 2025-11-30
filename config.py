import os
from pathlib import Path
from types import SimpleNamespace

class Config:
    SAM_CHECKPOINT = os.getenv("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "data"
    SAM_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_h")
    SAM_CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / SAM_CHECKPOINT

config = SimpleNamespace(**{k: v for k, v in vars(Config).items() if not k.startswith("_")})