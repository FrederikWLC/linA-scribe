from __future__ import annotations
from config import config
import sys
from pathlib import Path

WORKSPACE_SRC_ROOT = Path(__file__).resolve().parent
API_ROOT = WORKSPACE_SRC_ROOT / "fatesam2d_api"

for path in (API_ROOT, WORKSPACE_SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fatesam2d_api.ModalFATESAM2D import ModalFATESAM2D, app
from gf_sam_api.modal_gfsam import ModalGFSAM

@app.local_entrypoint()
def main() -> None:
    model = ModalGFSAM()
    print(model.interface.smoke.remote())
