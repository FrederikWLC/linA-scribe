from fastapi import APIRouter, Depends, HTTPException

from app.utils.auth import require_master
from app.utils.evaluation_runner import (
    get_ablation_files,
    get_evaluation_files,
    get_scribing_files,
    get_tuning_files,
)

router = APIRouter()


@router.get("/evaluation/tuning/files")
def evaluation_tuning_files(session: dict[str, str] = Depends(require_master)) -> dict[str, object]:
    return {"files": get_tuning_files(), "by": session["username"]}


@router.get("/evaluation/evaluation/files")
def evaluation_eval_files(session: dict[str, str] = Depends(require_master)) -> dict[str, object]:
    return {"files": get_evaluation_files(), "by": session["username"]}


@router.get("/evaluation/ablation/files")
def evaluation_ablation_files(session: dict[str, str] = Depends(require_master)) -> dict[str, object]:
    return {"files": get_ablation_files(), "by": session["username"]}


@router.get("/evaluation/scribing/files")
def evaluation_scribing_files(session: dict[str, str] = Depends(require_master)) -> dict[str, object]:
    return {"files": get_scribing_files(), "by": session["username"]}
