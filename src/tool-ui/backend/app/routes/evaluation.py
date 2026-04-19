from fastapi import APIRouter, Depends

from app.utils.auth import require_master

router = APIRouter()

# Placeholder funcs to be filled out later
# for retrieval of
# evaluation data, and
# scribed masks

def get_evaluation_files() -> list[dict[str, str]]:
    return [{"name": "Evaluation File 1"}, {"name": "Evaluation File 2"}]

def get_scribing_files() -> list[dict[str, str]]:
    return [{"name": "Scribing File 1"}, {"name": "Scribing File 2"}]

@router.get("/evaluation/evaluation/files")
def evaluation_eval_files(session: dict[str, str] = Depends(require_master)) -> dict[str, object]:
    return {"files": get_evaluation_files()}

@router.get("/evaluation/scribing/files")
def evaluation_scribing_files(session: dict[str, str] = Depends(require_master)) -> dict[str, object]:
    return {"files": get_scribing_files()}

