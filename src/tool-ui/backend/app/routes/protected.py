from fastapi import APIRouter, Depends

from app.utils.auth import require_master, require_session

router = APIRouter()


# Return protected page payload for authenticated users.
@router.get("/protected/page")
def protected_page(session: dict[str, str] = Depends(require_session)) -> dict[str, str]:
    username = session["username"]
    return {"username": username, "message": f"Protected page for {username}"}


# Return master-only evaluation page payload.
@router.get("/evaluation/page")
def evaluation_page(session: dict[str, str] = Depends(require_master)) -> dict[str, str]:
    return {
        "username": session["username"],
        "message": f"Evaluation page for master user {session['username']}",
    }
