from fastapi import APIRouter

router = APIRouter()


# Return simple health status for container checks.
@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
