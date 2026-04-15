from fastapi import APIRouter

from app.utils.api_models import LoginRequest, LoginResponse
from app.utils.auth import login_user

router = APIRouter()


# Authenticate by universal or master password and issue a session token.
@router.post("/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    result = login_user(payload.username, payload.password)
    return LoginResponse(
        username=str(result["username"]),
        created=bool(result["created"]),
        token=str(result["token"]),
        role=str(result["role"]),
    )
