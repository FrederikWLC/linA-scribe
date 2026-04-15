import os
import secrets
from threading import Lock

from fastapi import Depends, Header, HTTPException

_USERS: set[str] = set()
_USERS_LOCK = Lock()
_SESSIONS: dict[str, dict[str, str]] = {}
_SESSIONS_LOCK = Lock()


# Validate bearer token and return session payload.
def require_session(authorization: str | None = Header(default=None)) -> dict[str, str]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    with _SESSIONS_LOCK:
        session = _SESSIONS.get(token)

    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    return session


# Restrict access to users logged in with master role.
def require_master(session: dict[str, str] = Depends(require_session)) -> dict[str, str]:
    if session.get("role") != "master":
        raise HTTPException(status_code=403, detail="Master access required")
    return session


# Authenticate by universal or master password and issue a session token.
def login_user(username: str, password: str) -> dict[str, str | bool]:
    normalized_username = username.strip().lower()
    if not normalized_username:
        raise HTTPException(status_code=400, detail="Username is required")

    universal_password = os.getenv("UNIVERSAL_PASSWORD", "linascribe")
    master_password = os.getenv("MASTER_PASSWORD", "linascribe-master")

    if password == master_password:
        role = "master"
    elif password == universal_password:
        role = "user"
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    with _USERS_LOCK:
        created = normalized_username not in _USERS
        _USERS.add(normalized_username)

    token = secrets.token_urlsafe(24)
    with _SESSIONS_LOCK:
        _SESSIONS[token] = {"username": normalized_username, "role": role}

    return {
        "username": normalized_username,
        "created": created,
        "token": token,
        "role": role,
    }
