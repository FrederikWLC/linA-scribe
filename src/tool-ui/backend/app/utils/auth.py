import os
import secrets
import hmac
import hashlib
import base64
from threading import Lock

from fastapi import Depends, Header, HTTPException

_USERS: set[str] = set()
_USERS_LOCK = Lock()
_SESSIONS: dict[str, dict[str, str]] = {}
_SESSIONS_LOCK = Lock()


def _session_secret() -> bytes:
    return os.getenv("SESSION_SECRET", os.getenv("UNIVERSAL_PASSWORD", "linascribe")).encode("utf-8")


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _sign_session(username: str, role: str, nonce: str) -> str:
    payload = f"{username}:{role}:{nonce}"
    signature = hmac.new(_session_secret(), payload.encode("utf-8"), hashlib.sha256).digest()
    return f"{_b64url_encode(payload.encode('utf-8'))}.{_b64url_encode(signature)}"


def _verify_signed_session(token: str) -> dict[str, str] | None:
    if "." not in token:
        return None

    payload_part, signature_part = token.split(".", 1)
    try:
        payload_bytes = _b64url_decode(payload_part)
        signature = _b64url_decode(signature_part)
    except Exception:
        return None

    expected_signature = hmac.new(_session_secret(), payload_bytes, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected_signature):
        return None

    try:
        username, role, _nonce = payload_bytes.decode("utf-8").split(":", 2)
    except ValueError:
        return None

    if role not in {"user", "master"}:
        return None

    return {"username": username, "role": role}


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
        session = _verify_signed_session(token)

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

    token = _sign_session(normalized_username, role, secrets.token_urlsafe(16))
    with _SESSIONS_LOCK:
        _SESSIONS[token] = {"username": normalized_username, "role": role}

    return {
        "username": normalized_username,
        "created": created,
        "token": token,
        "role": role,
    }
