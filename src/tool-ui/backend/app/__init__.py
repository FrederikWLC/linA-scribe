import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routes.auth import router as auth_router
from app.routes.health import router as health_router
from app.routes.protected import router as protected_router
from app.routes.scribe import router as scribe_router
from app.routes.storage import router as storage_router

app = FastAPI(title="FastAPI + Svelte Demo")
logger = logging.getLogger(__name__)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled backend error for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )

# Keep CORS open for local development and containerized previews.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(protected_router, prefix="/api")
app.include_router(storage_router, prefix="/api")
app.include_router(scribe_router, prefix="/api")
