import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from app.routes.auth import router as auth_router
from app.routes.health import router as health_router
from app.routes.scribe import router as scribe_router

app = FastAPI(title="Scribe", version="1.0")
logger = logging.getLogger(__name__)

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
app.include_router(scribe_router, prefix="/api")

app_dir = Path(__file__).resolve().parent
static_dir_candidates = [
    app_dir.parent / "frontend" / "dist",
    app_dir.parent.parent / "frontend" / "dist",
]
static_dir = next((path for path in static_dir_candidates if path.exists()), static_dir_candidates[0])
assets_dir = static_dir / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")


@app.get("/{full_path:path}", include_in_schema=False)
async def frontend_route(full_path: str) -> FileResponse:
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")

    index_path = static_dir / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend build not found")

    return FileResponse(index_path)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled backend error for %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {exc}"},
    )
