from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.auth import router as auth_router
from app.routes.evaluation import router as evaluation_router
from app.routes.health import router as health_router
from app.routes.protected import router as protected_router
from app.routes.storage import router as storage_router

app = FastAPI(title="FastAPI + Svelte Demo")

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
app.include_router(evaluation_router, prefix="/api")
app.include_router(storage_router, prefix="/api")
