from fastapi import APIRouter, Depends, File, Query, Response, UploadFile
from pydantic import BaseModel, Field

from app.utils.sam_model import DEFAULT_MODEL_KEY, scribe_sam_service
from app.utils.auth import require_session

router = APIRouter()


class SetImagePredictRequest(BaseModel):
    x: list[float] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
    coordinate_space: str = Field(default="percent", pattern="^(percent|pixel)$")
    model: str = DEFAULT_MODEL_KEY


@router.get("/scribe/models")
def models(session: dict[str, str] = Depends(require_session)) -> dict[str, object]:
    return {
        "default_model": DEFAULT_MODEL_KEY,
        "models": scribe_sam_service.model_options,
    }


@router.post("/scribe/warmup")
def warmup(session: dict[str, str] = Depends(require_session)) -> dict[str, object]:
    return scribe_sam_service.warmup_user_models(session["username"])


@router.post("/scribe/set-image")
async def set_image(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL_KEY),
    session: dict[str, str] = Depends(require_session),
) -> dict[str, object]:
    return await scribe_sam_service.set_image_from_upload(session["username"], file, model_key=model)


@router.post("/scribe/setImage")
async def set_image_alias(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL_KEY),
    session: dict[str, str] = Depends(require_session),
) -> dict[str, object]:
    return await set_image(file, model, session)


@router.post("/scribe/set_mask")
async def set_mask_alias(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL_KEY),
    session: dict[str, str] = Depends(require_session),
) -> dict[str, object]:
    return await set_image(file, model, session)


@router.get("/scribe/predict")
def predict(
    x: list[float] = Query(default_factory=list),
    y: list[float] = Query(default_factory=list),
    labels: list[int] = Query(default_factory=list),
    coordinate_space: str = Query("percent", pattern="^(percent|pixel)$"),
    model: str = Query(DEFAULT_MODEL_KEY),
    session: dict[str, str] = Depends(require_session),
) -> Response:
    mask_png = scribe_sam_service.predict_mask_png(
        username=session["username"],
        xs=x,
        ys=y,
        labels=labels,
        coordinate_space=coordinate_space,
        model_key=model,
    )
    return Response(content=mask_png, media_type="image/png")


@router.post("/scribe/predict-set-image")
def predict_set_image(
    payload: SetImagePredictRequest,
    session: dict[str, str] = Depends(require_session),
) -> Response:
    mask_png = scribe_sam_service.predict_mask_png(
        username=session["username"],
        xs=payload.x,
        ys=payload.y,
        labels=payload.labels,
        coordinate_space=payload.coordinate_space,
        model_key=payload.model,
    )
    return Response(content=mask_png, media_type="image/png")


@router.post("/scribe/predict")
async def predict_upload(
    file: UploadFile = File(...),
    x: list[float] = Query(default_factory=list),
    y: list[float] = Query(default_factory=list),
    labels: list[int] = Query(default_factory=list),
    coordinate_space: str = Query("percent", pattern="^(percent|pixel)$"),
    model: str = Query(DEFAULT_MODEL_KEY),
    session: dict[str, str] = Depends(require_session),
) -> Response:
    mask_png = await scribe_sam_service.predict_upload_mask_png(
        username=session["username"],
        file=file,
        xs=x,
        ys=y,
        labels=labels,
        coordinate_space=coordinate_space,
        model_key=model,
    )
    return Response(content=mask_png, media_type="image/png")
