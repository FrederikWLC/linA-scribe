import base64
import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Query, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field
import cv2
import numpy as np

import logging

from app.utils.sam_model import DEFAULT_MODEL_KEY, scribe_sam_service
from app.utils.auth import require_session
from scribe.binary_mask import BinaryMask

logger = logging.getLogger(__name__)
router = APIRouter()


class BoxPromptRequest(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None


class SetImagePredictRequest(BaseModel):
    x: list[float] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
    x1s: list[float] = Field(default_factory=list)
    y1s: list[float] = Field(default_factory=list)
    x2s: list[float] = Field(default_factory=list)
    y2s: list[float] = Field(default_factory=list)
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


def _predict_response(mask_png: bytes) -> Response:
    image_array = np.frombuffer(mask_png, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=500, detail="Unable to decode prediction mask image.")

    original_shape = image.shape
    original_dtype = image.dtype.name
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    unique_values = np.unique(image)
    zero_count = int(np.count_nonzero(image == 0))
    nonzero_count = int(np.count_nonzero(image != 0))
    print(
        "predict response generated",
        {
            "original_shape": original_shape,
            "original_dtype": original_dtype,
            "shape": image.shape,
            "dtype": image.dtype.name,
            "unique_values": unique_values.tolist(),
            "zero_count": zero_count,
            "nonzero_count": nonzero_count,
        },
        flush=True,
    )
    logger.info(
        "predict response generated",
        extra={
            "original_shape": original_shape,
            "original_dtype": original_dtype,
            "shape": image.shape,
            "dtype": image.dtype.name,
        },
    )

    payload = {
        "mask_png": base64.b64encode(mask_png).decode("ascii"),
    }

    return Response(content=json.dumps(payload), media_type="application/json")


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
        x1s=payload.x1s,
        y1s=payload.y1s,
        x2s=payload.x2s,
        y2s=payload.y2s,
        coordinate_space=payload.coordinate_space,
        model_key=payload.model,
    )
    return _predict_response(mask_png)


@router.post("/scribe/predict")
async def predict_classical(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL_KEY),
    session: dict[str, str] = Depends(require_session),
) -> Response:
    mask_png = await scribe_sam_service.predict_upload_mask_png(
        username=session["username"],
        file=file,
        xs=[],
        ys=[],
        labels=[],
        coordinate_space="percent",
        model_key=model,
    )
    return _predict_response(mask_png)
