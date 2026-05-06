import base64
import json

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field
import cv2
import numpy as np
from scribe.binary_mask import BinaryMask

import logging

from app.utils.service import MODEL_OPTIONS, build_sam_prompts_from_raw, ScribeService
from app.utils.auth import require_session

logger = logging.getLogger(__name__)
router = APIRouter()
scribe_service = ScribeService()


class SetImagePredictRequest(BaseModel):
    x: list[float] = Field(default_factory=list)
    y: list[float] = Field(default_factory=list)
    labels: list[int] = Field(default_factory=list)
    x1s: list[float] = Field(default_factory=list)
    y1s: list[float] = Field(default_factory=list)
    x2s: list[float] = Field(default_factory=list)
    y2s: list[float] = Field(default_factory=list)
    coordinate_space: str = Field(default="percent", pattern="^(percent|pixel)$")


@router.get("/scribe/models")
def models(session: dict[str, str] = Depends(require_session)) -> dict[str, object]:
    return {"models": list(MODEL_OPTIONS.values())}


@router.post("/scribe/warmup-sam-for-user")
def warmup(session: dict[str, str] = Depends(require_session)) -> dict[str, object]:
    scribe_service.get_sam_instance_for_user(session["username"])
    return {"status": "ok"}


@router.post("/scribe/set-image-for-sam")
async def set_image(
    file: UploadFile = File(...),
    session: dict[str, str] = Depends(require_session),
) -> dict[str, object]:
    image = await _read_upload_as_grayscale(file)
    scribe_service.set_image_for_sam(session["username"], image)
    return {
        "status": "ok",
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


@router.post("/scribe/set-image-for-sam")
async def set_image_alias(
    file: UploadFile = File(...),
    session: dict[str, str] = Depends(require_session),
) -> dict[str, object]:
    return await set_image(file, session)


def _get_predict_response_from_output_png(mask_png: bytes) -> Response:
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


@router.post("/scribe/predict-with-sam")
def predict_with_sam(
    payload: SetImagePredictRequest,
    session: dict[str, str] = Depends(require_session),
) -> Response:
    image_hw = scribe_service.get_sam_image_hw(session["username"])
    if image_hw is None:
        raise HTTPException(status_code=400, detail="No image is set. Upload an image before running SAM.")
 
    try:
        prompts = build_sam_prompts_from_raw(
            xs=payload.x,
            ys=payload.y,
            labels=payload.labels,
            x1s=payload.x1s,
            y1s=payload.y1s,
            x2s=payload.x2s,
            y2s=payload.y2s,
            coordinate_space=payload.coordinate_space,
            image_hw=image_hw,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mask = scribe_service.predict_with_sam(session["username"], None, prompts=prompts)
    return _get_predict_response_from_output_png(_mask_to_png(mask))


@router.post("/scribe/predict-with-classical")
async def predict_with_classical(
    file: UploadFile = File(...),
    session: dict[str, str] = Depends(require_session),
) -> Response:
    image = await _read_upload_as_grayscale(file)
    mask = scribe_service.predict_with_classical(image)
    return _get_predict_response_from_output_png(_mask_to_png(mask))


async def _read_upload_as_grayscale(file: UploadFile) -> np.ndarray:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image file is empty")

    try:
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Unable to decode image")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc


def _mask_to_png(mask: np.ndarray | BinaryMask) -> bytes:
    mask = BinaryMask(mask)
    image = mask.to_image()
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise HTTPException(status_code=500, detail="Unable to encode mask image to PNG")
    return encoded.tobytes()
