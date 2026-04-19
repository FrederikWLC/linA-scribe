from fastapi import APIRouter, File, Query, Response, UploadFile

from app.utils.sam_model import scribe_sam_service

router = APIRouter()


@router.post("/scribe/set-image")
async def set_image(file: UploadFile = File(...)) -> dict[str, object]:
    return await scribe_sam_service.set_image_from_upload(file)


@router.post("/scribe/setImage")
async def set_image_alias(file: UploadFile = File(...)) -> dict[str, object]:
    return await set_image(file)


@router.get("/scribe/predict")
def predict(
    x: list[float] = Query(default_factory=list),
    y: list[float] = Query(default_factory=list),
    labels: list[int] = Query(default_factory=list),
    coordinate_space: str = Query("percent", pattern="^(percent|pixel)$"),
) -> Response:
    mask_png = scribe_sam_service.predict_mask_png(
        xs=x,
        ys=y,
        labels=labels,
        coordinate_space=coordinate_space,
    )
    return Response(content=mask_png, media_type="image/png")
