from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.utils.supabase_storage import get_storage

router = APIRouter()


# Upload an arbitrary file to Supabase storage at provided path.
@router.post("/storage/upload")
async def upload_file(
    file: UploadFile = File(...),
    path: str = Query(..., min_length=1),
) -> dict[str, str]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    try:
        get_storage().upload(
            path=path,
            file=content,
            file_options={
                "content-type": file.content_type or "application/octet-stream",
                "upsert": "true",
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase upload failed: {exc}") from exc

    return {"path": path}


# Create a temporary signed download URL for a Supabase storage path.
@router.get("/storage/download-url")
def download_url(
    path: str = Query(..., min_length=1),
    expires_in: int = Query(3600, ge=60, le=86400),
) -> dict[str, str]:
    try:
        result = get_storage().create_signed_url(path, expires_in)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Supabase signed URL failed: {exc}") from exc

    signed_url = result.get("signedURL") or result.get("signedUrl")
    if not signed_url:
        raise HTTPException(status_code=502, detail="Supabase did not return a signed URL")

    return {"path": path, "download_url": signed_url}
