"""
Image prediction routes.
"""

from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from backend.app.schemas import ImagePredictionResponse
from backend.app.model_service import model_service
from backend.app.config import (
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    DEFAULT_THRESHOLD,
)

router = APIRouter(prefix="/api/predict", tags=["Predictions"])


@router.post("/image", response_model=ImagePredictionResponse)
async def predict_single_image(
    file: UploadFile = File(...),
    threshold: float = Form(default=DEFAULT_THRESHOLD),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format '{ext}'. Allowed: {sorted(list(ALLOWED_IMAGE_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB",
        )

    try:
        result = model_service.predict_image(
            image_bytes=content,
            filename=file.filename or "uploaded_image",
            threshold=threshold,
        )
        return ImagePredictionResponse(**result)
    except RuntimeError as re:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(re),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )
