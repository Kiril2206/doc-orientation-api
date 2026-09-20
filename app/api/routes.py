"""API routes for the Document Orientation Correction service."""

import logging
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response, FileResponse

from app.core.config import get_settings
from app.core.logging import generate_request_id, request_id_ctx
from app.models.schemas import ErrorResponse, HealthResponse
from app.services.classifier import OrientationClassifier
from app.services.pdf_processing import correct_pdf
from app.services.image_processing import (
    get_media_type,
    get_output_format,
    image_to_bytes,
    load_image,
    rotate_image,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Will be set during app lifespan startup
_classifier: OrientationClassifier | None = None


def set_classifier(classifier: OrientationClassifier | None) -> None:
    """Set the global classifier instance (called during app startup)."""
    global _classifier
    _classifier = classifier


def get_classifier() -> OrientationClassifier:
    """Get the global classifier instance.

    Raises:
        HTTPException(503): If the classifier has not been initialized (model not loaded/trained).
    """
    if _classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded or weights have not been trained yet. Please train the model and place orientation_model.onnx in the model/ directory.",
        )
    return _classifier


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of the service and whether the model is loaded.",
)
async def health_check() -> HealthResponse:
    """Check service health and model availability."""
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        model_loaded=_classifier is not None,
        version=settings.app_version,
    )


@router.post(
    "/correct-orientation",
    summary="Correct document orientation",
    description=(
        "Upload an image or PDF and receive a corrected image or PDF. Angles are clockwise. "
        "The response includes orientation metadata in the headers:\n"
        "- `X-Original-Orientation`: detected orientation in degrees\n"
        "- `X-Rotation-Applied`: rotation applied to correct the image\n"
        "- `X-Confidence`: model confidence score"
    ),
    responses={
        200: {
            "content": {"image/jpeg": {}, "image/png": {}, "application/pdf": {}},
            "description": "Corrected image or PDF with orientation metadata in headers.",
        },
        400: {"model": ErrorResponse, "description": "Invalid input (bad file type, corrupt image, etc.)"},
        413: {"model": ErrorResponse, "description": "File too large"},
        503: {"model": ErrorResponse, "description": "Model weights not loaded or trained yet"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def correct_orientation(
    file: UploadFile = File(
        ...,
        description="Document image file (JPEG, PNG, TIFF, BMP, WEBP) or PDF. Max 10 MB.",
    ),
) -> Response:
    """Return a corrected image or an original PDF with per-page rotations.

    Angles in response headers are clockwise. For PDFs, angle headers describe
    the first page; confidence is the minimum across all pages.
    """
    # Set request ID for logging
    req_id = generate_request_id()
    request_id_ctx.set(req_id)

    settings = get_settings()
    classifier = get_classifier()

    # --- Validate file extension ---
    filename = file.filename or "upload.jpg"
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Allowed: {sorted(settings.allowed_extensions)}",
        )

    # --- Read file content ---
    try:
        content = await file.read(settings.max_image_size_mb * 1024 * 1024 + 1)
    except Exception as e:
        logger.error("Failed to read uploaded file: %s", e)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.") from e
    finally:
        await file.close()

    # --- Validate file size ---
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content) / (1024*1024):.1f} MB. Maximum: {settings.max_image_size_mb} MB.",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info(
        "Processing file: %s (%.1f KB)",
        filename,
        len(content) / 1024,
    )

    # Classify page previews and update the original PDF page rotations.
    if ext == ".pdf":
        try:
            output, results = correct_pdf(content, classifier, settings.max_pdf_pages)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("PDF processing failed")
            raise HTTPException(status_code=500, detail="PDF processing failed.") from exc
        return Response(
            content=output,
            media_type="application/pdf",
            headers={
                "X-Original-Orientation": str(results[0].predicted_orientation),
                "X-Rotation-Applied": str(results[0].correction_rotation),
                "X-Confidence": f"{min(r.confidence for r in results):.4f}",
                "X-Page-Count": str(len(results)),
                "X-Request-Id": req_id,
                "Content-Disposition": "attachment; filename*=UTF-8''" + quote("corrected_" + filename),
            },
        )

    # ===== Image handling =====
    try:
        image = load_image(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # --- Classify orientation ---
    try:
        result = classifier.predict(image)
    except Exception as e:
        logger.error("Model inference failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Model inference failed. Please try again.",
        ) from e

    # --- Rotate image ---
    corrected = rotate_image(image, result.correction_rotation)

    # --- Encode output ---
    output_format = get_output_format(filename)
    media_type = get_media_type(output_format)
    output_bytes = image_to_bytes(corrected, output_format=output_format)

    logger.info(
        "Response: orientation=%d°, correction=%d°, confidence=%.3f, output_size=%.1f KB",
        result.predicted_orientation,
        result.correction_rotation,
        result.confidence,
        len(output_bytes) / 1024,
    )

    # --- Build response with metadata headers ---
    return Response(
        content=output_bytes,
        media_type=media_type,
        headers={
            "X-Original-Orientation": str(result.predicted_orientation),
            "X-Rotation-Applied": str(result.correction_rotation),
            "X-Confidence": f"{result.confidence:.4f}",
            "X-Request-Id": req_id,
            "Content-Disposition": "attachment; filename*=UTF-8''" + quote("corrected_" + filename),
        },
    )


@router.get("/", include_in_schema=False)
async def homepage():
    return FileResponse(Path(__file__).resolve().parents[1] / "static" / "index.html")
