"""API routes for the Document Orientation Correction service."""

import json
import logging
import time
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response

from app.core.config import get_settings
from app.core.logging import generate_request_id, request_id_ctx
from app.models.schemas import ErrorResponse, HealthResponse
from app.services.classifier import ModelUnavailableError, OrientationClassifier, OrientationService
from app.services.genai import GenAIError
from app.services.image_processing import (
    get_media_type,
    get_output_format,
)
from app.services.pdf_processing import correct_pdf
from app.services.processing import DemoProcessor, ProcessingBusyError, correct_image

logger = logging.getLogger(__name__)

router = APIRouter()

# Will be set during app lifespan startup
_classifier: OrientationClassifier | OrientationService | None = None


def set_classifier(classifier: OrientationClassifier | OrientationService | None) -> None:
    """Set the global classifier instance (called during app startup)."""
    global _classifier
    _classifier = classifier


def get_classifier() -> OrientationClassifier | OrientationService:
    """Get the global classifier instance.

    Raises:
        HTTPException(503): If the classifier has not been initialized (model not loaded/trained).
    """
    if _classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded: inference service is not initialized. Check model paths and restart the server.",
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
        status="healthy" if is_ready() else "degraded",
        model_loaded=bool(_classifier.models) if isinstance(_classifier, OrientationService) else _classifier is not None,
        inference_mode=settings.inference_mode,
        modes=_classifier.availability() if isinstance(_classifier, OrientationService) else {},
        version=settings.app_version,
        limits={"file_mb": settings.max_image_size_mb, "pixels": settings.max_image_pixels,
                "pdf_pages": settings.max_pdf_pages, "genai_pdf_pages": settings.genai_max_pdf_pages,
                "timeout_seconds": settings.processing_timeout_seconds},
    )


@router.post(
    "/correct-orientation",
    summary="Correct document orientation",
    description=(
        "Upload an image or PDF and receive a corrected image or PDF. Angles are clockwise. "
        "The response includes orientation metadata in the headers:\n"
        "- `X-Original-Orientation`: detected orientation in degrees\n"
        "- `X-Rotation-Applied`: rotation applied to correct the image\n"
        "- `X-Confidence`: CNN score; omitted for GenAI\n"
        "Modes: pure, hybrid, genai (Gemini API). genai_prompt adds optional document context."
    ),
    responses={
        200: {
            "content": {"image/jpeg": {}, "image/png": {}, "application/pdf": {}},
            "description": "Corrected image or PDF with orientation metadata in headers.",
        },
        400: {"model": ErrorResponse, "description": "Invalid input (bad file type, corrupt image, etc.)"},
        413: {"model": ErrorResponse, "description": "File too large"},
        503: {"model": ErrorResponse, "description": "Model weights not loaded or trained yet"},
        502: {"model": ErrorResponse, "description": "Invalid response or connection failure from Gemini"},
        504: {"model": ErrorResponse, "description": "Gemini request timed out"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def correct_orientation(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Document image file (JPEG, PNG, TIFF, BMP, WEBP) or PDF. Max 10 MB.",
    ),
    mode: str | None = Form(None),
    genai_prompt: str = Form("", max_length=2000, description="Optional document context for Gemini."),
) -> Response:
    """Return a corrected image or an original PDF with per-page rotations.

    Angles in response headers are clockwise. For PDFs, angle headers describe
    the first page; confidence is the minimum across all pages.
    """
    req_id = request_id_ctx.get() or generate_request_id()

    settings = get_settings()
    mode = mode or settings.inference_mode
    if mode not in ("pure", "hybrid", "genai"):
        raise HTTPException(status_code=422, detail="mode must be pure, hybrid or genai")
    classifier = get_classifier()
    if isinstance(classifier, OrientationService):
        try:
            classifier.ensure_mode(mode)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    logger.info("[pipeline=%s] request accepted", mode)

    # --- Validate file extension ---
    filename = (file.filename or "upload.jpg").replace(chr(92), "/").rsplit("/", 1)[-1][:180]
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

    timeout = settings.processing_timeout_seconds
    try:
        if ext == ".pdf":
            max_pages = min(settings.max_pdf_pages, settings.genai_max_pdf_pages) if mode == "genai" else settings.max_pdf_pages
            output, results = await get_processor().run(
                correct_pdf, content, classifier, max_pages, mode=mode, prompt=genai_prompt,
                review_threshold=settings.review_confidence_threshold,
                deadline=time.monotonic() + timeout, timeout=timeout)
            media_type = "application/pdf"
        else:
            output_format = get_output_format(filename)
            output, results = await get_processor().run(
                correct_image, content, classifier, output_format, settings, mode, genai_prompt,
                timeout=timeout)
            media_type = get_media_type(output_format)
    except ProcessingBusyError as exc:
        raise HTTPException(429, str(exc), headers={"Retry-After": "2"}) from exc
    except TimeoutError as exc:
        raise HTTPException(504, "Processing timed out. Use fewer pages and try again shortly.") from exc
    except GenAIError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Document processing failed")
        raise HTTPException(500, "Document processing failed.") from exc
    logger.info("result mode=%s pages=%d review_pages=%d output_bytes=%d",
                mode, len(results), sum(r.needs_review for r in results), len(output))
    return Response(output, media_type=media_type, headers={
        **pipeline_headers(results, mode),
        "X-Page-Count": str(len(results)),
        "X-Original-Orientation": str(results[0].predicted_orientation),
        "X-Rotation-Applied": str(results[0].correction_rotation),
        "X-Request-Id": req_id,
        "Content-Disposition": "attachment; filename*=UTF-8''" + quote("corrected_" + filename, safe=""),
    })


@router.get("/", include_in_schema=False)
async def homepage():
    return FileResponse(Path(__file__).resolve().parents[1] / "static" / "index.html")


def pipeline_headers(results, mode):
    counts = {}
    for result in results:
        counts[result.decision_source] = counts.get(result.decision_source, 0) + 1
    headers = {
        "X-Inference-Mode": mode,
        "X-Model-Version": next((r.model_version for r in results if r.model_version != "none"), "none"),
        "X-Decision-Source": next(iter(counts)) if len(counts) == 1 else "mixed",
        "X-Decision-Counts": json.dumps(counts, separators=(",", ":")),
        "X-Confidence-Source": results[0].confidence_source,
        "X-Needs-Review": str(any(r.needs_review for r in results)).lower(),
        "X-Page-Results": json.dumps([
            {"page": i + 1, "rotation_cw": r.correction_rotation,
             "score": round(r.confidence, 4) if r.confidence is not None else None,
             "decision": r.decision_source, "needs_review": r.needs_review}
            for i, r in enumerate(results)], separators=(",", ":")),
    }
    if all(result.confidence is not None for result in results):
        headers["X-Confidence"] = f"{min(result.confidence for result in results):.4f}"
    return headers


_processor = None


def get_processor():
    global _processor
    if _processor is None:
        _processor = DemoProcessor()
    return _processor


def close_processor():
    global _processor
    if _processor is not None:
        _processor.close()
        _processor = None


def is_ready():
    if _classifier is None:
        return False
    if not isinstance(_classifier, OrientationService):
        return True
    settings = get_settings()
    return all(mode in _classifier.models
               for mode in {*settings.required_modes, settings.inference_mode})


@router.get("/live", summary="Process liveness")
async def live():
    return {"status": "alive"}


@router.get("/ready", summary="Required models loaded and warmed up")
async def ready():
    if not is_ready():
        raise HTTPException(503, "Required model pipeline is unavailable.")
    return {"status": "ready"}
