"""Document Orientation Correction API.

FastAPI application that classifies document image orientation (0°/90°/180°/270°)
and returns the image rotated to upright (0°) orientation.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import close_processor, router, set_classifier
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.middleware import DemoMiddleware
from app.services.classifier import OrientationService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: load model on startup, cleanup on shutdown."""
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # Clear any prior application instance before loading.
    set_classifier(None)
    # Load model
    classifier = None
    try:
        classifier = OrientationService(settings)
        set_classifier(classifier)
        logger.info("Available modes: %s", classifier.availability())
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("Failed to load model: %s", e)
        logger.warning("Service starting WITHOUT a model. /correct-orientation will fail.")

    try:
        yield
    finally:
        set_classifier(None)
        close_processor()
        if classifier is not None:
            classifier.close()
        logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "API for automatic document orientation correction. "
            "Upload a document image and receive the image rotated to upright (0°) orientation. "
            "Choose pure (v2 neural model), hybrid (v1 with text verification), "
            "or genai (remote Gemini API)."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Browser and API share one origin; AWS requires a demo password.
    app.add_middleware(DemoMiddleware, settings=settings)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    # Register routes
    app.include_router(router, tags=["Orientation"])

    return app


# Application instance for uvicorn
app = create_app()
