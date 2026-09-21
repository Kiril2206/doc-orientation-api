"""Document Orientation Correction API.

FastAPI application that classifies document image orientation (0°/90°/180°/270°)
and returns the image rotated to upright (0°) orientation.
"""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, set_classifier
from app.core.config import get_settings
from app.core.logging import setup_logging
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
    try:
        classifier = OrientationService(settings)
        set_classifier(classifier)
        logger.info("Available modes: %s", classifier.availability())
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("Failed to load model: %s", e)
        logger.warning("Service starting WITHOUT a model. /correct-orientation will fail.")

    yield

    set_classifier(None)
    # Shutdown
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
            "Choose pure (v2 neural model) or hybrid (v1 with optional text verification)."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Original-Orientation",
            "X-Rotation-Applied",
            "X-Confidence",
            "X-Request-Id",
            "X-Page-Count",
            "X-Inference-Mode",
            "X-Model-Version",
            "X-Decision-Source",
            "X-Decision-Counts",
            "X-Confidence-Source",
            "Content-Disposition",
        ],
    )

    # Register routes
    app.include_router(router, tags=["Orientation"])

    return app


# Application instance for uvicorn
app = create_app()
