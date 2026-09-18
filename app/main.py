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
from app.services.classifier import OrientationClassifier

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: load model on startup, cleanup on shutdown."""
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # Load model
    try:
        classifier = OrientationClassifier(settings.model_path)
        set_classifier(classifier)
        logger.info("Model loaded successfully from %s", settings.model_path)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("Failed to load model: %s", e)
        logger.warning("Service starting WITHOUT a model. /correct-orientation will fail.")

    yield

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
            "The service uses a fine-tuned ResNet-18 model to classify orientation."
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
        ],
    )

    # Register routes
    app.include_router(router, tags=["Orientation"])

    return app


# Application instance for uvicorn
app = create_app()
