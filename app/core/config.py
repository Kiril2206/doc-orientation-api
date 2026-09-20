"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "Document Orientation API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model
    model_path: str = str(Path(__file__).resolve().parent.parent.parent / "model" / "orientation_model.onnx")

    # Image constraints
    max_image_size_mb: int = 10
    max_pdf_pages: int = 100
    allowed_extensions: set[str] = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Logging
    log_level: str = "INFO"

    model_config = {"env_prefix": "APP_", "env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
