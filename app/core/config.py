"""Validated application settings loaded from APP_* environment variables."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


class Settings(BaseSettings):
    app_name: str = "Document Orientation API"
    app_version: str = "2.0.0"
    debug: bool = False
    inference_mode: Literal["pure", "hybrid"] = "pure"
    model_path_v1: str = str(MODEL_DIR / "orientation_model.onnx")
    model_path_v2: str = str(MODEL_DIR / "orientation_model_v2.onnx")
    model_path: str | None = None  # Legacy APP_MODEL_PATH override for v1 only.
    inference_threads: int = Field(default=2, ge=1)
    hybrid_confidence_threshold: float = Field(default=.90, ge=0, le=1)
    hybrid_margin: float = Field(default=.25, ge=0, le=1)
    max_image_size_mb: int = 10
    max_pdf_pages: int = 100
    allowed_extensions: set[str] = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "INFO"
    model_config = {"env_prefix": "APP_", "env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings():
    return Settings()
