"""Validated application settings loaded from APP_* environment variables."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"


class Settings(BaseSettings):
    app_name: str = "Document Orientation API"
    app_version: str = "2.1.0"
    debug: bool = False
    inference_mode: Literal["pure", "hybrid", "genai"] = "pure"
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = Field(default="gemini-3.5-flash-lite", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    gemini_timeout_seconds: float = Field(default=15, gt=0, le=30)
    genai_image_max_side: int = Field(default=1600, ge=384, le=4096)
    genai_max_pdf_pages: int = Field(default=3, ge=1, le=10)
    genai_max_pdf_pages: int = Field(default=10, ge=1, le=20)
    model_path_v1: str = str(MODEL_DIR / "orientation_model.onnx")
    model_path_v2: str = str(MODEL_DIR / "orientation_model_v2.onnx")
    model_path: str | None = None  # Legacy APP_MODEL_PATH override for v1 only.
    inference_threads: int = Field(default=2, ge=1)
    hybrid_confidence_threshold: float = Field(default=.90, ge=0, le=1)
    hybrid_margin: float = Field(default=.25, ge=0, le=1)
    max_image_size_mb: int = Field(default=10, ge=1, le=25)
    max_image_pixels: int = Field(default=25_000_000, ge=1, le=50_000_000)
    max_pdf_pages: int = Field(default=20, ge=1, le=20)
    processing_timeout_seconds: float = Field(default=45, gt=0, le=55)
    upload_timeout_seconds: float = Field(default=20, gt=0, le=60)
    review_confidence_threshold: float = Field(default=.60, ge=0, le=1)
    required_modes: list[Literal["pure", "hybrid", "genai"]] = ["pure"]
    require_auth: bool = False
    demo_username: str = "demo"
    demo_password: SecretStr = SecretStr("")
    requests_per_minute: int = Field(default=30, ge=1, le=300)
    allowed_extensions: set[str] = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "INFO"
    model_config = {"env_prefix": "APP_", "env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings():
    return Settings()
