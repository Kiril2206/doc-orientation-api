"""Pydantic response schemas for the API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""

    inference_mode: str = "pure"
    modes: dict = Field(default_factory=dict)
    limits: dict = Field(default_factory=dict)
    status: str = Field(default="healthy", examples=["healthy"])
    model_loaded: bool = Field(default=True, examples=[True])
    version: str = Field(default="2.1.0", examples=["2.1.0"])


class OrientationResult(BaseModel):
    """Metadata about the orientation correction result.

    Describes response header semantics; the correction endpoint returns files.
    """

    original_orientation_degrees: int = Field(
        ...,
        description="Detected orientation of the input image in degrees (0, 90, 180, or 270).",
        examples=[90],
    )
    rotation_applied_degrees: int = Field(
        ...,
        description="Rotation applied to correct the image (0, 270, 180, or 90).",
        examples=[270],
    )
    confidence: float | None = Field(
        ...,
        description="CNN score (0.0 to 1.0); absent for Gemini.",
        ge=0.0,
        le=1.0,
        examples=[0.987],
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., examples=["Unsupported file type: .gif"])
