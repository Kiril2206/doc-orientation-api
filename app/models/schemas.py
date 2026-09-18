"""Pydantic response schemas for the API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for the health check endpoint."""

    status: str = Field(default="healthy", examples=["healthy"])
    model_loaded: bool = Field(default=True, examples=[True])
    version: str = Field(default="1.0.0", examples=["1.0.0"])


class OrientationResult(BaseModel):
    """Metadata about the orientation correction result.

    Returned as JSON when the client requests metadata-only,
    or embedded in response headers for image responses.
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
    confidence: float = Field(
        ...,
        description="Model confidence for the predicted orientation (0.0 to 1.0).",
        ge=0.0,
        le=1.0,
        examples=[0.987],
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., examples=["Unsupported file type: .gif"])
