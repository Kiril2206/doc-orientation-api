"""Shared test fixtures and configuration."""

import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.services.classifier import OrientationClassifier, PredictionResult


@pytest.fixture
def sample_document_image() -> Image.Image:
    """Create a synthetic document-like image for testing.

    Returns a 300x400 white image with some black text-like rectangles,
    simulating a document that has a clear 'up' direction.
    """
    img = Image.new("RGB", (300, 400), color=(255, 255, 255))
    pixels = np.array(img)

    # Add dark horizontal bars to simulate text lines (top half only,
    # creating asymmetry so orientation matters)
    for y in range(30, 200, 20):
        pixels[y : y + 8, 30:270] = [30, 30, 30]

    # Add a header bar at the top
    pixels[5:20, 30:150] = [0, 0, 0]

    return Image.fromarray(pixels)


@pytest.fixture
def sample_image_bytes(sample_document_image: Image.Image) -> bytes:
    """Convert sample document image to JPEG bytes."""
    buffer = io.BytesIO()
    sample_document_image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def sample_png_bytes(sample_document_image: Image.Image) -> bytes:
    """Convert sample document image to PNG bytes."""
    buffer = io.BytesIO()
    sample_document_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def grayscale_image_bytes() -> bytes:
    """Create a grayscale image as bytes."""
    img = Image.new("L", (200, 300), color=200)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


@pytest.fixture
def corrupt_file_bytes() -> bytes:
    """Return bytes that cannot be decoded as an image."""
    return b"this is not an image file at all"


@pytest.fixture
def mock_classifier() -> MagicMock:
    """Create a mock classifier that always predicts 0° (upright)."""
    classifier = MagicMock(spec=OrientationClassifier)
    classifier.predict.return_value = PredictionResult(
        predicted_orientation=0,
        correction_rotation=0,
        confidence=0.99,
    )
    return classifier


@pytest.fixture
def mock_classifier_90() -> MagicMock:
    """Create a mock classifier that predicts 90° orientation."""
    classifier = MagicMock(spec=OrientationClassifier)
    classifier.predict.return_value = PredictionResult(
        predicted_orientation=90,
        correction_rotation=270,
        confidence=0.95,
    )
    return classifier
