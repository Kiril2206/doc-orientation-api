"""Unit tests for image processing utilities."""

import numpy as np
import pytest
from PIL import Image

from app.services.image_processing import (
    get_media_type,
    get_output_format,
    image_to_bytes,
    load_image,
    rotate_image,
)


class TestLoadImage:
    """Tests for load_image()."""

    def test_load_valid_jpeg(self, sample_image_bytes: bytes) -> None:
        """Should load a valid JPEG image."""
        image = load_image(sample_image_bytes)
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

    def test_load_valid_png(self, sample_png_bytes: bytes) -> None:
        """Should load a valid PNG image."""
        image = load_image(sample_png_bytes)
        assert isinstance(image, Image.Image)

    def test_load_grayscale(self, grayscale_image_bytes: bytes) -> None:
        """Should load a grayscale image."""
        image = load_image(grayscale_image_bytes)
        assert isinstance(image, Image.Image)
        assert image.mode == "L"

    def test_load_corrupt_file_raises(self, corrupt_file_bytes: bytes) -> None:
        """Should raise ValueError for corrupt/non-image data."""
        with pytest.raises(ValueError, match="Cannot decode image"):
            load_image(corrupt_file_bytes)

    def test_load_empty_bytes_raises(self) -> None:
        """Should raise ValueError for empty bytes."""
        with pytest.raises(ValueError):
            load_image(b"")


class TestRotateImage:
    """Tests for rotate_image()."""

    def test_rotate_0_degrees(self, sample_document_image: Image.Image) -> None:
        """Rotating by 0° should return an identical copy."""
        rotated = rotate_image(sample_document_image, 0)
        assert rotated.size == sample_document_image.size
        assert np.array_equal(np.array(rotated), np.array(sample_document_image))

    def test_rotate_90_degrees(self, sample_document_image: Image.Image) -> None:
        """Rotating by 90° should swap width and height."""
        original_w, original_h = sample_document_image.size
        rotated = rotate_image(sample_document_image, 90)
        assert rotated.size == (original_h, original_w)

    def test_rotate_180_degrees(self, sample_document_image: Image.Image) -> None:
        """Rotating by 180° should preserve dimensions."""
        rotated = rotate_image(sample_document_image, 180)
        assert rotated.size == sample_document_image.size

    def test_rotate_270_degrees(self, sample_document_image: Image.Image) -> None:
        """Rotating by 270° should swap width and height."""
        original_w, original_h = sample_document_image.size
        rotated = rotate_image(sample_document_image, 270)
        assert rotated.size == (original_h, original_w)

    def test_rotate_360_roundtrip(self, sample_document_image: Image.Image) -> None:
        """Four 90° rotations should return to the original."""
        img = sample_document_image
        for _ in range(4):
            img = rotate_image(img, 90)
        assert np.array_equal(
            np.array(img), np.array(sample_document_image)
        )

    def test_rotate_invalid_angle_raises(self, sample_document_image: Image.Image) -> None:
        """Should raise ValueError for non-standard angles."""
        with pytest.raises(ValueError, match="Invalid rotation angle"):
            rotate_image(sample_document_image, 45)

    def test_rotate_negative_angle_raises(self, sample_document_image: Image.Image) -> None:
        """Should raise ValueError for negative angles."""
        with pytest.raises(ValueError, match="Invalid rotation angle"):
            rotate_image(sample_document_image, -90)


class TestImageToBytes:
    """Tests for image_to_bytes()."""

    def test_to_jpeg(self, sample_document_image: Image.Image) -> None:
        """Should produce valid JPEG bytes."""
        result = image_to_bytes(sample_document_image, output_format="JPEG")
        assert isinstance(result, bytes)
        assert len(result) > 0
        # Verify it's valid JPEG by loading it back
        reloaded = load_image(result)
        assert reloaded.mode == "RGB"

    def test_to_png(self, sample_document_image: Image.Image) -> None:
        """Should produce valid PNG bytes."""
        result = image_to_bytes(sample_document_image, output_format="PNG")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_rgba_to_jpeg_conversion(self) -> None:
        """RGBA images should be converted to RGB for JPEG output."""
        rgba_img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        result = image_to_bytes(rgba_img, output_format="JPEG")
        reloaded = load_image(result)
        assert reloaded.mode == "RGB"


class TestGetOutputFormat:
    """Tests for get_output_format()."""

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("document.jpg", "JPEG"),
            ("document.jpeg", "JPEG"),
            ("document.JPG", "JPEG"),
            ("document.png", "PNG"),
            ("document.tiff", "TIFF"),
            ("document.tif", "TIFF"),
            ("document.bmp", "BMP"),
            ("document.webp", "WEBP"),
            ("document.unknown", "JPEG"),  # fallback
        ],
    )
    def test_format_detection(self, filename: str, expected: str) -> None:
        """Should map file extensions to PIL format names."""
        assert get_output_format(filename) == expected


class TestGetMediaType:
    """Tests for get_media_type()."""

    @pytest.mark.parametrize(
        "format_name, expected",
        [
            ("JPEG", "image/jpeg"),
            ("PNG", "image/png"),
            ("TIFF", "image/tiff"),
            ("BMP", "image/bmp"),
            ("WEBP", "image/webp"),
            ("UNKNOWN", "image/jpeg"),  # fallback
        ],
    )
    def test_media_type_mapping(self, format_name: str, expected: str) -> None:
        """Should map PIL formats to MIME types."""
        assert get_media_type(format_name) == expected
