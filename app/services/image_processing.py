"""Image processing utilities for rotation and format handling."""

import io
import logging
from pathlib import Path
from typing import Generator

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def pdf_to_images(file_content: bytes, dpi: int = 150) -> Generator[Image.Image, None, None]:
    """Render each page of a PDF as a PIL Image.

    Args:
        file_content: Raw bytes of the PDF file.
        dpi: Rendering resolution. 150 DPI is a good balance of quality vs. speed.

    Yields:
        PIL Image for each page (RGB).

    Raises:
        ValueError: If the bytes cannot be decoded as a PDF.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is required for PDF support. Install it with: pip install pymupdf"
        )

    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Cannot decode PDF: {e}") from e

    try:
        zoom = dpi / 72.0  # PDF base resolution is 72 DPI
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            img_bytes = pixmap.tobytes("ppm")
            yield Image.open(io.BytesIO(img_bytes)).convert("RGB")
    finally:
        doc.close()

# Map rotation degrees → PIL rotation constant
# PIL rotates counter-clockwise, so to rotate an image clockwise by N degrees,
# we use PIL's rotate with -N (or equivalently, 360-N).
# However, PIL.Image.rotate() has expand parameter and rotates CCW.
# For exact 90-degree increments, we use transpose which is lossless.
TRANSPOSE_MAP = {
    0: None,
    90: Image.Transpose.ROTATE_270,    # Rotate 90° CW = Transpose ROTATE_270
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,     # Rotate 270° CW = Transpose ROTATE_90
}

# File extension → PIL format name
FORMAT_MAP = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".tiff": "TIFF",
    ".tif": "TIFF",
    ".bmp": "BMP",
    ".webp": "WEBP",
}


def load_image(file_content: bytes) -> Image.Image:
    """Load an image from bytes.

    Args:
        file_content: Raw bytes of the image file.

    Returns:
        PIL Image object.

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    try:
        image = Image.open(io.BytesIO(file_content))
        image.load()  # Force full decode to catch corrupt images
        return ImageOps.exif_transpose(image)
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}") from e


def rotate_image(image: Image.Image, degrees: int) -> Image.Image:
    """Rotate an image clockwise by the specified degrees.

    Uses lossless transpose for exact 90-degree increments.

    Args:
        image: Input PIL Image.
        degrees: Rotation in degrees (must be 0, 90, 180, or 270).

    Returns:
        Rotated PIL Image.

    Raises:
        ValueError: If degrees is not a valid rotation angle.
    """
    if degrees not in TRANSPOSE_MAP:
        raise ValueError(
            f"Invalid rotation angle: {degrees}. Must be 0, 90, 180, or 270."
        )

    transpose_op = TRANSPOSE_MAP[degrees]

    if transpose_op is None:
        logger.debug("No rotation needed (0°)")
        return image.copy()

    logger.debug("Rotating image by %d° clockwise", degrees)
    return image.transpose(transpose_op)


def image_to_bytes(image: Image.Image, output_format: str = "JPEG", quality: int = 95) -> bytes:
    """Convert a PIL Image to bytes in the specified format.

    Args:
        image: PIL Image to convert.
        output_format: Output format (JPEG, PNG, etc.).
        quality: JPEG/WEBP quality (1-100). Ignored for lossless formats.

    Returns:
        Image encoded as bytes.
    """
    buffer = io.BytesIO()

    # Convert RGBA to RGB for JPEG (JPEG doesn't support alpha)
    if output_format.upper() == "JPEG" and image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")

    save_kwargs: dict = {}
    if output_format.upper() in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality

    image.save(buffer, format=output_format, **save_kwargs)
    buffer.seek(0)
    return buffer.read()


def get_output_format(filename: str) -> str:
    """Determine the PIL format name from a filename.

    Args:
        filename: Original filename with extension.

    Returns:
        PIL format string (e.g., "JPEG", "PNG").
    """
    ext = Path(filename).suffix.lower()
    return FORMAT_MAP.get(ext, "JPEG")


def get_media_type(format_name: str) -> str:
    """Get the MIME media type for a PIL format name.

    Args:
        format_name: PIL format string (e.g., "JPEG", "PNG").

    Returns:
        MIME type string (e.g., "image/jpeg").
    """
    mime_map = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "TIFF": "image/tiff",
        "BMP": "image/bmp",
        "WEBP": "image/webp",
    }
    return mime_map.get(format_name.upper(), "image/jpeg")
