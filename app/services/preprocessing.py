"""Shared pixel contract for training, export and serving (no torch dependency)."""
import numpy as np
from PIL import Image, ImageOps

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ANGLES_CCW = [0, 90, 180, 270]


def rgb_image(image):
    """Normalize EXIF and composite transparency onto white for classification."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def prepare_image(image, input_size=384, resize_mode="letterbox"):
    image = rgb_image(image)
    size = (input_size, input_size)
    if resize_mode == "stretch":
        return image.resize(size, Image.Resampling.BILINEAR)
    if resize_mode != "letterbox":
        raise ValueError(f"Unknown resize mode: {resize_mode}")
    # Preserve glyph proportions and retain the entire page, including margins.
    image = ImageOps.contain(image, size, Image.Resampling.BILINEAR)
    result = Image.new("RGB", size, "white")
    result.paste(image, ((input_size - image.width) // 2, (input_size - image.height) // 2))
    return result


def image_tensor(image):
    array = (np.asarray(image, dtype=np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(array.transpose(2, 0, 1))
