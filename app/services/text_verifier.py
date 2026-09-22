"""Text orientation verifier using RapidOCR text direction classifier (OSD)."""

import logging
import threading

import numpy as np
from PIL import Image

from app.services.image_processing import rotate_image

logger = logging.getLogger(__name__)


class TextOrientationVerifier:
    """Verifies document orientation based on detected text lines.

    Uses RapidOCR's lightweight DB text detector and MobileNet direction classifier (1.4 MB)
    to check whether readable text lines are horizontal/vertical and right-side up.
    """

    def __init__(self) -> None:
        self._inference_lock = threading.Lock()
        try:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
            self._available = True
            logger.info("TextOrientationVerifier (OSD) initialized successfully.")
        except Exception as e:
            logger.warning("Failed to initialize TextOrientationVerifier: %s", e)
            self._engine = None
            self._available = False

    @property
    def is_available(self) -> bool:
        """Whether the OCR text verifier is initialized and available."""
        return self._available and self._engine is not None

    def detect_orientation(self, image: Image.Image) -> int | None:
        # The detector mutates preprocessing state. Also safe for non-API callers.
        with self._inference_lock:
            return self._detect_orientation(image)

    def _detect_orientation(self, image: Image.Image) -> int | None:
        """Detect the orientation (0, 90, 180, 270) of text in an image.

        Returns:
            0, 90, 180, 270 if text orientation is determined with high confidence.
            None if no text is found or votes are ambiguous.
        """
        if not self.is_available or self._engine is None:
            return None

        try:
            img_np = np.array(image.convert("RGB"))
            dt_boxes, _ = self._engine.text_det(img_np)
            if dt_boxes is None or len(dt_boxes) == 0:
                return None

            # Determine whether text lines are predominantly horizontal or vertical
            horiz_count = 0
            vert_count = 0
            for box in dt_boxes:
                w = np.linalg.norm(box[0] - box[1])
                h = np.linalg.norm(box[1] - box[2])
                if w >= h * 1.2:
                    horiz_count += 1
                elif h >= w * 1.2:
                    vert_count += 1

            is_horizontal = horiz_count >= vert_count

            if is_horizontal:
                crop_list = self._engine.get_crop_img_list(img_np, dt_boxes)
                if not crop_list:
                    return None
                _, cls_res, _ = self._engine.text_cls(crop_list)
                if not cls_res:
                    return None

                votes_0 = sum(1 for r in cls_res if r[0] == "0" and r[1] > 0.7)
                votes_180 = sum(1 for r in cls_res if r[0] == "180" and r[1] > 0.7)
                total = votes_0 + votes_180
                if total == 0:
                    return None

                if votes_0 > votes_180 and (votes_0 / total) >= 0.6:
                    return 0
                elif votes_180 > votes_0 and (votes_180 / total) >= 0.6:
                    return 180
                return None
            else:
                # Text lines are vertical (rotated by 90° or 270°)
                # Rotate 90° clockwise to make lines horizontal, then check direction
                rot_90 = rotate_image(image, 90)
                sub_np = np.array(rot_90.convert("RGB"))
                sub_boxes, _ = self._engine.text_det(sub_np)
                if sub_boxes is None or len(sub_boxes) == 0:
                    return None

                sub_crops = self._engine.get_crop_img_list(sub_np, sub_boxes)
                if not sub_crops:
                    return None
                _, cls_res, _ = self._engine.text_cls(sub_crops)
                if not cls_res:
                    return None

                votes_0 = sum(1 for r in cls_res if r[0] == "0" and r[1] > 0.7)
                votes_180 = sum(1 for r in cls_res if r[0] == "180" and r[1] > 0.7)
                total = votes_0 + votes_180
                if total == 0:
                    return None

                if votes_0 > votes_180 and (votes_0 / total) >= 0.6:
                    return 270
                elif votes_180 > votes_0 and (votes_180 / total) >= 0.6:
                    return 90
                return None

        except Exception as e:
            logger.warning("Error during text orientation detection: %s", e)
            return None
