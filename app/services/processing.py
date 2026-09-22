"""Bounded processing and conservative, uncalibrated demo review rules."""

import asyncio
import contextvars
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app.services.classifier import PredictionResult
from app.services.image_processing import image_to_bytes, load_image, rotate_image
from app.services.preprocessing import rgb_image


class ProcessingBusyError(RuntimeError):
    pass


class DemoProcessor:
    """One dedicated thread per process: PDFs and OCR never run concurrently.

    A response timeout cannot kill native code. Keep the admission slot occupied
    until the underlying work finishes, even if its HTTP client has gone away.
    """
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="document")
        self.slot = threading.BoundedSemaphore(1)

    async def run(self, function, *args, timeout=45, **kwargs):
        if not self.slot.acquire(blocking=False):
            raise ProcessingBusyError("The demo is processing another document. Try again shortly.")
        context = contextvars.copy_context()
        try:
            future = self.executor.submit(context.run, function, *args, **kwargs)
        except Exception:
            self.slot.release()
            raise
        future.add_done_callback(lambda _: self.slot.release())
        wrapped = asyncio.wrap_future(future)
        # Observe late exceptions after HTTP timeout/cancellation.
        wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.wait_for(asyncio.shield(wrapped), timeout)

    def close(self):
        self.executor.shutdown(wait=True)


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Document processing deadline exceeded.")


def predict_page(image, classifier, mode="pure", prompt=None, review_threshold=None):
    if review_threshold is not None:
        preview = rgb_image(image)
        preview.thumbnail((256, 256))
        # Only near-uniform pages: this is not a general text/ambiguity detector.
        if np.ptp(np.asarray(preview.convert("L"))) <= 2:
            return PredictionResult(0, 0, None, mode, "none", "blank_unchanged",
                                    confidence_source="not-provided", needs_review=True)
    result = classifier.predict(image, mode=mode,
                                **({"prompt": prompt} if mode == "genai" else {}))
    if (review_threshold is not None and result.confidence is not None
            and result.confidence < review_threshold
            and result.decision_source not in ("ocr_override", "cv_ocr_confirmed")):
        return result._replace(predicted_orientation=0, correction_rotation=0,
                               decision_source="uncertain_unchanged", needs_review=True)
    return result


def correct_image(content, classifier, output_format, settings, mode, prompt):
    image = load_image(content, max_pixels=settings.max_image_pixels)
    result = predict_page(image, classifier, mode, prompt, settings.review_confidence_threshold)
    output = image_to_bytes(rotate_image(image, result.correction_rotation), output_format)
    return output, [result]
