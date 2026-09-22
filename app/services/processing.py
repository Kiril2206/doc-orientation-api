"""Document processing pipeline with concurrency limits and review heuristics."""

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


class DocumentProcessor:
    """One dedicated worker thread per process: PDFs and OCR never run concurrently.

    Native C++ libraries (PyMuPDF, ONNX Runtime) are protected from concurrent access.
    The admission slot remains occupied until active processing completes.
    """
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="document")
        self.slot = threading.BoundedSemaphore(1)

    async def run(self, function, *args, timeout=45, **kwargs):
        if not self.slot.acquire(blocking=False):
            raise ProcessingBusyError("The server is currently processing another document. Try again shortly.")
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


# Backwards compatibility alias
DemoProcessor = DocumentProcessor


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Document processing deadline exceeded.")


def predict_page(image, classifier, mode="pure", prompt=None, review_threshold=None):
    if review_threshold is not None:
        preview = rgb_image(image)
        preview.thumbnail((256, 256))
        # Near-uniform page detection for blank page review policy.
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
