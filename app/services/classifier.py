"""ONNX classification with explicit, isolated pure and hybrid pipelines."""
import json
import logging
import threading
from pathlib import Path
from typing import NamedTuple

import numpy as np
import onnxruntime as ort

from app.services.preprocessing import ANGLES_CCW, image_tensor, prepare_image

logger = logging.getLogger(__name__)
INPUT_SIZE = (224, 224)  # Legacy model and callers.
ORIENTATION_CLASSES = {0: 0, 1: 270, 2: 180, 3: 90}
CORRECTION_MAP = {0: 0, 90: 270, 180: 180, 270: 90}


class PredictionResult(NamedTuple):
    predicted_orientation: int
    correction_rotation: int
    confidence: float  # CNN probability of the RETURNED angle; never an invented OCR score.
    mode: str = "pure"
    model_version: str = "v1"
    decision_source: str = "cv"
    cnn_confidence: float = 0.0


class ModelUnavailableError(RuntimeError):
    pass


class OrientationClassifier:
    def __init__(self, model_path, text_verifier=None, model_version="v1",
                 threads=1, hybrid_confidence_threshold=.90, hybrid_margin=.25):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        try:
            self._session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
            inp = self._session.get_inputs()[0]
            self._input_name = inp.name
            self._output_name = self._session.get_outputs()[0].name
            metadata = self._session.get_modelmeta().custom_metadata_map
            shape = inp.shape
            if len(shape) != 4 or shape[1] != 3 or not isinstance(shape[2], int) or shape[2] != shape[3]:
                raise ValueError(f"Expected static square RGB input, got {shape}")
            self.input_size = shape[2]
            self.model_version = metadata.get("model_version", model_version)
            self.resize_mode = metadata.get("resize_mode", "stretch")
            self.angles_ccw = json.loads(metadata.get("angles_ccw", "[0,90,180,270]"))
            if self.angles_ccw != ANGLES_CCW:
                raise ValueError("Unsupported class ordering in ONNX metadata")
            if model_version == "v2":
                if self.model_version != "v2" or metadata.get("input_size") != str(self.input_size):
                    raise ValueError("v2 requires matching version/input metadata. Re-export the v2 checkpoint.")
                if self.input_size not in (384, 448) or self.resize_mode != "letterbox":
                    raise ValueError("v2 requires 384/448 letterbox preprocessing")
                if metadata.get("normalization") != "imagenet_rgb":
                    raise ValueError("v2 requires imagenet_rgb normalization metadata")
            if self.resize_mode not in ("stretch", "letterbox"):
                raise ValueError("Unsupported resize mode")
        except Exception as exc:
            raise RuntimeError(f"Failed to load ONNX model: {exc}") from exc
        # Do not import or initialize OCR on startup or during pure inference.
        self._text_verifier = text_verifier
        self._verifier_attempted = text_verifier is not None
        self._verifier_lock = threading.Lock()
        self.hybrid_confidence_threshold = hybrid_confidence_threshold
        self.hybrid_margin = hybrid_margin
        logger.info("Loaded %s: %s, input=%d, resize=%s",
                    self.model_version, model_path, self.input_size, self.resize_mode)

    def preprocess(self, image):
        prepared = prepare_image(image, getattr(self, "input_size", 224),
                                 getattr(self, "resize_mode", "stretch"))
        return image_tensor(prepared)[None]

    def _get_verifier(self):
        with self._verifier_lock:
            if not self._verifier_attempted:
                self._verifier_attempted = True
                try:
                    from app.services.text_verifier import TextOrientationVerifier
                    self._text_verifier = TextOrientationVerifier()
                except Exception:
                    logger.exception("Optional OCR initialization failed")
            return self._text_verifier

    def predict(self, image, mode="pure"):
        if mode not in ("pure", "hybrid"):
            raise ValueError("mode must be pure or hybrid")
        logits = self._session.run([self._output_name],
                                   {self._input_name: self.preprocess(image)})[0][0]
        if logits.shape != (4,) or not np.isfinite(logits).all():
            raise RuntimeError("Invalid model logits")
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        predicted_class = int(np.argmax(probs))
        top_confidence = float(probs[predicted_class])
        decision = "cv"
        if mode == "hybrid":
            top_two = set(np.argsort(probs)[-2:].tolist())
            ambiguous = top_two == {0, 2} and (
                top_confidence < self.hybrid_confidence_threshold
                or abs(float(probs[0] - probs[2])) < self.hybrid_margin)
            decision = "cv_no_ambiguity"
            if ambiguous:
                verifier = self._get_verifier()
                if verifier is None or not verifier.is_available:
                    decision = "cv_ocr_unavailable"
                else:
                    try:
                        angle = verifier.detect_orientation(image)
                        if angle in (0, 180):
                            ocr_class = 0 if angle == 0 else 2
                            decision = "ocr_override" if ocr_class != predicted_class else "cv_ocr_confirmed"
                            predicted_class = ocr_class
                        else:
                            decision = "cv_ocr_inconclusive"
                    except Exception:
                        logger.exception("Optional OCR failed; keeping CNN prediction")
                        decision = "cv_ocr_error"
        orientation = ORIENTATION_CLASSES[predicted_class]
        result = PredictionResult(orientation, CORRECTION_MAP[orientation],
                                  float(probs[predicted_class]), mode,
                                  getattr(self, "model_version", "v1"), decision, top_confidence)
        logger.info("[pipeline=%s model=%s decision=%s] orientation=%d correction=%d cnn_score=%.4f",
                    mode, result.model_version, decision, orientation, result.correction_rotation,
                    result.confidence)
        return result


class OrientationService:
    """Pure serves v2. Hybrid serves the legacy v1 demonstration model."""
    def __init__(self, settings):
        self.models = {}
        self.errors = {}
        for mode, path, version in (
            ("pure", settings.model_path_v2, "v2"),
            ("hybrid", settings.model_path or settings.model_path_v1, "v1"),
        ):
            try:
                self.models[mode] = OrientationClassifier(
                    path, model_version=version, threads=settings.inference_threads,
                    hybrid_confidence_threshold=settings.hybrid_confidence_threshold,
                    hybrid_margin=settings.hybrid_margin)
            except (FileNotFoundError, RuntimeError) as exc:
                self.errors[mode] = str(exc)
                logger.warning("%s unavailable: %s", mode, exc)

    def availability(self):
        return {mode: {"available": mode in self.models,
                       "model_version": "v2" if mode == "pure" else "v1"}
                for mode in ("pure", "hybrid")}

    def ensure_mode(self, mode):
        if mode not in self.models:
            target = "orientation_model_v2.onnx" if mode == "pure" else "orientation_model.onnx"
            raise ModelUnavailableError(
                f"{mode} model not loaded. Train/export {target} and restart the server.")

    def predict(self, image, mode="pure"):
        self.ensure_mode(mode)
        return self.models[mode].predict(image, mode=mode)
