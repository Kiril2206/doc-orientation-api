"""Orientation classifier service using ONNX Runtime.

Loads the exported ONNX model at startup and provides inference
for document orientation classification.
"""

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Class index → rotation in degrees that was applied to the image
ORIENTATION_CLASSES = {0: 0, 1: 90, 2: 180, 3: 270}

# To correct the image, we rotate by the inverse
CORRECTION_MAP = {0: 0, 90: 270, 180: 180, 270: 90}

INPUT_SIZE = (224, 224)


class PredictionResult(NamedTuple):
    """Result from the orientation classifier."""

    predicted_orientation: int  # degrees: 0, 90, 180, 270
    correction_rotation: int   # degrees to rotate to fix
    confidence: float          # softmax probability


class OrientationClassifier:
    """Document orientation classifier backed by ONNX Runtime.

    Usage:
        classifier = OrientationClassifier("model/orientation_model.onnx")
        result = classifier.predict(pil_image)
        print(result.predicted_orientation, result.confidence)
    """

    def __init__(self, model_path: str | Path) -> None:
        """Load the ONNX model.

        Args:
            model_path: Path to the ONNX model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If the model fails to load.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading ONNX model from %s", model_path)

        # Use CPU execution provider (no GPU needed for inference)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1  # t2.micro has 1 vCPU

        try:
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Model loaded successfully (%.1f MB). Input: %s, Output: %s",
            model_size_mb,
            self._input_name,
            self._output_name,
        )

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocess a PIL Image for model input.

        Args:
            image: Input PIL Image (any mode, any size).

        Returns:
            numpy array of shape (1, 3, 224, 224), float32, normalized.
        """
        # Convert to RGB if needed (handles grayscale, RGBA, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to model input size
        image = image.resize(INPUT_SIZE, Image.BILINEAR)

        # Convert to numpy float32, scale to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0

        # Normalize with ImageNet mean/std
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD

        # HWC → CHW and add batch dimension → (1, 3, 224, 224)
        img_array = np.transpose(img_array, (2, 0, 1))
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image: Image.Image) -> PredictionResult:
        """Predict the orientation of a document image.

        Args:
            image: Input PIL Image of a document.

        Returns:
            PredictionResult with predicted orientation, correction rotation,
            and confidence score.
        """
        input_tensor = self.preprocess(image)

        # Run inference
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: input_tensor},
        )

        logits = outputs[0][0]  # shape: (4,)

        # Softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # numerical stability
        probabilities = exp_logits / exp_logits.sum()

        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        predicted_orientation = ORIENTATION_CLASSES[predicted_class]
        correction_rotation = CORRECTION_MAP[predicted_orientation]

        logger.info(
            "Prediction: orientation=%d°, confidence=%.3f, correction=%d°",
            predicted_orientation,
            confidence,
            correction_rotation,
        )

        return PredictionResult(
            predicted_orientation=predicted_orientation,
            correction_rotation=correction_rotation,
            confidence=confidence,
        )
