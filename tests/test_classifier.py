"""Unit tests for the orientation classifier service."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from app.services.classifier import (
    CORRECTION_MAP,
    INPUT_SIZE,
    ORIENTATION_CLASSES,
    OrientationClassifier,
    PredictionResult,
)


class TestPredictionResult:
    """Tests for PredictionResult named tuple."""

    def test_prediction_result_fields(self) -> None:
        """Should have the expected fields."""
        result = PredictionResult(
            predicted_orientation=90,
            correction_rotation=270,
            confidence=0.95,
        )
        assert result.predicted_orientation == 90
        assert result.correction_rotation == 270
        assert result.confidence == 0.95


class TestOrientationMappings:
    """Tests for class/orientation mapping constants."""

    def test_orientation_classes_complete(self) -> None:
        """All 4 orientation classes should be mapped."""
        assert len(ORIENTATION_CLASSES) == 4
        assert set(ORIENTATION_CLASSES.values()) == {0, 90, 180, 270}

    def test_correction_map_inverse(self) -> None:
        """Correction should produce the inverse rotation."""
        # 0° → no correction
        assert CORRECTION_MAP[0] == 0
        # 90° → rotate 270° CW to fix
        assert CORRECTION_MAP[90] == 270
        # 180° → rotate 180° to fix
        assert CORRECTION_MAP[180] == 180
        # 270° → rotate 90° CW to fix
        assert CORRECTION_MAP[270] == 90

    def test_correction_roundtrip(self) -> None:
        """Orientation + correction should sum to 360 (or 0)."""
        for orientation, correction in CORRECTION_MAP.items():
            total = (orientation + correction) % 360
            assert total == 0, f"{orientation}° + {correction}° = {total}°, expected 0°"


class TestClassifierInit:
    """Tests for OrientationClassifier initialization."""

    def test_missing_model_file_raises(self, tmp_path) -> None:
        """Should raise FileNotFoundError for non-existent model."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            OrientationClassifier(tmp_path / "nonexistent.onnx")


class TestClassifierPreprocess:
    """Tests for the preprocessing pipeline.

    These tests use a mock session to avoid needing a real ONNX model.
    """

    @pytest.fixture
    def classifier_with_mock_session(self) -> OrientationClassifier:
        """Create a classifier with a mocked ONNX session."""
        with patch.object(OrientationClassifier, "__init__", lambda self, *a, **kw: None):
            clf = OrientationClassifier.__new__(OrientationClassifier)
            clf._session = MagicMock()
            clf._input_name = "input"
            clf._output_name = "output"
            return clf

    def test_preprocess_output_shape(
        self, classifier_with_mock_session: OrientationClassifier, sample_document_image: Image.Image
    ) -> None:
        """Preprocessed output should have shape (1, 3, 224, 224)."""
        result = classifier_with_mock_session.preprocess(sample_document_image)
        assert result.shape == (1, 3, INPUT_SIZE[0], INPUT_SIZE[1])

    def test_preprocess_dtype(
        self, classifier_with_mock_session: OrientationClassifier, sample_document_image: Image.Image
    ) -> None:
        """Preprocessed output should be float32."""
        result = classifier_with_mock_session.preprocess(sample_document_image)
        assert result.dtype == np.float32

    def test_preprocess_grayscale_to_rgb(
        self, classifier_with_mock_session: OrientationClassifier
    ) -> None:
        """Grayscale images should be converted to 3-channel RGB."""
        gray_img = Image.new("L", (200, 300), color=128)
        result = classifier_with_mock_session.preprocess(gray_img)
        assert result.shape == (1, 3, INPUT_SIZE[0], INPUT_SIZE[1])

    def test_preprocess_rgba_to_rgb(
        self, classifier_with_mock_session: OrientationClassifier
    ) -> None:
        """RGBA images should be converted to RGB."""
        rgba_img = Image.new("RGBA", (200, 300), color=(255, 0, 0, 128))
        result = classifier_with_mock_session.preprocess(rgba_img)
        assert result.shape == (1, 3, INPUT_SIZE[0], INPUT_SIZE[1])

    def test_preprocess_normalization_range(
        self, classifier_with_mock_session: OrientationClassifier, sample_document_image: Image.Image
    ) -> None:
        """Preprocessed values should be roughly in the normalized range."""
        result = classifier_with_mock_session.preprocess(sample_document_image)
        # After ImageNet normalization, values should typically be in [-3, 3]
        assert result.min() > -5.0
        assert result.max() < 5.0

    def test_predict_calls_session(
        self, classifier_with_mock_session: OrientationClassifier, sample_document_image: Image.Image
    ) -> None:
        """predict() should call the ONNX session with preprocessed input."""
        # Mock session to return logits for class 0 (upright)
        mock_logits = np.array([[10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        classifier_with_mock_session._session.run.return_value = [mock_logits]

        result = classifier_with_mock_session.predict(sample_document_image)

        assert result.predicted_orientation == 0
        assert result.correction_rotation == 0
        assert result.confidence > 0.99
        classifier_with_mock_session._session.run.assert_called_once()

    def test_predict_90_degree(
        self, classifier_with_mock_session: OrientationClassifier, sample_document_image: Image.Image
    ) -> None:
        """Should correctly predict 90° orientation."""
        mock_logits = np.array([[0.0, 10.0, 0.0, 0.0]], dtype=np.float32)
        classifier_with_mock_session._session.run.return_value = [mock_logits]

        result = classifier_with_mock_session.predict(sample_document_image)

        assert result.predicted_orientation == 270
        assert result.correction_rotation == 90

    def test_predict_180_degree(
        self, classifier_with_mock_session: OrientationClassifier, sample_document_image: Image.Image
    ) -> None:
        """Should correctly predict 180° orientation."""
        mock_logits = np.array([[0.0, 0.0, 10.0, 0.0]], dtype=np.float32)
        classifier_with_mock_session._session.run.return_value = [mock_logits]

        result = classifier_with_mock_session.predict(sample_document_image)

        assert result.predicted_orientation == 180
        assert result.correction_rotation == 180

    def test_predict_270_degree(
        self, classifier_with_mock_session: OrientationClassifier, sample_document_image: Image.Image
    ) -> None:
        """Should correctly predict 270° orientation."""
        mock_logits = np.array([[0.0, 0.0, 0.0, 10.0]], dtype=np.float32)
        classifier_with_mock_session._session.run.return_value = [mock_logits]

        result = classifier_with_mock_session.predict(sample_document_image)

        assert result.predicted_orientation == 90
        assert result.correction_rotation == 270
