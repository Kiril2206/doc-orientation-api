"""Integration tests for the FastAPI application."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.api.routes import set_classifier
from app.services.classifier import PredictionResult


@pytest.fixture
def mock_classifier_fixture() -> MagicMock:
    """Create a mock classifier for integration tests."""
    from app.services.classifier import OrientationClassifier

    classifier = MagicMock(spec=OrientationClassifier)
    classifier.predict.return_value = PredictionResult(
        predicted_orientation=90,
        correction_rotation=270,
        confidence=0.9567,
    )
    return classifier


@pytest.fixture
def client(mock_classifier_fixture: MagicMock) -> TestClient:
    """Create a test client with a mocked classifier.

    We patch the lifespan to avoid loading the real ONNX model,
    and inject our mock classifier instead.
    """
    from contextlib import asynccontextmanager
    from collections.abc import AsyncGenerator
    from fastapi import FastAPI

    @asynccontextmanager
    async def mock_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        set_classifier(mock_classifier_fixture)
        yield

    app = create_app()
    app.router.lifespan_context = mock_lifespan

    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_body(self, client: TestClient) -> None:
        """Health response should include status and model_loaded."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "version" in data


class TestCorrectOrientationEndpoint:
    """Tests for POST /correct-orientation."""

    def test_valid_jpeg_upload(self, client: TestClient, sample_image_bytes: bytes) -> None:
        """Should accept a valid JPEG and return corrected image."""
        response = client.post(
            "/correct-orientation",
            files={"file": ("document.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

    def test_response_orientation_headers(self, client: TestClient, sample_image_bytes: bytes) -> None:
        """Response should include orientation metadata in headers."""
        response = client.post(
            "/correct-orientation",
            files={"file": ("document.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.headers["X-Original-Orientation"] == "90"
        assert response.headers["X-Rotation-Applied"] == "270"
        assert "X-Confidence" in response.headers
        assert float(response.headers["X-Confidence"]) == pytest.approx(0.9567, rel=1e-3)

    def test_response_request_id_header(self, client: TestClient, sample_image_bytes: bytes) -> None:
        """Response should include a request ID."""
        response = client.post(
            "/correct-orientation",
            files={"file": ("document.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert "X-Request-Id" in response.headers
        assert len(response.headers["X-Request-Id"]) == 12

    def test_response_is_valid_image(self, client: TestClient, sample_image_bytes: bytes) -> None:
        """Returned bytes should be a valid image."""
        response = client.post(
            "/correct-orientation",
            files={"file": ("document.jpg", sample_image_bytes, "image/jpeg")},
        )
        image = Image.open(io.BytesIO(response.content))
        assert image.mode == "RGB"

    def test_png_upload_returns_png(self, client: TestClient, sample_png_bytes: bytes) -> None:
        """PNG input should return PNG output."""
        response = client.post(
            "/correct-orientation",
            files={"file": ("document.png", sample_png_bytes, "image/png")},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_unsupported_file_extension_returns_400(self, client: TestClient) -> None:
        """Should reject files with unsupported extensions."""
        response = client.post(
            "/correct-orientation",
            files={"file": ("animation.gif", b"fake gif data", "image/gif")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_corrupt_image_returns_400(self, client: TestClient, corrupt_file_bytes: bytes) -> None:
        """Should return 400 for corrupt/non-image data with valid extension."""
        response = client.post(
            "/correct-orientation",
            files={"file": ("document.jpg", corrupt_file_bytes, "image/jpeg")},
        )
        assert response.status_code == 400
        assert "Cannot decode image" in response.json()["detail"]

    def test_empty_file_returns_400(self, client: TestClient) -> None:
        """Should reject empty files."""
        response = client.post(
            "/correct-orientation",
            files={"file": ("document.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400

    def test_oversized_file_returns_413(self, client: TestClient) -> None:
        """Should reject files larger than the configured limit."""
        # Create a file larger than 10 MB
        large_content = b"x" * (11 * 1024 * 1024)
        response = client.post(
            "/correct-orientation",
            files={"file": ("large.jpg", large_content, "image/jpeg")},
        )
        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

    def test_no_file_returns_422(self, client: TestClient) -> None:
        """Should return 422 when no file is provided."""
        response = client.post("/correct-orientation")
        assert response.status_code == 422

    def test_grayscale_image_accepted(self, client: TestClient, grayscale_image_bytes: bytes) -> None:
        """Should accept and process grayscale images."""
        response = client.post(
            "/correct-orientation",
            files={"file": ("gray.jpg", grayscale_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200

    def test_model_not_loaded_returns_503(self, sample_image_bytes: bytes) -> None:
        """Should return 503 when model weights are not loaded."""
        from contextlib import asynccontextmanager
        from collections.abc import AsyncGenerator
        from fastapi import FastAPI
        from app.api import routes

        @asynccontextmanager
        async def unloaded_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            routes._classifier = None
            yield

        original = routes._classifier
        try:
            app = create_app()
            app.router.lifespan_context = unloaded_lifespan
            with TestClient(app) as uninitialized_client:
                routes._classifier = None
                # Health check reports model_loaded=False
                health_resp = uninitialized_client.get("/health")
                assert health_resp.status_code == 200
                assert health_resp.json()["model_loaded"] is False

                # Correct orientation returns 503
                response = uninitialized_client.post(
                    "/correct-orientation",
                    files={"file": ("document.jpg", sample_image_bytes, "image/jpeg")},
                )
                assert response.status_code == 503
                assert "not loaded" in response.json()["detail"]
        finally:
            routes._classifier = original


class TestSwaggerDocs:
    """Tests for auto-generated documentation."""

    def test_swagger_ui_available(self, client: TestClient) -> None:
        """Swagger UI should be accessible at /docs."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json_available(self, client: TestClient) -> None:
        """OpenAPI spec should be available at /openapi.json."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert "/correct-orientation" in spec["paths"]
        assert "/health" in spec["paths"]

    def test_redoc_available(self, client: TestClient) -> None:
        """ReDoc should be accessible at /redoc."""
        response = client.get("/redoc")
        assert response.status_code == 200
