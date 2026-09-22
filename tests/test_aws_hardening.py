"""Unit tests for AWS production readiness, security headers, limits, and safeguards."""
import base64
import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr

from app.api.routes import set_classifier
from app.core.config import Settings, get_settings
from app.main import create_app
from app.services.classifier import OrientationClassifier, OrientationService, PredictionResult
from app.services.image_processing import image_to_bytes, load_image
from app.services.processing import ProcessingBusyError, predict_page


@pytest.fixture
def mock_classifier() -> MagicMock:
    clf = MagicMock(spec=OrientationClassifier)
    clf.predict.return_value = PredictionResult(
        predicted_orientation=0,
        correction_rotation=0,
        confidence=0.98,
        mode="pure",
        model_version="v1",
        decision_source="v1_confident",
        needs_review=False,
    )
    clf.session = MagicMock()
    clf.session_v2 = None
    return clf


@pytest.fixture
def client(mock_classifier: MagicMock) -> TestClient:
    @asynccontextmanager
    async def mock_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        set_classifier(mock_classifier)
        yield

    app = create_app()
    app.router.lifespan_context = mock_lifespan
    with TestClient(app) as test_client:
        yield test_client


def test_liveness_endpoint(client):
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
    assert "x-request-id" in response.headers
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert "default-src 'self'" in response.headers.get("content-security-policy", "")


def test_readiness_endpoint_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readiness_endpoint_unhealthy_when_no_classifier():
    @asynccontextmanager
    async def empty_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        set_classifier(None)
        yield

    app = create_app()
    app.router.lifespan_context = empty_lifespan
    with TestClient(app) as test_client:
        response = test_client.get("/ready")
        assert response.status_code == 503
        assert "Required model pipeline is unavailable" in response.json()["detail"]


def test_readiness_endpoint_checks_genai_if_required(monkeypatch):
    settings = Settings(
        required_modes=["genai"],
        gemini_api_key=SecretStr(""),
        inference_mode="pure",
    )
    get_settings.cache_clear()
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.api.routes.get_settings", lambda: settings)

    # OrientationService with only pure/hybrid models loaded
    service = OrientationService(settings)

    @asynccontextmanager
    async def service_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        set_classifier(service)
        yield
        service.close()

    app = create_app()
    app.router.lifespan_context = service_lifespan
    try:
        with TestClient(app) as test_client:
            response = test_client.get("/ready")
            assert response.status_code == 503
            assert "Required model pipeline is unavailable" in response.json()["detail"]
    finally:
        get_settings.cache_clear()


def test_multipage_tiff_rejection(client):
    img1 = Image.new("RGB", (60, 60), color="white")
    img2 = Image.new("RGB", (60, 60), color="blue")
    buf = io.BytesIO()
    img1.save(buf, format="TIFF", save_all=True, append_images=[img2])
    buf.seek(0)

    response = client.post(
        "/correct-orientation",
        files={"file": ("multipage.tiff", buf.getvalue(), "image/tiff")},
    )
    assert response.status_code == 400
    assert "Multipage TIFF" in response.json()["detail"]


def test_max_pixels_limit_raises():
    img = Image.new("RGB", (1000, 1000), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    content = buf.getvalue()

    # max_pixels = 500_000, but image has 1_000_000 pixels
    with pytest.raises(ValueError, match="pixel limit"):
        load_image(content, max_pixels=500_000)


def test_cmyk_image_conversion_to_rgb():
    img = Image.new("CMYK", (80, 80), color=(100, 50, 0, 10))
    png_bytes = image_to_bytes(img, output_format="PNG")
    with Image.open(io.BytesIO(png_bytes)) as decoded:
        assert decoded.mode == "RGB"
        assert decoded.size == (80, 80)


def test_blank_page_triggers_needs_review(mock_classifier):
    blank = Image.new("RGB", (300, 400), color=(255, 255, 255))
    res = predict_page(blank, mock_classifier, mode="pure", review_threshold=0.60)
    assert res.needs_review is True
    assert res.decision_source == "blank_unchanged"


def test_blank_page_header_in_endpoint(client):
    blank = Image.new("RGB", (300, 400), color=(255, 255, 255))
    buf = io.BytesIO()
    blank.save(buf, format="PNG")

    response = client.post(
        "/correct-orientation",
        files={"file": ("blank.png", buf.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    assert response.headers.get("X-Needs-Review") == "true"


def test_demo_auth_workflow(monkeypatch, mock_classifier):
    secret_pass = "SuperSecureDemoPassword123!"
    settings = Settings(
        require_auth=True,
        demo_username="demouser",
        demo_password=SecretStr(secret_pass),
    )
    monkeypatch.setattr("app.core.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.main.get_settings", lambda: settings)

    @asynccontextmanager
    async def mock_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        set_classifier(mock_classifier)
        yield

    app = create_app()
    app.router.lifespan_context = mock_lifespan
    try:
        with TestClient(app) as test_client:
            # Liveness & Readiness should bypass auth
            res_live = test_client.get("/live")
            assert res_live.status_code == 200
            res_ready = test_client.get("/ready")
            assert res_ready.status_code == 200

            # UI home requires auth
            res_ui_unauth = test_client.get("/")
            assert res_ui_unauth.status_code == 401
            assert "Basic realm" in res_ui_unauth.headers.get("WWW-Authenticate", "")

            # Valid auth works
            creds = base64.b64encode(f"demouser:{secret_pass}".encode()).decode("ascii")
            res_ui_auth = test_client.get("/", headers={"Authorization": f"Basic {creds}"})
            assert res_ui_auth.status_code == 200
    finally:
        get_settings.cache_clear()


def test_concurrency_busy_returns_429(monkeypatch, client):
    from app.services import processing

    async def mock_busy_run(self, function, *args, **kwargs):
        raise ProcessingBusyError("The server is currently processing another document. Try again shortly.")

    monkeypatch.setattr(processing.DocumentProcessor, "run", mock_busy_run)

    img = Image.new("RGB", (50, 50), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    response = client.post(
        "/correct-orientation",
        files={"file": ("test.png", buf.getvalue(), "image/png")},
    )
    assert response.status_code == 429
    assert "currently processing another document" in response.json()["detail"]


def test_pre_parsing_size_limit_rejection(client):
    # Pass Content-Length header that exceeds 10MB + 64KB
    headers = {"Content-Length": "20000000", "Content-Type": "multipart/form-data"}
    response = client.post("/correct-orientation", headers=headers, content=b"")
    assert response.status_code == 413
    assert "File too large" in response.json()["detail"]
