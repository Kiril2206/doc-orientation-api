"""Exercise the real Gemini adapter and API without sending documents externally."""
import base64
import io
import json
from contextlib import ExitStack, asynccontextmanager
from unittest.mock import patch

import fitz
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import routes
from app.core.config import Settings
from app.main import create_app
from app.services.classifier import OrientationService
from app.services.genai import GenAIError, GeminiOrientationClassifier, parse_rotation


def gemini_response(angle=0, uncertain=False, finish="STOP"):
    return {"candidates": [{"finishReason": finish, "content": {"parts": [
        {"text": json.dumps({"rotation_cw": angle, "uncertain": uncertain})}
    ]}}]}


@pytest.fixture
def api_factory(tmp_path, monkeypatch):
    with ExitStack() as stack:
        def make(handler, **overrides):
            values = {"gemini_api_key": "test-server-secret",
                      "model_path_v1": str(tmp_path / "missing-v1.onnx"),
                      "model_path_v2": str(tmp_path / "missing-v2.onnx"), "model_path": None}
            values.update(overrides)
            settings = Settings(_env_file=None, **values)
            with patch("app.services.genai.GeminiOrientationClassifier",
                       side_effect=lambda s: GeminiOrientationClassifier(s, transport=httpx.MockTransport(handler))):
                service = OrientationService(settings)
            monkeypatch.setattr(routes, "get_settings", lambda: settings)
            previous = routes._classifier

            @asynccontextmanager
            async def lifespan(app):
                routes.set_classifier(service)
                try:
                    yield
                finally:
                    routes.set_classifier(previous)
                    service.close()

            app = create_app()
            app.router.lifespan_context = lifespan
            return stack.enter_context(TestClient(app))
        yield make


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_genai_rotates_original_pixels_clockwise(api_factory, sample_document_image, angle):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=gemini_response(angle))
    client = api_factory(handler)
    content = io.BytesIO()
    sample_document_image.save(content, format="PNG")
    response = client.post("/correct-orientation", data={"mode": "genai", "genai_prompt": "Passport cover"},
                           files={"file": ("page.png", content.getvalue(), "image/png")})
    assert response.status_code == 200
    with Image.open(io.BytesIO(response.content)) as corrected:
        np.testing.assert_array_equal(np.array(corrected), np.array(sample_document_image.rotate(-angle, expand=True)))
    assert response.headers["X-Rotation-Applied"] == str(angle)
    assert response.headers["X-Original-Orientation"] == str((-angle) % 360)
    assert response.headers["X-Inference-Mode"] == "genai"
    assert response.headers["X-Decision-Source"] == "gemini"
    assert response.headers["X-Confidence-Source"] == "not-provided"
    assert "X-Confidence" not in response.headers
    assert len(requests) == 1
    request = requests[0]
    assert request.url.host == "generativelanguage.googleapis.com"
    assert request.url.path == f"/v1beta/models/{routes.get_settings().gemini_model}:generateContent"
    assert request.headers["x-goog-api-key"] == "test-server-secret"
    assert "test-server-secret" not in str(request.url)
    assert b"test-server-secret" not in request.content
    body = json.loads(request.content)
    assert "CLOCKWISE" in body["systemInstruction"]["parts"][0]["text"]
    assert "Passport cover" in body["contents"][0]["parts"][1]["text"]
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    data = body["contents"][0]["parts"][0]["inlineData"]
    assert data["mimeType"] == "image/jpeg"
    with Image.open(io.BytesIO(base64.b64decode(data["data"]))) as preview:
        assert preview.size == sample_document_image.size


@pytest.mark.parametrize("angle", [45, -90, 360, True, "90", 90.0, None])
def test_invalid_angle_is_never_accepted(angle):
    with pytest.raises(GenAIError):
        parse_rotation(gemini_response(angle))


@pytest.mark.parametrize("payload", [
    {}, {"candidates": []}, {"candidates": None},
    gemini_response(90, finish="MAX_TOKENS"),
    gemini_response(0, uncertain="false"),
    {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "90"}]}}]},
])
def test_blocked_truncated_and_malformed_responses_fail(payload):
    with pytest.raises(GenAIError) as exc:
        parse_rotation(payload)
    assert exc.value.status_code == 502


@pytest.mark.parametrize("status,expected", [(401, 503), (403, 503), (429, 503), (404, 503), (500, 502)])
def test_provider_errors_are_sanitized(api_factory, sample_png_bytes, status, expected):
    client = api_factory(lambda _: httpx.Response(status, text="private provider error: test-server-secret"))
    result = client.post("/correct-orientation", data={"mode": "genai"},
                         files={"file": ("page.png", sample_png_bytes)})
    assert result.status_code == expected
    assert "test-server-secret" not in result.text
    assert "private provider" not in result.text


def test_timeout_has_explicit_status(api_factory, sample_png_bytes):
    def handler(request):
        raise httpx.ReadTimeout("private details", request=request)
    client = api_factory(handler)
    response = client.post("/correct-orientation", data={"mode": "genai"},
                           files={"file": ("page.png", sample_png_bytes)})
    assert response.status_code == 504


def test_missing_key_has_no_remote_call(api_factory, sample_png_bytes):
    def forbidden(_):
        pytest.fail("Remote call without a configured key")
    client = api_factory(forbidden, gemini_api_key="")
    assert not client.get("/health").json()["modes"]["genai"]["available"]
    response = client.post("/correct-orientation", data={"mode": "genai"},
                           files={"file": ("page.png", sample_png_bytes)})
    assert response.status_code == 503
    assert "APP_GEMINI_API_KEY" in response.json()["detail"]


def test_genai_independent_of_local_weights_and_default_mode(api_factory, sample_png_bytes):
    client = api_factory(lambda _: httpx.Response(200, json=gemini_response()), inference_mode="genai")
    health = client.get("/health")
    assert health.json()["modes"]["genai"]["available"]
    assert not health.json()["modes"]["pure"]["available"]
    assert "test-server-secret" not in health.text
    response = client.post("/correct-orientation", files={"file": ("page.png", sample_png_bytes)})
    assert response.status_code == 200
    assert response.headers["X-Inference-Mode"] == "genai"
    # Explicit pure must never call the available remote model as a fallback.
    response = client.post("/correct-orientation", data={"mode": "pure"},
                           files={"file": ("page.png", sample_png_bytes)})
    assert response.status_code == 503


def pdf_bytes():
    with fitz.open() as doc:
        for angle in [90, 180]:
            page = doc.new_page(width=300, height=400)
            page.insert_text((30, 40), "Preserve original colored text", color=(1, 0, 0))
            page.set_rotation(angle)
        return doc.tobytes()


def test_pdf_uses_each_page_and_preserves_text(api_factory):
    angles = iter([270, 180])
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=gemini_response(next(angles)))
    client = api_factory(handler)
    response = client.post("/correct-orientation", data={"mode": "genai", "genai_prompt": "Report"},
                           files={"file": ("report.pdf", pdf_bytes())})
    assert response.status_code == 200
    assert response.headers["X-Page-Count"] == "2"
    assert json.loads(response.headers["X-Decision-Counts"]) == {"gemini": 2}
    assert "X-Confidence" not in response.headers
    with fitz.open(stream=response.content, filetype="pdf") as doc:
        assert all(page.rotation == 0 for page in doc)
        assert all("Preserve original colored text" in page.get_text() for page in doc)
        assert doc[0].get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["color"] == 0xff0000
    assert len(requests) == 2
    assert all("Report" in json.loads(r.content)["contents"][0]["parts"][1]["text"] for r in requests)


def test_pdf_page_limit_is_checked_before_any_remote_call(api_factory):
    def forbidden(_):
        pytest.fail("Request sent before page limit validation")
    client = api_factory(forbidden, genai_max_pdf_pages=1)
    response = client.post("/correct-orientation", data={"mode": "genai"},
                           files={"file": ("report.pdf", pdf_bytes())})
    assert response.status_code == 400
    assert "1-page limit" in response.json()["detail"]


def test_uncertain_pdf_page_does_not_return_partial_result(api_factory):
    count = 0
    def handler(_):
        nonlocal count
        count += 1
        return httpx.Response(200, json=gemini_response(0, uncertain=count == 2))
    client = api_factory(handler)
    response = client.post("/correct-orientation", data={"mode": "genai"},
                           files={"file": ("report.pdf", pdf_bytes())})
    assert response.status_code == 422
    assert response.headers["content-type"] == "application/json"


def test_prompt_limit_is_checked_before_remote_call(api_factory, sample_png_bytes):
    def forbidden(_):
        pytest.fail("Oversized prompt sent")
    client = api_factory(forbidden)
    response = client.post("/correct-orientation", data={"mode": "genai", "genai_prompt": "x" * 2001},
                           files={"file": ("page.png", sample_png_bytes)})
    assert response.status_code == 422


def test_preview_respects_exif_and_size_without_changing_original():
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=gemini_response())
    settings = Settings(_env_file=None, gemini_api_key="test", genai_image_max_side=800)
    image = Image.new("RGB", (2400, 1200))
    image.getexif()[274] = 6
    classifier = GeminiOrientationClassifier(settings, transport=httpx.MockTransport(handler))
    try:
        classifier.predict(image)
    finally:
        classifier.close()
    encoded = captured[0]["contents"][0]["parts"][0]["inlineData"]["data"]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as preview:
        assert preview.size == (400, 800)
        assert 274 not in preview.getexif()
    assert image.size == (2400, 1200)
