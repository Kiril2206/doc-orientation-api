"""Behavioral regression tests for isolated pipelines and v2 training contracts."""
import json
import threading
from collections import Counter
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import routes
from app.core.config import Settings
from app.main import create_app
from app.services.classifier import (
    ModelUnavailableError,
    OrientationClassifier,
    OrientationService,
    PredictionResult,
)
from training.dataset import BalancedSourceSampler, ManifestOrientationDataset, read_manifest
from training.metrics import orientation_metrics
from training.prepare_data import mark_downloaded_pages_verified, save_json


def fake_classifier(logits, verifier=None):
    clf = OrientationClassifier.__new__(OrientationClassifier)
    clf._session = MagicMock()
    clf._session.run.return_value = [np.array([logits], dtype=np.float32)]
    clf._input_name, clf._output_name = "input", "output"
    clf.input_size, clf.resize_mode, clf.model_version = 384, "letterbox", "v2"
    clf._text_verifier = verifier
    clf._verifier_attempted = True
    clf._verifier_lock = threading.Lock()
    clf.hybrid_confidence_threshold, clf.hybrid_margin = .90, .25
    return clf


def test_pure_never_calls_ocr_even_when_ambiguous():
    clf = fake_classifier([.8, -3, .7, -3])
    with patch.object(clf, "_get_verifier", side_effect=AssertionError("OCR called")):
        result = clf.predict(Image.new("RGB", (400, 500)), mode="pure")
    assert result.decision_source == "cv"
    assert result.mode == "pure"


def test_hybrid_override_preserves_actual_cnn_score():
    verifier = MagicMock(is_available=True)
    verifier.detect_orientation.return_value = 180
    clf = fake_classifier([.8, -3, .7, -3], verifier)
    result = clf.predict(Image.new("RGB", (400, 500)), mode="hybrid")
    assert result.predicted_orientation == result.correction_rotation == 180
    assert result.decision_source == "ocr_override"
    assert 0 < result.confidence < result.cnn_confidence < .9
    verifier.detect_orientation.assert_called_once()


@pytest.mark.parametrize("logits", ([10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 0, 10]))
def test_hybrid_skips_ocr_for_clear_or_sideways_predictions(logits):
    clf = fake_classifier(logits)
    with patch.object(clf, "_get_verifier", side_effect=AssertionError("OCR called")):
        assert clf.predict(Image.new("RGB", (50, 70)), "hybrid").decision_source == "cv_no_ambiguity"


@pytest.mark.parametrize("ocr_angle", (None, 90, 270))
def test_hybrid_cannot_change_axis(ocr_angle):
    verifier = MagicMock(is_available=True)
    verifier.detect_orientation.return_value = ocr_angle
    clf = fake_classifier([.8, -3, .7, -3], verifier)
    result = clf.predict(Image.new("RGB", (50, 70)), "hybrid")
    assert result.predicted_orientation == 0
    assert result.decision_source == "cv_ocr_inconclusive"


def test_missing_ocr_is_explicit():
    clf = fake_classifier([.8, -3, .7, -3])
    result = clf.predict(Image.new("RGB", (50, 70)), "hybrid")
    assert result.decision_source == "cv_ocr_unavailable"
    assert result.predicted_orientation == 0


def test_service_does_not_substitute_v1_for_missing_v2(tmp_path):
    settings = Settings(model_path_v1=str(tmp_path/"v1"), model_path_v2=str(tmp_path/"v2"))
    with patch("app.services.classifier.OrientationClassifier", side_effect=[FileNotFoundError(), MagicMock()]):
        service = OrientationService(settings)
    assert service.availability()["pure"]["available"] is False
    with pytest.raises(ModelUnavailableError, match="v2"):
        service.predict(Image.new("RGB", (50, 70)), mode="pure")
    service.predict(Image.new("RGB", (50, 70)), mode="hybrid")
    service.models["hybrid"].predict.assert_called_once()


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for i, split in enumerate(("train", "train", "validation", "test")):
        image = Image.new("RGB", (480, 600), (20 + i * 25, 50, 100))
        image.paste((250, 20, 10), (10, 20, 50 + i, 90))
        image.save(tmp_path/f"{i}.png")
        rows.append({"path": f"{i}.png", "split": split, "source": "local" if i == 0 else "doclaynet",
                     "group_id": f"doc-{i}", "upright_verified": True})
    path = tmp_path/"manifest.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("label", range(4))
def test_v2_training_serving_pixels_match(manifest, label):
    dataset = ManifestOrientationDataset(manifest, "validation", input_size=384)
    clf = fake_classifier([1, 0, 0, 0])
    tensor, target = dataset[label]
    with Image.open(dataset.records[0]["resolved_path"]) as image:
        actual = clf.preprocess(image.rotate(label*90, expand=True))
    assert target == label
    np.testing.assert_allclose(tensor.numpy(), actual[0], atol=1e-6)
    assert tensor.shape == (3, 384, 384)


def test_manifest_blocks_unreviewed_pages(manifest):
    manifest.write_text(manifest.read_text().replace('"upright_verified": true', '"upright_verified": false'))
    with pytest.raises(ValueError, match="Unreviewed"):
        read_manifest(manifest)


def test_manifest_blocks_family_leakage(manifest):
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[-1]["group_id"] = rows[0]["group_id"]
    manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="Group leaks"):
        read_manifest(manifest)


def test_manifest_blocks_duplicate_exports(manifest):
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[-1]["path"] = rows[0]["path"]
    manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(manifest)


def test_source_sampling_keeps_exact_angle_balance(manifest):
    dataset = ManifestOrientationDataset(manifest, "train")
    sampler = BalancedSourceSampler(dataset, {"local": .5, "doclaynet": .5})
    indices = list(sampler)
    assert Counter(i % 4 for i in indices) == {0: 2, 1: 2, 2: 2, 3: 2}
    assert list(sampler) == indices


def test_metrics_report_real_0_180_confusions():
    metrics = orientation_metrics([0, 0, 2, 2, 1, 3], [0, 2, 0, 2, 1, 3])
    assert metrics["confusions_0_180"] == 2
    assert metrics["per_angle_cw"]["0"]["recall"] == .5
    assert metrics["per_angle_cw"]["180"]["precision"] == .5


def test_api_forwards_mode_to_every_pdf_page_and_reports_pipeline():
    clf = MagicMock()
    clf.predict.return_value = PredictionResult(0, 0, .7, "hybrid", "v1", "cv_ocr_confirmed", .7)
    previous = routes._classifier
    @asynccontextmanager
    async def lifespan(app):
        routes.set_classifier(clf)
        yield
    app = create_app()
    app.router.lifespan_context = lifespan
    with fitz.open() as pdf:
        for _ in range(2):
            page = pdf.new_page()
            page.insert_text((20, 50), "Example")
        data = pdf.tobytes()
    try:
        with TestClient(app) as client:
            result = client.post("/correct-orientation", data={"mode": "hybrid"},
                                 files={"file": ("x.pdf", data, "application/pdf")})
            assert result.status_code == 200
            assert result.headers["X-Inference-Mode"] == "hybrid"
            assert result.headers["X-Model-Version"] == "v1"
            assert json.loads(result.headers["X-Decision-Counts"]) == {"cv_ocr_confirmed": 2}
            assert all(call.kwargs["mode"] == "hybrid" for call in clf.predict.call_args_list)
            clf.reset_mock()
            bad = client.post("/correct-orientation", data={"mode": "bad"},
                              files={"file": ("x.pdf", data, "application/pdf")})
            assert bad.status_code == 422
            clf.predict.assert_not_called()
    finally:
        routes.set_classifier(previous)


def test_config_rejects_invalid_mode():
    with pytest.raises(ValueError):
        Settings(inference_mode="random")


def test_prepare_data_retries_transient_windows_file_lock(tmp_path, monkeypatch):
    destination = tmp_path / "state.json"
    destination.write_text("{}", encoding="utf-8")
    original_replace = type(destination).replace
    attempts = 0

    def flaky_replace(path, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient lock")
        return original_replace(path, target)

    monkeypatch.setattr(type(destination), "replace", flaky_replace)
    monkeypatch.setattr("training.prepare_data.time.sleep", lambda _: None)
    save_json(destination, {"complete": True})

    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {"complete": True}


def test_downloaded_corpora_are_trusted_as_upright():
    rows = [
        {"source": "doclaynet", "upright_verified": False},
        {"source": "cord", "upright_verified": False},
        {"source": "local", "upright_verified": False},
    ]

    assert mark_downloaded_pages_verified(rows) == 2
    assert [row["upright_verified"] for row in rows] == [True, True, False]
