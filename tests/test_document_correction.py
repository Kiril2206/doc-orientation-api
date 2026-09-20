"""Regression tests for actual rotation direction and PDF preservation."""
import io
from unittest.mock import MagicMock
import fitz
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import create_app
from app.api.routes import set_classifier
from app.services.classifier import OrientationClassifier, PredictionResult
from app.services.image_processing import load_image, rotate_image
from app.services.pdf_processing import correct_pdf


@pytest.mark.parametrize("label", range(4))
def test_training_rotation_is_reversed(label, sample_document_image):
    from training.dataset import OrientationDataset
    ds = OrientationDataset(images=[sample_document_image], is_train=False)
    tensor, target = ds[label]
    clf = OrientationClassifier.__new__(OrientationClassifier)
    clf._input_name, clf._output_name = "input", "output"
    clf._session = MagicMock()
    logits = np.full((1, 4), -10, dtype=np.float32)
    logits[0, target] = 10
    clf._session.run.return_value = [logits]
    rotated = sample_document_image.rotate(label * 90, expand=True)
    # Resize then rotate can differ by 1-2 pixel levels due to Pillow rounding.
    assert np.allclose(tensor.numpy(), clf.preprocess(rotated)[0], atol=2 / 255 / .224)
    result = clf.predict(rotated)
    corrected = rotate_image(rotated, result.correction_rotation)
    assert np.array_equal(np.array(corrected), np.array(sample_document_image))


def make_pdf():
    with fitz.open() as doc:
        for angle in (0, 90, 180, 270):
            page = doc.new_page(width=300, height=400)
            page.insert_text((30, 40), "Colored document", color=(1, 0, 0))
            page.set_rotation(angle)
        return doc.tobytes()


def test_pdf_preserves_content_and_composes_rotation():
    clf = MagicMock()
    clf.predict.side_effect = [PredictionResult(a, (-a) % 360, .99) for a in (0, 90, 180, 270)]
    corrected, results = correct_pdf(make_pdf(), clf)
    with fitz.open(stream=corrected, filetype="pdf") as doc:
        assert doc.page_count == len(results) == 4
        assert all(p.rotation == 0 for p in doc)
        assert all("Colored document" in p.get_text() for p in doc)
        assert doc[0].get_text("dict")["blocks"][0]["lines"][0]["spans"][0]["color"] == 0xff0000
    assert [call.args[0].size[0] > call.args[0].size[1] for call in clf.predict.call_args_list] == [False, True, False, True]


def test_invalid_and_oversized_pdf():
    with pytest.raises(ValueError, match="decode PDF"):
        correct_pdf(b"invalid", MagicMock())
    with pytest.raises(ValueError, match="page limit"):
        correct_pdf(make_pdf(), MagicMock(), max_pages=2)


def test_exif_orientation_normalized():
    image = Image.new("RGB", (30, 50))
    exif = Image.Exif()
    exif[274] = 6
    data = io.BytesIO()
    image.save(data, format="JPEG", exif=exif)
    result = load_image(data.getvalue())
    assert result.size == (50, 30)
    assert result.getexif().get(274) is None


def test_home_and_pdf_upload():
    from contextlib import asynccontextmanager
    from app.api import routes
    original = routes._classifier
    clf = MagicMock()
    clf.predict.return_value = PredictionResult(0, 0, .99)
    @asynccontextmanager
    async def lifespan(app):
        set_classifier(clf)
        yield
    app = create_app()
    app.router.lifespan_context = lifespan
    try:
        with TestClient(app) as client:
            assert "Choose a document" in client.get("/").text
            response = client.post("/correct-orientation", files={"file": ("report.pdf", make_pdf(), "application/pdf")})
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/pdf"
            assert response.headers["X-Page-Count"] == "4"
            with fitz.open(stream=response.content, filetype="pdf") as pdf:
                assert pdf.page_count == 4
            bad = client.post("/correct-orientation", files={"file": ("report.pdf", b"bad", "application/pdf")})
            assert bad.status_code == 400
    finally:
        set_classifier(original)
