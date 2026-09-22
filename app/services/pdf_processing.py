"""Correct PDF display rotation without rasterizing the output document."""
import fitz
from PIL import Image


def correct_pdf(content, classifier, max_pages=20, mode="pure", prompt=None,
                review_threshold=None, deadline=None):
    from app.services.processing import check_deadline, predict_page
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("Cannot decode PDF.") from exc
    with doc:
        if doc.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")
        if not doc.page_count:
            raise ValueError("PDF contains no pages.")
        if doc.page_count > max_pages:
            raise ValueError(f"PDF exceeds the {max_pages}-page limit.")
        results = []
        for page in doc:
            check_deadline(deadline)
            # Bound preview memory even for unusually large PDF page dimensions.
            scale = min(150 / 72, 1600 / max(page.rect.width, page.rect.height))
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale),
                                 colorspace=fitz.csRGB, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            result = predict_page(image, classifier, mode, prompt, review_threshold)
            # Rendering respects existing /Rotate. Add the visual correction.
            page.set_rotation((page.rotation + result.correction_rotation) % 360)
            results.append(result)
        check_deadline(deadline)
        return doc.tobytes(garbage=3, deflate=True), results
