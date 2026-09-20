# Project review

FastAPI receives files, Pillow handles images, and ONNX Runtime serves a ResNet-18 CNN. Training uses PyTorch with four counterclockwise rotation labels.

## Fixed

- Training and API rotation conventions disagreed, reversing 90/270 corrections. API metadata now consistently uses clockwise angles while checkpoint labels remain compatible.
- Added a browser upload page at the root URL, plus download and image preview.
- PDFs now return as PDFs. Each rendered page is classified, then its existing display rotation is adjusted without rasterizing the original text and page content.
- Added EXIF normalization for phone images, bounded PDF preview size and a page limit.
- Replaced the unsuitable default dataset mixture with DocLayNet v1.2 and CORD. Preserved official splits, grouped local PDFs by file and reduced training memory.
- Added a launcher and regression coverage for angle direction, PDF text/color preservation and dataset splits.

## Remaining limitations

The available weights have not been retrained on these datasets. Their training history records 95.99% peak validation accuracy, but lacks a complete original data manifest. This is not a measured accuracy claim for the updated app.

The classifier always selects an angle, including for ambiguous or blank pages. A calibrated confidence threshold and review/abstain behavior need validation. The model sees 224x224 square images, reducing text detail and aspect ratio. Higher-resolution or aspect-preserving preprocessing requires matching retraining and export.

CPU inference and PDF processing run inside the async route and can block concurrent requests; use bounded workers or a job queue for a deployed service. Multipage TIFF currently processes the first frame only. Skew, cropping, perspective correction and dewarping remain outside scope.

The regression tests use controlled predictions to verify geometry and file preservation, not to benchmark model accuracy.
