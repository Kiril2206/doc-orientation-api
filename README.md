# Document Orientation Correction

Upload an image or PDF at **http://127.0.0.1:8000/**.
Images return in their original format. PDFs return as PDFs with page rotation
adjusted while preserving their text and page content. The CNN remains ResNet-18.

## Start on Windows

~~~powershell
.\start-app.ps1
~~~

Leave the terminal running. If PowerShell script execution is restricted:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
~~~

- [Browser app](http://127.0.0.1:8000/)
- [API documentation](http://127.0.0.1:8000/docs)
- [Model health](http://127.0.0.1:8000/health)
- Upload API: POST /correct-orientation, multipart field named file

Localhost links work only on the computer running the server.
The service requires model/orientation_model.onnx.

## Setup

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
~~~

## Training

See [dataset recommendations and training commands](docs/DATASETS.md)
and [project analysis](docs/PROJECT_REVIEW.md).
The default corpus is DocLayNet v1.2 plus CORD.
Train into a separate candidate checkpoint until evaluation passes.

## Docker

~~~powershell
docker compose up --build
~~~

## Tests

~~~powershell
.\.venv\Scripts\python.exe -m pytest
~~~

The default limits are 10 MB and 100 PDF pages.
Only quarter-turn rotation is supported; arbitrary skew is not corrected.
API angles are clockwise. PDF angle headers describe the first page, and
confidence is the lowest page confidence. Checkpoint labels remain counterclockwise.
