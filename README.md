# Document Orientation Correction

FastAPI + Pillow + ONNX Runtime CPU + PyMuPDF. Upload images or PDFs; corrected PDFs retain their original text and page content.

Three explicit pipelines:
- **Pure CV:** Model v2, EfficientNet-B0 at 384×384 by default, no OCR.
- **Hybrid:** legacy v1 with optional RapidOCR verification for ambiguous 0/180 predictions.
- **GenAI:** Gemini receives a page preview and a prompt, returns a clockwise correction, and the server rotates the original. No local weights or training required.

V2 weights require training. If they are absent, Pure returns 503 and the UI explains why. There is no silent v1 fallback.

**[Пошаговая инструкция Model v2: данные, обучение, API, экспорт и benchmark](docs/MODEL_V2.md)**

**[GenAI: настройка Gemini API и третьего режима](docs/GENAI.md)**

Start on Windows:

~~~powershell
.\start-app.ps1
~~~

Or:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
~~~

Open http://127.0.0.1:8000/ or http://127.0.0.1:8000/docs. The server must remain running.

Dependencies:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
~~~

Optional hybrid dependencies:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-hybrid.txt
~~~

POST /correct-orientation accepts multipart file and mode=pure|hybrid|genai.
GenAI accepts an optional genai_prompt (up to 2000 characters).
If mode is omitted, APP_INFERENCE_MODE applies. /health lists available models.
Set APP_GEMINI_API_KEY in your server's .env to enable GenAI, then restart.
See .env.example for model paths, Gemini settings and thresholds.

Docker:

~~~powershell
docker compose up --build
~~~

For an OCR-enabled image, build with --build-arg INSTALL_HYBRID=true before starting Compose.

Tests:

~~~powershell
.\.venv\Scripts\python.exe -m pytest
~~~

Default limits: 10 MB, 100 PDF pages for local modes, 10 for GenAI. API angles are clockwise; checkpoint class angles remain counterclockwise. Only quarter turns are corrected, not arbitrary skew or perspective.
