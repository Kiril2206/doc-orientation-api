# ==============================================================================
# Stage 1: Builder — install Python dependencies
# ==============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies for wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==============================================================================
# Stage 2: Runtime — hardened, minimal production container
# ==============================================================================
FROM python:3.12-slim

# System runtime dependencies for OpenCV, PyMuPDF, and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -s /sbin/nologin appuser

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code and trained models
COPY app/ ./app/
COPY model/ ./model/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKERS=1 \
    PORT=8000

# Expose HTTP port
EXPOSE 8000

# Container liveness check (bypasses auth and does not invoke models)
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/live || exit 1

# Run single-worker uvicorn to protect native C++ libraries (RapidOCR/PyMuPDF/ONNX Runtime)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
