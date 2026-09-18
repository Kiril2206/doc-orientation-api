# ==============================================================================
# Stage 1: Builder — install Python dependencies
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install dependencies into a separate directory for clean copy
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==============================================================================
# Stage 2: Runtime — minimal image with app + model
# ==============================================================================
FROM python:3.11-slim

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -s /sbin/nologin appuser

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code and model
COPY app/ ./app/
COPY model/ ./model/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
