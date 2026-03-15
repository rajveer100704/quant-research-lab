# ============================================================
# AI Quant Trading Research Platform — Dockerfile
# Multi-stage build for lightweight production image.
# ============================================================

FROM python:3.12-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[dev]"

# Copy source
COPY . .

# Expose ports: Streamlit (8501)
EXPOSE 8501

# Default: run dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]