# Reconocimiento facial — Python
# Multi-stage: compilación de dlib en una imagen intermedia, runtime ligero.

ARG PYTHON_VERSION=3.11
ARG DEBIAN_VERSION=bookworm

FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} AS builder

# Dependencias del sistema para compilar dlib, numpy, OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# Pre-instalar dependencias en /install para reutilizar en la siguiente etapa.
RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    "numpy>=1.24" \
    "Pillow>=10.0" \
    "opencv-python>=4.8" \
    "face-recognition>=1.3.0" \
    "dlib>=19.24" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.27" \
    "typer>=0.9" \
    "rich>=13.7" \
    "pydantic>=2.5" \
    "pydantic-settings>=2.1" \
    "loguru>=0.7" \
    "python-multipart>=0.0.9"

# ---- Runtime ---------------------------------------------------------------

FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} AS runtime

# Solo libs runtime (sin compiladores).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* \
    && rm -rf /wheels

COPY src ./src
COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir --no-deps -e .

# Usuario no-root por seguridad.
RUN useradd --create-home --shell /bin/bash rfuser \
    && mkdir -p /app/data \
    && chown -R rfuser:rfuser /app
USER rfuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "reconocimiento_facial.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]