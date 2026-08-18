# ============================================================
# SupportPilot AI — CPU Production Image
# ============================================================

FROM python:3.11-slim


# ============================================================
# PYTHON / PIP CONFIGURATION
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1


# ============================================================
# WORK DIRECTORY
# ============================================================

WORKDIR /app


# ============================================================
# INSTALL PYTORCH CPU
# ============================================================

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        torch==2.12.0 \
        --index-url https://download.pytorch.org/whl/cpu


# ============================================================
# INSTALL API DEPENDENCIES
# ============================================================

COPY requirements-api.txt ./requirements-api.txt

RUN python -m pip install \
    --no-cache-dir \
    -r requirements-api.txt


# ============================================================
# CREATE NON-ROOT USER
# ============================================================

RUN groupadd --system appgroup \
    && useradd \
        --system \
        --gid appgroup \
        --create-home \
        appuser


# ============================================================
# COPY APPLICATION SOURCE
# ============================================================

COPY --chown=appuser:appgroup \
    src \
    ./src


# ============================================================
# COPY PRODUCTION MODEL
# ============================================================

COPY --chown=appuser:appgroup \
    models/distilbert_supportpilot/best_model \
    ./models/distilbert_supportpilot/best_model


# ============================================================
# SWITCH TO NON-ROOT USER
# ============================================================

USER appuser


# ============================================================
# PORT
# ============================================================

EXPOSE 8000


# ============================================================
# HEALTH CHECK
# ============================================================

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]


# ============================================================
# START FASTAPI
# ============================================================

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]