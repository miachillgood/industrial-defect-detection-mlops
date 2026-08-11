# Multi-stage image for the two serving surfaces.
#
#   docker build -f docker/Dockerfile --target api    -t spade-api .
#   docker build -f docker/Dockerfile --target review -t spade-review .
#
# CPU-only torch: the wheel from the default index pulls ~2 GB of CUDA
# libraries that are useless in a CPU container.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Heaviest layer first so it survives source edits.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.13.0" "torchvision==0.28.0"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve,review]"

COPY apps ./apps
COPY scripts ./scripts

# Bake the ImageNet weights in so a cold container does not reach the network.
RUN python -c "from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights; \
    wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)"

RUN useradd --create-home --uid 10001 spade && chown -R spade:spade /app
USER spade

ENV SPADE_BANK_DIR=/app/artifacts/banks \
    SPADE_DEVICE=cpu


# --------------------------------------------------------------------- api
FROM base AS api
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ------------------------------------------------------------------ review
FROM base AS review
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "apps/streamlit/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
