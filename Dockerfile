# Streamlit demo for Google Cloud Run (single service, no Vercel needed).
# Build context: repository root.  The model weights live in assets/.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RICE_E5_ROOT=/app \
    RICE_E5_ASSETS=/app/assets \
    RICE_E5_LOCAL_FILES_ONLY=1 \
    HF_HOME=/app/.hf \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# CPU-only torch: much smaller and cheaper than the default CUDA wheels.
RUN pip install --upgrade pip \
 && pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY assets ./assets

# Bake DINOv2-base into the image so cold starts do not download 330 MB.
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-base')" \
 && chmod -R 777 /app/.hf

EXPOSE 8080

# Cloud Run injects $PORT (default 8080). Shell form is required for expansion.
CMD streamlit run app/streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=${PORT:-8080} \
    --server.headless=true
