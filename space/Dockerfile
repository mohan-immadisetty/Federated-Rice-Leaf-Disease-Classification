FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RICE_E5_ROOT=/app \
    RICE_E5_ASSETS=/app/assets \
    HF_HOME=/app/.hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only torch keeps the image ~10x smaller than the default CUDA wheels.
RUN pip install --upgrade pip \
 && pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake DINOv2-base into the image so the first request is fast.
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-base')"

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
