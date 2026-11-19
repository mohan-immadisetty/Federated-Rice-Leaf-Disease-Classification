"""FastAPI inference service for the final E5 rice-leaf model.

Endpoints
  GET  /health          service + model status
  POST /predict         multipart form field `file` -> JSON predictions
  GET  /                small browser demo (static/index.html)
  GET  /static/*        static assets

This is the API consumed by the Vercel frontend (web/) and can also be used
directly. It runs on Hugging Face Spaces (Docker) or locally:

    uvicorn app:app --host 0.0.0.0 --port 7860
"""
from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from e5_infer import (
    E5Model,
    load_manifest,
    md5_bytes,
    official_test,
    pretty,
    top_k,
)

STATE: dict = {"model": None, "error": None}
LOCK = threading.Lock()
STATIC = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["model"] = E5Model()
    except Exception as exc:  # keep the service up so /health explains the failure
        STATE["error"] = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(
    title="Rice Leaf Disease E5 API",
    version="1.0.0",
    description="Federated EfficientNetV2-S + DINOv2 linear head (19 rice-leaf classes).",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    model = STATE["model"]
    return {
        "status": "ok" if model is not None else "error",
        "model": "E5 (concat, seed 42)" if model is not None else None,
        "head_round": getattr(model, "head_round", None),
        "classes": len(getattr(model, "classes", []) or []) or None,
        "device": str(getattr(model, "dev", "")) or None,
        "error": STATE["error"],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model: E5Model | None = STATE["model"]
    if model is None:
        raise HTTPException(503, detail=f"Model not loaded: {STATE['error']}")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, detail="Empty file")
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(413, detail="Image larger than 25 MB")

    t0 = time.perf_counter()
    try:
        # one CPU-bound inference at a time; keeps memory predictable on small hosts
        with LOCK:
            out = await asyncio.to_thread(model.predict, raw)
    except Exception as exc:
        raise HTTPException(400, detail=f"Could not read image: {type(exc).__name__}: {exc}") from exc
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    e5 = out["e5_prob"]
    fa = out["fedavg_prob"]
    classes = model.classes
    md5 = md5_bytes(raw)

    man = load_manifest()
    hit = man[man.md5 == md5]
    row = None if hit.empty else hit.iloc[0]
    official = set(official_test()["image_path"]) if row is not None else set()

    e5_idx, fa_idx = int(e5.argmax()), int(fa.argmax())
    return {
        "md5": md5,
        "e5": {"class": classes[e5_idx], "label": pretty(classes[e5_idx]), "prob": float(e5[e5_idx])},
        "fedavg": {"class": classes[fa_idx], "label": pretty(classes[fa_idx]), "prob": float(fa[fa_idx])},
        "top5": top_k(e5, classes, 5),
        "in_dataset": row is not None,
        "true_label": None if row is None else row.canonical_class,
        "true_label_pretty": None if row is None else pretty(row.canonical_class),
        "source": None if row is None else row.source,
        "split": None if row is None else row.split,
        "official_test": bool(row is not None and row.image_path in official),
        "exclusion_reason": None if row is None else (row.exclusion_reason or None),
        "latency_ms": latency_ms,
    }


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))
