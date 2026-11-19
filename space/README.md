---
title: Rice Leaf Disease E5
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Federated EfficientNetV2-S + DINOv2 rice-leaf classifier
---

# Rice Leaf Disease E5 — inference API

FastAPI service for the final **E5** model: frozen FedAvg EfficientNetV2-S
(1280-d) + frozen DINOv2-base (1536-d) → federated linear head (19 classes).

- `GET /health`
- `POST /predict` — multipart field `file`
- `GET /docs` — OpenAPI
- `GET /` — browser demo

This Space is the inference backend for the Vercel frontend. Model artifacts
(`assets/best_global.pth`, `assets/best_head.pth`, `assets/combined_manifest.csv`)
are copied in by `scripts/build_space.py` before pushing.
