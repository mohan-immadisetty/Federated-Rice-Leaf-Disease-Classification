# 🌾 Rice Leaf Disease Recognition with Federated Learning (E5)

**Final E5 model** — frozen FedAvg EfficientNetV2-S features fused with frozen
DINOv2-base features, classified by a **federated linear head** over **19 rice-leaf
disease classes** across **5 heterogeneous clients**.

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Live demo:** _add your Vercel URL here after deploying_ ·
> **API:** _add your Hugging Face Space URL here_

---

## Results

Official held-out test, **n = 2,793** (duplicate-family bootstrap, 2,000 resamples):

| Model | Accuracy | Macro-F1 | Strict acc / F1 (n = 2,484) |
|---|---|---|---|
| **E5 (ours)** | **92.34%** [90.21, 94.26] | **89.58%** [86.52, 92.38] | 91.59 / 89.01 |
| FedAvg + masked local loss (same backbone) | 85.82% [83.57, 87.94] | 67.39% [63.19, 70.54] | 84.46 / 66.65 |
| **Δ (E5 − FedAvg)** | **+6.52** [+5.40, +7.72] | **+22.19** [+18.69, +26.42] | +7.13 / +22.36 |

Backbone-seed robustness (42 primary, 43/44 robustness):

| Backbone seed | E5 acc / macro-F1 | FedAvg acc / macro-F1 |
|---|---|---|
| 42 | 92.34 / 89.58 | 85.82 / 67.39 |
| 43 | 93.45 / 89.81 | 85.96 / 68.47 |
| 44 | 92.52 / 87.54 | 84.78 / 65.64 |
| **mean ± SD** | **92.77 ± 0.60 / 88.97 ± 1.25** | 85.52 ± 0.64 / 67.17 ± 1.43 |

E5 has **no zero-recall class** for any seed. Full protocol, predictions,
bootstrap intervals and limitations: [`repro/results/final_test/`](repro/results/final_test/).

---

## How E5 works

```
rice leaf image
      │
      ├─ 384-px JPEG q95 cache ──► frozen EfficientNetV2-S ──► avgpool 1280-d ─┐
      │                            (FedAvg, seed 42, masked CE)               │
      │                                                                       ├─ concat 2816-d
      └─ 224×224 bicubic ───────► frozen DINOv2-base ──► [CLS, mean patch] ───┘
                                                                              │
                                              federated standardisation (μ, σ)
                                                                              │
                                                     linear head 19 classes ──┴──► softmax
```

- The backbone is **frozen** after 20 rounds of FedAvg with a masked local loss.
- DINOv2-base is **frozen** (never fine-tuned).
- Only the **19 × 2816 linear head** is trained federatedly: **FedSGD + server Adam**
  (one full-batch local gradient step per round, full participation).
- Training uses **data policy v1** (cross-source label-conflict near-copies removed).
- The FedAvg baseline is the backbone's own classifier, computed in the same forward pass.

---

## Repository layout

```
.
├── app/                  # shared inference core + Streamlit demo
│   ├── e5_infer.py       # exact E5 preprocessing + forward pass
│   └── streamlit_app.py
├── assets/               # E5 artifacts (Git LFS for *.pth)
│   ├── best_global.pth   # 81.7 MB FedAvg EfficientNetV2-S
│   ├── best_head.pth     # 239 KB E5 linear head
│   ├── combined_manifest.csv
│   ├── e5_split_audit.json
│   └── samples/          # 19 real held-out test images (one per class)
├── space/                # Hugging Face Docker Space (FastAPI + PyTorch)
│   ├── app.py
│   ├── Dockerfile
│   └── static/index.html
├── web/                  # Next.js frontend for Vercel
│   ├── app/page.tsx
│   └── app/api/predict/route.ts
├── repro/                # E5 training + final-test artifacts
│   ├── train_e5_head.py
│   ├── extract_fl_features.py
│   ├── configs/e5.json
│   └── results/
├── docs/                 # architecture, deployment, model card
├── scripts/build_space.py
└── requirements.txt
```

Only E5-related files are included. The research history (E1–E4, CCVR, seed
sweeps, notebooks, report/paper sources, caches, venvs) is intentionally excluded.

---

## Quickstart (local Streamlit)

```bash
git lfs install
git clone <your-repo-url>
cd rice-e5

python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

First run downloads `facebook/dinov2-base` (~330 MB) from Hugging Face.
The first inference also warms up CUDA/CPU kernels.

**GPU vs CPU:** the app uses CUDA if available, otherwise CPU. CPU inference
is a few seconds per image.

---

## API

The FastAPI service in [`space/`](space/) exposes:

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | model + device status |
| `POST` | `/predict` | multipart field `file` → predictions |
| `GET` | `/docs` | OpenAPI UI |
| `GET` | `/` | browser demo |

Example:

```bash
curl -X POST "$SPACE_URL/predict" -F "file=@assets/samples/00_bacterial_leaf_blight.jpg"
```

```json
{
  "e5": {"class": "bacterial_leaf_blight", "label": "Bacterial leaf blight", "prob": 0.6969},
  "fedavg": {"class": "tungro", "label": "Tungro", "prob": 0.7389},
  "top5": [{"class": "bacterial_leaf_blight", "label": "Bacterial leaf blight", "prob": 0.6969}, "..."],
  "in_dataset": true,
  "true_label": "bacterial_leaf_blight",
  "split": "test",
  "official_test": true,
  "latency_ms": 2480.1
}
```

---

## Deployment

**Why two services?** Vercel is serverless: it cannot run Streamlit, and the E5
assets (~420 MB including DINOv2) exceed the 250 MB function limit. So:

```
Browser ──► Vercel (Next.js UI + thin API proxy) ──► Hugging Face Space (FastAPI + PyTorch)
```

- **Model + API** → Hugging Face Docker Space (free CPU, 16 GB RAM).
- **UI** → Vercel (Next.js 15, no model weights).

`python scripts/build_space.py` assembles a self-contained Space in `dist/space/`
(source + weights + samples); that folder is pushed to Hugging Face.
The browser calls `/api/predict` on Vercel; the route forwards the image to the
Space. `SPACE_URL` is a server-side environment variable.

Step-by-step instructions: **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

---

## Reproduce E5

```bash
python repro/train_e5_head.py \
  --exp-id E5_concat_v1_lr1e-3 --features concat --train-manifest v1 \
  --lr 1e-3 --max-rounds 3000 --eval-every 10 --patience-rounds 300 \
  --weight-decay 0 --family-weight-power 0.5 --seed 42
```

See [`repro/README.md`](repro/README.md) for the exact pipeline, required
feature caches and environment variables.

---

## Dataset

Five public rice-leaf datasets, deduplicated and split **once**, before any
model was trained (`split_meta.json`):

| Client | Train images |
|---|---:|
| bd_rice8 | 1,282 |
| paddy_doctor | 7,222 |
| philippines | 1,331 |
| riceleafbd | 997 |
| sethy | 3,230 |
| **total** | **14,062** |

- 21,731 files found → 1,428 excluded (1,349 exact duplicates, 79 label-conflicting duplicates).
- 19 canonical classes; stratified group split with duplicate families kept together.
- Official test = 3,047 test images − 254 duplicate re-audit exclusions = **2,793**.
- The raw images are **not** redistributed here; use the original dataset sources
  and the included manifest.

---

## Limitations

Read [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) before using this model.

1. **Not a fresh independent test.** The test split was used earlier for the
   FedAvg baseline and head diagnostics. E5 was designed on validation only.
2. **Near-duplicates.** 309 official test images have a training neighbour at
   DINOv2 cosine ≥ 0.985; the strict subset (n = 2,484) removes them.
3. **Labels unreviewed.** Planned pathologist review was not performed.
4. **Rare classes.** Several classes have 11–15 test images; F1 is unstable.
5. **Demo only.** Not a substitute for an agronomist or lab diagnosis.

---

## Citation

```bibtex
@misc{naimmadisetty2026ricee5,
  title  = {Federated Learning with Frozen Dual-Backbone Features for Rice Leaf Disease Recognition across Heterogeneous Datasets},
  author = {NaImmadisetty, Mohan Venkata},
  year   = {2026},
  note   = {B.Tech mini project}
}
```

## License

Code: [MIT](LICENSE). Model weights and sample images follow the licenses of the
source datasets; check each dataset before redistribution.
