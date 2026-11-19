# Architecture

## Model

E5 is a **frozen dual-backbone feature fusion** model:

| Stage | Detail |
|---|---|
| Backbone A | EfficientNetV2-S, FedAvg-trained for 20 rounds with masked local CE (seed 42), then frozen |
| Backbone B | DINOv2-base (`facebook/dinov2-base`), frozen, never fine-tuned |
| Features | A: 1280-d avgpool; B: 1536-d `[CLS]` + mean patch tokens |
| Fusion | concatenation → 2816-d |
| Normalisation | head-specific training mean/std (`feat_mean`, `feat_std`) |
| Classifier | linear 19 × 2816, trained with FedSGD + server Adam |

The FedAvg baseline is the frozen EfficientNet's own 19-way classifier, read
from the same forward pass. That makes the comparison free and exact.

### Preprocessing (must match training)

**EfficientNet branch**

1. PIL `draft("RGB", (768, 768))` (JPEG DCT scaling).
2. `ImageOps.exif_transpose` → RGB.
3. `thumbnail((384, 384), LANCZOS)`.
4. JPEG q95 round trip (this reproduces the on-disk MD5 cache).
5. `Resize((256, 256))` → tensor → checkpoint mean/std.

**DINOv2 branch**

1. `ImageOps.exif_transpose` → RGB.
2. Bicubic resize to 224 × 224 (no crop).
3. ImageNet mean/std on NumPy `float32` → tensor.

The two branches intentionally see different resolutions; this is part of E5,
not a bug.

---

## Repository components

```
app/e5_infer.py          single source of truth for inference
app/streamlit_app.py     local/human demo (Streamlit)
space/app.py             FastAPI service (same e5_infer.py, copied by build_space.py)
space/static/index.html  zero-build browser demo served by the Space
web/app/page.tsx         Next.js UI (client component)
web/app/api/predict      server-side proxy to the Space (hides SPACE_URL, avoids CORS)
repro/                   training + evaluation scripts and frozen results
scripts/build_space.py   assembles a self-contained Space folder
```

`e5_infer.py` resolves paths from `RICE_E5_ROOT` / `RICE_E5_ASSETS`, so the same
file works in the repo (`assets/`) and in the Docker image (`/app/assets`).

---

## Request flow

```
1. Browser: user drops leaf.jpg
2. web/app/page.tsx: if > 3.5 MB, downscale to max 2200 px JPEG q90
3. POST /api/predict (Vercel, Node runtime, 60 s max)
4. fetch(SPACE_URL/predict, multipart)  (55 s abort)
5. Space: FastAPI -> threading.Lock -> E5Model.predict
6. Response JSON: e5, fedavg, top5, dataset metadata, latency
7. UI renders prediction, top-5 bars and true-label verdict
```

### Why the proxy?

- **CORS:** the browser never talks to `*.hf.space` directly.
- **Secret-ish config:** `SPACE_URL` stays server-side.
- **Uniform errors:** Vercel returns a clean JSON `detail` on upstream failure.

The trade-off is Vercel's 4.5 MB body limit and 60 s duration. The client
downscales large images; the Space is kept warm by normal demo traffic. For
heavy use, call the Space directly (CORS is open) or move to a GPU Space.

---

## Error handling

| Layer | Behaviour |
|---|---|
| `E5Model.__init__` | raises `FileNotFoundError` with the missing path and a fix hint |
| FastAPI lifespan | catches load errors, keeps the service up; `/health` reports them |
| `/predict` | 400 empty/unreadable, 413 > 25 MB, 503 model not loaded |
| Vercel route | 400 missing field, 500 missing `SPACE_URL`, 502 upstream failure |
| UI | inline error card; the page never crashes |

Inference is serialised with a lock: the free CPU Space has 2 vCPU and a single
model instance, so parallel requests would only add memory pressure.

---

## Performance

Measured locally (RTX GPU): model load ~60 s cold (first ever DINOv2 load),
inference ~2.5 s/image.

On the free HF CPU Space expect roughly 5–15 s per image after warm-up, and a
30–60 s cold start. That is acceptable for a demo and is why the UI says the
first request can be slow.

### Optimisation options (not implemented)

- Precompute DINOv2 features for the 19 samples and serve them instantly.
- Export both backbones to ONNX Runtime; the linear head is trivial.
- Use a GPU Space or a persistent inference host.
- Cache results by MD5 in the Space process.

---

## Security / privacy

- The Space is public and stateless; uploaded images are processed in memory
  and not written to disk.
- No analytics, accounts or database.
- The API accepts images only (Pillow decode); 25 MB cap.
- The dataset manifest exposes only paths and labels, no personal data.
