# Deploy on Google Cloud Run

Cloud Run can host either:

- **Option A — Streamlit UI only** (simplest, one service, no Vercel)
- **Option B — FastAPI API + Vercel UI** (the original split architecture)

Both use the same E5 weights in `assets/`.

---

## 0. Prerequisites

```powershell
gcloud auth login
gcloud config set project <your-project-id>
```

Enable the required APIs once:

```powershell
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

> **Git LFS check:** Cloud Run uploads your *local* files. If `assets/best_global.pth`
> is a 133-byte pointer, run `git lfs install && git lfs pull` first.
> It must be ~82 MB.

---

## Option A — Streamlit on Cloud Run (recommended)

The root [`Dockerfile`](../Dockerfile) builds the Streamlit app and bakes
DINOv2-base into the image.

From the repository root:

```powershell
gcloud run deploy rice-e5 `
  --source . `
  --region asia-south1 `
  --allow-unauthenticated `
  --memory 4Gi `
  --cpu 2 `
  --timeout 3600 `
  --max-instances 1 `
  --session-affinity `
  --port 8080
```

The first build takes ~10–20 minutes (CPU torch + 330 MB DINOv2 download).
Later deploys are much faster.

When it finishes, gcloud prints a URL like:

```
https://rice-e5-abc123-uc.a.run.app
```

Open it — that is the full Streamlit interface. No Vercel needed.

Health check:

```powershell
curl.exe https://rice-e5-abc123-uc.a.run.app/_stcore/health
# ok
```

### Useful flags

| Flag | Why |
|---|---|
| `--memory 4Gi` | Streamlit + PyTorch + DINOv2 + EffNet |
| `--cpu 2` | faster CPU inference |
| `--timeout 3600` | long-lived Streamlit WebSocket sessions |
| `--max-instances 1` | predictable memory; a demo does not need horizontal scale |
| `--session-affinity` | keeps a browser session on the same instance |
| `--min-instances 1` | optional: removes cold starts, but costs money |
| `--cpu-boost` | optional: faster container startup |

### Cost

With `--min-instances 0` the service scales to zero when idle. For a demo this
usually stays inside the Cloud Run free tier (180k vCPU-s, 360k GiB-s, 2M
requests per month). Cold start is ~30–90 s because the model must load.

---

## Option B — FastAPI + Vercel

Use this if you still want the Next.js frontend on Vercel.

Build the self-contained API folder (weights + code):

```powershell
python scripts/build_space.py
```

Deploy it to Cloud Run:

```powershell
gcloud run deploy rice-e5-api `
  --source dist/space `
  --region asia-south1 `
  --allow-unauthenticated `
  --memory 4Gi `
  --cpu 2 `
  --timeout 300 `
  --max-instances 1 `
  --port 8080
```

The FastAPI Dockerfile reads `$PORT`, so the same image works on Cloud Run and
Hugging Face.

Test:

```powershell
curl.exe https://rice-e5-api-abc123-uc.a.run.app/health
curl.exe -X POST https://rice-e5-api-abc123-uc.a.run.app/predict `
  -F "file=@assets/samples/00_bacterial_leaf_blight.jpg"
```

Then in Vercel set:

```
SPACE_URL = https://rice-e5-api-abc123-uc.a.run.app
```

and redeploy the `web/` project. The Vercel proxy works unchanged.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Container fails to start | Cloud Run sets `$PORT`; the Dockerfile uses `${PORT:-8080}`. Check logs with `gcloud run services logs read rice-e5`. |
| `best_global.pth` missing / 133 bytes | Git LFS not pulled. `git lfs install && git lfs pull`, then redeploy. |
| Build fails at `pip install torch` | Check Cloud Build logs; the CPU wheel index is `download.pytorch.org/whl/cpu`. |
| 503 / cold start timeout | Increase `--timeout`, or set `--min-instances 1`. |
| Streamlit disconnects | Keep `--session-affinity` and `--max-instances 1`; raise `--timeout`. |
| Out of memory | Increase `--memory 8Gi`; the model is ~2 GB resident. |
| Permission denied creating service | Your account needs `roles/run.admin`, `roles/cloudbuild.builds.editor`, `roles/artifactregistry.writer`. |

---

## Which should you pick?

- **Just want the working demo?** Option A. One deploy, one URL, Streamlit UI.
- **Want the polished Vercel UI?** Option B. Two services, but the UI is nicer and
  the model stays behind a JSON API.

You can also deploy Option A now and switch to Option B later; the model code is
shared.
