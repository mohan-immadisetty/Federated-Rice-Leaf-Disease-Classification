# Deployment guide

Two services, one repo:

```
Browser
   │
   ▼
Vercel  (web/ — Next.js UI + thin /api/predict proxy)
   │  HTTPS, server-side SPACE_URL
   ▼
Hugging Face Space  (space/ — Docker + FastAPI + PyTorch + E5 weights)
```

Vercel cannot host the model itself: Streamlit needs a long-running server, and
the E5 assets (~420 MB with DINOv2) exceed Vercel's 250 MB function bundle.

---

## 0. Prerequisites

| Tool | Why |
|---|---|
| Git + Git LFS | `best_global.pth` is 81.7 MB |
| GitHub account | source repo |
| Hugging Face account | model + API hosting (free CPU) |
| Vercel account | frontend hosting |
| Node 20+ | local Next.js build (optional) |

```bash
git lfs install
node --version   # 20.x or 22.x
```

---

## 1. Push the source to GitHub

Create an **empty** repository on GitHub (no README, no .gitignore), then:

```bash
cd D:\rice-e5

git init
git branch -M main
git lfs install

git add .gitattributes
git add .
git commit -m "E5: federated rice-leaf model, API and Vercel frontend"

git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

Check that LFS worked:

```bash
git lfs ls-files
# assets/best_global.pth  (should show an object id, not a 130-byte pointer)
```

If GitHub rejects the push, the repo was probably created with a README. Pull
first (`git pull --rebase origin main`) or delete and recreate it empty.

> **Size:** ~90 MB of LFS objects (backbone + head) plus ~15 MB of code, CSV and
> results. GitHub's free LFS quota is 1 GB.

---

## 2. Build and push the Hugging Face Space

The Space is a **Docker** Space. It needs the weights inside `space/`, so build
the folder first:

```bash
python scripts/build_space.py
# Space ready: D:\rice-e5\dist\space  (83.5 MB)
```

Create the Space on huggingface.co:

1. **New Space** → name `rice-e5` → **SDK: Docker** → **Blank** → CPU basic (free).
2. Make it **public** (the Vercel proxy calls it without a token).

> **Shortcut:** if `hf auth login` already stores a **write** token, this one
> command creates the Space and uploads it:
>
> ```bash
> python scripts/build_space.py
> python scripts/push_space.py
> ```
>
> A read-only token returns `403 You don't have the rights to create a space`.

Then push the generated folder as its own repository:

```bash
cd dist/space

git init
git branch -M main
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add .
git commit -m "E5 inference API"

git remote add space https://huggingface.co/spaces/<your-user>/rice-e5
git push space main
```

When prompted: username = your HF username, password = a **write** access token
(https://huggingface.co/settings/tokens). Or run `hf auth login` first.

The Space builds for ~3–8 minutes (CPU torch + DINOv2 download). Watch the
**Logs** tab. When it is running:

```bash
curl https://<your-user>-rice-e5.hf.space/health
# {"status":"ok","model":"E5 (concat, seed 42)","head_round":80,...}

curl -X POST https://<your-user>-rice-e5.hf.space/predict \
  -F "file=@assets/samples/00_bacterial_leaf_blight.jpg"
```

Open `https://<your-user>-rice-e5.hf.space/` for the built-in browser demo.

---

## 3. Deploy the frontend to Vercel

### Option A — Vercel dashboard (recommended)

1. **Add New → Project** → import the GitHub repo.
2. **Root Directory:** `web`
3. Framework preset: **Next.js** (auto-detected).
4. **Environment Variables:**
   - `SPACE_URL` = `https://<your-user>-rice-e5.hf.space` (no trailing slash)
5. **Deploy**.

### Option B — Vercel CLI

```bash
cd web
npm install
echo "SPACE_URL=https://<your-user>-rice-e5.hf.space" > .env.local
npm run build          # verify locally
npx vercel             # preview
npx vercel --prod      # production
```

Set the same `SPACE_URL` in **Project → Settings → Environment Variables** for
Production, Preview and Development.

Test the deployed site: upload a leaf image and click one of the 19 sample
thumbnails. A correct sample should show ✅ and the true label.

---

## 4. Updating

| Change | Command |
|---|---|
| App/UI/code | `git add . && git commit -m "..." && git push` (Vercel auto-deploys) |
| Space code/weights | `python scripts/build_space.py` then `cd dist/space && git add . && git commit -m "..." && git push space main` |
| New weights | replace `assets/best_global.pth` / `assets/best_head.pth`, rebuild the Space |

`assets/` is the single source of truth; `dist/space/` is generated and
git-ignored.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Missing E5 backbone: ...best_global.pth` | Cloned without LFS. Run `git lfs install && git lfs pull`. |
| GitHub file is 130 bytes | It is an LFS pointer, not the weight. Same fix. |
| Space build fails on `torch` | The Dockerfile installs CPU wheels from `download.pytorch.org`; check the build log for network/proxy errors. |
| `/health` returns 503/error | Weights or manifest missing from the Space image. Run `scripts/build_space.py` before pushing. |
| Vercel 500: `SPACE_URL is not configured` | Add the env var and redeploy. |
| Vercel 502: `Inference service unreachable` | Space is asleep, building, private, or the URL is wrong. Open it once in a browser to wake it. |
| Vercel 413 / body too large | Images > ~4.5 MB exceed the serverless request limit. The UI downscales >3.5 MB automatically; for the API, resize first. |
| Vercel 504 / 60 s timeout | Free Space cold start + CPU inference can exceed 60 s. Retry once it is warm, or use a larger Space/plan. |
| `true_label` is null for a dataset image | The image was resized/edited (MD5 changed). Upload the original bytes. |
| CORS error | You are calling the Space directly. Use the Vercel `/api/predict` proxy, which is same-origin. |

---

## Cost and limits (free tiers)

| Service | Limit that matters here |
|---|---|
| GitHub | 1 GB LFS storage/bandwidth per month |
| HF Space (CPU basic) | 2 vCPU, 16 GB RAM, sleeps after inactivity |
| Vercel Hobby | 4.5 MB request body, 60 s function duration |

Everything above fits in the free tiers for a demo.
