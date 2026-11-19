"""Assemble a self-contained Hugging Face Docker Space in `dist/space/`.

The GitHub repo keeps only the Space *source* (`space/`). The 83 MB of weights
and the sample gallery are generated into `dist/space/`, which is git-ignored
in the GitHub repo and pushed to Hugging Face as its own repository.

Usage:
    python scripts/build_space.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "space"
OUT = ROOT / "dist" / "space"
ASSETS = ROOT / "assets"


def copy_samples(samples: pd.DataFrame) -> None:
    dst = OUT / "static" / "samples"
    dst.mkdir(parents=True, exist_ok=True)
    for f in (ASSETS / "samples").iterdir():
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            shutil.copy2(f, dst / f.name)
    (dst / "samples.json").write_text(
        json.dumps(samples[["file", "true_class", "source"]].to_dict(orient="records"), indent=2)
    )


def main() -> None:
    if not (ASSETS / "best_global.pth").exists():
        raise SystemExit(
            "Missing assets/best_global.pth.\n"
            "If you cloned without Git LFS: git lfs install && git lfs pull"
        )

    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(
        SRC,
        OUT,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", ".hf"),
    )

    (OUT / "assets").mkdir(exist_ok=True)
    for name in ["best_global.pth", "best_head.pth", "combined_manifest.csv", "e5_split_audit.json"]:
        shutil.copy2(ASSETS / name, OUT / "assets" / name)

    shutil.copy2(ROOT / "app" / "e5_infer.py", OUT / "e5_infer.py")

    samples = pd.read_csv(ASSETS / "samples" / "samples.csv", keep_default_na=False)
    copy_samples(samples)

    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"Space ready: {OUT}  ({total / 2**20:.1f} MB)")
    print("Next: cd dist/space && git init && git lfs install && ... (see docs/DEPLOYMENT.md)")


if __name__ == "__main__":
    main()
