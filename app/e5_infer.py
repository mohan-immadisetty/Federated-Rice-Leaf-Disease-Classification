"""Inference for the final E5 rice-leaf model.

E5 = frozen FedAvg EfficientNetV2-S avgpool features (1280)
     ++ frozen DINOv2-base [CLS, mean patch] (1536)
     -> standardise with the head's training mean/std -> linear head (19 classes).

Preprocessing mirrors the training/test pipeline exactly:
  EffNet: JPEG draft, EXIF transpose, RGB, thumbnail to max side 384 (LANCZOS),
          JPEG q95 round trip (= the md5 cache), Resize 256x256, checkpoint
          mean/std, fp16 autocast on GPU.
  DINOv2: EXIF transpose, RGB, bicubic 224x224 (no crop), ImageNet mean/std.

The FedAvg baseline prediction (the backbone's own classifier) comes from the
same EffNet forward pass.

Paths are resolved relative to the repository unless overridden:
  RICE_E5_ROOT             repository root (default: parent of this file)
  RICE_E5_ASSETS           asset folder (default: <root>/assets)
  RICE_E5_BACKBONE         default: <assets>/best_global.pth
  RICE_E5_HEAD             default: <assets>/best_head.pth
  RICE_E5_MANIFEST         default: <assets>/combined_manifest.csv
  RICE_E5_AUDIT            default: <assets>/e5_split_audit.json
  RICE_E5_DINO             default: facebook/dinov2-base
  RICE_E5_LOCAL_FILES_ONLY "1" to forbid Hugging Face downloads
  RICE_E5_CACHE_DIR        optional Hugging Face cache directory
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s

ROOT = Path(os.environ.get("RICE_E5_ROOT", Path(__file__).resolve().parents[1]))
ASSETS = Path(os.environ.get("RICE_E5_ASSETS", ROOT / "assets"))
BACKBONE = Path(os.environ.get("RICE_E5_BACKBONE", ASSETS / "best_global.pth"))
HEAD = Path(os.environ.get("RICE_E5_HEAD", ASSETS / "best_head.pth"))
MANIFEST = Path(os.environ.get("RICE_E5_MANIFEST", ASSETS / "combined_manifest.csv"))
AUDIT = Path(os.environ.get("RICE_E5_AUDIT", ASSETS / "e5_split_audit.json"))
DINO = os.environ.get("RICE_E5_DINO", "facebook/dinov2-base")
LOCAL_ONLY = os.environ.get("RICE_E5_LOCAL_FILES_ONLY", "0") == "1"
CACHE_DIR = os.environ.get("RICE_E5_CACHE_DIR") or None

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)
UNKNOWN = "— I don't know / not specified —"


def pretty(class_name: str) -> str:
    """bacterial_leaf_blight -> Bacterial leaf blight"""
    return class_name.replace("_", " ").capitalize()


def md5_bytes(raw: bytes) -> str:
    return hashlib.md5(raw).hexdigest()


def _device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class E5Model:
    def __init__(self, device: str | None = None):
        from transformers import AutoModel

        for path, label in ((BACKBONE, "backbone"), (HEAD, "head"), (MANIFEST, "manifest")):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing E5 {label}: {path}\n"
                    "Run `python scripts/download_assets.py` or copy the files into assets/."
                )

        self.dev = _device(device)
        ck = torch.load(BACKBONE, map_location="cpu", weights_only=False)
        hd = torch.load(HEAD, map_location="cpu", weights_only=False)
        assert hd["feature_set"] == "concat" and tuple(hd["weight"].shape) == (19, 2816)
        assert list(hd["class_names"]) == list(ck["class_names"])
        self.classes = list(ck["class_names"])

        self.effnet = efficientnet_v2_s(weights=None)
        self.effnet.classifier[1] = nn.Linear(1280, 19)
        self.effnet.load_state_dict(ck["global_state"], strict=True)
        self.effnet.to(self.dev).eval()

        self.eff_tf = T.Compose([
            T.Resize((256, 256), antialias=True),
            T.ToTensor(),
            T.Normalize(list(ck["mean"]), list(ck["std"])),
        ])
        self.dino = AutoModel.from_pretrained(
            DINO, local_files_only=LOCAL_ONLY, cache_dir=CACHE_DIR
        ).to(self.dev).eval()

        self.W = hd["weight"].float()
        self.B = hd["bias"].float()
        self.mu = hd["feat_mean"].float()
        self.sd = hd["feat_std"].float()
        self.head_round = int(hd.get("round", -1))

    def _autocast(self):
        if self.dev.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return torch.autocast("cpu", enabled=False)

    @staticmethod
    def _cache_image(raw: bytes) -> Image.Image:
        """Reproduce the 384-px JPEG q95 cache the EffNet backbone was trained on."""
        with Image.open(io.BytesIO(raw)) as im:
            im.draft("RGB", (768, 768))
            im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((384, 384), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=95)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    @staticmethod
    def _dino_tensor(raw: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(raw)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB").resize((224, 224), Image.BICUBIC)
        arr = (np.asarray(im, np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())

    @torch.no_grad()
    def predict(self, raw: bytes) -> dict:
        x = self.eff_tf(self._cache_image(raw)).unsqueeze(0).to(self.dev)
        xd = self._dino_tensor(raw).unsqueeze(0).to(self.dev)
        with self._autocast():
            h = torch.flatten(self.effnet.avgpool(self.effnet.features(x)), 1)
            z_fedavg = self.effnet.classifier(h)
            hs = self.dino(pixel_values=xd).last_hidden_state
            d = torch.cat([hs[:, 0], hs[:, 1:].mean(1)], 1)
        # EffNet features were stored as fp16 when the E5 head was fitted.
        X = torch.cat([h.half().float().cpu(), d.float().cpu()], 1)
        logits = ((X - self.mu) / self.sd) @ self.W.T + self.B
        return {
            "e5_prob": torch.softmax(logits, 1)[0].numpy(),
            "fedavg_prob": torch.softmax(z_fedavg.float().cpu(), 1)[0].numpy(),
        }

    def predict_path(self, path: str | Path) -> dict:
        return self.predict(Path(path).read_bytes())


@lru_cache(maxsize=1)
def load_manifest() -> pd.DataFrame:
    cols = ["image_path", "source", "canonical_class", "md5", "split", "exclusion_reason"]
    man = pd.read_csv(MANIFEST, keep_default_na=False, usecols=cols)
    return man


@lru_cache(maxsize=1)
def load_audit() -> dict:
    if not AUDIT.exists():
        return {"eval_images_excluded_by_reaudit": []}
    return json.loads(AUDIT.read_text())


def official_test(man: pd.DataFrame | None = None) -> pd.DataFrame:
    man = load_manifest() if man is None else man
    leak = set(load_audit().get("eval_images_excluded_by_reaudit", []))
    return man[(man.split == "test") & ~man.image_path.isin(leak)]


def lookup(md5: str, man: pd.DataFrame | None = None):
    man = load_manifest() if man is None else man
    hit = man[man.md5 == md5]
    return None if hit.empty else hit.iloc[0]


def top_k(prob: np.ndarray, classes: list[str], k: int = 5) -> list[dict]:
    order = prob.argsort()[::-1][:k]
    return [
        {"class": classes[i], "label": pretty(classes[i]), "prob": float(prob[i])}
        for i in order
    ]
