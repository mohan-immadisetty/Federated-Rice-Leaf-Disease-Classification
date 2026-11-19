"""Frozen EfficientNetV2-S feature + full-model logit extraction for one backbone seed and one split (val or test).

The procedure is identical to tools/research_extract_fl_features.py:
  manifest order per client, batch 32, 6 loader threads, Resize 256x256 on the 384-px md5 cache,
  checkpoint mean/std, eval mode, fp16 autocast, avgpool.
It additionally saves the checkpoint's own classifier logits from the SAME forward pass
(= model(x), the FedAvg baseline output).
Label columns are dropped when the manifest is loaded and are never read. Writes one new file; refuses to overwrite.
Usage: extract_effnet.py --ckpt PATH --expect-seed S --split {val,test} --out FILE.pt
"""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s

R = Path(os.environ.get("RICE_E5_DATA_ROOT", r"D:\rice"))
SPLIT_DIR = R / "splits" / "combined_rice_datasets"
CACHE = Path(os.environ.get("RICE_E5_IMAGE_CACHE", R / "cache" / "combined_rice_datasets" / "max384_q95"))
EXP42 = R / "outputs" / "fedavg_masked_local_loss_efficientnetv2_s" / "experiments" / "20261014_104304"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLIENTS = ["bd_rice8", "paddy_doctor", "philippines", "riceleafbd", "sethy"]
LABEL_COLS = {"canonical_class", "label", "original_class", "y"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--expect-seed", type=int, required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    t0 = time.time()
    meta = json.loads((SPLIT_DIR / "split_meta.json").read_text())
    assert sha(SPLIT_DIR / "combined_manifest.csv") == meta["manifest_sha256"]
    leak = set(json.loads((EXP42 / "experiment_config.json").read_text())["split_fingerprint"]["eval_images_excluded_by_reaudit"])
    man = pd.read_csv(SPLIT_DIR / "combined_manifest.csv", keep_default_na=False, usecols=lambda c: c not in LABEL_COLS)
    kept = man[man.split != "excluded"]
    frame = kept[(kept.split == a.split) & ~kept.image_path.isin(leak)].reset_index(drop=True)
    assert len(frame) == {"val": 2854, "test": 2793}[a.split], len(frame)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ck_sha = sha(a.ckpt)
    assert ck["arch"] == "torchvision.models.efficientnet_v2_s" and list(ck["class_names"]) == list(meta["class_names"]) and ck["num_classes"] == 19
    assert ck["mapping_sha256"] == meta["mapping_sha256"] and ck["split_sha256"] == meta["manifest_sha256"]
    assert ck["arm"] == "B_masked_ce" and ck["seed"] == a.expect_seed, (ck["arm"], ck["seed"])
    model = efficientnet_v2_s(weights=None); model.classifier[1] = nn.Linear(1280, 19)
    model.load_state_dict(ck["global_state"], strict=True); model.to(DEV).eval()
    tf = T.Compose([T.Resize((256, 256), antialias=True), T.ToTensor(), T.Normalize(list(ck["mean"]), list(ck["std"]))])
    pool = ThreadPoolExecutor(max_workers=6)

    def load(p):
        with Image.open(p) as im:
            return tf(im.convert("RGB"))

    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    paths, md5s, srcs, feats, logits = [], [], [], [], []
    with torch.no_grad():
        for c in CLIENTS:
            f = frame[frame.source == c].reset_index(drop=True)
            files = [str(CACHE / f"{m}.jpg") for m in f.md5]
            for i in range(0, len(files), 32):
                x = torch.stack(list(pool.map(load, files[i:i + 32]))).to(DEV)
                with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                    h = torch.flatten(model.avgpool(model.features(x)), 1)
                    z = model.classifier(h)                       # == model(x): EfficientNet.forward = features -> avgpool -> flatten -> classifier
                feats.append(h.float().cpu()); logits.append(z.float().cpu())
            paths += f.image_path.tolist(); md5s += f.md5.tolist(); srcs += [c] * len(f)
            print(f"{a.split} {c}: {len(f)}", flush=True)
    F_ = torch.cat(feats); Z = torch.cat(logits)
    assert len(paths) == len(set(paths)) == F_.shape[0] == Z.shape[0] and F_.shape[1] == 1280 and Z.shape[1] == 19
    assert bool(torch.isfinite(F_).all() and torch.isfinite(Z).all())
    info = dict(split=a.split, n=len(paths), checkpoint=str(Path(a.ckpt).resolve()), checkpoint_sha256=ck_sha, arm=ck["arm"], seed=ck["seed"], round=ck["round"],
                procedure="Resize 256x256 on md5 384px cache, ckpt mean/std, eval, fp16 autocast, batch 32 per client in manifest order, avgpool; logits = ckpt classifier, same pass",
                label_columns_read=False, seconds=round(time.time() - t0, 1), peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3) if DEV == "cuda" else 0.0)
    torch.save(dict(paths=paths, md5=md5s, source=srcs, features_fp16=F_.half(), logits_fp32=Z, **info), out)
    out.with_suffix(".json").write_text(json.dumps(dict(info, file_sha256=sha(out)), indent=2))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
