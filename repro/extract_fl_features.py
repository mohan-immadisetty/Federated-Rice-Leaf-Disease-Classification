"""Per-client frozen-backbone feature caches (train + filtered val; NEVER test) for an FL EfficientNetV2-S checkpoint.

Same procedure as tools/federated_head_only_fedavg.py (which built the seed-42 caches): manifest order per client,
Resize 256x256 on the md5 cache, ImageNet normalisation, eval mode, fp16 autocast, batch 32, avgpool output (1280-d).
Output format identical: features_<client>.pt with train/val tensors (fp16), paths, labels, base_checkpoint_sha256.

Usage: python tools/research_extract_fl_features.py --ckpt <best_global.pth> --out <dir> [--expect-arm B_masked_ce --expect-seed 43]
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.models import efficientnet_v2_s

ROOT = Path(os.environ.get("RICE_E5_DATA_ROOT", r"D:\rice"))
SPLIT_DIR = ROOT / "splits" / "combined_rice_datasets"
CACHE = Path(os.environ.get("RICE_E5_IMAGE_CACHE", ROOT / "cache" / "combined_rice_datasets" / "max384_q95"))
EXP = ROOT / "outputs" / "fedavg_masked_local_loss_efficientnetv2_s" / "experiments" / "20261014_104304"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CLIENTS = ["bd_rice8", "paddy_doctor", "philippines", "riceleafbd", "sethy"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--expect-arm", default="B_masked_ce"); ap.add_argument("--expect-seed", type=int, required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((SPLIT_DIR / "split_meta.json").read_text())
    exp_cfg = json.loads((EXP / "experiment_config.json").read_text())
    CLASSES = list(meta["class_names"]); IDX = {c: i for i, c in enumerate(CLASSES)}
    leak = set(exp_cfg["split_fingerprint"]["eval_images_excluded_by_reaudit"])
    man = pd.read_csv(SPLIT_DIR / "combined_manifest.csv", keep_default_na=False)
    assert sha(SPLIT_DIR / "combined_manifest.csv") == meta["manifest_sha256"]
    kept = man[man.split != "excluded"]
    tr_all = kept[kept.split == "train"].reset_index(drop=True)
    va_all = kept[(kept.split == "val") & ~kept.image_path.isin(leak)].reset_index(drop=True)
    test_paths = set(kept[kept.split == "test"].image_path)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ck_sha = sha(a.ckpt)
    assert ck["arch"] == "torchvision.models.efficientnet_v2_s" and list(ck["class_names"]) == CLASSES and ck["num_classes"] == 19
    assert ck["mapping_sha256"] == meta["mapping_sha256"] and ck["split_sha256"] == meta["manifest_sha256"]
    assert ck["arm"] == a.expect_arm and ck["seed"] == a.expect_seed, (ck["arm"], ck["seed"])
    model = efficientnet_v2_s(weights=None); model.classifier[1] = nn.Linear(1280, 19)
    model.load_state_dict(ck["global_state"], strict=True); model.to(DEV).eval()
    tf = T.Compose([T.Resize((256, 256), antialias=True), T.ToTensor(), T.Normalize(list(ck["mean"]), list(ck["std"]))])
    pool = ThreadPoolExecutor(max_workers=6)

    def load(p):
        with Image.open(p) as im:
            return tf(im.convert("RGB"))

    @torch.no_grad()
    def encode(paths):
        F_ = []
        for i in range(0, len(paths), 32):
            x = torch.stack(list(pool.map(load, paths[i:i + 32]))).to(DEV)
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                F_.append(torch.flatten(model.avgpool(model.features(x)), 1).float().cpu())
        return torch.cat(F_)

    info = {}
    for c in CLIENTS:
        d = {}
        for s, frame in (("train", tr_all), ("val", va_all)):
            f = frame[frame.source == c].reset_index(drop=True)
            assert not set(f.image_path) & test_paths
            d[s] = (f, encode([str(CACHE / f"{m}.jpg") for m in f.md5]))
        torch.save(dict(client=c, train=d["train"][1].half(), val=d["val"][1].half(),
                        train_paths=d["train"][0].image_path.tolist(), val_paths=d["val"][0].image_path.tolist(),
                        train_y=d["train"][0].canonical_class.map(IDX).tolist(), val_y=d["val"][0].canonical_class.map(IDX).tolist(),
                        base_checkpoint_sha256=ck_sha), out / f"features_{c}.pt")
        info[c] = dict(train=len(d["train"][0]), val=len(d["val"][0]))
        print(f"{c}: train {info[c]['train']}, val {info[c]['val']}", flush=True)
    (out / "extraction_info.json").write_text(json.dumps(dict(checkpoint=a.ckpt, checkpoint_sha256=ck_sha, arm=ck["arm"], seed=ck["seed"],
                                                               round=ck["round"], counts=info, test_images=0,
                                                               peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3) if DEV == "cuda" else 0.0), indent=2))


if __name__ == "__main__":
    main()
