"""Apply a frozen E5 head (and record the FedAvg full-model argmax) to an extracted features file. No fitting; no labels read.

The ops mirror tools/research_federated_head.py:
  X = cat([effnet_fp16.float(), dinov2_fp32]); z = (X - feat_mean) / feat_std (stored in the head, training-derived);
  pred = argmax(z @ W.T + b) over 19 classes.
Usage: apply_e5.py --features FILE.pt --head HEAD.pth --expect-backbone-sha SHA --out PRED.csv
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

R = Path(os.environ.get("RICE_E5_DATA_ROOT", r"D:\rice"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True); ap.add_argument("--head", required=True)
    ap.add_argument("--expect-backbone-sha", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    fe = torch.load(a.features, weights_only=False)
    hd = torch.load(a.head, map_location="cpu", weights_only=False)
    assert fe["checkpoint_sha256"] == hd["base_checkpoint_sha256"] == a.expect_backbone_sha, "feature cache / head / backbone mismatch"
    assert hd["feature_set"] == "concat" and tuple(hd["weight"].shape) == (19, 2816)
    pm = pd.read_csv(R / "rice_v3_pack" / "manifest.csv")
    DX = np.load(R / "rice_v3_pack" / "features_dinov2-base.npy", mmap_mode="r")
    dpos = dict(zip(pm.md5, pm.idx))                                  # same mapping as research_federated_head.py:88
    X = torch.cat([fe["features_fp16"].float(), torch.tensor(np.asarray(DX[[dpos[m] for m in fe["md5"]]], dtype=np.float32))], 1)
    assert X.shape == (len(fe["paths"]), 2816)
    Xs = (X - hd["feat_mean"]) / hd["feat_std"]
    with torch.no_grad():
        logits = Xs @ hd["weight"].T + hd["bias"]
    pd.DataFrame(dict(image_path=fe["paths"], md5=fe["md5"], client=fe["source"], e5_pred=logits.argmax(1).numpy(),
                      fedavg_pred=fe["logits_fp32"].argmax(1).numpy())).to_csv(out, index=False)
    print(f"wrote {out} ({len(fe['paths'])} rows) | head {sha(a.head)[:12]} | features {fe['split']} seed {fe['seed']}")


if __name__ == "__main__":
    main()
