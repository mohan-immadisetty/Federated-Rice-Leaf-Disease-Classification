"""Training-manifest candidate v1: exclude cross-source label-conflicting near-copies (philippines sheath_rot <-> sethy BLB).

Rule (fixed before any model comparison; TRAINING images only; val/test untouched):
  a philippines `sheath_rot` training image whose nearest sethy `bacterial_leaf_blight` training image has DINOv2-base cosine
  >= 0.90 is a cross-source near-copy carrying a conflicting label. Following the split builder's existing policy for
  label-conflicting duplicate families ("label_conflict_near_duplicate": exclude every member), both the flagged
  sheath_rot image and every sethy BLB training image with cosine >= 0.90 to a flagged image are excluded from training.
Threshold evidence (stage1 audit): 0 of 417 images of the other philippines r__ classes reach 0.90 to any sethy image; the
flagged sheath_rot similarities are 0.908-0.992 and the next unflagged one is 0.751 (clear gap); 3 pairs inspected visually
are the same photographs.
No label is changed. Originals are not modified.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("RICE_E5_DATA_ROOT", r"D:\rice"))
SPLIT_DIR = ROOT / "splits" / "combined_rice_datasets"
PACK = Path(os.environ.get("RICE_E5_PACK", ROOT / "rice_v3_pack"))
OUT = Path(os.environ.get("RICE_E5_MANIFEST_OUT", ROOT / "outputs" / "model_improvement_research" / "data_manifests"))
THR = 0.90

man = pd.read_csv(SPLIT_DIR / "combined_manifest.csv", keep_default_na=False)
tr = man[man.split == "train"].reset_index(drop=True)
pm = pd.read_csv(PACK / "manifest.csv")
X = np.load(PACK / "features_dinov2-base.npy", mmap_mode="r")
pos = dict(zip(pm.md5, pm.idx))
E = F.normalize(torch.tensor(np.asarray(X[[pos[m] for m in tr.md5]], dtype=np.float32)), dim=1)
sr = np.flatnonzero(((tr.source == "philippines") & (tr.canonical_class == "sheath_rot")).to_numpy())
sb = np.flatnonzero(((tr.source == "sethy") & (tr.canonical_class == "bacterial_leaf_blight")).to_numpy())
S = (E[sr] @ E[sb].T).numpy()
flag_sr = sr[S.max(1) >= THR]
flag_sb = sb[(S[S.max(1) >= THR] >= THR).any(0)]
rows = []
for i in flag_sr:
    j = sb[S[list(sr).index(i)].argmax()]
    rows.append(dict(image_path=tr.image_path[i], source="philippines", original_class=tr.original_class[i], canonical_class="sheath_rot",
                     reason="cross-source near-copy with conflicting label", partner=tr.image_path[j],
                     partner_label="sethy/Bacterialblight -> bacterial_leaf_blight", dinov2_cos=float(S[list(sr).index(i)].max())))
for j in flag_sb:
    k = list(sb).index(j)
    i = sr[S[:, k].argmax()]
    rows.append(dict(image_path=tr.image_path[j], source="sethy", original_class=tr.original_class[j], canonical_class="bacterial_leaf_blight",
                     reason="cross-source near-copy with conflicting label", partner=tr.image_path[i],
                     partner_label="philippines/sheath_rot -> sheath_rot", dinov2_cos=float(S[:, k].max())))
ex = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
ex.to_csv(OUT / "train_v1_exclusions.csv", index=False)
v1 = tr[~tr.image_path.isin(set(ex.image_path))]
v1.to_csv(OUT / "train_manifest_v1.csv", index=False)
tr.to_csv(OUT / "train_manifest_v0_original.csv", index=False)
info = dict(version="v1", base="combined_manifest.csv split=='train' (v0)", rule=__doc__, threshold=THR,
            n_v0=len(tr), n_v1=len(v1), excluded=ex.groupby(["source", "canonical_class"]).size().rename("n").reset_index().to_dict("records"),
            sha256_v0=hashlib.sha256((OUT / "train_manifest_v0_original.csv").read_bytes()).hexdigest(),
            sha256_v1=hashlib.sha256((OUT / "train_manifest_v1.csv").read_bytes()).hexdigest(),
            val_and_test_changed=False, labels_changed=False)
(OUT / "train_manifest_v1_info.json").write_text(json.dumps(info, indent=2))
print(json.dumps({k: v for k, v in info.items() if k != "rule"}, indent=2))
