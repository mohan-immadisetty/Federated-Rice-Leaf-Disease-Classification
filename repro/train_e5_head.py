"""Federated linear-head training on frozen features with FedSGD + server Adam (FedAdam, one local gradient step).

Protocol per round (all 5 clients participate):
  client k: g_k = d/dtheta sum_{i in D_k} omega_i * CE(theta; x_i, y_i)  (full local batch, unmasked 19-way CE),
            s_k = sum_{i in D_k} omega_i ;  uploads (g_k, s_k) only.
  server:   g = sum_k g_k / sum_k s_k  (= exact gradient of the pooled weighted objective), Adam step on theta.
Feature sets: "fl" = avgpool features of the round-20 FedAvg-masked EfficientNetV2-S (head initialised from its global
head, raw features); "dino" = frozen DINOv2-base (1536-d, rice_v3_pack), "concat" = [fl, dino]; for dino/concat the
features are standardised with a FEDERATED mean/std (clients upload per-feature sums, sums of squares, counts) and the
head starts at zero. Model selection: validation macro-F1 (19 classes), evaluated every `eval_every` rounds.
Never loads test images or labels.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(os.environ.get("RICE_E5_DATA_ROOT", r"D:\rice"))
SPLIT_DIR = ROOT / "splits" / "combined_rice_datasets"
EXP = ROOT / "outputs" / "fedavg_masked_local_loss_efficientnetv2_s" / "experiments" / "20261014_104304"
BASE_CKPT = EXP / "B_masked_ce_seed42" / "best_global.pth"
FEAT_RUN = Path(os.environ.get("RICE_E5_FL_CACHE", ROOT / "outputs" / "federated_head_only_fedavg" / "run_20261021_120729"))
PACK = Path(os.environ.get("RICE_E5_PACK", ROOT / "rice_v3_pack"))
WS = Path(os.environ.get("RICE_E5_OUTPUT", ROOT / "outputs" / "model_improvement_research"))
CLIENTS = ["bd_rice8", "paddy_doctor", "philippines", "riceleafbd", "sethy"]
STRICT_COS = 0.985


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--features", choices=["fl", "dino", "concat"], default="fl")
    ap.add_argument("--train-manifest", choices=["v0", "v1"], default="v0")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-rounds", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--patience-rounds", type=int, default=300)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--family-weight-power", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--note", default="")
    ap.add_argument("--fl-cache-dir", default=str(FEAT_RUN), help="folder with features_<client>.pt of the FL backbone")
    ap.add_argument("--base-ckpt", default=str(BASE_CKPT), help="FL backbone checkpoint the caches were built from")
    a = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(a.seed)
    out = WS / "experiments" / a.exp_id
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    meta = json.loads((SPLIT_DIR / "split_meta.json").read_text())
    exp_cfg = json.loads((EXP / "experiment_config.json").read_text())
    CLASSES = list(meta["class_names"]); C = len(CLASSES); IDX = {c: i for i, c in enumerate(CLASSES)}
    assert sha_file(SPLIT_DIR / "combined_manifest.csv") == meta["manifest_sha256"]
    man = pd.read_csv(SPLIT_DIR / "combined_manifest.csv", keep_default_na=False)
    kept = man[man.split != "excluded"]
    test_paths = set(kept[kept.split == "test"].image_path)
    leak = set(exp_cfg["split_fingerprint"]["eval_images_excluded_by_reaudit"])
    mdir = WS / "data_manifests"
    trm = pd.read_csv(mdir / ("train_manifest_v0_original.csv" if a.train_manifest == "v0" else "train_manifest_v1.csv"), keep_default_na=False)
    assert set(trm.split) == {"train"} and not set(trm.image_path) & test_paths and set(trm.image_path) <= set(kept[kept.split == "train"].image_path)
    va = kept[(kept.split == "val") & ~kept.image_path.isin(leak)]
    assert len(va) == 2854 and not set(va.image_path) & set(trm.image_path)

    # ---------------- features
    fl = {c: torch.load(Path(a.fl_cache_dir) / f"features_{c}.pt", weights_only=False) for c in CLIENTS}
    base_ckpt = Path(a.base_ckpt)
    assert all(fl[c]["base_checkpoint_sha256"] == sha_file(base_ckpt) for c in CLIENTS), "FL caches were not built from --base-ckpt"
    fl_map = {}
    for c in CLIENTS:
        for s in ("train", "val"):
            for p, v in zip(fl[c][f"{s}_paths"], fl[c][s]):
                fl_map[p] = v
    pm = pd.read_csv(PACK / "manifest.csv")
    DX = np.load(PACK / "features_dinov2-base.npy", mmap_mode="r")
    dpos = dict(zip(pm.md5, pm.idx))

    def feats(frame):
        parts = []
        if a.features in ("fl", "concat"):
            parts.append(torch.stack([fl_map[p] for p in frame.image_path]).float())
        if a.features in ("dino", "concat"):
            parts.append(torch.tensor(np.asarray(DX[[dpos[m] for m in frame.md5]], dtype=np.float32)))
        return torch.cat(parts, 1)

    data = {}
    for c in CLIENTS:
        f = trm[trm.source == c].reset_index(drop=True)
        w = f.groupby("dup_group").image_path.transform("size").to_numpy(float) ** (-a.family_weight_power)
        data[c] = dict(X=feats(f), y=torch.tensor(f.canonical_class.map(IDX).to_numpy()), w=torch.tensor(w / w.mean(), dtype=torch.float32), n=len(f))
    va = va.reset_index(drop=True)
    Xva, yva, cva = feats(va), va.canonical_class.map(IDX).to_numpy(), va.source.to_numpy()
    D = Xva.shape[1]

    # strict validation subset (label-free): drop val images with a DINOv2 cosine >= STRICT_COS to any v0 training image
    tr0 = kept[kept.split == "train"]
    Etr = F.normalize(torch.tensor(np.asarray(DX[[dpos[m] for m in tr0.md5]], dtype=np.float32)), dim=1)
    Eva = F.normalize(torch.tensor(np.asarray(DX[[dpos[m] for m in va.md5]], dtype=np.float32)), dim=1)
    strict = (Eva @ Etr.T).max(1).values.numpy() < STRICT_COS

    # ---------------- federated standardisation (dino / concat): clients upload sum, sum of squares, count
    if a.features == "fl":
        mu, sd = torch.zeros(D), torch.ones(D)
    else:
        S1 = sum(d["X"].double().sum(0) for d in data.values()); S2 = sum((d["X"].double() ** 2).sum(0) for d in data.values())
        N = sum(d["n"] for d in data.values())
        mu = (S1 / N).float(); sd = ((S2 / N - (S1 / N) ** 2).clamp_min(1e-12).sqrt()).float()
    for d in data.values():
        d["X"] = (d["X"] - mu) / sd
    Xva = (Xva - mu) / sd

    ck = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    if a.features == "fl":
        W = ck["global_state"]["classifier.1.weight"].clone().float(); B = ck["global_state"]["classifier.1.bias"].clone().float()
    else:
        W, B = torch.zeros(C, D), torch.zeros(C)
    W.requires_grad_(True); B.requires_grad_(True)
    opt = torch.optim.Adam([W, B], lr=a.lr, weight_decay=a.weight_decay)     # server optimiser

    def evaluate(W_, B_, mask=None):
        with torch.no_grad():
            p = (Xva @ W_.T + B_).argmax(1).numpy()
        sel = np.ones(len(p), bool) if mask is None else mask
        y_, p_, c_ = yva[sel], p[sel], cva[sel]
        pr, rc, f1, sup = precision_recall_fscore_support(y_, p_, labels=list(range(C)), zero_division=0)
        return p, dict(n=int(sel.sum()), accuracy=float(accuracy_score(y_, p_)), macro_f1=float(f1.mean()),
                       weighted_f1=float(f1_score(y_, p_, average="weighted", labels=list(range(C)), zero_division=0)),
                       macro_precision=float(pr.mean()), macro_recall=float(rc.mean()),
                       zero_recall=[CLASSES[k] for k in range(C) if sup[k] > 0 and rc[k] == 0],
                       per_client={c: float(accuracy_score(y_[c_ == c], p_[c_ == c])) for c in CLIENTS},
                       per_class={CLASSES[k]: dict(precision=float(pr[k]), recall=float(rc[k]), f1=float(f1[k]), support=int(sup[k])) for k in range(C)})

    curve, best, since = [], dict(macro_f1=-1), 0
    up_floats = C * D + C + 1
    for rnd in range(0, a.max_rounds + 1):
        if rnd > 0:
            grads_W, grads_B, wsum = [], [], 0.0
            for c in CLIENTS:                                      # local computation; only (gW, gB, s) leave the client
                d = data[c]
                gW, gB = torch.zeros_like(W), torch.zeros_like(B)
                Wc, Bc = W.detach().clone().requires_grad_(True), B.detach().clone().requires_grad_(True)
                loss = (F.cross_entropy(d["X"] @ Wc.T + Bc, d["y"], reduction="none") * d["w"]).sum()
                loss.backward()
                grads_W.append(Wc.grad); grads_B.append(Bc.grad); wsum += float(d["w"].sum())
            opt.zero_grad(set_to_none=True)
            W.grad = sum(grads_W) / wsum; B.grad = sum(grads_B) / wsum
            if not (torch.isfinite(W.grad).all() and torch.isfinite(B.grad).all()):
                raise RuntimeError(f"non-finite gradient at round {rnd}")
            opt.step()
        if rnd % a.eval_every == 0:
            p, m = evaluate(W.detach(), B.detach())
            _, ms = evaluate(W.detach(), B.detach(), strict)
            curve.append(dict(round=rnd, val_accuracy=m["accuracy"], val_macro_f1=m["macro_f1"], val_weighted_f1=m["weighted_f1"],
                              n_zero_recall=len(m["zero_recall"]), strict_accuracy=ms["accuracy"], strict_macro_f1=ms["macro_f1"],
                              **{f"acc_{c}": v for c, v in m["per_client"].items()}))
            if rnd > 0 and (m["macro_f1"] > best["macro_f1"] + 1e-9):
                best = dict(round=rnd, macro_f1=m["macro_f1"], metrics=m, strict=ms, pred=p, W=W.detach().clone(), B=B.detach().clone())
                since = 0
            elif rnd > 0:
                since += a.eval_every
            if rnd % 100 == 0:
                print(f"[{a.exp_id}] round {rnd:4d} val acc {m['accuracy']:.4f} mF1 {m['macro_f1']:.4f} | strict acc {ms['accuracy']:.4f} "
                      f"mF1 {ms['macro_f1']:.4f} | best r{best.get('round')} mF1 {best['macro_f1']:.4f}", flush=True)
            if since >= a.patience_rounds:
                print(f"[{a.exp_id}] stop: no val macro-F1 gain for {a.patience_rounds} rounds", flush=True)
                break
    rounds_run = rnd
    pd.DataFrame(curve).to_csv(out / "validation_curve.csv", index=False)
    torch.save(dict(weight=best["W"], bias=best["B"], feature_set=a.features, feat_mean=mu, feat_std=sd, round=best["round"],
                    train_manifest=a.train_manifest, class_names=CLASSES, base_checkpoint=str(base_ckpt),
                    base_checkpoint_sha256=sha_file(base_ckpt), note="linear head for frozen features; apply (x-mean)/std first"),
               out / "best_head.pth")
    pd.DataFrame(dict(image_path=va.image_path, client=cva, y=yva, pred=best["pred"], strict_subset=strict)).to_csv(out / "validation_predictions.csv", index=False)
    m = best["metrics"]
    cfg = dict(exp_id=a.exp_id, created=datetime.now().isoformat(timespec="seconds"), args=vars(a), algorithm="FedSGD + server Adam (FedAdam, 1 local full-batch gradient step, full participation)",
               feature_dim=D, train_manifest_sha=sha_file(mdir / ("train_manifest_v0_original.csv" if a.train_manifest == "v0" else "train_manifest_v1.csv")),
               n_train={c: d["n"] for c, d in data.items()}, rounds_run=rounds_run, best_round=best["round"],
               upload_per_client_per_round_kb=round(4 * up_floats / 1024, 1),
               total_upload_mb_to_best=round(5 * best["round"] * 4 * up_floats / 2**20, 1),
               strict_subset=dict(rule=f"val images with DINOv2 cosine < {STRICT_COS} to every v0 training image", n=int(strict.sum())),
               runtime_s=round(time.time() - t0, 1), test_used=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "best_validation_metrics.json").write_text(json.dumps(dict(official=m, strict=best["strict"]), indent=2))
    cm = confusion_matrix(yva, best["pred"], labels=list(range(C)))
    (out / "confusion_matrix.json").write_text(json.dumps(dict(class_names=CLASSES, matrix=cm.tolist())))
    reg = WS / "experiment_registry.csv"
    row = dict(exp_id=a.exp_id, status="completed", method="federated linear head, FedSGD+server Adam", features=a.features,
               train_manifest=a.train_manifest, lr=a.lr, weight_decay=a.weight_decay, seed=a.seed, best_round=best["round"], rounds_run=rounds_run,
               val_accuracy=m["accuracy"], val_macro_f1=m["macro_f1"], val_weighted_f1=m["weighted_f1"], val_macro_recall=m["macro_recall"],
               n_zero_recall=len(m["zero_recall"]), mean_client_acc=float(np.mean(list(m["per_client"].values()))),
               min_client_acc=float(min(m["per_client"].values())),
               strict_accuracy=best["strict"]["accuracy"], strict_macro_f1=best["strict"]["macro_f1"],
               upload_mb_to_best=cfg["total_upload_mb_to_best"], runtime_s=cfg["runtime_s"], checkpoint=str(out / "best_head.pth"), note=a.note)
    pd.concat([pd.read_csv(reg), pd.DataFrame([row])], ignore_index=True).to_csv(reg, index=False) if reg.exists() else pd.DataFrame([row]).to_csv(reg, index=False)
    print(f"[{a.exp_id}] BEST round {best['round']}: val acc {m['accuracy']:.4f} mF1 {m['macro_f1']:.4f} wF1 {m['weighted_f1']:.4f} "
          f"zeroRec {len(m['zero_recall'])} | strict (n={int(strict.sum())}) acc {best['strict']['accuracy']:.4f} mF1 {best['strict']['macro_f1']:.4f} | "
          f"clients {json.dumps({k: round(v, 3) for k, v in m['per_client'].items()})} | {cfg['runtime_s']}s")


if __name__ == "__main__":
    main()
