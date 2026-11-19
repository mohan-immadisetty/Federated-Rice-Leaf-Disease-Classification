# E5 reproduction

This folder contains the code that produced the final **E5** head and the
artifacts needed to check the reported numbers. The raw image corpus
(21,731 files, ~3.3 GB) and the precomputed feature caches are **not** in the
repository.

## Pipeline

```
combined_manifest.csv (5 datasets, deduplicated split)
        │
        ├── FedAvg EfficientNetV2-S checkpoint (seed 42, masked local loss)
        │        └── features_<client>.pt   (train + val, 1280-d avgpool)
        │
        └── rice_v3_pack/features_dinov2-base.npy (1536-d, frozen DINOv2)
                 │
                 ▼
     train_e5_head.py  (FedSGD + server Adam, data policy v1)
                 │
                 ├── best_head.pth           (19 × 2816 linear head)
                 ├── config.json
                 ├── validation_curve.csv
                 └── best_validation_metrics.json
                          │
                          ▼
     extract_effnet.py + apply_e5.py  (val/test features, no fitting)
                          │
                          ▼
     final test evaluation (seed 42 primary, 43/44 robustness)
```

## What is included

| Path | Purpose |
|---|---|
| `train_e5_head.py` | Federated head training (FedSGD + server Adam) |
| `extract_fl_features.py` | EffNet feature caches per client |
| `extract_effnet.py` | Val/test EffNet features + FedAvg logits |
| `apply_e5.py` | Apply the frozen head to extracted features |
| `make_manifest_v1.py` | Training manifest v1 (cross-source label-conflict policy) |
| `verify_outputs.py` | Pure-numpy recomputation of the headline metrics |
| `configs/e5.json` | Exact E5 hyperparameters and metrics |
| `results/validation/` | E5 validation curves, metrics, predictions |
| `results/final_test/` | Final test protocol, metrics, predictions, labels |

## What you need to supply

The scripts default to the original layout under `D:\rice`. Override with
environment variables:

| Variable | Default |
|---|---|
| `RICE_E5_DATA_ROOT` | `D:\rice` |
| `RICE_E5_IMAGE_CACHE` | `<root>/cache/combined_rice_datasets/max384_q95` |
| `RICE_E5_PACK` | `<root>/rice_v3_pack` |
| `RICE_E5_FL_CACHE` | `<root>/outputs/federated_head_only_fedavg/run_20261021_120729` |
| `RICE_E5_OUTPUT` | `<root>/outputs/model_improvement_research` |
| `RICE_E5_FINAL_TEST` | `repro/results/final_test` |

Required inputs (not in this repo):

- `splits/combined_rice_datasets/{combined_manifest.csv,split_meta.json}`
- `rice_v3_pack/{manifest.csv,features_dinov2-base.npy}` (~133 MB)
- `outputs/federated_head_only_fedavg/run_20261021_120729/features_*.pt` (~44 MB)
- the 384-px image cache (`<md5>.jpg`) or the original images
- `outputs/.../B_masked_ce_seed42/best_global.pth` (included as `assets/best_global.pth`)

## Commands

Train the head (after the feature caches exist):

```bash
python repro/train_e5_head.py \
  --exp-id E5_concat_v1_lr1e-3 --features concat --train-manifest v1 \
  --lr 1e-3 --max-rounds 3000 --eval-every 10 --patience-rounds 300 \
  --weight-decay 0 --family-weight-power 0.5 --seed 42
```

Extract test features and apply the head:

```bash
python repro/extract_effnet.py --ckpt assets/best_global.pth --expect-seed 42 \
  --split test --out repro/results/final_test/features/test_seed42_effnet.pt

python repro/apply_e5.py \
  --features repro/results/final_test/features/test_seed42_effnet.pt \
  --head assets/best_head.pth \
  --expect-backbone-sha 3cba13a51f45e69bfc4d5934d526a9c7d91765cee01329fa13ed1eccf4d970d9 \
  --out repro/results/final_test/test_predictions_e5_seed42.csv
```

Recompute the headline metrics:

```bash
python repro/verify_outputs.py
```

The final-test evaluation is **not** a fresh independent test: the test split
was used earlier for the FedAvg baseline and head diagnostics. See
`results/final_test/README.md` and `docs/MODEL_CARD.md`.
