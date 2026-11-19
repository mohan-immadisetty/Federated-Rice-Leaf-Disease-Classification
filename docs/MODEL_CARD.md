# Model card — E5 rice-leaf disease classifier

## Overview

| Field | Value |
|---|---|
| Model | E5: frozen EfficientNetV2-S (FedAvg) + frozen DINOv2-base → linear head |
| Task | 19-class rice-leaf disease / condition classification |
| Input | RGB leaf image (any size; internal 384-px cache + 224-px DINO view) |
| Output | softmax over 19 classes + FedAvg baseline prediction |
| Parameters trained | 19 × 2816 + 19 = 53,523 (linear head only) |
| License | code MIT; weights inherit source dataset terms |

## Classes

`bacterial_leaf_blight`, `bacterial_leaf_streak`, `bacterial_panicle_blight`,
`bakanae`, `blast`, `brown_spot`, `dead_heart`, `downy_mildew`, `false_smut`,
`grassy_stunt_virus`, `healthy`, `hispa`, `leaf_scald`, `narrow_brown_spot`,
`ragged_stunt_virus`, `sheath_blight`, `sheath_rot`, `stem_rot`, `tungro`.

## Training data

Five public datasets, pooled and deduplicated **before** training:

| Client | Train images |
|---|---:|
| bd_rice8 | 1,282 |
| paddy_doctor | 7,222 |
| philippines | 1,331 |
| riceleafbd | 997 |
| sethy | 3,230 |
| **total** | **14,062** |

- 21,731 files → 1,428 excluded duplicates/label conflicts.
- Split: StratifiedGroupKFold by duplicate family, seed 42, 70/15/15.
- Official test: 2,793 images (3,047 test − 254 duplicate re-audit exclusions).
- Training manifest **v1** removes cross-source label-conflicting near-copies
  (philippines `sheath_rot` ↔ sethy `bacterial_leaf_blight`, DINOv2 cosine ≥ 0.90).

## Training procedure

1. FedAvg EfficientNetV2-S, 20 rounds, masked local CE, seed 42 (frozen afterwards).
2. Extract 1280-d avgpool features per client; DINOv2-base features are precomputed.
3. Federated standardisation: clients upload per-feature sum, sum of squares, count.
4. Head: FedSGD + server Adam, lr 1e-3, weight decay 0, family weight power 0.5,
   full participation, one full-batch gradient step per round, early stopping on
   validation macro-F1 (best round 80 of 380).

## Evaluation

Official held-out test, n = 2,793; 95% intervals from a paired duplicate-family
bootstrap (2,147 families, 2,000 resamples, seed 42).

| Model | Accuracy | Macro-F1 |
|---|---|---|
| E5 | **92.34%** [90.21, 94.26] | **89.58%** [86.52, 92.38] |
| FedAvg baseline | 85.82% [83.57, 87.94] | 67.39% [63.19, 70.54] |
| Δ | +6.52 [+5.40, +7.72] | +22.19 [+18.69, +26.42] |

Strict subset (n = 2,484; drops test images with a training neighbour at
DINOv2 cosine ≥ 0.985): E5 91.59% / 89.01.

Per-client accuracy (E5, seed 42):

| Client | Accuracy |
|---|---:|
| bd_rice8 | 79.0 |
| paddy_doctor | 95.9 |
| philippines | 92.4 |
| riceleafbd | 89.3 |
| sethy | 90.2 |

Weakest classes (seed 42): ragged_stunt 0.70, leaf_scald 0.71,
sheath_rot 0.78, narrow_brown_spot 0.78 F1.

## Intended use

- Educational demo and research reproduction.
- Screening aid for common rice-leaf conditions in a controlled setting.
- Comparing federated feature fusion against a FedAvg baseline.

## Out-of-scope use

- Agronomic decisions, pesticide application or crop-loss estimates.
- Images from unseen diseases, other crops, or non-leaf plant parts.
- Low-quality, heavily edited, synthetic or microscopy images.
- Any safety-critical or regulated diagnosis.

## Limitations

1. **Test reuse.** The test split was used earlier for the FedAvg baseline and
   head diagnostics; E5 was selected on validation only, but these numbers are
   not a fresh independent test.
2. **Near-duplicate leakage.** 309 test images have a training neighbour at
   DINOv2 cosine ≥ 0.985. The strict subset removes them and costs E5 0.75
   accuracy points.
3. **Label noise.** Cross-source conflicts exist; the planned pathologist review
   was not performed. No test label was changed.
4. **Rare classes.** Several classes have 11–15 test images; per-class F1 is
   unstable and bootstrap intervals do not include label noise.
5. **Domain shift.** Performance on phone photos from new regions, lighting,
   backgrounds or camera pipelines is unknown.
6. **Single crop stage.** The model sees whatever stage the datasets contain;
   it does not model disease progression.
7. **No abstention.** Softmax confidence is not calibrated; the model always
   returns a class.

## Ethical / privacy notes

- Only leaf images; no personally identifiable data.
- Public datasets may carry their own licenses and attribution requirements;
  check before redistributing images or weights.
- The demo stores nothing and logs no uploads.

## Reproducibility

- Exact config: [`repro/configs/e5.json`](../repro/configs/e5.json)
- Training code: [`repro/train_e5_head.py`](../repro/train_e5_head.py)
- Frozen test protocol + predictions: [`repro/results/final_test/`](../repro/results/final_test/)
- Checkpoints: `assets/best_global.pth`, `assets/best_head.pth`

## Contact

Mohan Venkata NaImmadisetty — B.Tech mini project, 2026.
