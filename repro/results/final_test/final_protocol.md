# E5 final test evaluation: frozen protocol

**Frozen 2026-10-21, before any test label, test prediction or test metric was computed in this study.** The file's SHA-256 is recorded in `execution_log.txt` and `artifact_manifest.json` before Stage D starts. Nothing below may change after that point. Any deviation is logged as a deviation, never applied silently.

## 1. Models (all frozen; nothing is trained, fine-tuned or refitted)

| Role | E5 head (validation-selected) | EfficientNetV2-S backbone (FedAvg + masked local loss) |
|---|---|---|
| **Primary E5 model** | Backbone seed 42: `experiments/E5_concat_v1_lr1e-3/best_head.pth` (sha `557380ec…`, round 80) | `fedavg_masked_local_loss_efficientnetv2_s/experiments/20261014_104304/B_masked_ce_seed42/best_global.pth` (sha `3cba13a5…`, round 20) |
| Robustness | Backbone seed 43: `experiments/E5_backbone_seed43_44_20261021/experiments/E5_bb_seed43/best_head.pth` (round 250) | `…/20261021_132145/B_masked_ce_seed43/best_global.pth` (sha `f5660938…`, round 19) |
| Robustness | Backbone seed 44: `…/E5_bb_seed44/best_head.pth` (round 640) | `…/20261021_132145/B_masked_ce_seed44/best_global.pth` (sha `4e6ee5d1…`, round 19) |

- **Seed 42 is the primary result.** It is the original, validation-selected E5 configuration (`final_recommendation.md`). Seeds 43 and 44 are backbone-robustness results.
- **Test results cannot change which seed is primary**, and no seed is selected or recommended for deployment based on test outcomes.
- **Second feature branch:** DINOv2-base. Precomputed features in `rice_v3_pack/features_dinov2-base.npy` (sha `618231fc…`), indexed by md5 via `rice_v3_pack/manifest.csv`. They were extracted once for all 21,731 images with identical preprocessing.
- **E5 inference (frozen):**
  1. x = [EffNet avgpool features (1,280), stored as fp16 exactly like the train/val caches; DINOv2 (1,536)].
  2. z = (x − feat_mean) / feat_std, using the training-derived federated statistics stored in each head file. They are never recomputed from test data.
  3. logits = z·Wᵀ + b.
  4. Prediction = unmasked argmax over 19 classes.
- **EfficientNet test features** are extracted fresh for each seed with that seed's frozen checkpoint and the frozen procedure of `tools/research_extract_fl_features.py`: Resize 256×256 on the 384-px md5 cache, the checkpoint's ImageNet mean/std, eval mode, fp16 autocast, batch 32, avgpool. The extraction code reads image paths only and never reads label columns.
- **FedAvg baseline:** the matching backbone's own full-model output, using the classifier stored in the same checkpoint and computed in the same forward pass (`classifier(avgpool(features(x)))`). This is the notebook's established protocol (`predict`/`evaluate_global`): the same eval transform and an unmasked 19-way argmax. The only difference is the evaluation batch size, 32 here versus 8 in the notebook.

## 2. Test set (fixed before evaluation)

- **Split:** `split == "test"` in `splits/combined_rice_datasets/combined_manifest.csv` (sha `65bf37f2…`).
- **Exclusions:** minus the pre-registered re-audit list `split_fingerprint.eval_images_excluded_by_reaudit` of experiment 20261014_104304. That list holds 444 val+test images, 254 of them test, from a dihedral dHash ≤ 4/64 and thumbnail MAE ≤ 3 check. The list must be identical in 20261021_132145.
- **Expected count:** 2,793 images. Stage C must confirm this from the manifest and list. It is never forced.
- **Labels:** `canonical_class` mapped by `split_meta.json` `class_names` (19 classes, mapping hash `da892aa2…`). No label is changed.
- **Labels are read only at evaluation time**, after every model is frozen, and are stored separately in `restricted_labels/`.

## 3. Metrics (identical rules for every model and seed)

1. **Official test:** all 2,793 images.
   - Accuracy.
   - Macro-F1 as the unweighted mean of per-class F1 over **all 19 classes**, the E5 convention: `labels=range(19)`, `zero_division=0`. If a class has no test support, the FedAvg-notebook convention (classes present in y) is also reported, and the difference is flagged.
   - Weighted-F1; per-class precision, recall, F1 and support; 19×19 confusion matrix.
2. **Strict subset** (label-free, fixed rule, mirroring the validation rule in `research_federated_head.py:107-111`):
   - Keep official test images whose DINOv2-base cosine to **every v0 training image** (14,212) is < 0.985.
   - Expected exclusions: 309, consistent with `stage1_audit/embedding_near_duplicates_test_vs_train.csv`. This is checked, not forced.
   - Report n, accuracy and Macro-F1 under the same 19-class convention.
3. **Per-client:** accuracy and n for each `source` client; macro-F1 over that client's present classes, as in `research_compile.py`.
4. **Family-weighted diagnostic:** each image is weighted by 1 / (number of official-test images in its `dup_group`), so each duplicate family counts once. Report family-weighted accuracy and macro-F1 (weighted confusion, 19 classes). This is a diagnostic only.
5. **Uncertainty:** paired **duplicate-family bootstrap**, the project's predefined method (`tools/research_compile.py`).
   - Sampling unit: `dup_group` among official test images.
   - 2,000 resamples, `numpy.random.default_rng(42)`, percentile 95% intervals (2.5 and 97.5).
   - Macro-F1 is computed from the weighted 19-class confusion matrix (`wmf1`).
   - Reported per seed for E5 accuracy and macro-F1, FedAvg accuracy and macro-F1, and the paired difference E5 − FedAvg. The same resample draws are used within each pair.
   - These intervals express sampling uncertainty over test families only. They do **not** correct for earlier test-set reuse, session-level near-duplicates with training, or label noise.
6. **Across seeds (supplementary):** mean and **sample** SD (ddof = 1, n = 3) of each metric. With three seeds no significance claim is made.

## 4. Mandatory verification gates (stop on failure)

- **Hashes:** every checkpoint, head, DINOv2 file, manifest and leak-list hash matches.
- **Head integrity:** each head's `base_checkpoint_sha256` equals its backbone; shape (19, 2816); class order equals `split_meta`.
- **Split integrity:** no md5 or duplicate family is shared between train, val and test; DINOv2 covers every test md5.
- **Extraction-code parity, before any test extraction:** the new extraction/evaluation code, run on seed-42 **validation** images, must reproduce:
  - the existing seed-42 val feature cache (max |diff| ≤ 1e-2, min cos ≥ 0.9999, the project tolerance);
  - E5's saved `validation_predictions.csv` exactly;
  - the notebook's seed-42 FedAvg validation accuracy and macro-F1 (round 20: 0.851437 / 0.680597) to within 0.002.
- **Test features:** row count 2,793, unique IDs, exact ID-set coverage, dims 1,280/1,536, cache backbone sha equals checkpoint sha, no label columns read.
- **FedAvg protocol parity:** seed-42 FedAvg test predictions must agree with the existing notebook output `20261014_104304/B_masked_ce_seed42/test_predictions.csv` on ≥ 99.5% of images. This is a repeat access to an already-computed result with the same weights. If it fails, the matched baseline comparison is not reported.

## 5. Disclosures (reported with every result)

- **The test split is not untouched.** It was evaluated before this study for:
  - FedAvg seed 42 (both arms, 2026-10-14);
  - the centralized and federated head diagnostics plus a repeat of the seed-42 checkpoint (`head_only_test_evaluation`, 2026-10-21 11:47).

  E5's design used validation only, but earlier test exposure in the project means these numbers are not a fresh, independent test.
- **Session-level near-duplicates:** at DINOv2 cosine ≥ 0.985, 309 official test images have a training neighbour, mostly paddy_doctor burst shots. The strict subset is reported for this reason.
- **Label quality:** cross-source label conflicts exist, for example philippines sheath_rot vs sethy BLB in training. sethy BLB → ragged_stunt confusion comes from a few heavily augmented leaves. The planned manual review by a plant pathologist has **not** been performed; labels are unchanged (see `label_review_decision.md`).
- **Rare classes:** several classes have very few test images, so their per-class F1 is unstable.
