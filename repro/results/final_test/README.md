# E5 final test evaluation (2026-10-21)

**TEST results, not validation.** The protocol is fixed in `final_protocol.md` (sha `864e443e…`); the actual sequence of events is recorded in `execution_log.txt`. **This is not a fresh or fully independent test** (see Limitations).

## Result: primary model (E5, backbone seed 42), official test n = 2,793

| | Accuracy | Macro-F1 (19 classes) | Strict acc / Macro-F1 (n = 2,484) | Family-weighted acc / Macro-F1 |
|---|---|---|---|---|
| **E5** | **92.34** [90.21, 94.26] | **89.58** [86.52, 92.38] | 91.59 / 89.01 | 93.11 / 91.68 |
| FedAvg + masked loss (same backbone) | 85.82 [83.57, 87.94] | 67.39 [63.19, 70.54] | 84.46 / 66.65 | 85.46 / 67.73 |
| E5 − FedAvg | +6.52 [+5.40, +7.72] | +22.19 [+18.69, +26.42] | +7.13 / +22.36 | +7.66 / +23.95 |

The bracketed values are 95% intervals from a **paired duplicate-family bootstrap**:
- the sampling unit is `dup_group` (2,147 families);
- 2,000 resamples with `default_rng(42)`;
- percentile intervals.

They cover sampling uncertainty over test families only.

## Robustness: other FedAvg backbone seeds (same frozen rules)

| Backbone seed | E5 acc / Macro-F1 | FedAvg acc / Macro-F1 | E5 − FedAvg (acc / Macro-F1) | E5 strict acc / Macro-F1 |
|---|---|---|---|---|
| 43 | 93.45 [91.45, 95.14] / 89.81 [86.67, 93.48] | 85.96 / 68.47 | +7.48 [5.96, 9.13] / +21.34 [18.00, 25.86] | 92.79 / 89.38 |
| 44 | 92.52 [89.96, 94.62] / 87.54 [84.08, 91.23] | 84.78 / 65.64 | +7.73 [6.41, 9.16] / +21.90 [18.22, 26.60] | 91.79 / 87.19 |
| **Mean ± sample SD (42/43/44)** | 92.77 ± 0.60 / 88.97 ± 1.25 | 85.52 ± 0.64 / 67.17 ± 1.43 | +7.24 ± 0.64 / +21.81 ± 0.43 | 92.06 ± 0.65 / 88.53 ± 1.17 |

- **Seed 42 stays primary** by the pre-registered rule; seeds 43 and 44 are robustness results, and no seed was chosen from test outcomes.
- **Three seeds support no significance claim.**
- **E5 has no zero-recall class for any seed.** FedAvg has one: ragged_stunt for seed 42, leaf_scald for seed 44.

## Checks

- **Integrity:** predictions and test features were hashed in `execution_log.txt` before any test label was read. Those hashes were re-verified before evaluation.
- **Parity:** the extraction and E5 code exactly reproduce the seed-42 validation cache, E5's validation predictions and the notebook's FedAvg validation metrics (`parity/`).
- **FedAvg baseline protocol:** seed-42 FedAvg test predictions agree with the notebook's earlier test output on 2,791/2,793 images (99.93%; the threshold was 99.5%). The labels are identical.
- **Independent recomputation:** `scripts/verify_outputs.py` recomputes every headline metric in pure numpy from the saved, label-free prediction files and the restricted labels. All match.

## Limitations

1. **Earlier test exposure.** The test split was used before:
   - for FedAvg seed 42 (2026-10-14);
   - for the head diagnostics, plus a repeat of the seed-42 checkpoint (2026-10-21, 11:47).

   E5 was designed on validation only, but these numbers are not a fresh or independent test.
2. **Session-level near-duplicates.** 309 official test images have a training neighbour at DINOv2 cosine ≥ 0.985. The strict subset removes them: E5 seed 42 falls 0.75 points in accuracy and 0.57 points in Macro-F1.
3. **Labels unreviewed.** Cross-source label conflicts exist in training, and the planned pathologist review was not performed (`label_review_decision.md`). No test image was flagged, and no label was changed.
4. **Rare classes.** Several classes have only 11–15 test images (sheath_rot 11; bakanae, grassy_stunt, ragged_stunt and stem_rot 15 each; false_smut 14). Their F1 is unstable.
   - Weakest E5 seed-42 classes: ragged_stunt 0.70, leaf_scald 0.71, sheath_rot 0.78, narrow_brown_spot 0.78.
   - The bootstrap intervals do not capture label noise, near-duplicate leakage or test reuse.
5. **Per-client accuracy (E5 seed 42).** bd_rice8 is the weakest client:

   | Client | Accuracy |
   |---|---|
   | bd_rice8 | 79.0 |
   | paddy_doctor | 95.9 |
   | philippines | 92.4 |
   | riceleafbd | 89.3 |
   | sethy | 90.2 |

## Files

| Path | Contents |
|---|---|
| `test_predictions_{e5,fedavg}_seed{42,43,44}.csv` | Label-free predictions |
| `restricted_labels/test_labels.csv` | Test labels (restricted) |
| `test_metrics_by_seed.json` | All metrics |
| `per_class_metrics.csv` | Per-class results |
| `per_client_metrics.csv` | Per-client results |
| `confusion_matrices.json` | Confusion matrices |
| `bootstrap_intervals.json` | Bootstrap intervals |
| `final_comparison.csv` | Comparison table |
| `fedavg_consistency_check.json` | FedAvg vs notebook check |
| `preflight/` | Preflight checks (51 gates) |
| `parity/` | Validation parity check |
| `features/` | Test features (EffNet + FedAvg logits) |
| `predictions/` | Frozen raw predictions |
| `scripts/` | All code |
| `logs/` | Run logs |
| `artifact_manifest.json` | Hashes of outputs and inputs |
| `execution_log.txt` | Append-only action log |
