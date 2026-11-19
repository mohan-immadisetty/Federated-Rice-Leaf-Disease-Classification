# Stage B: label-conflict review decision

**Decision: no labels changed, and the manual review is recorded as skipped.**

## Plan vs. what happened

- **Planned:** `final_recommendation.md` step 3 called for a manual review, by someone qualified to judge the symptoms, before any test evaluation. It covered `label_conflict_audit.csv` (140 medium and 322 low flags) and the sethy BLB → ragged_stunt families.
- **What happened:** no qualified reviewer decision exists in the project (searched for reviewer/adjudication records: none), and none can be invented.

## Why this does not change the test set

The review manifest `label_review_manifest.csv` maps each of the 541 flags to its split:

| Confidence | Split | n | Status |
|---|---|---:|---|
| high | excluded at build time | 79 | Already excluded from every split (unchanged) |
| medium | train | 140 | Pending manual review; training label unchanged |
| low | train | 322 | Pending manual review; training label unchanged |
| any | **test** | **0** | — |

- **No flagged image is in the test split**, so adjudication could not change test labels or test inclusion. The stop condition "label adjudication would change the test set" does not apply.
- **The frozen models stay as they are:** the E5 heads were trained on manifest v1, which already excludes the 150 confirmed sheath_rot/BLB cross-source near-copies. Changing training labels would require retraining, which is out of scope.
- **The sethy BLB → ragged_stunt issue** is a suspected label or visual ambiguity, not a flag in the audit file. It remains a disclosed limitation, and no test image is relabelled because of it.
