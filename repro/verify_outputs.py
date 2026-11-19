"""Independent check of the Stage E/F outputs (pure numpy, no sklearn): label-free prediction files, consistency with the
frozen raw predictions, and every headline metric recomputed from the saved predictions + restricted labels."""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
B = Path(os.environ.get("RICE_E5_FINAL_TEST", Path(__file__).resolve().parent / "results" / "final_test"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
COMMIT = {"predictions/test_predictions_raw_seed42.csv": "c9a8b386", "predictions/test_predictions_raw_seed43.csv": "8f89e91c", "predictions/test_predictions_raw_seed44.csv": "8ffefaf1"}
ok = all(sha(B / k).startswith(v) for k, v in COMMIT.items()); print("raw predictions unchanged:", ok)
L = pd.read_csv(B / "restricted_labels/test_labels.csv").set_index("image_path"); M = json.loads((B / "test_metrics_by_seed.json").read_text())
def f1m(y, p):
    f = []
    for c in range(19):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))
for s in (42, 43, 44):
    raw = pd.read_csv(B / f"predictions/test_predictions_raw_seed{s}.csv").set_index("image_path")
    for m in ("e5", "fedavg"):
        P = pd.read_csv(B / f"test_predictions_{m}_seed{s}.csv")
        nolab = not ({"y", "canonical_class", "label", "true", "correct"} & set(P.columns))
        same = P.image_path.is_unique and len(P) == 2793 and (P.pred.values == raw.loc[P.image_path, f"{m}_pred"].values).all()
        y = L.loc[P.image_path, "y"].to_numpy(); p = P.pred.to_numpy(); st = P.strict_subset.to_numpy(bool)
        acc, mf, sacc, smf = float((y == p).mean()), f1m(y, p), float((y[st] == p[st]).mean()), f1m(y[st], p[st])
        r = M["results"][f"{m}_seed{s}"]
        match = all(abs(a - b) < 1e-12 for a, b in ((acc, r["official"]["accuracy"]), (mf, r["official"]["macro_f1"]), (sacc, r["strict"]["accuracy"]), (smf, r["strict"]["macro_f1"])))
        match &= r["official"]["n"] == 2793 and r["strict"]["n"] == int(st.sum()) == 2484
        ok &= nolab and same and match
        print(f"{m:6s} seed {s}: label-free {nolab} | == frozen raw {same} | recomputed acc {acc:.4f} mF1 {mf:.4f} strict {sacc:.4f}/{smf:.4f} | matches json {match}")
print("VERIFY OUTPUTS PASS" if ok else "VERIFY OUTPUTS FAIL"); sys.exit(0 if ok else 1)
