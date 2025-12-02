#!/usr/bin/env bash
#
# build_noise_global_no_sid.sh
#
# Post-process NOISE (RIGID) sigma sweep results to compute GLOBAL AUROC **without SID**.
#
# What it does
# ------------
# - Starts from: $RESULTS_ROOT/sigma_sweep_global.csv
#     (one row per sigma value, with GLOBAL AUROC over all datasets)
# - For each row:
#     * Loads the corresponding per-image scores: <results_dir>/rigid_scores.csv
#     * Filters out rows with dataset == "SID"
#     * Recomputes GLOBAL AUROC over the remaining datasets only
# - Writes a new CSV:
#     $RESULTS_ROOT/sigma_sweep_global_no_sid.csv
#   with one row per sigma, containing:
#     * model
#     * sigma
#     * n_noise
#     * total_n              (number of samples without SID)
#     * global_auroc_wo_sid
#     * results_dir
#

set -euo pipefail

# Root directory where NOISE sigma sweep results were stored
RESULTS_ROOT="$HOME/results_sigma_dinov3"

GLOBAL_WITH="$RESULTS_ROOT/sigma_sweep_global.csv"
GLOBAL_NOSID="$RESULTS_ROOT/sigma_sweep_global_no_sid.csv"

if [ ! -f "$GLOBAL_WITH" ]; then
  echo "[ERR] File not found: $GLOBAL_WITH"
  exit 1
fi

echo "[INFO] Building NOISE GLOBAL without SID -> $GLOBAL_NOSID"

GLOBAL_WITH="$GLOBAL_WITH" GLOBAL_NOSID="$GLOBAL_NOSID" python - <<'PY'
import os
import pandas as pd
from sklearn.metrics import roc_auc_score

gw = os.environ["GLOBAL_WITH"]
gn = os.environ["GLOBAL_NOSID"]

# Existing global summary (includes SID)
df_global = pd.read_csv(gw)

rows = []

for _, row in df_global.iterrows():
    model = row["model"]
    sigma = row["sigma"]
    n_noise = row["n_noise"]
    results_dir = row["results_dir"]

    scores_path = os.path.join(results_dir, "rigid_scores.csv")
    if not os.path.isfile(scores_path):
        raise SystemExit(f"[ERR] Missing scores file: {scores_path}")

    scores = pd.read_csv(scores_path)

    # Required columns in rigid_scores.csv:
    # - dataset : dataset name (ADM / CollabDiff / SID / UNKNOWN)
    # - label   : 0/1 ground-truth (real/fake)
    # - score   : anomaly score (higher = more likely fake)
    required = {"dataset", "label", "score"}
    if not required.issubset(scores.columns):
        raise SystemExit(
            f"[ERR] Expected columns {required} in {scores_path}, got {scores.columns.tolist()}"
        )

    # Filter out SID
    scores_wo = scores[scores["dataset"] != "SID"]

    if scores_wo.empty:
        # No data after filtering
        total_n = 0
        auroc_wo = ""
    else:
        y_true = scores_wo["label"].to_numpy()
        y_score = scores_wo["score"].to_numpy()
        total_n = len(y_true)
        auroc_wo = f"{roc_auc_score(y_true, y_score):.6f}"

    rows.append(
        {
            "model": model,
            "sigma": sigma,
            "n_noise": n_noise,
            "total_n": total_n,
            "global_auroc_wo_sid": auroc_wo,
            "results_dir": results_dir,
        }
    )

out = pd.DataFrame(rows)
out.to_csv(gn, index=False)
print(f"[OK] Wrote {gn}")
PY
