#!/usr/bin/env bash
#
# build_minder_global_no_sid.sh
#
# Post-process MINDER sweep results to compute GLOBAL AUROC **without SID**.
#
# What it does
# ------------
# - Starts from: $RESULTS_ROOT/minder_sweep_global.csv
#     (one row per (sigma, sigma_blur), with GLOBAL AUROC over all datasets)
# - For each row:
#     * Loads the corresponding per-image scores: <results_dir>/minder_scores.csv
#     * Filters out rows with dataset == "SID"
#     * Recomputes GLOBAL AUROC over the remaining datasets only
# - Writes an updated CSV:
#     $RESULTS_ROOT/minder_sweep_global_no_sid.csv
#   which contains all original columns +:
#     * total_n_wo_sid        (number of samples without SID)
#     * global_auroc_wo_sid   (GLOBAL AUROC without SID)
#

set -euo pipefail

# Root directory where MINDER sweep results were stored
RESULTS_ROOT="$HOME/results_minder_dinov3"

GLOBAL_WITH="$RESULTS_ROOT/minder_sweep_global.csv"
GLOBAL_NOSID="$RESULTS_ROOT/minder_sweep_global_no_sid.csv"

if [ ! -f "$GLOBAL_WITH" ]; then
  echo "[ERR] File not found: $GLOBAL_WITH"
  exit 1
fi

echo "[INFO] Building MINDER GLOBAL without SID -> $GLOBAL_NOSID"

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
    # Start from all existing columns for this (sigma, sigma_blur, ...)
    row_out = row.to_dict()

    results_dir = row["results_dir"]
    scores_path = os.path.join(results_dir, "minder_scores.csv")
    if not os.path.isfile(scores_path):
        raise SystemExit(f"[ERR] Missing scores file: {scores_path}")

    scores = pd.read_csv(scores_path)

    # Required columns in minder_scores.csv:
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
        total_n_wo = 0
        auroc_wo = ""
    else:
        y_true = scores_wo["label"].to_numpy()
        y_score = scores_wo["score"].to_numpy()
        total_n_wo = len(y_true)
        auroc_wo = f"{roc_auc_score(y_true, y_score):.6f}"

    # Add "without SID" metrics
    row_out["total_n_wo_sid"] = total_n_wo
    row_out["global_auroc_wo_sid"] = auroc_wo

    rows.append(row_out)

out = pd.DataFrame(rows)
out.to_csv(gn, index=False)
print(f"[OK] Wrote {gn}")
PY
