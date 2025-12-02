#!/usr/bin/env bash
#
# build_blur_global_no_sid.sh
#
# Post-process BLUR sigma sweep results to compute GLOBAL AUROC **without SID**.
#
# What it does
# ------------
# - Starts from: $RESULTS_ROOT/sigma_blur_sweep_global.csv
#     (one row per sigma_blur value, with GLOBAL AUROC over all datasets)
# - For each row:
#     * Loads the corresponding per-image scores: <results_dir>/blur_scores.csv
#     * Filters out rows with dataset == "SID"
#     * Recomputes GLOBAL AUROC over the remaining datasets only
# - Writes a new CSV:
#     $RESULTS_ROOT/sigma_blur_sweep_global_no_sid.csv
#   with one row per sigma_blur, containing:
#     * model
#     * sigma_blur
#     * n_noise       (kept if present, otherwise empty)
#     * total_n       (number of samples without SID)
#     * global_auroc_wo_sid
#     * results_dir
#

set -euo pipefail

# Root directory where BLUR sigma sweep results were stored
RESULTS_ROOT="$HOME/results_sigma_blur_dinov3"

GLOBAL_WITH="$RESULTS_ROOT/sigma_blur_sweep_global.csv"
GLOBAL_NOSID="$RESULTS_ROOT/sigma_blur_sweep_global_no_sid.csv"

if [ ! -f "$GLOBAL_WITH" ]; then
  echo "[ERR] File not found: $GLOBAL_WITH"
  exit 1
fi

echo "[INFO] Building BLUR GLOBAL without SID -> $GLOBAL_NOSID"

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
    sigma_blur = row["sigma_blur"]
    # If 'n_noise' is absent, we keep it as empty (not relevant for blur-only)
    n_noise = row.get("n_noise", "")
    results_dir = row["results_dir"]

    scores_path = os.path.join(results_dir, "blur_scores.csv")
    if not os.path.isfile(scores_path):
        raise SystemExit(f"[ERR] Missing scores file: {scores_path}")

    scores = pd.read_csv(scores_path)

    # Required columns in blur_scores.csv:
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
            "sigma_blur": sigma_blur,
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
