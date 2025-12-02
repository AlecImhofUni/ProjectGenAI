#!/usr/bin/env bash
#
# sweep_sigma_dinov3.sh
#
# Sweep Gaussian noise σ values for the DINOv3 RIGID (noise-only) baseline and log AUROC.
#
# Requirements:
#   - Conda env with: python>=3.10, torch, torchvision, timm, pandas, scikit-learn, pillow.
#   - Evaluation script available at: $HOME/training_free_detect.py (supports --perturb noise).
#   - Pair dataset root with subfolders: $HOME/data/pairs_1000_eval/{real,fake}/
#     (filenames encode dataset tags like ADM / CollabDiff / SID).
#
# Script configuration (edit here as needed)
#   - DATA_ROOT : path to eval pairs (must contain real/ and fake/).
#   - MODEL     : timm model alias/name (e.g., dinov3-l16).
#   - SIGMAS    : list of σ values in pixel units [0,1] to test.
#   - N_NOISE   : number of noise samples to average per image.
#   - BATCH     : inference batch size.
#
# Outputs:
#   - For each σ: $RESULTS_ROOT/sigma_<σ>/ with:
#       * rigid_scores.csv   (per-image scores)
#       * rigid_summary.csv  (GLOBAL + per-dataset AUROC)
#   - Aggregated across all σ:
#       * $RESULTS_ROOT/sigma_sweep_global.csv        (GLOBAL AUROC per σ)
#       * $RESULTS_ROOT/sigma_sweep_per_dataset.csv   (AUROC per dataset per σ)
#

set -euo pipefail

# --- Config ---
ENV_NAME="rigid"
PY="$HOME/training_free_detect.py"            # evaluation script
DATA_ROOT="$HOME/data/pairs_1000_eval"       # must contain real/ and fake/
MODEL="dinov3-l16"                           # timm alias/name
BATCH=64
N_NOISE=3
RESULTS_ROOT="$HOME/results_sigma_dinov3"
SIGMAS=(0.004 0.006 0.008 0.010 0.012 0.014 0.016 0.018 0.020 0.022 0.024 0.026 0.028 0.030 0.032 0.034 0.036 0.038 0.040)

# --- Activate env ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "$ENV_NAME"

mkdir -p "$RESULTS_ROOT"
GLOBAL_CSV="$RESULTS_ROOT/sigma_sweep_global.csv"
PERDS_CSV="$RESULTS_ROOT/sigma_sweep_per_dataset.csv"

# Write header only if file missing/empty
[ -s "$GLOBAL_CSV" ] || echo "model,sigma,n_noise,total_n,global_auroc,results_dir" > "$GLOBAL_CSV"
[ -s "$PERDS_CSV" ] || echo "model,sigma,dataset,n,auroc,results_dir"              > "$PERDS_CSV"

for S in "${SIGMAS[@]}"; do
  OUT_DIR="$RESULTS_ROOT/sigma_${S}"
  echo "[RUN] σ=${S} -> $OUT_DIR"

  # Run NOISE-only to isolate the effect of sigma
  python "$PY" \
    --data_root "$DATA_ROOT" \
    --model "$MODEL" \
    --batch_size "$BATCH" \
    --sigma "$S" --n_noise "$N_NOISE" \
    --perturb noise \
    --results_dir "$OUT_DIR"

  SUM="$OUT_DIR/rigid_summary.csv"
  if [ ! -f "$SUM" ]; then
    echo "[ERR] Summary not found: $SUM"
    exit 1
  fi

  # Append GLOBAL row
  MODEL_VAL="$MODEL" S_VAL="$S" NNOISE_VAL="$N_NOISE" OUT_VAL="$OUT_DIR" SUM_VAL="$SUM" \
  python - <<'PY' >> "$GLOBAL_CSV"
import os
import pandas as pd

df = pd.read_csv(os.environ["SUM_VAL"])
g = df[df["dataset"].eq("GLOBAL")].iloc[0]
row = [
    os.environ["MODEL_VAL"],
    os.environ["S_VAL"],
    os.environ["NNOISE_VAL"],
    str(int(g["n"])),
    f"{float(g['auroc']):.6f}",
    os.environ["OUT_VAL"],
]
print(",".join(row))
PY

  # Append per-dataset rows
  MODEL_VAL="$MODEL" S_VAL="$S" OUT_VAL="$OUT_DIR" SUM_VAL="$SUM" \
  python - <<'PY' >> "$PERDS_CSV"
import os
import math
import pandas as pd

df = pd.read_csv(os.environ["SUM_VAL"])
df = df[df["dataset"].ne("GLOBAL")]
for _, r in df.iterrows():
    au = "" if (isinstance(r["auroc"], float) and math.isnan(r["auroc"])) else f"{float(r['auroc']):.6f}"
    n  = "" if (isinstance(r["n"],    float) and math.isnan(r["n"]))    else str(int(r["n"]))
    print(",".join([
        os.environ["MODEL_VAL"],
        os.environ["S_VAL"],
        str(r["dataset"]),
        n,
        au,
        os.environ["OUT_VAL"],
    ]))
PY

done

echo
echo "[OK] Done."
echo "Global summary  -> $GLOBAL_CSV"
echo "Per-dataset     -> $PERDS_CSV"
