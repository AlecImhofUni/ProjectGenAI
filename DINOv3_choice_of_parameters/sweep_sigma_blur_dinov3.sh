#!/usr/bin/env bash
#
# sweep_sigma_blur_dinov3.sh
#
# Sweep Gaussian blur σ for the DINOv3 Contrastive Blur baseline and log AUROC.
#
# Requirements:
#   - Conda env with: python>=3.10, torch, torchvision, timm, pandas, scikit-learn, pillow.
#   - Evaluation script available at: $HOME/training_free_detect.py
#       (supports: --perturb blur, --sigma_blur).
#   - Pair dataset root with subfolders: $HOME/data/pairs_1000_eval/{real,fake}/
#       (filenames encode dataset tags like ADM / CollabDiff / SID).
#
# Script configuration (edit here as needed):
#   - DATA_ROOT  : path to eval pairs (must contain real/ and fake/).
#   - MODEL      : timm model alias/name (e.g., dinov3-l16).
#   - BLURS      : list of σ_blur values (in pixels at 224×224) to test.
#   - BATCH      : inference batch size.
#
# Outputs:
#   - For each σ_blur: $RESULTS_ROOT/sigma_blur_<σ>/ with:
#       * blur_scores.csv    (per-image scores)
#       * blur_summary.csv   (GLOBAL + per-dataset AUROC)
#   - Aggregated across all σ_blur:
#       * $RESULTS_ROOT/sigma_blur_sweep_global.csv       (GLOBAL AUROC per σ_blur)
#       * $RESULTS_ROOT/sigma_blur_sweep_per_dataset.csv  (AUROC per dataset per σ_blur)
#

set -euo pipefail

# --- Config ---
ENV_NAME="rigid"
PY="$HOME/training_free_detect.py"           # evaluation script
DATA_ROOT="$HOME/data/pairs_1000_eval"      # must contain real/ and fake/
MODEL="dinov3-l16"                          # timm alias/name
BATCH=64
RESULTS_ROOT="$HOME/results_sigma_blur_dinov3"

# sigma_blur candidates (in pixel units)
BLURS=(0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)

# --- Activate env ---
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "$ENV_NAME"

mkdir -p "$RESULTS_ROOT"
GLOBAL_CSV="$RESULTS_ROOT/sigma_blur_sweep_global.csv"
PERDS_CSV="$RESULTS_ROOT/sigma_blur_sweep_per_dataset.csv"

# Write header only if file missing/empty
[ -s "$GLOBAL_CSV" ] || echo "model,sigma_blur,total_n,global_auroc,results_dir" > "$GLOBAL_CSV"
[ -s "$PERDS_CSV" ] || echo "model,sigma_blur,dataset,n,auroc,results_dir"       > "$PERDS_CSV"

for S in "${BLURS[@]}"; do
  OUT_DIR="$RESULTS_ROOT/sigma_blur_${S}"
  echo "[RUN] σ_blur=${S} -> $OUT_DIR"

  # Run BLUR-only to isolate the effect of sigma_blur
  python "$PY" \
    --data_root "$DATA_ROOT" \
    --model "$MODEL" \
    --batch_size "$BATCH" \
    --sigma_blur "$S" \
    --perturb blur \
    --results_dir "$OUT_DIR"

  SUM="$OUT_DIR/blur_summary.csv"
  if [ ! -f "$SUM" ]; then
    echo "[ERR] Summary not found: $SUM"
    exit 1
  fi

  # Append GLOBAL row
  MODEL_VAL="$MODEL" S_VAL="$S" OUT_VAL="$OUT_DIR" SUM_VAL="$SUM" \
  python - <<'PY' >> "$GLOBAL_CSV"
import os
import pandas as pd

df = pd.read_csv(os.environ["SUM_VAL"])
g = df[df["dataset"].eq("GLOBAL")].iloc[0]
row = [
    os.environ["MODEL_VAL"],
    os.environ["S_VAL"],
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
