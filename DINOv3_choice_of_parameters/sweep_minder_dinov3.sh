#!/usr/bin/env bash
#
# sweep_minder_dinov3.sh
#
# Sweep all (sigma, sigma_blur) combinations for the MINDER score using
# training_free_detect.py (which supports `--minder_from_csv`).
#
# What it does
# ------------
# - For each pair (sigma, sigma_blur), builds MINDER from existing noise/blur CSVs:
#     * Noise scores   from: $NOISE_ROOT/sigma_<sigma>/rigid_scores.csv
#     * Blur scores    from: $BLUR_ROOT/sigma_blur_<sigma_blur>/blur_scores.csv
#   MINDER is computed as min(noise_score, blur_score) in distance space.
# - Saves per-combo CSVs in a unique results folder.
# - Appends GLOBAL AUROC and per-dataset AUROC lines to two master CSVs.
#
# Requirements
# ------------
# - Conda env with: python>=3.10, torch, torchvision, timm, pandas, scikit-learn, pillow.
# - Eval script: $HOME/training_free_detect.py
#     (must support: --minder_from_csv, --rigid_csv, --blur_csv, --results_dir).
# - Precomputed sweeps:
#     * Noise sweep: $NOISE_ROOT/sigma_<sigma>/rigid_scores.csv
#     * Blur sweep : $BLUR_ROOT/sigma_blur_<sigma_blur>/blur_scores.csv
#
# Outputs
# -------
# - Per-combo directory: $RESULTS_ROOT/minder_sigma_<sigma>__sblur_<sigma_blur>/
#     * minder_scores.csv
#     * minder_summary.csv
# - Aggregates:
#     * $RESULTS_ROOT/minder_sweep_global.csv        (one GLOBAL row per (sigma, sigma_blur))
#     * $RESULTS_ROOT/minder_sweep_per_dataset.csv   (AUROC per dataset per (sigma, sigma_blur))
#

set -euo pipefail

# -------- Config --------
ENV_NAME="rigid"
PY="$HOME/training_free_detect.py"                 # evaluation script supporting --minder_from_csv
NOISE_ROOT="$HOME/results_sigma_dinov3"            # contains sigma_<sigma>/rigid_scores.csv
BLUR_ROOT="$HOME/results_sigma_blur_dinov3"        # contains sigma_blur_<sigma_blur>/blur_scores.csv
RESULTS_ROOT="$HOME/results_minder_dinov3"         # output per (sigma, sigma_blur)

# Sigmas and blur sigmas to sweep
SIGMAS=(0.004 0.006 0.008 0.009 0.010 0.012 0.014 0.016 0.018 0.020 0.022 0.024 0.026 0.028 0.030 0.032 0.034 0.036 0.038 0.040)
SBLURS=(0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00)

MODEL="dinov3-l16"

# -------- Activate env --------
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "$ENV_NAME"

mkdir -p "$RESULTS_ROOT"
GLOBAL_CSV="$RESULTS_ROOT/minder_sweep_global.csv"
PERDS_CSV="$RESULTS_ROOT/minder_sweep_per_dataset.csv"

# Headers if files are missing/empty
[ -s "$GLOBAL_CSV" ] || echo "model,sigma,sigma_blur,n_noise,total_n,global_auroc,results_dir" > "$GLOBAL_CSV"
[ -s "$PERDS_CSV"  ] || echo "model,sigma,sigma_blur,dataset,n,auroc,results_dir"              > "$PERDS_CSV"

for S in "${SIGMAS[@]}"; do
  for SB in "${SBLURS[@]}"; do
    OUT_DIR="$RESULTS_ROOT/minder_sigma_${S}__sblur_${SB}"
    mkdir -p "$OUT_DIR"

    RIGID_CSV="$NOISE_ROOT/sigma_${S}/rigid_scores.csv"
    RIGID_SUM="$NOISE_ROOT/sigma_${S}/rigid_summary.csv"
    BLUR_CSV="$BLUR_ROOT/sigma_blur_${SB}/blur_scores.csv"

    echo "[CHK] noise: $RIGID_CSV"
    echo "[CHK] blur : $BLUR_CSV"

    # Check that source CSVs exist
    if [ ! -f "$RIGID_CSV" ] || [ ! -f "$BLUR_CSV" ]; then
      echo "[MISS] Missing CSV(s) → skip (sigma=$S, sigma_blur=$SB)"
      continue
    fi

    # Build MINDER from CSVs (no recomputation of embeddings)
    if ! python "$PY" \
        --minder_from_csv \
        --sigma "$S" \
        --sigma_blur "$SB" \
        --rigid_csv "$RIGID_CSV" \
        --blur_csv  "$BLUR_CSV" \
        --results_dir "$OUT_DIR" ; then
      echo "[ERR] minder_from_csv failed → skip (sigma=$S, sigma_blur=$SB)"
      continue
    fi

    SUM="$OUT_DIR/minder_summary.csv"
    if [ ! -f "$SUM" ]; then
      echo "[ERR] Summary missing: $SUM → skip (sigma=$S, sigma_blur=$SB)"
      continue
    fi

    # Append GLOBAL row (try to retrieve n_noise from the noise summary if available)
    MODEL_VAL="$MODEL" S_VAL="$S" SB_VAL="$SB" OUT_VAL="$OUT_DIR" SUM_VAL="$SUM" NSUM_VAL="$RIGID_SUM" \
    python - <<'PY' >> "$GLOBAL_CSV"
import os
import pandas as pd

df = pd.read_csv(os.environ["SUM_VAL"])
g = df[df["dataset"].eq("GLOBAL")]
if len(g):
    g = g.iloc[0]
    n_tot = int(g["n"])
    auroc = float(g["auroc"])

    # n_noise (if available from noise summary CSV)
    n_noise = ""
    nsum = os.environ.get("NSUM_VAL", "")
    if nsum and os.path.isfile(nsum):
        try:
            dn = pd.read_csv(nsum)
            gn = dn[dn["dataset"].eq("GLOBAL")]
            if len(gn) and "n_noise" in gn.columns:
                val = gn.iloc[0]["n_noise"]
                if not pd.isna(val):
                    n_noise = str(int(val))
        except Exception:
            # If anything goes wrong, we just leave n_noise empty
            pass

    print(",".join([
        os.environ["MODEL_VAL"],
        os.environ["S_VAL"],
        os.environ["SB_VAL"],
        n_noise,
        str(n_tot),
        f"{auroc:.6f}",
        os.environ["OUT_VAL"],
    ]))
PY

    # Append per-dataset rows
    MODEL_VAL="$MODEL" S_VAL="$S" SB_VAL="$SB" OUT_VAL="$OUT_DIR" SUM_VAL="$SUM" \
    python - <<'PY' >> "$PERDS_CSV"
import os
import math
import pandas as pd

df = pd.read_csv(os.environ["SUM_VAL"])
df = df[df["dataset"].ne("GLOBAL")]
for _, r in df.iterrows():
    ds = str(r["dataset"])
    n  = "" if (isinstance(r["n"],    float) and math.isnan(r["n"]))    else str(int(r["n"]))
    au = "" if (isinstance(r["auroc"], float) and math.isnan(r["auroc"])) else f"{float(r['auroc']):.6f}"
    print(",".join([
        os.environ["MODEL_VAL"],
        os.environ["S_VAL"],
        os.environ["SB_VAL"],
        ds,
        n,
        au,
        os.environ["OUT_VAL"],
    ]))
PY

  done
done

echo
echo "[OK] Done."
echo "Global summary  -> $GLOBAL_CSV"
echo "Per-dataset     -> $PERDS_CSV"
