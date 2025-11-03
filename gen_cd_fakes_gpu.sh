#!/usr/bin/env bash

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate cdiff310gpu

cd "$HOME/data/Collaborative-Diffusion"
mkdir -p outputs "$HOME/data/CollabDiff/fake"

TARGET="${1:-270}"    # aim for >=250 fakes; adjust if necessary
BATCH="${2:-8}"
STEPS="${3:-10}"
PROMPT="${4:-A portrait photo of a person.}"

# current counter in outputs/
count_generated () { find outputs -type f -iregex '.*\.\(png\|jpg\|jpeg\)' | wc -l; }

seed=0
gen=$(count_generated)
echo "[INFO] départ: ${gen}/${TARGET} images"

# loop over seeds, process all masks at each seed
while [ "$gen" -lt "$TARGET" ]; do
  for m in test_data/512_masks/*.png; do
    python generate.py \
      --mask_path "$m" \
      --input_text "$PROMPT" \
      --ddim_steps "$STEPS" \
      --batch_size "$BATCH" \
      --save_folder "outputs/seed_${seed}_$(basename "$m" .png)"
  done
  seed=$((seed+1))
  gen=$(count_generated)
  echo "[INFO] générées (outputs/): $gen / $TARGET (seed=${seed})"
done

# move to fake/ while keeping the tree structure (no collisions)
rsync -av --ignore-existing outputs/ "$HOME/data/CollabDiff/fake/"

echo "[OK] FAKES CollabDiff (récursif):" \
  $(find "$HOME/data/CollabDiff/fake" -type f -iregex '.*\.\(png\|jpg\|jpeg\)' | wc -l)
