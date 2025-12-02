#!/usr/bin/env bash
#
# gen_cd_fakes_gpu.sh
#
# Generate fake images for the CollabDiff dataset using Collaborative-Diffusion on GPU.
#
# Requirements:
#   - Conda env "cdiff310gpu" with all Collaborative-Diffusion dependencies.
#   - Repository cloned at:  $HOME/data/Collaborative-Diffusion
#   - Masks available under: $HOME/data/Collaborative-Diffusion/test_data/512_masks/*.png
#
# Output:
#   - Intermediate generations under: $HOME/data/Collaborative-Diffusion/outputs/
#   - Final fake set (rsynced) under: $HOME/data/CollabDiff/fake/
#

set -euo pipefail

# Load conda so that "conda activate" works from a non-interactive shell.
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Activate the GPU environment for Collaborative-Diffusion.
conda activate cdiff310gpu

# Go to the Collaborative-Diffusion repo.
cd "$HOME/data/Collaborative-Diffusion"

# Ensure output directories exist:
#   - local "outputs/" for raw generations
#   - global "$HOME/data/CollabDiff/fake" for the consolidated fake set
mkdir -p outputs "$HOME/data/CollabDiff/fake"

# --- CLI arguments with defaults ---

TARGET="${1:-270}"    # (int) total number of images to aim for
BATCH="${2:-8}"       # (int) batch size per generate.py call
STEPS="${3:-10}"      # (int) number of DDIM steps in generate.py
PROMPT="${4:-A portrait photo of a person.}"  # (string) text prompt

# --- Helper function ---

# count_generated
# ---------------
# Count how many PNG/JPEG files currently exist in the local "outputs/" tree.
#
# Inputs :
#   - None (reads from ./outputs/)
# Outputs:
#   - Prints an integer count to stdout (used via command substitution).
count_generated () { find outputs -type f -iregex '.*\.\(png\|jpg\|jpeg\)' | wc -l; }

# Start from seed 0 and see how many images we already have (if any).
seed=0
gen=$(count_generated)
echo "[INFO] départ: ${gen}/${TARGET} images"

# --- Main generation loop ---
# Iterate over seeds; at each seed run through all masks in test_data/512_masks/.
# After each full pass, update the counter and stop once reach TARGET.
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

# --- Consolidate fakes ---
# Move (rsync) everything under outputs/ into $HOME/data/CollabDiff/fake/
# while:
#   - keeping directory structure (-a)
#   - not overwriting existing files (--ignore-existing).
rsync -av --ignore-existing outputs/ "$HOME/data/CollabDiff/fake/"

echo "[OK] FAKES CollabDiff (récursif):" \
  $(find "$HOME/data/CollabDiff/fake" -type f -iregex '.*\.\(png\|jpg\|jpeg\)' | wc -l)
