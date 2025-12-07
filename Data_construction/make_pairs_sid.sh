#!/usr/bin/env bash
#
# make_pairs_sid.sh
#
# Append SID pairs to an existing eval split (ADM + CollabDiff).
#
# It:
#   - Samples N real/fake SID images (authentic / fully_synthetic).
#   - Finds the next available pair index in an existing pairs_<K>_eval folder.
#   - For each new pair i, copies:
#       real ->  $EVAL_DIR/real/pair_000i_SID.<ext>
#       fake ->  $EVAL_DIR/fake/pair_000i_SID.<ext>
#   - Appends lines to an existing CSV:
#       pair_XXXX,SID,real_src,fake_src,eval_real,eval_fake
#
# Usage:
#   ./make_pairs_sid.sh [N]
#
#   N (int, optional): how many SID pairs to append (default: 1000).
#
# Environment variables (optional):
#   ROOT_DATA : root data dir (default: "$HOME/data").
#   SID_ROOT  : root of SID dataset (default: "$ROOT_DATA/SID_dataset").
#   SID_REAL  : folder with authentic SID images (default: "$SID_ROOT/authentic").
#   SID_FAKE  : folder with fully_synthetic SID images (default: "$SID_ROOT/fully_synthetic").
#   EVAL_DIR  : existing pairs_<K>_eval dir (default: "$ROOT_DATA/pairs_1000_eval").
#   CSV_PATH  : CSV file to append (default: "$EVAL_DIR/pairs_1000.csv").
#
# Expected layout:
#   $SID_REAL/*.png|jpg|jpeg
#   $SID_FAKE/*.png|jpg|jpeg
#   $EVAL_DIR/real/*.*
#   $EVAL_DIR/fake/*.*
#

set -euo pipefail

# log
# ---
# Print a message to stderr.
#
# Args:
#   $* (str...): Any number of strings to print.
#
# Returns:
#   None
log() {
  printf '%s\n' "$*" >&2
}

# ---------- Inputs ----------

N="${1:-1000}"                               # (int) how many SID pairs to append
ROOT="${ROOT_DATA:-$HOME/data}"             # (str) datasets root
SID_ROOT="${SID_ROOT:-$ROOT/SID_dataset}"   # (str) SID dataset root
SID_REAL="${SID_REAL:-$SID_ROOT/authentic}" # (str) folder for authentic images
SID_FAKE="${SID_FAKE:-$SID_ROOT/fully_synthetic}" # (str) folder for fully_synthetic images

# Where ADM/CD pairs already live:
EVAL_DIR="${EVAL_DIR:-$ROOT/pairs_1000_eval}"    # (str) eval dir with real/ and fake/
CSV_PATH="${CSV_PATH:-$EVAL_DIR/pairs_1000.csv}" # (str) existing CSV to append

TAG="SID"  # (str) dataset tag added in filename and CSV

# ---------- Checks ----------

[ -d "$SID_REAL" ]         || { log "[ERROR] Missing: $SID_REAL"; exit 1; }
[ -d "$SID_FAKE" ]         || { log "[ERROR] Missing: $SID_FAKE"; exit 1; }
[ -d "$EVAL_DIR/real" ]    || { log "[ERROR] Missing: $EVAL_DIR/real"; exit 1; }
[ -d "$EVAL_DIR/fake" ]    || { log "[ERROR] Missing: $EVAL_DIR/fake"; exit 1; }
[ -e "$CSV_PATH" ] || touch "$CSV_PATH"  # create if absent (no header)

# ---------- Helpers ----------

# count_imgs
# ----------
# Count how many image files are in a directory.
#
# Args:
#   $1 (str): Directory path.
#
# Output (stdout):
#   int: number of files whose extension matches png|jpg|jpeg (case-insensitive).
#
# Returns:
#   0 (success), exits non-zero on error.
count_imgs() {
  find "$1" -type f -iregex '.*\.\(png\|jpg\|jpeg\)' | wc -l | tr -d ' '
}

# pick
# ----
# Pick K random image files from a directory and print their paths.
# (Uses `shuf`, which est garanti présent dans ton environnement.)
#
# Args:
#   $1 (str): Directory path.
#   $2 (int): Number K of files to pick.
#
# Output (stdout):
#   str: K lines, each a file path.
#
# Returns:
#   0 on success. Non-zero if find/shuf/head fails.
pick() { # $1=dir  $2=K -> print K random file paths
  find "$1" -type f -iregex '.*\.\(png\|jpg\|jpeg\)' | shuf | head -n "$2"
}

# next_pair_id
# ------------
# Compute the next available pair ID by scanning existing filenames.
#
# It looks for files named like:
#   pair_0001_*.*
# in both $EVAL_DIR/real and $EVAL_DIR/fake, and returns max_id + 1.
#
# Args:
#   None (reads from $EVAL_DIR/{real,fake}).
#
# Output (stdout):
#   int: the next pair index (1-based).
#
# Returns:
#   0 on success, non-zero if find/sed fails.
next_pair_id() {
  # scan existing pair numbers in real/ and fake/
  local maxid=0 cur
  while read -r f; do
    cur=$(sed -n 's/^pair_\([0-9]\{4\}\)_.*/\1/p' <<<"$(basename "$f")")
    [ -n "${cur:-}" ] && ((10#$cur > maxid)) && maxid=$((10#$cur))
  done < <(
    (find "$EVAL_DIR/real" -maxdepth 1 -type f -printf '%f\n'
     find "$EVAL_DIR/fake" -maxdepth 1 -type f -printf '%f\n') 2>/dev/null
  )
  echo $((maxid + 1))
}

# mkpair
# ------
# Build one SID pair: copy real/fake images to eval dirs and append a line to CSV.
#
# Args:
#   $1 (int): Pair ID (1-based).
#   $2 (str): Path to source real image.
#   $3 (str): Path to source fake image.
#   $4 (str): TAG label to embed in the filename (e.g. "SID").
#
# Side effects:
#   - Copies images into:
#       $EVAL_DIR/real/pair_XXXX_TAG.ext
#       $EVAL_DIR/fake/pair_XXXX_TAG.ext
#   - Appends one row to $CSV_PATH:
#       pair_XXXX,TAG,real_src,fake_src,eval_real,eval_fake
#
# Returns:
#   0 on success, exits non-zero on copy or printf failure.
mkpair() { # $1=pair_id  $2=real_src  $3=fake_src  $4=TAG
  local pid="$1" rp="$2" fp="$3" tag="$4"
  local rext="${rp##*.}" fext="${fp##*.}"
  local real_out="$EVAL_DIR/real/pair_$(printf '%04d' "$pid")_${tag}.$rext"
  local fake_out="$EVAL_DIR/fake/pair_$(printf '%04d' "$pid")_${tag}.$fext"

  cp -f "$rp" "$real_out"
  cp -f "$fp" "$fake_out"

  printf 'pair_%04d,%s,%s,%s,%s,%s\n' \
    "$pid" "$tag" "$rp" "$fp" "$real_out" "$fake_out" >> "$CSV_PATH"
}

# ---------- Capacity & picks ----------

cap_r=$(count_imgs "$SID_REAL")
cap_f=$(count_imgs "$SID_FAKE")
CAP=$(( cap_r < cap_f ? cap_r : cap_f ))

if [ "$CAP" -lt 1 ]; then
  log "[ERROR] No SID images found."
  exit 1
fi

if [ "$N" -gt "$CAP" ]; then
  log "[WARN] request $N > capacity $CAP -> using $CAP"
  N="$CAP"
fi

# Sample N real and fake SID images.
mapfile -t R < <(pick "$SID_REAL" "$N")
mapfile -t F < <(pick "$SID_FAKE" "$N")

# ---------- Append ----------

PAIR=$(next_pair_id)
log "[INFO] Starting at pair_$(printf '%04d' "$PAIR") ; appending $N SID pairs"

for i in $(seq 0 $((N-1))); do
  mkpair "$PAIR" "${R[$i]}" "${F[$i]}" "$TAG"
  PAIR=$((PAIR + 1))
done

log "[OK] Appended $N SID pairs to $EVAL_DIR/{real,fake}"
log "     CSV appended: $CSV_PATH"
log "     Counts now: real=$(find "$EVAL_DIR/real" -type f | wc -l | tr -d ' ')  fake=$(find "$EVAL_DIR/fake" -type f | wc -l | tr -d ' ')"
