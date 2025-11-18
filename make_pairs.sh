#!/usr/bin/env bash
#
# make_pairs.sh
#
# Build an eval split of paired images for ADM + CollabDiff.
#
# It:
#   - Samples real/fake images from ADM and CollabDiff.
#   - Builds <TOTAL> pairs in total.
#   - For each pair i, copies:
#       real ->  $EVAL/real/pair_000i_<TAG>.<ext>
#       fake ->  $EVAL/fake/pair_000i_<TAG>.<ext>
#   - Logs everything to a CSV:
#       pair_id,source_tag,real_src,fake_src,eval_real,eval_fake
#
# Usage:
#   ./make_pairs.sh [TOTAL]
#
#   TOTAL (int, optional) : number of pairs to create (default: 1000).
#
# Environment variables (optional):
#   ROOT_DATA      : root directory for datasets (default: "$HOME/data").
#   TAKE_ADM       : desired number of ADM pairs (default: TOTAL/2).
#   TAKE_CD        : desired number of CollabDiff pairs (default: TOTAL - TAKE_ADM).
#   ADM_TAG_LABEL  : label used in ADM filenames (default: "AMD").
#   CD_TAG_LABEL   : label used in CD filenames (default: "CollabDiff").
#
# Expected layout:
#   $ROOT/ADM/val/real/*.png|jpg|jpeg
#   $ROOT/ADM/val/fake/*.png|jpg|jpeg
#   $ROOT/CollabDiff/real/*.png|jpg|jpeg
#   $ROOT/CollabDiff/fake/*.png|jpg|jpeg
#
# Output:
#   $ROOT/pairs_<TOTAL>_eval/
#     real/pair_0001_<TAG>.<ext>
#     fake/pair_0001_<TAG>.<ext>
#     ...
#   $ROOT/pairs_<TOTAL>_eval/pairs_<TOTAL>.csv
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

# ---------- Params ----------

TOTAL="${1:-1000}"                      # (int) number of pairs to create
ROOT="${ROOT_DATA:-$HOME/data}"         # (str) datasets root

ADM_REAL="$ROOT/ADM/val/real"
ADM_FAKE="$ROOT/ADM/val/fake"
CD_REAL="$ROOT/CollabDiff/real"
CD_FAKE="$ROOT/CollabDiff/fake"

TAKE_ADM="${TAKE_ADM:-$((TOTAL/2))}"    # (int) desired pairs from ADM
TAKE_CD="${TAKE_CD:-$((TOTAL-TAKE_ADM))}" # (int) desired pairs from CollabDiff

EVAL="$ROOT/pairs_${TOTAL}_eval"
CSV="$EVAL/pairs_${TOTAL}.csv"

# Labels used in filenames (e.g. pair_0001_AMD.png, pair_0001_CollabDiff.jpeg)
ADM_TAG_LABEL="${ADM_TAG_LABEL:-AMD}"
CD_TAG_LABEL="${CD_TAG_LABEL:-CollabDiff}"

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
#
# Args:
#   $1 (str): Directory path.
#   $2 (int): Number K of files to pick.
#
# Output (stdout):
#   str: K lines, each a file path.
#
# Returns:
#   0 on success. Non-zero if find/shuf/awk fails.
pick() { # $1=dir $2=K -> echo K random files
  if command -v shuf >/dev/null 2>&1; then
    # Fast path: use shuf if available.
    find "$1" -type f -iregex '.*\.\(png\|jpg\|jpeg\)' | shuf | head -n "$2"
  else
    # Fallback: randomize via awk if shuf is missing.
    find "$1" -type f -iregex '.*\.\(png\|jpg\|jpeg\)' \
    | awk 'BEGIN{srand()} {printf "%f %s\n", rand(), $0}' \
    | sort -n | cut -d" " -f2- | head -n "$2"
  fi
}

# mkpair
# ------
# Build one pair: copy real/fake images to eval dirs and append a line to the CSV.
#
# Args:
#   $1 (int): Pair ID (1-based or 0-based; used only for formatting).
#   $2 (str): Path to source real image.
#   $3 (str): Path to source fake image.
#   $4 (str): TAG label to embed in the filename (e.g. "AMD", "CollabDiff").
#
# Side effects:
#   - Copies images into:
#       $EVAL/real/pair_XXXX_TAG.ext
#       $EVAL/fake/pair_XXXX_TAG.ext
#   - Appends one row to $CSV:
#       pair_XXXX,TAG,real_src,fake_src,eval_real,eval_fake
#
# Returns:
#   0 on success, exits non-zero on copy or printf failure.
mkpair() { # $1=pair_id $2=real_src $3=fake_src $4=TAGLABEL
  local pid="$1" rp="$2" fp="$3" tag="$4"
  local rext="${rp##*.}" fext="${fp##*.}"

  # Filenames WITHOUT _real / _fake, only the pair id and source tag.
  local real_out="$EVAL/real/pair_$(printf '%04d' "$pid")_${tag}.$rext"
  local fake_out="$EVAL/fake/pair_$(printf '%04d' "$pid")_${tag}.$fext"

  cp -f "$rp" "$real_out"
  cp -f "$fp" "$fake_out"

  # CSV: pair_id,source_tag,real_src,fake_src,eval_real,eval_fake
  printf 'pair_%04d,%s,%s,%s,%s,%s\n' \
    "$pid" "$tag" "$rp" "$fp" "$real_out" "$fake_out" >> "$CSV"
}

# ---------- Prep ----------

# Create eval directories and reset CSV.
mkdir -p "$EVAL/real" "$EVAL/fake"
: > "$CSV"

# Capacities: how many balanced pairs we can form for ADM / CD
adm_r=$(count_imgs "$ADM_REAL"); adm_f=$(count_imgs "$ADM_FAKE")
cd_r=$(count_imgs "$CD_REAL");   cd_f=$(count_imgs "$CD_FAKE")
ADM_CAP=$(( adm_r < adm_f ? adm_r : adm_f ))
CD_CAP=$(( cd_r < cd_f ? cd_r : cd_f ))

# Adjust requested ADM/CD counts if they exceed what is available.
[ "$TAKE_ADM" -gt "$ADM_CAP" ] && {
  log "[WARN] ADM cap $ADM_CAP < req $TAKE_ADM -> using $ADM_CAP"
  TAKE_ADM="$ADM_CAP"
}
[ "$TAKE_CD"  -gt "$CD_CAP"  ] && {
  log "[WARN] CD  cap $CD_CAP  < req $TAKE_CD  -> using $CD_CAP"
  TAKE_CD="$CD_CAP"
}

# If we have leftover budget (SUM < TOTAL), try to top up ADM or CD.
SUM=$((TAKE_ADM + TAKE_CD))
if [ "$SUM" -lt "$TOTAL" ]; then
  REM=$((TOTAL - SUM))
  FREE_ADM=$((ADM_CAP - TAKE_ADM))
  FREE_CD=$((CD_CAP - TAKE_CD))
  if   [ "$FREE_ADM" -ge "$REM" ]; then
    TAKE_ADM=$((TAKE_ADM + REM))
  elif [ "$FREE_CD"  -ge "$REM" ]; then
    TAKE_CD=$((TAKE_CD + REM))
  else
    log "[ERROR] Not enough capacity (ADM=$ADM_CAP CD=$CD_CAP) for TOTAL=$TOTAL"
    exit 1
  fi
fi

log "[PLAN] TOTAL=$TOTAL (ADM=$TAKE_ADM, CD=$TAKE_CD)"

# ---------- Build ----------

PAIR=0

# ADM pairs
if [ "$TAKE_ADM" -gt 0 ]; then
  mapfile -t R_ADM < <(pick "$ADM_REAL" "$TAKE_ADM")
  mapfile -t F_ADM < <(pick "$ADM_FAKE" "$TAKE_ADM")
  for i in $(seq 0 $((TAKE_ADM-1))); do
    PAIR=$((PAIR+1))
    mkpair "$PAIR" "${R_ADM[$i]}" "${F_ADM[$i]}" "$ADM_TAG_LABEL"
  done
fi

# CollabDiff pairs
if [ "$TAKE_CD" -gt 0 ]; then
  mapfile -t R_CD < <(pick "$CD_REAL" "$TAKE_CD")
  mapfile -t F_CD < <(pick "$CD_FAKE" "$TAKE_CD")
  for i in $(seq 0 $((TAKE_CD-1))); do
    PAIR=$((PAIR+1))
    mkpair "$PAIR" "${R_CD[$i]}" "${F_CD[$i]}" "$CD_TAG_LABEL"
  done
fi

log "[OK] Eval split: $EVAL/{real,fake}"
log "     CSV: $CSV"
log "Counts: real=$(find "$EVAL/real" -type f | wc -l | tr -d ' ') fake=$(find "$EVAL/fake" -type f | wc -l | tr -d ' ')"
