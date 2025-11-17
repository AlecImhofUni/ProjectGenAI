# ~/append_pairs_sid.sh
#!/usr/bin/env bash
set -euo pipefail

log(){ printf '%s\n' "$*" >&2; }

# ---------- Inputs ----------
N="${1:-1000}"                               # how many SID pairs to append
ROOT="${ROOT_DATA:-$HOME/data}"
SID_ROOT="${SID_ROOT:-$ROOT/SID_dataset}"
SID_REAL="${SID_REAL:-$SID_ROOT/authentic}"
SID_FAKE="${SID_FAKE:-$SID_ROOT/fully_synthetic}"

# Where your ADM/CD pairs already live:
EVAL_DIR="${EVAL_DIR:-$ROOT/pairs_1000_eval}"         # has real/ and fake/
CSV_PATH="${CSV_PATH:-$EVAL_DIR/pairs_1000.csv}"      # existing CSV to append

TAG="SID"  # dataset tag added in filename and CSV

# ---------- Checks ----------
[ -d "$SID_REAL" ] || { log "[ERROR] Missing: $SID_REAL"; exit 1; }
[ -d "$SID_FAKE" ] || { log "[ERROR] Missing: $SID_FAKE"; exit 1; }
[ -d "$EVAL_DIR/real" ] || { log "[ERROR] Missing: $EVAL_DIR/real"; exit 1; }
[ -d "$EVAL_DIR/fake" ] || { log "[ERROR] Missing: $EVAL_DIR/fake"; exit 1; }
[ -e "$CSV_PATH" ] || touch "$CSV_PATH"  # create if absent (no header)

# ---------- Helpers ----------
count_imgs(){ find "$1" -type f -iregex '.*\.\(png\|jpg\|jpeg\)' | wc -l | tr -d ' '; }

pick(){ # $1=dir  $2=K -> print K random file paths
  if command -v shuf >/dev/null 2>&1; then
    find "$1" -type f -iregex '.*\.\(png\|jpg\|jpeg\)' | shuf | head -n "$2"
  else
    find "$1" -type f -iregex '.*\.\(png\|jpg\|jpeg\)' \
    | awk 'BEGIN{srand()} {printf "%f %s\n", rand(), $0}' \
    | sort -n | cut -d" " -f2- | head -n "$2"
  fi
}

next_pair_id(){
  # scan existing pair numbers in real/ and fake/
  local maxid=0 cur
  while read -r f; do
    cur=$(sed -n 's/^pair_\([0-9]\{4\}\)_.*/\1/p' <<<"$(basename "$f")")
    [ -n "${cur:-}" ] && ((10#$cur > maxid)) && maxid=$((10#$cur))
  done < <( (find "$EVAL_DIR/real" -maxdepth 1 -type f -printf '%f\n' ; \
             find "$EVAL_DIR/fake" -maxdepth 1 -type f -printf '%f\n') 2>/dev/null )
  echo $((maxid + 1))
}

mkpair(){ # $1=pair_id  $2=real_src  $3=fake_src  $4=TAG
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
if [ "$CAP" -lt 1 ]; then log "[ERROR] No SID images found."; exit 1; fi
if [ "$N" -gt "$CAP" ]; then log "[WARN] request $N > capacity $CAP -> using $CAP"; N="$CAP"; fi

mapfile -t R < <(pick "$SID_REAL" "$N")
mapfile -t F < <(pick "$SID_FAKE" "$N")

# ---------- Append ----------
PAIR=$(next_pair_id)
log "[INFO] Starting at pair_$(printf '%04d' "$PAIR") ; appending $N SID pairs"

for i in $(seq 0 $((N-1))); do
  mkpair "$PAIR" "${R[$i]}" "${F[$i]}" "$TAG"
  PAIR=$((PAIR+1))
done

log "[OK] Appended $N SID pairs to $EVAL_DIR/{real,fake}"
log "     CSV appended: $CSV_PATH"
log "     Counts now: real=$(find "$EVAL_DIR/real" -type f | wc -l | tr -d ' ')  fake=$(find "$EVAL_DIR/fake" -type f | wc -l | tr -d ' ')"
