# ~/make_pairs.sh
#!/usr/bin/env bash
set -euo pipefail

log(){ printf '%s\n' "$*" >&2; }

# ---------- Params ----------
TOTAL="${1:-1000}"                      # number of pairs to create
ROOT="${ROOT_DATA:-$HOME/data}"         # datasets root

ADM_REAL="$ROOT/ADM/val/real"
ADM_FAKE="$ROOT/ADM/val/fake"
CD_REAL="$ROOT/CollabDiff/real"
CD_FAKE="$ROOT/CollabDiff/fake"

TAKE_ADM="${TAKE_ADM:-$((TOTAL/2))}"    # desired pairs from ADM
TAKE_CD="${TAKE_CD:-$((TOTAL-TAKE_ADM))}"

EVAL="$ROOT/pairs_${TOTAL}_eval"
CSV="$EVAL/pairs_${TOTAL}.csv"

# Labels used in filenames
ADM_TAG_LABEL="${ADM_TAG_LABEL:-AMD}"
CD_TAG_LABEL="${CD_TAG_LABEL:-CollabDiff}"

# ---------- Helpers ----------
count_imgs(){ find "$1" -type f -iregex '.*\.\(png\|jpg\|jpeg\)' | wc -l | tr -d ' '; }

pick(){ # $1=dir $2=K -> echo K random files
  if command -v shuf >/dev/null 2>&1; then
    find "$1" -type f -iregex '.*\.\(png\|jpg\|jpeg\)' | shuf | head -n "$2"
  else
    find "$1" -type f -iregex '.*\.\(png\|jpg\|jpeg\)' \
    | awk 'BEGIN{srand()} {printf "%f %s\n", rand(), $0}' \
    | sort -n | cut -d" " -f2- | head -n "$2"
  fi
}

mkpair(){ # $1=pair_id $2=real_src $3=fake_src $4=TAGLABEL
  local pid="$1" rp="$2" fp="$3" tag="$4"
  local rext="${rp##*.}" fext="${fp##*.}"
  # <-- filenames WITHOUT _real / _fake -->
  local real_out="$EVAL/real/pair_$(printf '%04d' "$pid")_${tag}.$rext"
  local fake_out="$EVAL/fake/pair_$(printf '%04d' "$pid")_${tag}.$fext"
  cp -f "$rp" "$real_out"
  cp -f "$fp" "$fake_out"
  # CSV: pair_id,source_tag,real_src,fake_src,eval_real,eval_fake
  printf 'pair_%04d,%s,%s,%s,%s,%s\n' \
    "$pid" "$tag" "$rp" "$fp" "$real_out" "$fake_out" >> "$CSV"
}

# ---------- Prep ----------
mkdir -p "$EVAL/real" "$EVAL/fake"
: > "$CSV"

# Capacities
adm_r=$(count_imgs "$ADM_REAL"); adm_f=$(count_imgs "$ADM_FAKE")
cd_r=$(count_imgs "$CD_REAL");   cd_f=$(count_imgs "$CD_FAKE")
ADM_CAP=$(( adm_r < adm_f ? adm_r : adm_f ))
CD_CAP=$(( cd_r < cd_f ? cd_r : cd_f ))

# Adjust requests
[ "$TAKE_ADM" -gt "$ADM_CAP" ] && { log "[WARN] ADM cap $ADM_CAP < req $TAKE_ADM -> using $ADM_CAP"; TAKE_ADM="$ADM_CAP"; }
[ "$TAKE_CD"  -gt "$CD_CAP"  ] && { log "[WARN] CD  cap $CD_CAP  < req $TAKE_CD  -> using $CD_CAP";  TAKE_CD="$CD_CAP"; }

SUM=$((TAKE_ADM + TAKE_CD))
if [ "$SUM" -lt "$TOTAL" ]; then
  REM=$((TOTAL - SUM))
  FREE_ADM=$((ADM_CAP - TAKE_ADM))
  FREE_CD=$((CD_CAP - TAKE_CD))
  if   [ "$FREE_ADM" -ge "$REM" ]; then TAKE_ADM=$((TAKE_ADM + REM))
  elif [ "$FREE_CD"  -ge "$REM" ]; then TAKE_CD=$((TAKE_CD + REM))
  else log "[ERROR] Not enough capacity (ADM=$ADM_CAP CD=$CD_CAP) for TOTAL=$TOTAL"; exit 1; fi
fi

log "[PLAN] TOTAL=$TOTAL (ADM=$TAKE_ADM, CD=$TAKE_CD)"

# ---------- Build ----------
PAIR=0

if [ "$TAKE_ADM" -gt 0 ]; then
  mapfile -t R_ADM < <(pick "$ADM_REAL" "$TAKE_ADM")
  mapfile -t F_ADM < <(pick "$ADM_FAKE" "$TAKE_ADM")
  for i in $(seq 0 $((TAKE_ADM-1))); do
    PAIR=$((PAIR+1)); mkpair "$PAIR" "${R_ADM[$i]}" "${F_ADM[$i]}" "$ADM_TAG_LABEL"
  done
fi

if [ "$TAKE_CD" -gt 0 ]; then
  mapfile -t R_CD < <(pick "$CD_REAL" "$TAKE_CD")
  mapfile -t F_CD < <(pick "$CD_FAKE" "$TAKE_CD")
  for i in $(seq 0 $((TAKE_CD-1))); do
    PAIR=$((PAIR+1)); mkpair "$PAIR" "${R_CD[$i]}" "${F_CD[$i]}" "$CD_TAG_LABEL"
  done
fi

log "[OK] Eval split: $EVAL/{real,fake}"
log "     CSV: $CSV"
log "Counts: real=$(find "$EVAL/real" -type f | wc -l | tr -d ' ') fake=$(find "$EVAL/fake" -type f | wc -l | tr -d ' ')"
