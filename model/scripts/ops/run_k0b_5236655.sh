#!/usr/bin/env bash
# ops/run_k0b_5236655.sh -- route-2 driver for the K0b 12-digit truth reference (WP-A item 3).
# Scripts belong to the route-2 line; every hash in EXPECT is measured at one pin: 523665574319a3cb575b6cc25790fd0254c53e7a.
# No training, no GPU, no torch: the trip decides whether the REFERENCE resolves the quantity.
# Budget: ${BUDGET_S} (default 180 s). Past that it stops and reports what landed -- it does not
# extend itself.
#
#   bash run_k0b_5236655.sh check   fetch + hash the five scripts, run the gates that can go red
#   bash run_k0b_5236655.sh smoke   run the 1.5 s syntax probe (does v4.9 EXECUTE `floor`?), then
#                                   solve the coarsest level and prove the 8 companions land and
#                                   the reader merges them
#   bash run_k0b_5236655.sh run     the four S1 levels at 12 digits + the truth-side K0b scan
#
# Nothing is deleted and every file is hash-checked before it lands. A 6-digit artefact cannot
# pass as the K0b reference: `--require-12-digit-truth` makes the reader halt instead.
set -uo pipefail

FULL_PIN=523665574319a3cb575b6cc25790fd0254c53e7a
REPO=Ruler4396/pinn-platform-v4
WS="${WS:-/mnt/workspace/pinn-repro-2026}"
FFROOT="$WS/ffroot.tgz"
LOGD="$WS/out/k0b_5236655"
PHASE="${1:-check}"
BUDGET_S="${BUDGET_S:-180}"
CASE="${CASE:-TB-base}"
T_START=$(date +%s)
mkdir -p "$LOGD" || exit 1
export PYTHONPATH="$WS/pylibs:${PYTHONPATH:-}"
LEVELS_SMOKE="h1"
LEVELS_FULL="h1,h2,h3,hgrade"
LEVELS_SCAN="h1,hgrade"

declare -A EXPECT=(
  [model/scripts/route2/artifacts.py]=1febd1e1dc89ae66c8242483e8dab6f33509fea09d979617c6e96c165f83c35d
  [model/scripts/route2/residual_scorers.py]=86c96b1cbf9c6bcefaeca6e984a303a8e8c9a60029e5377a9369a93665bf8745
  [model/scripts/route2/t_geometry.py]=94329e67f178b7dfed18f40b897d8b0d82f0cb0a161a57374155fcf86857742c
  [model/scripts/route2/generate_t_case.py]=b31bf371cddeb444af6cf057aa2f2ef57fee7631e268208735f2136430370029
  [model/scripts/route2/k0_truth_gate.py]=bdfdc971e65c2ebe0da89ab6e628301c85b9f978218dc3112739842b2221f00e
  [model/cases/contraction_2d/cfd/C-base_ns_re1/probe_syntax.edp]=a4ca809f0b05b932d76a76d8e7d2d87dcefb1174c3cf0817eb5603c579f6bb25
)
R2="$WS/model/scripts/route2"

log() { printf '%s | %s\n' "$(date '+%H:%M:%S')" "$*"; }

over_budget() {
  local used=$(( $(date +%s) - T_START ))
  if [ "$used" -gt "$BUDGET_S" ]; then
    log "BUDGET: ${used}s > ${BUDGET_S}s -- stopping, not extending (what landed is in $LOGD)"
    return 0
  fi
  return 1
}

fetch_one() {
  local p="$1" tmp m got want
  want="${EXPECT[$p]}"
  tmp="$LOGD/$(basename "$p").part"
  mkdir -p "$(dirname "$WS/$p")" || return 1
  for m in "https://gh-proxy.com/https://raw.githubusercontent.com" "https://raw.githubusercontent.com"; do
    if curl -fsSL --max-time 90 --retry 2 -o "$tmp" "$m/$REPO/$FULL_PIN/$p"; then
      got="$(sha256sum "$tmp" | cut -d' ' -f1)"
      if [ "$got" = "$want" ]; then
        mv "$tmp" "$WS/$p"; log "FETCH ok $(basename "$p")"; return 0
      fi
      log "FETCH badhash $(basename "$p") got=${got:0:16} want=${want:0:16}"
    else
      log "FETCH netfail $(basename "$p") via $m"
    fi
  done
  rm -f "$tmp"; return 1
}

step() {
  local name="$1" fatal="$2" limit="$3"; shift 3
  local t0 rc out
  t0=$(date +%s); out="$LOGD/${name}.txt"
  timeout "$limit" bash -c "$*" >"$out" 2>&1; rc=$?
  log "STEP $name rc=$rc wall=$(( $(date +%s) - t0 ))s limit=${limit}s fatal=$fatal"
  if [ "$rc" != 0 ]; then
    tail -14 "$out" | sed "s/^/    E| /"
    if [ "$fatal" = fatal ]; then log "ABORT at $name (full log: $out)"; exit 1; fi
  else
    tail -5 "$out" | sed "s/^/    > /"
  fi
  return 0
}

expect_red() {         # a command that MUST fail: passing is the red case
  local name="$1"; shift
  local out="$LOGD/${name}.txt"
  bash -c "$*" >"$out" 2>&1; local rc=$?
  if [ "$rc" = 0 ]; then
    log "CONTROL $name rc=0 -- it should have REFUSED; tail:"; tail -6 "$out" | sed "s/^/    E| /"
    exit 1
  fi
  log "CONTROL $name refused as required (rc=$rc): $(grep -m1 -iE 'refus|refus|< 12|SystemExit' "$out" | cut -c1-96)"
}

restore_ff() {
  if ! command -v FreeFem++ >/dev/null 2>&1; then
    [ -f "$FFROOT" ] || { log "ABORT: FreeFem++ absent and no $FFROOT -- the container disk is not persistent, re-install first"; exit 1; }
    tar xzf "$FFROOT" -C / 2>/dev/null || true
    ldconfig 2>/dev/null || true
  fi
  command -v FreeFem++ >/dev/null 2>&1 || { log "ABORT: FreeFem++ still not on PATH"; exit 1; }
  log "FreeFem++ resolved: $(command -v FreeFem++)"
}

case "$PHASE" in
  check)
    for p in "${!EXPECT[@]}"; do fetch_one "$p" || { log "ABORT: fetch failed for $p"; exit 1; }; done
    step emission fatal 90 "cd '$R2' && python3 generate_t_case.py --selfcheck-emission"
    expect_red digits_refused "cd '$R2' && python3 generate_t_case.py --truth-digits 6 --dry-run --out-root '$LOGD/digits'"
    step suite fatal 300 "cd '$R2' && python3 selftest_route2_stdlib.py --json '$LOGD/route2_selftest.json' --cases TB-base"
    log "CHECK done: emission controls, the 6-digit refusal and the stdlib suite all had to bite."
    ;;
  smoke)
    restore_ff
    PROBE="$WS/model/cases/contraction_2d/cfd/C-base_ns_re1/probe_syntax.edp"
    [ -f "$PROBE" ] || { log "ABORT: probe $PROBE missing"; exit 1; }
    # `floor` has only ever been PARSED on v4.9, never executed. The whole staged reference rests
    # on it, so this 1.5 s probe is a precondition, not a courtesy.
    step probe_floor fatal 90 "cd '$(dirname "$PROBE")' && FreeFem++ -nw '$(basename "$PROBE")'"
    grep -q "PROBE OK" "$LOGD/probe_floor.txt" || { log "ABORT: no PROBE OK -- do not emit a staged truth on an unproven floor"; exit 1; }
    log "probe: floor executed on v4.9 (see $LOGD/probe_floor.txt)"
    step solve fatal 150 "cd '$R2' && python3 generate_t_case.py --case '$CASE' --levels '$LEVELS_SMOKE' --out-root '$LOGD/smoke'"
    n=$(find "$LOGD/smoke" -name "*${LEVELS_SMOKE}*${LEVELS_SMOKE}_samples_*_staged.csv" 2>/dev/null | wc -l)
    [ "$n" -ge 8 ] || n=$(find "$LOGD/smoke" -name "*_staged.csv" | wc -l)
    log "companions written: $n (want 8 for one level)"
    [ "$n" -ge 8 ] || { log "ABORT: $n companions, expected 8 -- the staged emission did not run"; exit 1; }
    step readback fatal 150 "cd '$R2' && python3 k0_truth_gate.py --case '$CASE' --case-root '$LOGD/smoke' --level '$LEVELS_SMOKE' --require-12-digit-truth --k0b-truth-scan '$LEVELS_SMOKE'"
    ;;
  run)
    restore_ff
    over_budget && exit 3
    step full fatal 300 "cd '$R2' && python3 generate_t_case.py --case '$CASE' --levels '$LEVELS_FULL' --out-root '$LOGD/k0b'"
    over_budget && log "WARN: budget spent during the levels -- reporting what landed"
    find "$LOGD/k0b" -name "*_staged.csv" | sort | while read -r f; do
      printf '  %-58s %s %sB\n' "$(basename "$f")" "$(sha256sum "$f" | cut -c1-16)" "$(stat -c%s "$f")"
    done | tee "$LOGD/hashes.txt"
    step scan fatal 240 "cd '$R2' && python3 k0_truth_gate.py --case '$CASE' --case-root '$LOGD/k0b' --require-12-digit-truth --k0b-truth-scan '$LEVELS_SCAN'"
    log "scan json: $LOGD/k0b/data/$CASE/k0b_truth_scan.json (no RESOLVED_* label is produced here -- that needs the model-side chain, which is not run)"
    ;;
  *) echo "usage: $0 {check|smoke|run}"; exit 2 ;;
esac

log "wall total $(( $(date +%s) - T_START ))s (budget ${BUDGET_S}s); logs in $LOGD"
date
