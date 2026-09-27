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

# The caller's pin wins: run_coldtrip_one.sh fetches this driver AND the same python files
# from its own table, so a private pin here could re-download an OLDER generate_t_case.py
# over the parent's newer bytes.  Measured on dsw-2213920 at 14:52: the solve aborted on a
# binary the 1.5 s probe had just used successfully (parent table pin 70317a5, driver pin
# 5236655) -- two nested tables must not be allowed to disagree, so the child inherits.
FULL_PIN="${FULL_PIN:-70317a5965fb342ed5212fac71b5f6b20334eb5f}"
REPO=Ruler4396/pinn-platform-v4
WS="${WS:-/mnt/workspace/pinn-repro-2026}"
FFROOT="$WS/ffroot.tgz"
LOGD="$WS/out/k0b_5236655"
SUITED="${SUITED:-/mnt/workspace/route2_selftest}"   # must be OUTSIDE the repo checkout
PHASE="${1:-check}"
BUDGET_S="${BUDGET_S:-180}"
CASE="${CASE:-TB-base}"
SCALE_SRC="${SCALE_SRC:-$WS/route2_out}"   # read-only 6-digit S1 truth; never the output tree
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
  [model/scripts/route2/generate_t_case.py]=4a15997314ae6c71a4813e93181bb454f7f7ec7f776cf329a4c286fdb7f821ae
  [model/scripts/route2/k0_truth_gate.py]=bdfdc971e65c2ebe0da89ab6e628301c85b9f978218dc3112739842b2221f00e
  [model/cases/contraction_2d/cfd/C-base_ns_re1/probe_syntax.edp]=a4ca809f0b05b932d76a76d8e7d2d87dcefb1174c3cf0817eb5603c579f6bb25
)
R2="$WS/model/scripts/route2"

# ---------------------------------------------------------------------------------------------
# The staging widths follow from the field magnitudes, so they are MEASURED, never guessed: the
# emitter refuses an empty --truth-scales with no prior samples to read (that refusal is what
# stopped the 15:03 smoke on this box, correctly -- but it means the first 12-digit run has to be
# told where to measure).  SCALE_SRC is a read-only tree of the archived 6-digit S1 truth; the
# per-level numbers are printed so the spread stays visible, and the plan uses the conservative
# max across levels (a hi that is printable for every level).
measure_trip_scales() {
  local src="$1" out
  out=$(cd "$R2" && python3 - "$src" <<'PY'
import csv, sys, pathlib
root = pathlib.Path(sys.argv[1])
want = ("x_star", "y_star", "u_star", "v_star", "p_star")
per = {}
for f in sorted(root.glob("**/*_samples_*.csv")):
    lv = f.name.split("_samples_")[0]
    with f.open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        if not rd.fieldnames or not all(w in rd.fieldnames for w in want):
            continue
        m = per.setdefault(lv, {w: 0.0 for w in want})
        for row in rd:
            for w in want:
                try:
                    v = abs(float(row[w]))
                except (TypeError, ValueError):
                    continue
                if v > m[w]:
                    m[w] = v
if not per:
    print("NO_SOURCE")
    raise SystemExit(3)
worst = {w: max(m[w] for m in per.values()) for w in want}
for k in sorted(per):
    print("PER_LEVEL %s %s" % (k, " ".join("%s=%.6g" % (w, per[k][w]) for w in want)))
print("SCALES " + ",".join("%.6g" % worst[w] for w in want))
PY
)
  local rc=$?
  printf '%s\n' "$out" > "$LOGD/scales.txt"
  if [ "$rc" != 0 ] || printf '%s' "$out" | grep -q NO_SOURCE; then
    echo "NO_SOURCE"
    return 1
  fi
  printf '%s' "$out" | awk '/^SCALES /{print $2}'
}

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
  # see run_coldtrip_one.sh fetch(): a verified file on disk is not worth another mirror round trip
  if [ -f "$WS/$p" ] && [ "$(sha256sum "$WS/$p" | cut -c1-16)" = "${want:0:16}" ]; then
    log "SKIP $p already matches the pinned hash (no download)"; return 0
  fi
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

# HARD CONSTRAINT (orchestrator, 9/27 14:2x): no extraction into `/`, no `cp -a` into /usr, no
# `ldconfig` -- the previous instance was replaced after exactly that recovery.  The solver must
# come from PATH already, or from run_coldtrip_one.sh's instance-local prefix via FFBIN.
FFHOME="${FFHOME:-$WS/ffrun}"
FFBIN="${FFBIN:-}"
restore_ff() {
  if [ -n "$FFBIN" ] && [ -x "$FFBIN" ]; then
    export FREEFEM_BIN="$FFBIN"
    log "FreeFem++ supplied by the caller: $FFBIN -> FREEFEM_BIN exported (LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset})"
    return 0
  fi
  if command -v FreeFem++ >/dev/null 2>&1; then
    FFBIN=$(command -v FreeFem++); export FREEFEM_BIN="$FFBIN"
    log "FreeFem++ on PATH: $FFBIN (FREEFEM_BIN exported so the python steps use the same binary)"; return 0
  fi
  local cand
  for cand in "$FFHOME/usr/bin/FreeFem++" "$FFHOME/bin/FreeFem++"; do
    if [ -x "$cand" ]; then
      FFBIN="$cand"
      local d; d=$(find "$FFHOME" -name '*.so*' -type f 2>/dev/null | sed 's|/[^/]*$||' | sort -u | tr '
' ':')
      export LD_LIBRARY_PATH="${d}${LD_LIBRARY_PATH:-}"
      export FREEFEM_BIN="$FFBIN"
      log "FreeFem++ from the instance-local prefix: $FFBIN (LD_LIBRARY_PATH + FREEFEM_BIN for this run only)"
      return 0
    fi
  done
  log "ABORT: no FreeFem++ on PATH and none under $FFHOME. This driver will NOT unpack into / or run ldconfig (the recovery that cost the last instance). Run run_coldtrip_one.sh preflight/restore first."
  exit 1
}

case "$PHASE" in
  check)
    for p in "${!EXPECT[@]}"; do fetch_one "$p" || { log "ABORT: fetch failed for $p"; exit 1; }; done
    step emission fatal 90 "cd '$R2' && python3 generate_t_case.py --selfcheck-emission"
    expect_red digits_refused "cd '$R2' && python3 generate_t_case.py --truth-digits 6 --dry-run --out-root '$LOGD/digits'"
    # TWO fixes found by running this on dsw-2213920 at 14:43:
    #  1. $LOGD lives under $WS, and $WS is a repo checkout on the instance, so the suite's own
    #     guard ("refusing to write self-test output inside the repo") correctly killed it.  The
    #     evidence directory must be outside the checkout.
    #  2. `--cases TB-base` made `s2_adversary_geometry_is_the_asymmetric_one` fail by
    #     construction (it reads the TB-asym summary), so it was a harness artefact, not a finding.
    mkdir -p "$SUITED"
    step suite fatal 300 "cd '$R2' && python3 selftest_route2_stdlib.py --json '$SUITED/route2_selftest.json'"
    log "CHECK done: emission controls, the 6-digit refusal and the stdlib suite all had to bite."
    ;;
  smoke)
    restore_ff
    PROBE="$WS/model/cases/contraction_2d/cfd/C-base_ns_re1/probe_syntax.edp"
    [ -f "$PROBE" ] || { log "ABORT: probe $PROBE missing"; exit 1; }
    # `floor` has only ever been PARSED on v4.9, never executed. The whole staged reference rests
    # on it, so this 1.5 s probe is a precondition, not a courtesy.
    step probe_floor fatal 90 "cd '$(dirname "$PROBE")' && "$FFBIN" -nw '$(basename "$PROBE")'"
    grep -q "PROBE OK" "$LOGD/probe_floor.txt" || { log "ABORT: no PROBE OK -- do not emit a staged truth on an unproven floor"; exit 1; }
    log "probe: floor executed on v4.9 (see $LOGD/probe_floor.txt)"
    SCALES=$(measure_trip_scales "$SCALE_SRC") || { log "ABORT: no archived *_samples_*.csv under $SCALE_SRC to measure -- do not guess a staging width"; exit 1; }
    log "measured scales (x,y,u,v,p) = $SCALES from $SCALE_SRC (per-level detail: $LOGD/scales.txt)"
    step solve fatal 150 "cd '$R2' && python3 generate_t_case.py --case '$CASE' --levels '$LEVELS_SMOKE' --truth-scales '$SCALES' --out-root '$LOGD/smoke'"
    n=$(find "$LOGD/smoke" -name "*${LEVELS_SMOKE}*${LEVELS_SMOKE}_samples_*_staged.csv" 2>/dev/null | wc -l)
    [ "$n" -ge 8 ] || n=$(find "$LOGD/smoke" -name "*_staged.csv" | wc -l)
    log "companions written: $n (want 8 for one level)"
    [ "$n" -ge 8 ] || { log "ABORT: $n companions, expected 8 -- the staged emission did not run"; exit 1; }
    step readback fatal 150 "cd '$R2' && python3 k0_truth_gate.py --case '$CASE' --case-root '$LOGD/smoke' --level '$LEVELS_SMOKE' --require-12-digit-truth --k0b-truth-scan '$LEVELS_SMOKE'"
    ;;
  run)
    restore_ff
    over_budget && exit 3
    SCALES=$(measure_trip_scales "$SCALE_SRC") || { log "ABORT: cannot measure scales from $SCALE_SRC before the full emission"; exit 1; }
    log "full-run scales = $SCALES (conservative max across the levels found under $SCALE_SRC)"
    step full fatal 300 "cd '$R2' && python3 generate_t_case.py --case '$CASE' --levels '$LEVELS_FULL' --truth-scales '$SCALES' --out-root '$LOGD/k0b'"
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
