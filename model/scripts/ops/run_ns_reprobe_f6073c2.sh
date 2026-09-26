#!/usr/bin/env bash
# ops/run_ns_reprobe_f6073c2.sh — orchestrator-side driver for the T6-Re truth probe
# (scripts belong to the route-2 line: gen_ns_re_edp.py / finalize_ns_truth.py /
#  check_ns_re_to_stokes.py, delivered at pin f6073c2).  No training, no GPU.
#   bash run_ns_reprobe_f6073c2.sh check   restore FreeFEM, fetch+hash scripts, diff proof,
#                                          both self-checks (each must be able to go red)
#   bash run_ns_reprobe_f6073c2.sh run     emit the .edp, solve Re=1e-3/1/10/50, merge, gate
# Re-running is safe: nothing is deleted and every file is hash-checked before it lands.
set -uo pipefail

PIN=50bc8532e8b1bb9054c6f6479e6fc137017bfa0f
REPO=Ruler4396/pinn-platform-v4
WS="${WS:-/mnt/workspace/pinn-repro-2026}"
FFROOT="$WS/ffroot.tgz"
LOGD="$WS/out/ns_reprobe"
PHASE="${1:-check}"
mkdir -p "$LOGD" || exit 1
export PYTHONPATH="$WS/pylibs:${PYTHONPATH:-}"
LEVELS="1e-3 1 10 50"

declare -A EXPECT=(
  [model/scripts/gen_ns_re_edp.py]=5e3a3c7e52fb9f97e4fb699f6095a5881e21e664d37e57b6e10e529a61046a9c
  [model/scripts/finalize_ns_truth.py]=52b89fc8f4c6e0b06dc3a201b68b1f5baaaaf4aa9cece180062e21525b86dcd6
  [model/scripts/check_ns_re_to_stokes.py]=d78495865b0fee8c4fe6a9d91146600e4ccdc0a18852451a4cc8156bb4af2b2c
  [model/scripts/selftest_ns_re.py]=8f7028ccda7785768808ca6ce13ec276cc58f4c73bb6fa0fed8261dae2f0423f
)

log() { printf '%s | %s\n' "$(date '+%H:%M:%S')" "$*"; }

fetch_one() {
  local p="$1" tmp m got
  tmp="$LOGD/$(basename "$1").part"
  mkdir -p "$(dirname "$WS/$1")" || return 1
  for m in "https://gh-proxy.com/https://raw.githubusercontent.com" "https://raw.githubusercontent.com"; do
    if curl -fsSL --max-time 90 --retry 2 -o "$tmp" "$m/$REPO/$PIN/$p"; then
      got="$(sha256sum "$tmp" | cut -d' ' -f1)"
      if [ "$got" = "${EXPECT[$p]}" ]; then
        [ -f "$WS/$1" ] && ! sha256sum "$WS/$1" | grep -q "^$got" && cp -p "$WS/$1" "$WS/$1.bak_pre_ns"
        mv "$tmp" "$WS/$1"; log "FETCH ok $p"; return 0
      fi
      log "FETCH badhash $p got=$got"
    else
      log "FETCH netfail $p via $m"
    fi
  done
  rm -f "$tmp"; return 1
}

step() {
  local name="$1" fatal="$2"; shift 2
  local t0 rc out
  t0=$(date +%s); out="$LOGD/step_${name}.txt"
  bash -c "$*" >"$out" 2>&1; rc=$?
  log "STEP $name rc=$rc wall=$(( $(date +%s) - t0 ))s fatal=$fatal"
  if [ "$rc" != 0 ]; then
    tail -16 "$out" | sed "s/^/    E| /"
    [ "$fatal" = fatal ] && { log "ABORT at $name (see $out)"; exit 1; }
  else
    tail -6 "$out" | sed "s/^/    > /"
  fi
  return 0
}

cd "$WS" || exit 1
log "PHASE=$PHASE PIN=$PIN"

# FreeFEM comes from the NAS-persisted archive; -nw is the only flag this build tolerates.
step ff_restore nonfatal "command -v FreeFem++ || { tar xzf $FFROOT -C /; }; command -v FreeFem++ && FreeFem++ -V 2>&1 | head -2 || FreeFem++ --version 2>&1 | head -2"
if ! command -v FreeFem++ >/dev/null 2>&1; then
  log "FreeFem++ still absent after restore attempt"; cat "$LOGD/step_ff_restore.txt" | tail -8
fi
log "delivering three scripts at $PIN"
_rc=0; for p in "${!EXPECT[@]}"; do fetch_one "$p" || _rc=1; done
[ "$_rc" != 0 ] && { log "ABORT at deliver (hash/network)"; exit 1; }

step gen_diff_proof fatal "python3 model/scripts/gen_ns_re_edp.py"
step ns_syntax_gate fatal "python3 model/scripts/gen_ns_re_edp.py --check-syntax"
step merger_selftest fatal "python3 model/scripts/selftest_ns_re.py"
step finalize_selfcheck fatal "python3 model/scripts/finalize_ns_truth.py --selfcheck"
step gate_selfcheck fatal "python3 model/scripts/check_ns_re_to_stokes.py --selfcheck"

if [ "$PHASE" = check ]; then
  step gen_list_diffs nonfatal "python3 model/scripts/gen_ns_re_edp.py --list-differences"
  log "CHECK_DONE"
  exit 0
fi

if [ "$PHASE" = run ]; then
  step gen_write fatal "python3 model/scripts/gen_ns_re_edp.py --write"
  step list_edp fatal "find model/cases/contraction_2d/cfd -maxdepth 2 -name 'C-base_ns_re*.edp' | sort"
  for L in $LEVELS; do
    d="model/cases/contraction_2d/cfd/C-base_ns_re${L}"
    f="$d/C-base_ns_re${L}.edp"
    step "solve_re_${L}" nonfatal "test -f $f || { echo 'MISSING $f'; exit 1; }; cd $d && nice -n 10 FreeFem++ -nw C-base_ns_re${L}.edp"
    step "merge_${L}" nonfatal "python3 model/scripts/finalize_ns_truth.py --level $L"
  done
  step gate fatal "python3 model/scripts/check_ns_re_to_stokes.py"
  step cost_table nonfatal "grep -aH 'NS it=\|du2=\|NOT CONVERGED' model/cases/contraction_2d/cfd/C-base_ns_re*/*.log 2>/dev/null | tail -40; ls -l model/cases/contraction_2d/cfd/C-base_ns_re*/*_raw.csv 2>/dev/null"
  log "RUN_DONE"
  exit 0
fi

log "unknown phase $PHASE"; exit 2
