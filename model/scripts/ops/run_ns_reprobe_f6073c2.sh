#!/usr/bin/env bash
# ops/run_ns_reprobe_f6073c2.sh — orchestrator-side driver for the T6-Re truth probe
# (scripts belong to the route-2 line: gen_ns_re_edp.py / finalize_ns_truth.py /
#  check_ns_re_to_stokes.py, delivered at pin f6073c2).  No training, no GPU.
#   bash run_ns_reprobe_f6073c2.sh check   restore FreeFEM, fetch+hash scripts, diff proof,
#                                          both self-checks (each must be able to go red)
#   bash run_ns_reprobe_f6073c2.sh run     emit the .edp, solve Re=1e-3/1/10/50, merge, gate
# Re-running is safe: nothing is deleted and every file is hash-checked before it lands.
set -uo pipefail

PIN=fe0b074a5b0e5e355a359de8ef04f5cf6817a8d9
REPO=Ruler4396/pinn-platform-v4
WS="${WS:-/mnt/workspace/pinn-repro-2026}"
FFROOT="$WS/ffroot.tgz"
LOGD="$WS/out/ns_reprobe"
PHASE="${1:-check}"
mkdir -p "$LOGD" || exit 1
export PYTHONPATH="$WS/pylibs:${PYTHONPATH:-}"
LEVELS="1e-3 1 10 50"

declare -A EXPECT=(
  [model/scripts/gen_ns_re_edp.py]=ab824d089a121b563467a316940288b0078a24bd9d40dedbf7fd2fa0aa84c407
  [model/scripts/finalize_ns_truth.py]=1af30ad14426fcdc6397948fc79eedcd0cdfa061a09193fcec61f6123fa456ae
  [model/scripts/check_ns_re_to_stokes.py]=912f17e0a8088ff0ccfb01cbb938823aa1ff642c10723d2fa5ff37d7d40b9336
  [model/scripts/selftest_ns_re.py]=e7c744f530ee68a35be1d40b8f7519c1bd171746d1bfbb0a2b3e950de37bf609
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
  # Probe FIRST, per route-2's order: at 21:12 it parsed clean and then died rc=8 on the
  # inherited /root/dev write path (fixed in fe0b074). A probe whose success signal is an error
  # code is not a probe, so require BOTH rc=0 and a non-empty raw csv.
  step probe_syntax fatal "cd model/cases/contraction_2d/cfd/C-base_ns_re1 && nice -n 10 FreeFem++ -nw probe_syntax.edp && test -s probe_syntax_raw.csv && echo PROBE_OK"
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
