#!/usr/bin/env bash
# ops/run_ns3p_stage01.sh -- driver for Stage 0/1 of the finite-Reynolds (Navier-Stokes)
# extension of the contraction-family PINN line.
#
#   check : environment, FreeFEM restore, deliver + hash-verify the truth tooling,
#           then the pure-python proofs (each of them must be able to go red)
#   probe : run probe_syntax.edp -- the four constructs no local machine can answer for
#   solve : solve C-base at Re = 1e-3 / 1 / 10 / 50, per-process wall clock (date +%s%N)
#   merge : merge hi/lo per level, assert the printing floor, run the Re->0 gate
#   all   : check -> probe -> solve -> merge
#
# Nothing is deleted. Every fetched file is hash-checked before it lands, and any file
# already in place is moved aside with a .bak_pre_ns3p suffix rather than overwritten.
# Exit codes: 0 ok, 1 a gate failed, 90 environment unavailable (INDETERMINATE, not a fail).
set -uo pipefail

WS="${WS:-/mnt/workspace/pinn-repro-2026}"
OPS="${OPS:-/mnt/workspace/_ops/ns3p/20261003}"
REPO=Ruler4396/pinn-platform-v4
PIN_TRUTH=26c88ac13c6a49f74f746a30148cfc790a78b4e0
PIN_MINE=13a10e4af4285d97a7570d4eb2964c91bd524373
CFD="$WS/model/cases/contraction_2d/cfd"
LOGD="$OPS/out"
LEVELS="1e-3 1 10 50"
mkdir -p "$LOGD" "$LOGD/pre_backup"
export PYTHONPATH="$WS/pylibs:${PYTHONPATH:-}"

# path -> "<pin> <sha256>"
fetch_spec() {
  case "$1" in
    model/scripts/gen_ns_re_edp.py)              echo "$PIN_TRUTH 530a74544dc046d75f6706478011e013425c1766636285c9fba771d9b57e4e4b";;
    model/scripts/finalize_ns_truth.py)          echo "$PIN_TRUTH adc427cf87bc8995c780b708fa9b134d737d2ba95d2fa24b70cce37def293b4d";;
    model/scripts/check_ns_re_to_stokes.py)      echo "$PIN_TRUTH 912f17e0a8088ff0ccfb01cbb938823aa1ff642c10723d2fa5ff37d7d40b9336";;
    model/scripts/selftest_ns_re.py)             echo "$PIN_TRUTH e7c744f530ee68a35be1d40b8f7519c1bd171746d1bfbb0a2b3e950de37bf609";;
    model/scripts/train_velocity_pressure_independent_ns.py) echo "$PIN_MINE 08a32c4c64c0dd25805bae7c97e7b6842b6b68e1873fd7684dad4dce443f79ae";;
    model/scripts/make_ns_case.py)               echo "$PIN_MINE edc6317126450b24b601ff944efc89c64fa777016ba5dbd695c2f10cd9ef262c";;
    model/cases/contraction_2d/cfd/C-base_ns_re1e-3/C-base_ns_re1e-3.edp) echo "$PIN_TRUTH b8a8039bfadb03c2295fdb4f7df3d8e58f4ad32f9f1f589642dbe101f52377e5";;
    model/cases/contraction_2d/cfd/C-base_ns_re1/C-base_ns_re1.edp)       echo "$PIN_TRUTH 660fbfe426c847f357a5b7f4031d168e266b901f49e222a18b21fa48f0d25392";;
    model/cases/contraction_2d/cfd/C-base_ns_re10/C-base_ns_re10.edp)     echo "$PIN_TRUTH 8aeae887b3a28128606779b4e1e92d355219336ac899f61146754cd82e0e8999";;
    model/cases/contraction_2d/cfd/C-base_ns_re50/C-base_ns_re50.edp)     echo "$PIN_TRUTH 6392daa76ce426b9bb2b28fd7104a7503342b4d3b71ca7860ada23d94207a01e";;
    model/cases/contraction_2d/cfd/C-base_ns_re1/probe_syntax.edp)        echo "$PIN_TRUTH a4ca809f0b05b932d76a76d8e7d2d87dcefb1174c3cf0817eb5603c579f6bb25";;
    model/cases/contraction_2d/cfd/C-base/C-base_raw.csv)                echo "$PIN_TRUTH 46bd0401cf0f92f571a0419c09dc184688176f306241dcb3f010c82cab2b1aee";;
    *) return 1;;
  esac
}

log() { printf '%s | %s\n' "$(date '+%H:%M:%S')" "$*"; }

fetch_one() {
  local p="$1" spec pin want tmp got cur m
  spec="$(fetch_spec "$p")" || { log "FETCH unknown $p"; return 1; }
  pin="${spec%% *}"; want="${spec##* }"
  tmp="$LOGD/$(basename "$p").part"
  cur="$(sha256sum "$WS/$p" 2>/dev/null | cut -d' ' -f1)"
  if [ "$cur" = "$want" ]; then log "FETCH already-in-place $p"; return 0; fi
  mkdir -p "$(dirname "$WS/$p")" || return 1
  for m in "https://raw.githubusercontent.com" "https://gh-proxy.com/https://raw.githubusercontent.com"; do
    if curl -fsSL --max-time 90 --retry 2 -o "$tmp" "$m/$REPO/$pin/$p"; then
      got="$(sha256sum "$tmp" | cut -d' ' -f1)"
      if [ "$got" = "$want" ]; then
        [ -n "$cur" ] && { cp -p "$WS/$p" "$LOGD/pre_backup/$(basename "$(dirname "$p")")__$(basename "$p").bak_pre_ns3p" 2>/dev/null; mv "$WS/$p" "$WS/$p.bak_pre_ns3p"; log "FETCH replaced $p (was ${cur:0:12})"; }
        mv "$tmp" "$WS/$p"; log "FETCH ok $p ${got:0:16}"; return 0
      fi
      log "FETCH badhash $p got=${got:0:16} want=${want:0:16}"
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
  bash -c "cd '$WS' && $*" >"$out" 2>&1; rc=$?
  log "STEP $name rc=$rc wall=$(( $(date +%s) - t0 ))s fatal=$fatal"
  if [ "$rc" != 0 ]; then
    tail -20 "$out" | sed "s/^/    E| /"
    [ "$fatal" = fatal ] && { log "ABORT at $name (see $out)"; exit 1; }
  else
    tail -8 "$out" | sed "s/^/    > /"
  fi
  return 0
}

phase_check() {
  step env nonfatal "python3 -V; python3 -c 'import torch,numpy,pandas;print(torch.__version__,numpy.__version__,pandas.__version__)'"
  step ff_restore nonfatal "command -v FreeFem++ || tar xzf $WS/ffroot.tgz -C / ; command -v FreeFem++ && FreeFem++ -V 2>&1 | head -2"
  if ! command -v FreeFem++ >/dev/null 2>&1; then
    log "FreeFem++ absent after restore attempt -> probe/solve are INDETERMINATE, not FAIL"
  fi
  local rc=0
  for p in model/scripts/gen_ns_re_edp.py model/scripts/finalize_ns_truth.py \
           model/scripts/check_ns_re_to_stokes.py model/scripts/selftest_ns_re.py \
           model/scripts/train_velocity_pressure_independent_ns.py model/scripts/make_ns_case.py \
           model/cases/contraction_2d/cfd/C-base/C-base_raw.csv \
           model/cases/contraction_2d/cfd/C-base_ns_re1e-3/C-base_ns_re1e-3.edp \
           model/cases/contraction_2d/cfd/C-base_ns_re1/C-base_ns_re1.edp \
           model/cases/contraction_2d/cfd/C-base_ns_re10/C-base_ns_re10.edp \
           model/cases/contraction_2d/cfd/C-base_ns_re50/C-base_ns_re50.edp \
           model/cases/contraction_2d/cfd/C-base_ns_re1/probe_syntax.edp; do
    fetch_one "$p" || rc=1
  done
  [ "$rc" != 0 ] && { log "ABORT at deliver (hash/network)"; exit 1; }
  step diff_proof fatal "python3 model/scripts/gen_ns_re_edp.py"
  step syntax_lint fatal "python3 model/scripts/gen_ns_re_edp.py --check-syntax"
  step merger_selftest fatal "python3 model/scripts/selftest_ns_re.py"
  step finalize_selfcheck fatal "python3 model/scripts/finalize_ns_truth.py --selfcheck"
  step gate_selfcheck fatal "python3 model/scripts/check_ns_re_to_stokes.py --selfcheck"
  step ns_selftest fatal "python3 model/scripts/train_velocity_pressure_independent_ns.py --ns-selftest"
  log "CHECK done. FreeFem++=$(command -v FreeFem++ || echo ABSENT)"
}

phase_probe() {
  command -v FreeFem++ >/dev/null 2>&1 || { log "INDETERMINATE: no FreeFem++"; exit 90; }
  local d="$CFD/C-base_ns_re1" out="$LOGD/probe.log" t0 t1
  t0=$(date +%s%N)
  ( cd "$d" && FreeFem++ -nw probe_syntax.edp ) >"$out" 2>&1
  local rc=$?
  t1=$(date +%s%N)
  log "PROBE rc=$rc wall_ms=$(( (t1 - t0) / 1000000 ))"
  tail -4 "$out" | sed "s/^/    P| /"
  # The success line is anchored: with -nw this build echoes the numbered source listing,
  # so an unanchored match on 'PROBE OK' also hits the cout statement inside the file.
  if [ "$rc" != 0 ] || ! grep -q "^PROBE OK" "$out"; then
    log "PROBE did not print its success line -> solve would be guessing at syntax"; exit 1
  fi
}

phase_solve() {
  command -v FreeFem++ >/dev/null 2>&1 || { log "INDETERMINATE: no FreeFem++"; exit 90; }
  local tsv="$LOGD/solve.tsv"
  [ -f "$tsv" ] || printf 'level\trc\twall_ms\tnot_converged\tpicard_iters\tlast_du2\tsha_raw+pair\trows_raw\n' > "$tsv"
  for L in $LEVELS; do
    local d="$CFD/C-base_ns_re$L" lg="$LOGD/run_${L}.log" t0 rc t1
    for f in "$d/C-base_ns_re${L}_raw.csv" "$d/C-base_ns_re${L}_raw_pair.csv"; do
      [ -f "$f" ] && cp -p "$f" "$LOGD/pre_backup/$(basename "$f").pre_solve_${L}"
    done
    t0=$(date +%s%N)
    ( cd "$d" && FreeFem++ -nw "C-base_ns_re$L.edp" ) >"$lg" 2>&1
    rc=$?
    t1=$(date +%s%N)
    local nc iters last rows sha1 sha2
    # Anchored at column 0 on purpose: `FreeFem++ -nw` echoes the numbered source listing,
    # so an unanchored match counts the code line and the header comment (measured: the
    # unanchored version reported not_converged=2 for a level that converged in 3).
    nc=$(grep -c "^NS NOT CONVERGED" "$lg")
    iters=$(grep -c "^NS it=" "$lg")
    last=$(grep "^NS it=" "$lg" | tail -1 | sed 's/.*du2=\([^ ]*\).*/\1/')
    rows=$(( $(wc -l < "$d/C-base_ns_re${L}_raw.csv") - 1 ))
    sha1=$(sha256sum "$d/C-base_ns_re${L}_raw.csv" 2>/dev/null | cut -c1-16)
    sha2=$(sha256sum "$d/C-base_ns_re${L}_raw_pair.csv" 2>/dev/null | cut -c1-16)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$L" "$rc" \
      "$(( (t1 - t0) / 1000000 ))" "$nc" "$iters" "$last" "$sha1$sha2" "$rows" >> "$tsv"
    log "SOLVE Re=$L rc=$rc wall_ms=$(( (t1 - t0) / 1000000 )) not_converged=$nc iters=$iters last_du2=$last rows=$rows"
    [ "$rc" != 0 ] && tail -6 "$lg" | sed "s/^/      E| /"
  done
  [ -f "$tsv" ] && cat "$tsv" | sed "s/^/    T| /"
}

phase_merge() {
  for L in $LEVELS; do
    step "merge_$L" nonfatal "python3 model/scripts/finalize_ns_truth.py --level $L"
  done
  step re0_gate fatal "python3 model/scripts/check_ns_re_to_stokes.py"
}

PH="${1:-check}"
log "PHASE=$PH WS=$WS OPS=$OPS"
case "$PH" in
  check) phase_check;;
  probe) phase_probe;;
  solve) phase_solve;;
  merge) phase_merge;;
  all)   phase_check; phase_probe; phase_solve; phase_merge;;
  *) log "usage: $0 check|probe|solve|merge|all"; exit 2;;
esac
log "SUMMARY phase=$PH ok=1 logs=$LOGD"
