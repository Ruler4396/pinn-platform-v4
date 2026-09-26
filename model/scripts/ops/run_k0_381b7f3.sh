#!/bin/bash
# ops/run_k0_381b7f3.sh — K0 rerun at pin 381b7f3 (defect ⑪ fixed on the producer side).
# Same shape as run_k0_e4f32a8.sh; three differences:
#   (1) new WANT hashes for k0_truth_gate.py / selftest, residual_scorers.py stays 86c96b1c… (D7);
#   (2) the selftest expectation is total=271 (the five new k0_chain_contract cases);
#   (3) it prints whether the guard actually exists in the fetched file, so a "green" run can
#       never be the old pin arriving under a new name.
# Verdict files are copied under a pin-specific name and only when rc=0.
set -u
PIN=381b7f3dc803500204af4f825fde06a94ff54326
BASE=https://gh-proxy.com/https://raw.githubusercontent.com/Ruler4396/pinn-platform-v4/$PIN/model/scripts/route2
D=/mnt/workspace/pinn-repro-2026/model/scripts/route2
W=/mnt/workspace/pinn-repro-2026/route2_out/s1_54f2
O=/mnt/workspace/pinn-repro-2026/out
L=$O/k0_381b.log
export PYTHONPATH=/mnt/workspace/pinn-repro-2026/pylibs
T00=$(date +%s)

declare -A WANT=(
  [k0_truth_gate.py]=61cd0d6e0a66f438
  [selftest_route2_stdlib.py]=b6857879eb46b7f2
  [residual_scorers.py]=86c96b1cbf9c6bce
)

get() {  # fetch, hash-check, only then replace
  local f=$1 got
  curl -fsS -m 90 -o "$D/nf.new" "$BASE/$f" || { echo "FETCH_NETFAIL $f" | tee -a "$L"; return 1; }
  got=$(sha256sum "$D/nf.new" | cut -c1-16)
  if [ "$got" != "${WANT[$f]}" ]; then
    echo "FETCH HASHMISMATCH $f got=$got want=${WANT[$f]}" | tee -a "$L"
    rm -f "$D/nf.new"; return 1
  fi
  mv -f "$D/nf.new" "$D/$f"; echo "FETCH ok $f $got" | tee -a "$L"
}

: > "$L"
{
  echo "=== k0 381b7f3 runner start $(date -u +%FT%TZ) pin=$PIN ==="
  cd "$D" || exit 9
  for f in residual_scorers.py k0_truth_gate.py selftest_route2_stdlib.py; do
    if [ "$f" = residual_scorers.py ]; then
      on=$(sha256sum residual_scorers.py | cut -c1-16)
      echo "RESIDUAL_ON_DISK $on (intentionally NOT refetched: D7 keeps this file untouched)"
      [ "$on" = "${WANT[$f]}" ] || { echo "ABORT residual_scorers drift on=$on want=${WANT[$f]}"; exit 1; }
    else
      get "$f" || { echo "ABORT at $f"; exit 1; }
    fi
  done
  echo "GUARD_PRESENT=$(grep -c 'def assert_chain_is_numeric' k0_truth_gate.py) SCAN_MOVES=$(grep -c 'second_scan\[plan\] = reference_resolution_scan' k0_truth_gate.py)"
  if [ "$(grep -c 'def assert_chain_is_numeric' k0_truth_gate.py)" != 1 ]; then echo "ABORT guard not in fetched file"; exit 1; fi
  echo "--- selftest (expect total=271 failed=0) ---"
  timeout 900 python3 selftest_route2_stdlib.py > /tmp/st271.txt 2>&1
  echo "ST_RC=$? PASS=$(grep -c '^\[PASS\]' /tmp/st271.txt) FAIL=$(grep -c '^\[FAIL\]' /tmp/st271.txt)"
  grep -E 'k0_chain_contract' /tmp/st271.txt | head -8
  tail -2 /tmp/st271.txt
  echo "--- K0 stdlib dry-run ---"
  timeout 600 python3 k0_truth_gate.py --dry-run --case-root "$W" > /tmp/k0d.txt 2>&1
  echo "K0DRY_RC=$?"; tail -6 /tmp/k0d.txt
  for LV in h3 hgrade; do
    echo "=== K0 real TB-base level=$LV plans=a,b ==="
    Tc=$(date +%s)
    timeout 2700 python3 k0_truth_gate.py --case TB-base --level "$LV" --case-root "$W" --plans a,b > "/tmp/k0_$LV.txt" 2>&1
    rc=$?
    echo "K0_${LV}_RC=$rc K0_${LV}_s=$(( $(date +%s) - Tc ))"
    grep -E 'truth|model_|balance|fd_step|momentum_mse|reference|Traceback|rror|verdict|FAIL|K0-' "/tmp/k0_$LV.txt" | head -34
    # Only copy the verdict when THIS run produced it; the gate writes k0_verdict.json at the end,
    # so after a crash the file on disk belongs to the previous pin.
    NEWF="$O/k0_381b7f3_verdict_$LV.json"
    if [ "$rc" = 0 ] && [ -f "$W/data/TB-base/k0_verdict.json" ]; then
      cp -f "$W/data/TB-base/k0_verdict.json" "$NEWF"
      echo "VERDICT_COPIED $LV mtime=$(stat -c %y "$NEWF" | cut -c1-19)"
    else
      echo "NO_VERDICT_FROM_THIS_RUN level=$LV rc=$rc (盘上的 k0_verdict.json 未采纳)"
      ls -l "$W/data/TB-base/k0_verdict.json" 2>&1 | cut -c1-70
      continue
    fi
    python3 - "$NEWF" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rr = d.get("reference_resolution") or {}
def fmt(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True)[:620]
print("  [%s] pass=%s failed=%s" % (d.get("level", "?"), d.get("pass"), fmt(d.get("failed"))))
print("  [%s] rule=%s" % (d.get("level", "?"), rr.get("rule")))
for k in ("truth_mse_step_scan", "chain_second_scan", "second_order_items"):
    print("  [%s] %s=%s" % (d.get("level", "?"), k, fmt(rr.get(k))))
# keys as published by k0_truth_gate.run_gate (verdict.update at the end of the gate): all top level
print("  [%s] model_momentum_mse=%s truth_momentum_mse=%s ratio=%s balance=%s" % (
    d.get("level", "?"),
    json.dumps(d.get("model_momentum_mse")),
    json.dumps(d.get("truth_momentum_mse")),
    json.dumps(d.get("ratio_truth_over_model")),
    json.dumps((d.get("checks") or {}).get("K0-S2_balance_ratio_in_band"))))
PY
  done
  echo "=== done $(date -u +%FT%TZ) elapsed_s=$(( $(date +%s) - T00 )) ==="
} >> "$L" 2>&1
tail -70 "$L"
