#!/bin/bash
# ops/run_k0_e4f32a8.sh — K0 rerun at pin e4f32a8 (the second-derivative referee + INDETERMINATE).
# Differs from run_k0_471228b.sh in two ways: (1) every fetched file must match its expected
# sha256 before it replaces the one on disk; (2) it reads back the new reference_resolution
# fields and model_momentum_mse, because those are what this pin's verdict turns on.
set -u
PIN=e4f32a8eea85a88f01ad31a905eab1e5c4cd7307
BASE=https://gh-proxy.com/https://raw.githubusercontent.com/Ruler4396/pinn-platform-v4/$PIN/model/scripts/route2
D=/mnt/workspace/pinn-repro-2026/model/scripts/route2
W=/mnt/workspace/pinn-repro-2026/route2_out/s1_54f2
O=/mnt/workspace/pinn-repro-2026/out
L=$O/k0_e4f3.log
export PYTHONPATH=/mnt/workspace/pinn-repro-2026/pylibs
T00=$(date +%s)

declare -A WANT=(
  [k0_truth_gate.py]=855190c1504a3487
  [selftest_route2_stdlib.py]=eccf141942cd3c46
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
  echo "=== k0 e4f32a8 runner start $(date -u +%FT%TZ) pin=$PIN ==="
  cd "$D" || exit 9
  for f in residual_scorers.py k0_truth_gate.py selftest_route2_stdlib.py; do
    if [ "$f" = residual_scorers.py ]; then
      echo "RESIDUAL_ON_DISK $(sha256sum residual_scorers.py | cut -c1-16) (intentionally NOT refetched: D7 keeps this file untouched)"
    else
      get "$f" || { echo "ABORT at $f"; exit 1; }
    fi
  done
  echo "--- selftest (expect total=266 failed=0) ---"
  timeout 900 python3 selftest_route2_stdlib.py > /tmp/st266.txt 2>&1
  echo "ST_RC=$? PASS=$(grep -c '^\[PASS\]' /tmp/st266.txt) FAIL=$(grep -c '^\[FAIL\]' /tmp/st266.txt)"
  tail -2 /tmp/st266.txt
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
    # Only copy the verdict when THIS run produced it. The gate writes k0_verdict.json at the
    # end, so after a crash the file on disk is the previous pin's — copying it unconditionally
    # (what run_k0_471228b.sh did) lets a stale verdict masquerade as a fresh reading.
    if [ "$rc" = 0 ] && [ -f "$W/data/TB-base/k0_verdict.json" ]; then
      cp -f "$W/data/TB-base/k0_verdict.json" "$O/k0_e4f32a8_verdict_$LV.json"
      echo "VERDICT_COPIED $LV mtime=$(stat -c %y "$O/k0_e4f32a8_verdict_$LV.json" | cut -c1-19)"
    else
      echo "NO_VERDICT_FROM_THIS_RUN level=$LV rc=$rc (盘上的 k0_verdict.json 未采纳，可能是上一枚留下的)"
      ls -l "$W/data/TB-base/k0_verdict.json" 2>&1 | cut -c1-70
      continue
    fi
    python3 - "$O/k0_e4f32a8_verdict_$LV.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rr = d.get("reference_resolution") or {}
def fmt(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True)[:520]
print("  [%s] rule=%s" % (d.get("level", "?"), rr.get("rule")))
for k in ("truth_mse_step_scan", "chain_second_scan", "second_order_items"):
    print("  [%s] %s=%s" % (d.get("level", "?"), k, fmt(rr.get(k))))
print("  [%s] model_momentum_mse=%s truth_momentum_mse=%s" % (
    d.get("level", "?"),
    json.dumps(d.get("model_momentum_mse")),
    json.dumps((d.get("truth_score") or {}).get("momentum_mse"))))
PY
  done
  echo "=== done $(date -u +%FT%TZ) elapsed_s=$(( $(date +%s) - T00 )) ==="
} >> "$L" 2>&1
tail -60 "$L"
