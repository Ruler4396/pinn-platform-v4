#!/bin/bash
L=/mnt/workspace/pinn-repro-2026/out/k0_4712.log
D=/mnt/workspace/pinn-repro-2026/model/scripts/route2
W=/mnt/workspace/pinn-repro-2026/route2_out/s1_54f2
O=/mnt/workspace/pinn-repro-2026/out
U=https://gh-proxy.com/https://raw.githubusercontent.com/Ruler4396/pinn-platform-v4/471228b5b87c6339eaa40e73e140af410f470b3d/model/scripts/route2
export PYTHONPATH=/mnt/workspace/pinn-repro-2026/pylibs
T00=$(date +%s)
{
echo "=== k0 4712 runner start $(date -u +%FT%TZ) ==="
cd "$D" || exit 9
for f in k0_truth_gate.py selftest_route2_stdlib.py; do
  curl -fsS -m 60 -o nf.new "$U/$f" && mv -f nf.new "$f" && echo GOT_$f
done
sha256sum k0_truth_gate.py selftest_route2_stdlib.py | cut -c1-16,65-99
echo "--- selftest (expect 261/0) ---"
python3 selftest_route2_stdlib.py > /tmp/st261.txt 2>&1
echo "ST_RC=$? PASS=$(grep -c '^\[PASS\]' /tmp/st261.txt) FAIL=$(grep -c '^\[FAIL\]' /tmp/st261.txt)"
tail -2 /tmp/st261.txt
echo "--- inputs present? ---"
ls "$W"/data/TB-base/ 2>&1 | head -6
echo "--- K0 stdlib dry-run ---"
timeout 600 python3 k0_truth_gate.py --dry-run --case-root "$W" > /tmp/k0d.txt 2>&1
echo "K0DRY_RC=$?"
tail -8 /tmp/k0d.txt
for LV in h3 hgrade; do
  echo "=== K0 real TB-base level=$LV plans=a,b ==="
  Tc=$(date +%s)
  timeout 2700 python3 k0_truth_gate.py --case TB-base --level "$LV" --case-root "$W" --plans a,b > "/tmp/k0_$LV.txt" 2>&1
  rc=$?
  echo "K0_${LV}_RC=$rc K0_${LV}_s=$(( $(date +%s) - Tc ))"
  grep -E 'truth|model|balance|fd_step|momentum_mse|Traceback|Error|error|verdict|FAIL|PASS|K0-' "/tmp/k0_$LV.txt" | head -40
  cp -f "$W/data/TB-base/k0_verdict.json" "$O/k0_verdict_$LV.json" 2>/dev/null
done
echo "=== verdict summary (both levels) ==="
for LV in h3 hgrade; do
  echo "## $LV"
  tr ',' '\n' < "$O/k0_verdict_$LV.json" 2>/dev/null | grep -E '"check"|"pass"|"failed"|K0-' | head -40
done
echo "=== done $(date -u +%FT%TZ) elapsed_s=$(( $(date +%s) - T00 )) ==="
} > "$L" 2>&1
