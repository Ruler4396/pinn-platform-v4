#!/usr/bin/env bash
# ops/run_t4_baselines_b76c6dc.sh
# Orchestrator-side runner for 必做1 (baseline trio) + the T6 multi-point units.
#   bash run_t4_baselines_b76c6dc.sh check   -> deliver, verify hashes, gates, two 6-epoch smokes
#   bash run_t4_baselines_b76c6dc.sh full    -> arm A/B (seg 09), then T6 (seg 10) if obs verify passed
# Nothing here deletes output. Re-running is a resume: sweep_t5.sh skips runs that already
# have metrics.json, and every file is re-fetched and hash-checked against the pinned commit.
set -uo pipefail

PIN=680083f48ee32621debae8b8d705eba516d4e9ba
REPO=Ruler4396/pinn-platform-v4
WS="${WS:-/mnt/workspace/pinn-repro-2026}"
PHASE="${1:-check}"
LOGD="$WS/out/t4_b76c6dc"
mkdir -p "$LOGD" || exit 1

export PYTHONPATH="$WS/pylibs:${PYTHONPATH:-}"
export SWEEP_WS_ROOT="$WS"
export SWEEP_OUT_DIR="$WS/out"
# Unit prices are the medians read back out of out/runs_index.psv for this instance
# (contraction dense 81.4s, sparse ~53.7s, bend sparse ~116s), rounded UP on purpose:
# a budget guard must fail early, never late.
export UNIT_DENSE_TRAIN_SEC=85 UNIT_SPARSE_TRAIN_SEC=56 BEND_UNIT_SEC=118 UNIT_EVAL_SEC=2

declare -A EXPECT=(
  [model/scripts/sweep_lib.sh]=830aeec5b5c88dbbd7b045bb624699b3d1be076cdd2236de0768d044794d5448
  [model/scripts/sweep_t5.sh]=fabc15b442f6eebe2a2eca941548db867e4756512cd5914f8a1e84e9cc9960e7
  [model/scripts/train_joint_upnp_pin.py]=878006f3688ab480e5f52c179228c2c513a633e47a20cbc53bb5cafc0ace58d2
  [model/scripts/baselines_pod.py]=a1bc77cea2f5d24a34b1d8209de10e6e1f6212a87ec7560160e2e046c0381e14
  [model/scripts/selftest_ledger.sh]=d43360b2d2c5216c74720bc71ebedcb77b0ecc53b4c512400e9962ec7d4da4cc
  [model/scripts/analyze_sweep.py]=0883a1066050127882b60661cdca2538374ef6d85063b30be6f34e53b9b20c69
  [model/scripts/generate_observations_seeded.py]=0cac210ca0a69f858c7669a66e14d8a268b14636117af353458ad278dbd47d28
  [model/scripts/ops/build_runs_index.py]=9ec4c8fc9f6c867695ace375f74f1fc9b7e5b6502690dbb8b4b14a3d39a9e95b
)

log() { printf '%s | %s\n' "$(date '+%H:%M:%S')" "$*"; }

fetch_one() {
  local p="$1" tmp m got
  tmp="$LOGD/$(basename "$1").part"
  mkdir -p "$(dirname "$WS/$1")" || return 1
  for m in "https://gh-proxy.com/https://raw.githubusercontent.com" \
           "https://raw.githubusercontent.com" \
           "https://ghproxy.net/https://raw.githubusercontent.com"; do
    if curl -fsSL --max-time 90 --retry 2 -o "$tmp" "$m/$REPO/$PIN/$p"; then
      got="$(sha256sum "$tmp" | cut -d' ' -f1)"
      if [ "$got" = "${EXPECT[$p]}" ]; then
        if [ -f "$WS/$1" ] && ! sha256sum "$WS/$1" | grep -q "^$got"; then
          cp -p "$WS/$1" "$WS/$1.bak_pre_b76c6dc" 2>/dev/null || true
        fi
        mv "$tmp" "$WS/$1"
        log "FETCH ok $p via $m"
        return 0
      fi
      log "FETCH badhash $p got=$got want=${EXPECT[$p]}"
    else
      log "FETCH netfail $p via $m"
    fi
  done
  rm -f "$tmp"
  return 1
}

deliver() {
  local rc=0 p
  for p in "${!EXPECT[@]}"; do fetch_one "$p" || rc=1; done
  return $rc
}

step() {
  local name="$1" fatal="$2"; shift 2
  local t0 rc out
  t0=$(date +%s); out="$LOGD/step_${name}.txt"
  bash -c "$*" >"$out" 2>&1; rc=$?
  log "STEP $name rc=$rc wall=$(( $(date +%s) - t0 ))s fatal=$fatal"
  if [ "$rc" != 0 ]; then
    tail -14 "$out" | sed "s/^/    E| /"
    [ "$fatal" = fatal ] && { log "ABORT at $name (see $out)"; exit 1; }
  else
    tail -5 "$out" | sed "s/^/    > /"
  fi
  return 0
}

cd "$WS" || exit 1
log "PHASE=$PHASE PIN=$PIN WS=$WS RUNNER_SHA=$(sha256sum "$0" | cut -c1-16)"

step env nonfatal "python3 -c 'import sys,torch,numpy,pandas,scipy;print(\"py\",sys.version.split()[0]);print(\"torch\",torch.__version__);print(\"numpy\",numpy.__version__);print(\"pandas\",pandas.__version__);print(\"scipy\",scipy.__version__)'"
_t0=$(date +%s); deliver > "$LOGD/step_deliver.txt" 2>&1; _rc=$?
log "STEP deliver rc=$_rc wall=$(( $(date +%s) - _t0 ))s fetched=$(grep -c 'FETCH ok' "$LOGD/step_deliver.txt")/8"
grep -v 'FETCH ok' "$LOGD/step_deliver.txt" | tail -12 | sed 's/^/    D| /'
[ "$_rc" != 0 ] && { log "ABORT at deliver (hash or network)"; exit 1; }
step ledger_selftest "$([ "$PHASE" = full ] && echo fatal || echo nonfatal)" "bash model/scripts/selftest_ledger.sh"
step index_regression fatal "python3 model/scripts/ops/build_runs_index.py"

if [ "$PHASE" = check ]; then
  # Both dry-runs write to the SAME out/dryrun_argv.txt, so the baseline plan has to be
  # snapshotted before --t6 overwrites it. (First run of this script aborted exactly here:
  # the count gate read the T6 plan and said rc=1. The gate was right; the script was wrong.)
  step dryrun_baseline fatal "bash model/scripts/sweep_t5.sh --baseline --dry-run > $LOGD/base_plan.txt 2>&1 && cp out/dryrun_argv.txt $LOGD/argv_baseline.txt && test \$(wc -l < $LOGD/argv_baseline.txt) -eq 20"
  step dryrun_t6 fatal "bash model/scripts/sweep_t5.sh --t6 --dry-run > $LOGD/t6_plan.txt 2>&1 && grep -q '实际要训练=12' $LOGD/t6_plan.txt"
  step argv_shapes fatal "test \$(grep -c train_joint_upnp_pin.py $LOGD/argv_baseline.txt) -eq 10 -a \$(grep -c scripts/train_supervised.py $LOGD/argv_baseline.txt) -eq 10"
  step armA_paramratio fatal "sed 's/--run-name rev2609b_t5c20__s42__o0/--run-name smokeparam__s42__o0/' $LOGD/argv_baseline.txt | head -1 | sed 's|\$| --dry-run --require-param-ratio 1 --param-ratio-tol 0.05|' > $LOGD/param.sh; cd model && bash $LOGD/param.sh"
  # Arm C's documented first step crashes on this instance: baselines_pod.py:153 assigns
  # args.eval_cases/obs_files as LISTS, and :163/:164 then call parse_list() on them
  # ((text or "").strip() -> AttributeError). Only the --self-check branch is affected, so
  # every real arm-C run is still untested. Reported to the baseline line; kept non-fatal here
  # so the arm A/B smokes get measured in the same trip instead of one defect per round.
  step armC_selfcheck nonfatal "cd model && python3 scripts/baselines_pod.py --self-check"
  step obs_verify nonfatal "python3 model/scripts/generate_observations_seeded.py --family contraction_2d --obs-seeds 0 --verify-committed"
  step smoke_armA fatal "sed 's/--run-name rev2609b_t5c20__s42__o0/--run-name smoke_joint6__s42__o0/; s/--epochs 480/--epochs 6/' '$LOGD/argv_baseline.txt' | head -1 > $LOGD/smokeA.sh; cd model && bash $LOGD/smokeA.sh"
  step smoke_armB fatal "sed 's/--run-name rev2609b_t5c22__s42__o0/--run-name smoke_mlp6__s42__o0/; s/--max-epochs 2000/--max-epochs 6/' '$LOGD/argv_baseline.txt' | sed -n '11p' > $LOGD/smokeB.sh; cd model && bash $LOGD/smokeB.sh"
  step smoke_artifacts nonfatal "ls -l model/results/pinn/smoke_joint6__s42__o0/ model/results/supervised/smoke_mlp6__s42__o0/ 2>&1; echo '--- rel_l2 from whatever metrics landed:'; find model/results/pinn/smoke_joint6__s42__o0 model/results/supervised/smoke_mlp6__s42__o0 -name 'metrics*.json' -exec cat {} \; 2>/dev/null | tr -d '\\n' | cut -c1-600"
  log "CHECK_DONE"
  exit 0
fi

if [ "$PHASE" = train ]; then
  # 20 trainings do not depend on arm C, and arm C currently crashes on its own self-check
  # (baselines_pod.py:153 vs :163/:164). Running the long part with --no-pod means a 2-line
  # fix in someone else's file doesn't sit on 24 minutes of CPU. Arm C runs in phase armc.
  step seg09_nopod fatal "bash model/scripts/sweep_t5.sh --baseline --no-pod --seg 09 --budget-min 45"
  step index_after_train fatal "python3 model/scripts/ops/build_runs_index.py"
  log "TRAIN_DONE"
  exit 0
fi

if [ "$PHASE" = sparse ]; then
  # Arm A's sparse cell crashed on a point-set mismatch (fixed in 310b673: the boundary terms
  # take the dense forward pass and a mismatch now raises a named error). 5 seeds x ~139 s.
  step seg13_armA_sparse fatal "bash model/scripts/sweep_t5.sh --baseline --no-pod --only-cells 21 --seg 13 --budget-min 25"
  step index_after_sparse fatal "python3 model/scripts/ops/build_runs_index.py"
  log "SPARSE_DONE"
  exit 0
fi

if [ "$PHASE" = armc ]; then
  step armC_selfcheck fatal "cd model && python3 scripts/baselines_pod.py --self-check"
  step armC_real fatal "cd model && python3 scripts/baselines_pod.py --family contraction_2d --eval-cases C-val,C-test-1,C-test-2 --obs-files obs_sparse_5pct.csv --out $SWEEP_OUT_DIR/pod_baseline_contraction.json"
  log "ARMC_DONE"
  exit 0
fi

if [ "$PHASE" = full ]; then
  # Arm C first: 必做1 needs three arms, so a still-broken POD baseline must stop the trip
  # in the first seconds rather than after 20 trainings that can't be claimed anyway.
  step armC_pre fatal "cd model && python3 scripts/baselines_pod.py --self-check"
  step seg09_baseline fatal "bash model/scripts/sweep_t5.sh --baseline --seg 09 --budget-min 45"
  log "FULL_TRAIN_DONE"
fi

if [ "$PHASE" = evalextra ]; then
  # 44559b2 gives arm A a test-evaluation chain (--eval-test-cases). The 10 arm-A runs already
  # have metrics.json, so the sweep takes the [skip-train] path and only produces evaluations:
  # this is a backfill, not a retrain.
  step seg14_armA_testeval fatal "bash model/scripts/sweep_t5.sh --baseline --no-pod --only-cells 20,21 --seg 14 --budget-min 15"
  step analyze_full nonfatal "python3 model/scripts/analyze_sweep.py --split test --metric rel_l2_speed --paired t5c04,t5c08 --paired t5c22,t5c01 --paired t5c23,t5c04 --paired t5c20,t5c01 --paired t5c21,t5c04 --out out/analyze_t4_full.json"
  step index_final2 fatal "python3 model/scripts/ops/build_runs_index.py"
  log "EVALEXTRA_DONE"
  exit 0
fi

if [ "$PHASE" = tail ] || [ "$PHASE" = full ]; then
  if grep -qa 'identical' "$LOGD/step_obs_verify.txt" 2>/dev/null; then
    step seg10_t6 nonfatal "bash model/scripts/sweep_t5.sh --t6 --seg 10 --budget-min 30"
  else
    log "SKIP seg10_t6: obs_seed=0 verify-committed did not print an identical verdict (T6 needs it; 主矩阵不受影响)"
  fi
  # NOTE: analyze_sweep.py only globs model/results/pinn/<prefix>*, so arm B (the pure-data
  # MLP, writes to model/results/supervised) is invisible to it and its pairs would be
  # silently skipped. Arm B numbers come out of the run index instead (metrics_root=supervised).
  step analyze nonfatal "python3 model/scripts/analyze_sweep.py --split test --metric rel_l2_speed --paired t5c21,t5c04 --paired t5c20,t5c01 --out out/analyze_t4.json"
  step index_final nonfatal "python3 model/scripts/ops/build_runs_index.py"
  log "TAIL_DONE"
  exit 0
fi

log "unknown phase $PHASE"; exit 2
