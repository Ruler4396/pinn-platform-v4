#!/usr/bin/env bash
# selftest_joint_evalonly.sh — 本机重放"补评估型段"的**装置路径**（实例 seg14 崩在这里，与封段逻辑无关）
#
# 为什么必须有这个文件（9/26 seg14 实测）：`train_joint` 的 `[skip-train-joint]` 分支里原先写的是
#   local 是补评估=0   …   ${是补评估}
# bash 的变量名只认 ASCII，中文标识符在 `local`、赋值、`${}` 三处分别报
#   `not a valid identifier` / `command not found` / `bad substitution`
# ⇒ 分支一走到就把 train_joint 弄死，`[seal]` 判级根本进不去。
# 两道既有防线都覆盖不到它：`--dry-run` 在分支之前就 return；`selftest_ledger.sh` 驱动的是
# seal_segment 本身，不经过 train_joint。所以这里把**真实函数**跑起来，并配两条必定红的正对照。
#
# 本机没有 torch ⇒ python 入口换成桩（要验的是 bash 分支能否走到封段，不是网络能否训完）。
# 跑法：bash model/scripts/selftest_joint_evalonly.sh    （全在临时目录，不写仓库、不碰实例）
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/joint_evalonly.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT
SEEDS="42 43 44 45 46"

# 非 ASCII 标识符静态闸：扫一棵目录下的所有 .sh，命中就打印 文件:行:内容
lint_nonascii() {  # $1=目录
  python3 - "$1" <<'PY'
import sys, pathlib, re
root = pathlib.Path(sys.argv[1])
pat = re.compile(
    r'\b(?:local|export|declare|readonly|typeset)\s+"?[^\x00-\x7f]'      # decl site: local <non-ascii>=...
    r'|\$\{?[^\x00-\x7f]'                                                # use site: ${...} / $... with a non-ascii name
    r'|^[ \t]*[^\x00-\x7f][^\x00=]*='                                    # 中文名=...
)
hits = 0
for p in sorted(root.rglob("*.sh")):
    for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        s = line.strip()
        if s.startswith("#") or s.startswith("echo") or s.startswith("printf"):
            continue        # 注释与文案里出现中文是正常交付物
        if pat.search(line):
            hits += 1
            print("  HIT %s:%d  %s" % (p.name, i, s[:100]))
print("HITS=%d" % hits)
PY
}

# 造一格"已训完、缺 test 评估件"的现场：metrics.json 存在且不是 smoke，evaluations/ 里没有 test 件
make_run_dir() {  # $1=root $2=run   （HAVE_TEST_EVAL=1 ⇒ 连 test 评估件也齐 ⇒ 本段只会 [skip-train]）
  mkdir -p "$1/$2"
  printf '{"note": "fixture：本段之前的正式训练产物（非 smoke）", "best_epoch": 480}\n' >"$1/$2/metrics.json"
  if [[ "${HAVE_TEST_EVAL:-0}" == 1 ]]; then
    mkdir -p "$1/$2/evaluations"
    printf '{"split_name": "test_dense", "global_metrics": {"rel_l2_speed": 0.02}}\n' >"$1/$2/evaluations/metrics_test_dense.json"
  fi
}

# 账本里预置"别的段（s00）已把这些 run 训成"——seg14 的真实形态，orphan 判据要能过
seed_prior_ledger() {  # $1=ledger $2=pinn root   （NO_PRIOR=1 ⇒ 什么都不预置，用来验"跳过无人背书"）
  [[ "${NO_PRIOR:-0}" == 1 ]] && { echo "[fixture] 不预置前段 train 行 ⇒ 10 个跳过都无人背书"; return 0; }
  local root="$2" s nn
  for s in ${SEEDS}; do
    for nn in 20 21; do
      printf '{"run": "rev2609b_t5c%s__s%s__o0", "phase": "train", "rc": 0, "pid": "100", "segment": "s00", "family": "contraction_2d", "cell": "t5c%s", "train_seed": %s, "obs_seed": 0, "wall_ms": 139000}\n' \
        "${nn}" "${s}" "${nn}" "${s}" >>"$1"
    done
  done
}

# python 桩：只写评估件并按 env 决定退出码
make_stub() {  # $1=stub dir
  mkdir -p "$1"
  cat >"$1/train_joint_upnp_pin.py" <<'PY'
import os, sys
argv = sys.argv[1:]
def get(flag, default=""):
    return argv[argv.index(flag) + 1] if flag in argv else default
run = get("--run-name")
root = os.environ.get("RESULT_ROOT", "")
if root:
    d = os.path.join(root, run, "evaluations")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "metrics_test_dense.json"), "w", encoding="utf-8") as fh:
        fh.write('{"run_name": "%s", "split_name": "test_dense", "global_metrics": {"rel_l2_speed": 0.0}}\n' % run)
print("[stub] eval-only run=%s eval_only=%s" % (run, "--eval-only" in argv))
fail = os.environ.get("STUB_FAIL_RUN", "")
sys.exit(7 if (fail and fail in run) else 0)
PY
}

# 驱动**真实**的 train_joint ×10 + 真实的 seal_segment
replay() {  # $1=case dir $2=lib
  local cdir="$1" lib="$2" root="${1}/model/results/pinn" s nn
  mkdir -p "${root}"
  for s in ${SEEDS}; do
    for nn in 20 21; do make_run_dir "${root}" "rev2609b_t5c${nn}__s${s}__o0"; done
  done
  mkdir -p "${cdir}/out"
  : >"${cdir}/out/progress.jsonl"
  seed_prior_ledger "${cdir}/out/progress.jsonl" "${root}"
  make_stub "${cdir}/stub"
  SWEEP_OUT_DIR="${cdir}/out" SWEEP_SEGMENT=s01 SWEEP_PID=111 RESULT_ROOT="${root}" \
    STUB_FAIL_RUN="${STUB_FAIL_RUN:-}" LIB="${lib}" MODEL_ROOT="${cdir}/model" STUB_DIR="${cdir}/stub" \
    bash -c '
      source "${LIB}" >/dev/null 2>&1 || exit 70
      PROJECT_ROOT="${MODEL_ROOT}"; SWEEP_LIB_DIR="${STUB_DIR}"
      for s in 42 43 44 45 46; do
        for row in "20|mainline-dense|dense" "21|strict-sparse|obs_sparse_5pct"; do
          nn="${row%%|*}"; rest="${row#*|}"; preset="${rest%%|*}"; src="${rest#*|}"
          train_joint "rev2609b_t5c${nn}__s${s}__o0" contraction_2d \
            "C-base,C-train-1,C-train-2,C-train-3,C-train-4,C-train-5" "C-val" "${src}" geometry \
            "inlet_profile_star" "t5c${nn}" "${s}" "0" "${preset}" "C-test-1,C-test-2" \
            || echo "[driver] train_joint 返回非 0 @t5c${nn} s${s}"
        done
      done
      echo "[driver] 计数 done=$RUNS_DONE fail=$RUNS_FAIL skip=$RUNS_SKIP evalonly=$RUNS_EVALONLY evalfail=$RUNS_EVAL_FAIL"
      seal_segment ""
    ' 2>&1
}

FAILS=0
BAD_PAT='not a valid identifier|invalid identifier|bad substitution|错误的替换|不是有效的标识符|command not found'

ck() {  # $1=用例名 $2=0(期望绿)|1(期望红) $3=实际 rc $4..=附加断言已在上层做过
  if [[ "$2" == "$3" ]]; then printf '  [OK ] %s\n' "$1"; return 0; fi
  printf '  [BAD] %s\n' "$1"; FAILS=$((FAILS + 1)); return 1
}

echo "== 0 静态闸：全仓 .sh 里不得有非 ASCII 变量名 =="
lint_out="$(lint_nonascii "${HERE}")"
printf '%s\n' "${lint_out}" | grep -av '^HITS=' | sed 's/^/       /'
if [[ "$(printf '%s' "${lint_out}" | tail -1)" == "HITS=0" ]]; then
  ck "真实脚本树 HITS=0" 0 0
else
  ck "真实脚本树 HITS=0" 0 1
fi

echo "== 1 真实分支重放：10 粒 [skip-train-joint] + eval-only ⇒ 必须封得住段 =="
out1="$(replay "${WORK}/case1" "${HERE}/sweep_lib.sh")" ; rc1=$?
printf '%s\n' "${out1}" | sed 's/^/       /' | tail -6
if printf '%s' "${out1}" | grep -qE "${BAD_PAT}"; then
  ck "输出里没有 bash 标识符/替换错误" 0 1
else
  ck "输出里没有 bash 标识符/替换错误" 0 0
fi
n_eval_only=$(printf '%s' "${out1}" | grep -ac '\[ok-eval-only\]' || true)
[[ "${n_eval_only}" == 10 ]] && ck "[ok-eval-only] 恰 10 次" 0 0 || ck "[ok-eval-only] 恰 10 次（实得 ${n_eval_only}）" 0 1
printf '%s' "${out1}" | grep -q "计数 done=0 fail=0 skip=0 evalonly=10 evalfail=0" \
  && ck "计数 evalonly=10 / skip=0 / fail=0" 0 0 || ck "计数 evalonly=10 / skip=0 / fail=0" 0 1
printf '%s' "${out1}" | grep -q '\[seal\] 判级=clean' && ck "打到 [seal] 判级=clean" 0 0 || ck "打到 [seal] 判级=clean" 0 1
ck "整段 rc=0" 0 "${rc1}"

echo "== 2 正对照·把变量名改回中文（= seg14 的真实缺陷），闸必须抓到 =="
mkdir -p "${WORK}/mut"
sed 's/is_eval_only/是补评估/g' "${HERE}/sweep_lib.sh" >"${WORK}/mut/sweep_lib.sh"
lint2="$(lint_nonascii "${WORK}/mut")"
printf '%s\n' "${lint2}" | grep -a 'HITS=' | sed 's/^/       /'
[[ "$(printf '%s' "${lint2}" | tail -1)" != "HITS=0" ]] \
  && ck "静态闸在变异副本上变红（证明它不是摆设）" 0 0 || ck "静态闸在变异副本上变红（证明它不是摆设）" 0 1
out2="$(replay "${WORK}/case2" "${WORK}/mut/sweep_lib.sh")" ; rc2=$?
printf '%s\n' "${out2}" | grep -aE "${BAD_PAT}" | head -2 | sed 's/^/       /'
if printf '%s' "${out2}" | grep -qE "${BAD_PAT}"; then ck "重放副本出现 bash 标识符错误（预期红）" 0 0; else ck "重放副本出现 bash 标识符错误（预期红）" 0 1; fi
[[ "${rc2}" != 0 ]] && ck "重放副本走不到封段、rc!=0（正是 seg14 的形态）" 0 0 || ck "重放副本走不到封段、rc!=0（正是 seg14 的形态）" 0 1
if printf '%s' "${out2}" | grep -q '\[seal\] 判级=clean'; then ck "变异副本不得打出 判级=clean" 0 1; else ck "变异副本不得打出 判级=clean" 0 0; fi

echo "== 3 正对照·补评估本身失败（python 桩 rc=7）⇒ 判级必须是 fatal(rc=1) 而不是 clean =="
out3="$(STUB_FAIL_RUN=t5c21__s44 replay "${WORK}/case3" "${HERE}/sweep_lib.sh")" ; rc3=$?
printf '%s\n' "${out3}" | grep -aE '\[FAIL\] eval-only|计数 |INVALID|\[seal\]' | head -4 | sed 's/^/       /'
printf '%s' "${out3}" | grep -q "fail=1" && ck "失败被计入 RUNS_FAIL" 0 0 || ck "失败被计入 RUNS_FAIL" 0 1
[[ "${rc3}" == 1 ]] && ck "退出码=1（数据不可信档）" 0 0 || ck "退出码=1（数据不可信档）实得=${rc3}" 0 1
printf '%s' "${out3}" | grep -q '判级=fatal' && ck "判级行写明 fatal" 0 0 || ck "判级行写明 fatal" 0 1

echo "== 4 纯跳过段（test 件已齐 ⇒ 10 个 [skip-train]、0 个补评估；SKIP_RUNS 由真实代码填）=="
out4="$(HAVE_TEST_EVAL=1 replay "${WORK}/case4" "${HERE}/sweep_lib.sh")" ; rc4=$?
printf '%s\n' "${out4}" | grep -aE 'skipped|seal|no-op' | tail -2 | sed 's/^/       /'
printf '%s' "${out4}" | grep -q "计数 done=0 fail=0 skip=10 evalonly=0 evalfail=0" \
  && ck "真实代码把 10 个跳过都进了名单" 0 0 || ck "真实代码把 10 个跳过都进了名单" 0 1
if printf '%s' "${out4}" | grep -q '\[ok-eval-only\]'; then ck "这一趟不该有补评估" 0 1; else ck "这一趟不该有补评估" 0 0; fi
printf '%s' "${out4}" | grep -q '\[seal\] 判级=clean' && ck "纯跳过段判级=clean（定档甲：按 run 归属，不按段计数）" 0 0 \
  || ck "纯跳过段判级=clean（定档甲：按 run 归属，不按段计数）" 0 1
[[ "${rc4}" == 0 ]] && ck "整段 rc=0" 0 0 || ck "整段 rc=0（实得 ${rc4}）" 0 1

echo "== 5 正对照·同样 10 个跳过，但账上没有任何前段成功行 ⇒ 必须红并点名 =="
out5="$(HAVE_TEST_EVAL=1 NO_PRIOR=1 replay "${WORK}/case5" "${HERE}/sweep_lib.sh")" ; rc5=$?
printf '%s\n' "${out5}" | grep -aE 'INVALID|seal' | tail -1 | cut -c1-160 | sed 's/^/       /'
printf '%s' "${out5}" | grep -q '被跳过但账上任何段都没有终态成功的 train 行' \
  && ck "名单点名为无人背书的跳过" 0 0 || ck "名单点名为无人背书的跳过" 0 1
[[ "${rc5}" == 4 ]] && ck "退出码=4（产物完好、仅记账不符）" 0 0 || ck "退出码=4（产物完好、仅记账不符）实得=${rc5}" 0 1

echo
if [[ ${FAILS} -eq 0 ]]; then
  echo "总体：五组全符合 ⇒ [skip-train-joint] 两条分支（补评估 / 纯跳过）都能走到封段并给出正确判级；中文变量名这一类缺陷有静态闸 + 动态正对照两道把守"
  exit 0
fi
echo "总体：有 ${FAILS} 条断言不符 ⇒ 这条分支仍不可信，别投实例"
exit 1
