#!/usr/bin/env bash
# selftest_ledger.sh — 驱动**真实的** seal_segment 对账函数，验证它在 9 种账本形态下该绿的绿、该红的红。
#
# 为什么要这个文件：9/26 段 01 收尾自判 `INVALID: 内存=87 账本=89 / 失败 内存=0 账本=1`，
# 而数据其实是 95/95 满种子 —— 闸门把"追加式账本 + 续跑/重试"当成了不一致。修完之后必须证明
# 这次改动**没有把闸门改成永远绿**，所以这里带两条"必定 INVALID"的正对照（C3、C6）。
#
# 跑法（本机即可，不碰实例、不写仓库；账本与 OUT_DIR 都在临时目录）：
#   bash model/scripts/selftest_ledger.sh
# 退出码 0 = 九例全符合预期。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ledger_selftest.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

# $1=用例名 $2=期望(ok|invalid) $3=done $4=fail $5=skip $6=expected $7=gate $8=evalfail
# $9..=账本行（printf 格式，%s=本进程 pid，%s=段名）
run_case() {
  local name="$1" want="$2" done_n="$3" fail_n="$4" skip_n="$5" exp="$6" gate="$7" evalf="$8"
  local want_sub="${EXPECT_SUB:-}"   # 判据子串走环境变量，账本行仍从 $9 起排
  shift 8
  local case_dir="${WORK}/${name}"
  mkdir -p "${case_dir}"
  : >"${case_dir}/progress.jsonl"
  local ln
  for ln in "$@"; do
    printf '%s\n' "${ln}" >>"${case_dir}/progress.jsonl"
  done
  local out rc
  out="$(
    SWEEP_OUT_DIR="${case_dir}" SWEEP_SEGMENT=s01 SWEEP_DRY_RUN=1 SWEEP_PID=111 \
      bash -c "
        source '${HERE}/sweep_lib.sh' >/dev/null 2>&1 || exit 70
        RUNS_DONE=${done_n}; RUNS_FAIL=${fail_n}; RUNS_SKIP=${skip_n}
        RUNS_GATE_FAIL=${gate}; RUNS_EVAL_FAIL=${evalf}
        seal_segment '${exp}' 2>&1
      "
  )"
  rc=$?
  local got="ok"
  [[ "${rc}" != 0 ]] && got="invalid"
  if [[ -n "${want_sub}" ]] && ! printf '%s' "${out}" | grep -aF -- "${want_sub}" >/dev/null; then
    printf '  [BAD] %-34s 期望判据 "%s" 未出现在输出里
' "${name}" "${want_sub}"
    printf '%s
' "${out}" | sed 's/^/         | /' | tail -4
    return 1
  fi
  if [[ "${got}" == "${want}" ]]; then
    printf '  [OK ] %-34s 期望=%-7s 实得=%-7s rc=%d  %s\n' "${name}" "${want}" "${got}" "${rc}" \
      "$(printf '%s' "${out}" | grep -aE '^(INVALID|[0-9]+ runs)' | head -1 | cut -c1-96)"
    return 0
  fi
  printf '  [BAD] %-34s 期望=%-7s 实得=%-7s rc=%d\n' "${name}" "${want}" "${got}" "${rc}"
  printf '%s\n' "${out}" | sed 's/^/         | /' | tail -4
  return 1
}

P='"segment": "s01", "family": "contraction_2d", "cell": "t5c04", "train_seed": 42, "obs_seed": 0, "wall_ms": 53700'
L() {  # L <run> <phase> <rc> <pid>
  printf '{"run": "%s", "phase": "%s", "rc": %s, "pid": "%s", %s}\n' "$1" "$2" "$3" "$4" "${P}"
}

FAILS=0
echo "== seal_segment 九例（驱动真实函数，不是复刻逻辑）=="
# ① 干净：3 个 run 各一行 train rc=0，内存 3/0/0
run_case "1 正常绿" ok 3 0 0 "" 0 0 \
  "$(L r1 train 0 111)" "$(L r2 train 0 111)" "$(L r3 train 0 111)" || FAILS=$((FAILS+1))
# ② **本次修的假红**：同段重投续跑 ⇒ 账本有历史行 + 本进程 2 成功 1 跳过 + 一条早先的失败行
EXPECT_SUB="重试行 1" run_case "2 续跑+重试历史行不该假红" ok 2 0 1 "" 0 0 \
  "$(L r1 train 1 100)" "$(L r2 train 0 100)" "$(L r1 train 0 111)" "$(L r3 train 0 111)" || FAILS=$((FAILS+1))
# ③ 正对照（必定 INVALID）：账本少一行 —— 内存说跑了 3 个，账本只有 2 个本进程成功行
EXPECT_SUB="本进程训练成功数 内存=3 账本=2" run_case "3 正对照·账本缺一行" invalid 3 0 0 "" 0 0 \
  "$(L r1 train 0 111)" "$(L r2 train 0 111)" || FAILS=$((FAILS+1))
# ④ rc!=0 必须进失败计数：账本有本进程 eval rc=7，内存 fail 仍写 0 ⇒ 红
EXPECT_SUB="本进程失败数 内存=0 账本=1" run_case "4 rc!=0 未计失败要报红" invalid 2 0 0 "" 0 0 \
  "$(L r1 train 0 111)" "$(L r1 eval_val 7 111)" "$(L r2 train 0 111)" || FAILS=$((FAILS+1))
# ⑤ 终态失败：某 run 最后一条 train 行 rc=1（重试也没救回来）⇒ 红
EXPECT_SUB="个 run 终态失败" run_case "5 终态失败 run" invalid 2 1 0 "" 0 0 \
  "$(L r1 train 0 111)" "$(L r2 train 0 111)" "$(L r3 train 1 111)" || FAILS=$((FAILS+1))
# ⑥ 正对照（必定 INVALID）：段内 0 行 —— 闸门不能把"没记账"读成"全通过"
run_case "6 正对照·账本 0 行" invalid 0 0 0 "" 0 0 || FAILS=$((FAILS+1))
# ⑦ 跳过归属：内存说跳了 1 个，但账本里没有它的成功行 ⇒ 红（skip 必须能对上终态成功）
EXPECT_SUB="段内终态成功 run 数=1" run_case "7 跳过无对应成功行" invalid 1 0 1 "" 0 0 \
  "$(L r1 train 0 111)" || FAILS=$((FAILS+1))
# ⑧ run-name 撞车：同一 run 两条成功 train 行 ⇒ 红（不许静默算一个单元）
EXPECT_SUB="同一 run 有多条成功 train 行" run_case "8 同 run 两条成功行" invalid 2 0 0 "" 0 0 \
  "$(L r1 train 0 111)" "$(L r1 train 0 112)" "$(L r2 train 0 111)" || FAILS=$((FAILS+1))
# ⑨ 前置闸门没过（观测预算/点位生成）⇒ 红，且不能被当成"跑完"
EXPECT_SUB="前置闸门没过" run_case "9 闸门失败计数" invalid 1 0 0 "" 1 0 \
  "$(L r1 train 0 111)" || FAILS=$((FAILS+1))

echo
if [[ ${FAILS} -eq 0 ]]; then
  echo "总体：九例全符合（含 3、6 两条'必定 INVALID'的正对照 ⇒ 闸门没被改成永远绿）"
  exit 0
fi
echo "总体：有 ${FAILS} 例不符 ⇒ 对账函数不可信"
exit 1
