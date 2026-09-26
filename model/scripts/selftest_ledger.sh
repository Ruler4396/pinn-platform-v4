#!/usr/bin/env bash
# selftest_ledger.sh — 驱动**真实的** seal_segment 对账函数，验证它在 17 种账本形态下该绿的绿、该红的红。
#
# 为什么要这个文件：9/26 段 01 收尾自判 `INVALID: 内存=87 账本=89 / 失败 内存=0 账本=1`，
# 而数据其实是 95/95 满种子 —— 闸门把"追加式账本 + 续跑/重试"当成了不一致。修完之后必须证明
# 这次改动**没有把闸门改成永远绿**，所以这里带两条"必定 INVALID"的正对照（C3、C6）。
#
# 跑法（本机即可，不碰实例、不写仓库；账本与 OUT_DIR 都在临时目录）：
#   bash model/scripts/selftest_ledger.sh
# 退出码 0 = 十七例全符合预期（其中 7 条是必定红/必定非零的正面对照）。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ledger_selftest.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

# $1=用例名 $2=期望(ok|invalid) $3=done $4=fail $5=skip $6=expected $7=gate $8=evalfail
# $9..=账本行（printf 格式，%s=本进程 pid，%s=段名）
run_case() {
  local name="$1" want="$2" done_n="$3" fail_n="$4" skip_n="$5" exp="$6" gate="$7" evalf="$8"
  local want_sub="${EXPECT_SUB:-}" evalonly_n="${EVALONLY:-0}" rc_want="${RCWANT:-}"   # 判据子串/补评估数/精确退出码走环境变量
  local skip_list="${SKIPLIST:-}"                              # 本段被 [skip-train] 的 run 名单（逗号分隔）
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
    SWEEP_OUT_DIR="${case_dir}" SWEEP_SEGMENT=s01 SWEEP_DRY_RUN=1 SWEEP_PID=111 SKIPLIST="${skip_list}" \
      bash -c "
        source '${HERE}/sweep_lib.sh' >/dev/null 2>&1 || exit 70
        RUNS_DONE=${done_n}; RUNS_FAIL=${fail_n}; RUNS_SKIP=${skip_n}
        RUNS_GATE_FAIL=${gate}; RUNS_EVAL_FAIL=${evalf}; RUNS_EVALONLY=${evalonly_n}
        SKIP_RUNS='${skip_list}'
        seal_segment '${exp}' 2>&1
      "
  )"
  rc=$?
  local got="ok"
  [[ "${rc}" != 0 ]] && got="invalid"
  if [[ -n "${rc_want}" && "${rc}" != "${rc_want}" ]]; then
    printf '  [BAD] %-34s 退出码期望=%s 实得=%s（判级选错了）
' "${name}" "${rc_want}" "${rc}"
    printf '%s
' "${out}" | grep -a "\[seal\]" | sed 's/^/         | /'
    return 1
  fi
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
echo "== seal_segment 十七例（驱动真实函数，不是复刻逻辑）=="
# ① 干净：3 个 run 各一行 train rc=0，内存 3/0/0
RCWANT=0 run_case "1 正常绿" ok 3 0 0 "" 0 0 \
  "$(L r1 train 0 111)" "$(L r2 train 0 111)" "$(L r3 train 0 111)" || FAILS=$((FAILS+1))
# ② **本次修的假红**：同段重投续跑 ⇒ 账本有历史行 + 本进程 2 成功 1 跳过 + 一条早先的失败行
EXPECT_SUB="重试行 1" SKIPLIST="r2" RCWANT=0 run_case "2 续跑+重试历史行不该假红" ok 2 0 1 "" 0 0 \
  "$(L r1 train 1 100)" "$(L r2 train 0 100)" "$(L r1 train 0 111)" "$(L r3 train 0 111)" || FAILS=$((FAILS+1))
# ③ 正对照（必定 INVALID）：账本少一行 —— 内存说跑了 3 个，账本只有 2 个本进程成功行
EXPECT_SUB="本进程训练成功数 内存=3 账本=2" RCWANT=4 run_case "3 正对照·账本缺一行(仅记账)" invalid 3 0 0 "" 0 0 \
  "$(L r1 train 0 111)" "$(L r2 train 0 111)" || FAILS=$((FAILS+1))
# ④ rc!=0 必须进失败计数：账本有本进程 eval rc=7，内存 fail 仍写 0 ⇒ 红
EXPECT_SUB="评估阶段终态失败" RCWANT=1 run_case "4 rc!=0 未计失败要报红(判级fatal)" invalid 2 0 0 "" 0 0 \
  "$(L r1 train 0 111)" "$(L r1 eval_val 7 111)" "$(L r2 train 0 111)" || FAILS=$((FAILS+1))
# ⑤ 终态失败：某 run 最后一条 train 行 rc=1（重试也没救回来）⇒ 红
EXPECT_SUB="个 run 终态失败" RCWANT=1 run_case "5 终态失败 run" invalid 2 1 0 "" 0 0 \
  "$(L r1 train 0 111)" "$(L r2 train 0 111)" "$(L r3 train 1 111)" || FAILS=$((FAILS+1))
# ⑥ 正对照（必定 INVALID）：段内 0 行 —— 闸门不能把"没记账"读成"全通过"
RCWANT=1 run_case "6 正对照·账本 0 行" invalid 0 0 0 "" 0 0 || FAILS=$((FAILS+1))
# ⑦ 跳过归属：内存说跳了 1 个，但账本里没有它的成功行 ⇒ 红（skip 必须能对上终态成功）
EXPECT_SUB="被跳过但账上任何段都没有终态成功的 train 行" SKIPLIST="r9" RCWANT=4 run_case "7 跳过无对应成功行" invalid 1 0 1 "" 0 0 \
  "$(L r1 train 0 111)" || FAILS=$((FAILS+1))
# ⑧ run-name 撞车：同一 run 两条成功 train 行 ⇒ 红（不许静默算一个单元）
EXPECT_SUB="同一 run 有多条成功 train 行" RCWANT=4 run_case "8 同 run 两条成功行" invalid 2 0 0 "" 0 0 \
  "$(L r1 train 0 111)" "$(L r1 train 0 112)" "$(L r2 train 0 111)" || FAILS=$((FAILS+1))
# ⑨ 前置闸门没过（观测预算/点位生成）⇒ 红，且不能被当成"跑完"
EXPECT_SUB="前置闸门没过" RCWANT=1 run_case "9 闸门失败计数" invalid 1 0 0 "" 1 0 \
  "$(L r1 train 0 111)" || FAILS=$((FAILS+1))

# ⑩ 补评估段（实例段 14 的真实现场）：本段 0 条 train 行、10 个 run 只有 eval_test rc=0，
#    且这些 run 的 train 终态成功行在**别的段**（s00）。旧等式会假红；修完必须绿。
L0() { printf '{"run": "%s", "phase": "%s", "rc": %s, "pid": "%s", "segment": "s00", "family": "contraction_2d", "cell": "t5c20", "train_seed": 42, "obs_seed": 0, "wall_ms": 139000}
' "$1" "$2" "$3" "$4"; }
EO_LINES=()
for i in 1 2 3; do EO_LINES+=("$(L0 "r$i" train 0 100)" "$(L r$i eval_test 0 111)"); done
RCWANT=0 EVALONLY=3 run_case "10 补评估段不该假红" ok 0 0 0 "" 0 0 "${EO_LINES[@]}" || FAILS=$((FAILS+1))
# ⑪ 正对照：给一个从没训练成功的 run 补评估 ⇒ 必须红（证明上一条不是把闸门改松）
EVALONLY=2 RCWANT=1 run_case "11 正对照·给失败run补评估要红" invalid 0 0 0 "" 0 0   "$(L0 r1 train 0 100)" "$(L r1 eval_test 0 111)" "$(L r2 eval_test 0 111)" || FAILS=$((FAILS+1))
# ⑫ 正对照：内存说补评估 3 个、账本只有 2 个 ⇒ 红
EVALONLY=3 RCWANT=4 run_case "12 正对照·补评估数不符要红(仅记账)" invalid 0 0 0 "" 0 0   "$(L0 r1 train 0 100)" "$(L r1 eval_test 0 111)" "$(L0 r2 train 0 100)" "$(L r2 eval_test 0 111)" || FAILS=$((FAILS+1))

# ⑬ 实例 seg14 的真实现场（9/26 20:37 统括官实测）：本段 10 个 [skip-train]、0 新训练、0 补评估，
#    成功 train 行都在别的段（s00），本段账本里只剩上一趟崩掉的补评估留下的 eval_test 行。
#    旧等式按"段内终态成功数 == 成功 + 跳过"判它红；定档（甲）之后必须 clean。
SKIP10=$(python3 -c "print(','.join('r%d'%i for i in range(1,11)))")
L13=()
for i in $(seq 1 10); do L13+=("$(L0 "r$i" train 0 100)" "$(L "r$i" eval_test 0 100)"); done
RCWANT=0 SKIPLIST="${SKIP10}" run_case "13 纯跳过段(seg14 现场)不该假红" ok 0 0 10 "" 0 0 "${L13[@]}" || FAILS=$((FAILS+1))
# ⑭ 正对照：同样 10 个跳过，但把 s00 的背书行全抽掉 ⇒ 产物在盘上而账上无凭 ⇒ 仅记账不符(4)并点名
L14=()
for i in $(seq 1 10); do L14+=("$(L "r$i" eval_test 0 100)"); done
EXPECT_SUB="被跳过但账上任何段都没有终态成功的 train 行" RCWANT=4 SKIPLIST="${SKIP10}" \
  run_case "14 正对照·跳过无人背书要红" invalid 0 0 10 "" 0 0 "${L14[@]}" || FAILS=$((FAILS+1))
# ⑮ 正对照：名单只交 8 个、内存说跳了 10 个 ⇒ 有 [skip-train] 忘了进名单 ⇒ 红
EXPECT_SUB="跳过 run 数 内存=10 名单=8" RCWANT=4 \
  SKIPLIST="$(python3 -c "print(','.join('r%d'%i for i in range(1,9)))")" \
  run_case "15 正对照·名单少两个要红" invalid 0 0 10 "" 0 0 "${L13[@]}" || FAILS=$((FAILS+1))
# ⑯ no-op 段：本段账本 0 行（成功行都在 s00）⇒ 放行，但必须打出 no-op 提示
L16=()
for i in $(seq 1 10); do L16+=("$(L0 "r$i" train 0 100)"); done
EXPECT_SUB="本段是纯续跑（no-op）" RCWANT=0 SKIPLIST="${SKIP10}" \
  run_case "16 纯跳过且本段 0 行(no-op)" ok 0 0 10 "" 0 0 "${L16[@]}" || FAILS=$((FAILS+1))
# ⑰ 正对照：本段 0 行但内存报了 1 个成功 ⇒ no-op 豁免不许扩大到"记账真的漏了"的形状 ⇒ fatal(1)
RCWANT=1 SKIPLIST="${SKIP10}" run_case "17 正对照·0 行却有成功读数要 fatal" invalid 1 0 10 "" 0 0 "${L16[@]}" || FAILS=$((FAILS+1))

echo
if [[ ${FAILS} -eq 0 ]]; then
  echo "总体：十七例全符合（七条必定红/必定非零的正对照 + 退出码判级 fatal=1 / 仅记账=4 / clean=0 全对 ⇒ 闸门没被改成永远绿）"
  exit 0
fi
echo "总体：有 ${FAILS} 例不符 ⇒ 对账函数不可信"
exit 1
