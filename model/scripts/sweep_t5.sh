#!/usr/bin/env bash
# sweep_t5.sh — T5 全章 mean±std 主矩阵 + T6 采样策略配对矩阵（修订方案 §3/§5 的执行体）。
#
# 用法
#   bash model/scripts/sweep_t5.sh --dry-run                      # 只打印 argv 与文件计划，不执行
#   bash model/scripts/sweep_t5.sh --seg 01 --budget-min 90       # 第 1 段；超预算提前退出
#   bash model/scripts/sweep_t5.sh --seg 01 --budget-min 90       # 同段重投即续跑（有 metrics.json 就跳过训练）
#   bash model/scripts/sweep_t5.sh --t6 --seg 02                  # 只跑 T6 配对矩阵
#   bash model/scripts/sweep_t5.sh --only-cells 1,2,19 --seg 03   # 只跑指定格
#
# 产物落点（写在仓外，不弄脏工作区；见 sweep_lib.sh:_pick_out_dir）
#   实例：/mnt/workspace/pinn-repro-2026/out    本机：<仓的上一级>/.scratch/sweep_out
#   覆盖：SWEEP_OUT_DIR=/path bash model/scripts/sweep_t5.sh …
#
# 两层种子
#   train_seed ∈ {42..46} → 只换 --seed（初始化/批序/物理配点，见 train_*_strict_sparse.py:105-108）
#   obs_seed   ∈ {0..3}   → 换点位：obs_seed=0 复用已入库 CSV；>0 先由
#                           generate_observations_seeded.py 生成 obs_*__s{k}.csv 再训练。
#                           训练侧不需要改：load_case_source 把源名当文件名（train_supervised.py:417-421）
#
# 与 run_strict_sparse_experiments.sh:38-76 的关系：argv 逐项照抄，只有三处不同
#   ① 多了 --seed（旧启动器不传，靠 argparse 默认 42）
#   ② 观测源名可带 __s{k} 后缀（obs_seed>0）
#   ③ run-name 换成 rev2609_t5cNN__s{train}__o{obs}
# 稠密格（1/2/19）改用 train_velocity_pressure_independent.py + 表4-4 权重档，因为它们对标
# v4 稠密主线批（41 个 run 的权重指纹：23 个用 0.5/1e-4/1.0、14 个用 0/0/0、4 个纯 MLP 无权重）。
#
# 实例硬约束：无 /usr/bin/time、无 bc ⇒ 墙钟只用 date +%s%N 的整数运算。
# numpy 必须 2.x：train_supervised.py:588 用 np.trapezoid，镜像自带 1.26.4 会 AttributeError。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=sweep_lib.sh
source "${SCRIPT_DIR}/sweep_lib.sh"

RUN_T5=1
RUN_T6=1
RUN_BASE=0        # 必做1 基线三件套（臂 A 单网络联合 PINN / 臂 B 纯数据 MLP / 臂 C POD+最小二乘）
RUN_POD=1
SEEDS="42 43 44 45 46"
OBS_SEEDS="0 1 2 3"
T6_SEEDS="42 43"
BUDGET_MIN=90
ONLY_CELLS=""
RUN_PREFIX="rev2609"
BUDGET_START="$(_now_ns)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seg) SEGMENT_TAG="$2"; shift 2 ;;
    --budget-min) BUDGET_MIN="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --obs-seeds) OBS_SEEDS="$2"; shift 2 ;;
    --only-cells) ONLY_CELLS="$2"; shift 2 ;;
    --prefix) RUN_PREFIX="$2"; shift 2 ;;
    --t5) RUN_T5=1; RUN_T6=0; shift ;;
    --baseline) RUN_T5=0; RUN_T6=0; RUN_BASE=1; shift ;;
    --no-pod) RUN_POD=0; shift ;;
    --t6) RUN_T6=1; RUN_T5=0; shift ;;
    --dry-run) SWEEP_DRY_RUN=1; shift ;;
    *) echo "未知参数 $1" >&2; exit 2 ;;
  esac
done
export SWEEP_DRY_RUN SEGMENT_TAG

# ------------------------------------------------- 列表参数规范化（逗号/分号/空格都收）
# 9/25 实例上 `--seeds 42,43,44,45,46` 被当成**一个**种子：plan 行显示"要训练=19"、
# run 名变成 rev2609_t5c01__s42,43,...__o0、训练全失败，最后靠段末记账闸门才抱住。
SEEDS_RAW="${SEEDS}"; OBS_SEEDS_RAW="${OBS_SEEDS}"; T6_SEEDS_RAW="${T6_SEEDS}"
ONLY_CELLS_RAW="${ONLY_CELLS}"
SEEDS="$(norm_list "${SEEDS}")"
OBS_SEEDS="$(norm_list "${OBS_SEEDS}")"
T6_SEEDS="$(norm_list "${T6_SEEDS}")"
ONLY_CELLS="$(norm_list "${ONLY_CELLS}")"
[[ -n "${ONLY_CELLS}" ]] && ONLY_CELLS="${ONLY_CELLS// /,}"   # cell_wanted 靠逗号包边匹配

validate_sweep_lists() {
  validate_int_list --seeds "${SEEDS_RAW}" "${SEEDS}" || return 1
  validate_int_list --obs-seeds "${OBS_SEEDS_RAW}" "${OBS_SEEDS}" || return 1
  validate_int_list --t6-seeds "${T6_SEEDS_RAW}" "${T6_SEEDS}" || return 1
  if [[ -n "${ONLY_CELLS}" ]]; then
    local -a toks=()
    IFS=',' read -r -a toks <<<"${ONLY_CELLS}"
    local tok
    for tok in "${toks[@]}"; do
      [[ "${tok}" == "t6" ]] && continue
      [[ "${tok}" =~ ^([1-9]|1[0-9]|2[0-3])$ ]] \
        || { echo "[FAIL] --only-cells 只许 1..19（主矩阵）、20..23（基线三件套）或 t6，收到 '${tok}'（原值='${ONLY_CELLS_RAW}'）" >&2; return 1; }
    done
  fi
  echo "[args] seeds=[${SEEDS}] obs_seeds=[${OBS_SEEDS}] t6_seeds=[${T6_SEEDS}] only_cells=[${ONLY_CELLS}]"
  return 0
}

CONTRA_TRAIN="C-base,C-train-1,C-train-2,C-train-3,C-train-4,C-train-5"
CONTRA_VAL="C-val"
CONTRA_TEST="C-test-1,C-test-2"
BEND_TRAIN="B-base__ip_blunted,B-train-1__ip_blunted,B-train-2__ip_blunted,B-train-3__ip_blunted"
BEND_VAL="B-val__ip_blunted"
BEND_TEST="B-test-1__ip_blunted"
STRICT="train_velocity_pressure_independent_strict_sparse.py"
MAIN="train_velocity_pressure_independent.py"

# T6 的两臂直接落在 T5 的格4（分层5%）/格8（均匀5%）命名空间里 ⇒ 同配置不重复训练
T6_REGION_CELL=4
T6_UNIFORM_CELL=8

osuffix() { [[ "$1" == "0" ]] && echo "" || echo "__s$1"; }
cell_of_num() { printf 't5c%02d' "$1"; }

# 过滤：--only-cells 接受格号（1,2,19）或 t6
cell_wanted() {
  [[ -z "${ONLY_CELLS}" ]] && return 0
  [[ ",${ONLY_CELLS}," == *",$1,"* ]] && return 0
  [[ ",${ONLY_CELLS}," == *",t6,"* ]] && return 0
  return 1
}

expand_case() {  # 记号 → 真实工况清单
  case "$1" in
    TC) echo "${CONTRA_TRAIN}" ;; CV) echo "${CONTRA_VAL}" ;; CT) echo "${CONTRA_TEST}" ;;
    BT) echo "${BEND_TRAIN}" ;; BV) echo "${BEND_VAL}" ;; BT1) echo "${BEND_TEST}" ;;
    *) echo "$1" ;;
  esac
}

# 段预算守卫：本单元预计 ${1} 秒，余量不够就提前退出（返回 1 让上层停止铺排）
budget_guard() {
  local need="$1" used left
  used="$(elapsed_sec "${BUDGET_START}")"
  left=$(( BUDGET_MIN * 60 - used ))
  if [[ "${need}" -gt "${left}" ]]; then
    echo "[segment-stop] 已用 ${used}s，余量 ${left}s < 本单元预计 ${need}s ⇒ 提前退出，重投同段续跑"
    return 1
  fi
  return 0
}

unit_cost() {  # $1=family $2=dense|sparse → 一格（训练 + 2 次评估）的秒数
  local family="$1" kind="$2" base
  if [[ "${family}" == "bend_2d" ]]; then base="${BEND_UNIT_SEC}"
  elif [[ "${kind}" == "dense" ]]; then base="${UNIT_DENSE_TRAIN_SEC}"
  else base="${UNIT_SPARSE_TRAIN_SEC}"; fi
  echo $(( ${base%.*} + 2 * ${UNIT_EVAL_SEC%.*} + 1 ))
}

# ------------------------------------------------------------------ 观测点位
generate_obs() {
  [[ "${RUN_T6}" == 1 ]] || return 0
  local s
  for s in ${OBS_SEEDS}; do
    [[ "${s}" == "0" ]] && continue          # obs_seed=0 = 已入库点位，绝不重写
    # T6 只需要收缩族 5% 的两臂点位；其余格固定用已入库文件，控住机时
    ensure_observations contraction_2d "0.05" "${s}" "${CONTRA_TRAIN},${CONTRA_VAL},${CONTRA_TEST}" || return 1
  done
  return 0
}

# ------------------------------------------------------------------ T5 · 19 格
# 列：id|desc|family|train|val|观测源|feature_mode|drop|eval_val|eval_test|script|权重档|强制壁面|单价类型
CELLS=(
  "1|稠密主线(表5-2/5-3/5-7 C行/5-8启用行)|contraction_2d|TC|CV|dense|geometry|inlet_profile_star|CV|CT|${MAIN}|mainline-dense||dense"
  "2|稠密·阶段内PDE关(表5-8不启用行,现无一手产物)|contraction_2d|TC|CV|dense|geometry|inlet_profile_star|CV|CT|${MAIN}|no-stage-pde||dense"
  "3|分层1%(表5-1/5-4)|contraction_2d|TC|CV|obs_sparse_1pct|geometry|inlet_profile_star|CV|CT|${STRICT}|strict-sparse||sparse"
  "4|分层5%(表5-5分层臂/表5-6无噪臂/图5-18)|contraction_2d|TC|CV|obs_sparse_5pct|geometry|inlet_profile_star|CV|CT|${STRICT}|strict-sparse||sparse"
  "5|分层10%(表5-4)|contraction_2d|TC|CV|obs_sparse_10pct|geometry|inlet_profile_star|CV|CT|${STRICT}|strict-sparse||sparse"
  "6|分层15%(表5-4)|contraction_2d|TC|CV|obs_sparse_15pct|geometry|inlet_profile_star|CV|CT|${STRICT}|strict-sparse||sparse"
  "7|均匀1%|contraction_2d|TC|CV|obs_uniform_1pct|geometry|inlet_profile_star|CV|CT|${STRICT}|strict-sparse||sparse"
  "8|均匀5%(表5-5均匀臂)|contraction_2d|TC|CV|obs_uniform_5pct|geometry|inlet_profile_star|CV|CT|${STRICT}|strict-sparse||sparse"
  "9|均匀10%|contraction_2d|TC|CV|obs_uniform_10pct|geometry|inlet_profile_star|CV|CT|${STRICT}|strict-sparse||sparse"
  "10|均匀15%|contraction_2d|TC|CV|obs_uniform_15pct|geometry|inlet_profile_star|CV|CT|${STRICT}|strict-sparse||sparse"
  "11|分层5%+噪声1%(补4.4.3承诺档)|contraction_2d|TC|CV|obs_sparse_5pct_noise_1pct|geometry|inlet_profile_star|CV|CT|${STRICT}|strict-sparse||sparse"
  "12|分层5%+噪声5%(补4.4.3承诺档)|contraction_2d|TC|CV|obs_sparse_5pct_noise_5pct|geometry|inlet_profile_star|CV|CT|${STRICT}|strict-sparse||sparse"
  "13|basic4特征(只能soft:硬包络要wall_distance_frac,basic集里没有)|contraction_2d|TC|CV|obs_sparse_5pct|basic||CV|CT|${STRICT}|strict-sparse|soft|sparse"
  "14|geometry14特征+soft(与格13只差特征集的单变量对照)|contraction_2d|TC|CV|obs_sparse_5pct|geometry|inlet_profile_star|CV|CT|${STRICT}|strict-sparse|soft|sparse"
  "15|弯曲分层1%(表5-4)|bend_2d|BT|BV|obs_sparse_1pct|geometry|inlet_profile_star|BV|BT1|${STRICT}|strict-sparse||sparse"
  "16|弯曲分层5%(表5-4/表5-7 B行)|bend_2d|BT|BV|obs_sparse_5pct|geometry|inlet_profile_star|BV|BT1|${STRICT}|strict-sparse||sparse"
  "17|弯曲分层10%(表5-4)|bend_2d|BT|BV|obs_sparse_10pct|geometry|inlet_profile_star|BV|BT1|${STRICT}|strict-sparse||sparse"
  "18|弯曲分层15%(表5-4)|bend_2d|BT|BV|obs_sparse_15pct|geometry|inlet_profile_star|BV|BT1|${STRICT}|strict-sparse||sparse"
  "19|弯曲稠密(同批次稠密对照,修【审计】#7跨批次比)|bend_2d|BT|BV|dense|geometry|inlet_profile_star|BV|BT1|${MAIN}|mainline-dense||dense"
)

t5_matrix() {
  local row nn desc family tc vc src fmode drop evl evt script weights fwall kind s run_name cost cell_id
  for row in "${CELLS[@]}"; do
    IFS='|' read -r nn desc family tc vc src fmode drop evl evt script weights fwall kind <<<"${row}"
    cell_wanted "${nn}" || continue
    tc="$(expand_case "${tc}")"; vc="$(expand_case "${vc}")"
    evl="$(expand_case "${evl}")"; evt="$(expand_case "${evt}")"
    cost="$(unit_cost "${family}" "${kind}")"
    cell_id="$(cell_of_num "${nn}")"
    for s in ${SEEDS}; do
      budget_guard "${cost}" || return 1
      run_name="${RUN_PREFIX}_${cell_id}__s${s}__o0"
      echo "### T5 格${nn} ${desc} train_seed=${s} obs_seed=0 单格≈${cost}s"
      train_dual "${run_name}" "${family}" "${tc}" "${vc}" "${src}" "${src}" \
        "${fmode}" "${drop}" "${evl}" "${evt}" \
        "${cell_id}" "${s}" "0" "${script}" "${weights}" "${fwall}" || return 1
    done
    # 多点位（obs_seed>0）不在此铺排：由 T6 用同一批格号生成，避免同一配置跑两遍
  done
  return 0
}

# ------------------------------------------------------------------ T6 · 8 配对单元
# 配对单位 = (obs_seed, train_seed)；两臂共享同一 obs_seed ⇒ 同批点位、同观测预算。
# run 名沿用 t5c04/t5c08 + __o{k}：obs_seed=0 那两个单元与 T5 同名 ⇒ skip-existing 自动去重。
t6_matrix() {
  [[ "${RUN_T6}" == 1 ]] || return 0
  local o s suf cell_id
  for o in ${OBS_SEEDS}; do
    for s in ${T6_SEEDS}; do
      suf="$(osuffix "${o}")"
      assert_obs_budget "${PROJECT_ROOT}/cases/contraction_2d/data/${CONTRA_VAL}" 5 "${suf}"         || { RUNS_GATE_FAIL=$((RUNS_GATE_FAIL + 1)); return 1; }
      budget_guard "$(unit_cost contraction_2d sparse)" || return 1
      echo "### T6 配对单元 obs_seed=${o} train_seed=${s}（后缀='${suf}'；格4=分层臂 格8=均匀臂）"
      cell_id="$(cell_of_num "${T6_REGION_CELL}")"
      train_dual "${RUN_PREFIX}_${cell_id}__s${s}__o${o}" contraction_2d "${CONTRA_TRAIN}" "${CONTRA_VAL}" \
        "obs_sparse_5pct${suf}" "obs_sparse_5pct${suf}" geometry "inlet_profile_star" \
        "${CONTRA_VAL}" "${CONTRA_TEST}" "${cell_id}" "${s}" "${o}" "${STRICT}" "strict-sparse" "" || return 1
      cell_id="$(cell_of_num "${T6_UNIFORM_CELL}")"
      train_dual "${RUN_PREFIX}_${cell_id}__s${s}__o${o}" contraction_2d "${CONTRA_TRAIN}" "${CONTRA_VAL}" \
        "obs_uniform_5pct${suf}" "obs_uniform_5pct${suf}" geometry "inlet_profile_star" \
        "${CONTRA_VAL}" "${CONTRA_TEST}" "${cell_id}" "${s}" "${o}" "${STRICT}" "strict-sparse" "" || return 1
    done
  done
  return 0
}

# ---------------------------------------------------------- 必做1 基线三件套（臂 A/B/C）
# 命名与主矩阵分开（前缀 rev2609b_）且格号续到 20-23 ⇒ 绝不与全量产物撞名。
# 臂 A：单网络联合 PINN（一次训练）；臂 B：纯数据 MLP（无物理项、无壁面包络）；
# 两臂都吃与格1/格4 **同一张观测表**（obs_seed=0），每档 5 个种子 42-46。
# 臂 C：POD + 观测点最小二乘，不训练 ⇒ 不进账本（它没有 train 阶段，塞一行 phase=train 等于骗账本），
#       单独落一个 JSON。
BL_PREFIX="rev2609b"
BL_CELLS=(
  "20|臂A 单网络联合PINN·稠密主线(对照格1)|dense|A|mainline-dense"
  "21|臂A 单网络联合PINN·分层5%(对照格4)|obs_sparse_5pct|A|strict-sparse"
  "22|臂B 纯数据MLP·稠密主线(无物理无壁面包络)|dense|B|mainline-dense"
  "23|臂B 纯数据MLP·分层5%(对照格4)|obs_sparse_5pct|B|strict-sparse"
)

baseline_matrix() {
  local row nn desc src arm preset s cell_id run_name cost
  for row in "${BL_CELLS[@]}"; do
    IFS='|' read -r nn desc src arm preset <<<"${row}"
    cell_wanted "${nn}" || continue
    cell_id="t5c$(printf '%02d' "${nn}")"
    cost="$(unit_cost contraction_2d "$([[ "${src}" == "dense" ]] && echo dense || echo sparse)")"
    for s in ${SEEDS}; do
      budget_guard "${cost}" || return 1
      run_name="${BL_PREFIX}_${cell_id}__s${s}__o0"
      echo "### 基线 格${nn} ${desc} train_seed=${s}（同观测表=${src}，与格1/格4 同源）"
      if [[ "${arm}" == "A" ]]; then
        train_joint "${run_name}" contraction_2d "${CONTRA_TRAIN}" "${CONTRA_VAL}" "${src}" geometry           "inlet_profile_star" "${cell_id}" "${s}" "0" "${preset}" "${CONTRA_TEST}" || return 1
      else
        train_mlp_with_test "${run_name}" contraction_2d "${CONTRA_TRAIN}" "${CONTRA_VAL}" "${src}"           "${cell_id}" "${s}" "0" "${CONTRA_TEST}" || return 1
      fi
    done
  done
  return 0
}

run_pod_baseline() {
  local cmd="python3 scripts/baselines_pod.py --family contraction_2d --eval-cases ${CONTRA_VAL},${CONTRA_TEST} --obs-files obs_sparse_5pct.csv --out ${OUT_DIR}/pod_baseline_contraction.json"
  if [[ "${SWEEP_DRY_RUN:-0}" == 1 ]]; then
    echo "[dry-run][armC] ${cmd}"
    echo "[dry-run][armC] 臂 C 不写 progress.jsonl（无训练阶段），产物是上面这个独立 JSON"
    return 0
  fi
  echo "[armC] POD + 观测点最小二乘（不训练网络，几秒）"
  python3 "${SWEEP_LIB_DIR}/baselines_pod.py" --family contraction_2d     --eval-cases "${CONTRA_VAL},${CONTRA_TEST}" --obs-files "obs_sparse_5pct.csv"     --out "${OUT_DIR}/pod_baseline_contraction.json"     || { echo "[FAIL] 臂 C 失败 ⇒ 基线三件套缺一臂，不能声称必做1 完成" >&2; RUNS_GATE_FAIL=$((RUNS_GATE_FAIL + 1)); return 1; }
  return 0
}

# ------------------------------------------------------------------ 主流程
preflight
n_seed=$(echo "${SEEDS}" | wc -w); n_obs=$(echo "${OBS_SEEDS}" | wc -w); n_t6s=$(echo "${T6_SEEDS}" | wc -w)

# 计划数/ETA 与 t5_matrix、t6_matrix 的铺排逐项同式，避免"计划数"与实际跑数分叉
n_planned=0; eta_sec=0
if [[ "${RUN_T5}" == 1 ]]; then
  for row in "${CELLS[@]}"; do
    IFS='|' read -r nn _d fam _tc _vc _src _fm _dr _ev _evt _sc _wp _fw kind <<<"${row}"
    cell_wanted "${nn}" || continue
    c="$(unit_cost "${fam}" "${kind}")"
    n_planned=$(( n_planned + n_seed )); eta_sec=$(( eta_sec + n_seed * c ))
  done
fi
if [[ "${RUN_T6}" == 1 ]]; then
  # obs_seed=0 的 2×n_t6s 个单元与 T5 的格4/格8 同名同配置，不重复计
  n_t6_new=$(( 2 * n_obs * n_t6s - 2 * n_t6s ))
  c="$(unit_cost contraction_2d sparse)"
  n_planned=$(( n_planned + n_t6_new )); eta_sec=$(( eta_sec + n_t6_new * c ))
fi
n_dup_t6=0
[[ "${RUN_T5}" == 1 && "${RUN_T6}" == 1 ]] && n_dup_t6=$(( 2 * n_t6s ))   # T6 的 obs_seed=0 两臂与 T5 格4/格8 同名
# 基线段（必做1 臂A/臂B）：计划数与 ETA 照 BL_CELLS 逐项累加，不留"实际要训练=0 却铺了 20 个单元"
# 这种自相矛盾的 plan 行（9/26 统括官指出）。臂 C 不训练 ⇒ 不进 n_planned，单独一行说明。
n_base_plan=0
if [[ "${RUN_BASE}" == 1 ]]; then
  for row in "${BL_CELLS[@]}"; do
    IFS='|' read -r bnn _desc bsrc _barm _bpreset <<<"${row}"
    cell_wanted "${bnn}" || continue
    bcost="$(unit_cost contraction_2d "$([[ "${bsrc}" == "dense" ]] && echo dense || echo sparse)")"
    n_base_plan=$(( n_base_plan + n_seed )); eta_sec=$(( eta_sec + n_seed * bcost ))
  done
  n_planned=$(( n_planned + n_base_plan ))
  echo "[plan] 基线三件套：本段要训练=${n_base_plan}（${#BL_CELLS[@]} 格 × ${n_seed} 种子，臂A/B 各两档）；"
  echo "[plan] 臂C（POD + 观测点最小二乘）不训练 ⇒ 不计入训练数、不写 progress.jsonl"
fi
echo "[plan] segment=${SEGMENT_TAG} 实际要训练=${n_planned}；铺排行=${n_planned}+${n_dup_t6} 条同名重复（运行时走 [skip-train]）（T5=${RUN_T5} T6=${RUN_T6}; n_seed=${n_seed} n_obs=${n_obs}）预算=${BUDGET_MIN}min dry-run=${SWEEP_DRY_RUN:-0}"
echo "[plan] 串行总墙钟≈$(( eta_sec / 60 )) min；6 路并行按 1/4 折损≈$(( eta_sec / 240 )) min ⇒ 约需 $(( eta_sec / 60 / BUDGET_MIN + 1 )) 个 ${BUDGET_MIN}min 段"
echo "[plan] 弯曲单价 ${BEND_UNIT_SEC}s（ESTIMATED=${BEND_IS_ESTIMATE}：0=已标定，1=未标定则本 ETA 不可信）"
echo "[plan] 产物目录 OUT_DIR=${OUT_DIR}（progress/logs/segment tar 都在这，与脚本仓分离；覆盖用 SWEEP_OUT_DIR）"

if [[ "${SWEEP_DRY_RUN:-0}" == 1 ]]; then
  mkdir -p "${OUT_DIR}"
  export SWEEP_DUMP_ARGV="${OUT_DIR}/dryrun_argv.txt"
  : >"${SWEEP_DUMP_ARGV}"
fi

generate_obs || { echo "[FAIL] 观测生成失败 ⇒ 前置闸门没过，不是'跑完了'" >&2; RUNS_GATE_FAIL=$((RUNS_GATE_FAIL + 1)); exit 1; }
SWEEP_STOPPED=0
if [[ "${RUN_T5}" == 1 ]]; then t5_matrix || { echo "[STOP] T5 未到预算上限即停 ⇒ 重投同段续跑" >&2; SWEEP_STOPPED=1; }; fi
if [[ "${RUN_BASE}" == 1 && "${SWEEP_STOPPED}" != 1 ]]; then baseline_matrix || { echo "[STOP] 基线段未到预算即停或被闸门拦下" >&2; SWEEP_STOPPED=1; }; fi
if [[ "${RUN_BASE}" == 1 && "${RUN_POD}" == 1 && "${SWEEP_STOPPED}" != 1 ]]; then run_pod_baseline; fi
if [[ "${RUN_T6}" == 1 && "${SWEEP_STOPPED}" != 1 ]]; then t6_matrix || { echo "[STOP] T6 提前退出" >&2; SWEEP_STOPPED=1; }; fi

if [[ "${SWEEP_DRY_RUN:-0}" == 1 ]]; then
  echo "[dry-run] argv 清单 → ${SWEEP_DUMP_ARGV}（逐行一条，已与两个原启动器逐项比对）"
  exit 0
fi

seal_segment "${EXPECTED_RUNS:-}" || exit 1
if [[ "${SWEEP_STOPPED}" == 1 ]]; then
  echo "[exit 3] 本段被预算截断，未跑完 ⇒ 用同一段名重投续跑（rc=3 专指'没跑完'，与'跑失败'区分）" >&2
  exit 3
fi
echo "[done] segment=${SEGMENT_TAG} 产物包=${OUT_DIR}/segment_${SEGMENT_TAG}.tar.gz ⇒ 立刻 push/回传；未落回持久层的结果视为未发生"
