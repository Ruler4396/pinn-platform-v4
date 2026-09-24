#!/usr/bin/env bash
# sweep_lib.sh — T5/T6 扫描的公共函数库（被 sweep_t5.sh source，不单独执行）。
#
# 计时口径：只用 date +%s%N 的整数运算。实例镜像里**没有 /usr/bin/time、也没有 bc**
# （T0 实测：第一轮因此 rc=127），所以不得引入这两者。
# 产物保护：绝不 rm -rf 输出目录；已有 metrics.json 的训练单元直接跳过（可续跑）。
# 失败不得静默：段末必须打印 "N runs / M failures"，M 读不到就打印 INVALID 并 exit 1。

set -euo pipefail

if [[ -n "${SWEEP_LIB_LOADED:-}" ]]; then
  return 0
fi
SWEEP_LIB_LOADED=1

SWEEP_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SWEEP_LIB_DIR}/.." && pwd)"           # = model/
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
OUT_DIR="${SWEEP_OUT_DIR:-${REPO_ROOT}/out}"
LOG_DIR="${OUT_DIR}/logs"
PROGRESS="${OUT_DIR}/progress.jsonl"
SEGMENT_TAG="${SWEEP_SEGMENT:-seg00}"

# 单价（秒）。训练单价来自仓库落盘的 /usr/bin/time 行，评估单价来自 T0 实测。
UNIT_DENSE_TRAIN_SEC="${UNIT_DENSE_TRAIN_SEC:-98.80}"       # mainline_v4.log:46 Elapsed 1:38.80
UNIT_SPARSE_TRAIN_SEC="${UNIT_SPARSE_TRAIN_SEC:-59.52}"     # …sparse5clean_stagepde_v4.log:46 0:59.52
UNIT_EVAL_SEC="${UNIT_EVAL_SEC:-1.55}"                      # T0 实测 1.48–1.59 s
# 弯曲族单价未标定 ⇒ 留环境变量。未设时按 1.3× 收缩稀疏估，并在段首打印 ESTIMATED
BEND_UNIT_SEC="${BEND_UNIT_SEC:-}"
BEND_IS_ESTIMATE=0
if [[ -z "${BEND_UNIT_SEC}" ]]; then
  BEND_UNIT_SEC="78"
  BEND_IS_ESTIMATE=1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------- 计时
_now_ns() {
  local raw
  raw="$(date +%s%N)"
  case "${raw}" in
    *N*|*n*) echo "$(( $(date +%s) * 1000000000 ))" ;;        # 不支持 %N 的 date
    *) echo "${raw}" ;;
  esac
}

elapsed_sec() {  # $1=start_ns → 已用秒（整数）
  local now start
  now="$(_now_ns)"; start="$1"
  echo $(( (now - start) / 1000000000 ))
}

elapsed_ms() {  # $1=start_ns → 毫秒
  local now start
  now="$(_now_ns)"; start="$1"
  echo $(( (now - start) / 1000000 ))
}

# ---------------------------------------------------------------- 自检
# 递归删除守卫：本套脚本靠"产物已存在就跳过"续跑，任何递归删除都会把要 resume 的产物吃掉。
# 模式串用变量拼出来，避免本文件的说明文字自己被匹配上（说明文字要留着，改指标不改注释）。
_RM_PAT='rm[[:space:]][[:space:]]*-[rRf]'
lint_no_recursive_delete() {  # $1=file → 命中则返回 1
  local f="$1" hits
  [[ -f "${f}" ]] || return 0
  # 剥掉整行注释再匹配：否则这条规则会逼人删掉说明文字来变绿
  hits="$(sed 's/^[[:space:]]*#.*//' "${f}" | grep -nE "${_RM_PAT}" || true)"
  if [[ -n "${hits}" ]]; then
    echo "[FAIL] ${f} 里出现递归删除 —— 续跑依赖已存在产物，禁止删输出目录" >&2
    printf '%s\n' "${hits}" >&2
    return 1
  fi
  return 0
}

preflight() {
  mkdir -p "${OUT_DIR}" "${LOG_DIR}"
  local f
  for f in "${SWEEP_LIB_DIR}/sweep_t5.sh" "${SWEEP_LIB_DIR}/sweep_lib.sh"; do
    lint_no_recursive_delete "${f}" || exit 1
  done
  command -v python3 >/dev/null 2>&1 || { echo "[FAIL] python3 不存在" >&2; exit 1; }
  local nver=""
  if [[ "${SWEEP_DRY_RUN:-0}" == 1 ]]; then
    echo "[preflight] dry-run 模式：跳过 torch/numpy 运行时检查（只校验 argv 与文件计划）"
  else
    python3 -c "import torch, numpy, pandas" 2>/dev/null \
      || { echo "[FAIL] torch/numpy/pandas 不齐。镜像自带 numpy 1.26.4 会在 train_supervised.py:588 的 np.trapezoid 上崩，必须 pip3 install numpy==2.2.6" >&2; exit 1; }
    nver="$(python3 -c 'import numpy;print(numpy.__version__)')"
    if [[ "${nver}" != 2.* ]]; then
      echo "[FAIL] numpy=${nver}，代码用到 np.trapezoid（numpy>=2.0），会 AttributeError" >&2
      exit 1
    fi
  fi
  if [[ ! -e /usr/bin/time ]]; then
    echo "[note] 本实例无 /usr/bin/time，墙钟一律用 date +%s%N 整数运算（与论文 4.2 引用的旧口径不同，须在表注声明）"
  fi
  echo "[preflight] PROJECT_ROOT=${PROJECT_ROOT} OUT_DIR=${OUT_DIR} segment=${SEGMENT_TAG} numpy=${nver:-未检}"
  echo "[preflight] 单价：稠密 ${UNIT_DENSE_TRAIN_SEC}s 稀疏 ${UNIT_SPARSE_TRAIN_SEC}s 弯曲 ${BEND_UNIT_SEC}s 评估 ${UNIT_EVAL_SEC}s"
  if [[ "${BEND_IS_ESTIMATE}" == 1 ]]; then
    echo "[preflight][ESTIMATED] BEND_UNIT_SEC 未标定，弯曲单元按 ${BEND_UNIT_SEC}s 估 ⇒ 本段的 ETA 与预算余量对弯曲格不可信，标定后重投"
  fi
}

# ---------------------------------------------------------------- 记账
RUNS_DONE=0
RUNS_FAIL=0
RUNS_SKIP=0

_progress_append() {  # $1=run $2=family $3=cell $4=train_seed $5=obs_seed $6=phase $7=wall_ms $8=rc $9=metrics_json
  python3 - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${SEGMENT_TAG}" "${PROGRESS}" <<'PY'
import json, sys, os
run, family, cell, tseed, oseed, phase, wall_ms, rc, metrics_raw, seg, path = sys.argv[1:12]
try:
    metrics = json.loads(metrics_raw) if metrics_raw.strip() else {}
except Exception:
    metrics = {"_metrics_parse_error": metrics_raw[:200]}
rec = {"run": run, "family": family, "cell": cell, "train_seed": int(tseed),
       "obs_seed": int(oseed), "phase": phase, "wall_ms": int(wall_ms), "rc": int(rc),
       "segment": seg, "ts": __import__("datetime").datetime.now().isoformat(timespec="seconds")}
rec.update(metrics)
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fh.flush()
PY
}

_run_metrics_json() {  # $1=run_dir → 把关键指标读成一行 JSON（读不到就返回 {}）
  python3 - "$1" <<'PY'
import json, os, sys
run_dir = sys.argv[1]
out = {}
def first(path, keys):
    if not os.path.exists(path):
        return
    try:
        j = json.load(open(path, encoding="utf-8"))
    except Exception:
        return
    for k in keys:
        node = j
        for part in k.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                node = None
                break
        if isinstance(node, (int, float)):
            out[k.split(".")[-1]] = round(float(node), 8)
m = os.path.join(run_dir, "metrics.json")
first(m, ["最终验证指标.rel_l2_u", "最终验证指标.rel_l2_v",
          "最终验证指标.rel_l2_p", "最终验证指标.rel_l2_speed",
          "最终验证指标.max_p_err_over_p_range"])
ev = os.path.join(run_dir, "evaluations")
first(os.path.join(ev, "metrics_val_dense.json"),
      ["global_metrics.rel_l2_u", "global_metrics.rel_l2_speed",
       "global_metrics.rel_l2_p", "case_metrics.0.pressure_drop_rel_error"])
print(json.dumps(out, ensure_ascii=False))
PY
}

# ---------------------------------------------------------------- 观测生成
ensure_observations() {  # $1=family $2=rates $3=obs_seeds $4=cases
  local family="$1" rates="$2" seeds="$3" cases="$4"
  if [[ "${SWEEP_DRY_RUN:-0}" == 1 ]]; then
    echo "[dry-run][obs] python3 scripts/generate_observations_seeded.py --family ${family} --cases ${cases} --rates ${rates} --obs-seeds ${seeds} --allow-existing"
    return 0
  fi
  python3 "${SWEEP_LIB_DIR}/generate_observations_seeded.py" \
    --family "${family}" --cases "${cases}" --rates "${rates}" --obs-seeds "${seeds}" --allow-existing
}

# 两臂预算对齐断言：同一 obs_seed 下分层与均匀的**总点数**必须相同（区域配额按设计不同）
assert_obs_budget() {  # $1=case_dir $2=rate_pct $3=obs_seed_suffix(可为空)
  local case_dir="$1" pct="$2" suffix="$3"
  [[ "${SWEEP_DRY_RUN:-0}" == 1 ]] && { echo "[dry-run][assert] 预算对齐 ${case_dir} ${pct}pct${suffix}"; return 0; }
  python3 - "${case_dir}" "${pct}" "${suffix}" <<'PY'
import os, sys
case_dir, pct, suffix = sys.argv[1], sys.argv[2], sys.argv[3]
def count(stem):
    p = os.path.join(case_dir, f"{stem}.csv")
    if not os.path.exists(p):
        return None, p
    with open(p, encoding="utf-8") as fh:
        return sum(1 for _ in fh) - 1, p
n_reg, p_reg = count(f"obs_sparse_{pct}pct{suffix}")
n_uni, p_uni = count(f"obs_uniform_{pct}pct{suffix}")
if n_reg is None or n_uni is None:
    print(f"[FAIL] 观测文件缺失：{p_reg if n_reg is None else p_uni} ⇒ 先跑 obs 生成", file=sys.stderr)
    raise SystemExit(1)
if n_reg != n_uni:
    print(f"[FAIL] 观测预算不对齐：{n_reg} != {n_uni}（{case_dir} {pct}%）⇒ 配对比较不成立", file=sys.stderr)
    raise SystemExit(1)
print(f"[obs-budget] {os.path.basename(case_dir)} {pct}%{suffix} 两臂同为 {n_reg} 点")
PY
}

# ---------------------------------------------------------------- 训练+评估
# 参数顺序固定照抄 run_strict_sparse_experiments.sh:38-76，只额外加 --seed、
# 源名可带 __s{k} 后缀、run-name 带种子标记。
train_dual() {  # $1..: name|family|train_cases|val_cases|src_train|src_val|feature_mode|drop_features|eval_val_cases|eval_test_cases|cell|train_seed|obs_seed
  local run_name="$1" family="$2" train_cases="$3" val_cases="$4"
  local src="$5" src_val="$6" feature_mode="$7" drop_features="$8"
  local eval_val="$9" eval_test="${10}" cell="${11}" train_seed="${12}" obs_seed="${13}"
  local script="${14:-train_velocity_pressure_independent_strict_sparse.py}"
  local weights_preset="${15:-strict-sparse}"
  local wall_mode="hard"
  local force_wall_mode="${16:-}"
  [[ -n "${force_wall_mode}" ]] && wall_mode="${force_wall_mode}"
  [[ -z "${force_wall_mode}" && "${feature_mode}" == "basic" ]] && wall_mode="soft"

  # 权重档位：strict-sparse = 论文 5.3-5.8 的实际口径（三个稠密真值损失全 0）
  #           mainline-dense = 论文表4-4 的稠密主线口径
  local inlet_w outlet_w drop_w vsc_w psm_w strict_flag print_every
  print_every=40
  [[ "${weights_preset}" != "strict-sparse" ]] && print_every=20
  case "${weights_preset}" in
    strict-sparse) inlet_w="0.0"; outlet_w="0.0"; drop_w="0.0"; vsc_w="0.3"; psm_w="0.5"; strict_flag="--strict-sparse-scalers" ;;
    mainline-dense) inlet_w="0.5"; outlet_w="1e-4"; drop_w="1.0"; vsc_w="0.3"; psm_w="0.5"; strict_flag="" ;;
    no-stage-pde)  inlet_w="0.5"; outlet_w="1e-4"; drop_w="1.0"; vsc_w="0.0"; psm_w="0.0"; strict_flag="" ;;
    *) echo "[FAIL] 未知权重档位 ${weights_preset}" >&2; return 2 ;;
  esac

  local run_dir="${PROJECT_ROOT}/results/pinn/${run_name}"
  local log="${LOG_DIR}/${run_name}.log"
  local t0 rc=0 wall_ms=0
  mkdir -p "${run_dir}" "${LOG_DIR}"

  local drop_display="${drop_features}"
  [[ -z "${drop_features}" ]] && drop_display='""'      # 实参是一个空串，打印成可粘贴的 ""

  if [[ "${SWEEP_DRY_RUN:-0}" == 1 ]]; then
    echo "[dry-run][train] ${run_name} cell=${cell} ts=${train_seed} os=${obs_seed} script=${script} wall_mode=${wall_mode} weights=${weights_preset}"
    local argv="python3 scripts/${script} --family ${family} --run-name ${run_name} --seed ${train_seed} --train-cases ${train_cases} --val-cases ${val_cases} --feature-mode ${feature_mode} --drop-features ${drop_display} --train-velocity-source ${src} --val-velocity-source ${src_val} --train-pressure-source ${src} --val-pressure-source ${src_val} --velocity-hidden-layers 128,128,128 --pressure-hidden-layers 128,128,128 --activation silu --velocity-epochs 200 --pressure-epochs 200 --coupling-epochs 80 --velocity-lr 6e-4 --pressure-lr 6e-4 --coupling-velocity-lr 1e-4 --coupling-pressure-lr 1e-4 --wall-weight 0.0 --inlet-flux-weight ${inlet_w} --continuity-weight 0.1 --velocity-stage-continuity-weight ${vsc_w} --velocity-stage-momentum-weight 0.0 --outlet-pressure-weight ${outlet_w} --pressure-drop-weight ${drop_w} --pressure-stage-momentum-weight ${psm_w} --velocity-wall-mode ${wall_mode} --hard-wall-sharpness 12 --coupling-momentum-weight 10.0 --coupling-continuity-weight 0.1 --coupling-velocity-supervision-weight 1.0 --coupling-pressure-supervision-weight 1.0 --max-physics-points 512 --print-every ${print_every} ${strict_flag} --max-retries 1"
    argv="$(printf '%s' "${argv}" | tr -s ' ')"
    echo "  ${argv}"
    [[ "${SWEEP_DUMP_ARGV:-}" != "" ]] && printf '%s\n' "${argv}" >>"${SWEEP_DUMP_ARGV}"
    echo "[dry-run][eval] ${run_name} val=${eval_val} test=${eval_test:-<none>}"
    return 0
  fi

  if [[ -f "${run_dir}/metrics.json" ]]; then
    echo "[skip-train] ${run_name}"
    RUNS_SKIP=$((RUNS_SKIP + 1))
  else
    echo "[train] ${run_name} (cell=${cell} ts=${train_seed} os=${obs_seed})" | tee "${log}"
    t0="$(_now_ns)"
    if nice -n 10 python3 "${SWEEP_LIB_DIR}/${script}" \
        --family "${family}" --run-name "${run_name}" --seed "${train_seed}" \
        --train-cases "${train_cases}" --val-cases "${val_cases}" \
        --feature-mode "${feature_mode}" --drop-features "${drop_features}" \
        --train-velocity-source "${src}" --val-velocity-source "${src_val}" \
        --train-pressure-source "${src}" --val-pressure-source "${src_val}" \
        --velocity-hidden-layers 128,128,128 --pressure-hidden-layers 128,128,128 \
        --activation silu --velocity-epochs 200 --pressure-epochs 200 --coupling-epochs 80 \
        --velocity-lr 6e-4 --pressure-lr 6e-4 --coupling-velocity-lr 1e-4 --coupling-pressure-lr 1e-4 \
        --wall-weight 0.0 --inlet-flux-weight "${inlet_w}" --continuity-weight 0.1 \
        --velocity-stage-continuity-weight "${vsc_w}" --velocity-stage-momentum-weight 0.0 \
        --outlet-pressure-weight "${outlet_w}" --pressure-drop-weight "${drop_w}" \
        --pressure-stage-momentum-weight "${psm_w}" \
        --velocity-wall-mode "${wall_mode}" --hard-wall-sharpness 12 \
        --coupling-momentum-weight 10.0 --coupling-continuity-weight 0.1 \
        --coupling-velocity-supervision-weight 1.0 --coupling-pressure-supervision-weight 1.0 \
        --max-physics-points 512 --print-every ${print_every} ${strict_flag} --max-retries 1 >>"${log}" 2>&1; then
      rc=0
    else
      rc=$?
    fi
    wall_ms="$(elapsed_ms "${t0}")"
    # rc=0 但 metrics.json 没落盘，同样是失败：账上记成非零，否则段末对账会假绿
    local book_rc="${rc}"
    [[ "${rc}" == 0 && ! -f "${run_dir}/metrics.json" ]] && book_rc=90
    _progress_append "${run_name}" "${family}" "${cell}" "${train_seed}" "${obs_seed}" train "${wall_ms}" "${book_rc}" "$(_run_metrics_json "${run_dir}")"
    if [[ "${book_rc}" != 0 ]]; then
      echo "[FAIL] train rc=${rc}→记账 ${book_rc} run=${run_name}（metrics.json 存在=$([[ -f ${run_dir}/metrics.json ]] && echo yes || echo no)）见 ${log}" >&2
      RUNS_FAIL=$((RUNS_FAIL + 1))
      return 1
    fi
    RUNS_DONE=$((RUNS_DONE + 1))
    echo "[ok-train] ${run_name} wall_ms=${wall_ms}"
  fi

  eval_run "${family}" "${run_name}" "${eval_val}" "${eval_test}" "${cell}" "${train_seed}" "${obs_seed}" "${src}"
}

eval_run() {  # family run_name eval_val eval_test cell train_seed obs_seed eval_source
  local family="$1" run_name="$2" eval_val="$3" eval_test="$4"
  local cell="$5" train_seed="$6" obs_seed="$7"
  local run_dir="${PROJECT_ROOT}/results/pinn/${run_name}"
  local log="${LOG_DIR}/${run_name}_eval.log"

  if [[ "${SWEEP_DRY_RUN:-0}" == 1 ]]; then
    echo "[dry-run][eval] ${run_name} val=${eval_val} test=${eval_test:-<none>}"
    return 0
  fi
  local t0 rc=0
  : >"${log}"
  echo "[eval-val] ${run_name}" | tee -a "${log}"
  t0="$(_now_ns)"
  nice -n 10 python3 "${SWEEP_LIB_DIR}/evaluate_velocity_pressure_independent.py" \
    --family "${family}" --run-name "${run_name}" --eval-cases "${eval_val}" \
    --split-name "val_dense" --eval-source dense --max-retries 1 >>"${log}" 2>&1 || rc=$?
  local wall_ms; wall_ms="$(elapsed_ms "${t0}")"
  if [[ "${rc}" != 0 ]]; then
    echo "[FAIL] eval-val rc=${rc} ${run_name} 见 ${log}" >&2
    _progress_append "${run_name}" "${family}" "${cell}" "${train_seed}" "${obs_seed}" eval_val "${wall_ms}" "${rc}" "{}"
    RUNS_FAIL=$((RUNS_FAIL + 1))
    return 1
  fi
  _progress_append "${run_name}" "${family}" "${cell}" "${train_seed}" "${obs_seed}" eval_val "${wall_ms}" 0 "$(_run_metrics_json "${run_dir}")"

  if [[ -n "${eval_test}" ]]; then
    rc=0; t0="$(_now_ns)"
    nice -n 10 python3 "${SWEEP_LIB_DIR}/evaluate_velocity_pressure_independent.py" \
      --family "${family}" --run-name "${run_name}" --eval-cases "${eval_test}" \
      --split-name "test_dense" --eval-source dense --max-retries 1 >>"${log}" 2>&1 || rc=$?
    wall_ms="$(elapsed_ms "${t0}")"
    if [[ "${rc}" != 0 ]]; then
      echo "[FAIL] eval-test rc=${rc} ${run_name} 见 ${log}" >&2
      _progress_append "${run_name}" "${family}" "${cell}" "${train_seed}" "${obs_seed}" eval_test "${wall_ms}" "${rc}" "{}"
      RUNS_FAIL=$((RUNS_FAIL + 1))
      return 1
    fi
    _progress_append "${run_name}" "${family}" "${cell}" "${train_seed}" "${obs_seed}" eval_test "${wall_ms}" 0 "$(_run_metrics_json "${run_dir}")"
  fi
  return 0
}

# 纯监督 MLP（图5-18 的现有对照臂，只补种子与分区口径，不改结构）
train_mlp() {
  local run_name="$1" family="$2" train_cases="$3" val_cases="$4" src="$5" cell="$6" train_seed="$7" obs_seed="$8"
  local run_dir="${PROJECT_ROOT}/results/supervised/${run_name}"
  local log="${LOG_DIR}/${run_name}.log"
  if [[ "${SWEEP_DRY_RUN:-0}" == 1 ]]; then
    echo "[dry-run][train-mlp] python3 scripts/train_supervised.py --family ${family} --run-name ${run_name} --seed ${train_seed} --train-cases ${train_cases} --val-cases ${val_cases} --feature-mode geometry --drop-features inlet_profile_star --train-observation-source ${src} --val-observation-source ${src} --hidden-layers 128,128,128 --activation silu --lr 6e-4 --max-epochs 2000 --patience 200 --print-every 100 --max-retries 1"
    return 0
  fi
  if [[ -f "${run_dir}/metrics.json" ]]; then
    echo "[skip-train-mlp] ${run_name}"; RUNS_SKIP=$((RUNS_SKIP + 1)); return 0
  fi
  mkdir -p "${run_dir}" "${LOG_DIR}"
  echo "[train-mlp] ${run_name}" | tee "${log}"
  local t0 rc=0; t0="$(_now_ns)"
  nice -n 10 python3 "${SWEEP_LIB_DIR}/train_supervised.py" \
    --family "${family}" --run-name "${run_name}" --seed "${train_seed}" \
    --train-cases "${train_cases}" --val-cases "${val_cases}" \
    --feature-mode geometry --drop-features inlet_profile_star \
    --train-observation-source "${src}" --val-observation-source "${src}" \
    --hidden-layers 128,128,128 --activation silu --lr 6e-4 \
    --max-epochs 2000 --patience 200 --print-every 100 --max-retries 1 >>"${log}" 2>&1 || rc=$?
  local wall_ms; wall_ms="$(elapsed_ms "${t0}")"
  local book_rc="${rc}"
  [[ "${rc}" == 0 && ! -f "${run_dir}/metrics.json" ]] && book_rc=90
  _progress_append "${run_name}" "${family}" "${cell}" "${train_seed}" "${obs_seed}" train "${wall_ms}" "${book_rc}" "$(_run_metrics_json "${run_dir}")"
  if [[ "${book_rc}" != 0 ]]; then
    echo "[FAIL] train-mlp rc=${rc}→记账 ${book_rc} ${run_name}" >&2; RUNS_FAIL=$((RUNS_FAIL + 1)); return 1
  fi
  RUNS_DONE=$((RUNS_DONE + 1))
  nice -n 10 python3 "${SWEEP_LIB_DIR}/evaluate_supervised.py" --family "${family}" --run-name "${run_name}" \
    --eval-cases "${val_cases}" --split-name val_dense --max-retries 1 >>"${log}" 2>&1 \
    || { echo "[FAIL] eval-mlp ${run_name}" >&2; RUNS_FAIL=$((RUNS_FAIL + 1)); return 1; }
  return 0
}

# ---------------------------------------------------------------- 段末收尾
seal_segment() {
  local expected="${1:-}"
  local -a payload=()
  [[ -f "${PROGRESS}" ]] && payload+=("${PROGRESS}")
  [[ -d "${LOG_DIR}" ]] && payload+=("logs")
  if [[ ${#payload[@]} -gt 0 && "${SWEEP_DRY_RUN:-0}" != 1 ]]; then
    ( cd "${OUT_DIR}" && tar czf "segment_${SEGMENT_TAG}.tar.gz" "${payload[@]}" ) \
      || echo "[WARN] tar 失败，产物仍在 ${OUT_DIR}，请手工回传" >&2
  fi
  # 对账：内存计数必须与 progress.jsonl 落盘的账吻合。
  # grep -c 在"没匹配"时返回合法的 0，所以"本段 0 行"不能当通过 —— 那正是这里要判 INVALID 的情形。
  python3 - "${PROGRESS}" "${SEGMENT_TAG}" "${RUNS_DONE}" "${RUNS_FAIL}" "${RUNS_SKIP}" "${expected}" \
    "${SWEEP_FORCE_LEDGER_BREAK:-}" <<'LEDGER_PY'
import json, os, sys
path, seg, done, fail, skip, expected, breakage = sys.argv[1:8]
done, fail, skip = int(done), int(fail), int(skip)
if breakage == "1":          # 自测开关：人为把账本读废，确认这条闸会变红
    print("[self-test] 已按要求模拟账本损坏", file=sys.stderr)
    raise SystemExit(1)
if not os.path.exists(path):
    print("INVALID: 记账文件 %s 不存在 ⇒ 本段什么都没记上，不能声称跑过" % path, file=sys.stderr)
    raise SystemExit(1)
rows, bad_lines = [], 0
for line in open(path, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except Exception:
        bad_lines += 1
        continue
    if rec.get("segment") == seg:
        rows.append(rec)
if bad_lines:
    print("INVALID: 账本里有 %d 行无法解析（被截断？）⇒ 本段结果不可信" % bad_lines, file=sys.stderr)
    raise SystemExit(1)
if not rows:
    print("INVALID: 段 %s 在账本里 0 行 ⇒ 这不是'全部通过'，是没记上账或段名写错" % seg, file=sys.stderr)
    raise SystemExit(1)
train = [r for r in rows if str(r.get("phase", "")).startswith("train")]
ok_train = sum(1 for r in train if r.get("rc") == 0)
bad_train = len(train) - ok_train
bad_any = sum(1 for r in rows if r.get("rc") != 0)
problems = []
if ok_train != done:
    problems.append("训练成功数 内存=%d 账本=%d" % (done, ok_train))
if bad_any != fail:
    problems.append("失败数 内存=%d 账本=%d" % (fail, bad_any))
print("%d runs / %d failures / %d skipped  (账本: 段 %s 共 %d 行；训练 %d 行 = %d 成 + %d 败；全阶段 rc!=0 共 %d)"
      % (done, fail, skip, seg, len(rows), len(train), ok_train, bad_train, bad_any))
if problems:
    print("INVALID: 内存计数与落盘账对不上：%s ⇒ 判本段不可信" % "; ".join(problems), file=sys.stderr)
    raise SystemExit(1)
if fail:
    print("[FAIL] 本段有 %d 个失败单元" % fail, file=sys.stderr)
    raise SystemExit(1)
if expected and done != int(expected):
    print("[FAIL] 完成 %d ≠ 期望 %s（EXPECTED_RUNS=<n> 传本段应完成数）" % (done, expected), file=sys.stderr)
    raise SystemExit(1)
LEDGER_PY
}
