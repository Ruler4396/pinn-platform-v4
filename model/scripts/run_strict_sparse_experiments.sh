#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

LOG_DIR="${PROJECT_ROOT}/results/pinn/logs_strict_sparse_20260503"
mkdir -p "${LOG_DIR}"

run_dual() {
  local family="$1"; shift
  local run_name="$1"; shift
  local train_cases="$1"; shift
  local val_cases="$1"; shift
  local train_source="$1"; shift
  local val_source="$1"; shift
  local feature_mode="$1"; shift
  local drop_features="$1"; shift
  local val_eval_cases="$1"; shift
  local test_eval_cases="$1"; shift
  local extra_name="$1"; shift
  local wall_mode="hard"
  if [[ "${feature_mode}" == "basic" ]]; then
    wall_mode="soft"
  fi

  local run_dir="${PROJECT_ROOT}/results/pinn/${run_name}"
  local log="${LOG_DIR}/${run_name}.log"
  if [[ ! -f "${run_dir}/metrics.json" ]]; then
    echo "[train] ${run_name}" | tee "${log}"
    nice -n 10 python3 "${PROJECT_ROOT}/scripts/train_velocity_pressure_independent_strict_sparse.py" \
      --family "${family}" \
      --run-name "${run_name}" \
      --train-cases "${train_cases}" \
      --val-cases "${val_cases}" \
      --feature-mode "${feature_mode}" \
      --drop-features "${drop_features}" \
      --train-velocity-source "${train_source}" \
      --val-velocity-source "${val_source}" \
      --train-pressure-source "${train_source}" \
      --val-pressure-source "${val_source}" \
      --velocity-hidden-layers 128,128,128 \
      --pressure-hidden-layers 128,128,128 \
      --activation silu \
      --velocity-epochs 200 \
      --pressure-epochs 200 \
      --coupling-epochs 80 \
      --velocity-lr 6e-4 \
      --pressure-lr 6e-4 \
      --coupling-velocity-lr 1e-4 \
      --coupling-pressure-lr 1e-4 \
      --wall-weight 0.0 \
      --inlet-flux-weight 0.0 \
      --continuity-weight 0.1 \
      --velocity-stage-continuity-weight 0.3 \
      --velocity-stage-momentum-weight 0.0 \
      --outlet-pressure-weight 0.0 \
      --pressure-drop-weight 0.0 \
      --pressure-stage-momentum-weight 0.5 \
      --velocity-wall-mode "${wall_mode}" \
      --hard-wall-sharpness 12 \
      --coupling-momentum-weight 10.0 \
      --coupling-continuity-weight 0.1 \
      --coupling-velocity-supervision-weight 1.0 \
      --coupling-pressure-supervision-weight 1.0 \
      --max-physics-points 512 \
      --print-every 40 \
      --strict-sparse-scalers \
      --max-retries 1 2>&1 | tee -a "${log}"
  else
    echo "[skip-train] ${run_name}" | tee "${log}"
  fi

  echo "[eval-val] ${run_name}" | tee -a "${log}"
  nice -n 10 python3 "${PROJECT_ROOT}/scripts/evaluate_velocity_pressure_independent.py" \
    --family "${family}" \
    --run-name "${run_name}" \
    --eval-cases "${val_eval_cases}" \
    --split-name "val_dense${extra_name}" \
    --eval-source dense \
    --max-retries 1 2>&1 | tee -a "${log}"

  if [[ -n "${test_eval_cases}" ]]; then
    echo "[eval-test] ${run_name}" | tee -a "${log}"
    nice -n 10 python3 "${PROJECT_ROOT}/scripts/evaluate_velocity_pressure_independent.py" \
      --family "${family}" \
      --run-name "${run_name}" \
      --eval-cases "${test_eval_cases}" \
      --split-name "test_dense${extra_name}" \
      --eval-source dense \
      --max-retries 1 2>&1 | tee -a "${log}"
  fi
}

run_single_sparse() {
  local run_name="contraction_single_mlp_geometry_sparse5_strict_20260503"
  local run_dir="${PROJECT_ROOT}/results/supervised/${run_name}"
  local log="${LOG_DIR}/${run_name}.log"
  if [[ ! -f "${run_dir}/metrics.json" ]]; then
    echo "[train-single] ${run_name}" | tee "${log}"
    nice -n 10 python3 "${PROJECT_ROOT}/scripts/train_supervised.py" \
      --family contraction_2d \
      --run-name "${run_name}" \
      --train-cases C-base,C-train-1,C-train-2,C-train-3,C-train-4,C-train-5 \
      --val-cases C-val \
      --feature-mode geometry \
      --drop-features inlet_profile_star \
      --train-observation-source obs_sparse_5pct \
      --val-observation-source obs_sparse_5pct \
      --hidden-layers 128,128,128 \
      --activation silu \
      --lr 6e-4 \
      --max-epochs 2000 \
      --patience 200 \
      --print-every 100 \
      --max-retries 1 2>&1 | tee -a "${log}"
  else
    echo "[skip-single] ${run_name}" | tee "${log}"
  fi
  echo "[eval-single-val-dense] ${run_name}" | tee -a "${log}"
  nice -n 10 python3 "${PROJECT_ROOT}/scripts/evaluate_supervised.py" \
    --family contraction_2d \
    --run-name "${run_name}" \
    --eval-cases C-val \
    --split-name val_dense \
    --max-retries 1 2>&1 | tee -a "${log}"
}

CONTRA_TRAIN="C-base,C-train-1,C-train-2,C-train-3,C-train-4,C-train-5"
BEND_BLUNT_TRAIN="B-base__ip_blunted,B-train-1__ip_blunted,B-train-2__ip_blunted,B-train-3__ip_blunted"

run_dual contraction_2d contraction_strict_geometry_sparse1_stagepde_20260503 "${CONTRA_TRAIN}" C-val obs_sparse_1pct obs_sparse_1pct geometry inlet_profile_star C-val C-test-1,C-test-2 ""
run_dual contraction_2d contraction_strict_geometry_sparse5_stagepde_20260503 "${CONTRA_TRAIN}" C-val obs_sparse_5pct obs_sparse_5pct geometry inlet_profile_star C-val C-test-1,C-test-2 ""
run_dual contraction_2d contraction_strict_geometry_sparse10_stagepde_20260503 "${CONTRA_TRAIN}" C-val obs_sparse_10pct obs_sparse_10pct geometry inlet_profile_star C-val C-test-1,C-test-2 ""
run_dual contraction_2d contraction_strict_geometry_sparse15_stagepde_20260503 "${CONTRA_TRAIN}" C-val obs_sparse_15pct obs_sparse_15pct geometry inlet_profile_star C-val C-test-1,C-test-2 ""

run_dual contraction_2d contraction_strict_geometry_uniform1_stagepde_20260503 "${CONTRA_TRAIN}" C-val obs_uniform_1pct obs_uniform_1pct geometry inlet_profile_star C-val C-test-1,C-test-2 ""
run_dual contraction_2d contraction_strict_geometry_uniform5_stagepde_20260503 "${CONTRA_TRAIN}" C-val obs_uniform_5pct obs_uniform_5pct geometry inlet_profile_star C-val C-test-1,C-test-2 ""
run_dual contraction_2d contraction_strict_geometry_uniform10_stagepde_20260503 "${CONTRA_TRAIN}" C-val obs_uniform_10pct obs_uniform_10pct geometry inlet_profile_star C-val C-test-1,C-test-2 ""
run_dual contraction_2d contraction_strict_geometry_uniform15_stagepde_20260503 "${CONTRA_TRAIN}" C-val obs_uniform_15pct obs_uniform_15pct geometry inlet_profile_star C-val C-test-1,C-test-2 ""

run_dual contraction_2d contraction_strict_geometry_sparse5noise3_stagepde_20260503 "${CONTRA_TRAIN}" C-val obs_sparse_5pct_noise_3pct obs_sparse_5pct_noise_3pct geometry inlet_profile_star C-val C-test-1,C-test-2 ""
run_dual contraction_2d contraction_strict_basic_sparse5_stagepde_20260503 "${CONTRA_TRAIN}" C-val obs_sparse_5pct obs_sparse_5pct basic "" C-val C-test-1,C-test-2 ""

run_dual bend_2d bend_strict_blunted_sparse1_stagepde_20260503 "${BEND_BLUNT_TRAIN}" B-val__ip_blunted obs_sparse_1pct obs_sparse_1pct geometry inlet_profile_star B-val__ip_blunted B-test-1__ip_blunted ""
run_dual bend_2d bend_strict_blunted_sparse5_stagepde_20260503 "${BEND_BLUNT_TRAIN}" B-val__ip_blunted obs_sparse_5pct obs_sparse_5pct geometry inlet_profile_star B-val__ip_blunted B-test-1__ip_blunted ""
run_dual bend_2d bend_strict_blunted_sparse10_stagepde_20260503 "${BEND_BLUNT_TRAIN}" B-val__ip_blunted obs_sparse_10pct obs_sparse_10pct geometry inlet_profile_star B-val__ip_blunted B-test-1__ip_blunted ""
run_dual bend_2d bend_strict_blunted_sparse15_stagepde_20260503 "${BEND_BLUNT_TRAIN}" B-val__ip_blunted obs_sparse_15pct obs_sparse_15pct geometry inlet_profile_star B-val__ip_blunted B-test-1__ip_blunted ""

run_single_sparse

echo "[done] strict sparse experiments"
