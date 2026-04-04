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

RUN_NAME="contraction_independent_geometry_notemplate_stagepde_mainline_v4"
TRAIN_LOG="${PROJECT_ROOT}/results/pinn/${RUN_NAME}.log"
VAL_LOG="${PROJECT_ROOT}/results/pinn/${RUN_NAME}_eval_val.log"
TEST_LOG="${PROJECT_ROOT}/results/pinn/${RUN_NAME}_eval_test.log"

COMMON_ARGS=(
  --family contraction_2d
  --train-cases C-base,C-train-1,C-train-2,C-train-3,C-train-4,C-train-5
  --val-cases C-val
  --feature-mode geometry
  --drop-features inlet_profile_star
  --train-velocity-source dense
  --val-velocity-source dense
  --train-pressure-source dense
  --val-pressure-source dense
  --velocity-hidden-layers 128,128,128
  --pressure-hidden-layers 128,128,128
  --activation silu
  --velocity-epochs 200
  --pressure-epochs 200
  --coupling-epochs 80
  --velocity-lr 6e-4
  --pressure-lr 6e-4
  --coupling-velocity-lr 1e-4
  --coupling-pressure-lr 1e-4
  --wall-weight 0.0
  --inlet-flux-weight 0.5
  --continuity-weight 0.1
  --velocity-stage-continuity-weight 0.3
  --velocity-stage-momentum-weight 0.0
  --outlet-pressure-weight 1e-4
  --pressure-drop-weight 1.0
  --pressure-stage-momentum-weight 0.5
  --velocity-wall-mode hard
  --hard-wall-sharpness 12
  --coupling-momentum-weight 10.0
  --coupling-continuity-weight 0.1
  --coupling-velocity-supervision-weight 1.0
  --coupling-pressure-supervision-weight 1.0
  --max-physics-points 512
  --print-every 20
  --max-retries 1
)

if [[ ! -f "${PROJECT_ROOT}/results/pinn/${RUN_NAME}/metrics.json" ]]; then
  echo "[run] ${RUN_NAME}" | tee "${TRAIN_LOG}"
  /usr/bin/time -v nice -n 10 python3 "${PROJECT_ROOT}/scripts/train_velocity_pressure_independent.py" \
    --run-name "${RUN_NAME}" \
    "${COMMON_ARGS[@]}" 2>&1 | tee -a "${TRAIN_LOG}"
else
  echo "[skip] ${RUN_NAME} 已存在，跳过训练" | tee "${TRAIN_LOG}"
fi

echo "[eval] val" | tee "${VAL_LOG}"
/usr/bin/time -v nice -n 10 python3 "${PROJECT_ROOT}/scripts/evaluate_velocity_pressure_independent.py" \
  --family contraction_2d \
  --run-name "${RUN_NAME}" \
  --eval-cases C-val \
  --split-name val \
  --eval-source dense \
  --max-retries 1 2>&1 | tee -a "${VAL_LOG}"

echo "[eval] test" | tee "${TEST_LOG}"
/usr/bin/time -v nice -n 10 python3 "${PROJECT_ROOT}/scripts/evaluate_velocity_pressure_independent.py" \
  --family contraction_2d \
  --run-name "${RUN_NAME}" \
  --eval-cases C-test-1,C-test-2 \
  --split-name test \
  --eval-source dense \
  --max-retries 1 2>&1 | tee -a "${TEST_LOG}"
