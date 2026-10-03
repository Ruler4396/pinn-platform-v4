#!/usr/bin/env bash
# ops/run_ns5vonly.py 的驱动 —— Stage 5：只看速度反推压力 × 经典压力重建同框。
# 判据/格点/控制在 paper-route2/速度推压力-经典基线同框-预注册设计-20261003.md，本件只执行。
#
#   preflight  运行库与装置字节核（FreeFEM 的 sonames 走实例本地目录 + LD_LIBRARY_PATH）
#   obs        给 7 工况 × 2 档补 1pct/15pct 的 NS 取值观测表，再派生"速度-only"副本
#   selftests  判决器自检 / RBF 正对照 / C7 三桩 / C10 已知答案+必红
#   c6         速度-only 表逐格核（同点数、坐标逐位相同、压力列 0 个有限值）
#   smoke      1 格短训练，并**核 config.json 里的四个 source 旗子确实等于速度-only 表**
#              （Stage 4 的教训：照抄 argv 漏抄选择器 ⇒ 所谓"5% 稀疏档"其实吃的是 dense）
#   matrix     PINN 48 格（main 36 + solo 12）
#   baseline   经典臂 12 次求解 + C9 上限对照 4 次
#   score      两臂同尺现算 -> matrix_pinn.tsv
#   judge      K1 / K2 / K3 + 四象限
set -uo pipefail

WS="${WS:-/mnt/workspace/pinn-repro-2026}"
OPS="${OPS:-/mnt/workspace/_ops/ns5/20261003}"
S="$WS/model/scripts"
RES="$WS/model/results/pinn"
LOGD="$OPS/out"
DATA="$WS/model/cases/contraction_2d/data"
FFHOME="$OPS/ffhome/usr/lib/x86_64-linux-gnu"
TRAIN_BASES="C-base C-train-1 C-train-2 C-train-3 C-train-4 C-train-5"
VAL_BASE=C-val
LEVELS="10 50"
QUOTAS="1pct 5pct 15pct"
SEEDS="42 43 44"
VOBS=obs_sparse_5pct
# 与 Stage 4 同一份 strict-sparse 档位（照抄 sweep_lib.sh:316 的权重表），
# 但四个 --*-source 旗子必须显式带上：default 是 dense（..._strict_sparse.py:439-442）。
SP_ARGS="--feature-mode geometry --drop-features= --velocity-hidden-layers 128,128,128 --pressure-hidden-layers 128,128,128 --activation silu --velocity-epochs 200 --pressure-epochs 200 --coupling-epochs 80 --velocity-lr 6e-4 --pressure-lr 6e-4 --coupling-velocity-lr 1e-4 --coupling-pressure-lr 1e-4 --wall-weight 0.0 --inlet-flux-weight 0.0 --continuity-weight 0.1 --velocity-stage-continuity-weight 0.3 --velocity-stage-momentum-weight 0.0 --outlet-pressure-weight 0.0 --pressure-drop-weight 0.0 --pressure-stage-momentum-weight 0.5 --velocity-wall-mode hard --hard-wall-sharpness 12 --coupling-momentum-weight 1.0 --coupling-continuity-weight 0.1 --coupling-velocity-supervision-weight 1.0 --coupling-pressure-supervision-weight 1.0 --max-physics-points 512 --print-every 40 --strict-sparse-scalers --max-retries 1"
mkdir -p "$LOGD"
export PYTHONPATH="$WS/pylibs:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$FFHOME:${LD_LIBRARY_PATH:-}"

log() { printf '%s | %s\n' "$(date '+%H:%M:%S')" "$*"; }

step() {
  local name="$1" fatal="$2"; shift 2
  local t0 t1 rc
  t0=$(date +%s%N)
  "$@" >"$LOGD/$name.log" 2>&1; rc=$?
  t1=$(date +%s%N)
  log "STEP $name rc=$rc wall=$(( (t1 - t0) / 1000000000 ))s fatal=$fatal"
  if [ "$rc" -ne 0 ] && [ "$fatal" = fatal ]; then
    tail -15 "$LOGD/$name.log"
    log "FATAL at $name —— 中止（控制没过就不出结论）"; exit 1
  fi
  return 0
}

join_cases() { local lvl="$1"; shift; local o=""; local c; for c in "$@"; do o="$o${o:+,}${c}_ns_re${lvl}"; done; printf '%s' "$o"; }

src_flags() { printf -- '--train-velocity-source obs_sparse_%s_velocity_only --train-pressure-source obs_sparse_%s_velocity_only --val-velocity-source dense --val-pressure-source dense' "$1" "$1"; }

phase_preflight() {
  log "PHASE=preflight"
  command -v FreeFem++ >/dev/null 2>&1 || step ff_restore nonfatal tar xzf "$WS/ffroot.tgz" -C /
  local miss; miss=$(ldd "$(command -v FreeFem++)" 2>/dev/null | awk '/not found/{c++} END{print c+0}')
  log "FreeFem=$(command -v FreeFem++) 未解析库=$miss LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  [ "$miss" = 0 ] || { log "FATAL：FreeFEM 运行库没恢复齐（见 go8.sh 那条路子）"; exit 1; }
  sha256sum "$S/train_velocity_pressure_independent_ns_vonly.py" "$S/assim_pressure_baseline.py" \
            "$S/ns5_verdict.py" "$S/train_velocity_pressure_independent_ns.py" | tee "$LOGD/device.sha256"
}

phase_obs() {
  log "PHASE=obs"
  local L b stem
  for L in $LEVELS; do
    for b in $TRAIN_BASES $VAL_BASE; do
      # 5pct 那批 Stage 2 就造过（同点位、NS 取值），存在就跳过；make_ns_obs.py 对已存在的目标件是
      # **拒绝覆盖**的，不跳过就会把这一步变成假红。
      if [ -f "$DATA/${b}_ns_re${L}/obs_sparse_5pct.csv" ]; then
        log "obs ${b}_ns_re${L}/5pct [skip]"
      else
        step "obs_${b}_${L}_5pct" fatal python3 "$S/make_ns_obs.py" --root "$WS" --base "$b" --level "$L" --obs-stem "$VOBS"
      fi
      for stem in obs_sparse_1pct obs_sparse_15pct; do
        if [ -f "$DATA/${b}_ns_re${L}/${stem}.csv" ]; then log "obs ${b}_ns_re${L}/${stem} [skip]"; continue; fi
        step "obs_${b}_${L}_${stem}" fatal python3 "$S/make_ns_obs.py" --root "$WS" --base "$b" --level "$L" --obs-stem "$stem"
      done
    done
  done
  # 速度-only 派生：用现成工具（它按 family 铺全部工况目录），跑完把非 NS 目录里新增的那批删掉——
  # 别的线不读这个名字，留着只是让我的改动污染别人的工况目录。
  step derive_vonly fatal bash -c "cd '$WS' && python3 model/scripts/generate_partial_observations.py --family contraction_2d --sources obs_sparse_1pct,obs_sparse_5pct,obs_sparse_15pct --observed-components u,v"
  local killed=0 f
  for f in "$DATA"/*/obs_sparse_*_velocity_only.csv; do
    case "$(basename "$(dirname "$f")")" in *_ns_re*) continue ;; esac
    rm -f "$f"; killed=$((killed + 1))
  done
  log "非 NS 目录里的 velocity_only 件已清：$killed 个（那些不是本件的输入）"
}

phase_selftests() {
  log "PHASE=selftests"
  step v_verdict fatal python3 "$S/ns5_verdict.py" --selftest
  step v_rbf fatal python3 "$S/assim_pressure_baseline.py" --rbf-check --outdir "$LOGD"
  # 一阶导的已知答案检查：这是**诊断**，不是登记闸门（63 点上导数不可靠本身就是关于这一臂的信息，
  # 由 C9（稠密点）来判定装置在原理上对不对），所以 nonfatal。
  step v_deriv nonfatal python3 "$S/assim_pressure_baseline.py" --deriv-check --reynolds 10
  step v_c7 fatal python3 "$S/train_velocity_pressure_independent_ns_vonly.py" --vonly-selftest
  step v_c10 fatal python3 "$S/assim_pressure_baseline.py" --selftest --case C-base_ns_re10 --reynolds 10 --outdir "$LOGD"
}

phase_c6() {
  log "PHASE=c6"
  step c6 fatal python3 "$S/ns5_verdict.py" --c6-check --root "$WS" --out "$LOGD" \
    --cases "$(join_cases 10 $TRAIN_BASES $VAL_BASE),$(join_cases 50 $TRAIN_BASES $VAL_BASE)"
}

check_sources() {   # 一格跑完后，config.json 里的四个 source 必须与我声明的一致
  local name="$1" quota="$2"
  python3 - "$RES/$name/config.json" "$quota" "$name" <<'PY'
import json, sys
cfg, quota, name = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(cfg, encoding="utf-8"))
a = d.get("args", d)
want = "obs_sparse_%s_velocity_only" % quota
bad = []
for k in ("train_velocity_source", "train_pressure_source"):
    if a.get(k) != want:
        bad.append(f"{k}={a.get(k)!r}")
for k in ("val_velocity_source", "val_pressure_source"):
    if a.get(k) != "dense":
        bad.append(f"{k}={a.get(k)!r}")
if bad:
    print(f"[FAIL] {name} 的观测源不是本件声明的那张表：" + "; ".join(bad))
    sys.exit(1)
print(f"[SRC-OK] {name} 训练源={want} 评价源=dense")
PY
}

run_cell() {  # run_cell <lvl> <quota> <arm> <seed> <block> <train_cases>
  local L="$1" Q="$2" A="$3" SEED="$4" BLK="$5" TC="$6"
  local re name extra=""
  case "$A" in
    ns) re="$L" ;;
    stokes) re=0 ;;
    *) log "未知臂 $A"; return 1 ;;
  esac
  # 动量权重取 Stage 4 标定后的值（NS 两档 1.0；Stokes 两档 10.0）——同一批"强度对齐"的约定
  [ "$A" = stokes ] && extra="--coupling-momentum-weight 10.0"
  if [ "$BLK" = solo ]; then name="ns5solo_${L}_${Q}_${A}_s${SEED}"; else name="ns5_${L}_${Q}_${A}_s${SEED}"; fi
  if [ -f "$RES/$name/metrics.json" ]; then log "CELL $name [skip]"; return 0; fi
  local t0 t1
  t0=$(date +%s%N)
  step "cell_${name}" nonfatal python3 "$S/train_velocity_pressure_independent_ns_vonly.py" \
    --base-script strict-sparse --reynolds "$re" --family contraction_2d \
    --train-cases "$TC" --val-cases "$(join_cases "$L" $VAL_BASE)" \
    --run-name "$name" --seed "$SEED" $SP_ARGS $(src_flags "$Q") $extra
  t1=$(date +%s%N)
  log "CELL $name wall_ms=$(( (t1 - t0) / 1000000 ))"
  if ! check_sources "$name" "$Q" >>"$LOGD/cell_${name}.log" 2>&1; then
    log "FATAL SOURCE-MISMATCH $name —— 观测源没挂上，这一格不进判决，整批停"
    exit 1
  fi
}

phase_smoke() {
  log "PHASE=smoke"
  local TC VC
  TC=$(join_cases 10 $TRAIN_BASES); VC=$(join_cases 10 $VAL_BASE)
  step smoke fatal python3 "$S/train_velocity_pressure_independent_ns_vonly.py" \
    --base-script strict-sparse --reynolds 10 --family contraction_2d \
    --train-cases "$TC" --val-cases "$VC" --run-name ns5smoke_10_5pct_ns_s42 --seed 42 \
    --velocity-epochs 12 --pressure-epochs 12 --coupling-epochs 6 $SP_ARGS $(src_flags 5pct) --print-every 5
  check_sources ns5smoke_10_5pct_ns_s42 5pct || exit 1
  python3 - "$RES/ns5smoke_10_5pct_ns_s42" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
m = json.load(open(d / "metrics.json", encoding="utf-8"))["最终验证指标"]
keys = ("rel_l2_u", "rel_l2_v", "rel_l2_speed", "rel_l2_p")
missing = [k for k in keys if k not in m]
pred = list((d / "predictions").glob("*_predictions.csv")) if (d / "predictions").is_dir() else []
print("SMOKE keys_missing=%s pred_files=%d 值=%s" % (missing or "无", len(pred),
      {k: m.get(k) for k in keys}))
if missing or not pred:
    sys.exit(1)
if not any(p.name.startswith("C-val_ns_re10") for p in pred):
    print("[FAIL] 预测件里没有 C-val_ns_re10 那一份"); sys.exit(1)
print("SMOKE PASS")
PY
}

phase_matrix() {
  log "PHASE=matrix"
  local TSV="$LOGD/matrix_runs.tsv" L Q A sd BLK TC
  [ -f "$TSV" ] || printf 'wall_line\n' > "$TSV"
  for L in $LEVELS; do
    TC=$(join_cases "$L" $TRAIN_BASES)
    for Q in $QUOTAS; do
      for A in ns stokes; do
        for sd in $SEEDS; do run_cell "$L" "$Q" "$A" "$sd" main "$TC"; done
      done
    done
    local SOLO; SOLO=$(join_cases "$L" $VAL_BASE)
    for A in ns stokes; do
      for sd in $SEEDS; do run_cell "$L" 5pct "$A" "$sd" solo "$SOLO"; done
    done
  done
}

phase_baseline() {
  log "PHASE=baseline"
  local TSV="$LOGD/matrix_baseline.tsv" L Q E CASE
  : > "$TSV.tmp"
  for L in $LEVELS; do
    CASE="${VAL_BASE}_ns_re${L}"
    for Q in $QUOTAS; do
      for E in ns stokes; do
        step "base_${L}_${Q}_${E}" nonfatal python3 "$S/assim_pressure_baseline.py" \
          --root "$WS" --case "$CASE" --quota "$Q" --equation "$E" --reynolds "$L" --outdir "$LOGD"
        grep -h '^BASELINE ' "$LOGD/base_${L}_${Q}_${E}.log" >> "$TSV.tmp" || log "缺 BASELINE 行：$L/$Q/$E"
      done
    done
    # C9 上限对照：喂完整稠密真值速度（等价配额→100%），装置若连这都做不到就是有 bug
    for E in ns stokes; do
      step "c9_${L}_${E}" nonfatal python3 "$S/assim_pressure_baseline.py" \
        --root "$WS" --case "$CASE" --quota full --equation "$E" --reynolds "$L" --outdir "$LOGD"
      grep -h '^BASELINE ' "$LOGD/c9_${L}_${E}.log" >> "$TSV.tmp" || log "缺 C9 行：$L/$E"
    done
  done
  mv -f "$TSV.tmp" "$TSV"
  wc -l "$TSV"
  python3 - "$TSV" <<'PY'
import sys
rows = [dict(t.split("=", 1) for t in line.split()[1:] if "=" in t) for line in open(sys.argv[1], encoding="utf-8") if line.startswith("BASELINE")]
c9 = [r for r in rows if r.get("quota") == "full"]
bad = [r for r in c9 if float(r["rel_l2_p_meanfree"]) > 0.05 or float(r["pressure_drop_rel_error"]) > 0.05]
print("C9 上限对照 n=%d；不达标 %d" % (len(c9), len(bad)))
for r in bad:
    print("  C9-FAIL case=%s re=%s eq=%s p=%s dp=%s" % (r["case"], r["reynolds"], r["equation"],
          r["rel_l2_p_meanfree"], r["pressure_drop_rel_error"]))
if len(c9) < 4 or bad:
    sys.exit(1)
print("C9 PASS")
PY
}

phase_sens() {
  log "PHASE=sens（压力回退尺度=1.0 的敏感性对照；不进登记判决）"
  local L TC
  for L in $LEVELS; do
    TC=$(join_cases "$L" $TRAIN_BASES)
    step "cell_ns5sens_${L}_5pct_ns_s42" nonfatal python3 "$S/train_velocity_pressure_independent_ns_vonly.py" \
      --base-script strict-sparse --reynolds "$L" --family contraction_2d \
      --train-cases "$TC" --val-cases "$(join_cases "$L" $VAL_BASE)" \
      --run-name "ns5sens_${L}_5pct_ns_s42" --seed 42 --vonly-pressure-scale one \
      $SP_ARGS $(src_flags 5pct)
    check_sources "ns5sens_${L}_5pct_ns_s42" 5pct >>"$LOGD/cell_ns5sens_${L}_5pct_ns_s42.log" 2>&1 \
      || log "FATAL SOURCE-MISMATCH ns5sens_${L}_5pct_ns_s42"
  done
}

phase_matrixfix() {
  log "PHASE=matrixfix（对流项符号修正后重跑的 NS 臂：Re{10,50}×3 种子，5pct）"
  local L sd TC
  for L in $LEVELS; do
    TC=$(join_cases "$L" $TRAIN_BASES)
    for sd in $SEEDS; do
      local name="ns5fix_${L}_5pct_ns_s${sd}"
      [ -f "$RES/$name/metrics.json" ] && { log "CELL $name [skip]"; continue; }
      step "cell_${name}" nonfatal python3 "$S/train_velocity_pressure_independent_ns_vonly.py" \
        --base-script strict-sparse --reynolds "$L" --family contraction_2d \
        --train-cases "$TC" --val-cases "$(join_cases "$L" $VAL_BASE)" \
        --run-name "$name" --seed "$sd" $SP_ARGS $(src_flags 5pct)
      check_sources "$name" 5pct >>"$LOGD/cell_${name}.log" 2>&1 || log "FATAL SOURCE-MISMATCH $name"
    done
  done
}

phase_score() {
  log "PHASE=score"
  step score fatal python3 "$S/ns5_verdict.py" --pinn-score --root "$WS" --out "$LOGD"
}

phase_judge() {
  log "PHASE=judge"
  step judge fatal python3 "$S/ns5_verdict.py" --judge --root "$WS" --out "$LOGD"
}

case "${1:-}" in
  preflight) phase_preflight;;
  obs) phase_obs;;
  selftests) phase_selftests;;
  c6) phase_c6;;
  smoke) phase_smoke;;
  matrix) phase_matrix;;
  baseline) phase_baseline;;
  sens) phase_sens;;
  matrixfix) phase_matrixfix;;
  score) phase_score;;
  judge) phase_judge;;
  all) phase_preflight; phase_selftests; phase_obs; phase_c6; phase_smoke; phase_matrix;
       phase_baseline; phase_sens; phase_score; phase_judge;;
  *) echo "usage: $0 {preflight|obs|selftests|c6|smoke|matrix|baseline|sens|score|judge|all}"; exit 2;;
esac
RC=$?
log "SUMMARY phase=${1:-all} ok=$((1 - (RC > 0))) rc=$RC logs=$LOGD"
exit "$RC"
