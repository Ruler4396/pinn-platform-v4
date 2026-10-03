#!/usr/bin/env bash
# ops/run_ns4sparse.sh -- 稀疏观测 × 方程对错（有限 Re）。判据与控制写在
#   paper-route2/稀疏档方程对照-预注册设计-20261003.md，本件只执行、不解释。
#
#   obs      : 给 7 个工况 × 3 个 Re 档造"同点位、NS 取值"的观测表（make_ns_obs.py）
#   controls : C3 尺度口径两格（--reynolds 0 对 1e-9）+ 观测表核对
#   matrix   : 3 臂 × 2 档 × 3 种子 = 18 格，另加 Re=1 的 1 格 sanity（不进判据）
#   pair     : J1（NS vs Stokes 残差）与 J2（NS vs 无物理），三条条件全过才算 MET
#   parity   : 与 Stage 3 那枚驱动的 metrics_dump 逐字符对账（同一份 metrics.json）
set -uo pipefail

WS="${WS:-/mnt/workspace/pinn-repro-2026}"
OPS="${OPS:-/mnt/workspace/_ops/ns3p/20261003}"
S="$WS/model/scripts"
DATD="$WS/model/cases/contraction_2d/data"
RES="$WS/model/results/pinn"
LOGD="$OPS/out"
OBS=obs_sparse_5pct
TRAIN_BASES="C-base C-train-1 C-train-2 C-train-3 C-train-4 C-train-5"
LEVELS="10 50"
SEEDS="42 43 44"
# strict-sparse 档位照抄 sweep_lib.sh 的 train_dual()，只额外加 --base-script/--reynolds/--seed
SP_ARGS="--feature-mode geometry --drop-features '' --velocity-hidden-layers 128,128,128 --pressure-hidden-layers 128,128,128 --activation silu --velocity-epochs 200 --pressure-epochs 200 --coupling-epochs 80 --velocity-lr 6e-4 --pressure-lr 6e-4 --coupling-velocity-lr 1e-4 --coupling-pressure-lr 1e-4 --wall-weight 0.0 --inlet-flux-weight 0.0 --continuity-weight 0.1 --velocity-stage-continuity-weight 0.3 --velocity-stage-momentum-weight 0.0 --outlet-pressure-weight 0.0 --pressure-drop-weight 0.0 --pressure-stage-momentum-weight 0.5 --velocity-wall-mode hard --hard-wall-sharpness 12 --coupling-momentum-weight 10.0 --coupling-continuity-weight 0.1 --coupling-velocity-supervision-weight 1.0 --coupling-pressure-supervision-weight 1.0 --max-physics-points 512 --print-every 40 --strict-sparse-scalers --max-retries 1"
NOPHY="--coupling-momentum-weight 0.0 --coupling-continuity-weight 0.0 --velocity-stage-continuity-weight 0.0 --pressure-stage-momentum-weight 0.0"
# 机制对照用：strict-sparse 档位把入口流量/出口压力/压降三项设成 0（为了避免稠密真值泄漏进稀疏训练），
# 于是那一版里的 PDE 项是**没有边界条件配套**的。加回主线稠密档的取值才能分清"物理项没用"与
# "没有边界条件的物理项没用"。取值照抄 sweep_lib.sh 的 mainline-dense 档：0.5 / 1e-4 / 1.0。
BC_ARGS="--inlet-flux-weight 0.5 --outlet-pressure-weight 1e-4 --pressure-drop-weight 1.0"
mkdir -p "$LOGD"
export PYTHONPATH="$WS/pylibs:${PYTHONPATH:-}"

log() { printf '%s | %s\n' "$(date '+%H:%M:%S')" "$*"; }

step() {
  local name="$1" fatal="$2"; shift 2
  local t0 rc out
  t0=$(date +%s); out="$LOGD/step_${name}.txt"
  bash -c "cd '$WS' && $*" >"$out" 2>&1; rc=$?
  log "STEP $name rc=$rc wall=$(( $(date +%s) - t0 ))s fatal=$fatal"
  if [ "$rc" != 0 ]; then
    tail -20 "$out" | sed "s/^/    E| /"
    [ "$fatal" = fatal ] && { log "ABORT at $name (see $out)"; exit 1; }
  else
    tail -6 "$out" | sed "s/^/    > /"
  fi
  return 0
}

join_cases() { local lvl="$1"; shift; local o="" b; for b in "$@"; do [ -n "$o" ] && o="$o,"; o="$o${b}_ns_re${lvl}"; done; printf '%s' "$o"; }

metrics_dump() {   # 与 run_ns3p_stage23.sh 同一段：尺必须点名到段，压降取 验证工况指标[0]
  local f="$RES/$1/metrics.json"
  [ -f "$f" ] || { printf 'MISSING'; return; }
  python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
sec = d.get("最终验证指标")
if not isinstance(sec, dict):
    print("MISSING-SECTION"); sys.exit(0)
want = ["rel_l2_u", "rel_l2_v", "rel_l2_speed", "rel_l2_p"]
vals = {k: sec.get(k) for k in want}
rows = d.get("验证工况指标") or []
first = rows[0] if rows and isinstance(rows[0], dict) else {}
vals["pressure_drop_rel_error"] = first.get("pressure_drop_rel_error")
print("\t".join(f"{k}={vals[k]}" for k in want + ["pressure_drop_rel_error"])
      + "\tfour_keys=" + ("OK" if all(vals[k] is not None for k in want) else "INCOMPLETE")
      + "\tsplit=val\tval_case=" + str(first.get("case_id")))
PY
}

first_momentum() {  # 耦合阶段第一条动量损失（C3 用）；读不到就印 NA，不许默认通过
  python3 - "$RES/$1/history.csv" <<'PY'
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
except FileNotFoundError:
    print("NA(no history.csv)"); sys.exit(0)
if not rows:
    print("NA(no rows)"); sys.exit(0)
cols = [c for c in rows[0].keys() if "动量" in c]
prefer = [c for c in cols if c.startswith("耦合_")] or cols
if not prefer:
    print("NA(no 动量 column; have: " + ",".join(list(rows[0].keys())[:6]) + ")"); sys.exit(0)
for c in prefer:
    for r in rows:
        if r.get(c) not in (None, ""):
            print(f"{c}={r[c]}"); sys.exit(0)
print("NA(empty column)")
PY
}

phase_obs() {
  local b L
  for b in $TRAIN_BASES C-val; do
    for L in $LEVELS 1; do
      step "obs_${b}_${L}" fatal "python3 model/scripts/make_ns_obs.py --root '$WS' --base $b --level $L --obs-stem $OBS"
    done
  done
}

phase_controls() {
  local TC VC n0 n1 m0 m1
  TC=$(join_cases 10 $TRAIN_BASES); VC=$(join_cases 10 C-val)
  step c3_re0 fatal "python3 model/scripts/train_velocity_pressure_independent_ns.py --base-script strict-sparse --reynolds 0 --family contraction_2d --train-cases $TC --val-cases $VC --run-name ns4_c3_re0_s42 --seed 42 $SP_ARGS"
  step c3_re1e9 fatal "python3 model/scripts/train_velocity_pressure_independent_ns.py --base-script strict-sparse --reynolds 1e-9 --family contraction_2d --train-cases $TC --val-cases $VC --run-name ns4_c3_re1e9_s42 --seed 42 $SP_ARGS"
  n0=ns4_c3_re0_s42; n1=ns4_c3_re1e9_s42
  m0=$(first_momentum "$n0"); m1=$(first_momentum "$n1")
  log "C3 第一条耦合动量损失：re0[$m0] re1e-9[$m1]"
  python3 - "$m0" "$m1" <<'PY' | tee "$LOGD/c3_verdict.txt"
import sys
def val(s):
    return float(s.split("=", 1)[1]) if "=" in s and not s.startswith("NA") else None
a, b = val(sys.argv[1]), val(sys.argv[2])
if a is None or b is None or a == 0:
    print("C3 = INDETERMINATE（读不到耦合动量损失首条，或基准为 0 无法取相对差）"); sys.exit(3)
rel = abs(a - b) / abs(a)
print(f"C3 相对差 = {rel:.3e}（判据：≤1e-5）")
print("C3 = " + ("PASS" if rel <= 1e-5 else "FAIL -> 停下排查，本件不出结论"))
sys.exit(0 if rel <= 1e-5 else 1)
PY
  local d0 d1
  d0=$(metrics_dump "$n0"); d1=$(metrics_dump "$n1")
  log "C3 metrics re0   : $d0"
  log "C3 metrics re1e-9: $d1"
  python3 - "$d0" "$d1" <<'PY' | tee -a "$LOGD/c3_verdict.txt"
import sys
def parse(s):
    o = {}
    for t in s.split():
        if "=" in t:
            k, v = t.split("=", 1)
            try: o[k] = float(v)
            except ValueError: pass
    return o
a, b = parse(sys.argv[1]), parse(sys.argv[2])
keys = [k for k in ("rel_l2_u", "rel_l2_v", "rel_l2_speed", "rel_l2_p", "pressure_drop_rel_error")
        if k in a and k in b]
d = max(abs(a[k] - b[k]) for k in keys)
print(f"C3 四项+压降指标最大绝对差 = {d:.3e}（界 1e-3，实测值随读数一起落纸）")
PY
  grep -q "C3 = PASS" "$LOGD/c3_verdict.txt" || { log "中止：C3 尺度口径没过"; exit 1; }
}

phase_matrix() {
  local tsv="$LOGD/matrix_sparse.tsv" L A SEED re extra name TC VC t0 t1
  [ -f "$tsv" ] || printf 'level\tarm\tseed\twall_ms\trun\tmetrics\tpde_weights\n' > "$tsv"
  for L in $LEVELS; do
    for A in ns stokes nophy; do
      for SEED in $SEEDS; do
        run_cell "$L" "$A" "$SEED" "$tsv" || { log "matrix 中断于 level=$L arm=$A seed=$SEED"; exit 1; }
      done
    done
  done
  run_cell 1 ns 42 "$tsv"
  log "matrix_sparse rows: $(tail -n +2 "$tsv" | wc -l)"
}

run_cell() {
  local L="$1" A="$2" SEED="$3" tsv="$4" TAG="${5:-}" EXTRA="${6:-}"
  local re=10 extra="" name TC VC t0 t1 w
  case "$A" in
    ns) re="$L" ;;
    stokes) re=0 ;;
    nophy) re=0; extra="$NOPHY" ;;
    *) log "未知臂 $A"; return 1 ;;
  esac
  [ -n "$EXTRA" ] && extra="$extra $EXTRA"
  name="ns4${TAG}_${L}_${A}_s${SEED}"
  if [ -f "$RES/$name/metrics.json" ] && awk -F'\t' -v n="$name" '$5==n{f=1}END{exit !f}' "$tsv"; then
    log "CELL $name [skip]"; return 0
  fi
  TC=$(join_cases "$L" $TRAIN_BASES); VC=$(join_cases "$L" C-val)
  t0=$(date +%s%N)
  step "cell_${TAG}_${L}_${A}_${SEED}" nonfatal "python3 model/scripts/train_velocity_pressure_independent_ns.py --base-script strict-sparse --reynolds $re --family contraction_2d --train-cases $TC --val-cases $VC --run-name $name --seed $SEED $SP_ARGS $extra"
  t1=$(date +%s%N)
  w=$(python3 - "$RES/$name/config.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception as exc:
    print("NA(config unreadable: " + type(exc).__name__ + ")"); sys.exit(0)
# config.json 把权重放在中文键的 "权重" 段里（train_..._strict_sparse.py:839-849）——
# 先按那四个名字取；取不到就退回扫 args 段，扫不到印 NA，绝不默认"权重为 0"。
sec = d.get("权重") if isinstance(d.get("权重"), dict) else {}
names = ("耦合动量", "耦合连续性", "速度阶段连续性", "压力阶段动量")
got = {k: sec[k] for k in names if k in sec}
if not got:
    flat = d.get("args", d)
    alt = ("coupling_momentum_weight", "coupling_continuity_weight",
           "velocity_stage_continuity_weight", "pressure_stage_momentum_weight")
    got = {k: flat[k] for k in alt if k in flat}
print(",".join(f"{k}={got[k]}" for k in got) if got else "NA(no 权重 section)")
PY
)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$L" "$A" "$SEED" "$(( (t1 - t0) / 1000000 ))" \
    "$name" "$(metrics_dump "$name" | tr '\t\n' '  ')" "$w" >> "$tsv"
  log "CELL $name wall_ms=$(( (t1 - t0) / 1000000 )) pde[$w]"
}

phase_bc() {
  # 机制对照（不是新的判决）：strict-sparse 档位把 入口流量/出口压力/压降 三项权重全设成 0，
  # 于是 PDE 项是**没有边界条件配套**的欠约束项。matrix_sparse 里"带物理反而更差"最可能来自这里。
  # 这一批把那三项按主线稠密档的取值加回来，三臂同口径重跑 Re=10 三种子——比较的仍是同一条管线内的臂。
  local tsv="$LOGD/matrix_bc.tsv" A SEED
  [ -f "$tsv" ] || printf 'level\tarm\tseed\twall_ms\trun\tmetrics\tpde_weights\n' > "$tsv"
  for A in ns stokes nophy; do
    for SEED in $SEEDS; do
      run_cell 10 "$A" "$SEED" "$tsv" _bc "$BC_ARGS" || { log "bc 中断于 arm=$A seed=$SEED"; exit 1; }
    done
  done
  log "bc rows: $(tail -n +2 "$tsv" | wc -l)"
}

phase_pilot() {   # §七 A1：两臂 × 两档，w=1.0，seed 42
  local tsv="$LOGD/matrix_pilot.tsv" L A
  [ -f "$tsv" ] || printf 'level\tarm\tseed\twall_ms\trun\tmetrics\tpde_weights\n' > "$tsv"
  for L in $LEVELS; do
    for A in ns stokes; do
      run_cell "$L" "$A" 42 "$tsv" _p "--coupling-momentum-weight 1.0" || exit 1
    done
  done
}

phase_calib() {   # §七 选权规则执行 + A3 的 12 格
  local tsv="$LOGD/matrix_calib.tsv" arm level w r st SEED
  step calib_choose fatal "python3 model/scripts/ns4_calib.py --run-root '$RES' --prefix ns4 --out '$LOGD/weights.tsv'"
  [ -f "$tsv" ] || printf 'level\tarm\tseed\twall_ms\trun\tmetrics\tpde_weights\n' > "$tsv"
  while IFS=$'\t' read -r arm level w r st; do
    case "$st" in aligned) ;; *) log "SKIP 未对齐 $arm Re=$level ($st) —— 按 §七 不进 J3"; continue;; esac
    for SEED in $SEEDS; do
      run_cell "$level" "$arm" "$SEED" "$tsv" _c "--coupling-momentum-weight $w" || exit 1
    done
  done < <(tail -n +2 "$LOGD/weights.tsv")
  log "calib rows: $(tail -n +2 "$tsv" | wc -l)"
}

phase_eval() {   # §八 B1（同档 test 口径）与 B2（跨档外推），全部走评估器、复用已训好的 run
  local tsv="$LOGD/eval.tsv" L A S out
  [ -f "$tsv" ] || printf 'run\ttrain_level\tsplit\tcases\tspeed\tp\tdrop\teval_file\n' > "$tsv"
  for L in 1 10 50; do
    for A in ns stokes; do
      for S in $SEEDS; do
        [ "$L" = "1" ] && [ "$A" = "stokes" ] && continue
        out="ns3p_${L}_${A}_s${S}"
        [ -f "$RES/$out/metrics.json" ] || { log "缺 run $out，跳过"; continue; }
        run_eval "$out" "$L" test "C-test-1_ns_re${L},C-test-2_ns_re${L}" || exit 1
      done
    done
  done
  for A in ns stokes; do
    for S in $SEEDS; do
      run_eval "ns3p_10_${A}_s${S}" 10 ext50 "C-val_ns_re50,C-test-1_ns_re50,C-test-2_ns_re50" || exit 1
      run_eval "ns3p_50_${A}_s${S}" 50 ext10 "C-val_ns_re10,C-test-1_ns_re10,C-test-2_ns_re10" || exit 1
    done
  done
}

run_eval() {   # run  train_level  split  cases
  local name="$1" tl="$2" split="$3" cases="$4" f
  f="$RES/$name/evaluations/metrics_${split}.json"
  if [ -f "$f" ]; then log "EVAL $name/$split [skip]"; return 0; fi
  step "eval_${name}_${split}" fatal "python3 model/scripts/eval_ns_case.py --family contraction_2d --run-name $name --eval-cases $cases --split-name $split"
  python3 - "$f" "$name" "$tl" "$split" "$cases" "$LOGD/eval.tsv" <<'PY'
import json, sys
path, name, tl, split, cases, tsv = sys.argv[1:7]
d = json.load(open(path, encoding="utf-8"))
s = d.get("summary", {})
row = "\t".join([name, tl, split, cases,
                 str(s.get("mean_rel_l2_speed")), str(s.get("mean_rel_l2_p")),
                 str(s.get("mean_pressure_drop_rel_error")), path]) + "\n"
with open(tsv, "a", encoding="utf-8", newline="\n") as fh:
    fh.write(row)
print("EVAL-ROW " + row.strip())
PY
}

phase_pair() {
  local tsv="${1:-$LOGD/matrix_sparse.tsv}" out="${2:-$LOGD/pair_sparse.tsv}"
  python3 - "$tsv" "$out" <<'PY'
import csv, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
def num(s):
    o = {}
    for tok in (s or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try: o[k] = float(v)
            except ValueError: pass
    return o
cell, wall, wts = {}, {}, {}
for r in rows:
    if r["arm"] not in ("ns", "stokes", "nophy"):
        continue
    cell[(r["level"], r["arm"], r["seed"])] = num(r["metrics"])
    try:
        wall.setdefault((r["level"], r["arm"]), []).append(float(r["wall_ms"]))
    except (TypeError, ValueError):
        pass
    wts[(r["level"], r["arm"], r["seed"])] = r.get("pde_weights", "")
cols = ["rel_l2_speed", "rel_l2_p", "pressure_drop_rel_error", "rel_l2_u", "rel_l2_v"]
JUD = ("rel_l2_speed", "rel_l2_p")
out = []
# C4：无物理臂的四个 PDE 权重必须确为 0；两物理臂必须是"其余三项按档位、动量项 >0"。
# **动量权重是对齐批的被试量，不能再钉死成 10.0**（第一版把它写死，于是对齐批一跑就 C4=FAIL，
# 把 J2 整条判成"未判"——那是检查器的错，不是数据的错）。现在：钉死其余三项 + 动量项要求 >0，
# 动量项的具体值由 weights.tsv 印在读数旁边，读者能看见每臂实际用了多少。
want_ns = {"耦合连续性": 0.1, "速度阶段连续性": 0.3, "压力阶段动量": 0.5}
c4_bad, c4_na = [], []
for key, w in wts.items():
    arm = key[1]
    if w.startswith("NA"):
        c4_na.append("|".join(key)); continue
    got = {}
    for tok in w.split(","):
        if "=" in tok:
            k, _, v = tok.partition("=")
            try: got[k] = float(v)
            except ValueError: pass
    if arm == "nophy":
        if any(abs(v) > 0 for v in got.values()): c4_bad.append(("nophy 权重非 0", key, got))
    else:
        if any(abs(got.get(k, -1) - v) > 1e-9 for k, v in want_ns.items()):
            c4_bad.append(("三项权重与档位不符", key, got))
        if not got.get("耦合动量", 0) > 0:
            c4_bad.append(("物理臂的动量权重不是正数", key, got))
out.append("== C4 权重核对（读 config.json 的\"权重\"段；无物理臂四项皆 0，两物理臂三项按档位 + 动量项 >0）")
if c4_bad:
    for b in c4_bad: out.append("C4 = FAIL " + repr(b))
    out.append("=> 装置污染：J2 不进判决，先排查")
elif c4_na:
    out.append("C4 = INDETERMINATE（这些格没读到权重：" + ";".join(c4_na[:6]) + "）=> J2 不进判决")
else:
    out.append("C4 = PASS（无物理臂四项皆 0；NS/Stokes 两臂 10.0/0.1/0.3/0.5）")
c4_ok = not c4_bad and not c4_na
out.append("")
out.append("== per-arm (mean+-sd over seeds, ddof=1; wall = 训练墙钟 ms)")
hdr = ["level", "arm", "n"] + cols + ["wall_ms_mean"]
out.append("\t".join(hdr))
def ms(v):
    if not v: return "NA"
    s = ("+-" + f"{st.stdev(v):.4g}") if len(v) > 1 else "+-NA"
    return f"{st.mean(v):.5g}{s}"
for lvl in ("10", "50", "1"):
    for arm in ("ns", "stokes", "nophy"):
        vs = {c: [] for c in cols}
        n = 0
        for (l, a, s), d in cell.items():
            if l == lvl and a == arm:
                n += 1
                for c in cols:
                    if d.get(c) is not None: vs[c].append(d[c])
        if n == 0: continue
        w = wall.get((lvl, arm), [])
        row = [lvl, arm, str(n)] + [ms(vs[c]) for c in cols] + [f"{st.mean(w):.0f}" if w else "NA"]
        assert len(row) == len(hdr), "per-arm 列数漂移"
        out.append("\t".join(row))
def pair(lvl, left, right, metric):
    """Δ = err(right 的参照臂) - err(right)；这里返回 left-right 的配对差列表。"""
    ds, rr = [], []
    for s in sorted({k[2] for k in cell}):
        a = cell.get((lvl, left, s), {}).get(metric)
        b = cell.get((lvl, right, s), {}).get(metric)
        if a is None or b is None: continue
        ds.append(a - b); rr.append(b)
    return ds, rr
out.append("")
out.append("== 判据：Δ = 误差(参照臂) - 误差(NS 臂)，正 = NS 臂更好；三条全过才 MET")
out.append("rule\tpairs\tlevel\tmetric\tmed_d\tmean_d\tns_sd\td_sd\tmed_d/d_sd\tsame_dir\tverdict")
verdict = {}
for name, left in (("J1 NS对Stokes残差", "stokes"), ("J2 NS对无物理", "nophy")):
    for lvl in ("10", "50"):
        for m in JUD:
            ds, ns_vals = pair(lvl, left, "ns", m)
            if not ds:
                out.append(f"{name}\t0\t{lvl}\t{m}\tNO-PAIRS\t\t\t\t\t\t不进判决")
                verdict[(name, lvl, m)] = False
                continue
            med = st.median(ds)
            try: sd = st.stdev(ns_vals)
            except st.StatisticsError: sd = float("nan")
            try: dsd = st.stdev(ds)
            except st.StatisticsError: dsd = float("nan")
            pos = sum(1 for d in ds if d > 0); n = len(ds)
            ok = med > 0 and pos >= (2 * n + 2) // 3 and (sd == sd and med >= sd / 3.0)
            verdict[(name, lvl, m)] = ok
            out.append(f"{name}\t{n}\t{lvl}\t{m}\t{med:+.6g}\t{st.mean(ds):+.6g}\t{sd:.4g}\t"
                       f"{dsd:.4g}\t" + (f"{med/dsd:+.2f}" if dsd == dsd and dsd > 0 else "NA")
                       + f"\t{pos}/{n}\t" + ("MET" if ok else "not met")
                       + "\t[" + ",".join(f"{d:+.5g}" for d in ds) + "]")
for m in cols:
    if m in JUD: continue
    for lvl in ("10", "50"):
        for name, left in (("登记外 J1", "stokes"), ("登记外 J2", "nophy")):
            ds, ns_vals = pair(lvl, left, "ns", m)
            if not ds: continue
            med = st.median(ds); pos = sum(1 for d in ds if d > 0)
            out.append(f"{name}\t{len(ds)}\t{lvl}\t{m}\t{med:+.6g}\t{st.mean(ds):+.6g}\t\t\t\t"
                       f"{pos}/{len(ds)}\t不进判决")
def rule(name):
    ks = [k for k in verdict if k[0] == name]
    return all(verdict[k] for k in ks) if ks else False
r1, r2 = rule("J1 NS对Stokes残差"), rule("J2 NS对无物理")
out.append("")
if not c4_ok:
    # C4 没过只废 J2（J1 不依赖无物理臂）。这里必须**分支互斥**：
    # 上一版两条 QUADRANT 都往外印，同一份产物里出现两个判决。
    out.append("QUADRANT  J1 -> " + ("MET" if r1 else "not met")
               + "；J2 未判（C4 没过：无物理臂的权重状态没核下来，"
                 "\"有没有物理\"这一对比不成立。先把 C4 修绿再谈 J2）")
else:
    out.append("QUADRANT  " + {(True, True): "J1达&J2达 -> 信息不足时对流项开始挣到位置；给表5-8 的负结果加限定（稠密/低Re 无收益），不外推稠密档",
                               (False, True): "J1不达&J2达 -> 物理项有位置，但方程写对写错不重要",
                               (False, False): "J1不达&J2不达 -> 稀疏档同样无可测贡献；'信息不足时物理项才挣到位置'在有限 Re 上不支持",
                               (True, False): "J1达&J2不达 -> 自相矛盾：先按装置污染排查，不产正文句子"}[(r1, r2)])
out.append("NOTE n<=3 时配对 Wilcoxon 最小可达双侧 p=0.0625 -> 任何一行都不许写'显著'；"
           "真值是 NS 场，与表5-5 不可并列；评分口径=val（最终验证指标）。")
print("\n".join(out))
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(out) + "\n")
PY
}

phase_parity() {
  local run="${1:-ns3p_10_ns_s42}" a b
  a=$(bash "$S/ops/run_ns3p_stage23.sh" dump "$run" 2>/dev/null || true)
  b=$(metrics_dump "$run" | tr '\t\n' '  ')
  log "parity run=$run"
  log "  stage23: $a"
  log "  stage4 : $b"
}

PH="${1:-obs}"
log "PHASE=$PH WS=$WS"
case "$PH" in
  obs) phase_obs;;
  controls) phase_controls;;
  matrix) phase_matrix;;
  pilot) phase_pilot;;
  calib) phase_calib;;
  eval) phase_eval;;
  j4tsv)   # J4 要的是"对齐强度的物理臂 vs 无物理"：无物理臂与权重无关，故直接取主批那 6 格拼进来
    { head -n 1 "$LOGD/matrix_calib.tsv"
      tail -n +2 "$LOGD/matrix_calib.tsv"
      tail -n +2 "$LOGD/matrix_sparse.tsv" | awk -F'\t' '$2=="nophy"'; } > "$LOGD/matrix_j4.tsv"
    wc -l "$LOGD/matrix_j4.tsv";;
  bc) phase_bc;;
  pair) phase_pair "${2:-}" "${3:-}";;
  dump) metrics_dump "${2:-}";;
  parity) phase_parity "${2:-}";;
  *) log "usage: $0 obs|controls|matrix|pilot|calib|bc|eval|j4tsv|pair [tsv out]|parity|dump"; exit 2;;
esac
log "SUMMARY phase=$PH ok=1 logs=$LOGD"
