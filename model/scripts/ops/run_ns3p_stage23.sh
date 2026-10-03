#!/usr/bin/env bash
# ops/run_ns3p_stage23.sh -- driver for Stage 2 and Stage 3 of the finite-Reynolds line.
#
#   derive   : prove+emit the NS .edp for the eight non-C-base Stokes cases, reusing
#              gen_ns_re_edp's own transform/invert/lint (control first: C-base must
#              reproduce the shipped file byte-for-byte)
#   solveall : solve those eight cases at Re = 1e-3 / 1 / 10 / 50, timed per process
#   cases    : build data/<case>_ns_re<level>/field_dense.csv from each level's raw truth
#   smoke    : the Stage 2 gate -- 8 epochs, Re=10 vs Re=0 on identical data and seed
#   matrix   : the Stage 3 cells (refuses to run unless smoke.tsv says PASS)
#   pair     : paired read-outs + the pre-registered decision rule, computed here not by hand
#
# Nothing is deleted; existing case dirs are refused by make_ns_case.py unless --allow-existing.
set -uo pipefail

WS="${WS:-/mnt/workspace/pinn-repro-2026}"
OPS="${OPS:-/mnt/workspace/_ops/ns3p/20261003}"
CFD="$WS/model/cases/contraction_2d/cfd"
DATD="$WS/model/cases/contraction_2d/data"
LOGD="$OPS/out"
RES="$WS/model/results/pinn"
LEVELS="1e-3 1 10 50"
TRAIN_BASES="C-base C-train-1 C-train-2 C-train-3 C-train-4 C-train-5"
OTHER_BASES="C-train-1 C-train-2 C-train-3 C-train-4 C-train-5 C-val C-test-1 C-test-2"
ALL_BASES="$TRAIN_BASES C-val C-test-1 C-test-2"
SEEDS="42 43 44"
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
    tail -25 "$out" | sed "s/^/    E| /"
    [ "$fatal" = fatal ] && { log "ABORT at $name (see $out)"; exit 1; }
  else
    tail -10 "$out" | sed "s/^/    > /"
  fi
  return 0
}

join_cases() {  # join_cases <level> <bases...>  ->  "C-base_ns_re10,C-train-1_ns_re10,..."
  local lvl="$1"; shift
  local out="" b
  for b in "$@"; do
    [ -n "$out" ] && out="$out,"
    out="$out${b}_ns_re${lvl}"
  done
  printf '%s' "$out"
}

phase_derive() {
  step derive_control fatal "python3 model/scripts/derive_ns_case_edp.py"
  step derive_write fatal "python3 model/scripts/derive_ns_case_edp.py --write --cases $OTHER_BASES"
  local n; n=$(find "$CFD" -maxdepth 1 -type d -name '*_ns_re*' | wc -l)
  log "ns case dirs under cfd = $n (expect 9 bases x 4 levels = 36)"
}

phase_solveall() {
  command -v FreeFem++ >/dev/null 2>&1 || { log "INDETERMINATE: no FreeFem++"; exit 90; }
  local tsv="$LOGD/solve_all.tsv"
  printf 'case\tlevel\trc\twall_ms\tnot_converged\tpicard_iters\tlast_du2\tsha_raw+pair\trows_raw\n' > "$tsv"
  local b L d lg rc t0 t1 nc iters last rows
  for b in $OTHER_BASES; do
    for L in $LEVELS; do
      d="$CFD/${b}_ns_re$L"; lg="$LOGD/run_${b}_${L}.log"
      if [ ! -f "$d/${b}_ns_re$L.edp" ]; then log "SKIP $b Re=$L (no .edp)"; continue; fi
      t0=$(date +%s%N)
      ( cd "$d" && FreeFem++ -nw "${b}_ns_re$L.edp" ) >"$lg" 2>&1
      rc=$?
      t1=$(date +%s%N)
      nc=$(grep -c "^NS NOT CONVERGED" "$lg")
      iters=$(grep -c "^NS it=" "$lg")
      last=$(grep "^NS it=" "$lg" | tail -1 | sed 's/.*du2=\([^ ]*\).*/\1/')
      rows=$(( $(wc -l < "$d/${b}_ns_re${L}_raw.csv") - 1 ))
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$b" "$L" "$rc" \
        "$(( (t1 - t0) / 1000000 ))" "$nc" "$iters" "$last" \
        "$(sha256sum "$d/${b}_ns_re${L}_raw.csv" | cut -c1-16)$(sha256sum "$d/${b}_ns_re${L}_raw_pair.csv" | cut -c1-16)" \
        "$rows" >> "$tsv"
      log "SOLVE $b Re=$L rc=$rc wall_ms=$(( (t1 - t0) / 1000000 )) iters=$iters notconv=$nc rows=$rows"
    done
  done
  awk -F'\t' 'NR>1{n++;s+=$4;if($3!=0)bad++;if($5!=0)nc++}END{printf "SUMMARY solves=%d sum_wall_ms=%d mean_ms=%.0f rc!=0=%d not_converged_levels=%d\n",n,s,s/n,bad,nc+0}' "$tsv" > "$LOGD/solve_all.summary"
  sed "s/^/    A| /" "$LOGD/solve_all.summary"
}

phase_cases() {
  local b L raw
  for b in $ALL_BASES; do
    for L in $LEVELS; do
      raw="$CFD/${b}_ns_re$L/${b}_ns_re${L}_raw.csv"
      [ -f "$raw" ] || { log "SKIP case $b Re=$L (no raw truth)"; continue; }
      # make_ns_case.py refuses to overwrite on purpose; say SKIP instead of aborting the phase
      [ -f "$DATD/${b}_ns_re$L/field_dense.csv" ] && { log "SKIP case $b Re=$L (dense 件已在)"; continue; }
      step "case_${b}_${L}" fatal "python3 model/scripts/make_ns_case.py --base $b --level $L --raw '$raw' --root '$WS'"
    done
  done
}

metrics_dump() {  # metrics_dump <run-name> -> one line of key=value
  local f="$RES/$1/metrics.json"
  [ -f "$f" ] || { printf 'MISSING'; return; }
  python3 - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
want = ["rel_l2_u", "rel_l2_v", "rel_l2_speed", "rel_l2_p"]
def find(k):
    if k in d:
        return d[k]
    for a in d.values():
        if isinstance(a, dict) and k in a:
            return a[k]
    return None
print("\t".join(f"{k}={find(k)}" for k in want) + "\tfour_keys=" +
      ("OK" if all(find(k) is not None for k in want) else "INCOMPLETE"))
PY
}

phase_smoke() {
  local tsv="$LOGD/smoke.tsv" L=10 TC VC
  TC=$(join_cases "$L" $TRAIN_BASES); VC=$(join_cases "$L" C-val)
  printf 'run\twall_ms\tmetrics\n' > "$tsv"
  local P S0 S1
  for P in ns stokes; do
    local re=10 name="ns3p_smoke_re10_${P}_s42"
    [ "$P" = stokes ] && re=0
    local t0=$(date +%s%N)
    step "smoke_$P" fatal "python3 model/scripts/train_velocity_pressure_independent_ns.py --family contraction_2d --reynolds $re --train-cases $TC --val-cases $VC --seed 42 --run-name $name --velocity-epochs 8 --pressure-epochs 8 --coupling-epochs 8"
    local t1=$(date +%s%N)
    printf '%s\t%s\t%s\n' "$name" "$(( (t1 - t0) / 1000000 ))" "$(metrics_dump "$name" | tr '\n' ' ')" >> "$tsv"
  done
  cat "$tsv" | sed "s/^/    S| /"
  local A B
  A=$(metrics_dump "ns3p_smoke_re10_ns_s42"); B=$(metrics_dump "ns3p_smoke_re10_stokes_s42")
  if [ "$A" = "$B" ]; then
    log "SMOKE FAIL: Re=10 and Re=0 give bit-identical metrics -> the convective term never entered the loss"
    printf 'FAIL\tidentical-metrics\n' >> "$tsv"; exit 1
  fi
  case "$A$B" in
    *INCOMPLETE*|*MISSING*) log "SMOKE FAIL: metrics keys incomplete"; printf 'FAIL\tkeys\n' >> "$tsv"; exit 1;;
  esac
  log "SMOKE PASS: four keys present and the two arms differ (convection is in the path)"
  printf 'PASS\tdiffering-metrics\n' >> "$tsv"
}

phase_matrix() {
  grep -q "^PASS" "$LOGD/smoke.tsv" || { log "matrix refused: smoke.tsv has no PASS line"; exit 1; }
  local tsv="$LOGD/matrix.tsv"
  [ -f "$tsv" ] || printf 'level\tphys\tseed\twall_ms\trun\tmetrics\n' > "$tsv"
  local done_cells; done_cells=$(tail -n +2 "$tsv" | wc -l)
  log "matrix: $done_cells rows already in $tsv (a re-run skips cells already recorded)"
  local L P SEED re name t0 t1 seeds
  for L in 1 10 50 1e-3; do
    # 事前登记的矩阵：Re=1/10/50 各两臂三种子（18 格）+ Re=1e-3 两臂各一种子（2 格复算控制）= 20 格
    if [ "$L" = "1e-3" ]; then seeds="42"; else seeds="$SEEDS"; fi
    for P in ns stokes; do
      for SEED in $seeds; do
        re=$L; [ "$P" = stokes ] && re=0
        name="ns3p_${L}_${P}_s${SEED}"
        if [ -f "$RES/$name/metrics.json" ]; then
          if ! awk -F'\t' -v n="$name" '$5==n{f=1}END{exit !f}' "$tsv"; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$L" "$P" "$SEED" "already" "$name" "$(metrics_dump "$name" | tr '\n' ' ')" >> "$tsv"
            log "CELL $name [skip-train]"
          fi
          continue
        fi
        TC=$(join_cases "$L" $TRAIN_BASES); VC=$(join_cases "$L" C-val)
        t0=$(date +%s%N)
        step "cell_${L}_${P}_${SEED}" nonfatal "python3 model/scripts/train_velocity_pressure_independent_ns.py --family contraction_2d --reynolds $re --train-cases $TC --val-cases $VC --seed $SEED --run-name $name"
        t1=$(date +%s%N)
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$L" "$P" "$SEED" "$(( (t1 - t0) / 1000000 ))" "$name" "$(metrics_dump "$name" | tr '\n' ' ')" >> "$tsv"
        log "CELL $name wall_ms=$(( (t1 - t0) / 1000000 ))"
      done
    done
  done
  log "matrix rows: $(tail -n +2 "$tsv" | wc -l)"
}

phase_pair() {
  python3 - "$LOGD/matrix.tsv" "$LOGD/pair.tsv" <<'PY'
import csv, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
def num(s):
    out = {}
    for tok in (s or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out
cell = {}
for r in rows:
    if r["phys"] not in ("ns", "stokes") or r["seed"] in ("", None):
        continue
    cell[(r["level"], r["phys"], r["seed"])] = num(r["metrics"])
METRICS = ("rel_l2_speed", "rel_l2_p")
lines = []
verdicts = {}
for lvl in ("1", "10", "50", "1e-3"):
    for m in METRICS:
        ds, ns_vals, seeds = [], [], []
        for s in sorted({k[2] for k in cell}):
            a = cell.get((lvl, "stokes", s), {}).get(m)
            b = cell.get((lvl, "ns", s), {}).get(m)
            if a is None or b is None:
                continue
            ds.append(a - b)          # >0 means the NS-residual arm is better
            ns_vals.append(b)
            seeds.append(s)
        if not ds:
            lines.append(f"{lvl}\t{m}\tNO-PAIRS")
            continue
        med = st.median(ds)
        # sd 一把尺 = 样本标准差（ddof=1），与 9/27 定档一致；n=1 时无定义 -> nan，不补 0
        try:
            sd = st.stdev(ns_vals)
        except st.StatisticsError:
            sd = float("nan")
        pos = sum(1 for d in ds if d > 0)
        n = len(ds)
        ok = med > 0 and pos >= (2 * n + 2) // 3 and (sd == sd and med >= sd / 3.0)
        verdicts[(lvl, m)] = ok
        lines.append(f"{lvl}\t{n}\t{m}\tmed_d={med:+.6g}\tmean_d={st.mean(ds):+.6g}"
                     f"\tns_sd={sd:.4g}\tNSarm_better={pos}/{n}\trule={'MET' if ok else 'not met'}")
print("\n".join(lines))
gate = all(verdicts.get(k, False) for k in (("10", "rel_l2_speed"), ("10", "rel_l2_p"),
                                            ("50", "rel_l2_speed"), ("50", "rel_l2_p")))
print(f"PRE-REGISTERED RULE (Re=10 and Re=50, both metrics): "
      f"{'MET -> 可写对流项可吸收' if gate else 'NOT MET -> 按负结果写'}")
print("NOTE: this is the val-split metric on the 6-case NS train split; n<=3 -> Wilcoxon cannot "
      "reach p<0.05, so no significance word may be attached to any line above.")
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(lines) + "\n")
PY
}

PH="${1:-derive}"
log "PHASE=$PH WS=$WS"
case "$PH" in
  derive) phase_derive;;
  solveall) phase_solveall;;
  cases) phase_cases;;
  smoke) phase_smoke;;
  matrix) phase_matrix;;
  pair) phase_pair;;
  *) log "usage: $0 derive|solveall|cases|smoke|matrix|pair"; exit 2;;
esac
log "SUMMARY phase=$PH ok=1 logs=$LOGD"
