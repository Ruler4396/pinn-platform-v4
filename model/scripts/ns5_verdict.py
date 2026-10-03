"""Stage 5 的读数与判决（只看速度反推压力 × 经典基线同框）。判据事前写在
paper-route2/速度推压力-经典基线同框-预注册设计-20261003.md §四 与 §八，本件只执行、不解释。

模式：
  --selftest     三条已知答案控制：常数偏移⇒去均值误差恰好 0；等比 1.1⇒恰好 0.1；
                 压降公式与 train_supervised.compute_case_metrics:639-640 同形。
  --c6-check     速度-only 观测表：与源表同点数、坐标逐位相同、压力列有限值==0。
  --pinn-score   从各 run 的 predictions/C-val_ns_re<档>_predictions.csv 现算三个量，
                 逐行落 out/matrix_pinn.tsv（run 名约定 ns5_<档>_<配额>_<臂>_s<种子>）。
  --judge        读 matrix_pinn.tsv + matrix_baseline.tsv，出 K1 / K2 / K3 与四象限。

尺子只有这一枚：两臂都走本文件的 mean_free_rel / drop_err。
配对单位=种子（§八 登记的更正：逐点预测只在训练器自报的稠密 val 分路上落盘，
C-test-* 没有点级产物 ⇒ 不许为凑"逐工况配对"临时编一个读数来源）。经典臂是确定性求解，
没有种子散布 ⇒ K2 的分母只能取 PINN 跨种子 sd，这行说明随读数一起印出来。
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from pathlib import Path

JUD_P = 0.20          # K1：去均值压力场相对误差界
JUD_DP = 0.15         # K1：压降恢复相对误差界
LEVELS = ("10", "50")
QUOTAS = ("1pct", "5pct", "15pct")
REF_QUOTA = "15pct"
ARMS = ("ns", "stokes")
BLOCKS = ("main", "solo", "sens", "fix", "fixsolo", "lr")   # main＝6 工况联合训练；solo＝只用被评的那一个工况训练（数据池优势要分开）；
                                    # sens＝尺度敏感性（回退尺度 1.0）；fix/fixsolo＝对流项符号修正后重跑；
                                    # lr＝诊断档（压力侧 lr 抬到 1e-2），三档诊断都不进登记判决
METRICS = ("rel_l2_p_meanfree", "pressure_drop_rel_error", "rel_l2_speed")
JUDGE_METRICS = ("rel_l2_p_meanfree", "pressure_drop_rel_error")


def rows_of(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def tsv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def mean_free_rel(pred, truth) -> float:
    if len(pred) != len(truth) or not pred:
        raise SystemExit(f"[FAIL] mean_free_rel 长度不齐：{len(pred)} vs {len(truth)}")
    dp = [x - st.mean(pred) for x in pred]
    dt = [x - st.mean(truth) for x in truth]
    num = math.sqrt(sum((a - b) ** 2 for a, b in zip(dp, dt)))
    den = math.sqrt(sum(a * a for a in dt)) + 1.0e-12
    return num / den


def drop_err(pred, truth, bc):
    """bc：1=入口、2=出口。返回 (Δp_true, Δp_pred, 相对误差)，与主线同式。"""
    it = [p for p, b in zip(pred, bc) if b == 1]
    ot = [p for p, b in zip(pred, bc) if b == 2]
    tt = [t for t, b in zip(truth, bc) if b == 1]
    to = [t for t, b in zip(truth, bc) if b == 2]
    if not it or not ot or not tt or not to:
        raise SystemExit("[FAIL] 入口/出口点缺失，压降不可算")
    dp_t = st.mean(tt) - st.mean(to)
    dp_p = st.mean(it) - st.mean(ot)
    return dp_t, dp_p, abs(dp_p - dp_t) / (abs(dp_t) + 1.0e-12)


def selftest() -> int:
    fails = []
    a = [3.0, 5.0, 7.0]
    if abs(mean_free_rel([x + 100.0 for x in a], a)) > 1e-15:
        fails.append("常数偏移的去均值误差不是恰好 0")
    if abs(mean_free_rel([1.1 * x for x in a], a) - 0.1) > 1e-12:
        fails.append("等比 1.1 的去均值误差不是 0.1")
    t, p, e = drop_err([10.0, 4.0], [12.0, 5.0], [1, 2])
    if not (abs(t - 7.0) < 1e-12 and abs(p - 6.0) < 1e-12 and abs(e - 1.0 / 7.0) < 1e-9):
        fails.append(f"压降公式走形：dp_true={t} dp_pred={p} err={e}")
    print("VERDICT_SELFTEST " + ("ALL GREEN" if not fails else "FAILED | " + " | ".join(fails)))
    return 1 if fails else 0


def c6_check(data_root: Path, cases, quotas) -> int:
    bad = 0
    for cid in cases:
        for q in quotas:
            base = data_root / cid / f"obs_sparse_{q}.csv"
            vo = data_root / cid / f"obs_sparse_{q}_velocity_only.csv"
            if not base.exists() or not vo.exists():
                print(f"C6-FAIL {cid}/{q} MISSING base={base.exists()} vonly={vo.exists()}")
                bad += 1
                continue
            rb, rv = rows_of(base), rows_of(vo)
            same = [(r["x_star"], r["y_star"]) for r in rb] == [(r["x_star"], r["y_star"]) for r in rv]
            pcol = "p_obs" if "p_obs" in rv[0] else "p_star"
            finite_p = sum(1 for r in rv if (r.get(pcol) or "").strip() not in ("", "nan", "NaN"))
            ok = same and len(rb) == len(rv) and finite_p == 0
            print(f"C6-{'ok' if ok else 'FAIL'} {cid}/{q} n_base={len(rb)} n_vonly={len(rv)} "
                  f"坐标逐位相同={same} 压力有限值={finite_p}")
            bad += 0 if ok else 1
    print("C6 = " + ("PASS" if bad == 0 else f"FAIL（不合格 {bad} 格）"))
    return 1 if bad else 0


def pinn_score(res_root: Path, out_tsv: Path) -> int:
    hdr = ["level", "quota", "arm", "seed", "block"] + list(METRICS) + ["pred_csv", "n_points"]
    lines, n_ok = ["\t".join(hdr)], 0
    for run in sorted(p.name for p in res_root.iterdir() if p.is_dir()):
        if not run.startswith(("ns5_", "ns5solo_", "ns5sens_", "ns5fix_", "ns5fixsolo_", "ns5lr_")):
            continue
        parts = run.split("_")
        if len(parts) < 5:
            print(f"SKIP {run}: run 名不符 ns5[_solo|_sens|_fix|_fixsolo|_lr]<档>_<配额>_<臂>_s<种子>")
            continue
        lvl, quota, arm, seed = parts[1], parts[2], parts[3], parts[4]
        block = {"ns5": "main", "ns5solo": "solo", "ns5sens": "sens", "ns5fix": "fix",
                 "ns5fixsolo": "fixsolo", "ns5lr": "lr"}.get(parts[0])
        if block is None:          # 前缀认不全就跳过并喊出来，绝不默认成 main（那会把新块混进登记判决）
            print(f"SKIP {run}: 未知前缀 {parts[0]!r}")
            continue
        f = res_root / run / "predictions" / f"C-val_ns_re{lvl}_predictions.csv"
        if not f.exists():
            print(f"SKIP {run}: 没有 {f.name}")
            continue
        rr = rows_of(f)
        pred = [float(r["p_pred"]) for r in rr]
        truth = [float(r["p_true"]) for r in rr]
        bc = [{"inlet": 1, "outlet": 2}.get(r["boundary_type"], 0) for r in rr]
        sp = [math.hypot(float(r["u_pred"]), float(r["v_pred"])) for r in rr]
        sv = [math.hypot(float(r["u_true"]), float(r["v_true"])) for r in rr]
        _, _, dp_err = drop_err(pred, truth, bc)
        lines.append("\t".join([lvl, quota, arm, seed, block,
                                f"{mean_free_rel(pred, truth):.6g}", f"{dp_err:.6g}",
                                f"{mean_free_rel(sp, sv):.6g}", str(f), str(len(rr))]))
        n_ok += 1
    out_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"PINN-SCORE rows={n_ok} -> {out_tsv}")
    return 0 if n_ok else 1


def load_pinn(path: Path) -> dict:
    cell = {}
    for r in tsv_rows(path):
        block = r.get("block") or "main"
        cell.setdefault((r["level"], r["quota"], r["arm"], block), []).append(
            {m: float(r[m]) for m in METRICS})
    return cell


def load_base(path: Path) -> dict:
    """matrix_baseline.tsv：驱动把 assim 打印的 `BASELINE k=v ...` 行原样入库；只取评价工况 C-val。"""
    cell = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("BASELINE"):
            continue
        kv = {}
        for tok in line.split()[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k] = v
        if "case" not in kv or not kv["case"].startswith("C-val"):
            continue
        lvl = ("%g" % float(kv["reynolds"])).replace(".0", "") if float(kv["reynolds"]) % 1 == 0 \
            else kv["reynolds"]
        cell.setdefault((str(lvl), kv["quota"], kv["equation"]), []).append(
            {m: float(kv[m]) for m in METRICS if m in kv})
    for key, rows in cell.items():
        if len(rows) > 1:
            print(f"NOTE 基线同一格有 {len(rows)} 行（{key}）⇒ 取中位，并把这当作重复运行留痕")
    return cell


def med_sd(vals):
    sd = st.stdev(vals) if len(vals) > 1 else float("nan")
    return st.median(vals), sd


def judge(pinn_tsv: Path, base_tsv: Path, out_path: Path) -> int:
    P, Bs = load_pinn(pinn_tsv), load_base(base_tsv)
    out = []
    # ---------------- K1 ----------------
    out.append("== K1 绝对门槛（5pct：去均值压力 ≤%g 且 压降 ≤%g，两量同时达标才算这一臂可用）"
               % (JUD_P, JUD_DP))
    k1 = {}
    for lvl in LEVELS:
        for arm in ARMS:
            for blk in BLOCKS:
                rows = P.get((lvl, "5pct", arm, blk))
                lab = f"PINN/{arm}({blk})"
                if not rows:
                    out.append(f"K1 {lvl} {lab} 未测（缺读数）")
                    continue
                pm, _ = med_sd([r["rel_l2_p_meanfree"] for r in rows])
                dm, _ = med_sd([r["pressure_drop_rel_error"] for r in rows])
                ok = pm <= JUD_P and dm <= JUD_DP
                k1[(lvl, lab)] = ok
                out.append(f"K1 {lvl} {lab} n={len(rows)} 去均值压力中位={pm:.5g} 压降中位={dm:.5g} "
                           f"-> {'达标' if ok else '不达标'}")
        for arm in ARMS:
            rows = Bs.get((lvl, "5pct", arm))
            lab = f"经典/{arm}(单工况)"
            if not rows:
                out.append(f"K1 {lvl} {lab} 未测（缺读数）")
                continue
            pm, _ = med_sd([r["rel_l2_p_meanfree"] for r in rows])
            dm, _ = med_sd([r["pressure_drop_rel_error"] for r in rows])
            ok = pm <= JUD_P and dm <= JUD_DP
            k1[(lvl, lab)] = ok
            out.append(f"K1 {lvl} {lab} n={len(rows)} 去均值压力中位={pm:.5g} 压降中位={dm:.5g} "
                       f"-> {'达标' if ok else '不达标'}")
    for lvl in LEVELS:
        got = sorted(a for (l, a), v in k1.items() if l == lvl and v)
        out.append(f"K1 @Re={lvl} = " + (f"达（{'、'.join(got)}）" if got else "不达"))
    k1_any = any(k1.values())
    # ---------------- K2 ----------------
    out.append("")
    out.append("== K2 配对：Δ = 误差(经典同方程) − 误差(PINN 同方程)，正 = PINN 更好；"
               "三条全过（med>0、≥2/3 同向、med ≥ PINN 跨种子 sd 的 1/3）才算该行 MET")
    out.append("block\tlevel\tmetric\tarm\tn\tmed_d\tmean_d\tpinn_sd\tsame_dir\tverdict")
    v2 = {}
    for blk in BLOCKS:
        for lvl in LEVELS:
            for metric in JUDGE_METRICS:
                for arm in ARMS:
                    prow = P.get((lvl, "5pct", arm, blk))
                    brow = Bs.get((lvl, "5pct", arm))
                    if not prow or not brow:
                        out.append(f"{blk}\t{lvl}\t{metric}\t{arm}\t0\tNO-PAIRS\t\t\t\t不进判决")
                        continue
                    bval = med_sd([r[metric] for r in brow])[0]
                    ds = [bval - r[metric] for r in prow]
                    _, sd = med_sd([r[metric] for r in prow])
                    dmed = st.median(ds)
                    pos = sum(1 for d in ds if d > 0)
                    n = len(ds)
                    ok = dmed > 0 and pos >= (2 * n + 2) // 3 and (sd == sd and dmed >= sd / 3.0)
                    v2[(blk, lvl, metric, arm)] = ok
                    out.append(f"{blk}\t{lvl}\t{metric}\t{arm}\t{n}\t{dmed:+.6g}\t{st.mean(ds):+.6g}\t"
                               f"{sd:.4g}\t{pos}/{n}\t" + ("MET" if ok else "not met") + "\t[" +
                               ",".join(f"{d:+.4g}" for d in ds) + "]")
    main_rows = {k: v for k, v in v2.items() if k[0] == "main"}
    k2 = bool(main_rows) and all(main_rows.values())
    out.append("K2(main) = " + ("达" if k2 else "不达") +
               f"（进判决的行 {sum(main_rows.values())}/{len(main_rows)}；"
               "分母取 PINN 跨种子 sd，经典臂是确定性求解 ⇒ 没有可对齐的种子散布，这一格只能单边用；"
               "经典臂本来就是单工况求解，所以它同时是 solo 块的对手）")
    # ---------------- K3 ----------------
    out.append("")
    out.append("== K3 观测密度曲线 R(q) = 误差(q)/误差(%s)，分母=同臂同档同机制的参照配额中位；"
               "只报形状，不设门槛" % REF_QUOTA)
    out.append("block\tlevel\t臂\t" + "\t".join(f"{q}:p/压降/R_p/R_dp" for q in QUOTAS))
    for blk in BLOCKS:
        for lvl in LEVELS:
            for arm in ARMS:
                for tag, getter in (("PINN", lambda q, a: P.get((lvl, q, a, blk))),
                                    ("经典", lambda q, a: (Bs.get((lvl, q, a)) if blk == "main" else None))):
                    # 经典臂是逐工况确定性求解，与 PINN 的训练块（main/solo/sens）无关；
                    # 挂在每个块下重复印一遍会让人以为有三份独立读数。
                    ref = getter(REF_QUOTA, arm)
                    if not ref:
                        continue
                    rp = med_sd([r["rel_l2_p_meanfree"] for r in ref])[0]
                    rd = med_sd([r["pressure_drop_rel_error"] for r in ref])[0]
                    cells = []
                    for q in QUOTAS:
                        rows = getter(q, arm)
                        if not rows:
                            cells.append("未测")
                            continue
                        pm = med_sd([r["rel_l2_p_meanfree"] for r in rows])[0]
                        dm = med_sd([r["pressure_drop_rel_error"] for r in rows])[0]
                        cells.append(f"{pm:.4g}/{dm:.4g}/{pm / rp if rp else float('nan'):.2f}/"
                                     f"{dm / rd if rd else float('nan'):.2f}")
                    out.append(f"{blk}\t{lvl}\t{arm}({tag})\t" + "\t".join(cells))
    # ---------------- 四象限 ----------------
    quad = {(True, True): "K1达&K2达 -> 只给速度观测也能把压力恢复出来，且在同批观测下优于经典压力重建",
            (True, False): "K1达&K2不达 -> 能做出来，但这批观测下经典重建不输/更准：卖点不能是精度，"
                           "只能是免网格与跨工况复用（那一格仍未测）",
            (False, True): "K1不达&K2达 -> 两臂都不够可用、PINN 相对占优：不许用相对占优冒充可用",
            (False, False): "K1不达&K2不达 -> 负结果：只看速度时，两种做法都恢复不出可用压力场"}
    out.append("")
    out.append("QUADRANT  " + quad[(k1_any, k2)])
    out.append("NOTE n=3 ⇒ 配对 Wilcoxon 最小可达双侧 p=0.0625，任何一行都不许写'显著'；"
               "两臂都不含任何压力测量；本表不与'有压力观测'那批（稀疏档 §二/§三）并表；"
               "去均值 ⇒ 压力常数未恢复，不许写成'压力水平也恢复了'。")
    text = "\n".join(out) + "\n"
    out_path.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--c6-check", action="store_true")
    ap.add_argument("--pinn-score", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--root", default="/mnt/workspace/pinn-repro-2026")
    ap.add_argument("--out", default="/mnt/workspace/_ops/ns5/20261003/out")
    ap.add_argument("--cases", default=",".join(f"{c}_ns_re{l}" for l in ("10", "50") for c in
                                                 ("C-base", "C-train-1", "C-train-2", "C-train-3",
                                                  "C-train-4", "C-train-5", "C-val")))
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    if a.selftest:
        return selftest()
    if a.c6_check:
        return c6_check(root / "model" / "cases" / "contraction_2d" / "data",
                        [c.strip() for c in a.cases.split(",")], QUOTAS)
    if a.pinn_score:
        return pinn_score(root / "model" / "results" / "pinn", out / "matrix_pinn.tsv")
    if a.judge:
        return judge(out / "matrix_pinn.tsv", out / "matrix_baseline.tsv", out / "verdict_K123.tsv")
    ap.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
