#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 `eval.tsv`（评估器产物，test 口径与跨档外推）读成两张表：同档 test 的两臂配对、跨档外推的迁移代价。

为什么单独一枚而不是塞进 shell 驱动：这批的键在 run 名里（`ns3p_<档>_<臂>_s<种子>`）、
列名也和训练器那批不同；再往 shell 里塞一段 python 只会让两处解析各漂一次。

口径纪律（预注册 §八）：评估器产物（`evaluations/metrics_<split>.json` 的 `summary.mean_*`）
与训练器自报的 val 段**不同文件不同名**，本件只读前者；两批数不许混填。
Stage 3 那条唯一判据是钉在 **val** 上的，所以这里的 test 配对一律标"登记外：不进判决"。
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
import sys

ARM_RE = re.compile(r"^ns3p_([\w.\-]+)_(ns|stokes)_s(\d+)$")


def load_eval(path):
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"):
        m = ARM_RE.match(r["run"])
        if not m:
            continue
        lvl, arm, seed = m.group(1), m.group(2), m.group(3)
        try:
            vals = {"speed": float(r["speed"]), "p": float(r["p"]), "drop": float(r["drop"])}
        except (TypeError, ValueError):
            continue
        rows.append({"train": lvl, "split": r["split"], "arm": arm, "seed": seed,
                     "cases": r["cases"], **vals})
    return rows


def ms(v):
    if not v:
        return "NA"
    s = ("+-" + f"{st.stdev(v):.4g}") if len(v) > 1 else "+-NA"
    return f"{st.mean(v):.5g}{s}"


def paired(left, right, metric):
    ds = [left[k][metric] - right[k][metric] for k in sorted(left) if k in right]
    return ds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-tsv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load_eval(a.eval_tsv)
    if not rows:
        print("[FAIL] eval.tsv 里没有可解析的 ns3p_* 行（run 名格式或列名不对）")
        return 1
    out = []
    cells = {}
    for r in rows:
        cells[(r["train"], r["split"], r["arm"], r["seed"])] = {
            k: r[k] for k in ("speed", "p", "drop") if r.get(k) is not None}

    out.append("== A. 同档 test 口径（评估器场级；登记外：Stage 3 的判据钉在 val 上）")
    hdr = ["train_Re", "split", "arm", "n", "speed", "p", "drop"]
    out.append("\t".join(hdr))
    for train in ("1", "1e-3", "10", "50"):
        for arm in ("ns", "stokes"):
            seeds = sorted({k[3] for k in cells if k[0] == train and k[2] == arm and k[1] == "test"})
            if not seeds:
                continue
            got = {m: [cells[(train, "test", arm, s)][m] for s in seeds
                       if m in cells.get((train, "test", arm, s), {})] for m in ("speed", "p", "drop")}
            out.append("\t".join([train, "test", arm, str(len(seeds)),
                                  ms(got["speed"]), ms(got["p"]), ms(got["drop"])]))
    out.append("")
    out.append("配对（Δ = 误差(stokes) − 误差(ns)，正 = 写对流那臂更好）：")
    out.append("train_Re\tmetric\tn\tmed_d\tmean_d\td_sd\tsame_dir\t三条条件")
    for train in ("10", "50"):
        for metric in ("speed", "p"):
            L = {s: cells[(train, "test", "stokes", s)] for s in
                 {k[3] for k in cells if k[0] == train and k[1] == "test" and k[2] == "stokes"}
                 if (train, "test", "stokes", s) in cells and (train, "test", "ns", s) in cells}
            R = {s: cells[(train, "test", "ns", s)] for s in L}
            ds = paired(L, R, metric)
            if not ds:
                out.append(f"{train}\t{metric}\t0\tNO-PAIRS\t\t\t")
                continue
            ns_vals = [R[s][metric] for s in sorted(R)]
            med = st.median(ds)
            try:
                sd = st.stdev(ns_vals)
                dsd = st.stdev(ds)
            except st.StatisticsError:
                sd = dsd = float("nan")
            pos = sum(1 for d in ds if d > 0)
            n = len(ds)
            ok = med > 0 and pos >= (2 * n + 2) // 3 and (sd == sd and med >= sd / 3.0)
            out.append(f"{train}\t{metric}\t{n}\t{med:+.6g}\t{st.mean(ds):+.6g}\t{dsd:.4g}\t"
                       f"{pos}/{n}\t" + ("三条全过" if ok else "not met")
                       + "\t[" + ",".join(f"{d:+.5g}" for d in ds) + "]")

    out.append("")
    out.append("== B. 跨雷诺数外推（同一 run 换场评；只报数，不设判据——它回答\"能用在哪\"）")
    out.append("方向\t臂\tn\tspeed(外推)\tp(外推)\tdrop(外推)")
    for split, label in (("ext50", "训 10 → 评 50 场"), ("ext10", "训 50 → 评 10 场")):
        for arm in ("ns", "stokes"):
            tr = "10" if split == "ext50" else "50"
            seeds = sorted({k[3] for k in cells if k[0] == tr and k[1] == split and k[2] == arm})
            if not seeds:
                continue
            got = {m: [cells[(tr, split, arm, s)][m] for s in seeds
                       if m in cells.get((tr, split, arm, s), {})] for m in ("speed", "p", "drop")}
            out.append(f"{label}\t{arm}\t{len(seeds)}\t{ms(got['speed'])}\t"
                       + "\t".join(ms(got[m]) for m in ("p", "drop")))
    out.append("")
    out.append("迁移代价 = 外推误差 / 同档训练误差（分母取 §A 的 test 口径，只列中位数比）")
    out.append("方向\t臂\tspeed 比\tp 比\tdrop 比")
    for split, tr, ref in (("ext50", "10", "50"), ("ext10", "50", "10")):
        for arm in ("ns", "stokes"):
            a_vals = [v for k, v in cells.items() if k[0] == tr and k[1] == split and k[2] == arm]
            b_vals = [v for k, v in cells.items() if k[0] == ref and k[1] == "test" and k[2] == arm]
            if not a_vals or not b_vals:
                continue
            def ratio(m):
                x = st.median([v[m] for v in a_vals if m in v])
                y = st.median([v[m] for v in b_vals if m in v])
                return f"{x / y:.2f}" if y else "NA"
            label = "训 10 → 评 50" if split == "ext50" else "训 50 → 评 10"
            out.append(f"{label}\t{arm}\t" + "\t".join(ratio(m) for m in ("speed", "p", "drop")))
    out.append("")
    out.append("NOTE 分母是\"在同一档上训练并在同一档上评\"的那一臂，不是别的臂；")
    out.append("     比值 >1 只说明跨档变差多少，不构成任何\"谁更好\"的判断。")
    txt = "\n".join(out) + "\n"
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
