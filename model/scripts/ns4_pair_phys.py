#!/usr/bin/env python3
"""ns4_pair_phys.py —— 把预注册 §七 J4 缺的那半补上：「Stokes 臂 vs 无物理」也要单列。

主批那枚 phase_pair 的比较臂钉死在 NS 上，所以它只会印「NS vs 无物理」一行；§七 要求
"任一物理臂 vs 无物理……两臂分别列，不合成平均"，那一半没被任何现有产物回答。

三条条件（med Δ > 0、≥2/3 种子同向、med ≥ 比较臂跨种子 sd 的 1/3，sd 用 ddof=1）
与 run_ns4sparse.sh:phase_pair 逐字符同尺——本件里那段配对代码是从它照抄的，不是重写的一把新尺。
自检：喂同一份 matrix_j4.tsv，本件 ns:nophy 那几行必须与 driver 的 J2 行逐位相同。
"""
import argparse
import csv
import statistics as st
import sys
from pathlib import Path

JUD = ("rel_l2_speed", "rel_l2_p")


def num(s):
    o = {}
    for tok in (s or "").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                o[k] = float(v)
            except ValueError:
                pass
    return o


def load(paths):
    cell, wts = {}, {}
    for p in paths:
        with open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r["arm"] not in ("ns", "stokes", "nophy"):
                    continue
                cell[(r["level"], r["arm"], r["seed"])] = num(r["metrics"])
                wts[(r["level"], r["arm"])] = r.get("pde_weights", "")
    return cell, wts


def pair(cell, lvl, ref, cmp_, m):
    seeds = sorted({s for (l, a, s) in cell if l == lvl and a == ref} &
                   {s for (l, a, s) in cell if l == lvl and a == cmp_})
    ds, cvals = [], []
    for s in seeds:
        a = cell[(lvl, ref, s)].get(m)
        b = cell[(lvl, cmp_, s)].get(m)
        if a is None or b is None:
            continue
        ds.append(a - b)
        cvals.append(b)
    return ds, cvals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", action="append", required=True, help="可多枚，同 level/arm/seed 以先读到的为准")
    ap.add_argument("--pairs", default="ns:nophy,stokes:nophy", help="比较臂:参照臂，逗号分隔；Δ=误差(参照)-误差(比较)")
    ap.add_argument("--levels", default="10,50")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cell, wts = load(a.tsv)
    out = ["== 每臂实际用的动量权重（§七 的报告义务：读数旁边要看得见 w）"]
    for (lvl, arm), w in sorted(wts.items()):
        mom = num(w.replace(",", " ")).get("耦合动量", "NA")
        out.append(f"{lvl}\t{arm}\t耦合动量={mom}\t原始={w}")
    out.append("")
    out.append("== 判据：Δ = 误差(参照臂) − 误差(比较臂)，正 = 比较臂更好；三条全过才 MET")
    out.append("rule\tpairs\tlevel\tmetric\tmed_d\tmean_d\tcmp_sd\td_sd\tmed_d/d_sd\tsame_dir\tverdict")
    state = {}
    skipped = {}
    for spec in a.pairs.split(","):
        cmp_, ref = spec.split(":")
        name = f"{cmp_} 对 {ref}"
        state.setdefault(name, [])
        skipped.setdefault(name, [])
        for lvl in a.levels.split(","):
            for m in JUD:
                ds, cvals = pair(cell, lvl, ref, cmp_, m)
                if not ds:
                    out.append(f"{name}\t0\t{lvl}\t{m}\tNO-PAIRS\t\t\t\t\t\t未判（这一对在这一档没有配齐的格）")
                    skipped[name].append(f"{lvl}/{m}")
                    continue
                med = st.median(ds)
                try:
                    sd = st.stdev(cvals)
                except st.StatisticsError:
                    sd = float("nan")
                try:
                    dsd = st.stdev(ds)
                except st.StatisticsError:
                    dsd = float("nan")
                pos = sum(1 for d in ds if d > 0)
                n = len(ds)
                ok = med > 0 and pos >= (2 * n + 2) // 3 and (sd == sd and med >= sd / 3.0)
                state.setdefault(name, []).append(ok)
                out.append(f"{name}\t{n}\t{lvl}\t{m}\t{med:+.6g}\t{st.mean(ds):+.6g}\t{sd:.4g}\t"
                           f"{dsd:.4g}\t"
                           + (f"{med/dsd:+.2f}" if dsd == dsd and dsd > 0 else "NA")
                           + f"\t{pos}/{n}\t" + ("MET" if ok else "not met")
                           + "\t[" + ",".join(f"{d:+.5g}" for d in ds) + "]")
    out.append("")
    for name, oks in state.items():
        # all([]) 在 python 里是 True——一行实测都没有时**不许**报 MET，也不许报 not met，只许报"未判"。
        if not oks:
            out.append(f"判决 {name}：未判（0 行实测可配，缺 " + ";".join(skipped[name]) + "）")
            continue
        v = "MET" if all(oks) else "not met"
        tail = f"（{sum(oks)}/{len(oks)} 行过；要求已配齐档位×两项判据指标全过）"
        if skipped[name]:
            tail += "；未进判决的缺行 " + ";".join(skipped[name])
        out.append(f"判决 {name}：{v}" + tail)
    out.append("NOTE n<=3 时配对 Wilcoxon 最小可达双侧 p=0.0625 -> 任何一行都不许写'显著'；"
               "评分口径=val（最终验证指标）；本件的 ns:nophy 行应与 driver phase_pair 的 J2 行逐位相同。")
    text = "\n".join(out) + "\n"
    Path(a.out).write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
