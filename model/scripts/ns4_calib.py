#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""强度对齐的选权件：从已跑的 run 里量出 r = w×mean(耦合_动量损失)/mean(耦合_速度监督损失)，
按预注册 §七 的规则给每个 (臂, Re) 选一个 w，使 |log10(r/1.0)| 最小。

规则写死在 `稀疏档方程对照-预注册设计-20261003.md` §七，本件只是把那条规则执行成代码：
  · 候选 W = {0.1, 0.3, 1.0, 3.16, 10.0}，但**只有实测过的 w 才算候选**（不外推猜 r）；
  · 某 (臂, Re) 的全部实测 r 都离 1.0 太远（min|log10 r| > log10(3)）⇒ 标 `unalignable`，
    该组合不进 J3，而不是挑一个"最不差"的蒙混过去；
  · 无物理臂不参与选权（它的 PDE 权重恒 0）。
输出：一张人类可读的表 + `weights.tsv`（臂 / Re / 选中的 w / 该 w 下实测 r / 状态），
      并在 stdout 末尾打 `ALIGN_OK n/m`——分母是"应当对齐的组合数"，让"对不齐几格"看得见。
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

CANDIDATES = (0.1, 0.3, 1.0, 3.16, 10.0)
TOL_LOG = math.log10(3.0)          # r 必须落在 [1/3, 3] 才算对齐（§七 写死）


def measure(run_root: str, prefix: str) -> dict:
    """对每个匹配 prefix 的 run 目录，量 w 与两项损失的均值，返回 {(arm,level): {w: (r, mom, sup)}}。"""
    import csv

    out: dict = {}
    for name in sorted(os.listdir(run_root)):
        d = os.path.join(run_root, name)
        if not (name.startswith(prefix) and os.path.isdir(d)):
            continue
        h = os.path.join(d, "history.csv")
        c = os.path.join(d, "config.json")
        if not (os.path.exists(h) and os.path.exists(c)):
            continue
        parts = name.split("_")
        # 只有 **主批(w=10)** 与 **pilot(w=1.0)** 是 §七 的输入。
        # 已经排除：`_c_`（对齐批，用它选权就是循环论证）、`_bc_`（§六 的边界项对照，另一口径）、
        # `ns4_smoke/c3` 之类的控制格。
        tag = parts[1] if len(parts) > 2 and parts[1] in ("c", "bc", "p") else ""
        if tag in ("c", "bc"):
            continue
        if "c3" in name or "smoke" in name:
            continue
        # run 名有两种形状：ns4_<lvl>_<arm>_s<seed>（主批）与 ns4_p_<lvl>_<arm>_s<seed>（pilot）。
        # 按"第一个纯数字段"认 Re 档而不是固定下标——写死下标会在加前缀的那刻起悄悄错位。
        level = arm = None
        for i, tok in enumerate(parts):
            if tok.replace(".", "").replace("-", "").isdigit() and i + 1 < len(parts):
                level, arm = tok, parts[i + 1]
                break
        if level is None or arm not in ("ns", "stokes"):
            continue
        try:
            cfg = json.load(open(c, encoding="utf-8"))
            w = float(cfg.get("权重", {}).get("耦合动量", float("nan")))
        except Exception:
            continue
        try:
            rows = list(csv.DictReader(open(h, encoding="utf-8")))
        except Exception:
            continue

        def mean(col):
            v = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
            return sum(v) / len(v) if v else None

        mom, sup = mean("耦合_动量损失"), mean("耦合_速度监督损失")
        if not mom or not sup or w != w:
            continue
        out.setdefault((arm, level), {})[w] = (w * mom / sup, mom, sup)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--prefix", default="ns4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meas = measure(a.run_root, a.prefix)
    rows, aligned, total = [], 0, 0
    print("arm   Re  实测(w→r)                          选中w   r@选中   状态")
    for key in sorted(meas, key=lambda k: (k[1], k[0])):
        arm, level = key
        seen = meas[key]
        total += 1
        table = " ".join(f"{w:g}->{seen[w][0]:.3f}" for w in sorted(seen))
        ok = [(abs(math.log10(seen[w][0])), w) for w in seen if w in CANDIDATES]
        if not ok:
            rows.append({"arm": arm, "level": level, "w": None, "r": None, "status": "no-candidate-measured"})
            print(f"{arm:7s} {level:4s} {table:36s}  --      --     无可测候选（要先跑 pilot）")
            continue
        best = min(ok)[1]
        r = seen[best][0]
        if abs(math.log10(r)) > TOL_LOG:
            rows.append({"arm": arm, "level": level, "w": best, "r": r, "status": "unalignable"})
            print(f"{arm:7s} {level:4s} {table:36s}  {best:<6g} {r:<8.3f} 对不齐（r 出带）→ 不进 J3")
            continue
        aligned += 1
        rows.append({"arm": arm, "level": level, "w": best, "r": r, "status": "aligned"})
        print(f"{arm:7s} {level:4s} {table:36s}  {best:<6g} {r:<8.3f} aligned")
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("arm\tlevel\tw\tr\tstatus\n")
        for r in rows:
            fh.write(f"{r['arm']}\t{r['level']}\t{r['w']}\t{r['r']}\t{r['status']}\n")
    print(f"ALIGN_OK {aligned}/{total}  写出 {a.out}")
    if aligned < total:
        print("NOTE 有组合对不齐：J3 只在齐的那些组合上判，缺的组合按\"未测/对不齐\"写，不硬凑")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
