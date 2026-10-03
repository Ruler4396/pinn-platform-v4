#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NS 档 case 件生成器：把某工况的 NS 真值 (u,v,p) 换进该工况已入库的 field_dense.csv。

为什么可以这样换（两条本回合现读的事实，不是假设）：
1. 工况 id 是**按路径解析**的，不是注册表里的：`train_supervised.load_case_source` 走
   `cases/<family>/data/<case_id>/<file>`；family 才需要在 DEFAULTS 里注册。⇒ 新建目录不会
   污染主线任何一格（主线吃的是 `spec.default_train_cases`），也不需要改注册表。
2. field_dense.csv 里的特征列（壁距、region、bc/边界类型、 inlet profile 等）全部是**几何的函数**，
   NS 档与 Stokes 档同网格同边界 ⇒ 只有 u_star/v_star/p_star 该换。本脚本据此把"其余列逐位不变"
   做成硬断言：任一非值列被改动就拒绝写出。

用法（实例上，带 pandas）：
  python3 make_ns_case.py --family contraction_2d --base C-base --level 10 \
      --raw /mnt/workspace/_ops/ns3p/truth/C-base_ns_re10_raw.csv --root /mnt/workspace/pinn-repro-2026
写出： <root>/model/cases/<family>/data/<base>_ns_re<level>/field_dense.csv  (+ meta.json 一行说明)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VALUE_COLS = ("u_star", "v_star", "p_star")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="contraction_2d")
    ap.add_argument("--base", required=True, help="已入库工况 id，如 C-base / C-val")
    ap.add_argument("--level", required=True, help="Re 档标签，如 1e-3 / 1 / 10 / 50")
    ap.add_argument("--raw", required=True, help="该档 NS 真值 csv（字段名须与 Stokes raw 相同）")
    ap.add_argument("--root", required=True, help="代码树根（含 model/cases）")
    ap.add_argument("--allow-existing", action="store_true")
    a = ap.parse_args()

    root = Path(a.root)
    src = root / "model" / "cases" / a.family / "data" / a.base / "field_dense.csv"
    out_dir = root / "model" / "cases" / a.family / "data" / f"{a.base}_ns_re{a.level}"
    out = out_dir / "field_dense.csv"
    if not src.exists():
        raise SystemExit(f"[FAIL] 找不到基准 dense 件：{src}")
    if out.exists() and not a.allow_existing:
        raise SystemExit(f"[FAIL] 目标已存在，拒绝覆盖（要覆盖加 --allow-existing）：{out}")

    import pandas as pd

    dense = pd.read_csv(src)
    ns = pd.read_csv(a.raw)
    for col in ("x_star", "y_star", "bc_tag"):
        if col not in dense.columns or col not in ns.columns:
            raise SystemExit(f"[FAIL] 网格对齐无法核对（缺列 {col}）：dense={list(dense.columns)[:6]} ns={list(ns.columns)[:6]}")
    if len(dense) != len(ns):
        raise SystemExit(f"[FAIL] 行数不等：dense={len(dense)} ns={len(ns)} —— NS 档必须与 Stokes 同网格")
    for col in ("x_star", "y_star", "bc_tag"):
        d, n = dense[col].to_numpy(), ns[col].to_numpy()
        if not (d.shape == n.shape and bool((d == n).all())):
            import numpy as np
            bad = int((~np.isclose(d.astype(float), n.astype(float))).sum())
            raise SystemExit(f"[FAIL] 几何列 {col} 不一致（{bad}/{len(d)} 行）——这不是同一张网格，不能这样换值")

    changed = 0
    for col in VALUE_COLS:
        if col not in ns.columns:
            raise SystemExit(f"[FAIL] NS 真值缺列 {col}：{list(ns.columns)}")
        dense[col] = ns[col].to_numpy()
        changed += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    dense.to_csv(out, index=False)
    meta = {
        "case_id": f"{a.base}_ns_re{a.level}", "derived_from_dense": str(src),
        "ns_truth_raw": a.raw, "reynolds_label": a.level, "family": a.family,
        "value_columns_replaced": list(VALUE_COLS),
        "geometry_columns_identical_asserted": ["x_star", "y_star", "bc_tag"],
        "note": "工况 id 按路径解析（train_supervised.load_case_source），新建目录不改动主线任何一格",
    }
    (out_dir / "ns_derivation.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {out.name} 行数={len(dense)} 换掉 {changed} 个值列；几何列已核逐位相同 -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
