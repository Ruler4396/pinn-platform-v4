#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NS 档 case 件生成器：把某工况的 NS 真值 (u,v,p) 换进该工况已入库的 field_dense.csv。

为什么可以这样换（三条本回合现读的事实，不是假设）：
1. 工况 id 是**按路径解析**的，不是注册表里的：`train_supervised.load_case_source` 走
   `cases/<family>/data/<case_id>/<file>`；family 才需要在 DEFAULTS 里注册。⇒ 新建目录不会
   污染主线任何一格（主线吃的是 `spec.default_train_cases`），也不需要改注册表。
2. field_dense.csv 的特征列（region_id、wall_distance_star、is_boundary、boundary_type）全是
   **几何的函数**，NS 档与 Stokes 档同网格同边界 ⇒ 只有三个值列该换。本脚本据此把"其余列逐位不变"
   做成硬断言：x_star/y_star 任一行不等就拒绝写出。
3. **口径＝6 位有效数字，与主线一致**：已入库的 `C-base/field_dense.csv` 第 2 行是
   `1.91421, 1.38812e-11, 1.09893e-44`，与 `cfd/C-base/C-base_raw.csv` 同格逐字符相同——
   主线自己就是 FreeFEM 那台 6 位打印的产物。所以这里吃 `*_raw.csv`（6 位）而不是
   `*_raw_10dig.csv`（伴流合并件），并且这一点要在产物旁边的 sidecar 里写明。
   `speed_star` 是 u、v 的派生量，换了值列必须重算，否则 dense 件自己跟自己不一致。

用法（实例上，带 pandas）：
  python3 make_ns_case.py --base C-base --level 10 \
      --raw /mnt/workspace/pinn-repro-2026/model/cases/contraction_2d/cfd/C-base_ns_re10/C-base_ns_re10_raw.csv \
      --root /mnt/workspace/pinn-repro-2026
写出： <root>/model/cases/<family>/data/<base>_ns_re<level>/field_dense.csv (+ ns_case_derivation.json)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VALUE_COLS = ("u_star", "v_star", "p_star")
GEOM_ASSERT = ("x_star", "y_star")
DERIVED_COL = "speed_star"


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
    data = root / "model" / "cases" / a.family / "data"
    src = data / a.base / "field_dense.csv"
    case_id = f"{a.base}_ns_re{a.level}"
    out_dir = data / case_id
    out = out_dir / "field_dense.csv"
    if not src.exists():
        raise SystemExit(f"[FAIL] 找不到基准 dense 件：{src}")
    if out.exists() and not a.allow_existing:
        raise SystemExit(f"[FAIL] 目标已存在，拒绝覆盖（要覆盖加 --allow-existing）：{out}")

    import numpy as np
    import pandas as pd

    dense = pd.read_csv(src)
    ns = pd.read_csv(a.raw)
    for col in (*GEOM_ASSERT, *VALUE_COLS):
        if col not in ns.columns:
            raise SystemExit(f"[FAIL] NS 真值缺列 {col}：{list(ns.columns)}")
    if len(dense) != len(ns):
        raise SystemExit(f"[FAIL] 行数不等：dense={len(dense)} ns={len(ns)} —— NS 档必须与 Stokes 同网格")
    for col in GEOM_ASSERT:
        d, n = dense[col].to_numpy(dtype=float), ns[col].to_numpy(dtype=float)
        bad = int((~(d == n)).sum())
        if bad:
            raise SystemExit(f"[FAIL] 几何列 {col} 有 {bad}/{len(d)} 行不等——这不是同一张网格，"
                             f"不能只换值列")
    before = dense[list(VALUE_COLS)].to_numpy(dtype=float)
    for col in VALUE_COLS:
        dense[col] = ns[col].to_numpy(dtype=float)
    if DERIVED_COL in dense.columns:
        dense[DERIVED_COL] = np.hypot(dense["u_star"].to_numpy(dtype=float),
                                      dense["v_star"].to_numpy(dtype=float))
    if "case_id" in dense.columns:
        dense["case_id"] = case_id
    after = dense[list(VALUE_COLS)].to_numpy(dtype=float)
    moved = float(np.abs(after - before).max())
    out_dir.mkdir(parents=True, exist_ok=True)
    dense.to_csv(out, index=False)
    meta = {
        "case_id": case_id,
        "derived_from_dense": str(src.relative_to(root)),
        "ns_truth_raw": a.raw,
        "reynolds_label": a.level,
        "family": a.family,
        "value_columns_replaced": list(VALUE_COLS),
        "derived_column_recomputed": [DERIVED_COL] if DERIVED_COL in dense.columns else [],
        "geometry_columns_identical_asserted": list(GEOM_ASSERT),
        "rows": int(len(dense)),
        "digits_convention": "6 位有效数字（与已入库 Stokes dense 件同一打印口径；10 位伴流合并件"
                             "只用于 C-base 的地板证明，不进训练目标）",
        "max_abs_change_of_any_value_column": moved,
        "note": "工况 id 按路径解析（train_supervised.load_case_source），新建目录不改动主线任何一格",
    }
    (out_dir / "ns_case_derivation.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    print(f"[OK] {case_id}: rows={len(dense)} 换掉 {len(VALUE_COLS)} 个值列"
          f"（最大幅度改变 {moved:.6g}），几何列已核逐位相同 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
