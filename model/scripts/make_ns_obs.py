#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 NS 档造**同点位**的稀疏观测表：拿主线 `obs_sparse_5pct.csv` 的坐标，值换成该 Re 档的 NS 真值。

为什么不能直接复制：观测表里的 u_obs/v_obs/p_obs 是 Stokes 真值。直接复制进 NS 工况目录，
两臂就会一起去拟合一个不属于该方程场的标签——被试变量（残差里写不写对流项）之外还混进了
"观测值来自哪个场"这一层，配对就不再干净。

为什么点位要逐位相同：这一格问的是"信息不足时方程写对有没有用"，
点位/点数不是被试变量。所以本件只换值、不动点位，并把这一点做成硬断言：
  · 行数 == 主线 obs 表行数；
  · 每个 (x_star, y_star) 在该档 NS dense 里**恰好命中一次**（坐标逐位相等，不做最近邻容差）；
  · sample_id / region_id / wall_distance_star / sampling_tag / noise_tag 五列原样保留。
另外打印"观测点上的物理修正量"（NS 值与 Stokes 值的最大绝对变化），让读者看得见换值换掉了多少。

用法（实例上）：
  python3 make_ns_obs.py --root /mnt/workspace/pinn-repro-2026 \
      --base C-val --level 10 --obs-stem obs_sparse_5pct
写出： <root>/model/cases/contraction_2d/data/C-val_ns_re10/obs_sparse_5pct.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VALUE_COLS = ("u_obs", "v_obs", "p_obs")
CARRY_COLS = ("sample_id", "family", "sampling_tag", "noise_tag",
              "region_id", "wall_distance_star")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--family", default="contraction_2d")
    ap.add_argument("--base", required=True, help="已入库工况 id，如 C-val / C-train-1")
    ap.add_argument("--level", required=True, help="Re 档标签，如 1e-3 / 1 / 10 / 50")
    ap.add_argument("--obs-stem", default="obs_sparse_5pct")
    ap.add_argument("--allow-existing", action="store_true")
    a = ap.parse_args()

    data = Path(a.root) / "model" / "cases" / a.family / "data"
    src_obs = data / a.base / f"{a.obs_stem}.csv"
    ns_case = f"{a.base}_ns_re{a.level}"
    ns_dense = data / ns_case / "field_dense.csv"
    out = data / ns_case / f"{a.obs_stem}.csv"
    for p, what in ((src_obs, "主线观测表"), (ns_dense, "NS 档 dense 件")):
        if not p.is_file():
            raise SystemExit(f"[FAIL] 缺{what}：{p}")
    if out.exists() and not a.allow_existing:
        raise SystemExit(f"[FAIL] 目标已存在，拒绝覆盖（要覆盖加 --allow-existing）：{out}")

    import pandas as pd

    obs = pd.read_csv(src_obs)
    dense = pd.read_csv(ns_dense)
    for col in ("x_star", "y_star", "u_star", "v_star", "p_star"):
        if col not in dense.columns:
            raise SystemExit(f"[FAIL] NS dense 缺列 {col}：{list(dense.columns)}")
    key = {}
    for row in dense.itertuples(index=False):
        k = (float(row.x_star), float(row.y_star))
        if k in key:
            raise SystemExit(f"[FAIL] NS dense 里坐标 {k} 出现两次，无法按点位对齐")
        key[k] = (row.u_star, row.v_star, row.p_star)

    new_vals, max_move = [], 0.0
    for row in obs.itertuples(index=False):
        k = (float(row.x_star), float(row.y_star))
        if k not in key:
            raise SystemExit(f"[FAIL] 主线观测点 {k} 在 NS dense 里找不到逐位相等的坐标"
                             f"（本件刻意不做最近邻容差：一旦容差介入了，'同点位'这句话就要重写）")
    for row in obs.itertuples(index=False):
        u, v, p = key[(float(row.x_star), float(row.y_star))]
        new_vals.append((u, v, p))
        max_move = max(max_move, abs(u - row.u_obs), abs(v - row.v_obs), abs(p - row.p_obs))
    for i, col in enumerate(VALUE_COLS):
        obs[col] = [t[i] for t in new_vals]
    obs["case_id"] = ns_case
    out.parent.mkdir(parents=True, exist_ok=True)
    obs.to_csv(out, index=False)
    meta = {
        "written_by": "model/scripts/make_ns_case 的观测表姊妹件 make_ns_obs.py",
        "case_id": ns_case, "obs_stem": a.obs_stem,
        "points_taken_from": str(src_obs.relative_to(Path(a.root))),
        "values_taken_from": str(ns_dense.relative_to(Path(a.root))),
        "rows": int(len(obs)), "carried_columns_unchanged": list(CARRY_COLS),
        "coordinate_matching": "逐位相等（无容差），每个点恰好命中一次",
        "max_abs_change_of_any_observed_value_at_those_points": float(max_move),
        "note": "主线 5% 口径是 u/v/p 三量在同 63 点上都有监督（压力不 mask）——"
                "与创新点3那条线的\"速度-only 配额\"不是同一口径，两批读数不可并列",
    }
    (out.parent / f"{a.obs_stem}_ns_derivation.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] {ns_case}/{a.obs_stem}.csv rows={len(obs)} "
          f"点位逐位沿用主线、值换成 Re={a.level} 的 NS 真值（观测点上最大改变 {max_move:.6g}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
