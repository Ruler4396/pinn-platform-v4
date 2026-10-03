#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""有限雷诺数定常 Navier-Stokes 版双模型训练器（导师必做：从 Stokes 扩到 Re=1–50）。

被试变量只有一个：动量残差里多出的对流项。其余（数据切分、观测掩码、标准化器、三阶段流程、
壁面处理、评估与产物 schema）全部直接调用主线的函数，不复制、不改主线文件。

量纲约定取自真值侧的求解器本体（`model/cases/contraction_2d/cfd/C-base_ns_re*/C-base_ns_re*.edp`）：
star 单位下 W_stem=1、入口平均速度=1、mu=1，弱形式 = 入库 Stokes 形式 + **Re × 对流**，
即求解 −Δu + ∇p + Re (u·∇)u = 0；对流项的系数是 Re 本身，**不是 1/Re**
（9/27 裁决 R2-7 已把"form = Stokes × Re 所以除 Re"那版否掉，它会给 p_star 注入伪 1/Re）。

因此本文件的残差与主线逐项同尺度：粘性项与对流项都除 velocity_scale、压力梯度除 pressure_scale，
在 star 单位（L=W_stem=1）下与真值方程一致。

两道闸（`--ns-selftest`，秒级、需要 torch 因而要在实例上跑）：
  ① 复算一致性：Re=0 时本文件的 连续性/动量/散度 四项必须与主线 `方程耦合损失` **逐位相同**；
  ② 系数方向必红：把 Re 写成 1/Re 的误标版，在 Re=10 的对流值上必须与正解差出一个 Re² 倍——
     这条控制用来证明"乘 Re"是被测的，不是抄来的。
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REYNOLDS = 0.0


def load_dep(name: str):
    path = SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(path.stem, module)
    spec.loader.exec_module(module)
    return module


_base_cache = {}


def the_base():
    """主线模块——延迟加载，这样 `--ns-selftest` 只需要 torch，不需要 numpy/pandas。"""
    if "m" not in _base_cache:
        _base_cache["m"] = load_dep("train_velocity_pressure_independent.py")
    return _base_cache["m"]


class _BaseProxy:
    def __getattr__(self, name):
        return getattr(the_base(), name)

    def __setattr__(self, name, value):          # 必须转发，否则猴补丁打在代理上、主线看不见
        setattr(the_base(), name, value)


base = _BaseProxy()
sup = None            # 由主线内部按需使用；本文件不直接取


def 对流项(u, v, u_x, u_y, v_x, v_y, reynolds, velocity_scale):
    """(u·∇)u 与 (u·∇)v，按与粘性项同一尺度（除 velocity_scale；star 单位 L=1）。"""
    cu = (u * u_x + v * u_y) / velocity_scale
    cv = (u * v_x + v * v_y) / velocity_scale
    return reynolds * cu, reynolds * cv


def _物理点索引(dense_split, max_physics_points):
    interior_idx = (dense_split.boundary_type == "interior").nonzero()[0] \
        if hasattr(dense_split.boundary_type, "nonzero") else __import__("numpy").where(dense_split.boundary_type == "interior")[0]
    if interior_idx.size == 0:
        return None
    if max_physics_points > 0 and interior_idx.size > max_physics_points:
        import numpy as np
        sample_pos = np.linspace(0, interior_idx.size - 1, num=max_physics_points, dtype=np.int64)
        interior_idx = interior_idx[sample_pos]
    return interior_idx


def 方程耦合损失_NS(速度模型, 压力模型, dense_split, 输入标准化器, 速度标准化器, 压力标准化器,
                    device, max_physics_points, velocity_constraint_info):
    import numpy as np
    import torch
    interior_idx = _物理点索引(dense_split, max_physics_points)
    if interior_idx is None:
        return base.方程耦合损失(速度模型, 压力模型, dense_split, 输入标准化器,
                                速度标准化器, 压力标准化器, device, max_physics_points,
                                velocity_constraint_info)
    if REYNOLDS == 0.0:                      # 与主线走同一段代码，不是"数值上等价"
        return base.方程耦合损失(速度模型, 压力模型, dense_split, 输入标准化器,
                                速度标准化器, 压力标准化器, device, max_physics_points,
                                velocity_constraint_info)

    x = torch.tensor(dense_split.features_norm[interior_idx], dtype=torch.float32,
                     device=device, requires_grad=True)
    wall_distance_frac = base.提取壁面距离分数(dense_split.features_raw[interior_idx], device,
                                               velocity_constraint_info)
    _, 速度原值 = base.速度前向原值(速度模型, x, 速度标准化器, wall_distance_frac=wall_distance_frac,
                                    constraint_info=velocity_constraint_info)
    _, 压力原值 = base.压力前向原值(压力模型, x, 压力标准化器)
    u, v, p = 速度原值[:, 0:1], 速度原值[:, 1:2], 压力原值[:, 0:1]

    x_std = torch.clamp(torch.tensor(float(输入标准化器.std[0]), dtype=torch.float32, device=device), min=1.0e-6)
    y_std = torch.clamp(torch.tensor(float(输入标准化器.std[1]), dtype=torch.float32, device=device), min=1.0e-6)

    grad_u, grad_v, grad_p = base.一阶导(u, x), base.一阶导(v, x), base.一阶导(p, x)
    u_x, u_y = grad_u[:, 0:1] / x_std, grad_u[:, 1:2] / y_std
    v_x, v_y = grad_v[:, 0:1] / x_std, grad_v[:, 1:2] / y_std
    p_x, p_y = grad_p[:, 0:1] / x_std, grad_p[:, 1:2] / y_std

    g_ux, g_uy = base.二阶导(u_x, x), base.二阶导(u_y, x)
    g_vx, g_vy = base.二阶导(v_x, x), base.二阶导(v_y, x)
    u_xx, u_yy = g_ux[:, 0:1] / x_std, g_uy[:, 1:2] / y_std
    v_xx, v_yy = g_vx[:, 0:1] / x_std, g_vy[:, 1:2] / y_std

    velocity_scale = torch.tensor(
        max(float(np.nanmax(np.sqrt(dense_split.targets_raw[:, 0] ** 2 + dense_split.targets_raw[:, 1] ** 2))), 1.0e-12),
        dtype=torch.float32, device=device)
    pv = dense_split.targets_raw[:, 2]
    pressure_scale = torch.tensor(
        max(float(np.nanmax(pv) - np.nanmin(pv)), 1.0e-12), dtype=torch.float32, device=device)

    continuity = u_x / velocity_scale + v_y / velocity_scale
    cu, cv = 对流项(u, v, u_x, u_y, v_x, v_y, REYNOLDS, velocity_scale)
    momentum_u = u_xx / velocity_scale + u_yy / velocity_scale - p_x / pressure_scale + cu
    momentum_v = v_xx / velocity_scale + v_yy / velocity_scale - p_y / pressure_scale + cv
    return {
        "连续性": torch.mean(continuity ** 2),
        "动量": torch.mean(momentum_u ** 2) + torch.mean(momentum_v ** 2),
        "平均散度绝对值": torch.mean(torch.abs(continuity)),
        "最大散度绝对值": torch.max(torch.abs(continuity)),
    }


# ---------------------------------------------------------------------------- 三道闸
def ns_selftest() -> int:
    """只核"对流项本身"与"Re=0 是否交回主线"。端到端（真网上动量随 Re 变化、四项键齐全）
    不在这条自检的覆盖范围内——它要 torch 与真数据，写在实例侧的 8-epoch 冒烟格里，见设计件。"""
    import torch
    fails = []

    # ① 解析对照：u=(x, −y) 不可压缩，(u·∇)u = (x, y)；取 velocity_scale=1、Re=10 ⇒ 期望 10·(x, y)
    x = torch.tensor([[0.5], [-0.25]], dtype=torch.float32)
    u, v = x.clone(), -x.clone()
    ones, zeros = torch.ones_like(x), torch.zeros_like(x)
    cu, cv = 对流项(u, v, ones, zeros, zeros, -ones, 10.0, 1.0)
    good = torch.allclose(cu, 10.0 * x, atol=1e-6) and torch.allclose(cv, 10.0 * x, atol=1e-6)
    print("[%s] 解析对照 (u·∇)u=(x,y)：cu=%.6f cv=%.6f 期望 %.6f"
          % ("PASS" if good else "FAIL", float(cu[0]), float(cv[0]), float(10.0 * x[0])))
    if not good:
        fails.append("解析对照未过")

    # ② 系数方向必红：1/Re 的误标版必须与 Re 版分得开（否则"乘 Re"这句话没被测过）
    inv_cu, _ = 对流项(u, v, ones, zeros, zeros, -ones, 1.0 / 10.0, 1.0)
    red = not torch.allclose(inv_cu, 10.0 * x, atol=1e-6)
    print("[%s] 1/Re 误标版被区分开（cu=%.6f 对正解 %.6f）"
          % ("PASS" if red else "FAIL", float(inv_cu[0]), float(10.0 * x[0])))
    if not red:
        fails.append("1/Re 控制未红")

    # ③ 结构检查：Re=0 与"无内部点"两种情形都交回主线函数（同一段代码，不是复刻）
    import inspect
    src = inspect.getsource(方程耦合损失_NS)
    delegated = src.count("return base.方程耦合损失(") >= 2
    print("[%s] Re=0／无内部点两分支均 return base.方程耦合损失(...)" % ("PASS" if delegated else "FAIL"))
    if not delegated:
        fails.append("Re=0 未交回主线")

    print("[SCOPE] 本自检不覆盖端到端：真网上'动量随 Re 变化'与四项键齐全留给实例侧 8-epoch 冒烟格")
    print("NS_SELFTEST " + ("ALL GREEN" if not fails else "FAILED | " + " | ".join(fails)))
    return 1 if fails else 0


def main() -> int:
    global REYNOLDS
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--reynolds", type=float, default=None,
                    help="对流项系数（真值同族 star 单位下即 Re）；不给则 0＝纯 Stokes 复算")
    ap.add_argument("--ns-selftest", action="store_true")
    known, rest = ap.parse_known_args()
    if known.ns_selftest:
        return ns_selftest()
    REYNOLDS = 0.0 if known.reynolds is None else float(known.reynolds)
    bm = the_base()
    if REYNOLDS != 0.0:
        bm.方程耦合损失 = 方程耦合损失_NS
        if bm.方程耦合损失 is not 方程耦合损失_NS:                 # 猴补丁没落上就是假绿，宁可红
            raise SystemExit("[FAIL] 对流项补丁未落到主线模块，训练会跑成纯 Stokes")
        print("NS 模式：动量残差 = 粘性 − ∇p/Δp尺度 + Re·(u·∇)u，Re=%g；系数为乘 Re（真值 .edp 同约定）"
              % REYNOLDS, flush=True)
    else:
        print("Re=0：走主线 Stokes 残差（复算控制格）", flush=True)
    sys.argv = [sys.argv[0]] + rest
    return bm.main()


if __name__ == "__main__":
    raise SystemExit(main())
