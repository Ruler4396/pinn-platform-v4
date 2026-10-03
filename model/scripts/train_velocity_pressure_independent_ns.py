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

四道闸（`--ns-selftest`）：
  ① 解析对照 (u·∇)u=(x,y)；② 1/Re 误标版必红（证明"乘 Re"是被测的，不是抄来的）；
  ③ Re=0／无内部点两分支均 return 主线函数（同一段代码，不是复刻）；
  ④ NS 档工况名挂表：非 NS 名不许动、缺 dense 件必须拒、临时目录里走成功路并核几何逐字段继承。
①②③ 要 torch（实例上跑）；④ 纯 stdlib，单独入口 `--ns-registry-selftest` 在本机就能跑，
所以"注册这一半"不必等到占机时才第一次执行——上一轮就是没覆盖写路径才让 mkdir 缺陷漏到实例上。
端到端（真网上动量随 Re 变化、四项键齐全）仍不在自检覆盖内，留给实例侧 8-epoch 冒烟格。
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REYNOLDS = 0.0
BASE_SCRIPT = "train_velocity_pressure_independent.py"
SCALE_VARIANT = "dense"      # dense＝主线稠密版从真值场取比例尺；scaler＝严格稀疏版从标准化器取

# 两版主线的比例尺表达式（逐字符照抄，本件不复用其中一版去套另一版）：
#   train_velocity_pressure_independent.py:369-376            ← nanmax(√(u²+v²)) ／ nanmax(p)−nanmin(p)
#   train_velocity_pressure_independent_strict_sparse.py:380-385 ← ‖速度标准化器.std‖ ／ max(压力标准化器.std)
# 挂错会怎样：稀疏臂里 NS 那一臂就会去读未观测点的真值，两臂差的就不只对流项——Stage 3 之所以站得住，
# 正因为稠密版的尺度与被替换函数逐字相同（本回合复核过，见 Stage3 读数件 §四）。


def 尺度对(variant: str, 速度标准化器, 压力标准化器, dense_split, np):
    """返回 (velocity_scale, pressure_scale) 两个 float。`np` 作参数传入，是为了让这条
    "读不读 dense 真值"的差别能在本机用纯 stdlib 测出来（真跑时传的是真 numpy）。"""
    if variant == "scaler":
        v = float(np.linalg.norm(np.asarray(速度标准化器.std, dtype=np.float64)))
        p = float(np.max(np.asarray(压力标准化器.std, dtype=np.float64)))
        return v, p
    if variant != "dense":
        raise SystemExit(f"[FAIL] 未知尺度口径 {variant!r}（只允许 dense／scaler）")
    v = float(np.nanmax(np.sqrt(dense_split.targets_raw[:, 0] ** 2 + dense_split.targets_raw[:, 1] ** 2)))
    pv = dense_split.targets_raw[:, 2]
    return v, float(np.nanmax(pv) - np.nanmin(pv))


def load_dep(name: str):
    path = SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(path.stem, module)
    spec.loader.exec_module(module)
    return module


_base_cache = {}


def the_base():
    """被挂的主线模块——延迟加载，这样 `--ns-selftest` 只需要 torch，不需要 numpy/pandas。"""
    if BASE_SCRIPT not in _base_cache:
        _base_cache[BASE_SCRIPT] = load_dep(BASE_SCRIPT)
    return _base_cache[BASE_SCRIPT]


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

    v_val, p_val = 尺度对(SCALE_VARIANT, 速度标准化器, 压力标准化器, dense_split, np)
    velocity_scale = torch.tensor(max(v_val, 1.0e-12), dtype=torch.float32, device=device)
    pressure_scale = torch.tensor(max(p_val, 1.0e-12), dtype=torch.float32, device=device)

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


def scale_selftest() -> int:
    """`尺度对` 的四条控制，纯 stdlib（np 是注入的假模块）⇒ 本机就能跑，不必等实例。
    要点是第 ②③ 条：scaler 分支**不许碰** dense 真值，dense 分支**必须碰**——
    一条只测"不读"的断言可以恒真，所以两条一起写。"""
    import math

    class _Vec(list):
        def __pow__(self, e): return _Vec([x ** e for x in self])
        def __add__(self, o): return _Vec([a + b for a, b in zip(self, o)])

    class _NP:
        float64 = float

        class linalg:
            @staticmethod
            def norm(v): return math.sqrt(sum(float(x) ** 2 for x in v))

        @staticmethod
        def asarray(x, dtype=None): return list(x)

        @staticmethod
        def max(v): return max(v)

        @staticmethod
        def nanmax(v): return max(v)

        @staticmethod
        def nanmin(v): return min(v)

        @staticmethod
        def sqrt(v): return _Vec([math.sqrt(x) for x in v])

    class _Boom:
        def __getattr__(self, name): raise AssertionError("scaler 分支读了 dense_split." + name)
        def __getitem__(self, key): raise AssertionError("scaler 分支读了 dense 真值")

    class _Cols:
        def __init__(self, cols): self.cols = cols
        def __getitem__(self, key): return _Vec(self.cols[key[1]])

    class Split:
        def __init__(self, cols): self.targets_raw = _Cols(cols)

    class Scaler:
        def __init__(self, std): self.std = std

    fails = []
    v, p = 尺度对("scaler", Scaler([3.0, 4.0]), Scaler([0.2, 0.7]), _Boom(), _NP)
    good1 = abs(v - 5.0) < 1e-12 and abs(p - 0.7) < 1e-12
    print("[%s] ① scaler 口径：v=‖std‖、p=max(std) 复算 = %g / %g（期望 5 / 0.7）"
          % ("PASS" if good1 else "FAIL", v, p))
    fails += [] if good1 else ["scaler 数值错"]
    try:
        尺度对("scaler", Scaler([3.0, 4.0]), Scaler([0.2, 0.7]), _Boom(), _NP)
        touched = False
    except AssertionError:
        touched = True
    good2 = not touched
    print("[%s] ② scaler 分支在 dense 真值\"一读就抛\"的桩上跑完 ⇒ 它确实不读未观测标签"
          % ("PASS" if good2 else "FAIL"))
    fails += [] if good2 else ["scaler 分支读了 dense 真值"]
    try:
        尺度对("dense", Scaler([3.0, 4.0]), Scaler([0.2, 0.7]), _Boom(), _NP)
        raised = False
    except AssertionError:
        raised = True
    good3 = raised
    print("[%s] ③ 反对照：dense 分支在同一个桩上必须抛（不抛＝②是恒真断言）"
          % ("PASS" if good3 else "FAIL"))
    fails += [] if good3 else ["dense 分支没读真值，②就不作数"]
    v, p = 尺度对("dense", Scaler([1.0]), Scaler([1.0]),
                  Split([_Vec([3.0, 3.0]), _Vec([4.0, 4.0]), _Vec([10.0, 2.0])]), _NP)
    good4 = abs(v - 5.0) < 1e-12 and abs(p - 8.0) < 1e-12
    print("[%s] ④ dense 口径数值复算 = %g / %g（期望 5 / 8）" % ("PASS" if good4 else "FAIL", v, p))
    fails += [] if good4 else ["dense 数值错"]
    try:
        尺度对("whatever", Scaler([1.0]), Scaler([1.0]), _Boom(), _NP)
        refused = False
    except SystemExit:
        refused = True
    print("[%s] ⑤ 未知口径名被拒（不静默退回默认）%s" % ("PASS" if refused else "FAIL", ""))
    fails += [] if refused else ["未知口径没拒"]
    print("SCALE_SELFTEST " + ("ALL GREEN" if not fails else "FAILED | " + " | ".join(fails)))
    return 1 if fails else 0


# ---------------------------------------------------- 运行期把 NS 档挂进收缩族工况表
# 为什么必须挂：`train_supervised` 在按路径取数据之前先过 `src.data.contraction_cases.get_case`
# （实例上实测：`KeyError: Unknown contraction case 'C-base_ns_re10'`）。设计件 §二.1 原先只读了
# `load_case_source` 就断言"新建目录不需要动注册表"——那句被这一跑否证了，改在这里补。
# 为什么在运行期挂而不是改那枚文件：主线的工况条目数是被别处读的对象，加 36 条会连带动到别人的账。
NS_CASE_RE = re.compile(r"^(C-[A-Za-z0-9.\-]+?)_ns_re([0-9][0-9eE.\-]*)$")


def the_registry():
    """取主线**同一个** contraction_cases 模块对象；取不到就明说，不许静默注册到一份副本上。"""
    model_root = SCRIPT_DIR.parents[0]
    if str(model_root) not in sys.path:
        sys.path.insert(0, str(model_root))
    try:
        return importlib.import_module("src.data.contraction_cases")
    except Exception as exc:                                # noqa: BLE001
        raise SystemExit(f"[FAIL] 取不到收缩族工况表模块（注册到副本上等于没注册）：{exc!r}")


def 从argv取工况(argv):
    ids = []
    for i, a in enumerate(argv):
        for flag in ("--train-cases", "--val-cases"):
            if a == flag and i + 1 < len(argv):
                ids += [s.strip() for s in argv[i + 1].split(",") if s.strip()]
            elif a.startswith(flag + "="):
                ids += [s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()]
    return ids


def 注册NS工况(case_ids, data_root=None):
    """把 `<base>_ns_re<lvl>` 挂进工况表：几何逐项继承 base，只有 case_id 与 note 变。
    dense 件必须已经在位——否则宁可停下，也不让它悄悄回落到 Stokes 那一格。"""
    root = Path(data_root) if data_root else SCRIPT_DIR.parents[0] / "cases" / "contraction_2d" / "data"
    mod = the_registry()
    done = []
    for cid in case_ids:
        m = NS_CASE_RE.match(cid)
        if not m:
            continue
        base_id, lvl = m.group(1), m.group(2)
        if cid not in mod._CASE_LIBRARY:
            try:
                b = mod.get_case(base_id)
            except KeyError as exc:
                raise SystemExit(f"[FAIL] NS 档 {cid} 的基准工况 {base_id} 不在收缩族表里：{exc}")
            mod._CASE_LIBRARY[cid] = dataclasses.replace(
                b, case_id=cid,
                note=f"NS 档 Re={lvl}：与 {base_id} 同几何同网格，只有 u/v/p 取该档定常 Navier-Stokes 解")
        dense = root / cid / "field_dense.csv"
        if not dense.is_file():
            raise SystemExit(f"[FAIL] NS 档 {cid} 的 dense 件不存在：{dense} —— 先用 make_ns_case.py 造这一格，"
                             f"不能让它悄悄回落到 {base_id}（那会把 Stokes 值当成 Re={lvl} 的真值来训）")
        done.append(cid)
    return done


def registry_selftest() -> int:
    """三条控制：非 NS 名不许动、缺 dense 件必须拒、给了临时目录就走成功路并核几何继承。纯 stdlib，
    所以这条在本机也能跑（`--ns-registry-selftest`）——不占机时把写路径以外的分支全覆盖掉。"""
    import tempfile
    fails = []
    mod = the_registry()
    n_before = len(mod._CASE_LIBRARY)
    ids = 从argv取工况(["--family", "contraction_2d", "--train-cases",
                       "C-base_ns_re10,C-train-1_ns_re10", "--val-cases=C-val_ns_re10", "--seed", "42"])
    good_parse = ids == ["C-base_ns_re10", "C-train-1_ns_re10", "C-val_ns_re10"]
    print("[%s] argv 取工况：%s" % ("PASS" if good_parse else "FAIL", ids))
    if not good_parse:
        fails.append("argv 解析少了格")
    untouched = 注册NS工况(["C-base", "B-val"])
    clean = (untouched == [] and len(mod._CASE_LIBRARY) == n_before)
    print("[%s] 主线工况名与 bend 族名一律不注册（返回 %s，表大小不变）"
          % ("PASS" if clean else "FAIL", untouched))
    if not clean:
        fails.append("非 NS 名被动了")
    try:
        注册NS工况(["C-train-9_ns_re7"])
        fails.append("基准工况不存在却没拒")
        print("[FAIL] 未知基准 C-train-9 被挂上了表")
    except SystemExit as exc:
        live = "不在收缩族表里" in str(exc)
        print("[%s] NS 名但基准工况不存在 → 被拒（报文：%s）" % ("PASS" if live else "FAIL", str(exc)[:48]))
        if not live:
            fails.append("拒是拒了，但报的不是这条原因")
    try:
        注册NS工况(["C-base_ns_re10"])
        fails.append("缺 dense 件却没拒")
        print("[FAIL] 临时目录之外注册了没有 dense 件的 NS 档： precondition 没执法")
    except SystemExit as exc:
        live = "dense 件不存在" in str(exc)
        print("[%s] 缺 dense 件的 NS 档被拒（报文：%s）" % ("PASS" if live else "FAIL", str(exc)[:60]))
        if not live:
            fails.append("拒是拒了，但报的不是这条原因")
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "C-base_ns_re10").mkdir(parents=True)
        (Path(td) / "C-base_ns_re10" / "field_dense.csv").write_text("x\n1\n", encoding="utf-8")
        got = 注册NS工况(["C-base_ns_re10"], data_root=td)
        c = mod.get_case("C-base_ns_re10")
        b = mod.get_case("C-base")
        ok = got == ["C-base_ns_re10"] and c.beta == b.beta and c.lc_over_w == b.lc_over_w \
            and c.case_id == "C-base_ns_re10" and "NS 档" in c.note
        print("[%s] 挂上之后：beta/lc_over_w 逐字段继承 %s→%s，note 写明是 NS 档"
              % ("PASS" if ok else "FAIL", b.beta, c.beta))
        if not ok:
            fails.append("成功路没走通")
        mod._CASE_LIBRARY.pop("C-base_ns_re10", None)
    print("REGISTRY_SELFTEST " + ("ALL GREEN" if not fails else "FAILED | " + " | ".join(fails)))
    return 1 if fails else 0


# ---------------------------------------------------------------------------- 四道闸
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
    rc_scale = scale_selftest()
    rc_reg = registry_selftest()
    print("NS_SELFTEST " + ("ALL GREEN" if not fails and rc_scale == 0 and rc_reg == 0 else "FAILED")
          + ("" if not fails else " | " + " | ".join(fails)))
    return 1 if (fails or rc_scale or rc_reg) else 0


def main() -> int:
    global REYNOLDS, BASE_SCRIPT, SCALE_VARIANT
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--reynolds", type=float, default=None,
                    help="对流项系数（真值同族 star 单位下即 Re）；不给则 0＝纯 Stokes 复算")
    ap.add_argument("--ns-selftest", action="store_true")
    ap.add_argument("--ns-registry-selftest", action="store_true",
                    help="只跑第四道闸（纯 stdlib，本机可跑）")
    ap.add_argument("--ns-scale-selftest", action="store_true",
                    help="只跑尺度口径的四条控制（纯 stdlib，本机可跑）")
    ap.add_argument("--base-script", choices=("mainline", "strict-sparse"), default="mainline",
                    help="挂哪一版主线：mainline＝稠密三阶段（尺度取自真值场）；"
                         "strict-sparse＝论文表5-5/5-6 那版（尺度取自观测点拟合的标准化器）")
    known, rest = ap.parse_known_args()
    if known.ns_scale_selftest:
        return scale_selftest()
    if known.base_script == "strict-sparse":
        BASE_SCRIPT = "train_velocity_pressure_independent_strict_sparse.py"
        SCALE_VARIANT = "scaler"
    if known.ns_selftest:
        return ns_selftest()
    if known.ns_registry_selftest:
        return registry_selftest()
    REYNOLDS = 0.0 if known.reynolds is None else float(known.reynolds)
    print("[NS-base] base=%s scale=%s Re=%g" % (BASE_SCRIPT, SCALE_VARIANT, REYNOLDS), flush=True)
    bm = the_base()
    registered = 注册NS工况(从argv取工况(rest))
    for cid in registered:
        print(f"[NS-case] {cid} 已按同几何注册进收缩族工况表（运行期，不改 model/src/data/contraction_cases.py）",
              flush=True)
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
