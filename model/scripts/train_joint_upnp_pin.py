#!/usr/bin/env python3
"""臂 A（必做1 基线三件套之一）：**单网络联合 PINN** —— 一个网络同时输出 (u,v,p)，一次训练不拆阶段。

为什么要这一臂：论文的创新点是"双模型分阶段 + PDE 耦合"，但对照表里从来没有"单网络 + 同样的物理项 +
同样的观测表 + 同样的参数量级"这一格。图5-18 用的是纯监督 MLP（没有物理项），格13/14 用的是双模型
（拆了阶段与构造项）。缺了这一臂，"分阶段"与"单网络"就分不开。

与双模型严格版（train_velocity_pressure_independent_strict_sparse.py）的**逐项对齐**：
- 数据切分、观测表来源、输入/输出标准化（含 `--strict-sparse-scalers` 的"只用观测点拟合"口径）全部
  复用该模块的函数（同一个 import，不改它）。
- 物理项完全复用其 `方程耦合损失`：本脚本把单网络包成两个"头代理"传进去，于是求导链、
  尺度因子（velocity_scale=‖速度标准化.std‖、pressure_scale=max(压力标准化.std)）、512 个内部物理点的
  等距取点、硬壁面包络的作用位置**与双模型逐字相同**。这不是近似对齐，是同一段代码。
- 权重默认取双模型**耦合阶段**的那一档（连续性 0.1、动量 10.0、两个监督各 1.0），
  即 `--weights-preset strict-sparse|mainline-dense|no-stage-pde` 与 sweep_lib 的三个档位同名同值。
- 唯一不同的两点，正是被试变量：① 一个网络而不是两个；② 一次训练而不是三阶段。
- 参数量：双模型 = 14→128×3→2 加 14→128×3→1；单网络要同量级须用 5 层 128（默认），
  脚本会打印 `trainable_params` 与比值，`--require-param-ratio` 给定时比值超窗直接失败。

产物结构与双模型评估产物同名同 schema（analyze_sweep.py 不需改）：
`results/pinn/<run>/{config.json,best.ckpt,history.csv,metrics.json,evaluations/metrics_{val,test}_dense.json}`。
`best.ckpt` 的键是 `联合模型参数`（单网络），双模型评估器加载不了本 run —— 评估由本脚本自己做完并落盘。
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent          # = model/


def _make_stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", "") or ""
        if enc.lower() not in ("utf-8", "utf8"):
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding="utf-8", errors="replace")


def load_dep(name: str):
    path = SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(path.stem, module)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ 参数量（纯 python，可本地核）
def 参数数(in_dim: int, out_dim: int, hidden: list[int]) -> int:
    dims = [in_dim, *hidden, out_dim]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


def 规模对照(in_dim: int, joint_hidden: list[int], dual_hidden: list[int]) -> dict:
    joint = 参数数(in_dim, 3, joint_hidden)
    dual = 参数数(in_dim, 2, dual_hidden) + 参数数(in_dim, 1, dual_hidden)
    return {"joint_params": joint, "dual_params": dual, "ratio": joint / dual,
            "joint_hidden": joint_hidden, "dual_hidden": dual_hidden}


# ------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="臂 A：单网络联合 (u,v,p) 一次训练")
    ap.add_argument("--family", default="contraction_2d", choices=["contraction_2d", "bend_2d"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-cases", default="")
    ap.add_argument("--val-cases", default="")
    ap.add_argument("--eval-cases", default="", help="逗号分隔；留空=用 val-cases")
    ap.add_argument("--eval-test-cases", default="", help="逗号分隔；非空则再写一份 test_dense 评估")
    ap.add_argument("--feature-mode", default="geometry")
    ap.add_argument("--drop-features", default="")
    ap.add_argument("--train-velocity-source", default="dense")
    ap.add_argument("--val-velocity-source", default="dense")
    ap.add_argument("--train-pressure-source", default="dense")
    ap.add_argument("--val-pressure-source", default="dense")
    ap.add_argument("--hidden-layers", default="128,128,128,128,128",
                    help="单网络隐层；默认 5×128 是为了把参数量抬到双模型（两个 3×128）的同量级")
    ap.add_argument("--dual-reference-hidden", default="128,128,128",
                    help="仅用于打印参数量比值（与 sweep_lib 的双模型 argv 一致）")
    ap.add_argument("--require-param-ratio", type=float, default=0.0,
                    help=">0 时启用闸门：比值须落在 [1-tol, 1+tol]")
    ap.add_argument("--param-ratio-tol", type=float, default=0.05)
    ap.add_argument("--activation", default="silu", choices=["tanh", "relu", "gelu", "silu"])
    ap.add_argument("--epochs", type=int, default=480, help="480 = 双模型 200+200+80 的总轮数")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--print-every", type=int, default=40)
    ap.add_argument("--device", default="cpu", choices=["cpu"])
    ap.add_argument("--max-retries", type=int, default=1)
    ap.add_argument("--strict-sparse-scalers", action="store_true")
    ap.add_argument("--weights-preset", default="strict-sparse",
                    choices=["strict-sparse", "mainline-dense", "no-stage-pde"])
    # 逐项同名，便于与双模型 argv 直接 diff
    ap.add_argument("--velocity-supervision-weight", type=float, default=1.0)
    ap.add_argument("--pressure-supervision-weight", type=float, default=1.0)
    ap.add_argument("--continuity-weight", type=float, default=0.1)
    ap.add_argument("--momentum-weight", type=float, default=10.0)
    ap.add_argument("--wall-weight", type=float, default=0.0)
    ap.add_argument("--inlet-flux-weight", type=float, default=0.0)
    ap.add_argument("--outlet-pressure-weight", type=float, default=0.0)
    ap.add_argument("--pressure-drop-weight", type=float, default=0.0)
    ap.add_argument("--velocity-wall-mode", default="hard", choices=["hard", "soft"])
    ap.add_argument("--hard-wall-sharpness", type=float, default=12.0)
    ap.add_argument("--max-physics-points", type=int, default=512)
    ap.add_argument("--in-dim", dest="in_dim", type=int, default=0,
                    help="--dry-run 专用的离线逃生口：显式给特征列数即可在无 numpy/torch 的机器上看计划与参数量"
                         "（geometry 去掉 inlet_profile_star = 14，basic = 4）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划与参数量对照，不 import torch")
    return ap


PRESETS = {   # 与 sweep_lib.sh 的 train_dual 三个档位同名同值（改一处必须改两处）
    "strict-sparse":   {"inlet": 0.0, "outlet": 0.0, "drop": 0.0, "continuity": 0.1, "momentum": 10.0},
    "mainline-dense":  {"inlet": 0.5, "outlet": 1e-4, "drop": 1.0, "continuity": 0.1, "momentum": 10.0},
    "no-stage-pde":    {"inlet": 0.5, "outlet": 1e-4, "drop": 1.0, "continuity": 0.1, "momentum": 10.0},
}


def 解析层(text: str) -> list[int]:
    return [int(x) for x in text.replace(",", " ").split() if x.strip()]


def main() -> int:
    args = build_parser().parse_args()
    joint_hidden = 解析层(args.hidden_layers)
    dual_hidden = 解析层(args.dual_reference_hidden)

    if args.dry_run and args.in_dim > 0:
        # 离线逃生口：本机无 numpy/torch 时也能核参数量与计划（实例上不传 --in-dim，走真特征列）
        scale = 规模对照(args.in_dim, joint_hidden, dual_hidden)
        print("[params] in_dim=%d(显式) 单网络=%s → %d 参；双模型参照=%s×2 → %d 参；比值=%.4f"
              % (args.in_dim, joint_hidden, scale["joint_params"], dual_hidden, scale["dual_params"], scale["ratio"]))
        if args.require_param_ratio > 0:
            lo, hi = 1.0 - args.param_ratio_tol, 1.0 + args.param_ratio_tol
            if not (lo <= scale["ratio"] <= hi):
                raise SystemExit(f"[FAIL] 参数量比值 {scale['ratio']:.4f} 不在 [{lo:.2f},{hi:.2f}] ⇒ 对照不对等")
        print("[dry-run] run_dir=%s 权重档=%s epochs=%d lr=%g seed=%d 观测源(train)=%s strict=%s 壁面=%s"
              % (PROJECT_ROOT / "results" / "pinn" / args.run_name, args.weights_preset, args.epochs,
                 args.lr, args.seed, args.train_velocity_source, args.strict_sparse_scalers, args.velocity_wall_mode))
        print("[dry-run] 物理项=复用双模型 方程耦合损失（头代理切片）；求导链/尺度因子/512 点取点规则逐字相同")
        print(json.dumps({"param_scale": scale, "in_dim_forced": args.in_dim}, ensure_ascii=False))
        return 0

    sup = load_dep("train_supervised.py")
    dual = load_dep("train_velocity_pressure_independent_strict_sparse.py")
    spec = sup.get_family_spec(args.family)
    基础特征列 = sup.resolve_feature_cols(spec, args.feature_mode.strip() or spec.default_feature_mode)
    特征列 = sup.apply_feature_drop(基础特征列, sup.parse_csv_list(args.drop_features))

    if args.weights_preset != "strict-sparse" and args.strict_sparse_scalers:
        raise SystemExit("[FAIL] --strict-sparse-scalers 只配 strict-sparse 档（稠密主线不吃观测点标准化）")
    preset = PRESETS[args.weights_preset]
    for key, flag, cli_val in (("inlet", "--inlet-flux-weight", args.inlet_flux_weight),
                                ("outlet", "--outlet-pressure-weight", args.outlet_pressure_weight),
                                ("drop", "--pressure-drop-weight", args.pressure_drop_weight),
                                ("continuity", "--continuity-weight", args.continuity_weight),
                                ("momentum", "--momentum-weight", args.momentum_weight)):
        if abs(cli_val - preset[key]) > 1e-12:
            print("[warn] %s=%g 与档位 %s 的 %g 不同，按命令行显式值执行" % (flag, cli_val, args.weights_preset, preset[key]))
    weights = {"inlet": args.inlet_flux_weight, "outlet": args.outlet_pressure_weight,
               "drop": args.pressure_drop_weight, "continuity": args.continuity_weight,
               "momentum": args.momentum_weight}

    scale = 规模对照(len(特征列), joint_hidden, dual_hidden)
    print("[params] in_dim=%d 单网络=%s → %d 参；双模型参照=%s×2 → %d 参；比值=%.4f"
          % (len(特征列), joint_hidden, scale["joint_params"], dual_hidden, scale["dual_params"], scale["ratio"]))
    if args.require_param_ratio > 0:
        lo, hi = 1.0 - args.param_ratio_tol, 1.0 + args.param_ratio_tol
        if not (lo <= scale["ratio"] <= hi):
            raise SystemExit(f"[FAIL] 参数量比值 {scale['ratio']:.4f} 不在 [{lo:.2f},{hi:.2f}] ⇒ 对照不对等，"
                             f"先调 --hidden-layers 再跑")
    out_dir = PROJECT_ROOT / "results" / "pinn" / args.run_name

    if args.dry_run:
        print("[dry-run] run_dir=%s" % out_dir)
        print("[dry-run] 权重档=%s 权重=%s epochs=%d lr=%g seed=%d 观测源(train)=%s strict=%s 壁面=%s"
              % (args.weights_preset, weights, args.epochs, args.lr, args.seed,
                 args.train_velocity_source, args.strict_sparse_scalers, args.velocity_wall_mode))
        print("[dry-run] 物理项=复用双模型 方程耦合损失（头代理切片）；求导链/尺度因子/取点规则逐字相同")
        print(json.dumps({"param_scale": scale, "feature_cols": 特征列}, ensure_ascii=False))
        return 0

    import torch    # 真跑才要（本地无 torch，靠 --dry-run 与闸门自测）

    dual.设置随机种子(args.seed)
    import numpy as np
    train_cases = sup.parse_csv_list(args.train_cases) if args.train_cases else list(spec.default_train_cases)
    val_cases = sup.parse_csv_list(args.val_cases) if args.val_cases else list(spec.default_val_cases)
    eval_cases = sup.parse_csv_list(args.eval_cases) if args.eval_cases else val_cases
    test_cases = sup.parse_csv_list(args.eval_test_cases)

    构建 = dual.构建数据切分
    if args.strict_sparse_scalers:
        vel_train_split, 输入标准化器 = 构建(spec, train_cases, 特征列, args.train_velocity_source)
        p_train_split, _ = 构建(spec, train_cases, 特征列, args.train_pressure_source, input_scaler=输入标准化器)
        dense_train_split, _ = 构建(spec, train_cases, 特征列, "dense", input_scaler=输入标准化器)
        速度标准化器 = dual.输出标准化器.fit(vel_train_split.targets_raw[:, :2])
        压力标准化器 = dual.输出标准化器.fit(p_train_split.targets_raw[:, 2:3])
    else:
        dense_train_split, 输入标准化器 = 构建(spec, train_cases, 特征列, "dense")
        vel_train_split, _ = 构建(spec, train_cases, 特征列, args.train_velocity_source, input_scaler=输入标准化器)
        p_train_split, _ = 构建(spec, train_cases, 特征列, args.train_pressure_source, input_scaler=输入标准化器)
        速度标准化器 = dual.输出标准化器.fit(dense_train_split.targets_raw[:, :2])
        压力标准化器 = dual.输出标准化器.fit(dense_train_split.targets_raw[:, 2:3])
    dense_val_split, _ = 构建(spec, val_cases, 特征列, "dense", input_scaler=输入标准化器)

    device = torch.device(args.device)
    网络 = dual.多层感知机(len(特征列), 3, joint_hidden, args.activation).to(device)
    n_params = sum(p.numel() for p in 网络.parameters() if p.requires_grad)
    if abs(n_params - scale["joint_params"]) > 1:
        print("[warn] 实际可训练参数 %d 与解析式估计 %d 不符 ⇒ 结构变了，核对 多层感知机" % (n_params, scale["joint_params"]))
    scale["actual_joint_params"] = n_params

    class 速度头(torch.nn.Module):
        """把单网络的前两列当成'速度模型'传给双模型的物理函数 —— 同一段物理代码，不是复刻。"""

        def forward(self, x):           # type: ignore[override]
            return 网络(x)[:, 0:2]

    class 压力头(torch.nn.Module):
        def forward(self, x):           # type: ignore[override]
            return 网络(x)[:, 2:3]

    vhead, phead = 速度头(), 压力头()
    约束信息 = dual.构建壁面硬约束信息(特征列, args.velocity_wall_mode, args.hard_wall_sharpness)

    def 联合前向(split):
        x = dual.张量化特征(split, device)
        wd = dual.提取壁面距离分数(split.features_raw, device, 约束信息)
        v_norm, v_raw = dual.速度前向原值(vhead, x, 速度标准化器, wall_distance_frac=wd, constraint_info=约束信息)
        p_norm, p_raw = dual.压力前向原值(phead, x, 压力标准化器)
        return v_norm, v_raw, p_norm, p_raw

    优化器 = torch.optim.Adam(网络.parameters(), lr=args.lr)
    vel_x, vel_y = dual.张量化特征(vel_train_split, device), dual.速度真值张量(vel_train_split, device)
    p_x, p_y = dual.张量化特征(p_train_split, device), dual.压力真值张量(p_train_split, device)
    出 = out_dir
    出.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | str]] = []
    best = {"loss": math.inf, "epoch": 0}
    坏 = 0
    for epoch in range(1, args.epochs + 1):
        网络.train()
        v_norm, v_raw, p_norm, p_raw = 联合前向(vel_train_split)
        l_vel = dual.速度监督损失(v_norm, vel_y, 速度标准化器)[0]
        l_pre = dual.压力监督损失(p_norm, p_y, 压力标准化器)
        物理 = dual.方程耦合损失(vhead, phead, dense_train_split, 输入标准化器, 速度标准化器,
                                  压力标准化器, device, args.max_physics_points, 约束信息)
        l_wall = dual.壁面无滑移损失(v_raw, dense_train_split)
        l_in = dual.入口流量损失(v_raw, dense_train_split, train_cases, device)
        l_out = dual.出口压力损失(p_raw, dense_train_split)
        l_drop = dual.压降损失(p_raw, dense_train_split, train_cases, device)
        total = (args.velocity_supervision_weight * l_vel + args.pressure_supervision_weight * l_pre
                 + weights["continuity"] * 物理["连续性"] + weights["momentum"] * 物理["动量"]
                 + args.wall_weight * l_wall + weights["inlet"] * l_in
                 + weights["outlet"] * l_out + weights["drop"] * l_drop)
        优化器.zero_grad(set_to_none=True)
        total.backward()
        优化器.step()
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            _, 验证指标 = dual.评估联合场(vhead, phead, dense_val_split, 速度标准化器, 压力标准化器,
                                          device, 约束信息)
            row = {"stage": "joint", "epoch": epoch, "total_loss": float(total.detach()),
                   "vel_sup": float(l_vel.detach()), "pre_sup": float(l_pre.detach()),
                   "continuity": float(物理["连续性"].detach()), "momentum": float(物理["动量"].detach()),
                   "val_rel_l2_speed": float(验证指标["rel_l2_speed"]), "val_rel_l2_p": float(验证指标["rel_l2_p"])}
            history.append(row)
            print("[joint] ep=%d loss=%.4e speed=%.4f p=%.4f mom=%.3e"
                  % (epoch, row["total_loss"], row["val_rel_l2_speed"], row["val_rel_l2_p"], row["momentum"]))
            if row["total_loss"] < best["loss"]:
                best = {"loss": row["total_loss"], "epoch": epoch}
                坏 = 0
                torch.save({"联合模型参数": 网络.state_dict(), "epoch": epoch}, 出 / "best.ckpt")
            else:
                坏 += 1
                if 坏 * args.print_every >= args.patience:
                    print("[early-stop] 连续 %d 轮无改善 ⇒ 停在 ep=%d" % (坏 * args.print_every, epoch))
                    break
    if not (出 / "best.ckpt").exists():
        raise SystemExit("[FAIL] 没有任何一轮写出 best.ckpt ⇒ 训练没跑起来，不能交空产物")

    配置 = {
        "臂": "A_单网络联合PINN", "family": args.family, "run_name": args.run_name, "seed": args.seed,
        "feature_mode": args.feature_mode, "drop_features": args.drop_features,
        "feature_cols": 特征列, "hidden_layers": joint_hidden, "activation": args.activation,
        "epochs": args.epochs, "lr": args.lr, "weights_preset": args.weights_preset, "weights": weights,
        "velocity_supervision_weight": args.velocity_supervision_weight,
        "pressure_supervision_weight": args.pressure_supervision_weight,
        "wall_weight": args.wall_weight, "velocity_wall_mode": args.velocity_wall_mode,
        "hard_wall_sharpness": args.hard_wall_sharpness, "max_physics_points": args.max_physics_points,
        "strict_sparse_scalers": bool(args.strict_sparse_scalers),
        "train_velocity_source": args.train_velocity_source, "train_pressure_source": args.train_pressure_source,
        "val_velocity_source": args.val_velocity_source, "val_pressure_source": args.val_pressure_source,
        "模型结构": "joint_single_stage(单网络 out=3，一次训练不拆阶段)",
        "物理项来源": "复用 train_velocity_pressure_independent_strict_sparse.方程耦合损失（头代理切片）",
        "输入标准化": {"x_mean": list(map(float, 输入标准化器.mean)), "x_std": list(map(float, 输入标准化器.std))},
        "速度标准化": {"mean": list(map(float, 速度标准化器.mean)), "std": list(map(float, 速度标准化器.std))},
        "压力标准化": {"mean": list(map(float, 压力标准化器.mean)), "std": list(map(float, 压力标准化器.std))},
        "param_scale": scale,
    }
    (出 / "config.json").write_text(json.dumps(配置, ensure_ascii=False, indent=2), encoding="utf-8")
    with (出 / "history.csv").open("w", encoding="utf-8", newline="") as fh:
        if history:
            writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    评估器 = load_dep("evaluate_velocity_pressure_independent.py")

    def 落评估(cases, split_name):
        split, _ = 构建(spec, cases, 特征列, "dense", input_scaler=输入标准化器)
        pred, metrics = dual.评估联合场(vhead, phead, split, 速度标准化器, 压力标准化器, device, 约束信息)
        case_metrics = [dataclasses.asdict(sup.compute_case_metrics(cid, split, pred))
                        for cid in sorted(set(split.case_ids.tolist()))]
        payload = {"family": args.family, "run_name": args.run_name, "split_name": split_name,
                   "eval_cases": cases, "eval_source": "dense", "feature_mode": args.feature_mode,
                   "feature_cols": 特征列, "臂": "A_单网络联合PINN",
                   "summary": 评估器.汇总(case_metrics), "global_metrics": metrics,
                   "case_metrics": case_metrics, "arm_note": "单网络联合 PINN，一次训练；物理项与双模型同源"}
        (出 / "evaluations").mkdir(exist_ok=True)
        (出 / "evaluations" / ("metrics_%s.json" % split_name)).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return metrics

    val_metrics = 落评估(eval_cases, "val_dense")
    test_metrics = 落评估(test_cases, "test_dense") if test_cases else {}
    (出 / "metrics.json").write_text(json.dumps(
        {"run_name": args.run_name, "臂": "A_单网络联合PINN", "seed": args.seed,
         "best_epoch": best["epoch"], "final_val": val_metrics, "test": test_metrics,
         "param_scale": scale}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] %s val speed=%.4f p=%.4f (ep=%d)"
          % (args.run_name, val_metrics["rel_l2_speed"], val_metrics["rel_l2_p"], best["epoch"]))
    return 0


if __name__ == "__main__":
    _make_stdout_utf8()
    raise SystemExit(main())
