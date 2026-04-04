#!/usr/bin/env python3
"""独立速度模型 + 独立压力模型 + 控制方程交替耦合 试探版训练入口。

设计原则：
- 保留参数化几何感知输入；
- 允许去掉固定流场模板特征，例如 inlet_profile_star；
- 速度模型和压力模型完全独立预训练；
- 在耦合阶段使用控制方程做低学习率交替微调；
- 默认仅在中央处理器上运行，且支持限制方程点数，避免高占用。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import train_supervised as sup


速度目标列 = ("u_star", "v_star")
压力目标列 = ("p_star",)


class 多层感知机(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: Sequence[int], activation: str = "silu"):
        super().__init__()
        激活映射 = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        激活类 = 激活映射.get(activation.lower(), nn.SiLU)
        dims = [in_dim, *hidden_layers, out_dim]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(激活类())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class 输出标准化器:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "输出标准化器":
        mean = np.mean(values, axis=0).astype(np.float32)
        std = np.std(values, axis=0).astype(np.float32)
        std = np.where(std < 1.0e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


def 解析层(text: str) -> list[int]:
    layers = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not layers:
        raise ValueError("网络层配置不能为空")
    return layers


def 设置随机种子(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def 一阶导(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def 二阶导(first: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        first,
        inputs,
        grad_outputs=torch.ones_like(first),
        create_graph=True,
        retain_graph=True,
    )[0]


def 设置模块可训练(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def 构建数据切分(
    spec: sup.FamilySpec,
    case_ids: Sequence[str],
    feature_cols: Sequence[str],
    source: str,
    input_scaler: sup.StandardScaler | None = None,
) -> tuple[sup.PreparedSplit, sup.StandardScaler]:
    split, scaler, _ = sup.build_split(spec, case_ids, feature_cols=feature_cols, source=source, input_scaler=input_scaler, output_scaler=None)
    return split, scaler


def 张量化特征(split: sup.PreparedSplit, device: torch.device) -> torch.Tensor:
    return torch.tensor(split.features_norm, dtype=torch.float32, device=device)


def 速度真值张量(split: sup.PreparedSplit, device: torch.device) -> torch.Tensor:
    return torch.tensor(split.targets_raw[:, :2], dtype=torch.float32, device=device)


def 压力真值张量(split: sup.PreparedSplit, device: torch.device) -> torch.Tensor:
    return torch.tensor(split.targets_raw[:, 2:3], dtype=torch.float32, device=device)


def 构建壁面硬约束信息(
    feature_cols: Sequence[str],
    mode: str,
    strength: float,
) -> dict[str, float | int | str]:
    mode_norm = mode.strip().lower()
    if mode_norm not in {"soft", "hard"}:
        raise ValueError(f"未知壁面约束模式: {mode}")
    info: dict[str, float | int | str] = {
        "mode": mode_norm,
        "strength": float(max(strength, 1.0e-6)),
        "wall_distance_index": -1,
    }
    if mode_norm == "hard":
        if "wall_distance_frac" not in feature_cols:
            raise ValueError("启用硬约束时，特征中必须包含 wall_distance_frac")
        info["wall_distance_index"] = int(feature_cols.index("wall_distance_frac"))
    return info


def 提取壁面距离分数(
    x_raw: np.ndarray | None,
    device: torch.device,
    constraint_info: dict[str, float | int | str],
) -> torch.Tensor | None:
    if str(constraint_info["mode"]) != "hard":
        return None
    if x_raw is None:
        raise ValueError("硬约束模式需要原始特征以提取 wall_distance_frac")
    idx = int(constraint_info["wall_distance_index"])
    if idx < 0:
        raise ValueError("硬约束模式缺少 wall_distance_frac 索引")
    values = np.asarray(x_raw[:, idx], dtype=np.float32)
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(1)


def 壁面速度硬约束包络(
    wall_distance_frac: torch.Tensor | None,
    constraint_info: dict[str, float | int | str],
) -> torch.Tensor | None:
    if str(constraint_info["mode"]) != "hard":
        return None
    if wall_distance_frac is None:
        raise ValueError("硬约束模式缺少 wall_distance_frac")
    d = torch.clamp(wall_distance_frac, min=0.0, max=1.0)
    sharpness = float(constraint_info["strength"])
    scale = 1.0 - math.exp(-sharpness)
    scale_t = torch.tensor(scale if scale > 1.0e-8 else 1.0, dtype=torch.float32, device=d.device)
    return (1.0 - torch.exp(-sharpness * d)) / scale_t


def 速度前向原值(
    model: 多层感知机,
    x_norm: torch.Tensor,
    标准化器: 输出标准化器,
    wall_distance_frac: torch.Tensor | None = None,
    constraint_info: dict[str, float | int | str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(标准化器.mean, dtype=torch.float32, device=x_norm.device)
    std = torch.tensor(标准化器.std, dtype=torch.float32, device=x_norm.device)
    pred_free_norm = model(x_norm)
    pred_raw = pred_free_norm * std + mean
    if constraint_info is not None and str(constraint_info["mode"]) == "hard":
        envelope = 壁面速度硬约束包络(wall_distance_frac, constraint_info)
        if envelope is None:
            raise ValueError("硬约束模式未能构造速度包络")
        pred_raw = pred_raw * envelope
    pred_norm = (pred_raw - mean) / std
    return pred_norm, pred_raw


def 压力前向原值(
    model: 多层感知机,
    x_norm: torch.Tensor,
    标准化器: 输出标准化器,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_norm = model(x_norm)
    mean = torch.tensor(标准化器.mean, dtype=torch.float32, device=x_norm.device)
    std = torch.tensor(标准化器.std, dtype=torch.float32, device=x_norm.device)
    pred_raw = pred_norm * std + mean
    return pred_norm, pred_raw


def 速度监督损失(pred_norm: torch.Tensor, truth_raw: torch.Tensor, 标准化器: 输出标准化器) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.tensor(标准化器.mean, dtype=torch.float32, device=pred_norm.device)
    std = torch.tensor(标准化器.std, dtype=torch.float32, device=pred_norm.device)
    truth_norm = (truth_raw - mean) / std
    valid_u = torch.isfinite(truth_norm[:, 0])
    valid_v = torch.isfinite(truth_norm[:, 1])
    zero = pred_norm.new_tensor(0.0)
    loss_u = torch.mean((pred_norm[valid_u, 0] - truth_norm[valid_u, 0]) ** 2) if torch.any(valid_u) else zero
    loss_v = torch.mean((pred_norm[valid_v, 1] - truth_norm[valid_v, 1]) ** 2) if torch.any(valid_v) else zero
    return loss_u + loss_v, loss_u, loss_v


def 压力监督损失(pred_norm: torch.Tensor, truth_raw: torch.Tensor, 标准化器: 输出标准化器) -> torch.Tensor:
    mean = torch.tensor(标准化器.mean, dtype=torch.float32, device=pred_norm.device)
    std = torch.tensor(标准化器.std, dtype=torch.float32, device=pred_norm.device)
    truth_norm = (truth_raw - mean) / std
    valid_p = torch.isfinite(truth_norm[:, 0])
    if not torch.any(valid_p):
        return pred_norm.new_tensor(0.0)
    return torch.mean((pred_norm[valid_p, 0] - truth_norm[valid_p, 0]) ** 2)


def 壁面无滑移损失(速度原值: torch.Tensor, dense_split: sup.PreparedSplit) -> torch.Tensor:
    device = 速度原值.device
    wall_mask = torch.tensor(dense_split.boundary_type == "wall", dtype=torch.bool, device=device)
    if not torch.any(wall_mask):
        return 速度原值.new_tensor(0.0)
    return torch.mean(速度原值[wall_mask, 0] ** 2 + 速度原值[wall_mask, 1] ** 2)


def 入口流量损失(速度原值: torch.Tensor, dense_split: sup.PreparedSplit, case_ids: Sequence[str], device: torch.device) -> torch.Tensor:
    inlet_mask_np = dense_split.boundary_type == "inlet"
    if np.count_nonzero(inlet_mask_np) < 2:
        return 速度原值.new_tensor(0.0)
    losses: list[torch.Tensor] = []
    y_star = dense_split.y_star
    真值速度 = dense_split.targets_raw[:, 0]
    case_id_arr = dense_split.case_ids
    for case_id in case_ids:
        mask_np = inlet_mask_np & (case_id_arr == case_id)
        if np.count_nonzero(mask_np) < 2:
            continue
        coord = torch.tensor(y_star[mask_np], dtype=torch.float32, device=device)
        order = torch.argsort(coord)
        coord_sorted = coord[order]
        pred_u_sorted = 速度原值[torch.tensor(mask_np, dtype=torch.bool, device=device), 0][order]
        true_u_sorted = torch.tensor(真值速度[mask_np], dtype=torch.float32, device=device)[order]
        pred_flux = torch.trapezoid(pred_u_sorted, coord_sorted)
        true_flux = torch.trapezoid(true_u_sorted, coord_sorted)
        scale = torch.clamp(torch.abs(true_flux), min=1.0e-12)
        losses.append(((pred_flux - true_flux) / scale) ** 2)
    return torch.mean(torch.stack(losses)) if losses else 速度原值.new_tensor(0.0)


def 出口压力损失(压力原值: torch.Tensor, dense_split: sup.PreparedSplit) -> torch.Tensor:
    device = 压力原值.device
    outlet_mask = torch.tensor(dense_split.boundary_type == "outlet", dtype=torch.bool, device=device)
    truth_p = torch.tensor(dense_split.targets_raw[:, 2], dtype=torch.float32, device=device)
    if not torch.any(outlet_mask):
        return 压力原值.new_tensor(0.0)
    return torch.mean((压力原值[outlet_mask, 0] - truth_p[outlet_mask]) ** 2)


def 压降损失(压力原值: torch.Tensor, dense_split: sup.PreparedSplit, case_ids: Sequence[str], device: torch.device) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    case_id_arr = dense_split.case_ids
    边界类型 = dense_split.boundary_type
    真值压力 = dense_split.targets_raw[:, 2]
    for case_id in case_ids:
        inlet_mask_np = (case_id_arr == case_id) & (边界类型 == "inlet")
        outlet_mask_np = (case_id_arr == case_id) & (边界类型 == "outlet")
        if np.count_nonzero(inlet_mask_np) == 0 or np.count_nonzero(outlet_mask_np) == 0:
            continue
        inlet_mask = torch.tensor(inlet_mask_np, dtype=torch.bool, device=device)
        outlet_mask = torch.tensor(outlet_mask_np, dtype=torch.bool, device=device)
        pred_drop = torch.mean(压力原值[inlet_mask, 0]) - torch.mean(压力原值[outlet_mask, 0])
        true_drop = float(np.mean(真值压力[inlet_mask_np]) - np.mean(真值压力[outlet_mask_np]))
        true_drop_t = torch.tensor(true_drop, dtype=torch.float32, device=device)
        scale = torch.clamp(torch.abs(true_drop_t), min=1.0e-12)
        losses.append(((pred_drop - true_drop_t) / scale) ** 2)
    return torch.mean(torch.stack(losses)) if losses else 压力原值.new_tensor(0.0)


def 方程耦合损失(
    速度模型: 多层感知机,
    压力模型: 多层感知机,
    dense_split: sup.PreparedSplit,
    输入标准化器: sup.StandardScaler,
    速度标准化器: 输出标准化器,
    压力标准化器: 输出标准化器,
    device: torch.device,
    max_physics_points: int,
    velocity_constraint_info: dict[str, float | int | str],
) -> dict[str, torch.Tensor]:
    interior_idx = np.where(dense_split.boundary_type == "interior")[0]
    if interior_idx.size == 0:
        zero = torch.tensor(0.0, dtype=torch.float32, device=device)
        return {
            "连续性": zero,
            "动量": zero,
            "平均散度绝对值": zero,
            "最大散度绝对值": zero,
        }
    if max_physics_points > 0 and interior_idx.size > max_physics_points:
        sample_pos = np.linspace(0, interior_idx.size - 1, num=max_physics_points, dtype=np.int64)
        interior_idx = interior_idx[sample_pos]
    x = torch.tensor(dense_split.features_norm[interior_idx], dtype=torch.float32, device=device, requires_grad=True)
    wall_distance_frac = 提取壁面距离分数(dense_split.features_raw[interior_idx], device, velocity_constraint_info)
    _, 速度原值 = 速度前向原值(速度模型, x, 速度标准化器, wall_distance_frac=wall_distance_frac, constraint_info=velocity_constraint_info)
    _, 压力原值 = 压力前向原值(压力模型, x, 压力标准化器)
    u = 速度原值[:, 0:1]
    v = 速度原值[:, 1:2]
    p = 压力原值[:, 0:1]

    x_std = torch.tensor(float(输入标准化器.std[0]), dtype=torch.float32, device=device)
    y_std = torch.tensor(float(输入标准化器.std[1]), dtype=torch.float32, device=device)
    x_std = torch.clamp(x_std, min=1.0e-6)
    y_std = torch.clamp(y_std, min=1.0e-6)

    grad_u = 一阶导(u, x)
    grad_v = 一阶导(v, x)
    grad_p = 一阶导(p, x)
    u_x = grad_u[:, 0:1] / x_std
    u_y = grad_u[:, 1:2] / y_std
    v_x = grad_v[:, 0:1] / x_std
    v_y = grad_v[:, 1:2] / y_std
    p_x = grad_p[:, 0:1] / x_std
    p_y = grad_p[:, 1:2] / y_std

    grad_u_x = 二阶导(u_x, x)
    grad_u_y = 二阶导(u_y, x)
    grad_v_x = 二阶导(v_x, x)
    grad_v_y = 二阶导(v_y, x)
    u_xx = grad_u_x[:, 0:1] / x_std
    u_yy = grad_u_y[:, 1:2] / y_std
    v_xx = grad_v_x[:, 0:1] / x_std
    v_yy = grad_v_y[:, 1:2] / y_std

    velocity_scale = torch.tensor(
        max(float(np.nanmax(np.sqrt(dense_split.targets_raw[:, 0] ** 2 + dense_split.targets_raw[:, 1] ** 2))), 1.0e-12),
        dtype=torch.float32,
        device=device,
    )
    pressure_vals = dense_split.targets_raw[:, 2]
    pressure_scale = torch.tensor(
        max(float(np.nanmax(pressure_vals) - np.nanmin(pressure_vals)), 1.0e-12),
        dtype=torch.float32,
        device=device,
    )

    continuity = u_x / velocity_scale + v_y / velocity_scale
    momentum_u = u_xx / velocity_scale + u_yy / velocity_scale - p_x / pressure_scale
    momentum_v = v_xx / velocity_scale + v_yy / velocity_scale - p_y / pressure_scale
    return {
        "连续性": torch.mean(continuity ** 2),
        "动量": torch.mean(momentum_u ** 2) + torch.mean(momentum_v ** 2),
        "平均散度绝对值": torch.mean(torch.abs(continuity)),
        "最大散度绝对值": torch.max(torch.abs(continuity)),
    }


def 评估联合场(
    速度模型: 多层感知机,
    压力模型: 多层感知机,
    split: sup.PreparedSplit,
    速度标准化器: 输出标准化器,
    压力标准化器: 输出标准化器,
    device: torch.device,
    velocity_constraint_info: dict[str, float | int | str],
) -> tuple[np.ndarray, dict[str, float]]:
    速度模型.eval()
    压力模型.eval()
    with torch.no_grad():
        x = 张量化特征(split, device)
        wall_distance_frac = 提取壁面距离分数(split.features_raw, device, velocity_constraint_info)
        _, vel_raw = 速度前向原值(速度模型, x, 速度标准化器, wall_distance_frac=wall_distance_frac, constraint_info=velocity_constraint_info)
        _, p_raw = 压力前向原值(压力模型, x, 压力标准化器)
    vel_np = vel_raw.cpu().numpy()
    p_np = p_raw.cpu().numpy()
    pred_np = np.concatenate([vel_np, p_np], axis=1)
    metrics = {
        "rel_l2_u": sup.relative_l2(pred_np[:, 0], split.targets_raw[:, 0]),
        "rel_l2_v": sup.relative_l2(pred_np[:, 1], split.targets_raw[:, 1]),
        "rel_l2_p": sup.relative_l2(pred_np[:, 2], split.targets_raw[:, 2]),
        "rel_l2_speed": sup.relative_l2(
            np.sqrt(pred_np[:, 0] ** 2 + pred_np[:, 1] ** 2),
            np.sqrt(split.targets_raw[:, 0] ** 2 + split.targets_raw[:, 1] ** 2),
        ),
    }
    metrics.update(sup.compute_extreme_error_metrics(pred_np, split))
    return pred_np, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="独立速度模型 + 独立压力模型 + 控制方程交替耦合 试探版")
    parser.add_argument("--family", default="bend_2d", choices=sorted(sup.DEFAULTS))
    parser.add_argument("--train-cases", default="", help="逗号分隔训练工况；默认使用 family 预设")
    parser.add_argument("--val-cases", default="", help="逗号分隔验证工况；默认使用 family 预设")
    parser.add_argument("--run-name", default="", help="输出目录名")
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--feature-mode", default="geometry")
    parser.add_argument("--drop-features", default="")
    parser.add_argument("--train-velocity-source", default="dense")
    parser.add_argument("--val-velocity-source", default="dense")
    parser.add_argument("--train-pressure-source", default="dense")
    parser.add_argument("--val-pressure-source", default="dense")
    parser.add_argument("--velocity-hidden-layers", default="128,128,128")
    parser.add_argument("--pressure-hidden-layers", default="128,128,128")
    parser.add_argument("--activation", default="silu", choices=["tanh", "relu", "gelu", "silu"])
    parser.add_argument("--velocity-epochs", type=int, default=160)
    parser.add_argument("--pressure-epochs", type=int, default=160)
    parser.add_argument("--coupling-epochs", type=int, default=80)
    parser.add_argument("--velocity-lr", type=float, default=8e-4)
    parser.add_argument("--pressure-lr", type=float, default=8e-4)
    parser.add_argument("--coupling-velocity-lr", type=float, default=2e-4)
    parser.add_argument("--coupling-pressure-lr", type=float, default=2e-4)
    parser.add_argument("--wall-weight", type=float, default=1.0)
    parser.add_argument("--inlet-flux-weight", type=float, default=0.5)
    parser.add_argument("--continuity-weight", type=float, default=0.1)
    parser.add_argument(
        "--velocity-stage-continuity-weight",
        type=float,
        default=-1.0,
        help="速度阶段连续性权重；小于 0 时回退到 continuity-weight",
    )
    parser.add_argument(
        "--velocity-stage-momentum-weight",
        type=float,
        default=0.0,
        help="速度阶段动量权重；默认关闭，避免未训练压力模型过早干扰速度学习",
    )
    parser.add_argument("--outlet-pressure-weight", type=float, default=1e-4)
    parser.add_argument("--pressure-drop-weight", type=float, default=1.0)
    parser.add_argument(
        "--pressure-stage-momentum-weight",
        type=float,
        default=0.0,
        help="压力阶段动量权重；依赖已冻结的速度模型，把压力模型往控制方程一致方向拉回",
    )
    parser.add_argument("--velocity-wall-mode", default="soft", choices=["soft", "hard"])
    parser.add_argument("--hard-wall-sharpness", type=float, default=12.0, help="硬约束指数饱和强度，仅在 velocity-wall-mode=hard 时生效")
    parser.add_argument("--coupling-momentum-weight", type=float, default=10.0)
    parser.add_argument("--coupling-continuity-weight", type=float, default=0.1)
    parser.add_argument("--coupling-velocity-supervision-weight", type=float, default=1.0)
    parser.add_argument("--coupling-pressure-supervision-weight", type=float, default=1.0)
    parser.add_argument("--max-physics-points", type=int, default=2048, help="每轮控制方程最多使用的内部点数；0 表示全量")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=1)
    return parser


def train_once(args: argparse.Namespace) -> dict:
    设置随机种子(args.seed)
    spec = sup.get_family_spec(args.family)
    train_cases = sup.parse_csv_list(args.train_cases) if args.train_cases else list(spec.default_train_cases)
    val_cases = sup.parse_csv_list(args.val_cases) if args.val_cases else list(spec.default_val_cases)
    基础特征列 = sup.resolve_feature_cols(spec, args.feature_mode.strip() if args.feature_mode else spec.default_feature_mode)
    删除特征 = sup.parse_csv_list(args.drop_features)
    特征列 = sup.apply_feature_drop(基础特征列, 删除特征)
    velocity_constraint_info = 构建壁面硬约束信息(特征列, args.velocity_wall_mode, args.hard_wall_sharpness)
    输出目录 = PROJECT_ROOT / "results" / "pinn" / (args.run_name or f"{spec.data_subdir.replace('_2d', '')}_independent_coupling")
    输出目录.mkdir(parents=True, exist_ok=True)

    dense_train_split, 输入标准化器 = 构建数据切分(spec, train_cases, 特征列, "dense")
    dense_val_split, _ = 构建数据切分(spec, val_cases, 特征列, "dense", input_scaler=输入标准化器)
    vel_train_split, _ = 构建数据切分(spec, train_cases, 特征列, args.train_velocity_source, input_scaler=输入标准化器)
    vel_val_split, _ = 构建数据切分(spec, val_cases, 特征列, args.val_velocity_source, input_scaler=输入标准化器)
    p_train_split, _ = 构建数据切分(spec, train_cases, 特征列, args.train_pressure_source, input_scaler=输入标准化器)
    p_val_split, _ = 构建数据切分(spec, val_cases, 特征列, args.val_pressure_source, input_scaler=输入标准化器)

    速度标准化器 = 输出标准化器.fit(dense_train_split.targets_raw[:, :2])
    压力标准化器 = 输出标准化器.fit(dense_train_split.targets_raw[:, 2:3])

    device = torch.device(args.device)
    速度模型 = 多层感知机(len(特征列), 2, 解析层(args.velocity_hidden_layers), args.activation).to(device)
    压力模型 = 多层感知机(len(特征列), 1, 解析层(args.pressure_hidden_layers), args.activation).to(device)

    vel_train_x = 张量化特征(vel_train_split, device)
    vel_train_y = 速度真值张量(vel_train_split, device)
    vel_train_wall_distance = 提取壁面距离分数(vel_train_split.features_raw, device, velocity_constraint_info)
    p_train_x = 张量化特征(p_train_split, device)
    p_train_y = 压力真值张量(p_train_split, device)
    dense_train_x = 张量化特征(dense_train_split, device)
    dense_train_wall_distance = 提取壁面距离分数(dense_train_split.features_raw, device, velocity_constraint_info)

    history: list[dict[str, float | str]] = []
    速度阶段连续性权重 = args.velocity_stage_continuity_weight if args.velocity_stage_continuity_weight >= 0.0 else args.continuity_weight
    速度阶段动量权重 = max(0.0, float(args.velocity_stage_momentum_weight))
    压力阶段动量权重 = max(0.0, float(args.pressure_stage_momentum_weight))

    def 记录(stage: str, epoch: int, total_loss: float, extra: dict[str, float]) -> None:
        _, 验证指标 = 评估联合场(
            速度模型,
            压力模型,
            dense_val_split,
            速度标准化器,
            压力标准化器,
            device,
            velocity_constraint_info,
        )
        row: dict[str, float | str] = {"阶段": stage, "轮次": epoch, "总损失": float(total_loss)}
        row.update({f"验证_{k}": float(v) for k, v in 验证指标.items()})
        row.update(extra)
        history.append(row)
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"[{stage} 第{epoch}轮] "
                f"总损失={total_loss:.4e} "
                f"验证速度相对二范数={验证指标['rel_l2_speed']:.4e} "
                f"验证压力相对二范数={验证指标['rel_l2_p']:.4e} "
                f"验证最大速度误差={验证指标['max_speed_err_over_speed_max']:.4e} "
                f"验证最大压力误差={验证指标['max_p_err_over_p_range']:.4e}"
            )

    # 第一阶段：独立训练速度模型
    设置模块可训练(速度模型, True)
    设置模块可训练(压力模型, False)
    速度优化器 = torch.optim.Adam(速度模型.parameters(), lr=args.velocity_lr)
    for epoch in range(1, args.velocity_epochs + 1):
        速度模型.train()
        速度优化器.zero_grad()
        vel_pred_norm, vel_pred_raw = 速度前向原值(
            速度模型,
            vel_train_x,
            速度标准化器,
            wall_distance_frac=vel_train_wall_distance,
            constraint_info=velocity_constraint_info,
        )
        速度监督, loss_u, loss_v = 速度监督损失(vel_pred_norm, vel_train_y, 速度标准化器)
        _, dense_vel_raw = 速度前向原值(
            速度模型,
            dense_train_x,
            速度标准化器,
            wall_distance_frac=dense_train_wall_distance,
            constraint_info=velocity_constraint_info,
        )
        壁面损失 = 壁面无滑移损失(dense_vel_raw, dense_train_split)
        流量损失 = 入口流量损失(dense_vel_raw, dense_train_split, train_cases, device)
        方程项 = 方程耦合损失(
            速度模型,
            压力模型,
            dense_train_split,
            输入标准化器,
            速度标准化器,
            压力标准化器,
            device,
            args.max_physics_points,
            velocity_constraint_info,
        )
        total = (
            速度监督
            + args.wall_weight * 壁面损失
            + args.inlet_flux_weight * 流量损失
            + 速度阶段连续性权重 * 方程项["连续性"]
            + 速度阶段动量权重 * 方程项["动量"]
        )
        total.backward()
        速度优化器.step()
        记录(
            "第一阶段_速度模型",
            epoch,
            float(total.item()),
            {
                "速度监督损失": float(速度监督.item()),
                "速度_u损失": float(loss_u.item()),
                "速度_v损失": float(loss_v.item()),
                "壁面损失": float(壁面损失.item()),
                "入口流量损失": float(流量损失.item()),
                "连续性损失": float(方程项["连续性"].item()),
                "动量损失": float(方程项["动量"].item()),
                "速度阶段连续性权重": float(速度阶段连续性权重),
                "速度阶段动量权重": float(速度阶段动量权重),
            },
        )

    _, 速度阶段验证指标 = 评估联合场(
        速度模型,
        压力模型,
        dense_val_split,
        速度标准化器,
        压力标准化器,
        device,
        velocity_constraint_info,
    )

    # 第二阶段：独立训练压力模型，完全忽视速度模型
    设置模块可训练(速度模型, False)
    设置模块可训练(压力模型, True)
    压力优化器 = torch.optim.Adam(压力模型.parameters(), lr=args.pressure_lr)
    for epoch in range(1, args.pressure_epochs + 1):
        压力模型.train()
        压力优化器.zero_grad()
        p_pred_norm, p_pred_raw = 压力前向原值(压力模型, p_train_x, 压力标准化器)
        压力监督 = 压力监督损失(p_pred_norm, p_train_y, 压力标准化器)
        _, dense_p_raw = 压力前向原值(压力模型, dense_train_x, 压力标准化器)
        出口损失 = 出口压力损失(dense_p_raw, dense_train_split)
        压降项 = 压降损失(dense_p_raw, dense_train_split, train_cases, device)
        方程项 = 方程耦合损失(
            速度模型,
            压力模型,
            dense_train_split,
            输入标准化器,
            速度标准化器,
            压力标准化器,
            device,
            args.max_physics_points,
            velocity_constraint_info,
        )
        total = (
            压力监督
            + args.outlet_pressure_weight * 出口损失
            + args.pressure_drop_weight * 压降项
            + 压力阶段动量权重 * 方程项["动量"]
        )
        total.backward()
        压力优化器.step()
        记录(
            "第二阶段_压力模型",
            epoch,
            float(total.item()),
            {
                "压力监督损失": float(压力监督.item()),
                "出口压力损失": float(出口损失.item()),
                "压降损失": float(压降项.item()),
                "动量损失": float(方程项["动量"].item()),
                "压力阶段动量权重": float(压力阶段动量权重),
            },
        )

    _, 压力阶段验证指标 = 评估联合场(
        速度模型,
        压力模型,
        dense_val_split,
        速度标准化器,
        压力标准化器,
        device,
        velocity_constraint_info,
    )

    # 第三阶段：交替耦合
    速度耦合优化器 = torch.optim.Adam(速度模型.parameters(), lr=args.coupling_velocity_lr)
    压力耦合优化器 = torch.optim.Adam(压力模型.parameters(), lr=args.coupling_pressure_lr)
    for epoch in range(1, args.coupling_epochs + 1):
        # 先更新速度模型
        设置模块可训练(速度模型, True)
        设置模块可训练(压力模型, False)
        速度模型.train()
        压力模型.eval()
        速度耦合优化器.zero_grad()
        vel_pred_norm, vel_pred_raw = 速度前向原值(
            速度模型,
            vel_train_x,
            速度标准化器,
            wall_distance_frac=vel_train_wall_distance,
            constraint_info=velocity_constraint_info,
        )
        速度监督, _, _ = 速度监督损失(vel_pred_norm, vel_train_y, 速度标准化器)
        _, dense_vel_raw = 速度前向原值(
            速度模型,
            dense_train_x,
            速度标准化器,
            wall_distance_frac=dense_train_wall_distance,
            constraint_info=velocity_constraint_info,
        )
        壁面损失 = 壁面无滑移损失(dense_vel_raw, dense_train_split)
        流量损失 = 入口流量损失(dense_vel_raw, dense_train_split, train_cases, device)
        方程项 = 方程耦合损失(
            速度模型,
            压力模型,
            dense_train_split,
            输入标准化器,
            速度标准化器,
            压力标准化器,
            device,
            args.max_physics_points,
            velocity_constraint_info,
        )
        速度总损失 = (
            args.coupling_velocity_supervision_weight * 速度监督
            + args.wall_weight * 壁面损失
            + args.inlet_flux_weight * 流量损失
            + args.coupling_continuity_weight * 方程项["连续性"]
            + args.coupling_momentum_weight * 方程项["动量"]
        )
        速度总损失.backward()
        速度耦合优化器.step()

        # 再更新压力模型
        设置模块可训练(速度模型, False)
        设置模块可训练(压力模型, True)
        速度模型.eval()
        压力模型.train()
        压力耦合优化器.zero_grad()
        p_pred_norm, p_pred_raw = 压力前向原值(压力模型, p_train_x, 压力标准化器)
        压力监督 = 压力监督损失(p_pred_norm, p_train_y, 压力标准化器)
        _, dense_p_raw = 压力前向原值(压力模型, dense_train_x, 压力标准化器)
        出口损失 = 出口压力损失(dense_p_raw, dense_train_split)
        压降项 = 压降损失(dense_p_raw, dense_train_split, train_cases, device)
        方程项 = 方程耦合损失(
            速度模型,
            压力模型,
            dense_train_split,
            输入标准化器,
            速度标准化器,
            压力标准化器,
            device,
            args.max_physics_points,
            velocity_constraint_info,
        )
        压力总损失 = (
            args.coupling_pressure_supervision_weight * 压力监督
            + args.outlet_pressure_weight * 出口损失
            + args.pressure_drop_weight * 压降项
            + args.coupling_momentum_weight * 方程项["动量"]
        )
        压力总损失.backward()
        压力耦合优化器.step()

        记录(
            "第三阶段_交替耦合",
            epoch,
            float((速度总损失 + 压力总损失).item()),
            {
                "耦合_速度监督损失": float(速度监督.item()),
                "耦合_压力监督损失": float(压力监督.item()),
                "耦合_壁面损失": float(壁面损失.item()),
                "耦合_入口流量损失": float(流量损失.item()),
                "耦合_出口压力损失": float(出口损失.item()),
                "耦合_压降损失": float(压降项.item()),
                "耦合_连续性损失": float(方程项["连续性"].item()),
                "耦合_动量损失": float(方程项["动量"].item()),
                "耦合_平均散度绝对值": float(方程项["平均散度绝对值"].item()),
                "耦合_最大散度绝对值": float(方程项["最大散度绝对值"].item()),
            },
        )

    train_pred, train_metrics = 评估联合场(
        速度模型,
        压力模型,
        dense_train_split,
        速度标准化器,
        压力标准化器,
        device,
        velocity_constraint_info,
    )
    val_pred, val_metrics = 评估联合场(
        速度模型,
        压力模型,
        dense_val_split,
        速度标准化器,
        压力标准化器,
        device,
        velocity_constraint_info,
    )
    train_case_metrics = [asdict(sup.compute_case_metrics(case_id, dense_train_split, train_pred)) for case_id in sorted(set(dense_train_split.case_ids.tolist()))]
    val_case_metrics = [asdict(sup.compute_case_metrics(case_id, dense_val_split, val_pred)) for case_id in sorted(set(dense_val_split.case_ids.tolist()))]
    sup.save_predictions(输出目录, dense_val_split, val_pred)
    pd.DataFrame(history).to_csv(输出目录 / "history.csv", index=False)

    config = {
        "family": spec.family,
        "data_subdir": spec.data_subdir,
        "feature_mode": args.feature_mode,
        "base_feature_cols": list(基础特征列),
        "drop_features": 删除特征,
        "feature_cols": list(特征列),
        "train_cases": train_cases,
        "val_cases": val_cases,
        "train_velocity_source": args.train_velocity_source,
        "val_velocity_source": args.val_velocity_source,
        "train_pressure_source": args.train_pressure_source,
        "val_pressure_source": args.val_pressure_source,
        "device": args.device,
        "velocity_hidden_layers": 解析层(args.velocity_hidden_layers),
        "pressure_hidden_layers": 解析层(args.pressure_hidden_layers),
        "activation": args.activation,
        "epochs": {
            "速度模型": args.velocity_epochs,
            "压力模型": args.pressure_epochs,
            "交替耦合": args.coupling_epochs,
        },
        "学习率": {
            "速度模型": args.velocity_lr,
            "压力模型": args.pressure_lr,
            "耦合_速度模型": args.coupling_velocity_lr,
            "耦合_压力模型": args.coupling_pressure_lr,
        },
        "权重": {
            "壁面": args.wall_weight,
            "入口流量": args.inlet_flux_weight,
            "连续性": args.continuity_weight,
            "速度阶段连续性": 速度阶段连续性权重,
            "速度阶段动量": 速度阶段动量权重,
            "出口压力": args.outlet_pressure_weight,
            "压降": args.pressure_drop_weight,
            "压力阶段动量": 压力阶段动量权重,
            "耦合动量": args.coupling_momentum_weight,
            "耦合连续性": args.coupling_continuity_weight,
            "耦合速度监督": args.coupling_velocity_supervision_weight,
            "耦合压力监督": args.coupling_pressure_supervision_weight,
        },
        "速度壁面约束": {
            "mode": args.velocity_wall_mode,
            "hard_wall_sharpness": args.hard_wall_sharpness,
        },
        "max_physics_points": args.max_physics_points,
        "seed": args.seed,
        "max_retries": args.max_retries,
        "runtime_guard": {"max_retries": args.max_retries},
        "输入标准化": {
            "x_mean": 输入标准化器.mean.tolist(),
            "x_std": 输入标准化器.std.tolist(),
        },
        "速度标准化": {
            "mean": 速度标准化器.mean.tolist(),
            "std": 速度标准化器.std.tolist(),
        },
        "压力标准化": {
            "mean": 压力标准化器.mean.tolist(),
            "std": 压力标准化器.std.tolist(),
        },
    }
    (输出目录 / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "速度阶段验证指标": 速度阶段验证指标,
        "压力阶段验证指标": 压力阶段验证指标,
        "最终训练指标": train_metrics,
        "最终验证指标": val_metrics,
        "训练工况指标": train_case_metrics,
        "验证工况指标": val_case_metrics,
    }
    (输出目录 / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(
        {
            "速度模型参数": 速度模型.state_dict(),
            "压力模型参数": 压力模型.state_dict(),
            "config": config,
        },
        输出目录 / "best.ckpt",
    )

    print(f"[完成] 输出目录={输出目录}")
    print(json.dumps({"最终验证指标": val_metrics}, ensure_ascii=False, indent=2))
    return {"output_dir": str(输出目录), "metrics": metrics}


def main() -> int:
    args = build_parser().parse_args()
    attempts = max(1, args.max_retries)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if attempt > 1:
                print(f"[重试] 第 {attempt}/{attempts} 次尝试")
            train_once(args)
            return 0
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[错误] 第 {attempt}/{attempts} 次尝试失败：{exc}", file=sys.stderr)
    if last_error is not None:
        raise last_error
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
