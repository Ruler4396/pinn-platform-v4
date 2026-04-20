#!/usr/bin/env python3
"""Run a supervised baseline for contraction_2d or bend_2d.

The script is intentionally conservative:
- full-batch CPU training by default
- explicit max_epochs / max_retries / patience
- no parallel workers
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bend_cases import BendCase, evaluate_inlet_profile, get_case as get_bend_case
from src.data.bend_geometry import BendGeometry
from src.data.contraction_cases import ContractionCase, get_case as get_contraction_case
from src.data.contraction_geometry import ContractionGeometry


@dataclass(frozen=True)
class FamilySpec:
    family: str
    data_subdir: str
    default_train_cases: tuple[str, ...]
    default_val_cases: tuple[str, ...]
    feature_sets: dict[str, tuple[str, ...]]
    default_feature_mode: str
    load_case: Callable[[str], ContractionCase | BendCase]
    enrich_frame: Callable[[pd.DataFrame, ContractionCase | BendCase], pd.DataFrame]


def _safe_region_normalized(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) / 2.0


def _safe_divide(num: np.ndarray, den: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.full_like(np.asarray(num, dtype=np.float64), fill, dtype=np.float64)
    den_arr = np.asarray(den, dtype=np.float64)
    return np.divide(num, den_arr, out=out, where=np.abs(den_arr) > 1.0e-12)


def enrich_contraction_frame(df: pd.DataFrame, case: ContractionCase) -> pd.DataFrame:
    out = df.copy()
    geom = ContractionGeometry(case)
    x = out["x_star"].to_numpy(dtype=np.float64)
    y = out["y_star"].to_numpy(dtype=np.float64)
    half_width = geom.half_width(x)
    width = geom.width(x)
    width_prime = geom.width_prime(x)
    wall_distance = out["wall_distance_star"].to_numpy(dtype=np.float64) if "wall_distance_star" in out.columns else geom.wall_distance(x, y)
    region_id = out["region_id"].to_numpy(dtype=np.int64) if "region_id" in out.columns else geom.region_id(x, y)
    eta_norm = _safe_divide(y, half_width)
    axial_frac = _safe_divide(x, np.full_like(x, case.total_length_over_w))
    wall_distance_frac = _safe_divide(wall_distance, half_width)
    inlet_profile = np.maximum(0.0, 1.5 * (1.0 - np.clip(_safe_divide(y, half_width), -1.0, 1.0) ** 2))
    ref_half_width = np.full_like(half_width, 0.5)
    half_width_ratio = _safe_divide(half_width, ref_half_width, fill=1.0)
    inv_half_width_ratio = _safe_divide(ref_half_width, half_width, fill=1.0)
    center_proximity = np.clip(1.0 - np.abs(np.clip(eta_norm, -1.0, 1.0)), 0.0, 1.0)
    wall_proximity = np.clip(1.0 - np.clip(wall_distance_frac, 0.0, 1.0), 0.0, 1.0)
    contraction_rate = np.maximum(0.0, -_safe_divide(width_prime, np.maximum(width, 1.0e-12)))
    out["beta"] = case.beta
    out["lc_over_w"] = case.lc_over_w
    out["wall_distance_star"] = wall_distance
    out["region_id"] = region_id
    out["half_width_star"] = half_width
    out["eta_norm_star"] = eta_norm
    out["axial_frac_star"] = axial_frac
    out["wall_distance_frac"] = np.clip(wall_distance_frac, 0.0, 1.0)
    out["region_id_norm"] = _safe_region_normalized(region_id)
    out["inlet_profile_star"] = inlet_profile
    out["tangent_x_star"] = 1.0
    out["tangent_y_star"] = 0.0
    out["segment_id_norm"] = np.where((x >= case.l_in_over_w) & (x <= case.l_in_over_w + case.lc_over_w), 1.0, 0.0)
    out["center_proximity_star"] = center_proximity
    out["wall_proximity_star"] = wall_proximity
    out["half_width_ratio_star"] = half_width_ratio
    out["inv_half_width_ratio_star"] = inv_half_width_ratio
    out["contraction_rate_star"] = contraction_rate
    out["curvature_star"] = np.zeros_like(x, dtype=np.float64)
    return out


def enrich_bend_frame(df: pd.DataFrame, case: BendCase) -> pd.DataFrame:
    out = df.copy()
    geom = BendGeometry(case)
    x = out["x_star"].to_numpy(dtype=np.float64)
    y = out["y_star"].to_numpy(dtype=np.float64)
    xi, eta, tx, ty, segment = geom.local_coordinates(x, y)
    wall_distance = out["wall_distance_star"].to_numpy(dtype=np.float64) if "wall_distance_star" in out.columns else geom.wall_distance(x, y)
    region_id = out["region_id"].to_numpy(dtype=np.int64) if "region_id" in out.columns else geom.region_id(x, y)
    axial_frac = _safe_divide(xi, np.full_like(xi, case.total_centerline_length_over_w))
    eta_norm = _safe_divide(eta, np.full_like(eta, geom.half_width))
    wall_distance_frac = _safe_divide(wall_distance, np.full_like(wall_distance, geom.half_width))
    # Keep the legacy parabolic proxy for backward compatibility with older checkpoints.
    inlet_profile = np.maximum(0.0, 1.5 * (1.0 - np.clip(eta_norm, -1.0, 1.0) ** 2))
    inlet_profile_shape = evaluate_inlet_profile(eta, geom.half_width, case.inlet_profile_name)
    profile_bias_map = {
        "parabolic": 0.0,
        "blunted": 0.0,
        "skewed_top": 1.0,
        "skewed_bottom": -1.0,
    }
    profile_flatness_map = {
        "parabolic": 0.0,
        "blunted": 1.0,
        "skewed_top": 0.0,
        "skewed_bottom": 0.0,
    }
    inlet_profile_bias = np.full_like(x, profile_bias_map.get(case.inlet_profile_name, 0.0), dtype=np.float64)
    inlet_profile_flatness = np.full_like(x, profile_flatness_map.get(case.inlet_profile_name, 0.0), dtype=np.float64)
    half_width_ratio = np.ones_like(x, dtype=np.float64)
    inv_half_width_ratio = np.ones_like(x, dtype=np.float64)
    center_proximity = np.clip(1.0 - np.abs(np.clip(eta_norm, -1.0, 1.0)), 0.0, 1.0)
    wall_proximity = np.clip(1.0 - np.clip(wall_distance_frac, 0.0, 1.0), 0.0, 1.0)
    contraction_rate = np.zeros_like(x, dtype=np.float64)
    curvature = np.where(np.asarray(segment) == 1, 1.0 / max(case.rc_over_w, 1.0e-12), 0.0)
    out["rc_over_w"] = case.rc_over_w
    out["theta_over_90"] = case.theta_deg / 90.0
    out["wall_distance_star"] = wall_distance
    out["region_id"] = region_id
    out["half_width_star"] = geom.half_width
    out["eta_norm_star"] = eta_norm
    out["axial_frac_star"] = axial_frac
    out["wall_distance_frac"] = np.clip(wall_distance_frac, 0.0, 1.0)
    out["region_id_norm"] = _safe_region_normalized(region_id)
    out["inlet_profile_star"] = inlet_profile
    out["inlet_profile_shape_star"] = inlet_profile_shape
    out["inlet_profile_bias_star"] = inlet_profile_bias
    out["inlet_profile_flatness_star"] = inlet_profile_flatness
    out["tangent_x_star"] = tx
    out["tangent_y_star"] = ty
    out["segment_id_norm"] = np.asarray(segment, dtype=np.float32) / 2.0
    out["center_proximity_star"] = center_proximity
    out["wall_proximity_star"] = wall_proximity
    out["half_width_ratio_star"] = half_width_ratio
    out["inv_half_width_ratio_star"] = inv_half_width_ratio
    out["contraction_rate_star"] = contraction_rate
    out["curvature_star"] = curvature
    return out


DEFAULTS: dict[str, FamilySpec] = {
    "contraction_2d": FamilySpec(
        family="contraction_2d",
        data_subdir="contraction_2d",
        default_train_cases=("C-base", "C-train-1", "C-train-2", "C-train-3", "C-train-4", "C-train-5"),
        default_val_cases=("C-val",),
        feature_sets={
            "basic": ("x_star", "y_star", "beta", "lc_over_w"),
            "geometry": (
                "x_star",
                "y_star",
                "beta",
                "lc_over_w",
                "wall_distance_frac",
                "region_id_norm",
                "eta_norm_star",
                "axial_frac_star",
                "inlet_profile_star",
                "segment_id_norm",
                "center_proximity_star",
                "wall_proximity_star",
                "half_width_ratio_star",
                "inv_half_width_ratio_star",
                "contraction_rate_star",
            ),
        },
        default_feature_mode="basic",
        load_case=lambda case_id: get_contraction_case(case_id),
        enrich_frame=lambda df, case: enrich_contraction_frame(df, case),
    ),
    "bend_2d": FamilySpec(
        family="bend_2d",
        data_subdir="bend_2d",
        default_train_cases=("B-base", "B-train-1", "B-train-2", "B-train-3"),
        default_val_cases=("B-val",),
        feature_sets={
            "basic": ("x_star", "y_star", "rc_over_w", "theta_over_90"),
            "geometry": (
                "x_star",
                "y_star",
                "rc_over_w",
                "theta_over_90",
                "wall_distance_frac",
                "region_id_norm",
                "eta_norm_star",
                "axial_frac_star",
                "inlet_profile_star",
                "tangent_x_star",
                "tangent_y_star",
                "segment_id_norm",
                "center_proximity_star",
                "wall_proximity_star",
                "curvature_star",
            ),
            "geometry_profileaware": (
                "x_star",
                "y_star",
                "rc_over_w",
                "theta_over_90",
                "wall_distance_frac",
                "region_id_norm",
                "eta_norm_star",
                "axial_frac_star",
                "inlet_profile_star",
                "inlet_profile_shape_star",
                "inlet_profile_bias_star",
                "inlet_profile_flatness_star",
                "tangent_x_star",
                "tangent_y_star",
                "segment_id_norm",
                "center_proximity_star",
                "wall_proximity_star",
                "curvature_star",
            ),
        },
        default_feature_mode="basic",
        load_case=lambda case_id: get_bend_case(case_id),
        enrich_frame=lambda df, case: enrich_bend_frame(df, case),
    ),
}
TARGET_COLS = ("u_star", "v_star", "p_star")


@dataclass
class NormalizationStats:
    x_mean: list[float]
    x_std: list[float]
    y_mean: list[float]
    y_std: list[float]


class SupervisedMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_layers: Sequence[int], activation: str = "silu"):
        super().__init__()
        act_map = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        act_cls = act_map.get(activation.lower(), nn.SiLU)
        dims = [in_dim, *hidden_layers, out_dim]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(act_cls())
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


class StandardScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler":
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"StandardScaler.fit expects 2D array, got shape={arr.shape}")
        mean = np.zeros((arr.shape[1],), dtype=np.float64)
        std = np.ones((arr.shape[1],), dtype=np.float64)
        for idx in range(arr.shape[1]):
            finite = arr[np.isfinite(arr[:, idx]), idx]
            if finite.size == 0:
                mean[idx] = 0.0
                std[idx] = 1.0
                continue
            mean[idx] = float(finite.mean())
            col_std = float(finite.std())
            std[idx] = 1.0 if col_std < 1.0e-8 else col_std
        return cls(mean, std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std + self.mean


@dataclass
class PreparedSplit:
    family: str
    features_raw: np.ndarray
    targets_raw: np.ndarray
    features_norm: np.ndarray
    targets_norm: np.ndarray
    case_ids: np.ndarray
    boundary_type: np.ndarray
    x_star: np.ndarray
    y_star: np.ndarray


@dataclass
class CaseMetrics:
    case_id: str
    num_points: int
    rel_l2_u: float
    rel_l2_v: float
    rel_l2_p: float
    rel_l2_speed: float
    mae_u: float
    mae_v: float
    mae_p: float
    wall_max_abs_u_pred: float
    wall_max_abs_v_pred: float
    pressure_drop_truth: float
    pressure_drop_pred: float
    pressure_drop_rel_error: float
    inlet_flux_truth: float | None
    inlet_flux_pred: float | None
    outlet_flux_truth: float | None
    outlet_flux_pred: float | None


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_hidden_layers(text: str) -> list[int]:
    layers = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not layers:
        raise ValueError("hidden layers must not be empty")
    return layers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def relative_l2(pred: np.ndarray, truth: np.ndarray, eps: float = 1.0e-12) -> float:
    return float(np.linalg.norm(pred - truth) / (np.linalg.norm(truth) + eps))


def get_family_spec(family: str) -> FamilySpec:
    try:
        return DEFAULTS[family]
    except KeyError as exc:
        raise KeyError(f"Unknown family '{family}'. Choices: {sorted(DEFAULTS)}") from exc


def resolve_feature_cols(spec: FamilySpec, feature_mode: str) -> tuple[str, ...]:
    try:
        return spec.feature_sets[feature_mode]
    except KeyError as exc:
        raise KeyError(
            f"Unknown feature_mode '{feature_mode}' for family '{spec.family}'. Choices: {sorted(spec.feature_sets)}"
        ) from exc


def apply_feature_drop(feature_cols: Sequence[str], drop_features: Sequence[str]) -> tuple[str, ...]:
    cols = list(feature_cols)
    drops = [item.strip() for item in drop_features if item.strip()]
    if not drops:
        return tuple(cols)
    missing = [item for item in drops if item not in cols]
    if missing:
        raise ValueError(f"Requested drop_features not present in selected feature set: {missing}")
    filtered = [col for col in cols if col not in set(drops)]
    if not filtered:
        raise ValueError("Dropping features would leave an empty feature set")
    return tuple(filtered)


def load_dense_case(spec: FamilySpec, case_id: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "cases" / spec.data_subdir / "data" / case_id / "field_dense.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing dense field for case {case_id}: {path}")
    df = pd.read_csv(path)
    required = {"case_id", "x_star", "y_star", "u_star", "v_star", "p_star", "boundary_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dense field for {case_id} missing columns: {sorted(missing)}")
    case = spec.load_case(case_id)
    return spec.enrich_frame(df.copy(), case)


def normalize_source_name(source: str) -> str:
    source_name = source.strip()
    if not source_name:
        return "dense"
    if source_name == "dense":
        return "dense"
    return source_name if source_name.endswith(".csv") else f"{source_name}.csv"


def load_case_source(spec: FamilySpec, case_id: str, source: str) -> pd.DataFrame:
    source_name = normalize_source_name(source)
    if source_name == "dense":
        return load_dense_case(spec, case_id)
    path = PROJECT_ROOT / "cases" / spec.data_subdir / "data" / case_id / source_name
    if not path.exists():
        raise FileNotFoundError(f"Missing observation file for case {case_id}: {path}")
    df = pd.read_csv(path)
    required = {"case_id", "x_star", "y_star", "u_obs", "v_obs", "p_obs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Observation file for {case_id} missing columns: {sorted(missing)}")
    for component in ("u", "v", "p"):
        mask_col = f"{component}_obs_mask"
        value_col = f"{component}_obs"
        if mask_col in df.columns:
            mask = df[mask_col].to_numpy(dtype=np.float32) > 0.5
            df.loc[~mask, value_col] = np.nan
    renamed = df.rename(columns={"u_obs": "u_star", "v_obs": "v_star", "p_obs": "p_star"}).copy()
    if "boundary_type" not in renamed.columns:
        renamed["boundary_type"] = "interior"
    case = spec.load_case(case_id)
    return spec.enrich_frame(renamed, case)


def build_split(
    spec: FamilySpec,
    case_ids: Sequence[str],
    feature_cols: Sequence[str],
    source: str = "dense",
    input_scaler: StandardScaler | None = None,
    output_scaler: StandardScaler | None = None,
) -> tuple[PreparedSplit, StandardScaler, StandardScaler]:
    frames = [load_case_source(spec, case_id, source) for case_id in case_ids]
    full = pd.concat(frames, ignore_index=True)

    x_raw = full[list(feature_cols)].to_numpy(dtype=np.float32)
    y_raw = full[list(TARGET_COLS)].to_numpy(dtype=np.float32)

    if input_scaler is None:
        input_scaler = StandardScaler.fit(x_raw)
    if output_scaler is None:
        output_scaler = StandardScaler.fit(y_raw)

    split = PreparedSplit(
        family=spec.family,
        features_raw=x_raw,
        targets_raw=y_raw,
        features_norm=input_scaler.transform(x_raw),
        targets_norm=output_scaler.transform(y_raw),
        case_ids=full["case_id"].to_numpy(),
        boundary_type=full["boundary_type"].to_numpy(),
        x_star=full["x_star"].to_numpy(dtype=np.float32),
        y_star=full["y_star"].to_numpy(dtype=np.float32),
    )
    return split, input_scaler, output_scaler


def compute_component_losses(pred: torch.Tensor, truth: torch.Tensor, pressure_weight: float) -> dict[str, torch.Tensor]:
    zero = pred.new_tensor(0.0)
    valid_u = torch.isfinite(truth[:, 0])
    valid_v = torch.isfinite(truth[:, 1])
    valid_p = torch.isfinite(truth[:, 2])
    loss_u = torch.mean((pred[valid_u, 0] - truth[valid_u, 0]) ** 2) if torch.any(valid_u) else zero
    loss_v = torch.mean((pred[valid_v, 1] - truth[valid_v, 1]) ** 2) if torch.any(valid_v) else zero
    loss_p = torch.mean((pred[valid_p, 2] - truth[valid_p, 2]) ** 2) if torch.any(valid_p) else zero
    total = loss_u + loss_v + pressure_weight * loss_p
    return {"total": total, "loss_u": loss_u, "loss_v": loss_v, "loss_p": loss_p}


def relative_extreme_loss(abs_err: torch.Tensor, scale: torch.Tensor, mode: str, p_order: float) -> torch.Tensor:
    if abs_err.numel() == 0:
        return abs_err.new_tensor(0.0)
    scale_safe = torch.clamp(scale, min=1.0e-12)
    rel_err = abs_err / scale_safe
    if mode == "mse":
        return torch.mean(rel_err ** 2)
    if mode == "max_abs":
        return torch.max(rel_err)
    if mode == "pnorm_abs":
        order = max(float(p_order), 2.0)
        return torch.mean(rel_err.pow(order)).pow(1.0 / order)
    raise ValueError(f"Unknown relative extreme loss mode: {mode}")


def curriculum_progress(
    epoch: int,
    max_epochs: int,
    start_ratio: float,
    end_ratio: float,
    power: float,
) -> float:
    if max_epochs <= 1:
        return 1.0
    ratio = np.clip((epoch - 1) / max(max_epochs - 1, 1), 0.0, 1.0)
    start = float(np.clip(start_ratio, 0.0, 1.0))
    end = float(np.clip(end_ratio, start + 1.0e-6, 1.0))
    if ratio <= start:
        return 0.0
    if ratio >= end:
        return 1.0
    local = (ratio - start) / max(end - start, 1.0e-6)
    shaped = local ** max(float(power), 1.0e-6)
    return float(np.clip(shaped, 0.0, 1.0))


def curriculum_p_order(
    epoch: int,
    max_epochs: int,
    p_start: float,
    p_end: float,
    start_ratio: float,
    end_ratio: float,
    power: float,
) -> float:
    progress = curriculum_progress(epoch, max_epochs, start_ratio, end_ratio, power)
    return float(p_start + (p_end - p_start) * progress)


def compute_tail_aware_losses(
    pred_norm: torch.Tensor,
    truth_norm: torch.Tensor,
    pred_raw: torch.Tensor,
    truth_raw: torch.Tensor,
    pressure_weight: float,
    pressure_loss_mode: str,
    pressure_loss_p: float,
    speed_tail_weight: float,
    speed_tail_mode: str,
    speed_tail_p: float,
    velocity_scale_value: float,
    pressure_scale_value: float,
) -> dict[str, torch.Tensor]:
    zero = pred_norm.new_tensor(0.0)
    valid_u = torch.isfinite(truth_norm[:, 0]) & torch.isfinite(truth_raw[:, 0])
    valid_v = torch.isfinite(truth_norm[:, 1]) & torch.isfinite(truth_raw[:, 1])
    valid_p = torch.isfinite(truth_raw[:, 2])
    valid_speed = torch.isfinite(truth_raw[:, 0]) & torch.isfinite(truth_raw[:, 1])
    loss_u = torch.mean((pred_norm[valid_u, 0] - truth_norm[valid_u, 0]) ** 2) if torch.any(valid_u) else zero
    loss_v = torch.mean((pred_norm[valid_v, 1] - truth_norm[valid_v, 1]) ** 2) if torch.any(valid_v) else zero
    velocity_scale = pred_norm.new_tensor(max(float(velocity_scale_value), 1.0e-12))
    pressure_scale = pred_norm.new_tensor(max(float(pressure_scale_value), 1.0e-12))
    pressure_abs_err = torch.abs(pred_raw[valid_p, 2] - truth_raw[valid_p, 2]) if torch.any(valid_p) else pred_norm.new_zeros((0,))
    loss_p = relative_extreme_loss(pressure_abs_err, pressure_scale, pressure_loss_mode, pressure_loss_p)
    if speed_tail_mode == "none" or speed_tail_weight <= 0.0 or not torch.any(valid_speed):
        loss_speed_tail = zero
    else:
        speed_true = torch.sqrt(torch.clamp(truth_raw[valid_speed, 0] ** 2 + truth_raw[valid_speed, 1] ** 2, min=1.0e-24))
        speed_pred = torch.sqrt(torch.clamp(pred_raw[valid_speed, 0] ** 2 + pred_raw[valid_speed, 1] ** 2, min=1.0e-24))
        speed_abs_err = torch.abs(speed_pred - speed_true)
        loss_speed_tail = relative_extreme_loss(speed_abs_err, velocity_scale, speed_tail_mode, speed_tail_p)
    total = loss_u + loss_v + pressure_weight * loss_p + speed_tail_weight * loss_speed_tail
    return {
        "total": total,
        "loss_u": loss_u,
        "loss_v": loss_v,
        "loss_p": loss_p,
        "loss_speed_tail": loss_speed_tail,
    }


def tensorize(split: PreparedSplit, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(split.features_norm, dtype=torch.float32, device=device)
    y = torch.tensor(split.targets_norm, dtype=torch.float32, device=device)
    return x, y


def compute_flux(coord: np.ndarray, normal_velocity: np.ndarray) -> float | None:
    if len(coord) < 2:
        return None
    order = np.argsort(coord)
    return float(np.trapezoid(normal_velocity[order], coord[order]))


def boundary_flux(case_id: str, family: str, boundary_type: str, x: np.ndarray, y: np.ndarray, velocity: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    if family == "contraction_2d":
        if boundary_type not in {"inlet", "outlet"}:
            return None
        coord = y
        normal_velocity = velocity[:, 0]
        return compute_flux(coord, normal_velocity)

    if family == "bend_2d":
        case = get_bend_case(case_id)
        if boundary_type == "inlet":
            return compute_flux(y, velocity[:, 0])
        if boundary_type == "outlet":
            alpha = math.radians(case.theta_deg)
            t_out = np.array([math.cos(alpha), math.sin(alpha)], dtype=np.float64)
            n_out = np.array([-math.sin(alpha), math.cos(alpha)], dtype=np.float64)
            center = np.array([case.l_in_over_w, case.rc_over_w], dtype=np.float64)
            theta1 = -0.5 * math.pi + alpha
            arc_end_center = center + case.rc_over_w * np.array([math.cos(theta1), math.sin(theta1)], dtype=np.float64)
            outlet_center = arc_end_center + case.l_out_over_w * t_out
            points = np.stack([x, y], axis=1)
            eta = (points - outlet_center) @ n_out
            normal_velocity = velocity @ t_out
            return compute_flux(eta, normal_velocity)
        return None

    return None


def compute_case_metrics(case_id: str, split: PreparedSplit, pred_raw: np.ndarray) -> CaseMetrics:
    mask = split.case_ids == case_id
    y_true = split.targets_raw[mask]
    y_pred = pred_raw[mask]
    boundary_type = split.boundary_type[mask]
    x_coord = split.x_star[mask].astype(np.float64)
    y_coord = split.y_star[mask].astype(np.float64)

    speed_true = np.sqrt(y_true[:, 0] ** 2 + y_true[:, 1] ** 2)
    speed_pred = np.sqrt(y_pred[:, 0] ** 2 + y_pred[:, 1] ** 2)

    wall_mask = boundary_type == "wall"
    inlet_mask = boundary_type == "inlet"
    outlet_mask = boundary_type == "outlet"

    pressure_drop_truth = float(y_true[inlet_mask, 2].mean() - y_true[outlet_mask, 2].mean()) if inlet_mask.any() and outlet_mask.any() else 0.0
    pressure_drop_pred = float(y_pred[inlet_mask, 2].mean() - y_pred[outlet_mask, 2].mean()) if inlet_mask.any() and outlet_mask.any() else 0.0
    denom = abs(pressure_drop_truth) + 1.0e-12
    pressure_drop_rel_error = float(abs(pressure_drop_pred - pressure_drop_truth) / denom)

    inlet_flux_truth = boundary_flux(case_id, split.family, "inlet", x_coord[inlet_mask], y_coord[inlet_mask], y_true[inlet_mask, :2]) if inlet_mask.any() else None
    inlet_flux_pred = boundary_flux(case_id, split.family, "inlet", x_coord[inlet_mask], y_coord[inlet_mask], y_pred[inlet_mask, :2]) if inlet_mask.any() else None
    outlet_flux_truth = boundary_flux(case_id, split.family, "outlet", x_coord[outlet_mask], y_coord[outlet_mask], y_true[outlet_mask, :2]) if outlet_mask.any() else None
    outlet_flux_pred = boundary_flux(case_id, split.family, "outlet", x_coord[outlet_mask], y_coord[outlet_mask], y_pred[outlet_mask, :2]) if outlet_mask.any() else None

    return CaseMetrics(
        case_id=case_id,
        num_points=int(mask.sum()),
        rel_l2_u=relative_l2(y_pred[:, 0], y_true[:, 0]),
        rel_l2_v=relative_l2(y_pred[:, 1], y_true[:, 1]),
        rel_l2_p=relative_l2(y_pred[:, 2], y_true[:, 2]),
        rel_l2_speed=relative_l2(speed_pred, speed_true),
        mae_u=float(np.mean(np.abs(y_pred[:, 0] - y_true[:, 0]))),
        mae_v=float(np.mean(np.abs(y_pred[:, 1] - y_true[:, 1]))),
        mae_p=float(np.mean(np.abs(y_pred[:, 2] - y_true[:, 2]))),
        wall_max_abs_u_pred=float(np.max(np.abs(y_pred[wall_mask, 0])) if wall_mask.any() else 0.0),
        wall_max_abs_v_pred=float(np.max(np.abs(y_pred[wall_mask, 1])) if wall_mask.any() else 0.0),
        pressure_drop_truth=pressure_drop_truth,
        pressure_drop_pred=pressure_drop_pred,
        pressure_drop_rel_error=pressure_drop_rel_error,
        inlet_flux_truth=inlet_flux_truth,
        inlet_flux_pred=inlet_flux_pred,
        outlet_flux_truth=outlet_flux_truth,
        outlet_flux_pred=outlet_flux_pred,
    )


def compute_extreme_error_metrics(pred_raw: np.ndarray, split: PreparedSplit) -> dict[str, float]:
    if len(pred_raw) == 0:
        return {
            "max_speed_err_over_speed_max": 0.0,
            "max_p_err_over_p_range": 0.0,
            "p95_speed_err_over_speed_max": 0.0,
            "p95_p_err_over_p_range": 0.0,
        }
    speed_true = np.sqrt(split.targets_raw[:, 0] ** 2 + split.targets_raw[:, 1] ** 2)
    speed_pred = np.sqrt(pred_raw[:, 0] ** 2 + pred_raw[:, 1] ** 2)
    speed_abs_err = np.abs(speed_pred - speed_true)
    pressure_abs_err = np.abs(pred_raw[:, 2] - split.targets_raw[:, 2])
    speed_max = float(np.max(speed_true)) if len(speed_true) else 1.0
    p_range = float(np.max(split.targets_raw[:, 2]) - np.min(split.targets_raw[:, 2])) if len(split.targets_raw) else 1.0
    speed_den = max(speed_max, 1.0e-12)
    p_den = max(p_range, 1.0e-12)
    return {
        "max_speed_err_over_speed_max": float(np.max(speed_abs_err) / speed_den),
        "max_p_err_over_p_range": float(np.max(pressure_abs_err) / p_den),
        "p95_speed_err_over_speed_max": float(np.quantile(speed_abs_err, 0.95) / speed_den),
        "p95_p_err_over_p_range": float(np.quantile(pressure_abs_err, 0.95) / p_den),
    }


def save_predictions(output_dir: Path, split: PreparedSplit, pred_raw: np.ndarray) -> None:
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    unique_cases = sorted(set(split.case_ids.tolist()))
    for case_id in unique_cases:
        mask = split.case_ids == case_id
        frame = pd.DataFrame(
            {
                "case_id": split.case_ids[mask],
                "x_star": split.x_star[mask],
                "y_star": split.y_star[mask],
                "boundary_type": split.boundary_type[mask],
                "u_true": split.targets_raw[mask, 0],
                "v_true": split.targets_raw[mask, 1],
                "p_true": split.targets_raw[mask, 2],
                "u_pred": pred_raw[mask, 0],
                "v_pred": pred_raw[mask, 1],
                "p_pred": pred_raw[mask, 2],
            }
        )
        frame.to_csv(predictions_dir / f"{case_id}_predictions.csv", index=False)


def maybe_write_figures(output_dir: Path, history: list[dict[str, float]]) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        (figures_dir / "README.txt").write_text("No history available.\n", encoding="utf-8")
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        (figures_dir / "README.txt").write_text(
            "matplotlib not available; figures were not generated. Use history.csv for plotting.\n",
            encoding="utf-8",
        )
        return

    hist = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist["epoch"], hist["train_total"], label="train_total")
    ax[0].plot(hist["epoch"], hist["val_total"], label="val_total")
    ax[0].set_title("Total loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()

    ax[1].plot(hist["epoch"], hist["val_rel_l2_u"], label="val_u")
    ax[1].plot(hist["epoch"], hist["val_rel_l2_v"], label="val_v")
    ax[1].plot(hist["epoch"], hist["val_rel_l2_p"], label="val_p")
    ax[1].set_title("Validation relative L2")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("relative L2")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train supervised contraction_2d or bend_2d baseline")
    parser.add_argument("--family", default="contraction_2d", choices=sorted(DEFAULTS))
    parser.add_argument("--train-cases", default="", help="comma-separated train cases; default depends on family")
    parser.add_argument("--val-cases", default="", help="comma-separated val cases; default depends on family")
    parser.add_argument("--run-name", default="", help="output run directory name; default depends on family")
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="cpu only by default to keep impact low")
    parser.add_argument("--feature-mode", default="", help="feature preset; default depends on family")
    parser.add_argument(
        "--drop-features",
        default="",
        help="comma-separated feature names to drop after resolving feature preset; useful for ablations like inlet_profile_star",
    )
    parser.add_argument("--train-observation-source", default="dense", help="dense or observation csv name for training")
    parser.add_argument("--val-observation-source", default="dense", help="dense or observation csv name for validation")
    parser.add_argument("--hidden-layers", default="128,128,128,128", help="comma-separated hidden layer widths")
    parser.add_argument("--activation", default="silu", choices=["tanh", "relu", "gelu", "silu"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pressure-weight", type=float, default=1.0)
    parser.add_argument("--pressure-loss-mode", default="mse", choices=["mse", "max_abs", "pnorm_abs"])
    parser.add_argument("--pressure-loss-p", type=float, default=16.0, help="used when pressure-loss-mode=pnorm_abs")
    parser.add_argument("--speed-tail-weight", type=float, default=0.0)
    parser.add_argument("--speed-tail-mode", default="none", choices=["none", "mse", "max_abs", "pnorm_abs"])
    parser.add_argument("--speed-tail-p", type=float, default=12.0, help="used when speed-tail-mode=pnorm_abs")
    parser.add_argument("--progressive-pnorm", action="store_true")
    parser.add_argument("--pressure-loss-p-start", type=float, default=4.0)
    parser.add_argument("--pressure-loss-p-end", type=float, default=16.0)
    parser.add_argument("--speed-tail-p-start", type=float, default=4.0)
    parser.add_argument("--speed-tail-p-end", type=float, default=12.0)
    parser.add_argument("--progressive-pnorm-start-ratio", type=float, default=0.0)
    parser.add_argument("--progressive-pnorm-end-ratio", type=float, default=1.0)
    parser.add_argument("--progressive-pnorm-power", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=2000, help="hard cap to avoid runaway training")
    parser.add_argument("--patience", type=int, default=200, help="early-stop patience")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=1, help="hard cap to avoid infinite retries")
    return parser


def train_once(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    spec = get_family_spec(args.family)
    train_cases = parse_csv_list(args.train_cases) if args.train_cases else list(spec.default_train_cases)
    val_cases = parse_csv_list(args.val_cases) if args.val_cases else list(spec.default_val_cases)
    feature_mode = args.feature_mode.strip() if args.feature_mode else spec.default_feature_mode
    requested_drop_features = parse_csv_list(args.drop_features)
    base_feature_cols = resolve_feature_cols(spec, feature_mode)
    feature_cols = apply_feature_drop(base_feature_cols, requested_drop_features)
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    run_name = args.run_name or f"{spec.data_subdir.replace('_2d', '')}_supervised_baseline"

    output_dir = PROJECT_ROOT / "results" / "supervised" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_source = args.train_observation_source
    val_source = args.val_observation_source
    train_split, input_scaler, output_scaler = build_split(spec, train_cases, feature_cols=feature_cols, source=train_source)
    val_split, _, _ = build_split(spec, val_cases, feature_cols=feature_cols, source=val_source, input_scaler=input_scaler, output_scaler=output_scaler)

    device = torch.device(args.device)
    train_x, train_y = tensorize(train_split, device)
    val_x, val_y = tensorize(val_split, device)
    train_y_raw = torch.tensor(train_split.targets_raw, dtype=torch.float32, device=device)
    val_y_raw = torch.tensor(val_split.targets_raw, dtype=torch.float32, device=device)
    y_mean_t = torch.tensor(output_scaler.mean, dtype=torch.float32, device=device)
    y_std_t = torch.tensor(output_scaler.std, dtype=torch.float32, device=device)
    train_speed_valid = np.isfinite(train_split.targets_raw[:, 0]) & np.isfinite(train_split.targets_raw[:, 1])
    val_speed_valid = np.isfinite(val_split.targets_raw[:, 0]) & np.isfinite(val_split.targets_raw[:, 1])
    train_velocity_scale = float(np.max(np.sqrt(train_split.targets_raw[train_speed_valid, 0] ** 2 + train_split.targets_raw[train_speed_valid, 1] ** 2))) if np.any(train_speed_valid) else 1.0
    val_velocity_scale = float(np.max(np.sqrt(val_split.targets_raw[val_speed_valid, 0] ** 2 + val_split.targets_raw[val_speed_valid, 1] ** 2))) if np.any(val_speed_valid) else 1.0
    train_p_valid = np.isfinite(train_split.targets_raw[:, 2])
    val_p_valid = np.isfinite(val_split.targets_raw[:, 2])
    train_pressure_scale = float(np.max(train_split.targets_raw[train_p_valid, 2]) - np.min(train_split.targets_raw[train_p_valid, 2])) if np.any(train_p_valid) else 1.0
    val_pressure_scale = float(np.max(val_split.targets_raw[val_p_valid, 2]) - np.min(val_split.targets_raw[val_p_valid, 2])) if np.any(val_p_valid) else 1.0

    model = SupervisedMLP(in_dim=train_x.shape[1], out_dim=train_y.shape[1], hidden_layers=hidden_layers, activation=args.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict[str, float]] = []
    best_val = math.inf
    best_epoch = 0
    patience_left = args.patience
    stop_epoch = args.max_epochs

    for epoch in range(1, args.max_epochs + 1):
        current_pressure_p = (
            curriculum_p_order(
                epoch,
                args.max_epochs,
                args.pressure_loss_p_start,
                args.pressure_loss_p_end,
                args.progressive_pnorm_start_ratio,
                args.progressive_pnorm_end_ratio,
                args.progressive_pnorm_power,
            )
            if args.progressive_pnorm and args.pressure_loss_mode == "pnorm_abs"
            else args.pressure_loss_p
        )
        current_speed_p = (
            curriculum_p_order(
                epoch,
                args.max_epochs,
                args.speed_tail_p_start,
                args.speed_tail_p_end,
                args.progressive_pnorm_start_ratio,
                args.progressive_pnorm_end_ratio,
                args.progressive_pnorm_power,
            )
            if args.progressive_pnorm and args.speed_tail_mode == "pnorm_abs"
            else args.speed_tail_p
        )

        model.train()
        optimizer.zero_grad()
        pred_train = model(train_x)
        pred_train_raw = pred_train * y_std_t + y_mean_t
        train_losses = compute_tail_aware_losses(
            pred_norm=pred_train,
            truth_norm=train_y,
            pred_raw=pred_train_raw,
            truth_raw=train_y_raw,
            pressure_weight=args.pressure_weight,
            pressure_loss_mode=args.pressure_loss_mode,
            pressure_loss_p=current_pressure_p,
            speed_tail_weight=args.speed_tail_weight,
            speed_tail_mode=args.speed_tail_mode,
            speed_tail_p=current_speed_p,
            velocity_scale_value=train_velocity_scale,
            pressure_scale_value=train_pressure_scale,
        )
        train_losses["total"].backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(val_x)
            pred_val_raw = pred_val * y_std_t + y_mean_t
            val_losses = compute_tail_aware_losses(
                pred_norm=pred_val,
                truth_norm=val_y,
                pred_raw=pred_val_raw,
                truth_raw=val_y_raw,
                pressure_weight=args.pressure_weight,
                pressure_loss_mode=args.pressure_loss_mode,
                pressure_loss_p=current_pressure_p,
                speed_tail_weight=args.speed_tail_weight,
                speed_tail_mode=args.speed_tail_mode,
                speed_tail_p=current_speed_p,
                velocity_scale_value=val_velocity_scale,
                pressure_scale_value=val_pressure_scale,
            )

        train_pred_raw = pred_train_raw.detach().cpu().numpy()
        val_pred_raw = pred_val_raw.detach().cpu().numpy()
        train_extremes = compute_extreme_error_metrics(train_pred_raw, train_split)
        val_extremes = compute_extreme_error_metrics(val_pred_raw, val_split)

        train_rel_u = relative_l2(train_pred_raw[:, 0], train_split.targets_raw[:, 0])
        val_rel_u = relative_l2(val_pred_raw[:, 0], val_split.targets_raw[:, 0])
        val_rel_v = relative_l2(val_pred_raw[:, 1], val_split.targets_raw[:, 1])
        val_rel_p = relative_l2(val_pred_raw[:, 2], val_split.targets_raw[:, 2])

        item = {
            "epoch": float(epoch),
            "train_total": float(train_losses["total"].item()),
            "train_loss_u": float(train_losses["loss_u"].item()),
            "train_loss_v": float(train_losses["loss_v"].item()),
            "train_loss_p": float(train_losses["loss_p"].item()),
            "train_loss_speed_tail": float(train_losses["loss_speed_tail"].item()),
            "val_total": float(val_losses["total"].item()),
            "val_loss_u": float(val_losses["loss_u"].item()),
            "val_loss_v": float(val_losses["loss_v"].item()),
            "val_loss_p": float(val_losses["loss_p"].item()),
            "val_loss_speed_tail": float(val_losses["loss_speed_tail"].item()),
            "train_rel_l2_u": train_rel_u,
            "val_rel_l2_u": val_rel_u,
            "val_rel_l2_v": val_rel_v,
            "val_rel_l2_p": val_rel_p,
            "train_max_speed_err": train_extremes["max_speed_err_over_speed_max"],
            "train_max_p_err": train_extremes["max_p_err_over_p_range"],
            "val_max_speed_err": val_extremes["max_speed_err_over_speed_max"],
            "val_max_p_err": val_extremes["max_p_err_over_p_range"],
            "pressure_loss_p_effective": float(current_pressure_p),
            "speed_tail_p_effective": float(current_speed_p),
        }
        history.append(item)

        if val_losses["total"].item() < best_val:
            best_val = float(val_losses["total"].item())
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_mean": input_scaler.mean.tolist(),
                    "input_std": input_scaler.std.tolist(),
                    "output_mean": output_scaler.mean.tolist(),
                    "output_std": output_scaler.std.tolist(),
                    "train_cases": train_cases,
                    "val_cases": val_cases,
                    "family": spec.family,
                    "feature_mode": feature_mode,
                    "base_feature_cols": list(base_feature_cols),
                    "drop_features": requested_drop_features,
                    "feature_cols": list(feature_cols),
                    "epoch": epoch,
                },
                output_dir / "best.ckpt",
            )
        else:
            patience_left -= 1

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"[epoch {epoch}] train={item['train_total']:.4e} val={item['val_total']:.4e} "
                f"val_u={item['val_rel_l2_u']:.4e} val_v={item['val_rel_l2_v']:.4e} val_p={item['val_rel_l2_p']:.4e} "
                f"val_max_speed={item['val_max_speed_err']:.4e} val_max_p={item['val_max_p_err']:.4e}"
            )

        if patience_left <= 0:
            stop_epoch = epoch
            print(f"[early-stop] epoch={epoch} best_epoch={best_epoch} best_val={best_val:.4e}")
            break

    checkpoint = torch.load(output_dir / "best.ckpt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        train_pred_best = model(train_x)
        val_pred_best = model(val_x)
    train_pred_best_raw = output_scaler.inverse_transform(train_pred_best.cpu().numpy())
    val_pred_best_raw = output_scaler.inverse_transform(val_pred_best.cpu().numpy())

    train_case_metrics = [asdict(compute_case_metrics(case_id, train_split, train_pred_best_raw)) for case_id in sorted(set(train_split.case_ids.tolist()))]
    val_case_metrics = [asdict(compute_case_metrics(case_id, val_split, val_pred_best_raw)) for case_id in sorted(set(val_split.case_ids.tolist()))]

    save_predictions(output_dir, val_split, val_pred_best_raw)
    maybe_write_figures(output_dir, history)

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    normalization = NormalizationStats(
        x_mean=input_scaler.mean.tolist(),
        x_std=input_scaler.std.tolist(),
        y_mean=output_scaler.mean.tolist(),
        y_std=output_scaler.std.tolist(),
    )
    config = {
        "family": args.family,
        "data_subdir": spec.data_subdir,
        "feature_mode": feature_mode,
        "base_feature_cols": list(base_feature_cols),
        "drop_features": requested_drop_features,
        "feature_cols": list(feature_cols),
        "target_cols": list(TARGET_COLS),
        "train_cases": train_cases,
        "val_cases": val_cases,
        "train_observation_source": normalize_source_name(train_source),
        "val_observation_source": normalize_source_name(val_source),
        "device": args.device,
        "hidden_layers": hidden_layers,
        "activation": args.activation,
        "lr": args.lr,
        "pressure_weight": args.pressure_weight,
        "pressure_loss_mode": args.pressure_loss_mode,
        "pressure_loss_p": args.pressure_loss_p,
        "pressure_loss_p_start": args.pressure_loss_p_start,
        "pressure_loss_p_end": args.pressure_loss_p_end,
        "speed_tail_weight": args.speed_tail_weight,
        "speed_tail_mode": args.speed_tail_mode,
        "speed_tail_p": args.speed_tail_p,
        "speed_tail_p_start": args.speed_tail_p_start,
        "speed_tail_p_end": args.speed_tail_p_end,
        "progressive_pnorm": args.progressive_pnorm,
        "progressive_pnorm_start_ratio": args.progressive_pnorm_start_ratio,
        "progressive_pnorm_end_ratio": args.progressive_pnorm_end_ratio,
        "progressive_pnorm_power": args.progressive_pnorm_power,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "max_retries": args.max_retries,
        "runtime_guard": {"max_retries": args.max_retries},
        "normalization": asdict(normalization),
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "stop_epoch": stop_epoch,
        "train_case_metrics": train_case_metrics,
        "val_case_metrics": val_case_metrics,
        "train_extreme_metrics": compute_extreme_error_metrics(train_pred_best_raw, train_split),
        "val_extreme_metrics": compute_extreme_error_metrics(val_pred_best_raw, val_split),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "stop_epoch": stop_epoch,
        "train_cases": train_cases,
        "val_cases": val_cases,
    }


def main() -> None:
    args = build_parser().parse_args()
    attempts = max(1, int(args.max_retries))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            summary = train_once(args)
            print(f"[done] output_dir={summary['output_dir']}")
            print(f"best_epoch={summary['best_epoch']} best_val_total={summary['best_val_total']:.6e}")
            print(f"stop_epoch={summary['stop_epoch']}")
            print(f"train_cases={','.join(summary['train_cases'])}")
            print(f"val_cases={','.join(summary['val_cases'])}")
            return
        except Exception as exc:  # pragma: no cover - runtime guard path
            last_error = exc
            print(f"[attempt {attempt}/{attempts}] failed: {exc}", file=sys.stderr)
            if attempt >= attempts:
                raise
    if last_error is not None:
        raise last_error


if __name__ == "__main__":
    main()
