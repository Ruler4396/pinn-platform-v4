"""Sparse observation builders for pinn_v3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SparseSamplingSpec:
    sample_rates: tuple[float, ...] = (0.01, 0.05, 0.10, 0.15)
    default_noise_rates: tuple[float, ...] = (0.01, 0.03, 0.05)
    seed: int = 42


def _interior_pool(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["boundary_type"] == "interior"].copy()


def sample_uniform(df: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    pool = _interior_pool(df)
    count = max(1, int(round(len(pool) * rate)))
    return pool.sample(n=min(count, len(pool)), random_state=seed, replace=False).copy()


def sample_region_aware(df: pd.DataFrame, rate: float, seed: int, family: str) -> pd.DataFrame:
    pool = _interior_pool(df)
    count = max(1, int(round(len(pool) * rate)))
    if family == "contraction":
        quotas = {1: 0.45, 2: 0.35, 0: 0.20}
    elif family == "bend":
        quotas = {1: 0.40, 2: 0.35, 0: 0.25}
    else:
        quotas = {0: 1.0}

    rng = np.random.default_rng(seed)
    parts = []
    taken_ids: set[int] = set()
    for region_id, ratio in quotas.items():
        region_df = pool.loc[pool["region_id"] == region_id]
        if region_df.empty:
            continue
        n_region = max(1, int(round(count * ratio)))
        n_region = min(n_region, len(region_df))
        indices = rng.choice(region_df.index.to_numpy(), size=n_region, replace=False)
        taken_ids.update(int(i) for i in indices)
        parts.append(pool.loc[indices])

    sampled = pd.concat(parts, ignore_index=False) if parts else pool.iloc[:0].copy()
    if len(sampled) < count:
        remain = pool.loc[~pool.index.isin(sampled.index)]
        if not remain.empty:
            extra_count = min(count - len(sampled), len(remain))
            extra = remain.sample(n=extra_count, random_state=seed + 17, replace=False)
            sampled = pd.concat([sampled, extra], ignore_index=False)
    sampled = sampled.head(count).copy()
    return sampled


def to_observation_frame(df: pd.DataFrame, sampling_tag: str, noise_tag: str = "clean") -> pd.DataFrame:
    out = df[[
        "sample_id",
        "case_id",
        "family",
        "x_star",
        "y_star",
        "u_star",
        "v_star",
        "p_star",
        "region_id",
        "wall_distance_star",
    ]].copy()
    out = out.rename(columns={"u_star": "u_obs", "v_star": "v_obs", "p_star": "p_obs"})
    out.insert(3, "sampling_tag", sampling_tag)
    out.insert(4, "noise_tag", noise_tag)
    return out


def add_gaussian_noise(df: pd.DataFrame, noise_rate: float, seed: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    for column in ["u_obs", "v_obs", "p_obs"]:
        scale = float(np.std(out[column].to_numpy(dtype=float)))
        if scale <= 0.0:
            scale = 1.0
        out[column] = out[column].to_numpy(dtype=float) + rng.normal(0.0, noise_rate * scale, len(out))
    return out
