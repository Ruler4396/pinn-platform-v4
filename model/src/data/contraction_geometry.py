"""Geometry and synthetic reference field helpers for contraction_2d."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from .contraction_cases import ContractionCase


@dataclass(frozen=True)
class GridSpec:
    nx: int = 241
    ny: int = 121
    boundary_samples: int = 401


def smoothstep5(x: np.ndarray) -> np.ndarray:
    return 6.0 * x**5 - 15.0 * x**4 + 10.0 * x**3


def smoothstep5_prime(x: np.ndarray) -> np.ndarray:
    return 30.0 * x**4 - 60.0 * x**3 + 30.0 * x**2


class ContractionGeometry:
    def __init__(self, case: ContractionCase):
        self.case = case
        self.l_in = case.l_in_over_w
        self.lc = case.lc_over_w
        self.l_out = case.l_out_over_w
        self.total_length = case.total_length_over_w
        self.beta = case.beta

    def width(self, x_star: np.ndarray) -> np.ndarray:
        x = np.asarray(x_star, dtype=np.float64)
        out = np.ones_like(x)
        mask = (x >= self.l_in) & (x <= self.l_in + self.lc)
        if np.any(mask):
            xi = (x[mask] - self.l_in) / self.lc
            out[mask] = 1.0 - (1.0 - self.beta) * smoothstep5(xi)
        out[x > self.l_in + self.lc] = self.beta
        return out

    def width_prime(self, x_star: np.ndarray) -> np.ndarray:
        x = np.asarray(x_star, dtype=np.float64)
        out = np.zeros_like(x)
        mask = (x >= self.l_in) & (x <= self.l_in + self.lc)
        if np.any(mask):
            xi = (x[mask] - self.l_in) / self.lc
            out[mask] = -((1.0 - self.beta) / self.lc) * smoothstep5_prime(xi)
        return out

    def half_width(self, x_star: np.ndarray) -> np.ndarray:
        return 0.5 * self.width(x_star)

    def top_wall(self, x_star: np.ndarray) -> np.ndarray:
        return self.half_width(x_star)

    def bottom_wall(self, x_star: np.ndarray) -> np.ndarray:
        return -self.half_width(x_star)

    def contains(self, x_star: np.ndarray, y_star: np.ndarray) -> np.ndarray:
        x = np.asarray(x_star, dtype=np.float64)
        y = np.asarray(y_star, dtype=np.float64)
        h = self.half_width(x)
        tol = 1.0e-9
        return (x >= -tol) & (x <= self.total_length + tol) & (np.abs(y) <= h + tol)

    def wall_distance(self, x_star: np.ndarray, y_star: np.ndarray) -> np.ndarray:
        h = self.half_width(x_star)
        return np.maximum(0.0, h - np.abs(y_star))

    def region_id(self, x_star: np.ndarray, y_star: np.ndarray) -> np.ndarray:
        x = np.asarray(x_star)
        y = np.asarray(y_star)
        d_wall = self.wall_distance(x, y)
        throat_mask = (x >= self.l_in - 0.5) & (x <= self.l_in + self.lc + 0.5)
        near_wall = d_wall < 0.15
        region = np.zeros_like(x, dtype=np.int64)
        region[throat_mask] = 1
        region[near_wall] = 2
        return region

    def interior_grid(self, grid: GridSpec) -> pd.DataFrame:
        x_values = np.linspace(0.0, self.total_length, grid.nx)
        y_values = np.linspace(-0.5, 0.5, grid.ny)
        rows = []
        sample_id = 0
        tol = 2.0 / max(grid.ny - 1, 1)
        for x in x_values:
            h = float(self.half_width(np.array([x]))[0])
            for y in y_values:
                if abs(y) <= h + 1.0e-12:
                    is_wall = abs(abs(y) - h) <= tol * 0.5
                    is_inlet = abs(x) <= 1.0e-12
                    is_outlet = abs(x - self.total_length) <= 1.0e-12
                    boundary_type = "interior"
                    if is_wall:
                        boundary_type = "wall"
                    if is_inlet:
                        boundary_type = "inlet"
                    if is_outlet:
                        boundary_type = "outlet"
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "x_star": x,
                            "y_star": y,
                            "is_boundary": int(is_wall or is_inlet or is_outlet),
                            "boundary_type": boundary_type,
                        }
                    )
                    sample_id += 1
        df = pd.DataFrame(rows)
        df["wall_distance_star"] = self.wall_distance(df["x_star"].to_numpy(), df["y_star"].to_numpy())
        df["region_id"] = self.region_id(df["x_star"].to_numpy(), df["y_star"].to_numpy())
        return df

    def boundary_points(self, grid: GridSpec) -> pd.DataFrame:
        samples = max(grid.boundary_samples, 8)
        rows = []
        sample_id = 0

        y_inlet = np.linspace(-0.5, 0.5, samples)
        for y in y_inlet:
            eta = 2.0 * y
            u_bc = max(0.0, 1.5 * (1.0 - eta**2))
            rows.append(
                {
                    "sample_id": sample_id,
                    "boundary_type": "inlet",
                    "x_star": 0.0,
                    "y_star": y,
                    "u_bc": u_bc,
                    "v_bc": 0.0,
                    "p_bc": np.nan,
                    "normal_x": -1.0,
                    "normal_y": 0.0,
                }
            )
            sample_id += 1

        x_wall = np.linspace(0.0, self.total_length, samples)
        for sign in (+1.0, -1.0):
            for x in x_wall:
                y = sign * float(self.half_width(np.array([x]))[0])
                rows.append(
                    {
                        "sample_id": sample_id,
                        "boundary_type": "wall",
                        "x_star": x,
                        "y_star": y,
                        "u_bc": 0.0,
                        "v_bc": 0.0,
                        "p_bc": np.nan,
                        "normal_x": 0.0,
                        "normal_y": sign,
                    }
                )
                sample_id += 1

        y_outlet = np.linspace(-0.5 * self.beta, 0.5 * self.beta, samples)
        for y in y_outlet:
            rows.append(
                {
                    "sample_id": sample_id,
                    "boundary_type": "outlet",
                    "x_star": self.total_length,
                    "y_star": y,
                    "u_bc": np.nan,
                    "v_bc": np.nan,
                    "p_bc": 0.0,
                    "normal_x": 1.0,
                    "normal_y": 0.0,
                }
            )
            sample_id += 1

        return pd.DataFrame(rows)

    def synthetic_reference_field(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df["x_star"].to_numpy(dtype=np.float64)
        y = df["y_star"].to_numpy(dtype=np.float64)
        h = self.half_width(x)
        h_x = 0.5 * self.width_prime(x)
        eta = np.divide(y, h, out=np.zeros_like(y), where=h > 0)

        u_star = 0.75 * np.divide(1.0 - eta**2, h, out=np.zeros_like(y), where=h > 0)
        v_star = 0.75 * (1.0 - eta**2) * eta * np.divide(h_x, h, out=np.zeros_like(y), where=h > 0)
        speed_star = np.sqrt(u_star**2 + v_star**2)

        # Synthetic pressure field for smoke / pipeline validation only.
        dpdx_star = -1.5 * np.divide(1.0, h**3, out=np.zeros_like(h), where=h > 0)
        x_unique = np.unique(x)
        h_unique = self.half_width(x_unique)
        dpdx_unique = -1.5 * np.divide(1.0, h_unique**3, out=np.zeros_like(h_unique), where=h_unique > 0)
        dx = np.diff(x_unique)
        trap = 0.5 * (dpdx_unique[:-1] + dpdx_unique[1:]) * dx
        cumulative = np.zeros_like(x_unique)
        cumulative[:-1] = np.cumsum(trap[::-1])[::-1]
        p_lookup = {float(xv): float(pv) for xv, pv in zip(x_unique, cumulative)}
        p_star = np.array([p_lookup[float(xv)] for xv in x], dtype=np.float64)

        out = df.copy()
        out["u_star"] = u_star
        out["v_star"] = v_star
        out["p_star"] = p_star
        out["speed_star"] = speed_star
        out["family"] = self.case.family
        out["case_id"] = self.case.case_id
        return out[
            [
                "sample_id",
                "case_id",
                "family",
                "x_star",
                "y_star",
                "u_star",
                "v_star",
                "p_star",
                "speed_star",
                "region_id",
                "wall_distance_star",
                "is_boundary",
                "boundary_type",
            ]
        ]

    def write_geometry_manifest(self, output_path: Path, field_source: str = "synthetic_streamfunction_smoke") -> None:
        payload = {
            "case_id": self.case.case_id,
            "family": self.case.family,
            "geometry": {
                "kind": "smooth_contraction",
                "W_star": 1.0,
                "beta": self.case.beta,
                "L_in_over_W": self.case.l_in_over_w,
                "L_c_over_W": self.case.lc_over_w,
                "L_out_over_W": self.case.l_out_over_w,
                "total_length_over_W": self.case.total_length_over_w,
            },
            "field_source": field_source,
        }
        if field_source == "synthetic_streamfunction_smoke":
            payload["warning"] = (
                "This manifest is suitable for geometry and pipeline validation only. "
                "Replace with CFD truth for formal experiments."
            )
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
