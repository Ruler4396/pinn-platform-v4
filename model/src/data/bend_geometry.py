"""Geometry and synthetic reference field helpers for bend_2d."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .bend_cases import BendCase, evaluate_inlet_profile


@dataclass(frozen=True)
class GridSpec:
    nx: int = 241
    ny: int = 241
    boundary_samples: int = 401


class BendGeometry:
    def __init__(self, case: BendCase):
        self.case = case
        self.width = 1.0
        self.half_width = 0.5
        self.l_in = case.l_in_over_w
        self.l_out = case.l_out_over_w
        self.rc = case.rc_over_w
        self.theta_deg = case.theta_deg
        self.alpha = np.deg2rad(case.theta_deg)
        self.theta0 = -0.5 * np.pi
        self.theta1 = self.theta0 + self.alpha
        self.center = np.array([self.l_in, self.rc], dtype=np.float64)
        self.arc_length = self.rc * self.alpha
        self.t_out = np.array([np.cos(self.alpha), np.sin(self.alpha)], dtype=np.float64)
        self.n_out = np.array([-np.sin(self.alpha), np.cos(self.alpha)], dtype=np.float64)
        self.arc_end_center = self.center + self.rc * np.array([np.cos(self.theta1), np.sin(self.theta1)], dtype=np.float64)
        self.outlet_center = self.arc_end_center + self.l_out * self.t_out
        boundary_points = np.vstack([
            np.array([[0.0, -self.half_width], [0.0, self.half_width]], dtype=np.float64),
            self.arc_point(self.theta0, -self.half_width),
            self.arc_point(self.theta0, self.half_width),
            self.arc_point(self.theta1, -self.half_width),
            self.arc_point(self.theta1, self.half_width),
            self.outlet_center + self.half_width * self.n_out,
            self.outlet_center - self.half_width * self.n_out,
        ])
        self.x_min = float(boundary_points[:, 0].min())
        self.x_max = float(boundary_points[:, 0].max())
        self.y_min = float(boundary_points[:, 1].min())
        self.y_max = float(boundary_points[:, 1].max())

    def arc_point(self, theta: np.ndarray | float, eta: np.ndarray | float) -> np.ndarray:
        theta_arr = np.asarray(theta, dtype=np.float64)
        eta_arr = np.asarray(eta, dtype=np.float64)
        radius = self.rc - eta_arr
        x = self.center[0] + radius * np.cos(theta_arr)
        y = self.center[1] + radius * np.sin(theta_arr)
        return np.stack([x, y], axis=-1)

    def _candidate_local_coordinates(self, x_star: np.ndarray, y_star: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(x_star, dtype=np.float64)
        y = np.asarray(y_star, dtype=np.float64)
        tol = 1.0e-3

        # inlet straight
        xi_in = x
        eta_in = y
        valid_in = (xi_in >= -tol) & (xi_in <= self.l_in + tol) & (np.abs(eta_in) <= self.half_width + tol)
        tx_in = np.ones_like(x)
        ty_in = np.zeros_like(x)

        # bend arc
        dx = x - self.center[0]
        dy = y - self.center[1]
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        valid_arc = (
            (theta >= self.theta0 - tol)
            & (theta <= self.theta1 + tol)
            & (r >= self.rc - self.half_width - tol)
            & (r <= self.rc + self.half_width + tol)
        )
        xi_arc = self.l_in + self.rc * (theta - self.theta0)
        eta_arc = self.rc - r
        tx_arc = -np.sin(theta)
        ty_arc = np.cos(theta)

        # outlet straight
        rel_x = x - self.arc_end_center[0]
        rel_y = y - self.arc_end_center[1]
        xi_out = rel_x * self.t_out[0] + rel_y * self.t_out[1]
        eta_out = rel_x * self.n_out[0] + rel_y * self.n_out[1]
        valid_out = (xi_out >= -tol) & (xi_out <= self.l_out + tol) & (np.abs(eta_out) <= self.half_width + tol)
        tx_out = np.full_like(x, self.t_out[0])
        ty_out = np.full_like(x, self.t_out[1])

        return {
            "valid_in": valid_in,
            "valid_arc": valid_arc,
            "valid_out": valid_out,
            "xi_in": xi_in,
            "eta_in": eta_in,
            "tx_in": tx_in,
            "ty_in": ty_in,
            "xi_arc": xi_arc,
            "eta_arc": eta_arc,
            "tx_arc": tx_arc,
            "ty_arc": ty_arc,
            "xi_out": xi_out,
            "eta_out": eta_out,
            "tx_out": tx_out,
            "ty_out": ty_out,
            "theta": theta,
        }

    def local_coordinates(self, x_star: np.ndarray, y_star: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cand = self._candidate_local_coordinates(x_star, y_star)
        x = np.asarray(x_star, dtype=np.float64)
        size = x.shape
        xi = np.full(size, np.nan, dtype=np.float64)
        eta = np.full(size, np.nan, dtype=np.float64)
        tx = np.full(size, np.nan, dtype=np.float64)
        ty = np.full(size, np.nan, dtype=np.float64)
        segment = np.full(size, -1, dtype=np.int64)

        for seg_id, key in [(0, "in"), (2, "out"), (1, "arc")]:
            valid = cand[f"valid_{key}"] & np.isnan(xi)
            if not np.any(valid):
                continue
            xi[valid] = cand[f"xi_{key}"][valid]
            eta[valid] = cand[f"eta_{key}"][valid]
            tx[valid] = cand[f"tx_{key}"][valid]
            ty[valid] = cand[f"ty_{key}"][valid]
            segment[valid] = seg_id

        unresolved = np.isnan(xi)
        if np.any(unresolved):
            for key, seg_id in [("in", 0), ("arc", 1), ("out", 2)]:
                valid = cand[f"valid_{key}"] & unresolved
                if not np.any(valid):
                    continue
                xi[valid] = cand[f"xi_{key}"][valid]
                eta[valid] = cand[f"eta_{key}"][valid]
                tx[valid] = cand[f"tx_{key}"][valid]
                ty[valid] = cand[f"ty_{key}"][valid]
                segment[valid] = seg_id
                unresolved = np.isnan(xi)
        return xi, eta, tx, ty, segment

    def contains(self, x_star: np.ndarray, y_star: np.ndarray) -> np.ndarray:
        cand = self._candidate_local_coordinates(x_star, y_star)
        return cand["valid_in"] | cand["valid_arc"] | cand["valid_out"]

    def wall_distance(self, x_star: np.ndarray, y_star: np.ndarray) -> np.ndarray:
        _, eta, _, _, _ = self.local_coordinates(x_star, y_star)
        distance = self.half_width - np.abs(eta)
        return np.maximum(0.0, distance)

    def region_id(self, x_star: np.ndarray, y_star: np.ndarray) -> np.ndarray:
        xi, eta, _, _, segment = self.local_coordinates(x_star, y_star)
        d_wall = self.half_width - np.abs(eta)
        near_wall = d_wall < 0.15
        bend_core = (segment == 1) | ((xi >= self.l_in - 0.5) & (xi <= self.l_in + self.arc_length + 0.5))
        region = np.zeros_like(xi, dtype=np.int64)
        region[bend_core] = 1
        region[near_wall] = 2
        return region

    def interior_grid(self, grid: GridSpec) -> pd.DataFrame:
        x_values = np.linspace(self.x_min, self.x_max, grid.nx)
        y_values = np.linspace(self.y_min, self.y_max, grid.ny)
        xx, yy = np.meshgrid(x_values, y_values)
        x_flat = xx.reshape(-1)
        y_flat = yy.reshape(-1)
        inside = self.contains(x_flat, y_flat)
        x_keep = x_flat[inside]
        y_keep = y_flat[inside]
        wall_distance = self.wall_distance(x_keep, y_keep)
        tol = max(self.x_max - self.x_min, self.y_max - self.y_min) / max(grid.nx, grid.ny) * 0.75
        rows = []
        for sample_id, (x, y, d_wall) in enumerate(zip(x_keep, y_keep, wall_distance, strict=False)):
            inlet = abs(float(x)) <= tol and abs(float(y)) <= self.half_width + tol
            outlet_rel = np.array([x, y]) - self.outlet_center
            outlet_eta = float(outlet_rel.dot(self.n_out))
            outlet_axial = float(outlet_rel.dot(self.t_out))
            outlet = abs(outlet_axial) <= tol and abs(outlet_eta) <= self.half_width + tol
            is_wall = d_wall <= tol
            boundary_type = "interior"
            if is_wall:
                boundary_type = "wall"
            if inlet:
                boundary_type = "inlet"
            if outlet:
                boundary_type = "outlet"
            rows.append(
                {
                    "sample_id": sample_id,
                    "x_star": float(x),
                    "y_star": float(y),
                    "is_boundary": int(is_wall or inlet or outlet),
                    "boundary_type": boundary_type,
                }
            )
        df = pd.DataFrame(rows)
        df["wall_distance_star"] = self.wall_distance(df["x_star"].to_numpy(), df["y_star"].to_numpy())
        df["region_id"] = self.region_id(df["x_star"].to_numpy(), df["y_star"].to_numpy())
        return df

    def boundary_points(self, grid: GridSpec) -> pd.DataFrame:
        samples = max(grid.boundary_samples, 8)
        rows = []
        sample_id = 0

        eta_samples = np.linspace(-self.half_width, self.half_width, samples)
        for eta in eta_samples:
            u_bc = float(evaluate_inlet_profile(np.asarray([eta]), self.half_width, self.case.inlet_profile_name)[0])
            rows.append(
                {
                    "sample_id": sample_id,
                    "boundary_type": "inlet",
                    "x_star": 0.0,
                    "y_star": float(eta),
                    "u_bc": u_bc,
                    "v_bc": 0.0,
                    "p_bc": np.nan,
                    "normal_x": -1.0,
                    "normal_y": 0.0,
                }
            )
            sample_id += 1

        xi_in = np.linspace(0.0, self.l_in, samples)
        for eta, normal in [(self.half_width, (0.0, 1.0)), (-self.half_width, (0.0, -1.0))]:
            for xi in xi_in:
                rows.append(
                    {
                        "sample_id": sample_id,
                        "boundary_type": "wall",
                        "x_star": float(xi),
                        "y_star": float(eta),
                        "u_bc": 0.0,
                        "v_bc": 0.0,
                        "p_bc": np.nan,
                        "normal_x": normal[0],
                        "normal_y": normal[1],
                    }
                )
                sample_id += 1

        theta_values = np.linspace(self.theta0, self.theta1, samples)
        for eta in [self.half_width, -self.half_width]:
            points = self.arc_point(theta_values, eta)
            if eta > 0.0:
                normals = -np.stack([np.cos(theta_values), np.sin(theta_values)], axis=1)
            else:
                normals = np.stack([np.cos(theta_values), np.sin(theta_values)], axis=1)
            for point, normal in zip(points, normals, strict=False):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "boundary_type": "wall",
                        "x_star": float(point[0]),
                        "y_star": float(point[1]),
                        "u_bc": 0.0,
                        "v_bc": 0.0,
                        "p_bc": np.nan,
                        "normal_x": float(normal[0]),
                        "normal_y": float(normal[1]),
                    }
                )
                sample_id += 1

        xi_out = np.linspace(0.0, self.l_out, samples)
        for eta in [self.half_width, -self.half_width]:
            normal = self.n_out if eta > 0.0 else -self.n_out
            for xi in xi_out:
                point = self.arc_end_center + xi * self.t_out + eta * self.n_out
                rows.append(
                    {
                        "sample_id": sample_id,
                        "boundary_type": "wall",
                        "x_star": float(point[0]),
                        "y_star": float(point[1]),
                        "u_bc": 0.0,
                        "v_bc": 0.0,
                        "p_bc": np.nan,
                        "normal_x": float(normal[0]),
                        "normal_y": float(normal[1]),
                    }
                )
                sample_id += 1

        eta_out_values = np.linspace(self.half_width, -self.half_width, samples)
        for eta in eta_out_values:
            point = self.outlet_center + eta * self.n_out
            rows.append(
                {
                    "sample_id": sample_id,
                    "boundary_type": "outlet",
                    "x_star": float(point[0]),
                    "y_star": float(point[1]),
                    "u_bc": np.nan,
                    "v_bc": np.nan,
                    "p_bc": 0.0,
                    "normal_x": float(self.t_out[0]),
                    "normal_y": float(self.t_out[1]),
                }
            )
            sample_id += 1

        return pd.DataFrame(rows)

    def synthetic_reference_field(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df["x_star"].to_numpy(dtype=np.float64)
        y = df["y_star"].to_numpy(dtype=np.float64)
        xi, eta, tx, ty, segment = self.local_coordinates(x, y)
        nx = -ty
        ny = tx

        base_speed = evaluate_inlet_profile(eta, self.half_width, self.case.inlet_profile_name)
        arc_progress = np.clip((xi - self.l_in) / max(self.arc_length, 1.0e-8), 0.0, 1.0)
        bend_mask = segment == 1
        tangential = base_speed * (1.0 + 0.08 * bend_mask * eta / max(self.rc, 1.0))
        normal = 0.04 * bend_mask * np.sin(np.pi * arc_progress) * (1.0 - (2.0 * eta) ** 2) * eta

        u_star = tangential * tx + normal * nx
        v_star = tangential * ty + normal * ny
        speed_star = np.sqrt(u_star**2 + v_star**2)
        p_star = (self.case.total_centerline_length_over_w - xi) * 24.0

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
                "kind": "smooth_constant_width_bend",
                "W_star": 1.0,
                "L_in_over_W": self.case.l_in_over_w,
                "R_c_over_W": self.case.rc_over_w,
                "theta_deg": self.case.theta_deg,
                "arc_length_over_W": self.case.arc_length_over_w,
                "L_out_over_W": self.case.l_out_over_w,
                "total_centerline_length_over_W": self.case.total_centerline_length_over_w,
                "outlet_direction": self.t_out.tolist(),
            },
            "inlet_profile": {
                "name": self.case.inlet_profile_name,
            },
            "field_source": field_source,
        }
        if field_source == "synthetic_streamfunction_smoke":
            payload["warning"] = (
                "This manifest is suitable for geometry and pipeline validation only. "
                "Replace with CFD truth for formal experiments."
            )
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
