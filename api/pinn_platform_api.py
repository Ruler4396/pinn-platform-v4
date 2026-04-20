#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get('PINN_PLATFORM_MODEL_ROOT', PROJECT_ROOT / 'model')).resolve()
DEFAULT_BEND_WORKSPACE_ROOT = MODEL_ROOT
LEGACY_BEND_WORKSPACE = PROJECT_ROOT / 'legacy' / 'pinn_v3'
if LEGACY_BEND_WORKSPACE.exists():
    DEFAULT_BEND_WORKSPACE_ROOT = LEGACY_BEND_WORKSPACE.resolve()
BEND_WORKSPACE_ROOT = Path(
    os.environ.get('PINN_PLATFORM_BEND_WORKSPACE_ROOT', str(DEFAULT_BEND_WORKSPACE_ROOT))
).resolve()
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from scripts import train_supervised as sup  # type: ignore
from scripts import train_velocity_pressure_independent as ind  # type: ignore
from src.data.contraction_cases import ContractionCase
from src.data.contraction_geometry import ContractionGeometry, GridSpec as ContractionGridSpec
from src.data.bend_cases import BendCase
from src.data.bend_geometry import BendGeometry, GridSpec as BendGridSpec


torch.set_num_threads(1)

DEFAULT_RUN_NAME = os.environ.get('PINN_V4_RUN_NAME', 'contraction_independent_geometry_notemplate_stagepde_mainline_v4')
BEND_PARABOLIC_RUN_NAME = os.environ.get(
    'PINN_BEND_PARABOLIC_RUN_NAME',
    'bend_independent_geometry_notemplate_parabolic_mainline_v1_20260419',
)
BEND_SKEWED_TOP_RUN_NAME = os.environ.get(
    'PINN_BEND_SKEWED_TOP_RUN_NAME',
    'bend_independent_geometry_skewed_top_mainline_v1_20260419',
)
BEND_SKEWED_BOTTOM_RUN_NAME = os.environ.get(
    'PINN_BEND_SKEWED_BOTTOM_RUN_NAME',
    'bend_independent_geometry_skewed_bottom_mainline_v1_20260419',
)
BEND_BLUNTED_RUN_NAME = os.environ.get('PINN_BEND_BLUNTED_RUN_NAME', 'bend_independent_blunted_geometry_notemplate_medium_v1_20260401')
MIN_VISCOSITY = 4.0e-4
MAX_VISCOSITY = 8.0e-3
RESPONSE_CACHE_MAX_ENTRIES = int(os.environ.get('PINN_V4_RESPONSE_CACHE_MAX_ENTRIES', '8'))
RESPONSE_CACHE_TTL_SECONDS = int(os.environ.get('PINN_V4_RESPONSE_CACHE_TTL_SECONDS', '1800'))
BEND_PREVIEW_TARGET_POINTS = int(os.environ.get('PINN_BEND_PREVIEW_TARGET_POINTS', '5600'))
BEND_FULL_TARGET_POINTS = int(os.environ.get('PINN_BEND_FULL_TARGET_POINTS', '11000'))


@dataclass
class GridBundle:
    frame_star: pd.DataFrame
    field_physical: list[dict[str, float]]
    xs_um: np.ndarray
    ys_um: np.ndarray
    velocity_lookup: dict[tuple[int, int], tuple[float, float]]
    inside_lookup: set[tuple[int, int]]
    x_step: float
    y_step: float


@dataclass
class CachedResponse:
    body: bytes
    created_at: float


class ResponseCache:
    def __init__(self, max_entries: int = RESPONSE_CACHE_MAX_ENTRIES, ttl_seconds: int = RESPONSE_CACHE_TTL_SECONDS):
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._store: OrderedDict[str, CachedResponse] = OrderedDict()

    def _purge_expired(self) -> None:
        now = time.time()
        expired_keys = [
            key for key, item in self._store.items() if now - item.created_at > self.ttl_seconds
        ]
        for key in expired_keys:
            self._store.pop(key, None)

    def get(self, key: str) -> bytes | None:
        self._purge_expired()
        item = self._store.get(key)
        if item is None:
            return None
        self._store.move_to_end(key)
        return item.body

    def set(self, key: str, body: bytes) -> None:
        self._purge_expired()
        self._store[key] = CachedResponse(body=body, created_at=time.time())
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)


class ContractionModelRuntime:
    def __init__(self, run_name: str = DEFAULT_RUN_NAME):
        run_dir = MODEL_ROOT / 'results' / 'pinn' / run_name
        ckpt_path = run_dir / 'best.ckpt'
        config_path = run_dir / 'config.json'
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
        if not config_path.exists():
            raise FileNotFoundError(f'Missing config: {config_path}')

        self.run_name = run_name
        self.config = json.loads(config_path.read_text(encoding='utf-8'))
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.feature_cols = tuple(self.config['feature_cols'])
        self.input_scaler = sup.StandardScaler(
            mean=np.array(self.config['输入标准化']['x_mean'], dtype=np.float32),
            std=np.array(self.config['输入标准化']['x_std'], dtype=np.float32),
        )
        self.velocity_scaler = ind.输出标准化器(
            mean=np.array(self.config['速度标准化']['mean'], dtype=np.float32),
            std=np.array(self.config['速度标准化']['std'], dtype=np.float32),
        )
        self.pressure_scaler = ind.输出标准化器(
            mean=np.array(self.config['压力标准化']['mean'], dtype=np.float32),
            std=np.array(self.config['压力标准化']['std'], dtype=np.float32),
        )
        self.velocity_model = ind.多层感知机(
            len(self.feature_cols),
            2,
            self.config.get('velocity_hidden_layers', [128, 128, 128]),
            activation=self.config.get('activation', 'silu'),
        )
        self.pressure_model = ind.多层感知机(
            len(self.feature_cols),
            1,
            self.config.get('pressure_hidden_layers', [128, 128, 128]),
            activation=self.config.get('activation', 'silu'),
        )
        self.velocity_model.load_state_dict(checkpoint['速度模型参数'])
        self.pressure_model.load_state_dict(checkpoint['压力模型参数'])
        self.velocity_model.eval()
        self.pressure_model.eval()
        self.constraint_info = ind.构建壁面硬约束信息(
            self.feature_cols,
            self.config.get('速度壁面约束', {}).get('mode', 'soft'),
            float(self.config.get('速度壁面约束', {}).get('hard_wall_sharpness', 12.0)),
        )

    def predict_star(self, frame_star: pd.DataFrame, case: ContractionCase) -> pd.DataFrame:
        enriched = sup.enrich_contraction_frame(frame_star, case)
        x_raw = enriched[list(self.feature_cols)].to_numpy(dtype=np.float32)
        x_norm = self.input_scaler.transform(x_raw)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32)
        wall_distance = ind.提取壁面距离分数(x_raw, torch.device('cpu'), self.constraint_info)
        with torch.no_grad():
            _, vel_raw = ind.速度前向原值(
                self.velocity_model,
                x_tensor,
                self.velocity_scaler,
                wall_distance_frac=wall_distance,
                constraint_info=self.constraint_info,
            )
            _, p_raw = ind.压力前向原值(self.pressure_model, x_tensor, self.pressure_scaler)
        out = enriched.copy()
        vel_np = vel_raw.detach().cpu().numpy()
        p_np = p_raw.detach().cpu().numpy().reshape(-1)
        out['u_star'] = vel_np[:, 0]
        out['v_star'] = vel_np[:, 1]
        out['p_star'] = p_np
        out['speed_star'] = np.sqrt(out['u_star'] ** 2 + out['v_star'] ** 2)
        return out


class SupervisedFieldRuntime:
    def __init__(self, run_root: Path, run_name: str):
        run_dir = run_root / 'results' / 'supervised' / run_name
        ckpt_path = run_dir / 'best.ckpt'
        config_path = run_dir / 'config.json'
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
        if not config_path.exists():
            raise FileNotFoundError(f'Missing config: {config_path}')

        self.run_name = run_name
        self.config = json.loads(config_path.read_text(encoding='utf-8'))
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.feature_cols = tuple(self.config['feature_cols'])
        self.input_scaler = sup.StandardScaler(
            mean=np.array(self.config['normalization']['x_mean'], dtype=np.float32),
            std=np.array(self.config['normalization']['x_std'], dtype=np.float32),
        )
        self.output_scaler = sup.StandardScaler(
            mean=np.array(self.config['normalization']['y_mean'], dtype=np.float32),
            std=np.array(self.config['normalization']['y_std'], dtype=np.float32),
        )
        self.model = sup.SupervisedMLP(
            len(self.feature_cols),
            3,
            self.config.get('hidden_layers', [128, 128, 128, 128]),
            activation=self.config.get('activation', 'silu'),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict_star(self, frame_star: pd.DataFrame, case: BendCase) -> pd.DataFrame:
        enriched = sup.enrich_bend_frame(frame_star, case)
        x_raw = enriched[list(self.feature_cols)].to_numpy(dtype=np.float32)
        x_norm = self.input_scaler.transform(x_raw)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32)
        with torch.no_grad():
            pred_norm = self.model(x_tensor).detach().cpu().numpy()
        pred_raw = self.output_scaler.inverse_transform(pred_norm)
        out = enriched.copy()
        out['u_star'] = pred_raw[:, 0]
        out['v_star'] = pred_raw[:, 1]
        out['p_star'] = pred_raw[:, 2]
        out['speed_star'] = np.sqrt(out['u_star'] ** 2 + out['v_star'] ** 2)
        return out


class IndependentFieldRuntime:
    def __init__(self, run_root: Path, run_name: str):
        run_dir = run_root / 'results' / 'pinn' / run_name
        ckpt_path = run_dir / 'best.ckpt'
        config_path = run_dir / 'config.json'
        if not ckpt_path.exists():
            raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
        if not config_path.exists():
            raise FileNotFoundError(f'Missing config: {config_path}')

        self.run_name = run_name
        self.config = json.loads(config_path.read_text(encoding='utf-8'))
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.feature_cols = tuple(self.config['feature_cols'])
        self.input_scaler = sup.StandardScaler(
            mean=np.array(self.config['输入标准化']['x_mean'], dtype=np.float32),
            std=np.array(self.config['输入标准化']['x_std'], dtype=np.float32),
        )
        self.velocity_scaler = ind.输出标准化器(
            mean=np.array(self.config['速度标准化']['mean'], dtype=np.float32),
            std=np.array(self.config['速度标准化']['std'], dtype=np.float32),
        )
        self.pressure_scaler = ind.输出标准化器(
            mean=np.array(self.config['压力标准化']['mean'], dtype=np.float32),
            std=np.array(self.config['压力标准化']['std'], dtype=np.float32),
        )
        self.velocity_model = ind.多层感知机(
            len(self.feature_cols),
            2,
            self.config.get('velocity_hidden_layers', [128, 128, 128]),
            activation=self.config.get('activation', 'silu'),
        )
        self.pressure_model = ind.多层感知机(
            len(self.feature_cols),
            1,
            self.config.get('pressure_hidden_layers', [128, 128, 128]),
            activation=self.config.get('activation', 'silu'),
        )
        self.velocity_model.load_state_dict(checkpoint['速度模型参数'])
        self.pressure_model.load_state_dict(checkpoint['压力模型参数'])
        self.velocity_model.eval()
        self.pressure_model.eval()
        self.constraint_info = ind.构建壁面硬约束信息(
            self.feature_cols,
            self.config.get('速度壁面约束', {}).get('mode', 'soft'),
            float(self.config.get('速度壁面约束', {}).get('hard_wall_sharpness', 12.0)),
        )

    def predict_bend_star(self, frame_star: pd.DataFrame, case: BendCase) -> pd.DataFrame:
        enriched = sup.enrich_bend_frame(frame_star, case)
        x_raw = enriched[list(self.feature_cols)].to_numpy(dtype=np.float32)
        x_norm = self.input_scaler.transform(x_raw)
        x_tensor = torch.tensor(x_norm, dtype=torch.float32)
        wall_distance = ind.提取壁面距离分数(x_raw, torch.device('cpu'), self.constraint_info)
        with torch.no_grad():
            _, vel_raw = ind.速度前向原值(
                self.velocity_model,
                x_tensor,
                self.velocity_scaler,
                wall_distance_frac=wall_distance,
                constraint_info=self.constraint_info,
            )
            _, p_raw = ind.压力前向原值(self.pressure_model, x_tensor, self.pressure_scaler)
        out = enriched.copy()
        vel_np = vel_raw.detach().cpu().numpy()
        p_np = p_raw.detach().cpu().numpy().reshape(-1)
        out['u_star'] = vel_np[:, 0]
        out['v_star'] = vel_np[:, 1]
        out['p_star'] = p_np
        out['speed_star'] = np.sqrt(out['u_star'] ** 2 + out['v_star'] ** 2)
        return out

    def predict_star(self, frame_star: pd.DataFrame, case: BendCase) -> pd.DataFrame:
        return self.predict_bend_star(frame_star, case)


class SyntheticBendRuntime:
    def __init__(self, run_name: str):
        self.run_name = run_name

    def predict_star(self, frame_star: pd.DataFrame, case: BendCase) -> pd.DataFrame:
        geometry = BendGeometry(case)
        return geometry.synthetic_reference_field(frame_star)

    def predict_bend_star(self, frame_star: pd.DataFrame, case: BendCase) -> pd.DataFrame:
        return self.predict_star(frame_star, case)


class ScenarioEngine:
    def __init__(self, run_name: str = DEFAULT_RUN_NAME):
        self.contraction_runtime = ContractionModelRuntime(run_name=run_name)
        try:
            self.bend_parabolic_runtime = IndependentFieldRuntime(BEND_WORKSPACE_ROOT, BEND_PARABOLIC_RUN_NAME)
        except FileNotFoundError:
            self.bend_parabolic_runtime = SyntheticBendRuntime('synthetic_bend_parabolic_fallback')
        try:
            self.bend_skewed_top_runtime = IndependentFieldRuntime(BEND_WORKSPACE_ROOT, BEND_SKEWED_TOP_RUN_NAME)
        except FileNotFoundError:
            self.bend_skewed_top_runtime = SyntheticBendRuntime('synthetic_bend_skewed_top_fallback')
        try:
            self.bend_skewed_bottom_runtime = IndependentFieldRuntime(BEND_WORKSPACE_ROOT, BEND_SKEWED_BOTTOM_RUN_NAME)
        except FileNotFoundError:
            self.bend_skewed_bottom_runtime = SyntheticBendRuntime('synthetic_bend_skewed_bottom_fallback')
        try:
            self.bend_blunted_runtime = IndependentFieldRuntime(BEND_WORKSPACE_ROOT, BEND_BLUNTED_RUN_NAME)
        except FileNotFoundError:
            self.bend_blunted_runtime = SyntheticBendRuntime('synthetic_bend_blunted_fallback')

    @staticmethod
    def _mean_velocity_ms(scenario: dict[str, Any]) -> float:
        return float(scenario['flow']['meanVelocity'])

    @staticmethod
    def _width_um(scenario: dict[str, Any]) -> float:
        return float(scenario['geometry']['wUm'])

    @staticmethod
    def _pressure_scale(scenario: dict[str, Any]) -> float:
        width_m = max(float(scenario['geometry']['wUm']) * 1.0e-6, 1.0e-12)
        mu = float(scenario['fluid']['viscosity'])
        u = float(scenario['flow']['meanVelocity'])
        return mu * u / width_m

    @staticmethod
    def _compute_reynolds(scenario: dict[str, Any]) -> float:
        rho = float(scenario['fluid']['density'])
        u = float(scenario['flow']['meanVelocity'])
        w = float(scenario['geometry']['wUm']) * 1.0e-6
        mu = float(scenario['fluid']['viscosity'])
        return rho * u * w / max(mu, 1.0e-12)

    def _build_contraction_case(self, scenario: dict[str, Any]) -> ContractionCase:
        geom = scenario['geometry']
        flow = scenario['flow']
        fluid = scenario['fluid']
        return ContractionCase(
            case_id='web-custom-contraction',
            beta=float(geom['beta']),
            lc_over_w=float(geom['lCOverW']),
            w_um=float(geom['wUm']),
            l_in_over_w=float(geom['lInOverW']),
            l_out_over_w=float(geom['lOutOverW']),
            u_mean_mm_s=float(flow['meanVelocity']) * 1000.0,
            rho=float(fluid['density']),
            mu=float(fluid['viscosity']),
            note='website remote inference',
        )

    def _build_bend_case(self, scenario: dict[str, Any]) -> BendCase:
        geom = scenario['geometry']
        flow = scenario['flow']
        fluid = scenario['fluid']
        return BendCase(
            case_id='web-custom-bend',
            rc_over_w=float(geom['rcOverW']),
            theta_deg=float(geom['thetaDeg']),
            w_um=float(geom['wUm']),
            l_in_over_w=float(geom['lInOverW']),
            l_out_over_w=float(geom['lOutOverW']),
            u_mean_mm_s=float(flow['meanVelocity']) * 1000.0,
            rho=float(fluid['density']),
            mu=float(fluid['viscosity']),
            nu=float(fluid['viscosity']) / max(float(fluid['density']), 1.0e-12),
            inlet_profile_name=str(geom.get('inletProfile', 'parabolic')),
            note='website remote bend checkpoint inference',
        )

    def _predict_bend_with_runtime(self, frame_star: pd.DataFrame, case: BendCase) -> pd.DataFrame:
        profile = case.inlet_profile_name
        if profile == 'blunted':
            return self.bend_blunted_runtime.predict_bend_star(frame_star, case)
        if profile == 'skewed_top':
            return self.bend_skewed_top_runtime.predict_star(frame_star, case)
        if profile == 'skewed_bottom':
            return self.bend_skewed_bottom_runtime.predict_star(frame_star, case)
        return self.bend_parabolic_runtime.predict_star(frame_star, case)

    @staticmethod
    def _grid_shape(span_x_star: float, span_y_star: float, resolution: str = 'full') -> tuple[int, int]:
        if resolution == 'preview':
            max_samples = 112
            min_samples = 72
        else:
            max_samples = 168
            min_samples = 104
        aspect = span_x_star / max(span_y_star, 1.0e-9)
        if aspect >= 1:
            nx = max_samples
            ny = int(round(max_samples / max(aspect, 1.0)))
        else:
            nx = int(round(max_samples * aspect))
            ny = max_samples
        nx = max(min_samples, min(max_samples, nx))
        ny = max(min_samples, min(max_samples, ny))
        return nx, ny

    def _build_contraction_frame_star(
        self,
        scenario: dict[str, Any],
        resolution: str = 'full',
    ) -> tuple[pd.DataFrame, ContractionCase, ContractionGeometry]:
        case = self._build_contraction_case(scenario)
        geometry = ContractionGeometry(case)
        nx, ny = self._grid_shape(case.total_length_over_w, 1.0, resolution=resolution)
        frame_star = geometry.interior_grid(ContractionGridSpec(nx=nx, ny=ny, boundary_samples=241))
        return frame_star, case, geometry

    def _build_bend_frame_star(
        self,
        scenario: dict[str, Any],
        resolution: str = 'full',
    ) -> tuple[pd.DataFrame, BendCase, BendGeometry]:
        case = self._build_bend_case(scenario)
        geometry = BendGeometry(case)
        span_x = geometry.x_max - geometry.x_min
        span_y = geometry.y_max - geometry.y_min
        nx, ny = self._grid_shape(span_x, span_y, resolution=resolution)
        boundary_samples = 241
        frame_star = geometry.interior_grid(BendGridSpec(nx=nx, ny=ny, boundary_samples=boundary_samples))
        target_points = BEND_PREVIEW_TARGET_POINTS if resolution == 'preview' else BEND_FULL_TARGET_POINTS
        if target_points > 0 and len(frame_star) > 0 and len(frame_star) < target_points:
            max_samples = 256 if resolution == 'preview' else 336
            for _ in range(4):
                scale = math.sqrt(target_points / max(len(frame_star), 1))
                nx = min(max_samples, max(nx + 8, int(math.ceil(nx * min(scale, 1.5)))))
                ny = min(max_samples, max(ny + 8, int(math.ceil(ny * min(scale, 1.5)))))
                frame_star = geometry.interior_grid(BendGridSpec(nx=nx, ny=ny, boundary_samples=boundary_samples))
                if len(frame_star) >= target_points or (nx >= max_samples and ny >= max_samples):
                    break
        return frame_star, case, geometry

    def _build_target_frame_star(
        self,
        scenario: dict[str, Any],
        resolution: str = 'full',
    ) -> tuple[pd.DataFrame, ContractionCase | BendCase, ContractionGeometry | BendGeometry]:
        if scenario['geometry']['type'] == 'contraction':
            return self._build_contraction_frame_star(scenario, resolution=resolution)
        return self._build_bend_frame_star(scenario, resolution=resolution)

    def _predict_frame_star(
        self,
        scenario: dict[str, Any],
        frame_star: pd.DataFrame,
    ) -> pd.DataFrame:
        if scenario['geometry']['type'] == 'contraction':
            case = self._build_contraction_case(scenario)
            return self.contraction_runtime.predict_star(frame_star, case)
        case = self._build_bend_case(scenario)
        return self._predict_bend_with_runtime(frame_star, case)

    def _predict_contraction_field_star(
        self,
        scenario: dict[str, Any],
        resolution: str = 'full',
    ) -> tuple[pd.DataFrame, ContractionCase, ContractionGeometry]:
        frame_star, case, geometry = self._build_contraction_frame_star(scenario, resolution=resolution)
        predicted = self.contraction_runtime.predict_star(frame_star, case)
        return predicted, case, geometry

    def _predict_bend_field_star(
        self,
        scenario: dict[str, Any],
        resolution: str = 'full',
    ) -> tuple[pd.DataFrame, BendCase, BendGeometry]:
        frame_star, case, geometry = self._build_bend_frame_star(scenario, resolution=resolution)
        predicted = self._predict_bend_with_runtime(frame_star, case)
        return predicted, case, geometry

    def _predict_single_point_star(self, scenario: dict[str, Any], point_um: dict[str, float]) -> pd.DataFrame | None:
        w_um = self._width_um(scenario)
        x_star = float(point_um['x']) / max(w_um, 1.0e-12)
        y_star = float(point_um['y']) / max(w_um, 1.0e-12)
        frame = pd.DataFrame([{
            'sample_id': 0,
            'x_star': x_star,
            'y_star': y_star,
            'is_boundary': 0,
            'boundary_type': 'interior',
        }])
        if scenario['geometry']['type'] == 'contraction':
            case = self._build_contraction_case(scenario)
            geometry = ContractionGeometry(case)
            inside = bool(geometry.contains(np.array([x_star]), np.array([y_star]))[0])
            if not inside:
                return None
            frame['wall_distance_star'] = geometry.wall_distance(frame['x_star'].to_numpy(), frame['y_star'].to_numpy())
            frame['region_id'] = geometry.region_id(frame['x_star'].to_numpy(), frame['y_star'].to_numpy())
            return self.contraction_runtime.predict_star(frame, case)
        case = self._build_bend_case(scenario)
        geometry = BendGeometry(case)
        inside = bool(geometry.contains(np.array([x_star]), np.array([y_star]))[0])
        if not inside:
            return None
        frame['wall_distance_star'] = geometry.wall_distance(frame['x_star'].to_numpy(), frame['y_star'].to_numpy())
        frame['region_id'] = geometry.region_id(frame['x_star'].to_numpy(), frame['y_star'].to_numpy())
        return self._predict_bend_with_runtime(frame, case)

    def _star_to_physical_points(self, scenario: dict[str, Any], frame_star: pd.DataFrame) -> list[dict[str, float]]:
        w_um = self._width_um(scenario)
        u_scale = self._mean_velocity_ms(scenario)
        p_scale = self._pressure_scale(scenario)
        outlet_pressure = float(scenario['flow']['outletPressure'])
        points: list[dict[str, float]] = []
        for row in frame_star.itertuples(index=False):
            ux = float(row.u_star) * u_scale
            uy = float(row.v_star) * u_scale
            p = outlet_pressure + float(row.p_star) * p_scale
            points.append(
                {
                    'x': float(row.x_star) * w_um,
                    'y': float(row.y_star) * w_um,
                    'ux': ux,
                    'uy': uy,
                    'p': p,
                    'speed': math.hypot(ux, uy),
                }
            )
        return points

    def _build_grid_bundle(self, scenario: dict[str, Any], frame_star: pd.DataFrame, field_physical: list[dict[str, float]]) -> GridBundle:
        xs_um = np.sort(frame_star['x_star'].drop_duplicates().to_numpy(dtype=np.float64) * self._width_um(scenario))
        ys_um = np.sort(frame_star['y_star'].drop_duplicates().to_numpy(dtype=np.float64) * self._width_um(scenario))
        x_step = float(np.median(np.diff(xs_um))) if len(xs_um) > 1 else max(self._width_um(scenario) * 0.05, 1.0)
        y_step = float(np.median(np.diff(ys_um))) if len(ys_um) > 1 else max(self._width_um(scenario) * 0.05, 1.0)
        velocity_lookup: dict[tuple[int, int], tuple[float, float]] = {}
        inside_lookup: set[tuple[int, int]] = set()
        for row, point in zip(frame_star.itertuples(index=False), field_physical, strict=False):
            ix = int(round(point['x'] / max(x_step, 1.0e-9)))
            iy = int(round(point['y'] / max(y_step, 1.0e-9)))
            velocity_lookup[(ix, iy)] = (point['ux'], point['uy'])
            inside_lookup.add((ix, iy))
        return GridBundle(
            frame_star=frame_star,
            field_physical=field_physical,
            xs_um=xs_um,
            ys_um=ys_um,
            velocity_lookup=velocity_lookup,
            inside_lookup=inside_lookup,
            x_step=x_step,
            y_step=y_step,
        )

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def _sample_velocity(self, bundle: GridBundle, x_um: float, y_um: float) -> tuple[float, float] | None:
        ix = int(round(x_um / max(bundle.x_step, 1.0e-9)))
        iy = int(round(y_um / max(bundle.y_step, 1.0e-9)))
        for radius in range(0, 3):
            best_key: tuple[int, int] | None = None
            best_score = float('inf')
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    key = (ix + dx, iy + dy)
                    if key not in bundle.velocity_lookup:
                        continue
                    score = dx * dx + dy * dy
                    if score < best_score:
                        best_score = score
                        best_key = key
            if best_key is not None:
                return bundle.velocity_lookup[best_key]
        return None

    def _geometry_contains_um(self, scenario: dict[str, Any], x_um: float, y_um: float) -> bool:
        w_um = self._width_um(scenario)
        x_star = x_um / max(w_um, 1.0e-12)
        y_star = y_um / max(w_um, 1.0e-12)
        if scenario['geometry']['type'] == 'contraction':
            geometry = ContractionGeometry(self._build_contraction_case(scenario))
            return bool(geometry.contains(np.array([x_star]), np.array([y_star]))[0])
        geometry = BendGeometry(self._build_bend_case(scenario))
        return bool(geometry.contains(np.array([x_star]), np.array([y_star]))[0])

    def _build_streamlines(self, scenario: dict[str, Any], bundle: GridBundle) -> list[list[dict[str, float]]]:
        width_um = self._width_um(scenario)
        step_um = max(min(width_um * 0.22, 40.0), 16.0)
        if scenario['geometry']['type'] == 'contraction':
            lin = float(scenario['geometry']['lInOverW']) * width_um
            total = (float(scenario['geometry']['lInOverW']) + float(scenario['geometry']['lCOverW']) + float(scenario['geometry']['lOutOverW'])) * width_um
            x0 = min(lin * 0.08, width_um * 0.6)
            offsets = np.linspace(-0.42 * width_um, 0.42 * width_um, 11)
            seeds = [(x0, float(offset)) for offset in offsets]
            max_x = total
        else:
            case = self._build_bend_case(scenario)
            geom = BendGeometry(case)
            x0 = 0.08 * case.l_in_over_w * width_um
            offsets = np.linspace(-0.42 * width_um, 0.42 * width_um, 10)
            seeds = [(x0, float(offset)) for offset in offsets]
            max_x = geom.x_max * width_um + case.l_out_over_w * 0.02 * width_um

        lines: list[list[dict[str, float]]] = []
        for sx, sy in seeds:
            if not self._geometry_contains_um(scenario, sx, sy):
                continue
            line = [{'x': sx, 'y': sy}]
            x, y = sx, sy
            for _ in range(180):
                velocity = self._sample_velocity(bundle, x, y)
                if velocity is None:
                    break
                ux, uy = velocity
                mag = math.hypot(ux, uy)
                if mag < 1.0e-12:
                    break
                nx = ux / mag
                ny = uy / mag
                x_next = x + nx * step_um
                y_next = y + ny * step_um
                if x_next > max_x or not self._geometry_contains_um(scenario, x_next, y_next):
                    break
                line.append({'x': x_next, 'y': y_next})
                x, y = x_next, y_next
            if len(line) > 2:
                lines.append(line)
        return lines

    def _sample_contraction_curves(self, scenario: dict[str, Any]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        width_um = self._width_um(scenario)
        geom = scenario['geometry']
        total_star = float(geom['lInOverW']) + float(geom['lCOverW']) + float(geom['lOutOverW'])
        total_um = total_star * width_um
        main_x = np.linspace(0.0, total_um, 120)
        outlet_start = (float(geom['lInOverW']) + float(geom['lCOverW'])) * width_um
        branch_x = np.linspace(outlet_start, total_um, 72)

        main_frame = pd.DataFrame({'x': main_x, 'y': np.zeros_like(main_x)})
        branch_frame = pd.DataFrame({'x': branch_x, 'y': np.zeros_like(branch_x)})
        main = []
        for idx, row in main_frame.iterrows():
            pred = self._predict_single_point_star(scenario, {'x': float(row.x), 'y': 0.0})
            if pred is None:
                continue
            point = self._star_to_physical_points(scenario, pred)[0]
            main.append({'s': float(row.x / max(total_um, 1.0e-12)), 'speed': point['speed'], 'p': point['p']})
        branch = []
        denom = max(total_um - outlet_start, 1.0e-12)
        for _, row in branch_frame.iterrows():
            pred = self._predict_single_point_star(scenario, {'x': float(row.x), 'y': 0.0})
            if pred is None:
                continue
            point = self._star_to_physical_points(scenario, pred)[0]
            branch.append({'s': float((row.x - outlet_start) / denom), 'speed': point['speed'], 'p': point['p']})
        return main, branch

    def _sample_bend_curves(self, scenario: dict[str, Any]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        case = self._build_bend_case(scenario)
        geometry = BendGeometry(case)
        points: list[tuple[float, float, float]] = []
        for x_star in np.linspace(0.0, case.l_in_over_w, 36):
            points.append((x_star, 0.0, x_star))
        arc_len = case.arc_length_over_w
        for theta in np.linspace(geometry.theta0, geometry.theta1, 72):
            xy = geometry.arc_point(theta, 0.0)
            xi = case.l_in_over_w + case.rc_over_w * (theta - geometry.theta0)
            points.append((float(xy[0]), float(xy[1]), float(xi)))
        for xi_local in np.linspace(0.0, case.l_out_over_w, 48):
            x_star = geometry.arc_end_center[0] + xi_local * geometry.t_out[0]
            y_star = geometry.arc_end_center[1] + xi_local * geometry.t_out[1]
            points.append((float(x_star), float(y_star), float(case.l_in_over_w + arc_len + xi_local)))

        total_len = max(case.total_centerline_length_over_w, 1.0e-12)
        main = []
        for x_star, y_star, s_star in points:
            pred = self._predict_single_point_star(
                scenario,
                {'x': x_star * case.w_um, 'y': y_star * case.w_um},
            )
            if pred is None:
                continue
            point = self._star_to_physical_points(scenario, pred)[0]
            main.append({'s': float(s_star / total_len), 'speed': point['speed'], 'p': point['p']})

        branch = []
        for xi_local in np.linspace(0.0, case.l_out_over_w, 48):
            x_star = geometry.arc_end_center[0] + xi_local * geometry.t_out[0]
            y_star = geometry.arc_end_center[1] + xi_local * geometry.t_out[1]
            pred = self._predict_single_point_star(
                scenario,
                {'x': x_star * case.w_um, 'y': y_star * case.w_um},
            )
            if pred is None:
                continue
            point = self._star_to_physical_points(scenario, pred)[0]
            branch.append({'s': float(xi_local / max(case.l_out_over_w, 1.0e-12)), 'speed': point['speed'], 'p': point['p']})
        return main, branch

    @staticmethod
    def _scenario_seed(scenario: dict[str, Any], salt: str) -> int:
        payload = json.dumps({'salt': salt, 'scenario': scenario}, sort_keys=True, ensure_ascii=False).encode('utf-8')
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], 'little', signed=False)

    def _build_observation_candidate_frame_star(
        self,
        scenario: dict[str, Any],
        resolution: str,
    ) -> pd.DataFrame:
        candidate_resolution = 'full' if resolution == 'preview' else resolution
        candidate_frame, _, _ = self._build_target_frame_star(scenario, resolution=candidate_resolution)
        return candidate_frame

    def _sample_sparse_frame_star(
        self,
        scenario: dict[str, Any],
        target_frame_star: pd.DataFrame,
        candidate_frame_star: pd.DataFrame,
    ) -> pd.DataFrame:
        sample_count = max(1, int(round(len(target_frame_star) * float(scenario['sparse']['sampleRatePct']) / 100.0)))
        working = candidate_frame_star
        if 'wall_distance_star' in working.columns:
            filtered = working[working['wall_distance_star'] >= 0.045].copy()
            if len(filtered) >= sample_count:
                working = filtered

        rng = np.random.default_rng(self._scenario_seed(scenario, 'sparse-observations'))
        indices = np.arange(len(working))
        if sample_count >= len(indices):
            selected = indices
        elif str(scenario['sparse'].get('strategy', 'region_aware')) == 'region_aware' and 'region_id' in working.columns:
            regions = working['region_id'].to_numpy(dtype=int)
            picked: list[int] = []
            quotas = {
                1: int(round(sample_count * 0.4)),
                2: int(round(sample_count * 0.25)),
            }
            quotas[0] = max(sample_count - quotas[1] - quotas[2], 0)
            for region_id, quota in quotas.items():
                region_indices = indices[regions == region_id]
                if len(region_indices) == 0 or quota <= 0:
                    continue
                take = min(quota, len(region_indices))
                picked.extend(rng.choice(region_indices, size=take, replace=False).tolist())
            if len(picked) < sample_count:
                remaining = np.setdiff1d(indices, np.array(picked, dtype=int), assume_unique=False)
                extra = rng.choice(remaining, size=min(sample_count - len(picked), len(remaining)), replace=False)
                picked.extend(extra.tolist())
            selected = np.array(sorted(set(picked)))[:sample_count]
        else:
            selected = np.sort(rng.choice(indices, size=sample_count, replace=False))
        return working.iloc[selected].reset_index(drop=True)

    def _apply_sparse_noise(
        self,
        scenario: dict[str, Any],
        field_points: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        noise_pct = float(scenario['sparse'].get('noisePct', 0.0)) / 100.0
        if noise_pct <= 0:
            return [dict(point) for point in field_points]
        rng = np.random.default_rng(self._scenario_seed(scenario, 'sparse-noise'))
        noisy: list[dict[str, float]] = []
        for point in field_points:
            sample = dict(point)
            sample['ux'] *= 1 + float(rng.normal(0, noise_pct * 0.35))
            sample['uy'] *= 1 + float(rng.normal(0, noise_pct * 0.35))
            sample['p'] *= 1 + float(rng.normal(0, noise_pct * 0.2))
            sample['speed'] = math.hypot(sample['ux'], sample['uy'])
            noisy.append(sample)
        return noisy

    def _sample_sparse_points(self, scenario: dict[str, Any], frame_star: pd.DataFrame, field_physical: list[dict[str, float]]) -> list[dict[str, float]]:
        sample_rate = max(1, int(round(len(field_physical) * float(scenario['sparse']['sampleRatePct']) / 100.0)))
        rng = np.random.default_rng(20260402)
        indices = np.arange(len(field_physical))
        if sample_rate >= len(indices):
            selected = indices
        elif str(scenario['sparse'].get('strategy', 'region_aware')) == 'region_aware' and 'region_id' in frame_star.columns:
            regions = frame_star['region_id'].to_numpy(dtype=int)
            picked: list[int] = []
            quotas = {
                1: int(round(sample_rate * 0.4)),
                2: int(round(sample_rate * 0.25)),
            }
            quotas[0] = max(sample_rate - quotas[1] - quotas[2], 0)
            for region_id, quota in quotas.items():
                region_indices = indices[regions == region_id]
                if len(region_indices) == 0 or quota <= 0:
                    continue
                take = min(quota, len(region_indices))
                picked.extend(rng.choice(region_indices, size=take, replace=False).tolist())
            if len(picked) < sample_rate:
                remaining = np.setdiff1d(indices, np.array(picked, dtype=int), assume_unique=False)
                extra = rng.choice(remaining, size=min(sample_rate - len(picked), len(remaining)), replace=False)
                picked.extend(extra.tolist())
            selected = np.array(sorted(set(picked)))[:sample_rate]
        else:
            selected = np.sort(rng.choice(indices, size=sample_rate, replace=False))

        noise_pct = float(scenario['sparse'].get('noisePct', 0)) / 100.0
        sparse_points: list[dict[str, float]] = []
        for idx in selected:
            point = dict(field_physical[int(idx)])
            if noise_pct > 0:
                point['ux'] *= 1 + float(rng.normal(0, noise_pct * 0.35))
                point['uy'] *= 1 + float(rng.normal(0, noise_pct * 0.35))
                point['p'] *= 1 + float(rng.normal(0, noise_pct * 0.2))
                point['speed'] = math.hypot(point['ux'], point['uy'])
            sparse_points.append(point)
        return sparse_points

    @staticmethod
    def _point_key(point: dict[str, float]) -> tuple[float, float]:
        return (round(float(point['x']), 6), round(float(point['y']), 6))

    @staticmethod
    def _fit_affine_delta(
        prior: np.ndarray,
        observed: np.ndarray,
        *,
        scale_ridge: float,
        bias_ridge: float,
    ) -> tuple[float, float]:
        if len(prior) == 0 or len(observed) == 0:
            return 1.0, 0.0
        design = np.column_stack([prior, np.ones_like(prior)])
        rhs = observed - prior
        lhs = design.T @ design + np.diag([max(scale_ridge, 1.0e-9), max(bias_ridge, 1.0e-9)])
        coeff = np.linalg.solve(lhs, design.T @ rhs)
        return 1.0 + float(coeff[0]), float(coeff[1])

    @staticmethod
    def _nearest_sparse_points(
        target: dict[str, float],
        sparse_points: list[dict[str, float]],
        count: int,
    ) -> list[dict[str, float]]:
        ranked = sorted(
            sparse_points,
            key=lambda point: max(math.hypot(point['x'] - target['x'], point['y'] - target['y']), 1.0),
        )
        return ranked[:max(1, min(count, len(ranked)))]

    @staticmethod
    def _select_anchor_points(
        sparse_points: list[dict[str, float]],
        max_anchors: int = 48,
    ) -> list[dict[str, float]]:
        if len(sparse_points) <= max_anchors:
            return sparse_points
        coords = np.array([[float(point['x']), float(point['y'])] for point in sparse_points], dtype=np.float64)
        centroid = coords.mean(axis=0)
        first = int(np.argmax(np.sum((coords - centroid) ** 2, axis=1)))
        selected = [first]
        min_dist2 = np.sum((coords - coords[first]) ** 2, axis=1)
        while len(selected) < max_anchors:
            next_idx = int(np.argmax(min_dist2))
            selected.append(next_idx)
            dist2 = np.sum((coords - coords[next_idx]) ** 2, axis=1)
            min_dist2 = np.minimum(min_dist2, dist2)
        return [sparse_points[idx] for idx in selected]

    @staticmethod
    def _rbf_kernel(query_xy: np.ndarray, anchor_xy: np.ndarray, radius: float) -> np.ndarray:
        radius2 = max(float(radius) ** 2, 1.0e-12)
        diff = query_xy[:, None, :] - anchor_xy[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        return np.exp(-dist2 / radius2)

    @staticmethod
    def _solve_ridge_weights(kernel_obs: np.ndarray, residual: np.ndarray, ridge: float) -> np.ndarray:
        lhs = kernel_obs.T @ kernel_obs + np.eye(kernel_obs.shape[1], dtype=np.float64) * max(float(ridge), 1.0e-9)
        rhs = kernel_obs.T @ residual
        return np.linalg.solve(lhs, rhs)

    def _reconstruct_field(
        self,
        scenario: dict[str, Any],
        field: list[dict[str, float]],
        sparse_points: list[dict[str, float]],
    ) -> list[dict[str, float]]:
        if not sparse_points:
            return field

        def point_key(point: dict[str, float]) -> tuple[float, float]:
            return (round(float(point['x']), 6), round(float(point['y']), 6))

        baseline_lookup = {point_key(point): point for point in field}
        sparse_lookup = {point_key(point): point for point in sparse_points}
        anchor_points = self._select_anchor_points(sparse_points, max_anchors=48)
        if not anchor_points:
            return [dict(point) for point in field]

        width_um = float(scenario['geometry']['wUm'])
        velocity_radius = max(width_um * 1.15, 36.0)
        pressure_radius = max(width_um * 1.65, 48.0)
        velocity_ridge = 2.0e-2
        pressure_ridge = 3.5e-2

        residuals: list[dict[str, float]] = []
        for point in sparse_points:
            baseline = baseline_lookup.get(point_key(point))
            if baseline is None:
                continue
            residuals.append(
                {
                    'x': float(point['x']),
                    'y': float(point['y']),
                    'dux': float(point['ux']) - float(baseline['ux']),
                    'duy': float(point['uy']) - float(baseline['uy']),
                    'dp': float(point['p']) - float(baseline['p']),
                }
            )

        if not residuals:
            return [dict(point) for point in field]

        sparse_xy = np.array([[item['x'], item['y']] for item in residuals], dtype=np.float64)
        anchor_xy = np.array([[float(point['x']), float(point['y'])] for point in anchor_points], dtype=np.float64)
        field_xy = np.array([[float(point['x']), float(point['y'])] for point in field], dtype=np.float64)

        vel_kernel_obs = self._rbf_kernel(sparse_xy, anchor_xy, velocity_radius)
        vel_kernel_field = self._rbf_kernel(field_xy, anchor_xy, velocity_radius)
        p_kernel_obs = self._rbf_kernel(sparse_xy, anchor_xy, pressure_radius)
        p_kernel_field = self._rbf_kernel(field_xy, anchor_xy, pressure_radius)

        dux_obs = np.array([item['dux'] for item in residuals], dtype=np.float64)
        duy_obs = np.array([item['duy'] for item in residuals], dtype=np.float64)
        dp_obs = np.array([item['dp'] for item in residuals], dtype=np.float64)

        dux_weights = self._solve_ridge_weights(vel_kernel_obs, dux_obs, velocity_ridge)
        duy_weights = self._solve_ridge_weights(vel_kernel_obs, duy_obs, velocity_ridge)
        dp_weights = self._solve_ridge_weights(p_kernel_obs, dp_obs, pressure_ridge)

        dux_field = vel_kernel_field @ dux_weights
        duy_field = vel_kernel_field @ duy_weights
        dp_field = p_kernel_field @ dp_weights
        vel_confidence = np.clip(np.sum(vel_kernel_field, axis=1) / 3.2, 0.0, 1.0)
        p_confidence = np.clip(np.sum(p_kernel_field, axis=1) / 3.8, 0.0, 1.0)

        reconstruction: list[dict[str, float]] = []
        for idx, target in enumerate(field):
            exact_match = sparse_lookup.get(point_key(target))
            if exact_match is not None:
                reconstruction.append(dict(exact_match))
                continue
            ux = float(target['ux']) + float(dux_field[idx]) * float(vel_confidence[idx])
            uy = float(target['uy']) + float(duy_field[idx]) * float(vel_confidence[idx])
            p = float(target['p']) + float(dp_field[idx]) * float(p_confidence[idx])
            reconstruction.append(
                {
                    **target,
                    'ux': float(ux),
                    'uy': float(uy),
                    'p': float(p),
                    'speed': float(math.hypot(ux, uy)),
                }
            )
        return reconstruction

    def _inverse_reconstruct_result(
        self,
        scenario: dict[str, Any],
        *,
        resolution: str = 'preview',
    ) -> dict[str, Any]:
        target_frame_star, _, _ = self._build_target_frame_star(scenario, resolution=resolution)
        candidate_frame_star = self._build_observation_candidate_frame_star(scenario, resolution=resolution)
        sparse_frame_star = self._sample_sparse_frame_star(scenario, target_frame_star, candidate_frame_star)

        sparse_truth_star = self._predict_frame_star(scenario, sparse_frame_star)
        sparse_truth = self._star_to_physical_points(scenario, sparse_truth_star)
        sparse_points = self._apply_sparse_noise(scenario, sparse_truth)

        prior_sparse_star = self._predict_frame_star(scenario, sparse_frame_star)
        prior_sparse = self._star_to_physical_points(scenario, prior_sparse_star)
        prior_field_star = self._predict_frame_star(scenario, target_frame_star)
        prior_field = self._star_to_physical_points(scenario, prior_field_star)

        prior_sparse_ux = np.array([point['ux'] for point in prior_sparse], dtype=np.float64)
        prior_sparse_uy = np.array([point['uy'] for point in prior_sparse], dtype=np.float64)
        prior_sparse_p = np.array([point['p'] for point in prior_sparse], dtype=np.float64)
        observed_ux = np.array([point['ux'] for point in sparse_points], dtype=np.float64)
        observed_uy = np.array([point['uy'] for point in sparse_points], dtype=np.float64)
        observed_p = np.array([point['p'] for point in sparse_points], dtype=np.float64)

        ux_scale, ux_bias = self._fit_affine_delta(prior_sparse_ux, observed_ux, scale_ridge=2.5e-2, bias_ridge=1.0e-8)
        uy_scale, uy_bias = self._fit_affine_delta(prior_sparse_uy, observed_uy, scale_ridge=2.5e-2, bias_ridge=1.0e-8)
        p_scale, p_bias = self._fit_affine_delta(prior_sparse_p, observed_p, scale_ridge=3.0e-2, bias_ridge=1.0e-8)

        corrected_sparse_ux = prior_sparse_ux * ux_scale + ux_bias
        corrected_sparse_uy = prior_sparse_uy * uy_scale + uy_bias
        corrected_sparse_p = prior_sparse_p * p_scale + p_bias

        residuals = [
            {
                'x': float(point['x']),
                'y': float(point['y']),
                'dux': float(observed_ux[idx] - corrected_sparse_ux[idx]),
                'duy': float(observed_uy[idx] - corrected_sparse_uy[idx]),
                'dp': float(observed_p[idx] - corrected_sparse_p[idx]),
            }
            for idx, point in enumerate(sparse_points)
        ]
        anchor_points = self._select_anchor_points(sparse_points, max_anchors=48)

        width_um = float(scenario['geometry']['wUm'])
        velocity_radius = max(width_um * 1.05, 30.0)
        pressure_radius = max(width_um * 1.55, 42.0)
        velocity_ridge = 1.8e-2
        pressure_ridge = 3.0e-2

        if anchor_points and residuals:
            sparse_xy = np.array([[item['x'], item['y']] for item in residuals], dtype=np.float64)
            anchor_xy = np.array([[float(point['x']), float(point['y'])] for point in anchor_points], dtype=np.float64)
            field_xy = np.array([[float(point['x']), float(point['y'])] for point in prior_field], dtype=np.float64)

            vel_kernel_obs = self._rbf_kernel(sparse_xy, anchor_xy, velocity_radius)
            vel_kernel_field = self._rbf_kernel(field_xy, anchor_xy, velocity_radius)
            p_kernel_obs = self._rbf_kernel(sparse_xy, anchor_xy, pressure_radius)
            p_kernel_field = self._rbf_kernel(field_xy, anchor_xy, pressure_radius)

            dux_obs = np.array([item['dux'] for item in residuals], dtype=np.float64)
            duy_obs = np.array([item['duy'] for item in residuals], dtype=np.float64)
            dp_obs = np.array([item['dp'] for item in residuals], dtype=np.float64)

            dux_weights = self._solve_ridge_weights(vel_kernel_obs, dux_obs, velocity_ridge)
            duy_weights = self._solve_ridge_weights(vel_kernel_obs, duy_obs, velocity_ridge)
            dp_weights = self._solve_ridge_weights(p_kernel_obs, dp_obs, pressure_ridge)

            dux_field = vel_kernel_field @ dux_weights
            duy_field = vel_kernel_field @ duy_weights
            dp_field = p_kernel_field @ dp_weights
            vel_confidence = np.clip(np.sum(vel_kernel_field, axis=1) / 3.4, 0.0, 1.0)
            p_confidence = np.clip(np.sum(p_kernel_field, axis=1) / 3.9, 0.0, 1.0)
        else:
            dux_field = np.zeros(len(prior_field), dtype=np.float64)
            duy_field = np.zeros(len(prior_field), dtype=np.float64)
            dp_field = np.zeros(len(prior_field), dtype=np.float64)
            vel_confidence = np.zeros(len(prior_field), dtype=np.float64)
            p_confidence = np.zeros(len(prior_field), dtype=np.float64)

        if 'wall_distance_star' in target_frame_star.columns:
            wall_distance = target_frame_star['wall_distance_star'].to_numpy(dtype=np.float64)
            wall_blend = np.clip(wall_distance / 0.08, 0.0, 1.0)
        else:
            wall_blend = np.ones(len(prior_field), dtype=np.float64)

        sparse_lookup = {self._point_key(point): point for point in sparse_points}
        reconstruction: list[dict[str, float]] = []
        for idx, prior_point in enumerate(prior_field):
            exact_match = sparse_lookup.get(self._point_key(prior_point))
            if exact_match is not None:
                reconstruction.append(dict(exact_match))
                continue
            ux = float(prior_point['ux']) * ux_scale + ux_bias * float(wall_blend[idx])
            uy = float(prior_point['uy']) * uy_scale + uy_bias * float(wall_blend[idx])
            p = float(prior_point['p']) * p_scale + p_bias
            ux += float(dux_field[idx]) * float(vel_confidence[idx]) * float(wall_blend[idx])
            uy += float(duy_field[idx]) * float(vel_confidence[idx]) * float(wall_blend[idx])
            p += float(dp_field[idx]) * float(p_confidence[idx])
            reconstruction.append(
                {
                    **prior_point,
                    'ux': float(ux),
                    'uy': float(uy),
                    'p': float(p),
                    'speed': float(math.hypot(ux, uy)),
                }
            )

        return {
            'field': prior_field,
            'sparsePoints': sparse_points,
            'reconstruction': reconstruction,
            'metrics': self._compute_metrics(scenario, reconstruction),
            'baselineMetrics': self._compute_metrics(scenario, prior_field),
        }

    @staticmethod
    def _compute_metrics(
        scenario: dict[str, Any],
        field: list[dict[str, float]],
        probes: dict[str, list[dict[str, float]]] | None = None,
    ) -> dict[str, float]:
        pressures = np.array([point['p'] for point in field], dtype=np.float64)
        speeds = np.array([point['speed'] for point in field], dtype=np.float64)
        avg_pressure_drop = float(np.max(pressures) - np.min(pressures)) if len(pressures) else 0.0
        centerline_gradient = 0.0
        if probes is not None:
            centerline = probes['mainCenterline']
            if len(centerline) >= 2:
                centerline_gradient = abs(centerline[0]['p'] - centerline[-1]['p']) / max(float(len(centerline)), 1.0)
        elif field:
            if scenario['geometry']['type'] == 'contraction':
                total_length = (
                    float(scenario['geometry']['lInOverW'])
                    + float(scenario['geometry']['lCOverW'])
                    + float(scenario['geometry']['lOutOverW'])
                ) * float(scenario['geometry']['wUm'])
            else:
                theta_rad = math.radians(float(scenario['geometry']['thetaDeg']))
                total_length = (
                    float(scenario['geometry']['lInOverW'])
                    + float(scenario['geometry']['rcOverW']) * theta_rad
                    + float(scenario['geometry']['lOutOverW'])
                ) * float(scenario['geometry']['wUm'])
            centerline_gradient = avg_pressure_drop / max(float(total_length), 1.0e-12)
        width_m = float(scenario['geometry']['wUm']) * 1.0e-6
        wall_shear_proxy = float(np.max(speeds) / max(width_m * 0.5, 1.0e-12)) if len(speeds) else 0.0
        if scenario['geometry']['type'] == 'bend':
            curvature_proxy = 1.0 / max(float(scenario['geometry']['rcOverW']), 1.0e-12)
        else:
            curvature_proxy = (1.0 - float(scenario['geometry']['beta'])) / max(float(scenario['geometry']['lCOverW']), 1.0e-12)
        return {
            'reynolds': ScenarioEngine._compute_reynolds(scenario),
            'maxSpeed': float(np.max(speeds)) if len(speeds) else 0.0,
            'avgPressureDrop': avg_pressure_drop,
            'wallShearProxy': wall_shear_proxy,
            'streamlineCurvatureProxy': float(curvature_proxy),
            'centerlinePressureGradient': float(centerline_gradient),
        }

    def simulate(
        self,
        scenario: dict[str, Any],
        *,
        resolution: str = 'full',
        include_streamlines: bool = True,
        include_probes: bool = True,
        include_sparse: bool = False,
        include_reconstruction: bool = False,
    ) -> dict[str, Any]:
        if scenario['geometry']['type'] == 'contraction':
            frame_star, _, _ = self._predict_contraction_field_star(scenario, resolution=resolution)
            if include_probes:
                main_curve, branch_curve = self._sample_contraction_curves(scenario)
            else:
                main_curve, branch_curve = [], []
        else:
            frame_star, _, _ = self._predict_bend_field_star(scenario, resolution=resolution)
            if include_probes:
                main_curve, branch_curve = self._sample_bend_curves(scenario)
            else:
                main_curve, branch_curve = [], []
        field = self._star_to_physical_points(scenario, frame_star)
        streamlines = None
        if include_streamlines:
            bundle = self._build_grid_bundle(scenario, frame_star, field)
            streamlines = self._build_streamlines(scenario, bundle)
        sparse_points = self._sample_sparse_points(scenario, frame_star, field) if include_sparse else None
        probes = None
        if include_probes:
            probes = {
                'mainCenterline': main_curve,
                'branchCenterline': branch_curve,
            }
        metrics_field = field
        reconstruction = None
        if include_reconstruction:
            reconstruction = self._reconstruct_field(scenario, field, sparse_points or [])
            metrics_field = reconstruction
        result: dict[str, Any] = {
            'field': field,
            'metrics': self._compute_metrics(scenario, metrics_field, probes),
        }
        if streamlines is not None:
            result['streamlines'] = streamlines
        if probes is not None:
            result['probes'] = probes
        if sparse_points is not None:
            result['sparsePoints'] = sparse_points
        if reconstruction is not None:
            result['reconstruction'] = reconstruction
        return result

    def query_point(self, scenario: dict[str, Any], point: dict[str, float]) -> dict[str, float] | None:
        pred = self._predict_single_point_star(scenario, point)
        if pred is None:
            return None
        return self._star_to_physical_points(scenario, pred)[0]

    def reconstruct(self, scenario: dict[str, Any]) -> dict[str, Any]:
        return self._inverse_reconstruct_result(scenario, resolution='preview')

    def streamlines(self, scenario: dict[str, Any], resolution: str = 'preview') -> dict[str, Any]:
        result = self.simulate(
            scenario,
            resolution=resolution,
            include_streamlines=True,
            include_probes=False,
            include_sparse=False,
            include_reconstruction=False,
        )
        return {
            'streamlines': result.get('streamlines', []),
        }

    def probes(self, scenario: dict[str, Any]) -> dict[str, Any]:
        if scenario['geometry']['type'] == 'contraction':
            main_curve, branch_curve = self._sample_contraction_curves(scenario)
        else:
            main_curve, branch_curve = self._sample_bend_curves(scenario)
        return {
            'probes': {
                'mainCenterline': main_curve,
                'branchCenterline': branch_curve,
            }
        }

    def calibrate_viscosity(self, scenario: dict[str, Any], target_points: list[dict[str, float]]) -> dict[str, float]:
        if not target_points:
            return {'bestViscosity': float(scenario['fluid']['viscosity']), 'error': 0.0}
        width_m = float(scenario['geometry']['wUm']) * 1.0e-6
        velocity = float(scenario['flow']['meanVelocity'])
        outlet_pressure = float(scenario['flow']['outletPressure'])
        a_terms: list[float] = []
        b_terms: list[float] = []
        for point in target_points:
            pred = self._predict_single_point_star(scenario, {'x': float(point['x']), 'y': float(point['y'])})
            if pred is None:
                continue
            p_star = float(pred.iloc[0]['p_star'])
            a = p_star * velocity / max(width_m, 1.0e-12)
            b = float(point['p']) - outlet_pressure
            a_terms.append(a)
            b_terms.append(b)
        if not a_terms:
            return {'bestViscosity': float(scenario['fluid']['viscosity']), 'error': 0.0}
        a_vec = np.array(a_terms, dtype=np.float64)
        b_vec = np.array(b_terms, dtype=np.float64)
        denom = float(np.dot(a_vec, a_vec))
        if denom <= 1.0e-12:
            best_mu = float(scenario['fluid']['viscosity'])
        else:
            best_mu = float(np.dot(a_vec, b_vec) / denom)
        best_mu = self._clamp(best_mu, MIN_VISCOSITY, MAX_VISCOSITY)
        error = float(np.sqrt(np.mean((a_vec * best_mu - b_vec) ** 2)))
        return {'bestViscosity': best_mu, 'error': error}

    def sweep(self, scenario: dict[str, Any], variable: str, values: list[float]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for value in values:
            next_scenario = json.loads(json.dumps(scenario))
            if variable == 'meanVelocity':
                next_scenario['flow']['meanVelocity'] = float(value)
            elif variable == 'viscosity':
                next_scenario['fluid']['viscosity'] = float(value)
                next_scenario['fluid']['preset'] = 'custom'
            else:
                raise ValueError(f'Unsupported sweep variable: {variable}')
            result = self.simulate(next_scenario, include_sparse=False, include_reconstruction=False)
            results.append({'value': float(value), 'metrics': result['metrics']})
        return results


ENGINE = ScenarioEngine()
RESPONSE_CACHE = ResponseCache()


def _extract_input_and_options(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(payload, dict) and 'input' in payload:
        return payload['input'], payload.get('options', {}) or {}
    return payload, {}


class RequestHandler(BaseHTTPRequestHandler):
    server_version = 'PinnV4API/0.1'

    def _send_body(self, status: int, body: bytes, cache_status: str | None = None) -> None:
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        if cache_status:
            self.send_header('X-Response-Cache', cache_status)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            return

    def _send_json(self, status: int, payload: Any, cache_status: str | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self._send_body(status, body, cache_status=cache_status)

    def _read_raw_body(self) -> bytes:
        length = int(self.headers.get('Content-Length', '0'))
        return self.rfile.read(length) if length > 0 else b'{}'

    @staticmethod
    def _load_json(raw: bytes) -> Any:
        return json.loads(raw.decode('utf-8'))

    @staticmethod
    def _build_cache_key(path: str, raw: bytes) -> str:
        digest = hashlib.sha256()
        digest.update(path.encode('utf-8'))
        digest.update(b'\n')
        digest.update(raw)
        return digest.hexdigest()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip('/') or '/'
        if path == '/healthz':
            self._send_json(
                200,
                {
                    'ok': True,
                    'service': 'pinn-flow-visual-demo-v4-api',
                    'contraction_run': ENGINE.contraction_runtime.run_name,
                    'bend_parabolic_run': ENGINE.bend_parabolic_runtime.run_name,
                    'bend_skewed_top_run': ENGINE.bend_skewed_top_runtime.run_name,
                    'bend_skewed_bottom_run': ENGINE.bend_skewed_bottom_runtime.run_name,
                    'bend_blunted_run': ENGINE.bend_blunted_runtime.run_name,
                },
            )
            return
        self._send_json(404, {'error': 'Not found'})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip('/')
        try:
            raw = self._read_raw_body()
            cacheable = path in {'/simulate', '/reconstruct', '/sweep', '/streamlines', '/probes'}
            if cacheable:
                cache_key = self._build_cache_key(path, raw)
                cached = RESPONSE_CACHE.get(cache_key)
                if cached is not None:
                    self._send_body(200, cached, cache_status='HIT')
                    return
            payload = self._load_json(raw)
            if path == '/simulate':
                scenario, options = _extract_input_and_options(payload)
                body = json.dumps(
                    ENGINE.simulate(
                        scenario,
                        resolution=str(options.get('resolution', 'full')),
                        include_streamlines=bool(options.get('includeStreamlines', True)),
                        include_probes=bool(options.get('includeProbes', True)),
                        include_sparse=bool(options.get('includeSparsePoints', False)),
                        include_reconstruction=bool(options.get('includeReconstruction', False)),
                    ),
                    ensure_ascii=False,
                ).encode('utf-8')
                if cacheable:
                    RESPONSE_CACHE.set(cache_key, body)
                self._send_body(200, body, cache_status='MISS' if cacheable else None)
                return
            if path == '/query-point':
                self._send_json(200, ENGINE.query_point(payload['input'], payload['point']))
                return
            if path == '/reconstruct':
                scenario, _ = _extract_input_and_options(payload)
                body = json.dumps(ENGINE.reconstruct(scenario), ensure_ascii=False).encode('utf-8')
                if cacheable:
                    RESPONSE_CACHE.set(cache_key, body)
                self._send_body(200, body, cache_status='MISS' if cacheable else None)
                return
            if path == '/streamlines':
                scenario, options = _extract_input_and_options(payload)
                body = json.dumps(
                    ENGINE.streamlines(scenario, resolution=str(options.get('resolution', 'preview'))),
                    ensure_ascii=False,
                ).encode('utf-8')
                if cacheable:
                    RESPONSE_CACHE.set(cache_key, body)
                self._send_body(200, body, cache_status='MISS' if cacheable else None)
                return
            if path == '/probes':
                scenario, _ = _extract_input_and_options(payload)
                body = json.dumps(ENGINE.probes(scenario), ensure_ascii=False).encode('utf-8')
                if cacheable:
                    RESPONSE_CACHE.set(cache_key, body)
                self._send_body(200, body, cache_status='MISS' if cacheable else None)
                return
            if path == '/calibrate-viscosity':
                self._send_json(200, ENGINE.calibrate_viscosity(payload['input'], payload['targetPoints']))
                return
            if path == '/sweep':
                body = json.dumps(
                    ENGINE.sweep(payload['input'], payload['variable'], payload['values']),
                    ensure_ascii=False,
                ).encode('utf-8')
                if cacheable:
                    RESPONSE_CACHE.set(cache_key, body)
                self._send_body(200, body, cache_status='MISS' if cacheable else None)
                return
            self._send_json(404, {'error': 'Not found'})
        except BrokenPipeError:
            return
        except Exception as exc:  # pragma: no cover
            self._send_json(500, {'error': str(exc)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Minimal PINN v4 HTTP API for the visual demo site')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8011)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f'[pinn-v4-api] listening on http://{args.host}:{args.port} (run={ENGINE.contraction_runtime.run_name})', flush=True)
    httpd.serve_forever()


if __name__ == '__main__':
    main()
