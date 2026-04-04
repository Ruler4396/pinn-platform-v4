"""Case registry for bend_2d experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, List

import numpy as np


SUPPORTED_INLET_PROFILES = ("parabolic", "blunted", "skewed_top", "skewed_bottom")
VARIANT_TOKEN = "__ip_"


@dataclass(frozen=True)
class BendCase:
    case_id: str
    rc_over_w: float
    theta_deg: float
    family: str = "bend"
    w_um: float = 200.0
    l_in_over_w: float = 4.0
    l_out_over_w: float = 6.0
    u_mean_mm_s: float = 0.1
    rho: float = 997.05
    mu: float = 8.902e-4
    nu: float = 8.928e-7
    inlet_profile_name: str = "parabolic"
    note: str = ""

    @property
    def theta_rad(self) -> float:
        return self.theta_deg * 3.141592653589793 / 180.0

    @property
    def arc_length_over_w(self) -> float:
        return self.rc_over_w * self.theta_rad

    @property
    def total_centerline_length_over_w(self) -> float:
        return self.l_in_over_w + self.arc_length_over_w + self.l_out_over_w

    @property
    def hydraulic_diameter_um(self) -> float:
        return self.w_um

    @property
    def reynolds_number(self) -> float:
        u_mean_m_s = self.u_mean_mm_s * 1.0e-3
        d_h_m = self.hydraulic_diameter_um * 1.0e-6
        return self.rho * u_mean_m_s * d_h_m / self.mu

    def to_metadata(self) -> dict:
        payload = asdict(self)
        payload["geometry"] = {
            "W_um": self.w_um,
            "L_in_over_W": self.l_in_over_w,
            "R_c_over_W": self.rc_over_w,
            "theta_deg": self.theta_deg,
            "arc_length_over_W": self.arc_length_over_w,
            "L_out_over_W": self.l_out_over_w,
            "total_centerline_length_over_W": self.total_centerline_length_over_w,
        }
        payload["fluid"] = {
            "rho": self.rho,
            "mu": self.mu,
            "nu": self.nu,
        }
        payload["flow"] = {
            "u_mean_mm_s": self.u_mean_mm_s,
            "Re": self.reynolds_number,
        }
        payload["inlet_profile"] = {
            "name": self.inlet_profile_name,
            "description": inlet_profile_description(self.inlet_profile_name),
        }
        return payload


_CASE_LIBRARY: Dict[str, BendCase] = {
    "B-base": BendCase(case_id="B-base", rc_over_w=6.0, theta_deg=90.0, note="单工况基线"),
    "B-train-1": BendCase(case_id="B-train-1", rc_over_w=6.0, theta_deg=90.0, note="低曲率训练"),
    "B-train-2": BendCase(case_id="B-train-2", rc_over_w=5.0, theta_deg=90.0, note="低-中曲率训练"),
    "B-train-3": BendCase(case_id="B-train-3", rc_over_w=4.0, theta_deg=90.0, note="中曲率训练"),
    "B-val": BendCase(case_id="B-val", rc_over_w=3.5, theta_deg=90.0, note="验证集"),
    "B-test-1": BendCase(case_id="B-test-1", rc_over_w=3.0, theta_deg=90.0, note="高曲率挑战"),
    "B-test-2": BendCase(case_id="B-test-2", rc_over_w=4.0, theta_deg=60.0, note="转角泛化扩展"),
}


def get_case(case_id: str) -> BendCase:
    if case_id in _CASE_LIBRARY:
        return _CASE_LIBRARY[case_id]
    if VARIANT_TOKEN in case_id:
        base_case_id, profile_name = case_id.split(VARIANT_TOKEN, 1)
        if base_case_id in _CASE_LIBRARY and profile_name in SUPPORTED_INLET_PROFILES:
            base = _CASE_LIBRARY[base_case_id]
            return replace(
                base,
                case_id=case_id,
                inlet_profile_name=profile_name,
                note=f"{base.note} + inlet_profile={profile_name}",
            )
    try:
        return _CASE_LIBRARY[case_id]
    except KeyError as exc:
        known = ", ".join(sorted(_CASE_LIBRARY))
        raise KeyError(f"Unknown bend case '{case_id}'. Known cases: {known}") from exc


def list_cases() -> List[BendCase]:
    return [
        _CASE_LIBRARY[key]
        for key in sorted(_CASE_LIBRARY, key=lambda item: (
            item.replace("B-base", "B-00"),
            item,
        ))
    ]


def inlet_profile_description(name: str) -> str:
    mapping = {
        "parabolic": "baseline symmetric parabolic profile",
        "blunted": "symmetric blunted quartic profile with flatter core",
        "skewed_top": "asymmetric profile biased toward positive eta / top wall",
        "skewed_bottom": "asymmetric profile biased toward negative eta / bottom wall",
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported inlet profile '{name}'. Supported: {SUPPORTED_INLET_PROFILES}") from exc


def build_variant_case_id(base_case_id: str, inlet_profile_name: str) -> str:
    if inlet_profile_name == "parabolic":
        return base_case_id
    if inlet_profile_name not in SUPPORTED_INLET_PROFILES:
        raise ValueError(f"Unsupported inlet profile '{inlet_profile_name}'. Supported: {SUPPORTED_INLET_PROFILES}")
    return f"{base_case_id}{VARIANT_TOKEN}{inlet_profile_name}"


def evaluate_inlet_profile(eta: np.ndarray | float, half_width: float, profile_name: str) -> np.ndarray:
    eta_arr = np.asarray(eta, dtype=np.float64)
    eta_norm = np.clip(eta_arr / max(float(half_width), 1.0e-12), -1.0, 1.0)
    if profile_name == "parabolic":
        profile = 1.5 * (1.0 - eta_norm**2)
    elif profile_name == "blunted":
        profile = 1.25 * (1.0 - eta_norm**4)
    elif profile_name == "skewed_top":
        profile = 1.5 * (1.0 - eta_norm**2) * (1.0 + 0.4 * eta_norm)
    elif profile_name == "skewed_bottom":
        profile = 1.5 * (1.0 - eta_norm**2) * (1.0 - 0.4 * eta_norm)
    else:
        raise ValueError(f"Unsupported inlet profile '{profile_name}'. Supported: {SUPPORTED_INLET_PROFILES}")
    return np.maximum(0.0, profile)
