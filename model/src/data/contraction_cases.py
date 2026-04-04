"""Case registry for contraction_2d experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ContractionCase:
    case_id: str
    beta: float
    lc_over_w: float
    family: str = "contraction"
    w_um: float = 200.0
    l_in_over_w: float = 4.0
    l_out_over_w: float = 8.0
    u_mean_mm_s: float = 0.1
    rho: float = 997.05
    mu: float = 8.902e-4
    nu: float = 8.928e-7
    note: str = ""

    @property
    def total_length_over_w(self) -> float:
        return self.l_in_over_w + self.lc_over_w + self.l_out_over_w

    @property
    def reynolds_number(self) -> float:
        u_mean_m_s = self.u_mean_mm_s * 1.0e-3
        w_m = self.w_um * 1.0e-6
        return self.rho * u_mean_m_s * w_m / self.mu

    def to_metadata(self) -> dict:
        payload = asdict(self)
        payload["geometry"] = {
            "W_um": self.w_um,
            "beta": self.beta,
            "L_in_over_W": self.l_in_over_w,
            "L_c_over_W": self.lc_over_w,
            "L_out_over_W": self.l_out_over_w,
            "total_length_over_W": self.total_length_over_w,
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
        return payload


_CASE_LIBRARY: Dict[str, ContractionCase] = {
    "C-base": ContractionCase(case_id="C-base", beta=0.70, lc_over_w=4.0, note="单工况基线"),
    "C-train-1": ContractionCase(case_id="C-train-1", beta=0.85, lc_over_w=4.0, note="弱收缩训练"),
    "C-train-2": ContractionCase(case_id="C-train-2", beta=0.70, lc_over_w=4.0, note="中等收缩训练"),
    "C-train-3": ContractionCase(case_id="C-train-3", beta=0.50, lc_over_w=4.0, note="强收缩训练"),
    "C-train-4": ContractionCase(case_id="C-train-4", beta=0.85, lc_over_w=6.0, note="长平滑收缩训练"),
    "C-train-5": ContractionCase(case_id="C-train-5", beta=0.70, lc_over_w=6.0, note="中等收缩+长过渡训练"),
    "C-val": ContractionCase(case_id="C-val", beta=0.50, lc_over_w=6.0, note="验证集"),
    "C-test-1": ContractionCase(case_id="C-test-1", beta=0.60, lc_over_w=5.0, note="插值泛化"),
    "C-test-2": ContractionCase(case_id="C-test-2", beta=0.40, lc_over_w=6.0, note="外推挑战"),
}


def get_case(case_id: str) -> ContractionCase:
    try:
        return _CASE_LIBRARY[case_id]
    except KeyError as exc:
        known = ", ".join(sorted(_CASE_LIBRARY))
        raise KeyError(f"Unknown contraction case '{case_id}'. Known cases: {known}") from exc


def list_cases() -> List[ContractionCase]:
    return [
        _CASE_LIBRARY[key]
        for key in sorted(_CASE_LIBRARY, key=lambda item: (
            item.replace("C-base", "C-00"),
            item,
        ))
    ]
