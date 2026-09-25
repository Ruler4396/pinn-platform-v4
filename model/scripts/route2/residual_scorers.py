"""Momentum/continuity residual scorers and the frozen K0 / mesh-independence gates.

Pure stdlib: this module is the arithmetic referee. It is imported by the local
self-test (no numpy/torch needed) and by the on-instance gate script.

Two scorers are kept side by side on purpose:

* ``as_found_v4_terms``  -- what pinn-platform-v4 currently optimises
  (``train_velocity_pressure_independent.py:369-383``): the viscous term is
  divided by ``max||u||`` and the pressure-gradient term by ``max(p)-min(p)``,
  so a field that satisfies Stokes exactly scores far *worse* than a flattened
  one.  Reproduced here so the fix is aimed at a measured defect, not a story.
* ``corrected_terms`` -- star units, one common derivative scale for both sides.
  Because the case is nondimensionalised with W_stem and U_inlet (mu = 1), the
  momentum equation reads ``lap(u) = grad(p)`` and the only admissible scale is
  ``U*/W*^2``.  Nothing is fitted from data, so the strict no-dense-leakage
  protocol is satisfied by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------- frozen gates
K0_TRUTH_OVER_MODEL_MAX = 10.0        # score(truth) / score(converged reading)
K0_CHAIN_FIRST_REL_MAX = 0.02         # d/dx* total derivative vs analytic
K0_CHAIN_SECOND_REL_MAX = 0.05        # lap / grad vs analytic
K0_FD_STEP_HALVE_MAX = 0.30           # residual estimate change when FD step halves
K0_BALANCE_RATIO_LO = 1.0 / 1.2       # rms(lap u) / rms(grad p) for a Stokes field
K0_BALANCE_RATIO_HI = 1.2
K0_ABS_RESIDUAL_MAX = 1.0             # star units: same order as the terms themselves
MESH_INDEPENDENCE_REL_MAX = 0.10      # autopsy P1
MASS_CLOSURE_REL_TOL = 1.0e-3         # |Q_in - Q_up - Q_down| / Q_in, steady incompressible
SYMMETRY_TOL = 0.02                   # |Q_up/Q_in - 0.5| for a geometrically symmetric T
BEND_SCALES_FOR_AUDIT = (
    # (label, U=max||u||, P=max(p)-min(p), rms(lap u)) from the 2026-09-25 audit
    ("contraction mainline", 3.0000, 985.253, 70.869),
    ("contraction strict_sparse", 0.50022, 245.894, 70.869),
)
AUDIT_AS_FOUND_REFERENCE = {"contraction mainline": 554.6, "contraction strict_sparse": 19990.0}


# ------------------------------------------------------------------ scorers
@dataclass(frozen=True)
class ResidualTerms:
    momentum: float
    continuity: float
    balance_ratio: float          # rms(lap)/rms(grad p); 1.0 <=> the two sides balance
    detail: Dict[str, float]


def corrected_terms(lap_rms: float, gradp_rms: float, div_rms: float,
                    velocity_scale: float = 1.0, length_scale: float = 1.0) -> ResidualTerms:
    """One common scale on both sides of the momentum equation (star units).

    Momentum:  lap(u) - grad(p) = 0, both terms carry U*/W*^2.
    Continuity: div(u) = 0, scale U*/W*.
    With W* = U* = 1 both scales are exactly 1, so nothing is read from data.
    """
    l = max(abs(length_scale), 1.0e-30)
    s_m = velocity_scale / l ** 2
    s_c = velocity_scale / l
    return ResidualTerms(
        momentum=((lap_rms - gradp_rms) / s_m) ** 2,
        continuity=(div_rms / s_c) ** 2,
        balance_ratio=_safe_ratio(lap_rms, gradp_rms),
        detail={"lap_rms": lap_rms, "gradp_rms": gradp_rms, "div_rms": div_rms,
                "momentum_scale": s_m, "continuity_scale": s_c},
    )


def as_found_v4_momentum(lap_rms: float, gradp_rms: float,
                         velocity_scale: float, pressure_scale: float) -> float:
    """The u-component of what v4 currently optimises (:369-383).

    ``lap/V - gradp/P`` with V = max||u|| and P = max(p)-min(p).  For a
    Stokes-balanced field lap ~= gradp pointwise, so this collapses to
    lap^2 (1/V - 1/P)^2 -- minimised by flattening the profile, not by satisfying
    the equation.  Single component, matching the 2026-09-25 audit's convention.
    """
    v = max(abs(velocity_scale), 1.0e-12)
    p = max(abs(pressure_scale), 1.0e-12)
    return (lap_rms / v - gradp_rms / p) ** 2


def _safe_ratio(a: float, b: float) -> float:
    return a / max(b, 1.0e-30)


# ------------------------------------------------------ manufactured controls
@dataclass(frozen=True)
class Poiseuille:
    """Exact 2-D Stokes solution in a straight branch frame.

    u(eta) = 1.5*U*(1-(eta/h)^2) along the branch direction, p = p_ref - (3U/h^2)*xi.
    lap(u) = -3U/h^2 = grad(p) componentwise, so the true residual is zero.
    """

    mean_velocity: float
    half_width: float

    @property
    def grad_p(self) -> float:
        return -3.0 * self.mean_velocity / (self.half_width ** 2)

    def axial_velocity(self, eta: float) -> float:
        return 1.5 * self.mean_velocity * (1.0 - (eta / self.half_width) ** 2)

    def laplacian(self) -> float:
        return self.grad_p

    def pressure(self, xi: float, p_ref: float = 0.0) -> float:
        return p_ref + self.grad_p * xi

    def peak_speed(self) -> float:
        return 1.5 * self.mean_velocity


def poiseuille_as_found_score(sol: Poiseuille, xi_span: float) -> float:
    """as-found momentum score of an *exact* Stokes solution (the failure mode)."""
    lap = abs(sol.laplacian())
    gradp = abs(sol.grad_p)
    v = sol.peak_speed()
    p = abs(sol.grad_p) * max(xi_span, 1.0e-12)
    return as_found_v4_momentum(lap, gradp, v, p)


# --------------------------------------------------------------- gate helpers
def mesh_independence_gate(values_by_level: Dict[str, Sequence[float]],
                           limit: float = MESH_INDEPENDENCE_REL_MAX) -> dict:
    """values_by_level[name] = [level1, level2, ...] ordered coarse -> fine.

    Verdict uses the *last* refinement pair (the finest comparison available);
    every pair is reported so a reviewer can re-check the arithmetic.
    """
    out: Dict[str, dict] = {}
    worst = 0.0
    worst_name = ""
    for name, vals in values_by_level.items():
        if len(vals) < 2:
            out[name] = {"ok": False, "reason": "need >=2 refinement levels", "pairs": []}
            worst = math.inf
            worst_name = name
            continue
        pairs = []
        for i in range(len(vals) - 1):
            rel = relative_change(vals[i], vals[i + 1])
            pairs.append({"from_level": i + 1, "to_level": i + 2, "rel_change": rel})
        last = pairs[-1]["rel_change"]
        out[name] = {"ok": last < limit, "last_rel_change": last, "pairs": pairs}
        if last > worst:
            worst, worst_name = last, name
    return {
        "limit": limit,
        "quantities": out,
        "worst_rel_change": worst,
        "worst_quantity": worst_name,
        "pass": bool(out) and worst < limit,
    }


def relative_change(coarse: float, fine: float, floor: float = 1.0e-12) -> float:
    return abs(coarse - fine) / max(abs(fine), floor)


def fd_step_convergence_gate(estimates: Sequence[float],
                             limit: float = K0_FD_STEP_HALVE_MAX) -> dict:
    """estimates ordered coarse-step -> fine-step. Guarded: fewer than 2 -> INVALID."""
    if len(estimates) < 2:
        return {"pass": False, "reason": "need >=2 FD step sizes", "rel_change": math.inf}
    rel = relative_change(estimates[0], estimates[-1])
    return {"pass": bool(math.isfinite(rel) and rel < limit), "rel_change": rel, "limit": limit}


def absolute_gates(q_in: float, q_up: float, q_down: float,
                   flux_conservation_max: float = float("nan")) -> dict:
    """Non-relative gates: mass closure and the symmetry sanity check.

    A near-zero residual cannot be judged by relative change across meshes, so it
    gets its own absolute threshold here instead of sitting in the <10% gate.
    """
    q_in = abs(q_in)
    if not (math.isfinite(q_in) and q_in > 0.0):
        return {"pass": False, "reason": "q_in not a positive finite number", "q_in": q_in}
    closure = abs(q_in - q_up - q_down) / q_in
    split = q_up / q_in
    out = {
        "mass_closure": {"value": closure, "limit": MASS_CLOSURE_REL_TOL,
                         "pass": bool(closure <= MASS_CLOSURE_REL_TOL)},
        "symmetry_of_split": {"value": abs(split - 0.5), "limit": SYMMETRY_TOL,
                              "pass": bool(abs(split - 0.5) <= SYMMETRY_TOL)},
        "flux_conservation_along_branch": {
            "value": flux_conservation_max, "limit": MASS_CLOSURE_REL_TOL,
            "pass": bool(math.isfinite(flux_conservation_max)
                         and flux_conservation_max <= MASS_CLOSURE_REL_TOL)},
    }
    out["pass"] = all(bool(v.get("pass")) for v in out.values())
    return out


def k0_verdict(truth_score: float, model_score: float,
               chain: Dict[str, Dict[str, float]],
               fd_gate: dict, balance_ratio: float) -> dict:
    """Assemble the frozen K0 gate.  Any missing measurement => INVALID, never True."""
    ratio = (truth_score / model_score) if model_score > 0.0 else math.inf
    checks = {
        "K0-S1_truth_over_model<=10x": {
            "value": ratio, "limit": K0_TRUTH_OVER_MODEL_MAX,
            "pass": bool(math.isfinite(ratio) and ratio <= K0_TRUTH_OVER_MODEL_MAX),
        },
        "K0-S2_balance_ratio_in_band": {
            "value": balance_ratio, "band": [K0_BALANCE_RATIO_LO, K0_BALANCE_RATIO_HI],
            "pass": bool(math.isfinite(balance_ratio)
                         and K0_BALANCE_RATIO_LO <= balance_ratio <= K0_BALANCE_RATIO_HI),
        },
        "K0-S3_abs_residual_star_units<=1": {
            "value": truth_score, "limit": K0_ABS_RESIDUAL_MAX,
            "pass": bool(truth_score <= K0_ABS_RESIDUAL_MAX),
        },
        "K0-S4_fd_step_converged": fd_gate,
    }
    for plan, errs in chain.items():
        for order, value in errs.items():
            limit = K0_CHAIN_FIRST_REL_MAX if order.startswith("first") else K0_CHAIN_SECOND_REL_MAX
            checks[f"K0-C_{plan}_{order}"] = {
                "value": value, "limit": limit,
                "pass": bool(math.isfinite(value) and value <= limit),
            }
    all_pass = all(bool(c.get("pass")) for c in checks.values())
    return {"pass": all_pass, "checks": checks,
            "failed": sorted(k for k, c in checks.items() if not c.get("pass")),
            "note": "thresholds frozen 2026-09-25 before any route-2 run; do not relax"}


# ------------------------------------------------------- audit reproduction
def audit_reproduction() -> List[dict]:
    """Recompute the 2026-09-25 audit's 'truth scores worse than the model' numbers."""
    rows = []
    for label, v, p, lap in BEND_SCALES_FOR_AUDIT:
        as_found = as_found_v4_momentum(lap, lap, v, p)
        rows.append({
            "run": label,
            "velocity_scale": v,
            "pressure_scale": p,
            "imbalance_P_over_U": p / v,
            "as_found_exact_stokes_score": as_found,
            "audit_reference": AUDIT_AS_FOUND_REFERENCE[label],
            "rel_err_vs_audit": relative_change(as_found, AUDIT_AS_FOUND_REFERENCE[label]),
            "corrected_score": corrected_terms(lap, lap, 0.0).momentum,
        })
    return rows
