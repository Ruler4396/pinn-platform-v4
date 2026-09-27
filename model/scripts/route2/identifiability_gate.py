#!/usr/bin/env python3
"""gateA identifiability gate: how fine a defect can THIS rig tell apart?

Not "is the prediction accurate" -- that is K0/S2/S3, and K0 has not passed, so those
arms stay frozen.  This module answers a different question and unlocks no accuracy claim.

Pre-registration: paper-route2/格A可辨识性-预注册-20260927.md (written before any number
below was computed; that file is frozen and this module implements its clauses).  The
governing rulings are §十.3(a-c) + §十二.1 of T6Re实例复验-1eba988-20260926.md.

Three hard rules this module is built to obey, each as a mechanism rather than a promise:

1. RANK BY ELIMINATION ONLY.  rank_of() is the single entry point and it delegates to
   impedance_baseline.rank_by_elimination (row-pivoted Gaussian elimination).  There is
   no code path in which a rank claim comes from a spectrum: the first version of this
   line of work used an eigensolver, got a wrong spectrum on a repeated-root matrix
   (all-ones 4x4 came out [3.33, 0.67, 1.9e-4, 0] instead of [4, 0, 0, 0]) and thereby
   dressed up a WRONG PHYSICAL CLAIM ("node data is full rank") as numerical support.
2. NO EIGENDECOMPOSITION AVAILABLE AT ALL.  The import allow-list below excludes
   numpy/scipy/torch, so a decomposition cannot be reached even by accident.  sigma_min
   comes from one-sided Jacobi orthogonalisation of the columns of the rectangular
   Jacobian itself -- J^T J is never formed, which matters because squaring the condition
   number is exactly where clustered/ repeated singular values get mis-resolved.
3. THE THRESHOLD IS A FORMULA, NOT A MEASURED VALUE.  tau(sigma_rel) = sigma_rel /
   DELTA_TARGET takes only declared quantities.  The plateau region of the pre-fixed 2x2
   resolution matrix does NOT supply tau (that would be picking a number to pass a gate);
   it licenses which cells may be quoted at all.

Model: the stem's effective width is a K-element piecewise-constant profile (plus w_up,
w_dn, kappa), which is what "how fine" has to mean here -- a single per-branch width has
no "fine" to lose.  Everything is 1-D star-unit Poiseuille, shared with the read-only
reference module, and the self-test pins that the two forward maps agree.

Run:  python model/scripts/route2/identifiability_gate.py [--json OUT]   (OUT outside repo)
Exit: 0 = every requested claim is supported, 1 = a claim was refused (GateError), 3 = the
gate itself could not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import impedance_baseline as ib                       # noqa: E402  READ-ONLY reuse

# ---------------------------------------------------------------- pre-registered terms
K_STEM_ELEMENTS = 8          # pre-reg §3.1; no sensitivity ladder is claimed (see §十一)
DELTA_TARGET = 0.10          # pre-reg §5: the repo's own registered defect amplitude
                             # (impedance_baseline.station_data_with_distributed_defect
                             # defaults defect=0.10), not a number chosen to pass
PLATEAU_REL_TOL = 0.10       # pre-reg §4, verbatim from §十.3(a): "平台" = <10% drift
DETECTION_K = 1.0            # pre-reg §5: a definition, not a knob.  Any factor >1 would
                             # be free and would need a false-alarm budget we do not have
NULL_PROBE_TOL = 1.0e-9      # pre-reg §9.2: inherited from impedance_baseline
                             # .flat_direction_probe ("along flat < 1e-9 <= transversal");
                             # this gate adds no new constant
FD_STEP = 1.0e-5             # matches impedance_baseline.jacobian_wrt_params
PRINT_SIG = 6                # the repo's 6-significant-digit print width
N_P_MATRIX = (16, 40)        # §十.3(a)'s station axis
SIGMA_P_KPAS = (1.0, 0.1)    # §十.3(a)'s noise axis; absolute, see FULL_SCALE below
NOISE_TIER_MAIN = ib.NOISE_FLOOR_DEFAULT      # 3%: pre-registered, needs no citation
NOISE_TIER_QUIET = 0.01                        # 1%: requires instrument provenance
RANK_SOURCE = "gaussian_elimination_partial_pivot@impedance_baseline.rank_by_elimination"
PARAM_NAMES_D = tuple(f"w_stem_{i + 1}" for i in range(K_STEM_ELEMENTS)) + ("w_up", "w_dn",
                                                                            "kappa")
N_PARAMS = len(PARAM_NAMES_D)
# the kPa -> relative conversion needs a device full-scale pressure drop.  G1 of
# paper-route2/观测噪声出处-20260926.md: nothing citable exists in-repo, so the two kPa
# cells report sigma_bar_min and the plateau (both scale-free) but their delta_resolve is
# UNMEASURED.  This is a missing input, not a folded-away arm.
FULL_SCALE_DP_KPA: Optional[float] = None
BANNED_IMPORT_ROOTS = ("numpy", "scipy", "torch")
IMPORT_ALLOW_LIST = ("__future__", "argparse", "hashlib", "json", "math", "sys", "time",
                     "pathlib", "typing", "impedance_baseline")


class GateError(RuntimeError):
    """A claim the data does not support, or a gate precondition that broke."""


# --------------------------------------------------------------------- forward map
def theta_from_uniform(widths: Sequence[float], w_up: float, w_dn: float,
                       kappa: float) -> List[float]:
    return [float(w) for w in widths] + [w_up, w_dn, kappa]


def truth_theta(theta_node: Sequence[float], k: int = K_STEM_ELEMENTS) -> List[float]:
    """Spread the reference 4-parameter truth (w_stem, w_up, w_dn, kappa) onto K elements."""
    return theta_from_uniform([theta_node[0]] * k, theta_node[1], theta_node[2], theta_node[3])


def stem_element_resistances(theta_dist: Sequence[float], lengths: Dict[str, float]
                             ) -> List[float]:
    k = len(theta_dist) - 3
    seg = lengths["stem"] / k
    return [12.0 * seg / max(theta_dist[i] ** 3, 1.0e-12) for i in range(k)]


def solve_distributed(theta_dist: Sequence[float], lengths: Dict[str, float],
                      p_in: float) -> dict:
    """Closed-form two-node solution for the profile version of the same network.

    Only R_stem changes (a sum over elements instead of 12L/w^3), so the node algebra is
    the read-only reference's -- which is what lets the self-test pin the two maps
    together on a uniform profile.
    """
    r_el = stem_element_resistances(theta_dist, lengths)
    r_stem = sum(r_el)
    w_up, w_dn, kappa = theta_dist[-3], theta_dist[-2], theta_dist[-1]
    r_up = 12.0 * lengths["up"] / max(w_up ** 3, 1.0e-12)
    r_dn = 12.0 * lengths["down"] / max(w_dn ** 3, 1.0e-12)
    r_series = r_stem * (1.0 + kappa)
    y_tot = 1.0 / r_series + 1.0 / r_up + 1.0 / r_dn
    p_j = p_in * (1.0 / r_series) / y_tot
    q_in = (p_in - p_j) / r_series
    return {"r_el": r_el, "R_stem": r_stem, "R_up": r_up, "R_dn": r_dn,
            "R_series": r_series, "p_junction": p_j, "q_in": q_in,
            "q_up": p_j / r_up, "q_down": p_j / r_dn,
            "split_up": (p_j / r_up) / q_in if q_in else float("nan")}


def stem_pressure(theta_dist: Sequence[float], lengths: Dict[str, float], p_in: float,
                  frac: float) -> float:
    """Analytic piecewise version of station_data_with_distributed_defect's quadrature."""
    sol = solve_distributed(theta_dist, lengths, p_in)
    k = len(theta_dist) - 3
    d = frac * k
    drop = 0.0
    for i in range(k):
        overlap = min(1.0, max(0.0, d - i))          # element i is [i/k, (i+1)/k)
        if overlap <= 0.0:
            break
        drop += overlap * sol["r_el"][i]
    return p_in - sol["q_in"] * drop


def stations_for(n_p: int) -> Tuple[Tuple[str, float], ...]:
    """Cell-centre stem stations.  Pre-reg §3.3: branch stations carry no *profile*
    information (along a uniform branch p(xi) = p_J(1-xi) is independent of w), so putting
    them in n_p would dilute what "densifying stations" is supposed to measure."""
    return tuple(("stem", (j + 0.5) / n_p) for j in range(n_p))


def observables_distributed(theta_dist: Sequence[float], lengths: Dict[str, float],
                            p_in: float, stations: Sequence[Tuple[str, float]]
                            ) -> List[float]:
    """Node block first (same five names as impedance_baseline.OBSERVABLES), then stations."""
    sol = solve_distributed(theta_dist, lengths, p_in)
    vals = [sol["q_in"], sol["q_up"], sol["q_down"], p_in, sol["p_junction"]]
    for branch, frac in stations:
        if branch != "stem":
            # pre-reg §3.3 puts the n_p axis on the stem only.  Refuse rather than silently
            # treating a branch station as a stem station -- that would be a wrong number
            # that still looks finite.
            raise GateError(f"this gate's station set is stem-only (pre-reg 3.3); got "
                            f"branch={branch!r}")
        vals.append(stem_pressure(theta_dist, lengths, p_in, frac))
    return vals


def jacobian_distributed(theta_dist: Sequence[float], lengths: Dict[str, float],
                         p_in: float, stations: Sequence[Tuple[str, float]],
                         fd_step: float = FD_STEP) -> List[List[float]]:
    """rows = parameters, cols = observables, entry = d ln(observable) / d ln(param).

    Dimensionally: delta_o_a / |o_a| = sum_i J[i][a] * delta_theta_i / theta_i, so the
    finite difference is (o_plus - o_minus) / (2h) / |o_a| -- theta_i is already carried by
    the multiplicative step.  NOTE the read-only reference module writes
    `/ (2.0 * h * theta[idx])`, i.e. it divides by theta once more, rescaling every column
    by 1/theta_i.  That is rank-preserving (so its "node data rank 3 < 4" and "rank 4 with
    stations" claims stand untouched), but a resolution limit is a number in
    relative-parameter units, so the extra factor would move delta_resolve by up to 1/kappa
    = 8.3x here.  This module therefore uses the dimensionally correct form and the
    self-test pins both the difference and the fact that rank is unaffected by it.
    """
    base = observables_distributed(theta_dist, lengths, p_in, stations)
    scale = [max(abs(o), 1.0e-12) for o in base]
    rows = []
    for idx in range(len(theta_dist)):
        tp = list(theta_dist); tp[idx] *= (1.0 + fd_step)
        tm = list(theta_dist); tm[idx] *= (1.0 - fd_step)
        rp = observables_distributed(tp, lengths, p_in, stations)
        rm = observables_distributed(tm, lengths, p_in, stations)
        rows.append([(a - b) / (2.0 * fd_step) / sc for a, b, sc in zip(rp, rm, scale)])
    return rows


# ------------------------------------------------------------------ rank and spectrum
def rank_of(jac: List[List[float]]) -> int:
    """The one and only rank entry point (rule 1).  Elimination, never a spectrum."""
    return ib.rank_by_elimination([row[:] for row in jac])


def transpose(mat: List[List[float]]) -> List[List[float]]:
    return [[mat[i][j] for i in range(len(mat))] for j in range(len(mat[0]))]


def rref_with_pivots(mat: List[List[float]], tol: float = 1.0e-9
                     ) -> Tuple[List[List[float]], List[int]]:
    """Row-reduced echelon form with row pivoting; also returns the pivot columns.

    Same elimination machinery as rank_of, so a null direction comes from the same kernel
    that produced the rank -- and the self-test then checks it against the analytic
    trade-off, which is what keeps "elimination says rank 3" from being self-certaining.
    """
    rows = [row[:] for row in mat]
    n_rows, n_cols = len(rows), len(rows[0])
    pivots: List[int] = []
    r = 0
    for col in range(n_cols):
        piv = max(range(r, n_rows), key=lambda i: abs(rows[i][col])) if r < n_rows else -1
        if piv < 0 or abs(rows[piv][col]) <= tol * max(1.0, max(abs(x[col]) for x in rows)):
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        lead = rows[r][col]
        rows[r] = [v / lead for v in rows[r]]
        for i in range(n_rows):
            if i == r or abs(rows[i][col]) == 0.0:
                continue
            f = rows[i][col]
            rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        pivots.append(col)
        r += 1
        if r == n_rows:
            break
    return rows, pivots


def null_directions(mat: List[List[float]], tol: float = 1.0e-9) -> List[List[float]]:
    """Basis of {v : mat v = 0} read off the RREF.  mat is m x n, vectors live in R^n."""
    red, pivots = rref_with_pivots([row[:] for row in mat], tol=tol)
    n_cols = len(mat[0])
    free = [c for c in range(n_cols) if c not in pivots]
    out = []
    for fc in free:
        v = [0.0] * n_cols
        v[fc] = 1.0
        for row_i, pc in enumerate(pivots):
            v[pc] = -red[row_i][fc]
        out.append(v)
    return out


def singular_values(mat: List[List[float]], sweeps: int = 60) -> List[float]:
    """Singular values by ONE-SIDED Jacobi column orthogonalisation (no J^T J, no eigensolver).

    `mat` is m x n with m >= n; returns the n singular values descending.  Rotating column
    pairs directly keeps relative accuracy for small and repeated singular values, which is
    the regime this gate lives in -- see rule 2 in the module docstring.  Sweeps stop when
    every column pair is orthogonal to 1e-15 of the geometric mean of its two norms, so the
    exit condition is about the rotation residual, not about a guessed iteration count.
    """
    n_rows, n_cols = len(mat), len(mat[0])
    cols = [[mat[r][c] for r in range(n_rows)] for c in range(n_cols)]
    norms = [math.sqrt(sum(v * v for v in col)) for col in cols]
    for _ in range(sweeps):
        worst = 0.0
        for p in range(n_cols):
            for q in range(p + 1, n_cols):
                gamma = sum(a * b for a, b in zip(cols[p], cols[q]))
                gate = 1.0e-15 * norms[p] * norms[q]
                worst = max(worst, abs(gamma) / gate if gate > 0.0 else 0.0)
                if abs(gamma) <= gate:
                    continue
                alpha, beta = norms[p] ** 2, norms[q] ** 2
                if abs(gamma) <= 1.0e-300:
                    continue
                if abs(alpha - beta) <= 1.0e-300:
                    t = 1.0 if gamma >= 0.0 else -1.0
                else:
                    zeta = (beta - alpha) / (2.0 * gamma)
                    t = (1.0 if zeta >= 0.0 else -1.0) / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                cp, cq = cols[p], cols[q]
                for r in range(n_rows):
                    ap, aq = cp[r], cq[r]
                    cp[r] = c * ap - s * aq
                    cq[r] = s * ap + c * aq
                norms[p] = math.sqrt(sum(v * v for v in cp))
                norms[q] = math.sqrt(sum(v * v for v in cq))
        if worst <= 1.0:
            break
    return sorted(norms, reverse=True)


def informative_columns(jac: List[List[float]]) -> List[int]:
    """Columns that are not identically zero.

    p_in is a boundary condition, so its Jacobian column is zero for every theta.  Noise
    accounting uses the informative count (pre-reg §3.4); the observable itself stays in
    the table because the observation-table format is shared with the reference module.
    """
    m = len(jac[0])
    return [a for a in range(m) if any(abs(jac[i][a]) > 0.0 for i in range(len(jac)))]


def sigma_bar_min(jac: List[List[float]]) -> dict:
    """sigma_min / sqrt(m_inf) -- the plateau-testable form (pre-reg §3.5)."""
    n_params = len(jac)
    a = transpose(jac)                                  # m x n, the linear map theta -> o
    inf = informative_columns(jac)
    sig = singular_values(a)
    m_inf = len(inf)
    raw = sig[min(len(sig), n_params) - 1] if sig else 0.0
    r = rank_of(jac)
    return {"n_params": n_params, "n_observables": len(jac[0]), "m_inf": m_inf,
            "spectrum": sig, "sigma_min": raw,
            "sigma_bar_min": (raw / math.sqrt(m_inf)) if m_inf else 0.0,
            "rank": r, "rank_source": RANK_SOURCE,
            "rank_deficient": bool(r < n_params),
            "spectral_zero_agrees_with_elimination": bool((raw <= 1.0e-9) == (r < n_params))}


# ------------------------------------------------------------------ threshold and gate
def tau(sigma_rel: float) -> float:
    """The threshold, as a formula over declared quantities only (pre-reg §5)."""
    if sigma_rel <= 0.0:
        raise GateError(f"sigma_rel must be > 0, got {sigma_rel}")
    return DETECTION_K * sigma_rel / DELTA_TARGET


def delta_resolve(sigma_rel: float, sigma_bar: float) -> Optional[float]:
    """Smallest relative width perturbation whose linearised signal reaches the noise floor."""
    if sigma_bar <= 0.0:
        return None
    return DETECTION_K * sigma_rel / sigma_bar


def drift_vs_coarse(cells: Dict[int, float], tol_base: int = N_P_MATRIX[0]) -> float:
    """Relative change of sigma_bar_min when stations are densified.  Base = the coarse
    cell, fixed in advance (pre-reg §4), so the two orderings cannot disagree later."""
    if len(N_P_MATRIX) != 2 or tol_base not in cells:
        raise GateError(f"drift needs the {N_P_MATRIX} pair, got {sorted(cells)}")
    fine = [n for n in N_P_MATRIX if n != tol_base][0]
    base = cells[tol_base]
    if base <= 0.0:
        return math.inf
    return abs(cells[fine] - base) / base


def require_plateau(cells: Dict[int, float], label: str = "") -> dict:
    """The <10% clause of §十.3(a), as something that RAISES."""
    d = drift_vs_coarse(cells)
    ok = d < PLATEAU_REL_TOL
    if not ok:
        raise GateError(f"{label or 'cells'}: sigma_bar_min drifts {d:.4%} between "
                        f"n_p={N_P_MATRIX[0]} and n_p={N_P_MATRIX[1]}, "
                        f">= {PLATEAU_REL_TOL:.0%} -> not a plateau, so no cell here may be "
                        f"quoted as the device's resolution limit")
    return {"drift": d, "in_plateau": True, "tol": PLATEAU_REL_TOL}


def classify(rank: int, n_params: int, sigma_bar: float, sigma_rel: Optional[float],
             in_plateau: bool) -> dict:
    """Three states only (the set §十.3(a) named), with `cause` keeping the two flavours
    of NOT_IDENTIFIABLE from being conflated in prose.

    Per 统括官 10:2x ②: `state` and `quotable` are TWO SEPARATE AXES.  `quotable` is
    produced by the plateau check alone and never by this function's caller; `state` is the
    three-state judgement.  A cell can therefore read "红 but 可引用" (a stable measurement
    whose answer is negative) or "过判据 but 不可引用" (drift >= 10%).  `meets_noise_floor`
    is carried alongside so the noise verdict is never masked by the plateau branch -- that
    is what closes the ordering ambiguity logged as A2.
    """
    if rank < n_params:
        return {"state": "NOT_IDENTIFIABLE", "cause": "rank_deficient",
                "meets_noise_floor": None,
                "note": "structural and noise-independent: no sigma_rel buys this back"}
    if sigma_rel is None:
        return {"state": "INDETERMINATE", "cause": "no_full_scale_pressure",
                "meets_noise_floor": None,
                "note": "kPa tier cannot be turned into a relative noise without a device "
                        "full-scale pressure drop"}
    meets = bool(sigma_bar >= tau(sigma_rel))
    if not in_plateau:
        return {"state": "INDETERMINATE", "cause": "station_density_not_saturated",
                "meets_noise_floor": meets,
                "note": "the noise-floor criterion is met but densifying stations still "
                        "moves sigma_bar_min"}
    if not meets:
        return {"state": "NOT_IDENTIFIABLE", "cause": "below_noise_floor",
                "meets_noise_floor": False,
                "note": "full rank, plateau, but the weakest direction sits under the noise "
                        "floor at this tier -- practically, not structurally"}
    return {"state": "RESOLVED", "cause": "above_noise_floor", "meets_noise_floor": True,
            "note": "target detail is resolved at this tier"}


def resolution_claim(cell: dict) -> str:
    """The gate's teeth: a "the device resolves delta" sentence needs BOTH axes.

    state == RESOLVED says the device can do it; quotable (plateau alone) says the number
    was measured somewhere the station density no longer matters.  Either one missing -> raise.
    """
    if cell.get("state") != "RESOLVED" or not cell.get("quotable"):
        raise GateError(f"refused: state={cell.get('state')} cause={cell.get('cause')} "
                        f"quotable={cell.get('quotable')} -- this cell does not support a "
                        f"resolution claim")
    return (f"device resolves relative width perturbation >= {cell['delta_resolve']:.3e} "
            f"of {DELTA_TARGET:.2f} target at sigma_rel={cell['sigma_rel']:.3e} "
            f"(drift {cell['drift_vs_coarse']:.3%} < {PLATEAU_REL_TOL:.0%})")


def require_provenance(noise_frac: float, provenance: Optional[str]) -> None:
    """Delegate, do not reimplement: the 3% floor and its threshold stay word-for-word the
    reference module's (pre-reg §7, "阈值一字不动")."""
    ib.require_provenance(noise_frac, provenance)


# ------------------------------------------------------------------------- the matrix
def build_cells(lengths: Dict[str, float], theta_true: Sequence[float], p_in: float
                ) -> List[dict]:
    out = []
    for n_p in N_P_MATRIX:
        st = stations_for(n_p)
        sm = sigma_bar_min(jacobian_distributed(theta_true, lengths, p_in, st))
        out.append({"n_p": n_p, **sm})
    by_np = {c["n_p"]: c["sigma_bar_min"] for c in out}
    d = drift_vs_coarse(by_np)
    for c in out:
        c["drift_vs_coarse"] = d
        # the plateau axis, on its own: it says "this number may be quoted", nothing about
        # whether the device resolves anything (统括官 10:2x ②).
        c["quotable"] = bool(d < PLATEAU_REL_TOL)
    return out


def annotate_kpa_cells(cells: List[dict]) -> None:
    """The sigma_p axis does not move sigma_bar_min at all (it is a noiseless Jacobian
    property), so the kPa label rides along as provenance status, not as a second scan."""
    for c in cells:
        c["sigma_bar_min_is_independent_of_sigma_p"] = True
        if FULL_SCALE_DP_KPA is None:
            c["kpa_tier_delta_resolve"] = "UNMEASURED_NO_FULL_SCALE"
        else:
            c["kpa_tier_delta_resolve"] = {f"{s:g}kPa": delta_resolve(s / FULL_SCALE_DP_KPA,
                                                                     c["sigma_bar_min"])
                                           for s in SIGMA_P_KPAS}


def run_tier(cells: List[dict], sigma_rel: Optional[float], tier_label: str,
             provenance: Optional[str] = None) -> List[dict]:
    rows = []
    for c in cells:
        v = dict(c)                      # v["quotable"] is the plateau axis, set in build_cells
        v["tier"] = tier_label
        v["tau"] = tau(sigma_rel) if sigma_rel else None
        v["sigma_rel"] = sigma_rel
        v["delta_resolve"] = (delta_resolve(sigma_rel, c["sigma_bar_min"])
                              if sigma_rel else None)
        v.update(classify(c["rank"], c["n_params"], c["sigma_bar_min"], sigma_rel,
                          c["quotable"]))
        if provenance is not None:
            v["noise_provenance"] = provenance
        rows.append(v)
    return rows


def format_row(r: dict) -> str:
    """One line per cell, with BOTH axes visible so the three shapes read apart:
    green+quotable / passes-criteria-but-drift-too-large / red."""
    dr = "n/a" if r["delta_resolve"] is None else f"{r['delta_resolve']:.3e}"
    tv = "n/a" if r["tau"] is None else f"{r['tau']:.4g}"
    mn = "n/a" if r["meets_noise_floor"] is None else r["meets_noise_floor"]
    return (f"[gateA] cell K={K_STEM_ELEMENTS} n_p={r['n_p']} tier={r['tier']} "
            f"m_inf={r['m_inf']} rank={r['rank']}/{r['n_params']} "
            f"sigma_min={r['sigma_min']:.4e} sigma_bar_min={r['sigma_bar_min']:.4e} "
            f"drift={r['drift_vs_coarse']:.3%} quotable_by_plateau={r['quotable']} "
            f"tau={tv} delta_resolve={dr} meets_noise_floor={mn} "
            f"state={r['state']} cause={r['cause']}")


def print_rows(rows: List[dict]) -> None:
    for r in rows:
        print(format_row(r))


def report(lengths: Dict[str, float], theta_true: Sequence[float], p_in: float,
           quiet_provenance: Optional[str] = None) -> dict:
    cells = build_cells(lengths, theta_true, p_in)
    annotate_kpa_cells(cells)
    rows = run_tier(cells, NOISE_TIER_MAIN, f"{NOISE_TIER_MAIN:.0%}relative")
    try:
        require_provenance(NOISE_TIER_QUIET, quiet_provenance)
        rows += run_tier(cells, NOISE_TIER_QUIET, f"{NOISE_TIER_QUIET:.0%}relative",
                         quiet_provenance)
    except ValueError as exc:
        rows += [{**c, "tier": f"{NOISE_TIER_QUIET:.0%}relative", "sigma_rel": None,
                  "tau": None, "delta_resolve": None, "meets_noise_floor": None,
                  "state": "INDETERMINATE", "cause": "unmeasured_no_provenance",
                  "note": f"require_provenance refused the tier: {exc}"} for c in cells]
    out: dict = {"pre_registration": "paper-route2/格A可辨识性-预注册-20260927.md",
                 "warning": "1-D synthetic world, identifiability only. This unlocks no "
                            "accuracy claim and does not touch K0: S2/S3 stay frozen.",
                 "constants": {"K_STEM_ELEMENTS": K_STEM_ELEMENTS, "N_PARAMS": N_PARAMS,
                               "DELTA_TARGET": DELTA_TARGET, "PLATEAU_REL_TOL":
                               PLATEAU_REL_TOL, "DETECTION_K": DETECTION_K,
                               "NULL_PROBE_TOL": NULL_PROBE_TOL, "FD_STEP": FD_STEP,
                               "N_P_MATRIX": list(N_P_MATRIX),
                               "SIGMA_P_KPAS": list(SIGMA_P_KPAS),
                               "FULL_SCALE_DP_KPA": FULL_SCALE_DP_KPA,
                               "rank_source": RANK_SOURCE},
                 "cells_plateau": cells,
                 # the 2x2 matrix as §十.3(a) fixes it: both kPa rows are carried, judged
                 # with sigma_rel=None, so they come out INDETERMINATE / no_full_scale and
                 # the missing conversion is a printed fact rather than a silently dropped
                 # arm.  sigma_bar_min is identical along the sigma_p axis by construction.
                 "rows": run_tier(cells, None, "1kPa") + run_tier(cells, None, "0.1kPa")
                        + rows}
    try:
        out["plateau_verdict"] = require_plateau({c["n_p"]: c["sigma_bar_min"]
                                                 for c in cells}, label="gateA matrix")
    except GateError as exc:
        out["plateau_verdict"] = {"in_plateau": False, "raised": str(exc)}
    return out


def _is_inside_repo(path: Path) -> bool:
    root = HERE.parents[2]
    return root in path.resolve().parents or str(path).startswith(str(root))


def main() -> int:
    ap = argparse.ArgumentParser(description="gateA identifiability gate (stdlib, zero "
                                             "instance time, unlocks no accuracy claim)")
    ap.add_argument("--json", default="", help="write the report here (must be outside "
                                              "the repository)")
    ap.add_argument("--force", action="store_true", help="allow overwriting a non-empty "
                                                         "report target")
    ap.add_argument("--noise-frac", type=float, default=None,
                    help="extra relative tier to judge; below the 3%% floor needs provenance")
    ap.add_argument("--noise-provenance", default="",
                    help="instrument citation; mandatory below the 3%% floor")
    ap.add_argument("--claim", action="store_true",
                    help="assert a resolution claim on the 3%% cell pair; the gate RAISES "
                         "unless every one of them is RESOLVED")
    args = ap.parse_args()
    t0 = time_now()

    geom = ib.tg.TGeometry(ib.tg.case_by_id("TB-asym"))
    lengths = ib.lengths_from_geometry(geom)
    theta_node = [geom.case.w_stem, geom.case.w_branch_up, geom.case.w_branch_down, 0.12]
    p_in = ib._p_in_from(theta_node, lengths)
    theta_true = truth_theta(theta_node)

    rep = report(lengths, theta_true, p_in, args.noise_provenance or None)
    rows = rep["rows"]
    if args.noise_frac is not None:
        require_provenance(args.noise_frac, args.noise_provenance or None)
        rows = run_tier(rep["cells_plateau"], args.noise_frac, f"{args.noise_frac:.0%}relative",
                        args.noise_provenance or None)
        rep["rows"] = rows
    print_rows(rows)
    print(f"[gateA] plateau={rep['plateau_verdict']}")
    print(f"[gateA] claim_locked=S2/S3_frozen K0_status=unchanged "
          f"instance_uses=0 cpu_s={time_now() - t0:.2f}")

    rc = 0
    if args.claim:
        try:
            for r in rows:
                if r["tier"] == f"{NOISE_TIER_MAIN:.0%}relative":
                    print("[gateA] claim:", resolution_claim(r))
        except GateError as exc:
            print(f"[gateA] REFUSED: {exc}")
            rc = 1

    if args.json:
        out = Path(args.json)
        if _is_inside_repo(out):
            raise SystemExit(f"refusing to write inside the repository: {out}")
        if out.is_file() and out.stat().st_size > 0 and not args.force:
            raise SystemExit(f"refusing to overwrite non-empty {out} (pass --force)")
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(rep, ensure_ascii=False, indent=2) + "\n"
        # write_bytes, not write_text: text mode turns every \n into \r\n on Windows, so a
        # digest taken from the string would not be the digest of the file anyone re-hashes.
        out.write_bytes(text.encode("utf-8"))
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"[gateA] artifact_selfcert: path={out} bytes={out.stat().st_size} sha256={digest}")
    print(f"[gateA] elapsed_s={time_now() - t0:.3f} rc={rc}")
    return rc


def time_now() -> float:
    return time.perf_counter()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as exc:
        print(f"[gateA] GATE ERROR: {exc}")
        sys.exit(1)
    except ImportError as exc:
        print(f"[gateA] CANNOT RUN: {exc}")
        sys.exit(3)
