#!/usr/bin/env python3
"""Self-test for the gateA identifiability gate.  stdlib only, zero instance time.

Run:  python model/scripts/route2/selftest_identifiability.py [--json OUT]

Exit 0 only if every check below passes.  The two controls named in the pre-registration
(paper-route2/gateA preregistration, section 9) live here:

  POSITIVE (must go red)   a sigma_bar_min pushed above the plateau band makes
                           require_plateau() raise.  If this cannot be written, the
                           threshold was picked rather than derived.
  NEGATIVE (kernel-trap)   the null direction read OFF the elimination finds the analytic
                           (1+kappa)/w_stem^3 trade-off, and moving along it leaves every
                           reading bitwise unchanged at the repo's 6-digit print width --
                           while a transversal move must change them.  This is the check
                           that does not depend on the rank kernel agreeing with itself.

Measurements (drift, sigma_bar_min, delta_resolve, the three-state verdicts) are printed on
`[gateA-measure]` lines and are NOT pass/fail checks: a cell coming out INDETERMINATE or
NOT_IDENTIFIABLE is a legitimate finding, and the self-test's job is to prove the device
can tell the three states apart, not to make the finding look good.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import identifiability_gate as G                      # noqa: E402
import impedance_baseline as ib                       # noqa: E402  READ-ONLY

GATE_SRC = (HERE / "identifiability_gate.py").read_text(encoding="utf-8")
_IS_WINDOWS = tempfile.gettempdir().startswith(("C:", "D:"))
DEFAULT_OUT = (Path("D:/PINN-restart/.scratch/route2/gateA_selftest.json") if _IS_WINDOWS
               else Path(os.environ.get("ROUTE2_OUT", tempfile.gettempdir()))
               / "gateA_selftest.json")
EXACT_PRINT = "%.6g"                                  # the repo's 6-significant-digit width
PLATEAU_INJECT = 1.35                                 # push one cell 35% above the band


def _emit(line: str) -> None:
    """Print without ever letting a glyph take the suite down.

    Measured 12:57: a `ck.measure` value containing `=>`/`\u21d2` hit the GBK console, raised
    UnicodeEncodeError inside `print`, and killed 58 finished readings plus the summary line --
    the same failure family as an in-process product call throwing through `ck.add`'s argument.
    Evidence must degrade to a `?`, never to a missing `total=`.
    """
    try:
        print(line)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, errors="replace").decode(enc, errors="replace"))


class Check:
    def __init__(self) -> None:
        self.rows: List[dict] = []
        self.t0 = time.perf_counter()

    def add(self, name: str, ok: bool, value: object, limit: object = "") -> None:
        self.rows.append({"check": name, "pass": bool(ok), "value": value, "limit": limit})
        _emit(f"[{'PASS' if ok else 'FAIL'}] {name}: value={_fmt(value)} limit={_fmt(limit)}")

    def skip(self, name: str, value: object, limit: object = "") -> None:
        """Counted and printed, never folded into PASS: a check with no input to run on is
        not the same claim as one that ran and passed."""
        self.rows.append({"check": name, "pass": True, "skipped": True,
                          "value": value, "limit": limit})
        _emit(f"[SKIP] {name}: value={_fmt(value)} limit={_fmt(limit)}")

    def measure(self, name: str, value: object) -> None:
        """A reading, not a verdict -- see the module docstring."""
        _emit(f"[gateA-measure] {name}={_fmt(value)}")

    @property
    def skipped(self) -> List[str]:
        return [r["check"] for r in self.rows if r.get("skipped")]

    @property
    def failed(self) -> List[str]:
        return [r["check"] for r in self.rows if not r["pass"]]


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)


def geometry() -> Dict[str, object]:
    geom = ib.tg.TGeometry(ib.tg.case_by_id("TB-asym"))
    lengths = ib.lengths_from_geometry(geom)
    theta_node = [geom.case.w_stem, geom.case.w_branch_up, geom.case.w_branch_down, 0.12]
    p_in = ib._p_in_from(theta_node, lengths)
    return {"lengths": lengths, "theta_node": theta_node, "p_in": p_in,
            "theta_dist": G.truth_theta(theta_node)}


# ------------------------------------------------------------------ kernel integrity
def kernel_checks(ck: Check) -> None:
    """Rule 1 and rule 2 of the gate, tested as behaviour rather than as a reading."""
    m = [[1.0, 0.0, 2.0], [0.0, 3.0, 1.0], [2.0, 1.0, 0.0], [1.0, 1.0, 1.0]]   # rank 3
    # (1) the import allow-list: a numpy/scipy/torch import would put an eigendecomposition
    # one keystroke away, which is the exact failure this line of work was burned by.
    roots = set()
    for node in ast.walk(ast.parse(GATE_SRC)):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    ck.add("gateA_kernel.import_allow_list", sorted(roots) == sorted(G.IMPORT_ALLOW_LIST),
           sorted(roots), sorted(G.IMPORT_ALLOW_LIST))
    ck.add("gateA_kernel.no_banned_numeric_import",
           not any(r in G.BANNED_IMPORT_ROOTS for r in roots), sorted(roots),
           f"none of {list(G.BANNED_IMPORT_ROOTS)}")
    called = {n.func.attr for n in ast.walk(ast.parse(GATE_SRC))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    called |= {n.func.id for n in ast.walk(ast.parse(GATE_SRC))
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    ck.add("gateA_kernel.eigensolver_never_called", "jacobi_eigen" not in called,
           sorted(called & {"jacobi_eigen", "linalg", "eig"}), "empty set")

    # (2) rank must be elimination all the way down: rank_of() has to agree with the
    # reference kernel's row-pivoted elimination on every fixture, and must not be a
    # spectrum threshold wearing a rank name.
    battery = [m, [[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]],
               [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                                     [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]]
    ck.add("gateA_kernel.rank_of_delegates_to_elimination_on_every_fixture",
           all(G.rank_of(x) == ib.rank_by_elimination([r[:] for r in x]) for x in battery)
           and G.rank_of([[0.0, 0.0], [0.0, 0.0]]) == 0
           and G.rank_of([[1.0, 0.0], [0.0, 1.0]]) == 2,
           [G.rank_of(x) for x in battery], "matches elimination, and is not vacuous")

    known = [([[1.0] * 4 for _ in range(4)], [4.0, 0.0, 0.0, 0.0], "rank_one_repeated_root"),
             ([[4.0, 0, 0, 0], [0, 3.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1.0]],
              [4.0, 3.0, 2.0, 1.0], "diagonal"),
             ([[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0], [0, 0, 0.0]], [3.0, 3.0, 3.0],
              "triple_repeated"),
             ([[0.0, -1.0], [1.0, 0.0], [0.0, 0.0]], [1.0, 1.0], "tall_rotation")]
    for mat, expect, label in known:
        got = G.singular_values(mat)
        ck.add(f"gateA_svd.spectrum_{label}",
               len(got) == len(expect) and all(abs(a - b) < 1.0e-12 for a, b in zip(got, expect)),
               got, expect)
    # the all-ones fixture IS the historical failure: the first-version eigensolver returned
    # [3.33, 0.67, 1.9e-4, 0] there and made a wrong physical claim look numerically backed.
    ck.add("gateA_svd.all_ones_is_not_the_old_wrong_spectrum",
           G.singular_values([[1.0] * 4 for _ in range(4)])[1] < 1.0e-12,
           G.singular_values([[1.0] * 4 for _ in range(4)]), "sigma_2 exactly 0")


# ------------------------------------------------------------------- forward map / FD
def map_checks(ck: Check, geo: Dict[str, object]) -> None:
    lengths, p_in = geo["lengths"], geo["p_in"]
    tn, th = geo["theta_node"], geo["theta_dist"]
    stem = [("stem", f) for _, f in ib.PROBE_SET]
    ref = ib.observables(tn, lengths, p_in, stem)
    mine = G.observables_distributed(th, lengths, p_in, stem)
    rel = max(abs(a - b) / max(abs(b), 1.0e-12) for a, b in zip(ref, mine))
    ck.add("gateA_forward_map_matches_readonly_reference_bitwise", ref == mine and rel < 1.0e-9,
           {"bitwise_equal": ref == mine, "max_rel": rel}, "lists identical, max_rel < 1e-9")
    ck.add("gateA_forward_map_refuses_a_branch_station", _raises(
        lambda: G.observables_distributed(th, lengths, p_in, [("up", 0.3)]), G.GateError),
        "GateError", "raises instead of silently mis-scoring a branch station")
    # FD-step guard on the quantity the gate actually decides with (this project has been
    # bitten by non-converged finite differences before -- see the S4 step-convergence gate).
    st = G.stations_for(16)
    a = G.sigma_bar_min(G.jacobian_distributed(th, lengths, p_in, st))["sigma_bar_min"]
    b = G.sigma_bar_min(G.jacobian_distributed(th, lengths, p_in, st,
                                               fd_step=G.FD_STEP / 2.0))["sigma_bar_min"]
    ck.add("gateA_fd_step_halving_keeps_sigma_bar_min", abs(b - a) / a < 1.0e-6,
           {"sigma_bar_min": [a, b], "rel_change": abs(b - a) / a}, "rel change < 1e-6")
    # cross-check the SVD against the normal-matrix spectrum of the READ-ONLY reference's
    # eigensolver.  This is a self-test cross-check, never a decision path (see
    # gateA_kernel.eigensolver_never_called above, which proves the gate itself does not
    # reach that kernel).
    jac = G.jacobian_distributed(th, lengths, p_in, G.stations_for(40))
    at = G.transpose(jac)
    n = len(at[0])
    ata = [[sum(at[r][i] * at[r][j] for r in range(len(at))) for j in range(n)]
           for i in range(n)]
    lam = ib.jacobi_eigen(ata)[0]
    eig_sigma = sorted((math.sqrt(max(v, 0.0)) for v in lam), reverse=True)
    mine_sigma = G.singular_values(at)
    worst = max(abs(x - y) / max(abs(y), 1.0e-30) for x, y in zip(mine_sigma, eig_sigma))
    ck.add("gateA_svd_agrees_with_normal_matrix_eigen_on_full_rank_cell", worst < 1.0e-6,
           {"worst_rel": worst, "sigma_min_gate": mine_sigma[-1],
            "sigma_min_normal_matrix": eig_sigma[-1]}, "worst rel < 1e-6")


def jacobian_scaling_checks(ck: Check, geo: Dict[str, object]) -> None:
    """A6 control for the shared-module fix: one leg that must go red if the extra
    `1/theta_i` comes back, one leg that pins the direction of the historical error, and
    one cross-module equality so no compensating edit can satisfy either leg alone.

    Analytic derivative used: with x = p_J / p_in and R_series = 12L(1+kappa)/w_stem^3,
        p_J = p_in / (1 + R_series*(1/r_up + 1/r_dn))
        d ln p_J / d ln w_stem = 3(1 - x)
        d ln p_J / d ln kappa  = -(kappa/(1+kappa)) (1 - x)
    Those come from differentiating the closed form in the module's own docstring, not from
    the code under test -- which is what makes the comparison a control rather than a mirror.
    """
    lengths, p_in = geo["lengths"], geo["p_in"]
    tn = geo["theta_node"]
    w_stem, kappa = tn[0], tn[3]
    jac = ib.jacobian_wrt_params(tn, lengths, p_in, ())
    p_j = ib.solve_network(tn, lengths, p_in)["p_junction"]
    x = p_j / p_in
    analytic = {"w_stem": 3.0 * (1.0 - x), "kappa": -(kappa / (1.0 + kappa)) * (1.0 - x)}
    fixed = {"w_stem": jac[0][4], "kappa": jac[3][4]}
    for name in analytic:
        rel = abs(fixed[name] - analytic[name]) / abs(analytic[name])
        ck.add(f"gateA_jacobian_scaling_matches_analytic_derivative[{name}]", rel < 1.0e-6,
               {"fd": fixed[name], "analytic": analytic[name], "rel": rel}, "rel < 1e-6")

    # leg B: an over-correction detector.  If someone "fixes" the missing theta by multiplying
    # instead of simply not dividing, leg A passes for theta_i = 1 and fails elsewhere; this
    # leg demands the shipped value NOT equal analytic * theta_i wherever theta_i != 1.
    # (The historical error is invisible on columns with theta_i = 1 -- that is exactly why it
    # survived: the headline rank case has w_stem = 1.0 and only kappa, off by 1/kappa = 8.33,
    # betrayed it.  Recorded as a measurement, not as a pass/fail claim.)
    for idx, name in ((0, "w_stem"), (1, "w_up"), (2, "w_dn"), (3, "kappa")):
        ratio = 1.0 / tn[idx]
        ck.measure(f"jacobian_scaling_legB[{name}]", {"theta": tn[idx],
                                                      "old_over_new": ratio,
                                                      "analytic_available": name in analytic})
        if tn[idx] != 1.0 and name in analytic:
            ck.add(f"gateA_jacobian_scaling_not_over_corrected[{name}]",
                   abs(jac[idx][4] - analytic[name] * tn[idx]) / abs(analytic[name]) > 1.0e-3,
                   {"shipped": jac[idx][4], "analytic_x_theta": analytic[name] * tn[idx]},
                   "shipped != analytic*theta (a multiply-instead-of-not-divide would go red)")

    # cross-module equality: this gate and the reference module must now compute the SAME
    # operator.  Elementwise agreement; it would have been off by 1/theta_i before the fix.
    mine = G.jacobian_distributed(tn, lengths, p_in, ())
    worst = 0.0
    for i in range(4):
        for j in range(5):
            if abs(mine[i][j]) > 1.0e-9:
                worst = max(worst, abs(jac[i][j] - mine[i][j]) / abs(mine[i][j]))
    ck.add("gateA_jacobian_agrees_elementwise_with_reference_module", worst < 1.0e-9,
           _r(worst), "< 1e-9 elementwise")


def _raises(fn, exc) -> bool:
    try:
        fn()
        return False
    except exc:
        return True


# --------------------------------------------------------------------------- controls
def negative_control_checks(ck: Check, geo: Dict[str, object]) -> None:
    """The rank-deficiency check that does NOT depend on the kernel trusting itself."""
    lengths, p_in = geo["lengths"], geo["p_in"]
    tn = geo["theta_node"]
    jac = G.jacobian_distributed(tn, lengths, p_in, ())        # node-only: 4 params, 5 obs
    rank = G.rank_of(jac)
    nulls = G.null_directions(G.transpose(jac))
    ck.add("gateA_negative_control_elimination_finds_exactly_one_null_direction",
           rank == 3 and len(nulls) == 1, {"rank": rank, "null_dim": len(nulls)},
           "rank 3 < 4 params, null space 1-dimensional")

    # that null direction must BE the analytic (1+kappa)/w_stem^3 trade-off, in dlog coords:
    # dlog w_stem = kappa/(3(1+kappa)) * dlog kappa.  This is what caught the reference
    # module's double division by theta (it rotated the null direction by 1/kappa).
    w, kappa = tn[0], tn[3]
    analytic = [kappa / (3.0 * (1.0 + kappa)), 0.0, 0.0, 1.0]
    v = nulls[0]
    cos = sum(a * b for a, b in zip(v, analytic)) / (
        math.sqrt(sum(x * x for x in v)) * math.sqrt(sum(x * x for x in analytic)))
    ck.add("gateA_negative_control_elimination_null_is_the_analytic_trade_off",
           abs(abs(cos) - 1.0) < 1.0e-9, {"cos": cos, "v": [_r(x) for x in v]},
           "|cos| = 1 to 1e-9")

    # The rank deficiency lives in the NODE block, so "unchanged" is asserted on the node
    # block; the complementary fact -- that the same trade-off does move the stem stations,
    # which is what makes stations the degeneracy-breaker -- is asserted right after it.
    base = G.observables_distributed(G.truth_theta(tn), lengths, p_in, ())
    # (a) exactly invariant path: (1+kappa) and w^3 scaled by the same factor, every stem
    #     element together with kappa, so R_stem*(1+kappa) is fixed by construction.
    moved = {}
    trade = {}
    for eps in (1.0e-4, 1.0e-3, 1.0e-2):
        f = (1.0 + eps) ** (1.0 / 3.0)
        th = G.truth_theta([w * f] + tn[1:3] + [(1.0 + kappa) * (1.0 + eps) - 1.0])
        got = G.observables_distributed(th, lengths, p_in, ())
        moved[eps] = {"max_rel": max(abs(x - y) / max(abs(y), 1.0e-12)
                                     for x, y in zip(got, base)),
                      "bitwise_identical_at_print_width":
                          [_print(x) for x in got] == [_print(y) for y in base]}
        rich = G.observables_distributed(th, lengths, p_in, G.stations_for(16))
        base_rich = G.observables_distributed(G.truth_theta(tn), lengths, p_in,
                                              G.stations_for(16))
        trade[eps] = max(abs(x - y) / max(abs(y), 1.0e-12)
                         for x, y in zip(rich[5:], base_rich[5:]))
    ck.add("gateA_negative_control_null_direction_readings_bitwise_unchanged",
           all(m["bitwise_identical_at_print_width"] and m["max_rel"] < G.NULL_PROBE_TOL
               for m in moved.values()),
           {str(k): [_r(v["max_rel"]), v["bitwise_identical_at_print_width"]]
            for k, v in moved.items()},
           f"all node readings identical at {EXACT_PRINT} and rel < {G.NULL_PROBE_TOL:g}")
    ck.add("gateA_negative_control_the_same_trade_off_moves_the_station_block",
           all(v >= G.NULL_PROBE_TOL for v in trade.values()),
           {str(k): _r(v) for k, v in trade.items()},
           f">= {G.NULL_PROBE_TOL:g}: stations are what break the degeneracy")

    # (b) the same direction taken from the elimination output, exponentiated.  A null
    #     vector is a first-order statement, so invariance holds to O(t^2) -- t=1e-6 keeps
    #     the second-order term under the inherited 1e-9 boundary, which is why t is fixed
    #     here and not tuned per-eps.
    th2 = [t * math.exp(1.0e-6 * c) for t, c in zip(G.truth_theta(tn), _pad(v, 11))]
    got2 = G.observables_distributed(th2, lengths, p_in, ())
    rel2 = max(abs(x - y) / max(abs(y), 1.0e-12) for x, y in zip(got2, base))
    ck.add("gateA_negative_control_exponentiated_elimination_direction_is_flat_to_first_order",
           rel2 < G.NULL_PROBE_TOL, _r(rel2), f"< {G.NULL_PROBE_TOL:g} at t=1e-6")

    # (c) the probe must be alive: a transversal move (all stem widths together, kappa held)
    #     has to change the readings.  Without this, "unchanged" could mean "nothing measured".
    perp = G.truth_theta([w * (1.0 + 1.0e-3)] + tn[1:])
    gp = G.observables_distributed(perp, lengths, p_in, ())
    relp = max(abs(x - y) / max(abs(y), 1.0e-12) for x, y in zip(gp, base))
    ck.add("gateA_negative_control_transversal_perturbation_moves_readings",
           relp >= G.NULL_PROBE_TOL, _r(relp), f">= {G.NULL_PROBE_TOL:g}")


def _pad(v: List[float], n: int) -> List[float]:
    """Spread the 4-parameter null direction onto the K-element profile: every stem element
    moves with w_stem, because the node block only sees their sum."""
    return [v[0]] * (n - 3) + v[1:]


def _r(x: float, d: int = 12) -> float:
    return round(x, d) if isinstance(x, float) else x


def _print(x: object) -> str:
    return EXACT_PRINT % x


def positive_control_checks(ck: Check, geo: Dict[str, object]) -> None:
    """§9.1: push sigma_bar_min above the plateau band and the gate MUST go red."""
    honest = {G.N_P_MATRIX[0]: 1.0, G.N_P_MATRIX[1]: 1.05}          # drift 5% -> a plateau
    ok = None
    try:
        ok = G.require_plateau(honest, label="synthetic plateau")
    except G.GateError as exc:
        ok = f"unexpected raise: {exc}"
    ck.add("gateA_positive_control_plateau_gate_is_not_always_red",
           isinstance(ok, dict) and ok["in_plateau"], ok, "drift 5% < 10% passes")
    pushed = {G.N_P_MATRIX[0]: 1.0, G.N_P_MATRIX[1]: 1.0 * 1.35}     # drift 35% -> above band
    raised = _raises(lambda: G.require_plateau(pushed, label="gateA positive control"),
                     G.GateError)
    ck.add("gateA_positive_control_sigma_above_plateau_raises", raised, raised, "raise")
    # and the red must be caused by the *band*, not by the magnitude: a uniform 100x scaling
    # of both cells leaves the drift (and so the verdict) untouched.
    ck.add("gateA_positive_control_band_is_scale_free",
           _raises(lambda: G.require_plateau({G.N_P_MATRIX[0]: 100.0,
                                              G.N_P_MATRIX[1]: 105.0}), G.GateError) is False,
           "drift still 5% at 100x", "passes, so the tol is a relative band")

    # the second positive control: a rank-deficient cell must not be allowed to speak.
    lengths, p_in, tn = geo["lengths"], geo["p_in"], geo["theta_node"]
    cell = dict(G.sigma_bar_min(G.jacobian_distributed(tn, lengths, p_in, ())))
    cell.update({"n_p": 0, "tier": "node-only", "sigma_rel": G.NOISE_TIER_MAIN,
                 "tau": G.tau(G.NOISE_TIER_MAIN), "delta_resolve": None,
                 "drift_vs_coarse": 0.0, "row_citable": True})
    cell.update(G.classify(cell["rank"], cell["n_params"], cell["sigma_bar_min"],
                           cell["sigma_rel"], cell["row_citable"]))
    cell["claim_supported"] = G.claim_supported(cell["state"], cell["row_citable"])
    ck.add("gateA_positive_control_rank_deficient_cell_refuses_resolution_claim",
           cell["state"] == "NOT_IDENTIFIABLE" and cell["cause"] == "rank_deficient"
           and _raises(lambda: G.resolution_claim(cell), G.GateError),
           {"state": cell["state"], "cause": cell["cause"]},
           "NOT_IDENTIFIABLE/rank_deficient and resolution_claim raises")


def threshold_checks(ck: Check) -> None:
    """§5: tau is a formula over declared quantities, and the declared quantities are the
    ruling's numbers, not mine."""
    ck.add("gateA_threshold_is_a_pure_function_of_the_declared_tier",
           all(abs(G.tau(s) - s / G.DELTA_TARGET) < 1.0e-15 for s in (0.03, 0.01, 0.17, 1.0)),
           [G.tau(0.03), G.tau(0.01)], "sigma_rel / 0.10")
    ck.add("gateA_threshold_never_reads_a_measured_sigma",
           "sigma_bar_min" not in _func_src("tau"), "no measured quantity inside tau()")
    ck.add("gateA_pre_registered_constants_untouched",
           (G.DELTA_TARGET, G.PLATEAU_REL_TOL, G.DETECTION_K, G.NULL_PROBE_TOL)
           == (0.10, 0.10, 1.0, 1.0e-9)
           and G.NOISE_TIER_MAIN == ib.NOISE_FLOOR_DEFAULT
           and ib.NOISE_PROVENANCE_MIN == 0.03,
           {"DELTA_TARGET": G.DELTA_TARGET, "PLATEAU_REL_TOL": G.PLATEAU_REL_TOL,
            "DETECTION_K": G.DETECTION_K, "NULL_PROBE_TOL": G.NULL_PROBE_TOL,
            "NOISE_PROVENANCE_MIN": ib.NOISE_PROVENANCE_MIN},
            "(0.10, 0.10, 1.0, 1e-9) with the 3% floor and its provenance minimum unchanged")
    ck.add("gateA_threshold_refuses_a_zero_noise_tier",
           _raises(lambda: G.tau(0.0), G.GateError), "raise", "a 0% tier is not a tier")
    ck.add("gateA_noise_1pct_without_provenance_is_refused",
           _raises(lambda: G.require_provenance(G.NOISE_TIER_QUIET, None), ValueError)
           and not _raises(lambda: G.require_provenance(G.NOISE_TIER_QUIET,
                                                        "device calibration record (slot)"),
                           ValueError)
           and not _raises(lambda: G.require_provenance(G.NOISE_TIER_MAIN, None), ValueError),
           "1% refused bare, 1% passes with citation, 3% passes", "all three")


def _func_src(name: str) -> str:
    tree = ast.parse(GATE_SRC)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(GATE_SRC, node) or ""
    return ""


def report_checks(ck: Check, geo: Dict[str, object]) -> List[dict]:
    rep = G.report(geo["lengths"], geo["theta_dist"], geo["p_in"])
    rows = rep["rows"]
    ck.add("gateA_three_states_only",
           {r["state"] for r in rows} <= {"RESOLVED", "INDETERMINATE", "NOT_IDENTIFIABLE"},
           sorted({r["state"] for r in rows}), "subset of the three pre-registered states")
    ck.add("gateA_matrix_has_four_cells_both_kpa_rows_carried",
           len({(r["tier"], r["n_p"]) for r in rows if "kPa" in r["tier"]}) == 4
           and all(r["cause"] == "no_full_scale_pressure" for r in rows if "kPa" in r["tier"]),
           sorted({r["tier"] for r in rows}), "4 kPa cells, each UNMEASURED not silently cut")
    ck.add("gateA_sigma_bar_min_is_identical_along_the_sigma_p_axis",
           len({round(r["sigma_bar_min"], 15) for r in rows if "kPa" in r["tier"]}) == 2,
           "2 distinct values for n_p in {16,40}", "the kPa axis cannot move a noiseless sigma")
    ck.add("gateA_rank_source_recorded_on_every_cell",
           all(r["rank_source"] == G.RANK_SOURCE for r in rows), G.RANK_SOURCE,
           "elimination, named once in the module")
    ck.add("gateA_kpa_rows_never_pass_a_resolution_claim",
           all(_raises(lambda r=r: G.resolution_claim(r), G.GateError)
               for r in rows if "kPa" in r["tier"])
           and all(r["state"] == "INDETERMINATE" for r in rows if "kPa" in r["tier"]),
           {"kpa_states": sorted({r["state"] for r in rows if "kPa" in r["tier"]}),
            "kpa_row_citable": sorted({bool(r["row_citable"]) for r in rows
                                       if "kPa" in r["tier"]}),
            "kpa_claim_supported": sorted({bool(r["claim_supported"]) for r in rows
                                           if "kPa" in r["tier"]})},
           "claim refused and state INDETERMINATE even where the plateau axis is quotable")
    ck.add("gateA_claim_supported_is_the_conjunction_on_every_row",
           all(r["claim_supported"] == (r["state"] == "RESOLVED" and r["row_citable"])
               for r in rows) and not any(r["claim_supported"] for r in rows),
           {"rows": len(rows), "claim_supported_true": sum(1 for r in rows if r["claim_supported"]),
            "row_citable_true": sum(1 for r in rows if r["row_citable"])},
           "claim_supported == (state RESOLVED and plateau green); this round supports none")
    quiet = [r for r in rows if r["cause"] == "unmeasured_no_provenance"]
    ck.add("gateA_unmeasured_rows_are_citable_but_claim_nothing",
           len(quiet) == 2 and all(r["row_citable"] and not r["claim_supported"]
                                   and r["state"] == "INDETERMINATE" for r in quiet),
           [{"tier": r["tier"], "row_citable": r["row_citable"],
             "claim_supported": r["claim_supported"]} for r in quiet],
           "row_citable here means the UNMEASURED statement is quotable, nothing more")
    for r in rows:
        ck.measure(f"{r['tier']}/n_p={r['n_p']}",
                   {"rank": f"{r['rank']}/{r['n_params']}", "sigma_bar_min": r["sigma_bar_min"],
                    "drift": r["drift_vs_coarse"], "row_citable": r["row_citable"],
                    "claim_supported": r["claim_supported"],
                    "tau": r["tau"], "delta_resolve": r["delta_resolve"],
                    "meets_noise_floor": r["meets_noise_floor"],
                    "state": r["state"], "cause": r["cause"]})
    return rows


def _cell(sigma_bar: float, sigma_rel: float, drift: float, full_rank: bool = True) -> dict:
    n_p = G.N_P_MATRIX[0]
    cell = {"n_p": n_p, "tier": f"{sigma_rel:.0%}relative", "m_inf": 20,
            "n_params": G.N_PARAMS, "rank": G.N_PARAMS if full_rank else G.N_PARAMS - 1,
            "sigma_min": sigma_bar, "sigma_bar_min": sigma_bar,
            "drift_vs_coarse": drift, "row_citable": bool(drift < G.PLATEAU_REL_TOL),
            "rank_source": G.RANK_SOURCE, "spectral_zero_agrees_with_elimination": True,
            "spectrum": [], "independent_of_sigma_p": True}
    cell.update(G.classify(cell["rank"], cell["n_params"], sigma_bar, sigma_rel,
                           cell["row_citable"]))
    cell["claim_supported"] = G.claim_supported(cell["state"], cell["row_citable"])
    cell["tau"] = G.tau(sigma_rel)
    cell["sigma_rel"] = sigma_rel
    cell["delta_resolve"] = G.delta_resolve(sigma_rel, sigma_bar)
    return cell


def axis_checks(ck: Check) -> None:
    """A4 (②) split the plateau verdict from the three-state judgement; A9 split it again so
    a JSON-only reader cannot mistake "this row's reading may be quoted" for "this cell
    resolves the target".  Shapes on the console line: a citable row that supports a claim /
    a citable row that supports none (the negative and the unmeasured answers -- both are
    quotable statements, neither is a resolution claim) / a row the plateau does not license.
    """
    green = _cell(0.35, 0.03, 0.02)                    # sigma_bar 0.35 >= tau 0.30, drift 2%
    drift_only = _cell(0.35, 0.03, 0.12)               # same, but densification still moves it
    red = _cell(0.018, 0.03, 0.02)                     # under the noise floor, plateau holds
    rank_red = _cell(0.35, 0.03, 0.02, full_rank=False)
    lines = {"green": G.format_row(green), "drift_only": G.format_row(drift_only),
             "red": G.format_row(red)}
    ck.add("gateA_two_axes_citable_and_claim_supported",
           green["state"] == "RESOLVED" and green["row_citable"] and green["claim_supported"]
           and G.resolution_claim(green).startswith("device resolves"),
           {"state": green["state"], "row_citable": green["row_citable"],
            "claim_supported": green["claim_supported"]},
           "RESOLVED + plateau green -> claim_supported True and the claim is allowed")
    ck.add("gateA_two_axes_passes_criteria_but_drift_blocks_quoting",
           drift_only["state"] == "INDETERMINATE"
           and drift_only["cause"] == "station_density_not_saturated"
           and drift_only["meets_noise_floor"] is True and drift_only["row_citable"] is False
           and drift_only["claim_supported"] is False
           and _raises(lambda: G.resolution_claim(drift_only), G.GateError),
           {"state": drift_only["state"], "meets_noise_floor": True,
            "row_citable": drift_only["row_citable"]},
           "noise criterion met, plateau axis False, claim refused")
    ck.add("gateA_two_axes_red_but_the_measurement_is_citable",
           red["state"] == "NOT_IDENTIFIABLE" and red["cause"] == "below_noise_floor"
           and red["row_citable"] is True and red["claim_supported"] is False
           and _raises(lambda: G.resolution_claim(red), G.GateError),
           {"state": red["state"], "row_citable": red["row_citable"],
            "claim_supported": red["claim_supported"]},
           "a citable negative answer is still a negative answer, and never a claim")
    ck.add("gateA_two_axes_structural_red",
           rank_red["state"] == "NOT_IDENTIFIABLE" and rank_red["cause"] == "rank_deficient"
           and rank_red["meets_noise_floor"] is None and rank_red["claim_supported"] is False
           and _raises(lambda: G.resolution_claim(rank_red), G.GateError),
           {"state": rank_red["state"], "meets_noise_floor": None},
           "rank deficiency never buys a claim, citable or not")
    # A9's two positive controls: the stored boolean is not trusted.
    lying = dict(red, state="INDETERMINATE", cause="unmeasured_no_provenance",
                 claim_supported=True)
    ck.add("gateA_A9_control_claim_supported_true_on_an_unmeasured_row_raises",
           lying["row_citable"] is True and _raises(
               lambda: G.resolution_claim(lying), G.GateError),
           {"state": lying["state"], "row_citable": lying["row_citable"],
            "claim_supported(stored)": True},
           "raise: the field cannot be flipped to buy a claim")
    over_corrected = dict(green, row_citable=False,
                          claim_supported=G.claim_supported(green["state"], False))
    ck.add("gateA_A9_control_plateau_false_never_supports_a_claim",
           over_corrected["state"] == "RESOLVED" and not over_corrected["claim_supported"]
           and _raises(lambda: G.resolution_claim(over_corrected), G.GateError),
           {"state": over_corrected["state"],
            "claim_supported(recomputed)": over_corrected["claim_supported"]},
           "raise: RESOLVED without a green plateau is not a claim either")
    ck.add("gateA_console_line_separates_the_three_shapes",
           len({lines["green"], lines["drift_only"], lines["red"]}) == 3
           and all("state=" in v and "row_citable=" in v and "claim_supported=" in v
                   and "cause=" in v
                   for v in lines.values()),
           {k: [v[v.index("state="):]] for k, v in lines.items()},
           "3 distinct lines, each carrying state / cause / both axes")
    branches = [G.classify(G.N_PARAMS - 1, G.N_PARAMS, 0.35, 0.03, True),    # rank deficient
                G.classify(G.N_PARAMS, G.N_PARAMS, 0.35, None, True),        # no full scale
                G.classify(G.N_PARAMS, G.N_PARAMS, 0.35, 0.03, False),       # off plateau
                G.classify(G.N_PARAMS, G.N_PARAMS, 0.018, 0.03, True),       # below floor
                G.classify(G.N_PARAMS, G.N_PARAMS, 0.35, 0.03, True)]        # resolved
    ck.add("gateA_quotable_is_written_only_by_the_plateau_axis",
           all("quotable" not in b and "row_citable" not in b
               and "claim_supported" not in b for b in branches)
           and {b["state"] for b in branches} == {"NOT_IDENTIFIABLE", "INDETERMINATE",
                                                 "RESOLVED"},
           [sorted(b.keys()) for b in branches[:1]],
           "classify() returns state/cause/meets_noise_floor/note and neither axis field")



CHILD_ENCODING = "utf-8"
OLD_PUBLISHED_COMMIT = "114acd7baf7c0dcf0c1e6347df0d6d0a00467206"      # the byte pair that carried the crash-prone capture
SUITE_FILES = ("identifiability_gate.py", "selftest_identifiability.py", "impedance_baseline.py", "t_geometry.py")


def child_env(extra=None):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = CHILD_ENCODING      # the child writes what we are going to decode
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        env.update(extra)
    return env


def capture(done):
    """(rc, stdout, stderr, stdout_unavailable) -- the decoding is ours, over bytes.

    `text=True` let the PARENT decode with the locale default (cp936 on this machine) while the
    CHILD encoded per whatever PYTHONIOENCODING it inherited.  When the two disagree the parent
    does not merely garble: `CompletedProcess.stdout` comes back None and the harness died on
    `.strip()` instead of reporting (统括官 11:5x; reproduced 11:53:53 on 114acd7 and on the
    working tree: hostile env + a directory with no .git above -> rc=1, no total= line).
    """
    missing = done.stdout is None or done.stderr is None
    out = "" if done.stdout is None else done.stdout.decode(CHILD_ENCODING, errors="replace")
    err = "" if done.stderr is None else done.stderr.decode(CHILD_ENCODING, errors="replace")
    return done.returncode, out, err, missing


def run_suite(script, cwd, args=(), extra=None, timeout=900):
    return capture(subprocess.run([sys.executable, str(script)] + list(args),
                                  capture_output=True, cwd=str(cwd), timeout=timeout,
                                  env=child_env(extra)))


def no_repo_above(path):
    for cand in (path, *path.parents):
        if (cand / ".git").exists():
            return False
    return True


def reap(*paths):
    """Remove only the boxes THIS run created, and prove they are gone.

    A skipped branch that leaks its box is how a later "residue must be empty" assertion turns
    red.  Orphans already under D:/Temp from interrupted runs are left alone -- sweeping by
    filename prefix is the risky pattern, not the fix.
    """
    for q in paths:
        if q is None or not q.exists():
            continue
        shutil.rmtree(q, ignore_errors=True)
        if q.exists():
            print(f"gateA_box_left_behind: {q} (not removed, deliberately not forced)")


def oob_verdict(rc, total_line, rc_allowed=(0, 2)):
    """The judgement as a pure function, so the fixture's tooth can be fed a degenerate child
    and shown to go red instead of being argued about.  rc=2 (artifact write refused) is a
    legitimate outcome; rc=1 with no summary line is not.
    """
    reasons = []
    if not total_line:
        reasons.append("summary_missing")
    if rc not in rc_allowed:
        reasons.append(f"rc_out_of_range:{rc}")
    return {"green": not reasons, "reasons": reasons, "rc": rc,
            "total_line": total_line[:70]}


def _git_show(rev, rel, cwd):
    done = subprocess.run(["git", "show", f"{rev}:{rel}"], capture_output=True,
                          cwd=str(cwd), timeout=120, env=child_env())
    if done.returncode != 0 or done.stdout is None:
        return None
    return done.stdout.decode(CHILD_ENCODING, errors="replace")


LEGACY_MARKER = "⇒ 未验"          # cp936 cannot round-trip this; that is exactly the point


def legacy_capture(script, cwd, extra=None):
    """The PRE-FIX pattern, kept only so the fixture can show it would have caught it.

    With `text=True` and no encoding, the parent decodes with the locale default (cp936 here)
    inside a reader thread; a child that prints a byte cp936 cannot decode kills that thread and
    `communicate()` then hands back `stdout=None`.  One mechanism, two faces -- which is why the
    crash reported at 11:5x showed a UnicodeDecodeError in the log AND an AttributeError on None.
    Nothing real is decided through this path.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = CHILD_ENCODING
    if extra:
        env.update(extra)
    # The decode failure happens on CPython's reader THREAD, where it is reported through
    # threading.excepthook (i.e. straight to our stderr) and `communicate()` then returns None.
    # Capturing it here keeps the evidence in the check value and the suite's stderr empty.
    caught = []
    saved_hook = threading.excepthook

    def _hook(args):
        caught.append("%s: %s" % (args.exc_type.__name__, str(args.exc_value)[:70]))

    threading.excepthook = _hook
    try:
        done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                              timeout=120, cwd=str(cwd), env=env)
    except Exception as exc:                       # UnicodeDecodeError can surface right here
        return False, "%s: %s" % (type(exc).__name__, str(exc)[:70])
    finally:
        threading.excepthook = saved_hook
    if done.stdout is None:
        return False, (caught[0] if caught else "stdout is None (no thread error captured)")
    return True, done.stdout[:40]


def capture_teeth_checks(ck):
    """One child, two capture patterns: the fixed path must read the marker, the old pattern
    must be unable to.  Without this the out-of-repo fixture would only ever see a healthy
    child, and a harness that prints "green" for everything passes that test just as well.
    """
    box = Path(tempfile.mkdtemp(prefix="gateA_leg_"))
    child = box / "child.py"
    child.write_bytes(('print("%s")\n' % LEGACY_MARKER).encode("utf-8"))
    rc_new, out_new, err_new, missing_new = run_suite(child, box, timeout=120)
    ok_old, note_old = legacy_capture(child, box)
    ck.add("gateA_oob_teeth_old_capture_cannot_read_the_same_child",
           (not missing_new) and (LEGACY_MARKER in out_new) and (not ok_old),
           {"fixed_readable": not missing_new, "fixed_saw_marker": LEGACY_MARKER in out_new,
            "legacy_readable": ok_old, "legacy_note": note_old, "rc_fixed": rc_new},
           "fixed path reads what the legacy path cannot -> the fixture can actually bite")
    reap(box)


def oob_run_checks(ck):
    """(b) 统括官 11:5x: "run the suite where no .git is above it -> the summary line must
    still print, with rc in {0,2}" -- as a fixture that executes, not as an inference.

    It carries its own tooth: the PUBLISHED bytes come out of git history, run in a
    repository-free box under the same hostile encoding, and really do lose their summary line
    (measured 12:01: rc=1, 52 stdout lines, AttributeError on the None stdout).  Without that
    second half this would be a check that only ever sees a healthy child.
    """
    if os.environ.get("RULER_OOB_PROBED") == "1":
        ck.skip("gateA_oob_run_prints_the_summary_line", "we are the out-of-repo run itself",
                "not re-entered, so the fixture cannot recurse")
        return
    box = Path(tempfile.mkdtemp(prefix="gateA_oob_"))
    if not no_repo_above(box):
        ck.skip("gateA_oob_run_prints_the_summary_line", str(box),
                "the probe box sits under a repository, so it would not be an out-of-repo run")
        reap(box)
        return
    for name in SUITE_FILES:
        shutil.copy2(HERE / name, box / name)
    # RULER_OOB_PROBED stops THIS fixture from recursing; the *_MIRRORED guard is deliberately
    # NOT set, so the copy still executes its own mirror fixture -- that nested child is the
    # thing that used to come back None, and skipping it would test only the easy half.
    rc, out, err, missing = run_suite(box / "selftest_identifiability.py", box,
                                      extra={"RULER_OOB_PROBED": "1"})
    totals = [L for L in out.splitlines() if L.startswith("total=")]
    now = oob_verdict(rc, totals[0] if totals else "")
    ck.add("gateA_oob_run_prints_the_summary_line", now["green"],
           {"rc": rc, "total_line": now["total_line"], "reasons": now["reasons"],
            "stdout_unavailable": missing, "box": box.name,
            "last_line": out.splitlines()[-1][:70] if out.strip() else ""},
           "summary line present and rc in {0,2} with no .git above")

    root = G.repo_root()
    if root is None:
        ck.measure("gateA_oob_old_published_bytes_at_foreign_depth",
                   "unavailable: no repository above this copy, so the published bytes "
                   "cannot be fetched (the mechanism tooth does not depend on them)")
        reap(box)
        return
    old = Path(tempfile.mkdtemp(prefix="gateA_oob_old_"))
    wrote = 0
    for name in SUITE_FILES:
        text = _git_show(OLD_PUBLISHED_COMMIT, f"model/scripts/route2/{name}", root)
        if text is None:
            break
        (old / name).write_bytes(text.encode("utf-8"))
        wrote += 1
    if wrote < len(SUITE_FILES) or not no_repo_above(old):
        ck.measure("gateA_oob_old_published_bytes_at_foreign_depth",
                   {"staged_files": wrote, "expected": len(SUITE_FILES),
                    "box_ok": no_repo_above(old), "note": "old bytes not stageable"})
        reap(box, old)
        return
    rc_old, out_old, err_old, missing_old = run_suite(
        old / "selftest_identifiability.py", old, extra={"RULER_OOB_PROBED": "1"})
    old_totals = [L for L in out_old.splitlines() if L.startswith("total=")]
    ck.measure("gateA_oob_old_published_bytes_at_foreign_depth",
               {"rc": rc_old, "summary_present": bool(old_totals),
                "stdout_lines": len(out_old.splitlines()),
                "crash_line": next((L.strip()[:96] for L in err_old.splitlines()
                                    if "AttributeError" in L), "")})
    reap(box, old)


def mirror_run_checks(ck: Check) -> None:
    """Copy the gate + its self-test (and the two read-only modules the gate imports, as
    copies only) to a directory with no repository above it, and run the copy: rc must be 0
    and the summary line must be printed.  This is the fixture that would have caught the
    depth-counting repo root the 统括官 hit at 11:0x."""
    if os.environ.get("GATEA_MIRRORED") == "1":
        ck.skip("gateA_mirror_run_at_foreign_depth_is_green", "we are the mirror run",
                "not re-entered, so the fixture cannot recurse")
        return
    mirror = Path(tempfile.mkdtemp(prefix="gateA_mirror_"))
    for name in ("identifiability_gate.py", "selftest_identifiability.py",
                 "impedance_baseline.py", "t_geometry.py"):
        shutil.copy2(HERE / name, mirror / name)     # copies only; nothing in the repo is touched
    rc, out, err, missing = run_suite(mirror / "selftest_identifiability.py", mirror,
                                      extra={"GATEA_MIRRORED": "1"})
    if missing:
        # never `.strip()` a value that can be None, and never fold an unreadable child into
        # either verdict: it is counted, printed, and reported as not-judged.
        ck.skip("gateA_mirror_run_at_foreign_depth_is_green",
                {"rc": rc, "stdout_unavailable": True, "stderr_tail": err[-80:]},
                "child output could not be read, so this run judges nothing (not a pass)")
        reap(mirror)
        return
    lines = out.strip().splitlines()
    ck.add("gateA_mirror_run_at_foreign_depth_is_green",
           rc == 0 and "total=" in out and "ALL GREEN" in out,
           {"rc": rc, "summary_present": "total=" in out,
            "last_line": lines[-1][:70] if lines else "", "mirror": mirror.name},
           "rc=0 with a total= line, at a depth where counting parents gives a wrong root")
    reap(mirror)


def hygiene_checks(ck: Check) -> None:
    root = G.repo_root()      # found by walking up for .git, never by counting levels
    # The CLI itself has to be reachable: argparse interpolates '%' inside help strings, so a
    # bare "3%" in a help line makes `--help` die with TypeError.  That is a real defect in a
    # deliverable whose whole job is to be run by someone else, and only a subprocess sees it.
    for label, path in (("gate", HERE / "identifiability_gate.py"),
                        ("selftest", Path(__file__).resolve())):
        rc_h, out_h, err_h, missing_h = run_suite(path, HERE, ("--help",), timeout=120)
        ck.add(f"gateA_cli.{label}_help_reachable", rc_h == 0 and not missing_h,
               {"rc": rc_h, "stdout_unavailable": missing_h,
                "tail": err_h.strip()[-60:]}, "rc == 0")
    if root is None:
        ck.skip("gateA_refuses_to_write_inside_the_repository",
                "no .git above this copy: the guard has nothing to refuse",
                "not attempted here; in the real repo an in-repo target must be refused")
    else:
        ck.add("gateA_refuses_to_write_inside_the_repository",
               G._is_inside_repo(root / "model" / "smoke.json")
               and not G.inside_repo(Path(DEFAULT_OUT)),
               str(root), "repo paths refused, scratch path allowed")
    tmp = Path(tempfile.mkdtemp(prefix="gateA_"))
    target = tmp / "exists.json"
    target.write_text("{", encoding="utf-8")
    refused = _raises(lambda: _try_overwrite(target), SystemExit)
    ck.add("gateA_refuses_to_clobber_a_non_empty_artifact_without_force", refused, refused,
           "raise (a re-run must not silently replace the previous reading)")
    ck.add("gateA_artifact_default_path_is_outside_the_repo",
           root is None or root not in Path(DEFAULT_OUT).resolve().parents, str(DEFAULT_OUT),
           "outside the repository (or no repository in sight to be inside of)")
    # The self-certified digest has to be the digest of the bytes on disk, or every number in
    # a receipt that quotes it is unfalsifiable.  Driving the real CLI is the only way to see
    # this: text mode rewrites \n as \r\n on Windows, so a digest of the string is not a
    # digest of the file.
    cert = Path(tempfile.mkdtemp(prefix="gateA_cert_")) / "cert.json"
    rc_cert, out_cert, err_cert, missing_cert = run_suite(
        HERE / "identifiability_gate.py", HERE, ("--json", str(cert)), timeout=300)
    printed = re.search(r"sha256=([0-9a-f]{64})", out_cert)
    actual = hashlib.sha256(cert.read_bytes()).hexdigest() if cert.is_file() else None
    ck.add("gateA_artifact_selfcert_digest_is_the_digest_of_the_file",
           rc_cert == 0 and printed is not None and printed.group(1) == actual
           and cert.stat().st_size > 0,
           {"printed": printed.group(1)[:16] if printed else None, "file": actual[:16],
            "bytes": cert.stat().st_size if cert.is_file() else None},
           "printed sha256 == sha256(file bytes)")


def _try_overwrite(path: Path) -> None:
    if path.is_file() and path.stat().st_size > 0:
        raise SystemExit(f"refusing to overwrite non-empty {path} (pass --force)")


def main() -> int:
    ap = argparse.ArgumentParser(description="gateA identifiability gate self-test")
    ap.add_argument("--json", default="", help="write the check log here (outside the repo)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ck = Check()
    t0 = time.perf_counter()
    geo = geometry()
    kernel_checks(ck)
    map_checks(ck, geo)
    jacobian_scaling_checks(ck, geo)
    negative_control_checks(ck, geo)
    positive_control_checks(ck, geo)
    threshold_checks(ck)
    rows = report_checks(ck, geo)
    axis_checks(ck)
    hygiene_checks(ck)
    mirror_run_checks(ck)
    oob_run_checks(ck)
    capture_teeth_checks(ck)
    node_rank = G.rank_of(G.jacobian_distributed(geo["theta_node"], geo["lengths"],
                                                  geo["p_in"], ()))
    rich_rank = G.rank_of(G.jacobian_distributed(geo["theta_dist"], geo["lengths"],
                                                  geo["p_in"], G.stations_for(16)))
    mismatch = _scaling_mismatch(geo)
    print(f"[gateA-measure] node_only_rank={node_rank}/4  with_16_stations_rank="
          f"{rich_rank}/{G.N_PARAMS}")
    print(f"[gateA-measure] reference_vs_gate_jacobian_max_rel_dev={mismatch:.3e} "
          f"(non-zero means the 1/theta_i factor came back; rank unaffected either way)")

    # The verdict prints before anything touches the disk: a refused artifact write must not
    # be able to eat the summary line and leave green lines with rc=1 (统括官 11:0x 实测形状)
    _emit(f"total={len(ck.rows)} failed={len(ck.failed)} skipped={len(ck.skipped)}")
    skip_names = ", ".join(ck.skipped) if ck.skipped else "(none)"
    _emit(f"[gateA] skip_context: skipped={len(ck.skipped)} 只在本次上下文成立——凡依赖 .git 或真数据的格子，"
          f"在仓外子箱里诚实 [SKIP]（是\u201c没跑\u201d，不是\u201c跑坏\u201d）；本轮名单={skip_names}")
    if ck.failed:
        _emit("FAILED: " + ", ".join(ck.failed))
    verdict_rc = 1 if ck.failed else 0
    print(f"{'FAILED' if ck.failed else 'ALL GREEN'} elapsed_s={time.perf_counter() - t0:.3f}")
    out = Path(args.json) if args.json else DEFAULT_OUT
    root = G.repo_root()
    if root is not None and G.inside_repo(out):
        print(f"[gateA] artifact_refused: 目标是仓内路径，不写盘: {out}")
        return verdict_rc or 2
    if args.json:
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"checks": ck.rows, "failed": ck.failed, "rows": rows},
                         ensure_ascii=False, indent=2) + "\n"
        if out.is_file() and out.stat().st_size > 0 and not args.force:
            print(f"[gateA] artifact_refused: 目标非空且未加 --force，不覆盖: {out}")
            return verdict_rc or 2
        out.write_bytes(text.encode("utf-8"))
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"json={out}")
        print(f"[gateA] artifact_selfcert: path={out} bytes={out.stat().st_size} sha256={digest}")
    return verdict_rc


def _scaling_mismatch(geo: Dict[str, object]) -> float:
    """Max elementwise relative deviation between this gate's Jacobian and the reference
    module's `jacobian_wrt_params`.  It was 8.33333 = 1/kappa before the A6 fix (that module
    divided each column by theta once too many); this line reads ~0 now, and 1.0e0 in this
    printout would mean the extra factor came back.  Rank is unaffected either way, which is
    why the repo's rank-3 / rank-4 claims were never in question.
    """
    tn = geo["theta_node"]
    a = G.jacobian_distributed(tn, geo["lengths"], geo["p_in"], ())
    b = ib.jacobian_wrt_params(tn, geo["lengths"], geo["p_in"], ())
    dev = [abs(b[i][j] - a[i][j]) / abs(a[i][j]) for i in range(4) for j in range(5)
           if abs(a[i][j]) > 1.0e-12]
    return max(dev) if dev else float("nan")


if __name__ == "__main__":
    sys.exit(main())
