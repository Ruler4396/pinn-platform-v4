#!/usr/bin/env python3
"""Local, dependency-free self-test for the route-2 K0/S1 machinery.

Run:  python model/scripts/route2/selftest_route2_stdlib.py [--json OUT.json]

stdlib only (no numpy / scipy / torch / pandas): this is the part of K0 that can
be falsified on the laptop.  The torch-side half (chain completeness of the two
fixed residual implementations) is k0_truth_gate.py and runs on the instance.

Exit code 0 only if every gate below passes; artifacts default to the scratch
directory outside the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import artifacts as art                             # noqa: E402
import fd_stencils as fd                      # noqa: E402
import impedance_baseline as ib                # noqa: E402
import residual_scorers as rs                  # noqa: E402
import t_geometry as tg                                     # noqa: E402
from t_geometry import (GEOMETRY_FEATURES, STEM, UP, DOWN, TCase, TGeometry,  # noqa: E402
                        border_counts, case_by_id, mesh_levels)

import tempfile                                    # noqa: E402
_IS_WINDOWS = tempfile.gettempdir().startswith(("C:", "D:"))
# a hard-coded Windows default produced a junk file named with backslashes when the same
# script ran on the Linux instance; the default must follow the platform
DEFAULT_OUT = (Path("D:/PINN-restart/.scratch/route2/selftest_k0_s1.json") if _IS_WINDOWS
               else Path(__import__("os").environ.get("ROUTE2_OUT",
                                                      tempfile.gettempdir()))
               / "route2_selftest_k0_s1.json")
SIGMAS = (0.15, 0.30)
_AGREEMENT: dict = {}
# 1e-4 rather than 1e-8: the analytic gradient is exact but one-sided where a
# frame's clearance hits max(0, 1-eta^2)=0, so the central difference has an O(step)
# floor at those points.  sigma=0.30 reaches 1.1e-06, which confirms it is discretisation.
JACOBIAN_TOL_SINGLE_FRAME = 1.0e-4
JACOBIAN_TOL_BLENDED = 1.0e-2
FD_STEP = 1.0e-4


class Check:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.prefix = ""

    def add(self, name: str, ok: bool, value: object, limit: object = "") -> None:
        full = self.prefix + name
        self.rows.append({"check": full, "pass": bool(ok), "value": value, "limit": limit})
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {full}: value={_fmt(value)} limit={_fmt(limit)}")

    def skip(self, name: str, value: object, limit: object = "") -> None:
        """A check that could not run at all.  Counted and printed, never folded into PASS:
        "229 green" and "223 green + 6 not attempted" are different claims, and the
        instance/223-vs-229 gap was exactly that difference."""
        full = self.prefix + name
        self.rows.append({"check": full, "pass": True, "skipped": True,
                          "value": value, "limit": limit})
        print(f"[SKIP] {full}: value={_fmt(value)} limit={_fmt(limit)}")

    @property
    def failed(self) -> list[str]:
        return [r["check"] for r in self.rows if not r["pass"]]

    @property
    def skipped(self) -> list[str]:
        return [r["check"] for r in self.rows if r.get("skipped")]


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)


def geometry_checks(ck: Check, geom: TGeometry) -> None:
    poly = geom.polygon
    ck.add("polygon.CCW_and_positive_area", poly.is_ccw() and poly.area() > 0, poly.area(), "> 0")
    ck.add("polygon.borders_chain_head_to_tail", _borders_chain(poly),
           [(round(b[0], 4), round(b[1], 4)) for a, b in poly.edge_ends][:3], "closed walk")
    agree = _polygon_matches_frames(geom)
    _AGREEMENT.clear()
    _AGREEMENT.update(agree)
    ck.add("polygon.boundary_equals_union_of_frames", agree["mismatch"] == 0,
           agree["mismatch"], "0 cells")
    if geom.case.is_geometrically_symmetric:
        ck.add("polygon.top_bottom_symmetric", agree["symmetry_violations"] == 0,
               agree["symmetry_violations"], "0 cells")
    else:
        ck.add("polygon.crotch_off_axis_for_asymmetric_branches",
               abs(geom.crotch[1]) > 1.0e-3, geom.crotch[1], "|y| > 1e-3")
    ck.add("polygon.area_equals_union_minus_lens",
           abs(poly.area() - (geom.case.l_stem * geom.case.w_stem
                              + geom.case.l_branch * (geom.case.w_branch_up
                                                      + geom.case.w_branch_down)
                              - agree["lens_area"])) < 0.05,
           [round(poly.area(), 4), round(agree["lens_area"], 4)], "12 - lens, tol 0.05")
    ck.add("crotch_downstream_of_junction", geom.crotch[0] > geom.j_point[0],
           geom.crotch[0], f"> {geom.j_point[0]}")
    for p in [(2.0, 0.0), (2.0, 0.4), (4.3536, 0.0), geom.j_point, (5.0, 1.0),
              (5.5, 2.0), (6.8, 2.8)]:
        ck.add(f"contains.inside{tuple(round(c, 4) for c in p)}", geom.contains(*p),
               geom.frames_at(*p), "non-empty")
    for p in [(6.0, 0.0), (5.2, 0.0), (8.5, 3.0), (-0.1, 0.0), (7.5, 0.0), (5.5, 2.6)]:
        ck.add(f"contains.outside{tuple(round(c, 4) for c in p)}", not geom.contains(*p),
               geom.contains(*p), "False")
    ck.add("wall_distance.wall_is_zero",
           all(geom.wall_distance_exact(*p) < 1.0e-9 for p in
               [(2.0, geom.h_stem), (2.0, -geom.h_stem), (1.0, -geom.h_stem)]),
           [round(geom.wall_distance_exact(2.0, geom.h_stem), 12)], "< 1e-9")
    ck.add("wall_distance.centreline_is_half_width",
           abs(geom.wall_distance_exact(2.0, 0.0) - geom.h_stem) < 1.0e-9,
           geom.wall_distance_exact(2.0, 0.0), geom.h_stem)
    if geom.case.is_geometrically_symmetric:
        ck.add("wall_distance.mirror_symmetric",
               abs(geom.wall_distance_exact(5.3, 1.1) - geom.wall_distance_exact(5.3, -1.1)) < 1e-12,
               [geom.wall_distance_exact(5.3, 1.1), geom.wall_distance_exact(5.3, -1.1)], "equal")
    else:
        ck.add("wall_distance.branch_half_widths_differ",
               abs(geom.frames[UP].half_width - geom.frames[DOWN].half_width) > 1e-9,
               [geom.frames[UP].half_width, geom.frames[DOWN].half_width], "up != down")
        ck.add("wall_distance.matches_own_branch_half_width",
               abs(geom.wall_distance_exact(*geom.frames[UP].global_xy(2.0, 0.0))
                   - geom.frames[UP].half_width) < 1e-9,
               geom.wall_distance_exact(*geom.frames[UP].global_xy(2.0, 0.0)),
               geom.frames[UP].half_width)


def _borders_chain(poly) -> bool:
    """FreeFEM's buildmesh only works if consecutive borders join head-to-tail."""
    n = len(poly.edge_ends)
    if n != len(poly.verts):
        return False
    for i in range(n):
        a, b = poly.edge_ends[i]
        if any(abs(a[k] - poly.verts[i][k]) > 1e-12 for k in (0, 1)):
            return False
        if any(abs(b[k] - poly.verts[(i + 1) % n][k]) > 1e-12 for k in (0, 1)):
            return False
    return True


def _polygon_matches_frames(geom: TGeometry, n: int = 180) -> dict:
    """Coarse-cell agreement between the contour walk and the frame union."""
    x0, x1 = -0.2, geom.corners[UP][0][0] + 0.5
    y0, y1 = -geom.corners[UP][0][1] - 0.5, geom.corners[UP][0][1] + 0.5
    cell = (x1 - x0) / n * (y1 - y0) / n
    mismatch = sym_bad = 0
    lens = 0.0
    for i in range(n):
        x = x0 + (x1 - x0) * (i + 0.5) / n
        for j in range(n):
            y = y0 + (y1 - y0) * (j + 0.5) / n
            frames = geom.frames_at(x, y)
            union = bool(frames)
            if len(frames) > 1:
                lens += cell
            if union != geom._in_polygon(x, y, 0.0):  # noqa: SLF001 - deliberate white-box check
                mismatch += 1
            if geom.contains(x, y) != geom.contains(x, -y):
                sym_bad += 1
    return {"mismatch": mismatch, "symmetry_violations": sym_bad, "lens_area": lens}


def jacobian_checks(ck: Check, geom: TGeometry) -> None:
    for sigma in SIGMAS:
        single = _jacobian_error(geom, sigma, prefer="single")
        blended = _jacobian_error(geom, sigma, prefer="blended")
        ck.add(f"jacobian.single_frame.sigma{sigma}", single["max_rel"] <= JACOBIAN_TOL_SINGLE_FRAME,
               single["max_rel"], JACOBIAN_TOL_SINGLE_FRAME)
        print("        single-frame worst @sigma=%s: %s  per-feature: %s" % (
            sigma, single["worst_feature"],
            ", ".join(f"{k}={v:.2g}" for k, v in sorted(
                single["per_feature"].items(), key=lambda kv: -kv[1])[:3])))
        ck.add(f"jacobian.blended_lens.sigma{sigma}", blended["max_rel"] <= JACOBIAN_TOL_BLENDED,
               blended["max_rel"], JACOBIAN_TOL_BLENDED)
        if blended["worst_feature"]:
            print(f"        worst blended feature @sigma={sigma}: {blended['worst_feature']}")
            print("        per-feature max rel err: "
                  + ", ".join(f"{k}={v:.2g}" for k, v in sorted(
                      blended["per_feature"].items(), key=lambda kv: -kv[1])[:4]))
    ck.add("blend.weights_sum_to_one",
           all(abs(sum(w for w, _ in geom.blend_weights(x, y, 0.15).values()) - 1.0) < 1e-9
               for x, y in [(2.0, 0.1), (4.3, 0.2), (5.6, 1.4)]),
           [round(sum(w for w, _ in geom.blend_weights(4.3, 0.2, 0.15).values()), 12)], "1.0")


def _jacobian_error(geom: TGeometry, sigma: float, prefer: str) -> dict:
    """Central-difference check of the analytic feature Jacobian."""
    pts = _sample_points(geom, prefer)
    worst = 0.0
    worst_feat = ""
    per_feature = {name: 0.0 for name in GEOMETRY_FEATURES}
    for x, y in pts:
        ana = geom.features_jacobian(x, y, sigma)
        for name in GEOMETRY_FEATURES:
            for axis, (dx, dy) in enumerate(((FD_STEP, 0.0), (0.0, FD_STEP))):
                fp = geom.features(x + dx, y + dy, sigma)[name]
                fm = geom.features(x - dx, y - dy, sigma)[name]
                num = (fp - fm) / (2.0 * (dx if dx else dy))
                den = max(abs(ana[name][axis]), abs(num), 1.0e-6)
                rel = abs(num - ana[name][axis]) / den
                per_feature[name] = max(per_feature[name], rel)
                if rel > worst:
                    worst, worst_feat = rel, f"{name}{'xy'[axis]}@({x:.3f},{y:.3f})"
    return {"max_rel": worst, "worst_feature": worst_feat, "n_points": len(pts),
            "per_feature": per_feature}


def _sample_points(geom: TGeometry, prefer: str) -> list[tuple[float, float]]:
    pts = []
    n = 26
    for i in range(n):
        for j in range(n):
            x = 0.3 + (geom.crotch[0] - 0.6) * i / (n - 1)
            y = -3.0 + 6.0 * j / (n - 1)
            if not geom.contains(x, y):
                continue
            frames = geom.frames_at(x, y)
            if prefer == "single" and len(frames) == 1:
                fr = geom.frames[frames[0]]
                xi, eta = fr.local(x, y)
                if min(xi, fr.length - xi) < 0.4 or abs(abs(eta) - fr.half_width) < 0.4:
                    continue
            if prefer == "blended" and len(frames) < 2:
                continue
            pts.append((x, y))
    if prefer == "blended" and len(pts) < 5:  # fall back to the overlap band near the junction
        for i in range(60):
            x = geom.j_point[0] - 0.6 + 1.2 * i / 59.0
            for k in range(-8, 9):
                y = 0.06 * k
                if len(geom.frames_at(x, y)) >= 2 and geom.contains(x, y):
                    pts.append((x, y))
    return pts[:400]


def manufactured_field_checks(ck: Check, geom: TGeometry) -> None:
    """An exact Stokes solution must score ~0 corrected and ~huge as-found."""
    sol = rs.Poiseuille(mean_velocity=1.0, half_width=geom.h_stem)
    corr = rs.corrected_terms(abs(sol.laplacian()), abs(sol.grad_p), 0.0)
    ck.add("poiseuille.corrected_residual_is_zero", corr.momentum < 1.0e-24,
           corr.momentum, "< 1e-24")
    ck.add("poiseuille.balance_ratio", abs(corr.balance_ratio - 1.0) < 1.0e-12,
           corr.balance_ratio, "1.0")
    as_found = rs.poiseuille_as_found_score(sol, xi_span=geom.case.l_stem)
    ck.add("poiseuille.as_found_is_pathological", as_found > 10.0 * corr.momentum + 1.0e-3,
           as_found, "> 1e-3 (exact solution scored as a large residual)")
    audit = rs.audit_reproduction()
    for row in audit:
        ck.add(f"audit.reproduces.{row['run']}",
               row["rel_err_vs_audit"] < 5.0e-3, row["as_found_exact_stokes_score"],
               row["audit_reference"])
    # 43.3x amplitude deflation (the 2026-03 pilot failure mode) must be caught
    flat = rs.Poiseuille(mean_velocity=1.0 / 43.3, half_width=geom.h_stem)
    corr_flat = rs.corrected_terms(abs(flat.laplacian()), abs(sol.grad_p), 0.0)
    ck.add("negative_control.deflation_43x_is_caught",
           corr_flat.momentum > 100.0 * max(corr.momentum, 1.0e-30),
           corr_flat.momentum, ">> exact-solution score")


def fd_checks(ck: Check, geom: TGeometry) -> None:
    """FD on a branch-local grid must recover the analytic derivatives."""
    sol = rs.Poiseuille(mean_velocity=1.0, half_width=geom.h_stem)
    spec = geom.branch_grid(STEM, spacing=0.05)
    n_xi, n_eta = spec["n_xi"], spec["n_eta"]
    h0 = (spec["xi1"] - spec["xi0"]) / (n_xi - 1)          # axis 0 = xi
    h1 = 2.0 * spec["eta_max"] / (n_eta - 1)               # axis 1 = eta
    axial = [[0.0] * n_eta for _ in range(n_xi)]
    press = [[0.0] * n_eta for _ in range(n_xi)]
    for i in range(n_xi):
        xi = spec["xi0"] + i * h0
        for j in range(n_eta):
            eta = -spec["eta_max"] + j * h1
            axial[i][j] = sol.axial_velocity(eta)
            press[i][j] = sol.pressure(xi)
    lap = fd.laplacian_2d(axial, h0, h1)
    grad = fd.gradient_2d(press, h0, h1)
    inner_lap = [lap[i][j] for i in range(1, n_xi - 1) for j in range(1, n_eta - 1)]
    inner_gx = [grad[i][j][0] for i in range(1, n_xi - 1) for j in range(1, n_eta - 1)]
    inner_gy = [grad[i][j][1] for i in range(1, n_xi - 1) for j in range(1, n_eta - 1)]
    ck.add("fd.laplacian_exact_for_quadratic",
           fd.rms([v - sol.laplacian() for v in inner_lap]) < 1.0e-6,
           fd.rms([v - sol.laplacian() for v in inner_lap]), "< 1e-6")
    ck.add("fd.pressure_gradient_exact_for_linear",
           fd.rms([v - sol.grad_p for v in inner_gx]) < 1.0e-9,
           fd.rms([v - sol.grad_p for v in inner_gx]), "< 1e-9")
    ck.add("fd.transverse_pressure_gradient_is_zero", abs(fd.rms(inner_gy)) < 1.0e-9,
           fd.rms(inner_gy), "0")
    terms = rs.corrected_terms(fd.rms(inner_lap), fd.rms(inner_gx), 0.0)
    ck.add("fd.manufactured_residual_scores_zero", terms.momentum < 1.0e-20,
           terms.momentum, "< 1e-20")
    ck.add("fd.grid_shape", n_xi >= 5 and n_eta % 2 == 1, [n_xi, n_eta], "n_eta odd, n_xi>=5")

    # A field that is NOT a solution must score large, and the estimate must be
    # stable when the FD step halves (K0-S4's arithmetic on synthetic data).
    def _resid(step: float) -> float:
        n = max(5, int(round((spec["xi1"] - spec["xi0"]) / step)) + 1)
        m = max(3, int(round(2.0 * spec["eta_max"] / step)) | 1)
        g0 = (spec["xi1"] - spec["xi0"]) / (n - 1)
        g1 = 2.0 * spec["eta_max"] / (m - 1)
        f = [[0.0] * m for _ in range(n)]
        q = [[0.0] * m for _ in range(n)]
        for i in range(n):
            xi = spec["xi0"] + i * g0
            for j in range(m):
                eta = -spec["eta_max"] + j * g1
                f[i][j] = sol.axial_velocity(eta) + 0.3 * math.sin(3.0 * xi) * (1 - eta * 4)
                q[i][j] = sol.pressure(xi)
        lp = fd.laplacian_2d(f, g0, g1)
        gr = fd.gradient_2d(q, g0, g1)
        flat_lp = [lp[i][j] for i in range(1, n - 1) for j in range(1, m - 1)]
        flat_gx = [gr[i][j][0] for i in range(1, n - 1) for j in range(1, m - 1)]
        return rs.corrected_terms(fd.rms(flat_lp), fd.rms(flat_gx), 0.0).momentum

    r_fine, r_coarse = _resid(0.05), _resid(0.10)
    gate = rs.fd_step_convergence_gate([r_coarse, r_fine])
    ck.add("fd.step_halving_stable_on_smooth_non_solution", gate["pass"],
           gate.get("rel_change"), gate["limit"])
    ck.add("fd.non_solution_scores_nonzero", r_fine > 1.0e-3, r_fine, "> 1e-3")


def mesh_gate_checks(ck: Check) -> None:
    good = rs.mesh_independence_gate({
        "q_stem": [1.0, 1.004, 1.0005],
        "dp_stem_to_up": [12.0, 11.2, 11.0],
    })
    bad = rs.mesh_independence_gate({
        "p_junction_over_outlet": [3.0, 2.0, 1.4],
    })
    degenerate = rs.mesh_independence_gate({"q_up": [1.0]})
    empty = rs.mesh_independence_gate({})
    ck.add("mesh_gate.passes_when_below_10pct", good["pass"], good["worst_rel_change"],
           good["limit"])
    ck.add("mesh_gate.fails_when_above_10pct", not bad["pass"], bad["worst_rel_change"],
           bad["limit"])
    ck.add("mesh_gate.degenerate_series_is_INVALID_not_true",
           (not degenerate["pass"]) and (not empty["pass"]),
           [degenerate["pass"], empty["pass"]], "[False, False]")
    halve_ok = rs.fd_step_convergence_gate([0.9, 0.87])
    halve_bad = rs.fd_step_convergence_gate([5.0, 0.8])
    ck.add("fd_step_gate", halve_ok["pass"] and not halve_bad["pass"]
           and not rs.fd_step_convergence_gate([1.0])["pass"],
           [halve_ok["rel_change"], halve_bad["rel_change"]], "<0.30 / >=0.30 / single=INVALID")


def k0_verdict_checks(ck: Check) -> None:
    chain = {"plan_a": {"first_total_derivative": 0.001, "second_total_derivative": 0.01},
             "plan_b": {"first_total_derivative": 0.002, "second_total_derivative": 0.02}}
    ok = rs.k0_verdict(0.02, 0.01, chain, rs.fd_step_convergence_gate([0.021, 0.020]), 1.003)
    ck.add("k0_verdict.all_green", ok["pass"], ok["failed"], "[]")
    bad_chain = {"plan_a": dict(chain["plan_a"]), "plan_b": {
        "first_total_derivative": 0.5, "second_total_derivative": 0.02}}
    v2 = rs.k0_verdict(0.02, 0.01, bad_chain, rs.fd_step_convergence_gate([0.021, 0.020]), 1.003)
    ck.add("k0_verdict.catches_asfound_like_chain_failure",
           (not v2["pass"]) and "K0-C_plan_b_first_total_derivative" in v2["failed"],
           v2["failed"], "contains the plan_b first-derivative check")
    v3 = rs.k0_verdict(0.02, 2.0e-9, chain, rs.fd_step_convergence_gate([0.021, 0.020]), 492.0)
    ck.add("k0_verdict.catches_v4_scale_failure",
           (not v3["pass"])
           and {"K0-S1_truth_over_model<=10x", "K0-S2_balance_ratio_in_band"} <= set(v3["failed"]),
           v3["failed"], "both scale checks red")
    v4 = rs.k0_verdict(float("nan"), 0.01, chain, rs.fd_step_convergence_gate([0.021, 0.020]), 1.0)
    ck.add("k0_verdict.nan_is_never_a_pass", not v4["pass"], v4["failed"], "non-empty")


def s1_pipeline_checks(ck: Check, geom: TGeometry, tmp: Path) -> None:
    """End-to-end check of the S1 post-processing on a manufactured Poiseuille truth.

    FreeFEM cannot run here, but everything it hands back -- raw vertex CSV and section
    CSV -> field_dense rows -> Q(xi), pressure drops, mass closure, mesh gate -- is
    exercised on an analytic field whose exact answers are known.  Per-branch mean
    velocities are chosen so the expected section fluxes are Q = mean * width.
    """
    import generate_t_case as gc

    m_up, m_down = (0.5, 0.5) if geom.case.is_geometrically_symmetric else (0.55, 0.45)
    w_up = geom.frames[UP].half_width * 2.0
    w_dn = geom.frames[DOWN].half_width * 2.0
    q_up_exp, q_dn_exp = m_up * w_up, m_down * w_dn
    q_in_exp = q_up_exp + q_dn_exp
    stem = rs.Poiseuille(mean_velocity=q_in_exp / (geom.h_stem * 2.0), half_width=geom.h_stem)
    sol_of = {STEM: stem,
              UP: rs.Poiseuille(mean_velocity=m_up, half_width=geom.frames[UP].half_width),
              DOWN: rs.Poiseuille(mean_velocity=m_down, half_width=geom.frames[DOWN].half_width)}

    raw_rows, sec_rows = [], []
    for key, fr in geom.frames.items():
        sol = sol_of[key]
        n_xi = max(6, int(round(fr.length / 0.2)) + 1)
        n_eta = 51
        for i in range(n_xi):
            xi = i * fr.length / (n_xi - 1)
            for j in range(n_eta):
                eta = -fr.half_width + 2.0 * fr.half_width * j / (n_eta - 1)
                if abs(eta) > fr.half_width + 1e-12:
                    continue
                x, y = fr.global_xy(xi, eta)
                u_ax = sol.axial_velocity(eta)
                u, v = u_ax * fr.d[0], u_ax * fr.d[1]
                p = sol.pressure(xi, p_ref=12.0 * (1.0 - xi / fr.length))
                raw_rows.append([x, y, u, v, p, 4 if abs(abs(eta) - fr.half_width) < 1e-9 else 0])
                sec_rows.append([fr.name, xi, eta, x, y, u, v, p])
    lvl = tmp / "TB-fake_h1"
    lvl.mkdir(parents=True, exist_ok=True)
    raw_p, sec_p = lvl / "raw.csv", lvl / "sections.csv"
    art.write_csv(raw_p, ["x_star", "y_star", "u_star", "v_star", "p_star", "bc_tag"], raw_rows)
    art.write_csv(sec_p, ["branch", "xi", "eta", "x_star", "y_star", "u_star", "v_star",
                          "p_star"], sec_rows)
    art.write_csv(lvl / "summary.csv", ["key", "value"],
                  [["nv", len(raw_rows)], ["nt", 2 * len(raw_rows)],
                   ["q_in_edp", q_in_exp], ["q_up_edp", q_up_exp], ["q_down_edp", q_dn_exp]])
    dense, stats = gc.build_field_dense(raw_p, geom)
    ck.add("s1.field_dense_keeps_all_indomain_points",
           stats["outside"] == 0 and stats["n_kept"] == len(raw_rows), stats, "outside=0")
    kinds = {d["boundary_type"] for d in dense}
    ck.add("s1.boundary_classification_present",
           {"interior", "wall"} <= kinds, sorted(kinds), "at least interior+wall")
    ck.add("s1.features_written_for_plan_b",
           all("feat_" + name in dense[0] for name in GEOMETRY_FEATURES),
           sorted(k for k in dense[0] if k.startswith("feat_"))[:3], "feat_* present")

    ints = gc.section_integrals(sec_p, geom)
    q = gc.quantities(gc._read_summary(lvl / "summary.csv"), ints, [1.0, 1.0])
    ck.add("s1.inlet_flow_recovers_analytic", abs(q["q_stem"] - q_in_exp) < 2.0e-3 * q_in_exp,
           [q["q_stem"], q_in_exp], "Q_in = mean x width")
    ck.add("s1.branch_flows_recover_analytic",
           abs(q["q_up"] - q_up_exp) < 2.0e-3 * q_up_exp
           and abs(q["q_down"] - q_dn_exp) < 2.0e-3 * q_dn_exp,
           [q["q_up"], q_up_exp, q["q_down"], q_dn_exp], "Q = mean x width per branch")
    ck.add("s1.mass_closure_is_a_number_not_a_default",
           q["mass_closure_residual"] == q["mass_closure_residual"]
           and abs(q["mass_closure_residual"]) < 4.0e-3,
           q["mass_closure_residual"], "< 4e-3")
    ck.add("s1.split_fraction_recovers_analytic",
           abs(q["split_fraction_up"] - q_up_exp / q_in_exp) < 2.0e-3,
           [q["split_fraction_up"], q_up_exp / q_in_exp], "Q_up / Q_in")
    dp_expected = stem.pressure(0.0, 12.0) - sol_of[UP].pressure(geom.case.l_branch, 0.0)
    ck.add("s1.dp_matches_analytic_poiseuille",
           abs(q["dp_stem_to_up"] - dp_expected) < 0.05,
           [q["dp_stem_to_up"], dp_expected], "stem p_in - branch p_out")
    expected_split = 0.5 if geom.case.is_geometrically_symmetric else None
    gate = rs.absolute_gates(q["q_stem"], q["q_up"], q["q_down"],
                             q["flux_conservation_max_rel"], expected_split=expected_split)
    ck.add("s1.absolute_gates_pass_on_analytic_field", gate["pass"],
           {k: (round(v["value"], 6) if isinstance(v, dict) else v)
            for k, v in gate.items()}, "all True")
    ck.add("s1.split_gate_marks_inapplicable_when_asymmetric",
           gate["split_sanity"]["applicable"] == geom.case.is_geometrically_symmetric,
           gate["split_sanity"]["applicable"], "matches case symmetry")
    names = tg_names = ("q_stem", "q_up", "q_down", "dp_stem_to_up", "dp_stem_to_down",
                        "p_junction_over_outlet")
    fake_levels = {name: [q[name] * 1.02, q[name] * 1.005, q[name]] for name in names}
    mg = rs.mesh_independence_gate(fake_levels)
    ck.add("s1.mesh_gate_wireable_from_quantities", mg["pass"], mg["worst_rel_change"],
           mg["limit"])
    mg_bad = rs.mesh_independence_gate(dict(fake_levels, q_up=[q["q_up"] * 1.4,
                                                              q["q_up"] * 1.2, q["q_up"]]))
    ck.add("s1.mesh_gate_catches_flow_drift", not mg_bad["pass"],
           [round(mg_bad["worst_rel_change"], 4), mg_bad["worst_quantity"]], "> 0.10 on q_up")


def membership_checks(ck: Check, geom: TGeometry, tmp: Path) -> None:
    """Defect 3 as a positive example: `contains` and frame assignment cannot disagree.

    The real FreeFEM mesh put vertices 1e-8..1e-6 off the inlet plane; `contains` used a
    1e-6 band while `Frame.inside` used 1e-9, so those vertices were "in the domain" with
    no branch and the post-process died.  Sweeping the contour with jitter reproduces the
    band by construction, so this test does not need the instance file to be meaningful --
    and `generate_t_case.py --selfcheck-raw` exists for running the real one.
    """
    import random
    import generate_t_case as gc

    random.seed(11)
    checked = unhandled = 0
    for (a, b), lab in zip(geom.polygon.edge_ends, geom.polygon.edge_labels):
        for k in range(120):
            t = k / 119.0
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            for eps in (0.0, 1.0e-9, 1.0e-8, 1.0e-7, 5.0e-7, 1.0e-6):
                ang = random.uniform(0.0, 2.0 * math.pi)
                xx, yy = x + eps * math.cos(ang), y + eps * math.sin(ang)
                checked += 1
                if geom.contains(xx, yy):
                    try:
                        geom.primary_frame(xx, yy)
                    except ValueError:
                        unhandled += 1
    ck.add("membership.contains_implies_a_frame", unhandled == 0,
           {"samples": checked, "unhandled": unhandled}, "0 unhandled")
    witness = (-1.0e-8, 0.4974937343)
    verdict, key, gap = geom.membership(*witness)
    ck.add("membership.the_shipped_witness_is_absorbed_not_lost",
           key is not None and verdict in set(tg.MEMBERSHIP_VERDICTS) - {"outside"},
           [verdict, key, gap], "must resolve to a branch")
    # drops must be counted, never swallowed
    far = [(9.0, 0.0)]     # one genuine outlier out of 400 => counted, below the halt share
    rows = [[x, y, 0.0, 0.0, 0.0, 0] for x, y in far]
    rows += [[0.2 + 0.009 * i, 0.1, 1.0, 0.0, 5.0, 0] for i in range(400)]
    fake = tmp / "membership_raw.csv"
    art.write_csv(fake, ["x_star", "y_star", "u_star", "v_star", "p_star", "bc_tag"], rows)
    dense, stats = gc.build_field_dense(fake, geom)
    ck.add("membership.rejections_are_counted_not_swallowed",
           stats["outside"] == 1 and len(stats["worst_rejected"]) == 1
           and stats["n_kept"] == len(dense) == 400,
           {"outside": stats["outside"], "kept": stats["n_kept"],
            "worst_gap": stats["worst_rejected"][0]["gap_star"]},
           "1 counted with its gap, 400 kept")
    # and past the halt fraction it must stop rather than quietly keep a partial truth
    bad = [[9.0 + 0.1 * i, 4.0, 0.0, 0.0, 0.0, 0] for i in range(20)]
    bad += [[0.5 + 0.05 * i, 0.0, 1.0, 0.0, 5.0, 0] for i in range(20)]
    fake_bad = tmp / "membership_raw_halt.csv"
    art.write_csv(fake_bad, ["x_star", "y_star", "u_star", "v_star", "p_star", "bc_tag"], bad)
    try:
        gc.build_field_dense(fake_bad, geom)
        halted = False
        detail = "no exception"
    except ValueError as exc:
        halted, detail = True, str(exc)[:70]
    ck.add("membership.over_the_halt_fraction_it_stops", halted, detail,
           f"> {tg.MAX_REJECT_FRAC:.1%} rejected => ValueError")
    ck.add("membership.absorbed_points_are_labelled_in_the_output",
           "membership" in dense[0] and "rect_gap_star" in dense[0],
           sorted(k for k in dense[0] if k in ("membership", "rect_gap_star")), "labels present")
    # a mesh-like vertex set: contour vertices + interior lattice, must post-process clean
    mesh_rows = []
    for (a, b), _ in zip(geom.polygon.edge_ends, geom.polygon.edge_labels):
        n = 60
        for k in range(n + 1):
            f = k / n
            x, y = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            mesh_rows.append([x, y, 1.0, 0.0, -12.0 * x, 0])
    for i in range(70):
        for j in range(40):
            x = 8.0 * i / 69.0
            y = -3.6 + 7.2 * j / 39.0
            if geom.contains(x, y):
                mesh_rows.append([x, y, 1.0, 0.0, -12.0 * x, 0])
    raw = tmp / "mesh_like_raw.csv"
    art.write_csv(raw, ["x_star", "y_star", "u_star", "v_star", "p_star", "bc_tag"], mesh_rows)
    dense2, stats2 = gc.build_field_dense(raw, geom)
    ck.add("membership.mesh_like_vertex_set_post_processes_clean",
           stats2["polygon_without_frame"] == 0 and len(dense2) == stats2["n_kept"] > 1000,
           {"kept": stats2["n_kept"], "absorbed": stats2["absorbed_print"],
            "outside": stats2["outside"], "pwof": stats2["polygon_without_frame"]},
           "no contour-without-frame vertices")
    ck.add("bound_arithmetic_is_the_measured_half_ulp",
           geom.half_ulp(0.114329) == 5.0e-7 and geom.half_ulp(1.91421) == 5.0e-6
           and geom.half_ulp(16.0) == 5.0e-5
           and geom.representation_bound(16.0, 0.0) == 1.0e-4
           and geom.representation_bound(0.5, 0.0) == tg.ABSORB_TOL,
           [geom.representation_bound(16.0, 0.0), geom.representation_bound(7.18, 3.18),
            geom.representation_bound(0.5, 0.0)],
           "2 x half-ulp, floored at ABSORB_TOL; the directive's ~5e-6/8e-6 is 10x this")
    ck.add("bound_does_not_depend_on_mesh_spacing",
           geom.representation_bound(6.5, 3.1) == geom.representation_bound(6.5, 3.1),
           geom.representation_bound(6.5, 3.1), "1e-5 here, and unchanged across levels")
    # ---- the scaling law the instance measured (11.25 / 5.73 / 2.86 %) --------------
    # FreeFEM's ofstream prints 6 significant digits; a wall node has no slack to absorb
    # that, so nodes whose printed coordinate leaves the exact-geometry band are counted
    # as absorbed_print instead of being dropped.  Measured on our own repo file
    # (model/cases/contraction_2d/cfd/C-base/C-base_raw.csv): **all 2113/2113** x_star
    # values are fixed points of a 6-sig-digit round trip, max |x - sig6(x)| = 0 exactly.
    sig6 = gc.significant_digit_roundtrip

    def wall_nodes(h: float) -> list:
        out = []
        for (a, b) in geom.polygon.edge_ends:
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(round(L / h)))
            out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                    for k in range(n + 1)]
        return out

    counts = {}
    for h in (0.16, 0.08, 0.04):
        nodes = wall_nodes(h)
        tally = {v: 0 for v in tg.MEMBERSHIP_VERDICTS}      # derived: cannot go stale
        tally_exact = dict(tally)
        worst = 0.0
        for x, y in nodes:
            verdict, _key, gap = geom.membership(sig6(x), sig6(y))
            tally[verdict] += 1
            worst = max(worst, gap)
            tally_exact[geom.membership(x, y)[0]] += 1
        counts[h] = {"nodes": len(nodes), "absorbed": tally["absorbed_print"],
                     "outside": tally["outside"] + tally["polygon_without_frame"],
                     "outside_exact": tally_exact["outside"] + tally_exact["polygon_without_frame"],
                     "max_gap": worst}
    pred = {h: counts[h]["absorbed"] for h in counts}
    ck.add("membership.printed_wall_nodes_are_absorbed_not_dropped",
           all(counts[h]["outside"] == 0 and counts[h]["absorbed"] > 0 for h in counts)
           and (geom.case.case_id != "TB-base"
                or all(abs(pred[h] - ref) <= 12 for h, ref in
                       ((0.16, 62), (0.08, 124), (0.04, 236)))),
           {str(h): [counts[h]["absorbed"], counts[h]["outside"],
                     "%.1e" % counts[h]["max_gap"]] for h in counts},
           "absorbed ~62/124/236 (instance: 63/123/231 rejected), outside 0")
    ck.add("membership.exact_coordinates_need_no_absorption",
           all(counts[h]["outside_exact"] == 0 for h in counts),
           {str(h): counts[h]["outside_exact"] for h in counts}, "0 at every h")
    ck.add("membership.absorbed_gap_never_exceeds_the_printing_bound",
           all(counts[h]["max_gap"] <= geom.representation_bound(8.0, 4.0) for h in counts),
           ["%.2e" % counts[h]["max_gap"] for h in counts],
           f"<= 2*half-ulp = {geom.representation_bound(8.0, 4.0):.1e}")
    # the same nodes at the old 1e-6-only band is exactly what the instance saw rejected,
    # and the reject *count* must still scale like the perimeter (1/h) => share ~ h
    ratios = [pred[0.08] / pred[0.16], pred[0.04] / pred[0.08]]
    ck.add("membership.rejected_count_scales_like_the_perimeter",
           all(abs(r - 2.0) < 0.25 for r in ratios), [round(r, 3) for r in ratios],
           "doubling per halving of h => count ~ 1/h")
    instance = {0.16: (560, 63), 0.08: (2147, 123), 0.04: (8080, 231)}   # TB-base only
    if geom.case.case_id == "TB-base":
        share = {h: pred[h] / instance[h][0] for h in instance}
        ck.add("membership.share_reproduces_the_instance_with_the_instance_denominator",
               all(abs(share[h] * 100.0 - instance[h][1] * 100.0 / instance[h][0]) < 0.6
                   for h in instance),
               {str(h): [round(share[h] * 100, 2),
                         round(instance[h][1] * 100.0 / instance[h][0], 2)] for h in instance},
               "predicted vs reported share, within 0.6 pp")
        ratios = [share[0.16] / share[0.08], share[0.08] / share[0.04]]
        ck.add("membership.share_halves_with_h_so_the_old_halt_was_right",
               all(abs(r - 2.0) < 0.15 for r in ratios), [round(r, 3) for r in ratios],
               "as reported: 11.25 -> 5.73 -> 2.86 %")
        ck.add("membership.all_three_levels_were_over_the_halt_share",
               all(pred[h] / instance[h][0] > tg.MAX_REJECT_FRAC for h in instance),
               {str(h): round(pred[h] / instance[h][0], 4) for h in instance},
               "> 0.5%: the breaker had to refuse the level, not absorb silently")
    else:
        ck.add("membership.share_check_not_applicable_off_tb-base",
               all(pred[h] > 0 for h in pred), {"absorbed": pred},
               "no instance totals for this case; only the count law is checked")


    # ---- defect 5: the counter and the enum must be one source, and the input must be
    # the artefact's own form.  None of the 194 green assertions had ever fed a real
    # *_raw.csv to build_field_dense, so a stale counter key lived next to a membership()
    # that returned 'absorbed_print' -- and the post-process died 500 s after the solve.
    def print6(v: float) -> str:
        return "%.6g" % v                      # what FreeFEM's ofstream actually writes

    hdr6 = ["x_star", "y_star", "u_star", "v_star", "p_star", "bc_tag"]
    h_probe = 0.08
    art_rows = [[print6(x), print6(y), "1", "0", print6(-12.0 * x), "0"]
                for x, y in wall_nodes(h_probe)]
    n_wall = len(art_rows)
    for i in range(70):
        for j in range(40):
            x, y = 8.0 * i / 69.0, -3.6 + 7.2 * j / 39.0
            if geom.contains(x, y):
                art_rows.append([print6(x), print6(y), "1", "0", print6(-12.0 * x), "0"])
    printed_raw = tmp / f"printed_raw_{geom.case.case_id}.csv"
    art.write_csv(printed_raw, hdr6, art_rows)
    _hdr_p, body_p = art.read_csv_rows(printed_raw)
    ck.add("defect5.the_input_really_is_six_digit_text",
           len(body_p) == len(art_rows) > n_wall > 0
           and all(r[0] == print6(float(r[0])) and r[1] == print6(float(r[1]))
                   for r in body_p),
           [len(body_p), n_wall, body_p[0][:2]],
           "every coordinate is a fixed point of %.6g: memory floats cannot be involved")
    dense_p, stats_p = gc.build_field_dense(printed_raw, geom)
    ck.add("defect5.printed_artefact_runs_end_to_end_and_reports_the_three_numbers",
           stats_p["outside"] == 0 and stats_p["absorbed_print"] > 0
           and 0.0 < stats_p["max_absorbed_gap_star"] <= stats_p["max_absorbed_gap_bound_star"]
           and all(v in stats_p for v in tg.MEMBERSHIP_VERDICTS)
           and len(dense_p) == stats_p["n_kept"] == len(art_rows),
           {"absorbed": stats_p["absorbed_print"], "outside": stats_p["outside"],
            "max_gap_star": "%.2e" % stats_p["max_absorbed_gap_star"],
            "bound_star": "%.2e" % stats_p["max_absorbed_gap_bound_star"],
            "kept": stats_p["n_kept"]},
           "outside=0, a counter for every verdict, the biggest gap swallowed <= its bound")

    one = tmp / "verdict_one.csv"
    art.write_csv(one, hdr6, [[print6(dense_p[0]["x_star"]), print6(dense_p[0]["y_star"]),
                              "1", "0", "-12", "0"]])
    real_membership = geom.membership
    sweep: Dict[str, tuple] = {}
    try:
        for v in tg.MEMBERSHIP_VERDICTS:
            geom.membership = (lambda x, y, _v=v: (_v, None if _v == "outside" else tg.STEM,
                                                   9.9 if _v == "outside" else 1.0e-9))
            try:
                gc.build_field_dense(one, geom)
                sweep[v] = ("returned", "")
            except ValueError as exc:
                sweep[v] = ("ValueError", str(exc)[:40])
            except Exception as exc:                      # KeyError/RuntimeError = broken
                sweep[v] = (type(exc).__name__, str(exc)[:40])
    finally:
        geom.membership = real_membership
    designed = {"polygon_without_frame": "1 vertices inside the contour",
                "outside": "rejected 1/1"}
    ck.add("defect5.counter_keys_are_derived_from_the_verdict_enum",
           set(sweep) == set(tg.MEMBERSHIP_VERDICTS)
           and sweep["frame"][0] == "returned" and sweep["absorbed_print"][0] == "returned"
           and all(sweep[k][1].startswith(t) for k, t in designed.items()),
           {k: list(v) for k, v in sweep.items()},
           "each verdict is counted once; the two that must halt report the count of 1")

    witness_abs = next(d for d in dense_p if d["membership"] == "absorbed_print")
    abs_row = tmp / "verdict_absorbed.csv"
    art.write_csv(abs_row, hdr6, [[print6(witness_abs["x_star"]), print6(witness_abs["y_star"]),
                                  "1", "0", "-12", "0"]])
    full_enum = tg.MEMBERSHIP_VERDICTS
    try:
        tg.MEMBERSHIP_VERDICTS = tuple(v for v in full_enum if v != "absorbed_print")
        try:
            gc.build_field_dense(abs_row, geom)
            neg = "no exception"
        except RuntimeError as exc:
            neg = str(exc)[:46]
        except Exception as exc:
            neg = "wrong type: " + type(exc).__name__
    finally:
        tg.MEMBERSHIP_VERDICTS = full_enum
    pos = gc.build_field_dense(abs_row, geom)[1]
    ck.add("defect5.deleting_a_counter_key_must_raise_by_name",
           neg.startswith("undeclared membership verdict")
           and pos["absorbed_print"] == 1 and pos["outside"] == 0,
           {"with the key removed": neg, "with the enum": [pos["absorbed_print"],
                                                           pos["outside"],
                                                           "%.1e" % pos["max_absorbed_gap_star"]]},
           "shrinking the enum => named RuntimeError, never a KeyError after a solved mesh")

    # the command he actually types, not the function I call: `--selfcheck-raw` died on
    # an unimported `json` once the KeyError stopped covering it up
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(HERE / "generate_t_case.py"), "--case", geom.case.case_id,
         "--selfcheck-raw", str(printed_raw)],
        capture_output=True, text=True, cwd=str(HERE))
    try:
        cli = json.loads(proc.stdout)["stats"]
    except Exception as exc:                         # noqa: BLE001 - report, do not crash
        cli = {"_unparsed": type(exc).__name__, "_stderr": proc.stderr[-160:]}
    ck.add("defect5.selfcheck_raw_cli_runs_and_agrees_with_the_device",
           proc.returncode == 0 and cli.get("outside") == 0
           and cli.get("absorbed_print") == stats_p["absorbed_print"] > 0
           and cli.get("n_kept") == stats_p["n_kept"],
           {"rc": proc.returncode, "absorbed": cli.get("absorbed_print"),
            "outside": cli.get("outside"),
            "max_gap": cli.get("max_absorbed_gap_star"),
            "device_absorbed": stats_p["absorbed_print"]},
           "rc=0 and the same three numbers as the in-process run")

    ck.add("membership.reject_fraction_gate_exists",
           stats2["outside"] / max(stats2["n_vertices"], 1) <= tg.MAX_REJECT_FRAC,
           [stats2["outside"], stats2["n_vertices"], tg.MAX_REJECT_FRAC],
           "below the halt threshold")


def impedance_checks(ck: Check, summaries: Dict[str, dict], tmp: Path) -> None:
    """S2 opponent: device checks that need no numpy, no torch and no instance time.

    The load-bearing claim under test is not "the network fits" but "what the network can
    and cannot identify".  Every threshold below is a number the module prints, so a
    future change to the model has to break one of these to be noticed.
    """
    import impedance_baseline as ib

    # 1) the eigensolver the rank argument leans on, against known spectra
    known = [([[4.0, 0, 0, 0], [0, 3.0, 0, 0], [0, 0, 2.0, 0], [0, 0, 0, 1.0]],
              [4.0, 3.0, 2.0, 1.0], "diagonal"),
             ([[1.0, 1, 1, 1]] * 4, [4.0, 0.0, 0.0, 0.0], "rank_one_repeated_root"),
             ([[2.0, 1.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0],
               [0.0, 0.0, 3.0, -1.0], [0.0, 0.0, -1.0, 3.0]], [4.0, 3.0, 2.0, 1.0],
              "two_blocks")]
    for mat, expect, label in known:
        got = ib.jacobi_eigen([row[:] for row in mat])[0]
        ck.add(f"s2_jacobi.spectrum_{label}",
               all(abs(a - b) < 1.0e-9 for a, b in zip(got, expect)),
               [round(g, 9) for g in got], expect)
    ck.add("s2_rank.node_only_jacobian_is_rank_deficient",
           ib.rank_by_elimination(ib.jacobian_wrt_params(
               [1.0, 0.85, 1.0, 0.12], {"stem": 4.0, "up": 4.0, "down": 4.0}, 1.0, ())) < 4,
           ib.rank_by_elimination(ib.jacobian_wrt_params(
               [1.0, 0.85, 1.0, 0.12], {"stem": 4.0, "up": 4.0, "down": 4.0}, 1.0, ())), "< 4")

    ev = ib.evidence()
    pc = ev["positive_control_with_stations"]
    ck.add("s2_positive_control_recovers_synthetic_theta",
           max(pc["param_errors"].values()) < 1.0e-6,
           {k: round(v, 9) for k, v in pc["param_errors"].items()}, "all < 1e-6")
    nd = ev["node_data_only"]
    ck.add("s2_node_data_matches_observables_but_not_parameters",
           nd["split_up_rel_error"] < 1.0e-9 and nd["kappa_rel_error"] > 0.05
           and nd["w_stem_rel_error"] < 0.05,
           {"split_err": nd["split_up_rel_error"], "kappa_err": round(nd["kappa_rel_error"], 4),
            "w_stem_err": round(nd["w_stem_rel_error"], 5)},
           "observables exact while kappa is off >5%")
    idn = ev["identifiability"]
    ck.add("s2_identifiability_node_only_is_degenerate",
           idn["node_only"]["rank_deficient"]
           and idn["flat_direction_node_only"]["invariant_under_trade_off"],
           {"rank": idn["node_only"]["jacobian_rank"],
            "invariant": idn["flat_direction_node_only"]["invariant_under_trade_off"]},
           "rank<4 and trade-off leaves every observable fixed")
    ck.add("s2_identifiability_one_stem_station_breaks_it",
           (not idn["node_plus_stations"]["rank_deficient"])
           and (not idn["flat_direction_with_stations"]["invariant_under_trade_off"]),
           {"rank": idn["node_plus_stations"]["jacobian_rank"],
            "invariant": idn["flat_direction_with_stations"]["invariant_under_trade_off"]},
           "rank==4 and trade-off moves the stations")
    jc = ev["junction_correction_identifiability"]
    ck.add("s2_junction_correction_free_on_node_data_only",
           jc["sse_free_on_station_data"] < 1.0e-20
           and jc["sse_kappa_fixed_zero_on_station_data"] > 1.0e-4,
           {"sse_free": jc["sse_free_on_station_data"],
            "sse_kappa0": jc["sse_kappa_fixed_zero_on_station_data"]},
           "correction costs nothing at the node, everything at the stations")
    dd = ev["distributed_defect_floor"]
    ck.add("s2_distributed_defect_leaves_a_non_zero_floor",
           dd["sse"] > 1.0e-9 and dd["node_observables_still_matched"] < 1.0e-9
           and dd["worst_station_rel_error"] > 1.0e-4,
           {"sse": dd["sse"], "node_match": dd["node_observables_still_matched"],
            "worst_station_rel_error": dd["worst_station_rel_error"]},
           "node exact, along-stem shape missed")
    ck.add("s2_distributed_defect_drives_effective_widths_nonsense",
           abs(dd["fit"]["theta"]["w_stem"] - 1.0) > 0.2,
           dd["fit"]["theta"], "w_stem far from the true 1.0 while nodes are matched")
    nz = ev["noise_3pct_with_stations"]["param_errors"]
    ck.add("s2_three_percent_noise_keeps_wide_and_branch_widths",
           nz["w_up"] < 0.05 and nz["w_dn"] < 0.05 and nz["w_stem"] < 0.05,
           {k: round(v, 4) for k, v in nz.items()}, "widths < 5%; kappa is the weak one")
    fo = ev["field_output_check"]
    ck.add("s2_field_output_finite_and_comparable",
           fo["all_finite"] and fo["n_points"] > 100,
           {"n_points": fo["n_points"], "rel_l2_u": fo["rel_l2_u"]}, "same metric as S3")
    # ---- ruling R2-2: convergence must be stated and tested, not assumed -----------
    arms = {"positive": pc, "noise": ev["noise_3pct_with_stations"],
            "defect": dd}
    for label, arm in arms.items():
        fit = arm["fit"]
        ck.add(f"s2_convergence.{label}_reaches_first_order_optimality",
               fit["converged"] and fit.get("stop_reason") in
               ("first_order_optimality", "exact_consistency"),
               {"converged": fit["converged"], "stop_reason": fit.get("stop_reason"),
                "iterations": fit["iterations"], "grad_norm": fit.get("gradient_norm_inf")},
               "converged via a stated criterion")
    ck.add("s2_convergence.all_starts_converge_and_agree",
           ev["noise_3pct_with_stations"]["every_start_converged"]
           and len(ev["noise_3pct_with_stations"]["stop_reasons"]) == 3,
           ev["noise_3pct_with_stations"]["stop_reasons"], "3 starts, all converged")
    ck.add("s2_convergence.the_old_false_negative_is_gone",
           ev["noise_3pct_with_stations"]["fit"]["iterations"] < 21,
           ev["noise_3pct_with_stations"]["fit"]["iterations"],
           "<21: the 20:0x reading stalled on the damping ceiling, not on the optimum")
    tiers = ev["observation_tiers"]
    ck.add("s2_tiers_TA_is_rank_deficient_TB_is_not",
           tiers["T-A"]["jacobian_rank"] < 4 and tiers["T-B"]["jacobian_rank"] == 4,
           {k: tiers[k]["jacobian_rank"] for k in ("T-A", "T-B")}, "3 then 4")
    ck.add("s2_tiers_TB_station_count_equals_the_proved_minimum",
           len(tiers["T-B"]["probes"]) == 7, len(tiers["T-B"]["probes"]),
           "3 stem + 4 branch stations, from the rank condition")
    try:
        ib.require_provenance(0.01, None)
        refused = False
    except ValueError:
        refused = True
    ib.require_provenance(0.01, "µPIV repeatability reference (slot)")
    ib.require_provenance(0.03, None)
    ck.add("s2_noise_gate_1pct_without_provenance_is_refused", refused, refused, "raise")
    obs = [float(i) for i in range(len(ib.OBS_NAMES_FULL))]
    fA = tmp / "obs_TA.csv"
    fB = tmp / "obs_TB.csv"
    shaA = ib.write_obs_table(fA, ib.observables(theta := [1.0, 0.85, 1.0, 0.12],
                                {"stem": 4.0, "up": 4.0, "down": 4.0}, 30.0, ()),
                              {"tier": "T-A", "lengths": {"stem": 4.0, "up": 4.0,
                                                          "down": 4.0},
                               "p_in": 30.0, "probes": [], "noise_frac": 0.03})
    shaB = ib.write_obs_table(fB, ib.observables(theta, {"stem": 4.0, "up": 4.0,
                                                         "down": 4.0}, 30.0,
                                                 ib.OBS_TIERS["T-B"]),
                              {"tier": "T-B", "lengths": {"stem": 4.0, "up": 4.0,
                                                          "down": 4.0},
                               "p_in": 30.0,
                               "probes": [list(x) for x in ib.OBS_TIERS["T-B"]],
                               "noise_frac": 0.03})
    back, meta, sha_again = ib.read_obs_table(fA)
    ck.add("s2_obs_table_roundtrip_hash_is_stable",
           sha_again == shaA and len(back) == 5 + 0 and meta["tier"] == "T-A",
           [shaA[:12], sha_again[:12], len(back), meta["tier"]], "same sha256 on re-read")
    ck.add("s2_obs_table_tiers_are_different_bytes", shaA != shaB, [shaA[:12], shaB[:12]],
           "different data => different hash")
    ck.add("s2_obs_table_pinned_hash_defends_the_same_data_claim",
           ib.read_obs_table(fB)[1]["probes"] == [list(x) for x in ib.OBS_TIERS["T-B"]],
           ib.read_obs_table(fB)[1]["probes"][:2], "tier provenance travels with the file")


    asym = summaries.get("TB-asym", {})
    ck.add("s2_adversary_geometry_is_the_asymmetric_one",
           "adversary" in asym.get("metadata", {}).get("role", "")
           and "self-check" in summaries.get("TB-base", {}).get("metadata", {}).get("role", ""),
           [asym.get("metadata", {}).get("role"),
            summaries.get("TB-base", {}).get("metadata", {}).get("role")],
           "TB-asym = adversary table, TB-base = self-check")


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True


def k0_truth_side_checks(
        ck: Check, geom: TGeometry, tmp: Path) -> None:
    """Run the real K0 truth-half code on an exact Stokes solution written in S1 layout.

    The stem Poiseuille field u = 1.5(1-4y^2), v = 0, p = -12x is an exact solution of
    Stokes on the whole plane, so after rotation into a branch frame it stays exact and
    central differences are exact for it (quadratic).  score(truth) must therefore be
    ~0 and the balance ratio ~1 -- if this fails, the gate is broken, not the model.
    """
    import k0_truth_gate as kg

    root = tmp / "k0_case_root"
    level = "htest"
    case_id = geom.case.case_id
    d = root / "cfd" / f"{case_id}_{level}"
    d.mkdir(parents=True, exist_ok=True)
    dpdx = -12.0

    def field_global(x: float, y: float):
        return 1.5 * (1.0 - 4.0 * y * y), 0.0, dpdx * x

    for key in (STEM, UP, DOWN):
        fr = geom.frames[key]
        for mult, tag in ((0.05, "h"), (0.10, "h2")):
            spec = geom.branch_grid(key, mult)
            rows = []
            for x, y in geom.grid_points(spec):
                u, v, p = field_global(x, y)
                xi, eta = fr.local(x, y)
                rows.append([x, y, u, v, p, xi, eta, fr.name])
            art.write_csv(d / f"{case_id}_{level}_samples_{fr.name}_{tag}.csv",
                          ["x_star", "y_star", "u_star", "v_star", "p_star", "xi", "eta",
                           "branch"], rows)
    for mult, tag in ((0.05, "h"), (0.10, "h2")):
        spec = geom.junction_grid(mult)
        rows = []
        for x, y in geom.grid_points(spec):
            u, v, p = field_global(x, y)
            rows.append([x, y, u, v, p, x, y, "junction"])
        art.write_csv(d / f"{case_id}_{level}_samples_junction_{tag}.csv",
                      ["x_star", "y_star", "u_star", "v_star", "p_star", "xi", "eta",
                       "branch"], rows)
    score = kg.truth_score(root, case_id, level, geom)
    ck.add("k0_truth_side.exact_solution_scores_zero", score["momentum_mse"] < 1.0e-20,
           score["momentum_mse"], "< 1e-20")
    ck.add("k0_truth_side.balance_ratio_is_one", abs(score["balance_ratio"] - 1.0) < 0.01,
           score["balance_ratio"], "1.0 +- 1%")
    ck.add("k0_truth_side.fd_step_gate_passes", score["fd_step_gate"]["pass"],
           score["fd_step_gate"].get("rel_change"), rs.K0_FD_STEP_HALVE_MAX)
    ck.add("k0_truth_side.crotch_disk_excluded_and_stencils_survive",
           score["n_stencils"] > 3000 and score["per_grid"]["junction"]["h"]["n_stencils"] > 50,
           {b: score["per_grid"][b]["h"]["n_stencils"] for b in score["per_grid"]},
           "all grids non-empty")
    ck.add("k0_truth_side.continuity_scores_zero", score["continuity_mse"] < 1.0e-20,
           score["continuity_mse"], "< 1e-20")
    # a deflated-amplitude field (the 2026-03 failure mode) must be caught by the same code
    for key in (STEM, UP, DOWN):
        fr = geom.frames[key]
        for mult, tag in ((0.05, "h"), (0.10, "h2")):
            spec = geom.branch_grid(key, mult)
            rows = [[x, y, 0.0, 0.0, dpdx * x, *fr.local(x, y), fr.name]
                    for x, y in geom.grid_points(spec)]
            art.write_csv(d / f"{case_id}_{level}_samples_{fr.name}_{tag}.csv",
                          ["x_star", "y_star", "u_star", "v_star", "p_star", "xi", "eta",
                           "branch"], rows)
    plug = kg.truth_score(root, case_id, level, geom)
    ck.add("k0_truth_side.catches_plug_flow_with_wrong_pressure",
           plug["momentum_mse"] > 1.0e2, plug["momentum_mse"], "> 1e2 (|grad p| = 12)")
    ck.add("k0_truth_side.balance_ratio_flags_the_mismatch",
           abs(plug["balance_ratio"] - 1.0) > 0.5, plug["balance_ratio"], "far from 1")

    # ---- the same gate against the artefact's own form: 6 significant digits ----------
    # Everything above wrote full-precision floats, so K0 -- alone among the modules --
    # never touched a file in the shape FreeFEM actually writes.  That is exactly how the
    # instance found it: the 6-digit jitter on the station coordinates (spread 1.2e-5 on
    # branch_up at h3, 1.1e-5 on the graded stem_h2) tripped `_uniform_step`, which asked
    # for 4e-8.  So: rewrite every grid at the digits the solver prints, and let the
    # numbers say what is representation and what is the method.
    def rewrite_grids(digits: int) -> None:
        def pr(v: float) -> str:
            return "%.*g" % (digits, v)
        for key in (STEM, UP, DOWN):
            fr = geom.frames[key]
            for mult, tag in ((0.05, "h"), (0.10, "h2")):
                rows = []
                for x, y in geom.grid_points(geom.branch_grid(key, mult)):
                    u, v, p = field_global(x, y)
                    xi, eta = fr.local(x, y)
                    rows.append([pr(q) for q in (x, y, u, v, p, xi, eta)] + [fr.name])
                art.write_csv(d / f"{case_id}_{level}_samples_{fr.name}_{tag}.csv",
                              ["x_star", "y_star", "u_star", "v_star", "p_star", "xi",
                               "eta", "branch"], rows)
        for mult, tag in ((0.05, "h"), (0.10, "h2")):
            rows = [[pr(q) for q in (x, y, *field_global(x, y), x, y)] + ["junction"]
                    for x, y in geom.grid_points(geom.junction_grid(mult))]
            art.write_csv(d / f"{case_id}_{level}_samples_junction_{tag}.csv",
                          ["x_star", "y_star", "u_star", "v_star", "p_star", "xi", "eta",
                           "branch"], rows)

    rewrite_grids(6)
    printed = kg.truth_score(root, case_id, level, geom)
    worst = printed["axis_uniformity_worst"]
    ck.add("k0_printed.six_digit_lattice_survives_the_uniformity_guard",
           printed["n_stencils"] > 3000 and worst["share_of_printing_bound"] <= 1.0,
           {"which": worst["which"], "spread": "%.2e" % worst["spread_star"],
            "printing_bound": "%.1e" % worst["printing_bound_star"],
            "share_of_bound": round(worst["share_of_printing_bound"], 3)},
           "4 x half-ulp at |xi|max=4 is 2e-5; the instance saw spreads 1.2e-5 / 1.1e-5")
    # the floor this file size implies: u is rounded to 6 digits, so the second difference
    # carries ~4*half_ulp(u)/h^2 -- not a tuned number, and asserted below as a bound
    rewrite_grids(7)
    printed7 = kg.truth_score(root, case_id, level, geom)
    ratio = printed["momentum_mse"] / max(printed7["momentum_mse"], 1e-300)
    ck.add("k0_printed.residual_floor_is_the_printing_not_the_method",
           printed["momentum_mse"] > 1.0e-18 and 60.0 < ratio < 170.0,
           {"mse at 6 digits": "%.3e" % printed["momentum_mse"],
            "mse at 7 digits": "%.3e" % printed7["momentum_mse"],
            "ratio": round(ratio, 1),
            "why 100": "error ~ delta_u/h^2, so mse falls 10^2 per extra printed digit"},
           "a representation floor that falls 100x per digit is printing, not the stencil")
    ck.add("k0_printed.guard_still_bites_a_real_displacement",
           all(_raises(lambda vv=vals: kg._uniform_step(vv, Path("fake.csv")))
               for vals in ([0.0, 0.05, 0.10 + 1.0e-3, 0.15],
                            [0.0, 0.04, 0.08 - 2.0e-3, 0.12])),
           "a 1e-3 node displacement (>> 4*half_ulp) must raise",
           "positive control for the widened tolerance: it is derived, not loose")
    ck.add("k0_printed.tolerance_is_the_derived_bound_not_a_fudge",
           abs(tg.spacing_tolerance([0.0, 4.0]) - 2.0e-5) < 1e-20
           and abs(tg.spacing_tolerance([0.0, 0.4]) - 2.0e-6) < 1e-25
           and abs(tg.spacing_tolerance([0.0, 4.0], digits=17) - tg.ABSORB_TOL) < 1e-20,
           {"|xi|max=4, 6 digits": tg.spacing_tolerance([0.0, 4.0]),
            "|xi|=0.4, 6 digits": tg.spacing_tolerance([0.0, 0.4]),
            "17 digits -> floor": tg.spacing_tolerance([0.0, 4.0], digits=17)},
           "4*half_ulp per the derivation; at 17 digits it collapses to ABSORB_TOL")


def meshing_checks(ck: Check, geom: TGeometry) -> None:
    levels = mesh_levels()
    ck.add("levels.three_global_plus_graded",
           [lv["name"] for lv in levels] == ["h1", "h2", "h3", "hgrade"],
           [lv["name"] for lv in levels], "h1,h2,h3,hgrade")
    counts = {lv["name"]: border_counts(geom, lv["spacing"], lv["graded"]) for lv in levels}
    fine_vs_coarse = all(sum(counts["h2"]) > sum(counts["h1"]) and
                         sum(counts["h3"]) > sum(counts["h2"]) for _ in [0])
    ck.add("levels.borders_refine_monotonically", fine_vs_coarse,
           {k: sum(v) for k, v in counts.items()}, "increasing total")
    import generate_t_case as gc
    lvl0 = dict(mesh_levels()[0], counts=border_counts(geom, mesh_levels()[0]["spacing"], False))
    text = gc.render_edp(geom, geom.case, lvl0, Path("lintcase"))
    problems = gc.lint_edp(text)
    ck.add("edp_lint.rendered_h1_is_clean", problems == [], problems[:4], "no problems")
    sick = '  fo << "a," << xi << ","\n     << u(xx,yy) << endl;\n'
    ck.add("edp_lint_catches_the_shipped_defect",
           any("continuation" in p or "mid-expression" in p
               for p in gc.lint_edp("{" + sick + "}")),
           gc.lint_edp("{" + sick + "}")[:2], "must flag the split stream statement")
    ck.add("edp_lint_catches_unbalanced_and_CR",
           len(gc.lint_edp("int i = 0;\n{\n")) >= 1 and len(gc.lint_edp("a\r\n")) >= 1,
           [gc.lint_edp("int i = 0;\n{\n"), gc.lint_edp("a\r\n")], "brace + CR flagged")
    ck.add("levels.graded_densest_near_junction",
           counts["hgrade"][:2] > [c // 2 for c in counts["h1"][:2]],
           counts["hgrade"][:2], "> half of h1 stem-wall counts")


_OFSTREAM = re.compile(r'ofstream\s+(\w+)\("([^"]+)"\);')
_HEADER_LINE = re.compile(r'^\s*(\w+)\s*<<\s*"([a-z_0-9,]+)"\s*<<\s*endl')
_SUMMARY_LABEL = re.compile(r'^\s*(\w+)\s*<<\s*"([a-z_0-9_]+),"')


def _edp_artifacts(edp: Path):
    """(path, header) for every ofstream in a rendered .edp, in the order it writes them.

    Reading the shipped script rather than a list I keep by hand is the point: if the
    emitter renames or reorders an artefact, the stand-in solver below follows and the
    runtime path stays covered -- defect 5 was exactly a hand-kept list going stale.
    """
    out, pending = [], None
    for line in edp.read_text(encoding="utf-8").splitlines():
        m = _OFSTREAM.search(line)
        if m:
            pending = (m.group(1), m.group(2))
            continue
        if pending:
            h = _HEADER_LINE.match(line)
            if h and h.group(1) == pending[0]:
                out.append((pending[1], h.group(2)))
                pending = None
    return out


def _standin_solver(geom: TGeometry, spacing: float):
    """A fake FreeFem++ that writes the artefacts the real .edp asks for, at 6 digits.

    Returns `run(cmd, *a, **kw)`; patch it over `generate_t_case.subprocess.run` to drive
    run_case(execute=True) through every post-processing line without an instance.
    """
    import os
    import subprocess

    level_h = {lv["name"]: lv["spacing"] for lv in mesh_levels(spacing)}

    m_up, m_down = (0.5, 0.5) if geom.case.is_geometrically_symmetric else (0.55, 0.45)
    q_in = (m_up * 2.0 * geom.frames[UP].half_width
            + m_down * 2.0 * geom.frames[DOWN].half_width)
    sol_of = {STEM: rs.Poiseuille(mean_velocity=q_in / (2.0 * geom.h_stem),
                                  half_width=geom.h_stem),
              UP: rs.Poiseuille(mean_velocity=m_up, half_width=geom.frames[UP].half_width),
              DOWN: rs.Poiseuille(mean_velocity=m_down,
                                  half_width=geom.frames[DOWN].half_width)}

    def print6(v: float) -> str:
        return "%.6g" % v

    def field(x: float, y: float):
        _v, key, _g = geom.membership(x, y)
        if key is None:
            return None
        fr = geom.frames[key]
        xi, eta = fr.local(x, y)
        u_ax = sol_of[key].axial_velocity(eta)
        return (u_ax * fr.d[0], u_ax * fr.d[1],
                sol_of[key].pressure(xi, p_ref=12.0 * (1.0 - xi / fr.length)), xi, eta, fr)

    def vertices(h: float):
        pts = []
        for (a, b) in geom.polygon.edge_ends:
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(round(L / h)))
            pts += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
                    for k in range(n + 1)]
        step = h / 2.0
        nx = int(round(8.0 / step)) + 1
        ny = int(round(7.2 / step)) + 1
        pts += [(i * step, -3.6 + j * step) for i in range(nx) for j in range(ny)
                if geom.contains(i * step, -3.6 + j * step)]
        return pts

    def rows_for(path: Path, header: str, text: str, h: float):
        cols = header.split(",")
        name = Path(path).name
        if cols[:2] == ["x_star", "y_star"] and "bc_tag" in cols:
            out = []
            for x, y in vertices(h):
                st = field(x, y)
                if st is None:
                    continue
                u, v, p, xi, eta, fr = st
                tag = 0
                if fr.key == STEM and abs(xi) <= 1e-9:
                    tag = 1
                elif fr.key in (UP, DOWN) and abs(xi - fr.length) <= 1e-9:
                    tag = 2 if fr.key == UP else 3
                out.append([print6(t) for t in (x, y, u, v, p)] + [str(tag)])
            return cols, out
        if cols == ["key", "value"]:
            labels = [_SUMMARY_LABEL.match(l).group(2)
                      for l in text.splitlines() if _SUMMARY_LABEL.match(l)]
            vals = {"nv": len(vertices(h)), "nt": 2 * len(vertices(h)),
                    "q_in_edp": q_in, "q_up_edp": m_up * 2.0 * geom.frames[UP].half_width,
                    "q_down_edp": m_down * 2.0 * geom.frames[DOWN].half_width}
            return cols, [[l, print6(vals.get(l, 1.0))] for l in labels]
        if cols[-1] == "branch":                 # the 8 structured sample grids
            stem = name.split("_samples_")[1]
            br, tag = stem.rsplit("_", 1)
            key = {"stem": STEM, "branch_up": UP, "branch_down": DOWN}.get(br)
            spec = (geom.junction_grid(h if tag == "h" else 2.0 * h)
                    if key is None else geom.branch_grid(key, h if tag == "h" else 2.0 * h))
            out = []
            for x, y in geom.grid_points(spec):
                st = field(x, y)
                if st is None:
                    continue
                u, v, p, xi, eta, _fr = st
                out.append([print6(t) for t in (x, y, u, v, p, xi, eta)] + [br])
            return cols, out
        if cols[0] == "branch":                  # cross-section profiles
            out = []
            for key in (STEM, UP, DOWN):
                fr = geom.frames[key]
                xi0 = 0.0 if key == STEM else 0.25
                n_sta = int(round((fr.length - xi0) / 0.25)) + 1
                for s in range(n_sta):
                    xi = xi0 + s * 0.25
                    j = 0
                    eta = -fr.half_width
                    while eta <= fr.half_width + 1e-12:
                        x, y = fr.global_xy(xi, eta)
                        u_ax = sol_of[key].axial_velocity(eta)
                        out.append([fr.name] + [print6(t) for t in (
                            xi, eta, x, y, u_ax * fr.d[0], u_ax * fr.d[1],
                            sol_of[key].pressure(xi, p_ref=12.0 * (1.0 - xi / fr.length)))])
                        j += 1
                        eta = -fr.half_width + j * 0.02
            return cols, out
        raise AssertionError(f"stand-in solver has no writer for {name} columns {cols}")

    state = {"t0": time.time(), "i": 0}

    def run(cmd, *a, **kw):
        cmd = [str(c) for c in cmd]
        edp = next((c for c in cmd if c.endswith(".edp")), None)
        if edp is None:                          # the manifest's version probe
            return subprocess.CompletedProcess(cmd, 0, "4.9-fake\n", "")
        text = Path(edp).read_text(encoding="utf-8")
        h = level_h.get(Path(edp).stem.split("_")[-1], spacing)
        for path, header in _edp_artifacts(Path(edp)):
            cols, rws = rows_for(path, header, text, h)
            art.write_csv(Path(path), cols, rws)
            # stagger the mtimes so the plan's artifact_age_s column is tested against a
            # known spread instead of against "the OS happened to tick"
            state["i"] += 1
            age = state["t0"] + 1.7 * state["i"]
            os.utime(Path(path), (age, age))
        return subprocess.CompletedProcess(cmd, 0, "fake solve ok\n", "")

    return run


def runtime_path_checks(ck: Check, geom: TGeometry, tmp: Path) -> None:
    """run_case(execute=True) end to end against the stand-in solver -- the path the
    device had never taken, which is why 194 green assertions coexisted with a product
    that died on its first real post-process.
    """
    import generate_t_case as gc

    out_root = tmp / "s1_runtime_subset_h1_h2"      # subset name: never a full-run name
    levels = [lv for lv in mesh_levels(0.16) if lv["name"] in ("h1", "h2")]
    real_exec, real_run = gc.freefem_executable, gc.subprocess.run
    gc.freefem_executable = lambda: sys.executable
    gc.subprocess.run = _standin_solver(geom, 0.16)
    try:
        res = gc.run_case(geom.case, out_root, levels, True, sigma=0.15, coord_digits=0)
    except Exception as exc:                        # noqa: BLE001 - report, do not crash
        res = {"plan": {"_raised": f"{type(exc).__name__}: {exc}"}}
    finally:
        gc.freefem_executable, gc.subprocess.run = real_exec, real_run
    plan = res.get("plan", {})
    ck.add("runtime.run_case_execute_true_completes", "_raised" not in plan,
           plan.get("_raised", [e["level"] + ":" + e["status"] for e in plan.get("levels", [])]),
           "no exception from the real solve/post-process/manifest path")
    if "_raised" in plan:
        return
    ages = plan["levels"][0].get("artifact_age_s", {})
    ck.add("runtime.artifact_age_column_is_live_and_ordered",
           len(ages) == 11 and max(ages.values()) > 5.0
           and sorted(ages.values())[0] > 0.0,
           {"n": len(ages), "first": min(ages.values()), "last": max(ages.values())},
           "11 artefacts, the staggered spread readable from it")
    timing = plan.get("timing_s", {})
    ck.add("runtime.stage_timers_accumulate",
           all(k in timing for k in ("render_s", "solve_s", "postprocess_s", "manifest_s"))
           and timing["postprocess_s"] > 0.0 and timing["solve_s"] > 0.0,
           timing, "postprocess_s and solve_s non-zero (an inert timer would read 0.0)")
    stats = plan["levels"][0]["dense_stats"]
    ck.add("runtime.solved_level_reports_the_membership_triple",
           stats["outside"] == 0 and stats["absorbed_print"] > 0
           and stats["bc_tag_disagree"] == 0 and stats["n_kept"] == stats["n_vertices"],
           {k: stats[k] for k in ("outside", "absorbed_print", "polygon_without_frame",
                                  "bc_tag_disagree", "n_kept", "n_vertices")},
           "outside=0, absorbed>0, the reader's inlet/outlet tags agree with the artefact")
    # the paired negative: the same vertices classified with the old fixed 1e-6 band
    raw_lvl = (out_root / "cfd" / f"{geom.case.case_id}_h1"
               / f"{geom.case.case_id}_h1_raw.csv")
    _h, rows_r = art.read_csv_rows(raw_lvl)

    def dirichlet_count(tol_of) -> int:
        n = 0
        for r in rows_r:
            x, y = float(r[0]), float(r[1])
            _v, key, _g = geom.membership(x, y)
            if key is None:
                continue
            fr = geom.frames[key]
            xi, _eta = fr.local(x, y)
            tol = tol_of(x, y)
            if (fr.key in (UP, DOWN) and abs(xi - fr.length) <= tol) or \
               (fr.key == STEM and abs(xi) <= tol):
                n += 1
        return n

    n_old = dirichlet_count(lambda x, y: tg.ABSORB_TOL)
    n_new = dirichlet_count(geom.representation_bound)
    ck.add("boundary.ruler_is_the_printing_bound_not_a_fixed_band",
           n_new > n_old > 0 and stats["bc_tag_disagree"] == 0,
           {"with the 1e-6 band": n_old, "with the printing bound": n_new},
           "the old band loses rotated-outlet vertices to 6-digit printing (defect 6)")
    ck.add("runtime.mesh_gate_reached_with_two_levels",
           len(plan.get("level_order", [])) == 2
           and plan.get("mesh_independence", {}).get("relative", {}).get("quantities"),
           {"levels": plan.get("level_order"),
            "worst_rel": plan.get("mesh_independence", {}).get("relative", {})
                        .get("worst_rel_change"),
            "pass": plan.get("mesh_independence", {}).get("relative", {}).get("pass")},
           "the <10% gate actually computed over h1,h2")
    q1 = plan.get("quantities_by_level", {}).get(plan["level_order"][0], {})
    ck.add("runtime.quantities_report_gated_and_disclosed_flux",
           q1.get("flux_stations_excluded", 0) >= 1
           and set(q1.get("flux_per_branch", {})) == {"stem", "branch_up", "branch_down"}
           and q1["flux_conservation_max_rel"]
           <= q1["flux_conservation_all_stations_rel"] + 1e-15,
           {"gated": q1.get("flux_conservation_max_rel"),
            "all_stations": q1.get("flux_conservation_all_stations_rel"),
            "n_gated": q1.get("flux_stations_gated"),
            "n_excluded": q1.get("flux_stations_excluded"),
            "excluded_xi": {b: v["excluded_xi"]
                            for b, v in q1.get("flux_per_branch", {}).items()}},
           "junction stations excluded from the gate, still reported per branch")
    ck.add("runtime.artefacts_landed_where_the_manifest_hashes_them",
           (out_root / "data" / geom.case.case_id / "field_dense.csv").is_file()
           and (out_root / "data" / geom.case.case_id / "mesh_independence.json").is_file()
           and plan.get("manifest_files", 0) >= 24,
           plan.get("manifest_files"), ">=24 files hashed")
    man = art.read_json(out_root / "data" / geom.case.case_id / "sha256sums.json")
    ck.add("runtime.manifest_carries_the_freefem_startup_probe",
           man.get("env", {}).get("freefem_version") == "4.9-fake"
           and isinstance(man.get("env", {}).get("freefem_version_probe_wall_s"), float),
           {k: man.get("env", {}).get(k) for k in ("freefem_version",
                                                   "freefem_version_probe_wall_s")},
           "the column that answers 'was the 497 s process start-up' is populated")


def _find_tracked(name: str, *relatives: Path) -> Path:
    """Locate a repo artefact without assuming the checkout took the whole repository.

    The instance ran `git checkout <sha> -- model/scripts/route2` (my own delivery line),
    so anything under `model/cases/` was simply not there: the fixture assertion went red
    on Linux and 6 checks never registered (223/1 instead of 229).  The fix is to ship the
    small artefact *inside* the delivered directory and to name, not hide, what is missing.
    """
    import os
    env = Path(os.environ.get("ROUTE2_FIXTURES", "") or ".") / name
    cands = [HERE / "fixtures" / name, env, *relatives]
    for c in cands:
        if c.is_file():
            return c
    return cands[0]


REAL_FIXTURE_NAME = "TB-base_h1_real_rows.csv"
CBASE_RAW_NAME = "C-base_raw.csv"
FDENSE_NAME = "field_dense.csv"


def real_artefact_checks(ck: Check) -> None:
    """The mandated end-to-end assertion in its strongest form: a file FreeFEM itself
    printed (committed in the repo), not coordinates I hold in memory.

    It also pins defect 6, which the stand-in run found: four of these eight rows are
    branch-outlet vertices, and on the 45-degree outlet plane 6-digit printing leaves them
    3.0e-6 / 4.1e-6 from their own plane -- outside the fixed 1e-6 band the boundary
    classification used, so the post-process called real Dirichlet nodes "wall".
    """
    import generate_t_case as gc
    geom = TGeometry(case_by_id("TB-base"))
    REAL = _find_tracked(REAL_FIXTURE_NAME,
                         HERE.parents[1] / "cases" / "tbif_2d" / "fixtures" / REAL_FIXTURE_NAME)
    if not REAL.is_file():
        ck.add("fixture.solver_printed_rows_are_in_the_repository", False, str(REAL),
               "shipped inside model/scripts/route2/fixtures/ so the delivery line brings it")
        for nm in ("fixture.every_token_is_a_6_significant_digit_fixed_point",
                   "fixture.printed_outlet_vertices_are_classified_as_outlets",
                   "fixture.the_old_fixed_band_would_have_lost_them"):
            ck.skip(nm, "no fixture file", "cannot judge without the solver-printed rows")
        return
    ck.add("fixture.solver_printed_rows_are_in_the_repository", True, str(REAL),
           "shipped inside model/scripts/route2/fixtures/ so the delivery line brings it")
    hdr, body = art.read_csv_rows(REAL)
    ck.add("fixture.every_token_is_a_6_significant_digit_fixed_point",
           all("%.6g" % float(t) == t for r in body for t in r if t.strip()),
           {"rows": len(body), "sample": body[0][:3]},
           "the file really is printed text, so memory floats cannot be involved")
    dense, stats = gc.build_field_dense(REAL, geom)
    kinds = {}
    for r in dense:
        kinds[r["boundary_type"]] = kinds.get(r["boundary_type"], 0) + 1
    ck.add("fixture.printed_outlet_vertices_are_classified_as_outlets",
           stats["outside"] == 0 and stats["polygon_without_frame"] == 0
           and stats["n_kept"] == len(body) == 8 and kinds == {"inlet": 4, "outlet_down": 4}
           and stats["bc_tag_disagree"] == 0,
           {"kinds": kinds, "absorbed": stats["absorbed_print"],
            "max_gap": "%.3e" % stats["max_absorbed_gap_star"],
            "bound": "%.1e" % stats["max_absorbed_gap_bound_star"]},
           "8 kept, 4 inlet + 4 outlet_down, zero disagreement with the solver's own tags")
    lost = 0
    for r in dense:
        x, y = r["x_star"], r["y_star"]
        _v, key, _g = geom.membership(x, y)
        fr = geom.frames[key]
        xi, _eta = fr.local(x, y)
        plane = abs(xi) if fr.key == STEM else abs(xi - fr.length)
        if r["boundary_type"].startswith("outlet") and plane > tg.ABSORB_TOL:
            lost += 1
    ck.add("fixture.the_old_fixed_band_would_have_lost_them", lost == 4,
           {"outlets beyond 1e-6": lost},
           "4 of 4 real outlet nodes: the positive control for the same-ruler fix")

    # ---- the digit probe, against the file that started the argument -----------------
    # My first version of this read 58.5%; 统括官's toPrecision(6) read 100.0%.  Both were
    # computed on the same file, and his is right: `round(v/q)*q` is not the same operation
    # as printing 6 digits and parsing it back, and the difference is ~1e-15 of binary
    # noise that a *fixed-point share* turns into a 41% error.  Pinned in both directions.
    cbase = _find_tracked(CBASE_RAW_NAME, HERE.parents[1] / "cases" / "contraction_2d"
                          / "cfd" / "C-base" / CBASE_RAW_NAME)
    if not cbase.is_file():
        # the paper's own 2113-row artefact cannot be vendored into route2; say which
        # checkout line brings it, and never fold "not attempted" into "passed"
        for nm in ("probe.the_paper_artefact_is_readable",
                   "probe.reads_100_percent_on_a_6_digit_file",
                   "probe.calls_the_pandas_column_not_6_digit"):
            ck.skip(nm, str(cbase),
                    "git checkout <sha> -- model/cases/contraction_2d to run these two")
        return
    ck.add("probe.the_paper_artefact_is_readable", True, str(cbase),
           "the 6-digit reference the argument was about")
    if cbase.is_file():
        hdr_c, rows_c = art.read_csv_rows(cbase)
        probe = gc.coordinate_digits_used(rows_c, hdr_c)
        cols6 = ["x_star", "y_star", "u_star", "v_star", "p_star"]
        all_one = all(probe[c]["share_unchanged_by_6sig_roundtrip"] == 1.0 for c in cols6)
        zero_dev = max(probe[c]["max_abs_deviation_from_6sig"] for c in cols6) == 0.0
        ck.add("probe.reads_100_percent_on_a_6_digit_file", all_one and zero_dev,
               {c: probe[c]["share_unchanged_by_6sig_roundtrip"] for c in cols6},
               "his 2113/2113 = 100.0%, max_dev 0.0 (my earlier 58.5% was the probe's own bug)")
        # negative control, and it has to be a *different value*, not a different string:
        # the comparison is on doubles, so printing 15.9813 as 17 digits still parses back
        # to the same 6-digit-exact double.  pandas' own column is the honest opposite.
        fdense = _find_tracked(FDENSE_NAME, HERE.parents[1] / "cases" / "contraction_2d"
                               / "data" / "C-base" / FDENSE_NAME)
        if not fdense.is_file():
            ck.skip("probe.calls_the_pandas_column_not_6_digit", str(fdense),
                    "needs model/cases/contraction_2d/data as well")
            return
        hdr_f, rows_f = art.read_csv_rows(fdense)
        probe_f = gc.coordinate_digits_used(rows_f, hdr_f)
        ck.add("probe.calls_the_pandas_column_not_6_digit",
               probe_f["wall_distance_star"]["share_unchanged_by_6sig_roundtrip"] < 1.0
               and probe_f["x_star"]["share_unchanged_by_6sig_roundtrip"] == 1.0,
               {"wall_distance_star(pandas)":
                    probe_f["wall_distance_star"]["share_unchanged_by_6sig_roundtrip"],
                "x_star(copied from the solver)":
                    probe_f["x_star"]["share_unchanged_by_6sig_roundtrip"]},
               "same probe, same directory: solver text 1.0, computed column < 1.0")


def flux_section_checks(ck: Check, geom: TGeometry) -> None:
    """The `q'(xi) = 0` requirement, and which statistic sees which way of getting it wrong.

    The instance read 0.1454 (TB-base) / 0.2164 (TB-asym) at the 1e-3 along-branch flux
    gate, and the number did not shrink over four refinements -- that signature is a
    definition, not a mesh.  Here the definition is pinned to machine precision, and the
    two ways it can be broken (fixed axis component; junction stations) are made to fail
    on purpose.
    """
    import generate_t_case as gc

    c, s = geom.cos_t, geom.sin_t
    hu, hd = geom.frames[UP].half_width, geom.frames[DOWN].half_width
    means = {UP: 0.5, DOWN: 0.5} if geom.case.is_geometrically_symmetric else \
        {UP: 0.55, DOWN: 0.45}
    q_in = means[UP] * 2 * hu + means[DOWN] * 2 * hd
    means[STEM] = q_in / (2 * geom.h_stem)
    sol = {k: rs.Poiseuille(mean_velocity=means[k], half_width=geom.frames[k].half_width)
           for k in (STEM, UP, DOWN)}

    def section_q(key: int, xi: float, direction,
                  eta_step: float = gc.SECTION_ETA_STEP) -> float:
        """Q = integral of (u . direction) d eta over the section grid the .edp emits."""
        fr = geom.frames[key]
        etas = gc.section_eta_nodes(fr.half_width, eta_step)
        vals = [sol[key].axial_velocity(e) * (direction[0] * fr.d[0]
                                             + direction[1] * fr.d[1]) for e in etas]
        return gc._trapz(vals, etas)

    ck.add("flux.section_grid_ends_on_both_walls",
           all(abs(gc.section_eta_nodes(geom.frames[k].half_width)[-1]
                   - geom.frames[k].half_width) < 1e-15
               and gc.section_eta_nodes(geom.frames[k].half_width)[0]
               == -geom.frames[k].half_width
               for k in (STEM, UP, DOWN)),
           {geom.frames[k].name: [len(gc.section_eta_nodes(geom.frames[k].half_width)),
                                  round(gc.section_eta_nodes(geom.frames[k].half_width)[1]
                                        - geom.frames[k].half_width, 6)]
            for k in (STEM, UP, DOWN)},
           "last eta == +HW exactly (the 0.85W branch used to stop 0.01 short)")

    # (1) orthonormal frame => eta really is arc length, and u.d really is the normal flux
    ck.add("flux.frame_is_orthonormal_so_deta_is_arc_length",
           abs(math.hypot(*geom.frames[UP].d) - 1.0) < 1e-15
           and abs(math.hypot(*geom.frames[UP].m) - 1.0) < 1e-15
           and abs(geom.frames[UP].d[0] * geom.frames[UP].m[0]
                   + geom.frames[UP].d[1] * geom.frames[UP].m[1]) < 1e-15,
           [geom.frames[UP].d, geom.frames[UP].m], "|d|=|m|=1, d.m=0 for every frame")

    # (2) the geometric material test must agree with the closed-form junction geometry
    agree, spans, excluded = True, {}, {}
    for key in (STEM, UP, DOWN):
        fr = geom.frames[key]
        lo, hi = geom.material_span_closed_form(key)
        mat = {xi: geom.section_is_material(key, xi) for xi in geom.station_xis(key)}
        excluded[fr.name] = sorted(round(xi, 6) for xi, m in mat.items() if not m)
        spans[fr.name] = [round(lo, 6), round(hi, 6),
                          min([xi for xi, m in mat.items() if m], default=None),
                          max([xi for xi, m in mat.items() if m], default=None)]
        for xi, m in mat.items():
            if m != (lo - 1e-12 <= xi <= hi + 1e-12):
                agree = False
    ck.add("flux.material_station_test_agrees_with_the_closed_form", agree,
           {"span_and_witness": spans, "excluded_stations": excluded},
           "geometric wall-distance test == analytic crotch / wall-end formula")
    ck.add("flux.junction_stations_are_the_ones_excluded",
           4.0 in excluded["stem"] and 0.25 in excluded["branch_up"]
           and 0.25 in excluded["branch_down"]
           and all(xi not in excluded["branch_up"] for xi in (0.5, 1.0, 4.0)),
           excluded, "stem xi=4.0 (past the wall end) and branch xi=0.25 (before crotch)")

    # (3) POSITIVE: on an exactly-conserving field, q is constant to machine precision
    stats_all: Dict[str, dict] = {}
    for key in (STEM, UP, DOWN):
        fr = geom.frames[key]
        xis = geom.station_xis(key)
        q = {xi: section_q(key, xi, fr.d) for xi in xis}
        flags = {xi: geom.section_is_material(key, xi) for xi in xis}
        rel, worst = gc._conservation(q, flags)
        rel_all, worst_all = gc._conservation(q, {xi: True for xi in xis})
        stats_all[fr.name] = {"material": rel, "all": rel_all, "want": q[xis[-1]]}
    ck.add("flux.q_is_constant_to_machine_precision_along_material_sections",
           all(v["material"] < 1.0e-12 for v in stats_all.values()),
           {k: "%.2e" % v["material"] for k, v in stats_all.items()},
           "q'(xi)=0 within 1e-12 relative, per branch")

    # (4) NEGATIVE 1: a fixed axis component instead of u.n.  This is what cos(theta) does
    # to the branch fluxes -- it survives along-branch conservation (a constant factor) and
    # is caught by the node closure instead, so the two statistics are pinned separately.
    wrong_dir = (1.0, 0.0)
    q_wrong = {fr: section_q(k, geom.station_xis(k)[-1], wrong_dir)
               for k, fr in ((STEM, "stem"), (UP, "branch_up"), (DOWN, "branch_down"))}
    closure_wrong = abs(q_wrong["stem"] - q_wrong["branch_up"] - q_wrong["branch_down"]) \
        / abs(q_wrong["stem"])
    along_wrong = 0.0
    for key in (UP, DOWN):
        xis = geom.station_xis(key)
        along_wrong = max(along_wrong,
                          gc._conservation({xi: section_q(key, xi, wrong_dir) for xi in xis},
                                           {xi: True for xi in xis})[0])
    right = [section_q(k, geom.station_xis(k)[-1], geom.frames[k].d)
             for k in (STEM, UP, DOWN)]
    closure_right = abs(right[0] - right[1] - right[2]) / abs(right[0])
    ck.add("flux.fixed_axis_component_is_caught_by_the_node_closure",
           closure_wrong > 1.0e-3 and closure_right < 1.0e-4 and along_wrong < 1.0e-12,
           {"closure with u_x": round(closure_wrong, 6), "closure with u.n": closure_right,
            "along-branch with u_x": along_wrong, "cos(theta)": round(abs(c), 6)},
           "u_x misses the branch flux by 1-cos: closure red, conservation blind (1e-3 gate)")

    # (4b) what is left in `closure with u.n` must be the trapezoid's own 1/n^2 term, not a
    # refinement-independent bias -- that is exactly what the eta-grid truncation was.
    def closure_at(step: float) -> float:
        qq = [section_q(k, geom.station_xis(k)[-1], geom.frames[k].d, step)
              for k in (STEM, UP, DOWN)]
        return abs(qq[0] - qq[1] - qq[2]) / abs(qq[0])

    c4, c16 = closure_at(0.02), closure_at(0.005)
    # TB-base gives every section the same n, and the trapezoid's relative 1/n^2 error is
    # the same factor on all three fluxes, so it cancels in the closure to 1e-16 already;
    # only where n differs (TB-asym's 0.85W branch) is there a term left to watch shrink.
    cancels = c4 < 1.0e-14
    ck.add("flux.closure_leftover_shrinks_like_1_over_n_squared",
           cancels or (c16 > 0.0 and 12.0 < c4 / c16 < 20.0 and c4 < 1.0e-4),
           {"regime": "cancels (equal n on every section)" if cancels else "1/n^2 visible",
            "with deta=0.02": "%.3e" % c4, "with deta=0.005": "%.3e" % c16,
            "ratio": (round(c4 / c16, 2) if c16 > 0 else None)},
           "either already machine-zero, or 4x finer eta must cut it ~16x: a residual that "
           "refuses to shrink is a definition, not quadrature noise")

    # (5) NEGATIVE 2: a junction station that is NOT a material section, perturbed by the
    # share the instance saw.  The gate must ignore it; the disclosure column must show it.
    stem_xis = geom.station_xis(STEM)
    q_stem = {xi: section_q(STEM, xi, geom.frames[STEM].d) for xi in stem_xis}
    q_stem[max(q_stem)] *= 0.85
    flags_mat = {xi: geom.section_is_material(STEM, xi) for xi in stem_xis}
    rel_mat, _ = gc._conservation(q_stem, flags_mat)
    rel_all, worst_all = gc._conservation(q_stem, {xi: True for xi in stem_xis})
    # Note what the *old* definition does with this fixture: the last stem station is both
    # the reference AND a non-material line, so every real section is measured against the
    # junction -- which is how the instance got 0.1454 / 0.2094 out of a conserving field.
    ck.add("flux.junction_station_cannot_move_the_gated_number",
           rel_mat < 1.0e-12 and abs(rel_all - (1.0 / 0.85 - 1.0)) < 1.0e-12
           and worst_all == min(q_stem) and max(q_stem) not in
           [xi for xi in stem_xis if geom.section_is_material(STEM, xi)],
           {"gated (material only)": rel_mat, "disclosed (all stations)": rel_all,
            "worst when all are used, at xi": worst_all,
            "expected 1/0.85-1": round(1.0 / 0.85 - 1.0, 12)},
           "a 15% defect on the non-material last station: gate 0, disclosure 0.1765 "
           "and the reference station itself was the excluded one")

    # (6) too few material stations must be an error, never a silent 0.0
    try:
        gc._conservation({0.0: 1.0, 0.25: 1.0}, {0.0: True, 0.25: False})
        guard = "no exception"
    except ValueError as exc:
        guard = str(exc)[:46]
    ck.add("flux.too_few_material_sections_raises", guard != "no exception", guard,
           "anti-vacuous: 'nothing to compare' may not read as 'perfectly conserved'")


class _FakeArray:
    """The smallest thing with .detach().cpu().numpy().ravel().tolist() -- so defect 8's
    tensor side can be exercised on a machine without torch."""

    def __init__(self, rows: List[List[float]]) -> None:
        self.rows = rows

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return _FakeFlat([v for row in self.rows for v in row])


class _FakeFlat(list):
    def ravel(self):
        return self

    def tolist(self):
        return [float(v) for v in self]


def defect8_checks(ck: Check) -> None:
    """Why 257 green assertions coexisted with a plan-b comparison that never ran."""
    import k0_truth_gate as kg

    rows = [[1.0, 2.0], [3.0, 4.0]]
    pert = [[1.0, 2.0], [3.0, 5.0]]
    want = 1.0 / math.sqrt(39.0)                     # derived: |4-5| / sqrt(sum ref^2)
    ck.add("defect8.list_rows_reach_a_number",
           kg._rel_err(rows, rows) == 0.0 and abs(kg._rel_err(rows, pert) - want) < 1.0e-12,
           {"list vs list": kg._rel_err(rows, rows), "list vs perturbed": kg._rel_err(rows, pert),
            "hand-derived": want}, "0.0 and 1/sqrt(39)=0.1601281537...")
    ck.add("defect8.tensor_and_list_agree_to_bit",
           abs(kg._rel_err(_FakeArray(rows), pert) - kg._rel_err(rows, pert)) < 1.0e-15
           and kg._rel_err(_FakeArray(rows), rows) == 0.0,
           kg._rel_err(_FakeArray(rows), pert), "same number through either door")
    try:
        kg._rel_err([[1.0, 2.0, 3.0]], [[1.0, 2.0]])
        mismatch = "no exception"
    except ValueError as exc:
        mismatch = str(exc)[:56]
    ck.add("defect8.shape_mismatch_is_refused", mismatch != "no exception", mismatch,
           "anti-vacuous: zip() would have compared a truncated pair and looked fine")
    ck.add("defect8.why_the_device_missed_it",
           not hasattr(rows, "detach") and not hasattr(pert, "detach"),
           {"the analytic path hands over": type(rows).__name__,
            "the old first line asked for": "detach()"},
           "list has no .detach, and no assertion ever called _rel_err with a list before")



def k0_referee_checks(ck: Check) -> None:
    """A second-derivative referee that does not depend on the file's digits, and the
    rule that says when the FD-based reference is not allowed to call anything a FAIL.

    Field: G(x,y) = sum_k c_k exp(-r_k^2 / 2 sigma^2) over the three branch frames -- the
    same Gaussian proximity the plan-b features are built from.  d2G/dx2 is derived by
    hand (below) and used as the referee, so nothing here reads a printed file.
    """
    import k0_truth_gate as kg
    geom = TGeometry(case_by_id("TB-base"))
    frames = {k: geom.frames[k] for k in (STEM, UP, DOWN)}
    coef = {STEM: 0.7, UP: -0.3, DOWN: 0.5}
    sig = 0.15

    def comp(k: int, x: float, y: float) -> float:
        xi, eta = frames[k].local(x, y)
        return coef[k] * math.exp(-(xi * xi + eta * eta) / (2.0 * sig * sig))

    def comp_d2_xx(k: int, x: float, y: float) -> float:
        # psi = exp(-q), q = (xi^2+eta^2)/(2 sig^2); d2psi/dx2 = psi (qx^2 - qxx) with
        # qx = (xi dx + eta mx)/sig^2 and qxx = (dx^2+mx^2)/sig^2 = 1/sig^2 (unit vectors)
        fr = frames[k]
        xi, eta = fr.local(x, y)
        qx = (xi * fr.d[0] + eta * fr.m[0]) / (sig * sig)
        qxx = (fr.d[0] ** 2 + fr.m[0] ** 2) / (sig * sig)
        return coef[k] * math.exp(-(xi * xi + eta * eta) / (2.0 * sig * sig)) * (qx * qx - qxx)

    nodes = []
    for k in frames:
        fr = frames[k]
        for i in range(41):
            for j in (0, 8, 17, 25, 34):
                nodes.append((k,) + fr.global_xy(fr.length * i / 40.0,
                                                 -0.9 * fr.half_width
                                                 + 1.8 * fr.half_width * j / 34.0))

    def rel_err_at(step: float, round_to: int | None) -> float:
        num = den = 0.0
        for (k, x, y) in nodes:
            val = (lambda v: round(v, round_to) if round_to is not None else v)
            fd = (val(comp(k, x + step, y)) - 2.0 * val(comp(k, x, y))
                  + val(comp(k, x - step, y))) / (step * step)
            ex = comp_d2_xx(k, x, y)
            num += (fd - ex) ** 2
            den += ex * ex
        return math.sqrt(num / den)

    steps = [4.0e-3, 1.0e-3, 2.5e-4]
    exact = kg.reference_resolution_scan([rel_err_at(s, None) for s in steps], steps)
    six = kg.reference_resolution_scan([rel_err_at(s, 6) for s in steps], steps)
    six_sq = kg.reference_resolution_scan([rel_err_at(s, 6) ** 2 for s in steps], steps,
                                          kind="squared_error")
    ck.add("referee.hand_derived_reference_is_reached_by_exact_data",
           exact["status"] == "TRUNCATION" and 0.18 < exact["factor_per_halving"] < 0.32
           and exact["endpoints"][1][1] < 1.0e-5,
           {"factors": exact["pairwise_factors"], "err at finest": exact["endpoints"][1][1]},
           "error falls 4x per halving (h^2 truncation) -> the referee resolves the quantity")
    ck.add("referee.six_digit_data_flips_the_direction",
           six["status"] == "ROUND_OFF" and 3.0 < six["factor_per_halving"] < 5.0
           and six_sq["status"] == "ROUND_OFF" and 12.0 < six_sq["factor_per_halving"] < 22.0,
           {"per_halving_error": six["factor_per_halving"],
            "per_halving_squared": six_sq["factor_per_halving"],
            "theory": "4x per halving for the error, 16x for a squared residual"},
           "shrinking the step makes it WORSE -> this reference cannot judge the stencil")
    bound = [kg.second_derivative_roundoff_bound(tg.TGeometry.half_ulp(1.5), s) for s in steps]
    meas = [rel_err_at(s, 6) for s in steps]
    rms_scale = math.sqrt(sum(comp_d2_xx(k, x, y) ** 2 for (k, x, y) in nodes) / len(nodes))
    ratio = [max(b / rms_scale, 1e-30) / max(m, 1e-30) for b, m in zip(bound, meas)]
    ck.add("referee.roundoff_bound_holds_without_being_loose",
           all(b >= m for b, m in zip(bound, meas)) and max(ratio) < 1.0e3,
           {"bound/measured": [round(r, 1) for r in ratio],
            "bound_abs": ["%.1e" % b for b in bound], "measured": ["%.1e" % m for m in meas]},
           "4*ulp/h^2 is an upper bound on the move and within 3 decades of the measurement")
    flat = kg.reference_resolution_scan([0.05, 0.05, 0.05], steps)
    ck.add("referee.a_flat_series_is_called_what_it_is", flat["status"] == "INDETERMINATE",
           flat["pairwise_factors"], "no slope -> no verdict, and no silent pass either")
    ck.add("referee.status_rule_cannot_be_used_as_an_acquittal",
           kg.second_order_status(exact, 0.4, rs.K0_CHAIN_SECOND_REL_MAX) == "RESOLVED_FAIL"
           and kg.second_order_status(six, 0.4, rs.K0_CHAIN_SECOND_REL_MAX) == "INDETERMINATE"
           and kg.second_order_status(exact, 1.0e-6, rs.K0_CHAIN_SECOND_REL_MAX)
           == "RESOLVED_PASS" and kg.second_order_status(None, 0.0, 0.05) == "INDETERMINATE",
           {"resolving_ref+big": "RESOLVED_FAIL", "resolving_ref+small": "RESOLVED_PASS",
            "noise_ref": "INDETERMINATE", "no scan": "INDETERMINATE"},
           "INDETERMINATE appears only when the reference is blind, never to hide a fail")


def _lens_area_note(geom: TGeometry) -> float:
    if "lens_area" not in _AGREEMENT:
        _AGREEMENT.update(_polygon_matches_frames(geom))
    return _AGREEMENT["lens_area"]


_MODULE_DUNDER = {"__name__", "__file__", "__doc__", "__all__", "__spec__", "__loader__",
                  "__package__", "__builtins__", "__debug__"}


def _unbound_global_refs(src: str, name: str):
    """Names a module reads as globals but never binds anywhere -- a NameError waiting for
    the one code path that has never been run.  Deliberately conservative: a name bound in
    any scope counts as bound, so only genuinely-missing imports show up."""
    import symtable
    top = symtable.symtable(src.replace("\r\n", "\n"), name, "exec")
    bound: set = set()
    refs: set = set()

    def visit(tbl):
        for sym in tbl.get_symbols():
            if sym.is_assigned() or sym.is_imported() or sym.is_namespace():
                bound.add(sym.get_name())
            if sym.is_global():
                refs.add(sym.get_name())
        for child in tbl.get_children():
            visit(child)

    visit(top)
    import builtins
    return sorted(refs - bound - set(dir(builtins)) - _MODULE_DUNDER)


def module_hygiene_checks(ck: Check) -> None:
    """Defect 5's second face: `--selfcheck-raw` also used `json` without importing it.

    A branch that has never executed carries its NameError silently, and no in-process
    assertion will ever see it.  The scan is compiled-only (stdlib symtable), so it runs
    on the laptop, on his machine and on the instance identically.
    """
    offenders = {}
    for path in sorted(HERE.glob("*.py")):
        found = _unbound_global_refs(path.read_text(encoding="utf-8"), path.name)
        if found:
            offenders[path.name] = found
    ck.add("hygiene.no_reference_to_an_unbound_module_name", offenders == {}, offenders,
           "every global name is bound somewhere")
    caught = _unbound_global_refs("import os\n\n\ndef f():\n    return json.dumps(1)\n",
                                  "sample.py")
    clean = _unbound_global_refs("import json\n\n\ndef f():\n    return json.dumps(1)\n",
                                 "sample_ok.py")
    ck.add("hygiene.the_scan_is_discriminative", caught == ["json"] and clean == [],
           {"unimported": caught, "imported": clean},
           "positive control: the first is reported, the second is not")


def main() -> int:
    ap = argparse.ArgumentParser(description="route2 K0/S1 stdlib self-test")
    ap.add_argument("--json", default=str(DEFAULT_OUT), help="output json (scratch dir)")
    ap.add_argument("--cases", default="TB-base,TB-asym",
                    help="comma list; both run by default because TB-asym is now the "
                         "adversary-table geometry (ruling R2-1)")
    args = ap.parse_args()

    ck = Check()
    tmp_root = Path(args.json).parent / "selftest_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, dict] = {}
    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    for case_id in case_ids:
        geom = TGeometry(case_by_id(case_id))
        ck.prefix = "" if len(case_ids) == 1 else f"{case_id}."
        print(f"# route2 stdlib self-test  case={geom.case.case_id} theta={geom.case.theta_deg}"
              f" W_up={geom.case.w_branch_up} W_dn={geom.case.w_branch_down}"
              f" area={geom.area():.4f} perimeter={geom.perimeter():.4f}"
              f" crotch=({geom.crotch[0]:.5f},{geom.crotch[1]:.5f})")
        geometry_checks(ck, geom)
        jacobian_checks(ck, geom)
        manufactured_field_checks(ck, geom)
        fd_checks(ck, geom)
        tmp = tmp_root / case_id
        tmp.mkdir(parents=True, exist_ok=True)
        s1_pipeline_checks(ck, geom, tmp)
        flux_section_checks(ck, geom)
        k0_truth_side_checks(ck, geom, tmp)
        membership_checks(ck, geom, tmp)
        runtime_path_checks(ck, geom, tmp)
        meshing_checks(ck, geom)
        lens = _lens_area_note(geom)
        summaries[case_id] = {"metadata": geom.case.to_metadata(), "area": geom.area(),
                              "perimeter": geom.perimeter(), "crotch": geom.crotch,
                              "lens_area": lens,
                              "lens_fraction": lens / geom.area()}
        print(f"# {case_id}: overlap-lens area {lens:.4f} "
              f"({100.0 * lens / geom.area():.1f}% of domain) -- frame assignment is a "
              f"convention there, made single-valued by blend weights for plan (b)")
    ck.prefix = ""
    mesh_gate_checks(ck)
    k0_verdict_checks(ck)
    module_hygiene_checks(ck)
    defect8_checks(ck)
    k0_referee_checks(ck)
    real_artefact_checks(ck)
    impedance_checks(ck, summaries, tmp_root)

    out = Path(args.json)
    repo_root = HERE.parents[2]  # .../pinn-platform-v4
    if repo_root in out.parents or str(out).startswith(str(repo_root)):
        raise SystemExit(f"refusing to write self-test output inside the repo: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checks": ck.rows, "failed": ck.failed,
                               "cases": summaries},
                              ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
    print(f"json={out}")
    print(f"total={len(ck.rows)} failed={len(ck.failed)} "
          f"skipped={len(ck.skipped)}")
    if ck.failed:
        print("FAILED: " + ", ".join(ck.failed))
        return 1
    print("ALL GREEN")
    return 0


def geom_case_kwargs(case_id: str) -> dict:
    from t_geometry import case_by_id
    return {k: v for k, v in vars(case_by_id(case_id)).items()}


if __name__ == "__main__":
    sys.exit(main())
