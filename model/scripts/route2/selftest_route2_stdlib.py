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
import sys
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

    @property
    def failed(self) -> list[str]:
        return [r["check"] for r in self.rows if not r["pass"]]


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
    for (a, b), lab in zip(geom.polygon.verts, geom.polygon.edge_ends):
        pass
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
           key is not None and verdict in ("frame", "absorbed"),
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
           {"kept": stats2["n_kept"], "absorbed": stats2["absorbed"],
            "outside": stats2["outside"], "pwof": stats2["polygon_without_frame"]},
           "no contour-without-frame vertices")
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


def k0_truth_side_checks(ck: Check, geom: TGeometry, tmp: Path) -> None:
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


def _lens_area_note(geom: TGeometry) -> float:
    if "lens_area" not in _AGREEMENT:
        _AGREEMENT.update(_polygon_matches_frames(geom))
    return _AGREEMENT["lens_area"]


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
        k0_truth_side_checks(ck, geom, tmp)
        membership_checks(ck, geom, tmp)
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
    print(f"total={len(ck.rows)} failed={len(ck.failed)}")
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
