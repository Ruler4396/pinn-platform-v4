#!/usr/bin/env python3
"""K0: does the route-2 loss function actually contain Stokes?  (truth self-scoring gate)

The failure definition being tested (frozen 2026-09-25 in
paper-route2/K0-S1设计与预注册-20260925.md): substituting the CFD *truth* into the
current v4 momentum term scores 2.44e6 / 9.62e6 times worse than a converged model,
because the two sides of the momentum equation are divided by different scales
(``train_velocity_pressure_independent.py:369-383``) and because autodiff differentiates
only columns 0-1 of a 14-column input while the other 12 geometry columns are treated
as constants (``:337,350-365``).  A PINN-vs-impedance comparison built on that loss
measures nothing, so nothing downstream runs until the same comparison passes at <= 10x.

Two fixes, each of which must clear the gate on its own:

  plan a (PRIMARY)   the network eats only (x*, y*); geometry enters through a
                     differentiable hard envelope and through evaluation masks, so the
                     chain rule is complete by construction.
  plan b (ABLATION)  the geometry features stay in the network input, but d/dx* is made
                     a *total* derivative.  Implemented twice and cross-checked:
                       b-graph   features rebuilt inside the graph from (x*, y*), so
                                 autograd sees the whole chain (used for the residual,
                                 since it also gives the second derivatives);
                       b-contract the hand-written analytic Jacobian from t_geometry
                                 contracted with the input-layer gradient (first
                                 derivatives only, where the contract is complete).
                     Both must agree with a central difference of the identical composed
                     map; the v4-style "features detached" variant is reported alongside
                     so the size of the dropped-chain term is a number, not an assertion.

Gate arithmetic (thresholds, verdict assembly, the v4 as-found reproduction) lives in
residual_scorers.py, is stdlib-only, and is exercised by selftest_route2_stdlib.py -- so
the arithmetic is falsifiable on the laptop even though the torch half is not.

torch is imported only inside the real code path: ``--dry-run`` works with no numpy
and no torch installed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import artifacts as art                          # noqa: E402
import residual_scorers as rs                    # noqa: E402
import t_geometry as tg                          # noqa: E402

DEFAULT_CASE = "TB-base"
SIGMA_PRIMARY = 0.15
SIGMA_ABLATION = 0.30
X_NORM = 4.0                     # x* / X_NORM -> O(1) network input; derivatives rescaled
Y_NORM = 1.0
FD_STEP_NORM = 1.0e-3            # central-difference step in normalized units
TRAIN_BUDGET = {"steps": (600, 600), "lrs": (1.0e-3, 1.0e-4), "hidden": (64, 64, 64),
                "activation": "Tanh", "collocation": 1024, "seed": 20260925,
                "dtype": "float64", "envelope_sharpness": 12.0}
BRANCH_SAMPLE_SETS = ("stem", "branch_up", "branch_down")


# ============================================================ truth side (stdlib)
def load_lattice(case_root: Path, case_id: str, level: str, branch: str,
                 tag: str) -> dict:
    """One structured sample grid as an indexed point set with its spacings.

    Branch grids are stored in branch-local (xi, eta); because all three branches are
    straight, (xi, eta) is an orthonormal rotation of (x*, y*) and the Laplacian keeps
    its 5-point form -- no curvilinear terms.  The junction grid is stored in global
    (x*, y*) with the crotch disk removed, so the lattice has holes: stencils are only
    evaluated where all four neighbours exist (see lattice_residual).
    """
    d = case_root / "cfd" / f"{case_id}_{level}"
    path = d / f"{case_id}_{level}_samples_{branch}_{tag}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing S1 sample grid: {path}")
    header, rows = art.read_csv_rows(path)
    idx = {n: i for i, n in enumerate(header)}
    pts = [(float(r[idx["x_star"]]), float(r[idx["y_star"]]), float(r[idx["u_star"]]),
            float(r[idx["v_star"]]), float(r[idx["p_star"]]),
            float(r[idx["xi"]]) if "xi" in idx else float(r[idx["x_star"]]),
            float(r[idx["eta"]]) if "eta" in idx else float(r[idx["y_star"]]))
           for r in rows]
    if not pts:
        raise ValueError(f"{path.name}: empty sample grid")
    a0 = sorted({round(p[5], 9) for p in pts})
    a1 = sorted({round(p[6], 9) for p in pts})
    pos0 = {v: i for i, v in enumerate(a0)}
    pos1 = {v: i for i, v in enumerate(a1)}
    table: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    for p in pts:
        table[(pos0[round(p[5], 9)], pos1[round(p[6], 9)])] = (p[2], p[3], p[4])
    if len(a0) < 3 or len(a1) < 3:
        raise ValueError(f"{path.name}: lattice {len(a0)}x{len(a1)} too small for FD")
    h0, uni0 = _uniform_step(a0, path)
    h1, uni1 = _uniform_step(a1, path)
    return {"branch": branch, "tag": tag, "path": path, "n0": len(a0), "n1": len(a1),
            "h0": h0, "h1": h1, "table": table, "n_points": len(pts),
            "uniformity": {"xi": uni0, "eta": uni1}}


def _uniform_step(values: Sequence[float], path: Path,
                  tol: Optional[float] = None) -> Tuple[float, dict]:
    """Step of a lattice axis, with uniformity judged by the file's own precision.

    `tol=None` takes the derived printing bound (t_geometry.spacing_tolerance: 4 x
    half-ulp at the largest magnitude present).  Pass `tol` explicitly only to prove the
    guard still bites -- the self-test uses it to reject a genuine 1e-3 displacement, so
    "we made room for rounding" can never quietly become "we stopped checking".
    The measured spread and the bound travel with the lattice into `k0_verdict.json`.
    """
    diffs = [b - a for a, b in zip(values, values[1:])]
    h = (values[-1] - values[0]) / (len(values) - 1)
    spread = max(diffs) - min(diffs)
    bound = tg.spacing_tolerance(values) if tol is None else tol
    if spread > bound:
        raise ValueError(f"{path.name}: axis is not evenly spaced -- spread {spread:.3e} "
                         f"exceeds the printing bound {bound:.3e} "
                         f"(4 x half-ulp at |max|={max(abs(v) for v in values):.6g}); "
                         f"min {min(diffs):.6g} max {max(diffs):.6g}. Central differences "
                         f"would be invalid, so this is refused, not absorbed")
    return h, {"step_star": h, "spread_star": spread, "printing_bound_star": bound,
               "share_of_bound": spread / bound if bound > 0.0 else 0.0,
               "n_nodes": len(values)}


def lattice_residual(lat: dict, geom: tg.TGeometry) -> dict:
    """Corrected Stokes residual of a lattice, by central differences (holes allowed).

    Sample CSVs store the *global* (u*, v*) components, as FreeFEM exports them.  On a
    branch lattice the differencing axes are (xi, eta), so the vector components are
    rotated into that frame first: differencing global components along rotated axes is
    not the momentum equation (it is the bug this gate exists to catch, in miniature).
    """
    h0, h1 = lat["h0"], lat["h1"]
    n0, n1 = lat["n0"], lat["n1"]
    rot = None
    if lat["branch"] != "junction":
        fr = geom.frames[{"stem": tg.STEM, "branch_up": tg.UP,
                          "branch_down": tg.DOWN}[lat["branch"]]]
        rot = (fr.d, fr.m)
    table = {(i, j): _rotate(v, rot) for (i, j), v in lat["table"].items()}
    mom = div = lap_sq = grad_sq = 0.0
    n = 0
    for i in range(1, n0 - 1):
        for j in range(1, n1 - 1):
            c = table.get((i, j))
            pts5 = (table.get((i + 1, j)), table.get((i - 1, j)),
                    table.get((i, j + 1)), table.get((i, j - 1)))
            if c is None or any(q is None for q in pts5):
                continue
            up, um, vp, vm = pts5
            lu = ((up[0] - 2.0 * c[0] + um[0]) / (h0 * h0)
                  + (vp[0] - 2.0 * c[0] + vm[0]) / (h1 * h1))
            lv = ((up[1] - 2.0 * c[1] + um[1]) / (h0 * h0)
                  + (vp[1] - 2.0 * c[1] + vm[1]) / (h1 * h1))
            px = (up[2] - um[2]) / (2.0 * h0)
            py = (vp[2] - vm[2]) / (2.0 * h1)
            ux = (up[0] - um[0]) / (2.0 * h0)
            vy = (vp[1] - vm[1]) / (2.0 * h1)
            if not all(_is_num(z) for z in (lu, lv, px, py, ux, vy)):
                continue
            mom += (lu - px) ** 2 + (lv - py) ** 2
            div += (ux + vy) ** 2
            lap_sq += lu * lu + lv * lv
            grad_sq += px * px + py * py
            n += 1
    if n == 0:
        raise ValueError(f"{lat['path'].name}: no usable FD stencil")
    return {"branch": lat["branch"], "tag": lat["tag"], "n_stencils": n,
            "uniformity": lat.get("uniformity"),
            "spacing": [h0, h1], "n_points": lat["n_points"],
            "momentum_mse": mom / n, "continuity_mse": div / n,
            "lap_rms": math.sqrt(lap_sq / n), "gradp_rms": math.sqrt(grad_sq / n),
            "balance_ratio": (math.sqrt(lap_sq / grad_sq) if grad_sq > 0 else float("inf"))}


def _rotate(uv_p, rot):
    if rot is None:
        return uv_p
    (dx, dy), (mx, my) = rot
    u, v, p = uv_p
    return (u * dx + v * dy, u * mx + v * my, p)


def _is_num(z: float) -> bool:
    return isinstance(z, (int, float)) and z == z and abs(z) != float("inf")


def truth_score(case_root: Path, case_id: str, level: str, geom: tg.TGeometry) -> dict:
    """score(truth) per grid at FD step h and 2h -> K0-S1/S2/S3/S4 inputs."""
    per: Dict[str, dict] = {}
    for branch in BRANCH_SAMPLE_SETS + ("junction",):
        fine = lattice_residual(load_lattice(case_root, case_id, level, branch, "h"), geom)
        coarse = lattice_residual(load_lattice(case_root, case_id, level, branch, "h2"), geom)
        per[branch] = {"h": fine, "h2": coarse}
    # The FD step gate below compares two scores computed on two *different* lattices
    # (h vs 2h).  On a 6-significant-digit file the coordinates themselves carry up to
    # 4 x half-ulp of jitter, so that comparison would mix "the stencil is right" with
    # "the file is coarse"; publish the spread so nobody has to guess which one it saw.
    uni = {f"{b}_{ax}_{tag}": per[b][tag]["uniformity"][ax]
           for b in per for ax in ("xi", "eta") for tag in ("h", "h2")}
    worst_key = max(uni, key=lambda k: uni[k]["share_of_bound"])
    w = {b: per[b]["h"]["n_stencils"] for b in per}
    tot = float(sum(w.values()))
    mom = sum(per[b]["h"]["momentum_mse"] * w[b] for b in per) / tot
    mom_coarse = sum(per[b]["h2"]["momentum_mse"] * w[b] for b in per) / tot
    cont = sum(per[b]["h"]["continuity_mse"] * w[b] for b in per) / tot
    lap_sq = sum(per[b]["h"]["lap_rms"] ** 2 * w[b] for b in per)
    grad_sq = sum(per[b]["h"]["gradp_rms"] ** 2 * w[b] for b in per)
    return {"per_grid": per, "momentum_mse": mom, "continuity_mse": cont,
            "balance_ratio": math.sqrt(lap_sq / grad_sq) if grad_sq > 0 else float("inf"),
            "n_stencils": int(tot),
            "fd_step_same_lattice": {"note": "h vs 2h are different lattices, so a change "
                                   "here is not a pure convergence test; see axis_uniformity",
                                     "fine_momentum_mse": mom, "coarse_momentum_mse": mom_coarse},
            "axis_uniformity": uni,
            "axis_uniformity_worst": {"which": worst_key,
                                      "share_of_printing_bound": uni[worst_key]["share_of_bound"],
                                      "spread_star": uni[worst_key]["spread_star"],
                                      "printing_bound_star": uni[worst_key]["printing_bound_star"]},
            "fd_step_gate": rs.fd_step_convergence_gate([mom_coarse, mom])}


# =============================================================== model side (torch)
def forward_field(t, nets: Dict[str, object], xy_norm: "t.Tensor", plan: str,
                  geom: tg.TGeometry, sigma: float, sharpness: float,
                  detach_features: bool = False):
    """Physical (u, v, p) in star units as a function of normalized (x*, y*).

    plan 'a': nets take (x*, y*) only; velocity carries the hard envelope.
    plan 'b': nets take (x*, y*, 10 geometry features) built in-graph from (x*, y*),
              so any autograd derivative w.r.t. (x*, y*) is a *total* derivative.
              detach_features=True reproduces the v4 hole (features as constants).
    """
    x_s = xy_norm[:, 0:1] * X_NORM
    y_s = xy_norm[:, 1:2] * Y_NORM
    if plan == "a":
        u = nets["u"](xy_norm) * envelope(t, geom, x_s, y_s, sigma, sharpness)
        v = nets["v"](xy_norm) * envelope(t, geom, x_s, y_s, sigma, sharpness)
        p = nets["p"](xy_norm)
    elif plan == "b":
        feats = feature_tensor(t, geom, x_s, y_s, sigma)
        if detach_features:
            feats = feats.detach()
        inp = t.cat([xy_norm, feats], dim=1)
        u, v, p = nets["u"](inp), nets["v"](inp), nets["p"](inp)
    else:
        raise ValueError(f"unknown plan {plan!r}")
    return u, v, p


def clearance(t, geom: tg.TGeometry, x_s: "t.Tensor", y_s: "t.Tensor", sigma: float
              ) -> "t.Tensor":
    """Blended 1-(eta/h)^2 computed with torch ops (identical formula to t_geometry)."""
    frames = geom.frames
    chi = {}
    s2 = sigma * sigma
    for key, fr in frames.items():
        rx, ry = x_s - fr.origin[0], y_s - fr.origin[1]
        xi = rx * fr.d[0] + ry * fr.d[1]
        eta = rx * fr.m[0] + ry * fr.m[1]
        g_lo = t.clamp(-xi, min=0.0)
        g_hi = t.clamp(xi - fr.length, min=0.0)
        g_eta = t.clamp(t.abs(eta) - fr.half_width, min=0.0)
        chi[key] = (t.exp(-(g_lo * g_lo + g_hi * g_hi + g_eta * g_eta) / (2.0 * s2)),
                    xi, eta, fr)
    tot = sum(c[0] for c in chi.values())
    acc = t.zeros_like(x_s)
    for key, (val, xi, eta, fr) in chi.items():
        clear = t.clamp(1.0 - (eta / fr.half_width) ** 2, min=0.0) ** 2
        acc = acc + (val / tot) * clear
    return acc


def envelope(t, geom, x_s, y_s, sigma, sharpness) -> "t.Tensor":
    d = t.clamp(clearance(t, geom, x_s, y_s, sigma), min=0.0, max=1.0)
    return (1.0 - t.exp(-sharpness * d)) / (1.0 - math.exp(-sharpness))


def feature_tensor(t, geom: tg.TGeometry, x_s, y_s, sigma) -> "t.Tensor":
    """The 10 route-2 geometry features, in-graph, mirroring TGeometry.features()."""
    frames = geom.frames
    s2 = sigma * sigma
    chi = {}
    for key, fr in frames.items():
        rx, ry = x_s - fr.origin[0], y_s - fr.origin[1]
        xi = rx * fr.d[0] + ry * fr.d[1]
        eta = rx * fr.m[0] + ry * fr.m[1]
        g_lo, g_hi = t.clamp(-xi, min=0.0), t.clamp(xi - fr.length, min=0.0)
        g_eta = t.clamp(t.abs(eta) - fr.half_width, min=0.0)
        chi[key] = (t.exp(-(g_lo * g_lo + g_hi * g_hi + g_eta * g_eta) / (2.0 * s2)), xi, eta)
    tot = sum(c[0] for c in chi.values())
    w = {k: chi[k][0] / tot for k in frames}

    def blend(per: Dict[int, "t.Tensor"]) -> "t.Tensor":
        acc = None
        for key in frames:
            term = w[key] * per[key]
            acc = term if acc is None else acc + term
        return acc

    eta_n = {k: chi[k][2] / frames[k].half_width for k in frames}
    xi_n = {k: chi[k][1] / frames[k].length for k in frames}
    clear = {k: t.clamp(1.0 - eta_n[k] ** 2, min=0.0) ** 2 for k in frames}
    one = t.ones_like(x_s)
    r2 = (x_s - geom.j_point[0]) ** 2 + (y_s - geom.j_point[1]) ** 2
    jp = t.exp(-r2 / (2.0 * geom.h_stem * geom.h_stem))
    cols = {
        "eta_norm_star": blend(eta_n),
        "axial_frac_star": blend(xi_n),
        "wall_distance_frac": blend(clear),
        "half_width_ratio_star": blend({k: one * (frames[k].half_width / geom.h_stem)
                                        for k in frames}),
        "tangent_x_star": blend({k: one * frames[k].d[0] for k in frames}),
        "tangent_y_star": blend({k: one * frames[k].d[1] for k in frames}),
        "branch_stem_flag": w[tg.STEM],
        "branch_up_flag": w[tg.UP],
        "branch_down_flag": w[tg.DOWN],
        "junction_proximity_star": jp,
    }
    return t.cat([cols[name] for name in tg.GEOMETRY_FEATURES], dim=1)


def autograd_derivs(t, nets, xy_norm, plan, geom, sigma, sharpness,
                    detach_features=False) -> Dict[str, "t.Tensor"]:
    u, v, p = forward_field(t, nets, xy_norm, plan, geom, sigma, sharpness, detach_features)
    gx = 1.0 / X_NORM
    gy = 1.0 / Y_NORM

    def d1(f):
        g = t.autograd.grad(f.sum(), xy_norm, create_graph=True, retain_graph=True)[0]
        return g[:, 0:1] * gx, g[:, 1:2] * gy

    u_x, u_y = d1(u)
    v_x, v_y = d1(v)
    p_x, p_y = d1(p)
    u_xx = d1(u_x)[0] * gx
    u_yy = d1(u_y)[1] * gy
    v_xx = d1(v_x)[0] * gx
    v_yy = d1(v_y)[1] * gy
    return {"u": u, "v": v, "p": p, "u_x": u_x, "u_y": u_y, "v_x": v_x, "v_y": v_y,
            "p_x": p_x, "p_y": p_y, "u_xx": u_xx, "u_yy": u_yy, "v_xx": v_xx, "v_yy": v_yy}


def residual_from(der: Dict[str, "t.Tensor"]) -> Dict[str, float]:
    """Corrected Stokes residual: one common scale, both sides of the momentum equation."""
    mom = ((der["u_xx"] + der["u_yy"] - der["p_x"]) ** 2
           + (der["v_xx"] + der["v_yy"] - der["p_y"]) ** 2).mean()
    cont = (der["u_x"] + der["v_y"]) ** 2
    return {"momentum_mse": float(mom.detach().cpu()),
            "continuity_mse": float(cont.mean().detach().cpu())}


def fd_derivs(t, nets, xy_norm, plan, geom, sigma, sharpness, detach_features=False,
              step: float = FD_STEP_NORM) -> Dict[str, List[List[float]]]:
    """Central differences of the identical composed map: implementation-independent ref."""
    def at(delta0: float = 0.0, delta1: float = 0.0):
        x = xy_norm.clone()
        x[:, 0:1] += delta0
        x[:, 1:2] += delta1
        with t.no_grad():
            return [f.detach().cpu().numpy().ravel().tolist()
                    for f in forward_field(t, nets, x, plan, geom, sigma, sharpness,
                                           detach_features)]

    out: Dict[str, List[List[float]]] = {}
    base = at()
    field_index = {"u": 0, "v": 1, "p": 2}
    first_terms = (("u_x", "u", 0, X_NORM), ("u_y", "u", 1, Y_NORM),
                   ("v_x", "v", 0, X_NORM), ("v_y", "v", 1, Y_NORM),
                   ("p_x", "p", 0, X_NORM), ("p_y", "p", 1, Y_NORM))
    for name, field, axis, phys in first_terms:
        hi = at(step, 0.0) if axis == 0 else at(0.0, step)
        lo = at(-step, 0.0) if axis == 0 else at(0.0, -step)
        den = 2.0 * step * phys          # d/dx* = d/d(x*/phys) / phys
        out[name] = [(a - b) / den for a, b in zip(hi[field_index[field]],
                                                   lo[field_index[field]])]
    second_terms = (("u_xx", "u", 0, X_NORM), ("u_yy", "u", 1, Y_NORM),
                    ("v_xx", "v", 0, X_NORM), ("v_yy", "v", 1, Y_NORM))
    for name, field, axis, phys in second_terms:
        hi = at(step, 0.0) if axis == 0 else at(0.0, step)
        lo = at(-step, 0.0) if axis == 0 else at(0.0, -step)
        den = (step * phys) ** 2
        out[name] = [(a - 2.0 * b + c) / den
                     for a, b, c in zip(hi[field_index[field]], base[field_index[field]],
                                        lo[field_index[field]])]
    return out


def _rel_err(auto: "t.Tensor", ref: List[List[float]]) -> float:
    a = auto.detach().cpu().numpy().ravel().tolist()
    num = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, ref)))
    den = math.sqrt(sum(y * y for y in ref))
    return num / max(den, 1.0e-30)


def contract_first_derivatives(t, nets, xy_norm, geom, sigma) -> Dict[str, List[List[float]]]:
    """Analytic Jacobian contracted with the input-layer gradient (plan b, first order).

    du/dx* = partial(u)/partial(x*)_feats + sum_k partial(u)/partial(f_k) * df_k/dx*,
    with df_k/dx* from t_geometry (whose own correctness is checked against finite
    differences by the local stdlib self-test).  Compared against the finite-difference
    reference, so a shared bug between the two torch paths cannot pass as agreement.
    """
    x_s = xy_norm[:, 0:1] * X_NORM
    y_s = xy_norm[:, 1:2] * Y_NORM
    feats_np = feature_tensor(t, geom, x_s, y_s, sigma).detach()
    inp = t.cat([xy_norm.detach(), feats_np], dim=1).requires_grad_(True)
    jx, jy = _jacobian_columns(t, geom, x_s, y_s, sigma)
    out: Dict[str, List[List[float]]] = {}
    for field in ("u", "v", "p"):
        g = t.autograd.grad(nets[field](inp).sum(), inp)[0]
        base_x, base_y = g[:, 0:1] / X_NORM, g[:, 1:2] / Y_NORM
        add_x, add_y = t.zeros_like(base_x), t.zeros_like(base_y)
        for i, name in enumerate(tg.GEOMETRY_FEATURES):
            add_x = add_x + g[:, i + 2:i + 3] * jx[name]
            add_y = add_y + g[:, i + 2:i + 3] * jy[name]
        out[f"{field}_x"] = (base_x + add_x).detach().cpu().numpy().ravel().tolist()
        out[f"{field}_y"] = (base_y + add_y).detach().cpu().numpy().ravel().tolist()
    return out


def _jacobian_columns(t, geom: tg.TGeometry, x_s, y_s, sigma):
    xs = x_s.detach().cpu().numpy().ravel().tolist()
    ys = y_s.detach().cpu().numpy().ravel().tolist()
    jx: Dict[str, "t.Tensor"] = {}
    jy: Dict[str, "t.Tensor"] = {}
    for name in tg.GEOMETRY_FEATURES:
        ax = [geom.features_jacobian(x, y, sigma)[name][0] for x, y in zip(xs, ys)]
        ay = [geom.features_jacobian(x, y, sigma)[name][1] for x, y in zip(xs, ys)]
        jx[name] = t.tensor(ax, dtype=t.float64).reshape(-1, 1)
        jy[name] = t.tensor(ay, dtype=t.float64).reshape(-1, 1)
    return jx, jy


def build_nets(t, in_dim: int, hidden: Sequence[int], activation: str, seed: int):
    torch_nn = t.nn
    t.manual_seed(seed)

    def one(out_dim: int = 1):
        mods: List[object] = []
        d = in_dim
        for h in hidden:
            mods.append(torch_nn.Linear(d, h))
            mods.append(getattr(torch_nn, activation)())
            d = h
        mods.append(torch_nn.Linear(d, out_dim))
        return torch_nn.Sequential(*mods).double()

    return {"u": one(), "v": one(), "p": one()}


def train_nets(t, nets, xy_norm, targets_z, plan, geom, sigma, sharpness,
               steps: Sequence[int], lrs: Sequence[float]):
    """Fixed-budget supervised fit in standardized space (the K0 denominator)."""
    xy_fit = xy_norm.detach()      # fit only needs parameter gradients, not input gradients
    for stage, (n_steps, lr) in enumerate(zip(steps, lrs)):
        opt = t.optim.Adam([q for n in nets.values() for q in n.parameters()], lr=lr)
        for _ in range(n_steps):
            opt.zero_grad()
            u, v, p = forward_field(t, nets, xy_fit, plan, geom, sigma, sharpness)
            loss = ((u - targets_z[:, 0:1]) ** 2 + (v - targets_z[:, 1:2]) ** 2
                    + (p - targets_z[:, 2:3]) ** 2).mean()
            loss.backward()
            opt.step()
    return float(loss.detach().cpu())


# --------------------------------------------------------------------- data prep
def load_dense(data_dir: Path) -> List[dict]:
    path = data_dir / "field_dense.csv"
    if not path.is_file():
        raise FileNotFoundError(f"S1 truth missing: {path}")
    header, rows = art.read_csv_rows(path)
    return [dict(zip(header, r)) for r in rows]


def pick_collocation(rows: List[dict], n: int, seed: int) -> List[dict]:
    interior = [r for r in rows if r.get("boundary_type") == "interior"]
    if not interior:
        raise ValueError("field_dense.csv has no interior points")
    if len(interior) <= n:
        return interior
    step = len(interior) / float(n)
    return [interior[int(i * step)] for i in range(n)]


# --------------------------------------------------------------------- gate main
def run_gate(case_root: Path, case_id: str, level: str, sigma: float,
             plans: Sequence[str]) -> dict:
    import torch                                        # instance-only dependency
    t = torch
    t0 = time.time()
    data_dir = case_root / "data" / case_id
    geom = tg.TGeometry(tg.case_by_id(case_id))
    truth = truth_score(case_root, case_id, level, geom)
    print(f"[truth] momentum_mse={truth['momentum_mse']:.6g} "
          f"balance={truth['balance_ratio']:.4g} n={truth['n_stencils']} "
          f"fd_step={truth['fd_step_gate']}")

    rows = load_dense(data_dir)
    coll = pick_collocation(rows, TRAIN_BUDGET["collocation"], TRAIN_BUDGET["seed"])
    xy = t.tensor([[float(r["x_star"]) / X_NORM, float(r["y_star"]) / Y_NORM]
                   for r in coll], dtype=t.float64, requires_grad=True)
    raw = t.tensor([[float(r["u_star"]), float(r["v_star"]), float(r["p_star"])]
                    for r in coll], dtype=t.float64)
    mean, std = raw.mean(dim=0), raw.std(dim=0).clamp(min=1.0e-12)
    targets_z = (raw - mean) / std
    sharp = TRAIN_BUDGET["envelope_sharpness"]

    chain: Dict[str, Dict[str, float]] = {}
    model_scores: Dict[str, float] = {}
    extra: Dict[str, object] = {}
    for plan in plans:
        in_dim = 2 if plan == "a" else 2 + len(tg.GEOMETRY_FEATURES)
        nets = build_nets(t, in_dim, TRAIN_BUDGET["hidden"], TRAIN_BUDGET["activation"],
                          TRAIN_BUDGET["seed"])
        final_loss = train_nets(t, nets, xy, targets_z, plan, geom, sigma, sharp,
                               TRAIN_BUDGET["steps"], TRAIN_BUDGET["lrs"])
        with t.enable_grad():
            auto = autograd_derivs(t, nets, xy, plan, geom, sigma, sharp)
        ref = fd_derivs(t, nets, xy, plan, geom, sigma, sharp)
        first = max(_rel_err(auto[k], ref[k]) for k in ("u_x", "u_y", "v_x", "v_y",
                                                        "p_x", "p_y"))
        second = max(_rel_err(auto[k], ref[k]) for k in ("u_xx", "u_yy", "v_xx", "v_yy"))
        reading = residual_from({k: auto[k] for k in ("u_xx", "u_yy", "v_xx", "v_yy",
                                                      "p_x", "p_y", "u_x", "v_y")})
        model_scores[plan] = reading["momentum_mse"]
        chain[plan] = {"first_total_derivative": first, "second_total_derivative": second}
        extra[f"plan_{plan}"] = {"supervised_loss_z": final_loss,
                                 "continuity_mse": reading["continuity_mse"],
                                 "momentum_by_fd": residual_from({
                                     k: t.tensor(ref[k], dtype=t.float64).reshape(-1, 1)
                                     for k in ("u_xx", "u_yy", "v_xx", "v_yy",
                                               "p_x", "p_y", "u_x", "v_y")})}
        if plan == "b":
            with t.enable_grad():
                frozen = autograd_derivs(t, nets, xy, "b", geom, sigma, sharp,
                                         detach_features=True)
            extra["v4_style_detached_features_momentum_mse"] = residual_from(frozen)[
                "momentum_mse"]
            extra["v4_style_detached_vs_fd_first"] = max(
                _rel_err(frozen[k], ref[k]) for k in ("u_x", "u_y", "v_x", "v_y",
                                                      "p_x", "p_y"))
            contract = contract_first_derivatives(t, nets, xy, geom, sigma)
            contract_first_err = max(_rel_err(contract[k], ref[k])
                                    for k in ("u_x", "u_y", "v_x", "v_y"))
            extra["plan_b_contract_vs_fd_first"] = contract_first_err
            # the contract path cannot produce the second derivatives without the
            # feature Hessian, so it is gated on first derivatives only and inherits
            # the graph path's second-derivative reading.
            chain["b_contract"] = {"first_total_derivative": contract_first_err,
                                   "second_total_derivative": chain["b"][
                                       "second_total_derivative"]}
    if not model_scores:
        raise ValueError("no plan gated")
    denominator = min(model_scores.values())
    verdict = rs.k0_verdict(truth["momentum_mse"], denominator, chain,
                            truth["fd_step_gate"], truth["balance_ratio"])
    verdict.update({
        "case": case_id, "level": level, "sigma": sigma, "plans": list(plans),
        "truth_momentum_mse": truth["momentum_mse"],
        "truth_continuity_mse": truth["continuity_mse"],
        "truth_n_stencils": truth["n_stencils"],
        "model_momentum_mse": model_scores,
        "ratio_truth_over_model": {k: truth["momentum_mse"] / max(v, 1.0e-30)
                                   for k, v in model_scores.items()},
        "chain": chain, "extra": extra,
        "denominator_is": "fixed-budget supervised fit on the dense truth "
                          "(a best-case model reading, not the final S3 PINN)",
        "elapsed_s": round(time.time() - t0, 1), "env": art.env_lock(),
    })
    art.write_json(data_dir / "k0_verdict.json", verdict)
    return verdict


def dry_run_report(case_root: Path, case_id: str, level: str, sigma: float) -> dict:
    geom = tg.TGeometry(tg.case_by_id(case_id))
    data_dir = case_root / "data" / case_id
    required = [f"cfd/{case_id}_{level}/{case_id}_{level}_samples_{b}_{tag}.csv"
                for b in BRANCH_SAMPLE_SETS + ("junction",) for tag in ("h", "h2")]
    audit = rs.audit_reproduction()
    exact = rs.corrected_terms(12.0, 12.0, 0.0).momentum
    rep = {
        "thresholds_frozen": {
            "K0-S1_truth_over_model": rs.K0_TRUTH_OVER_MODEL_MAX,
            "K0-S2_balance_band": [rs.K0_BALANCE_RATIO_LO, rs.K0_BALANCE_RATIO_HI],
            "K0-S3_abs_residual_star_units": rs.K0_ABS_RESIDUAL_MAX,
            "K0-S4_fd_step_halve": rs.K0_FD_STEP_HALVE_MAX,
            "K0-C_first": rs.K0_CHAIN_FIRST_REL_MAX,
            "K0-C_second": rs.K0_CHAIN_SECOND_REL_MAX},
        "train_budget": TRAIN_BUDGET,
        "sigma": {"primary": SIGMA_PRIMARY, "ablation": SIGMA_ABLATION, "requested": sigma},
        "geometry": {"area": geom.area(), "j_point": geom.j_point, "crotch": geom.crotch,
                     "features": list(tg.GEOMETRY_FEATURES)},
        "required_inputs": required,
        "missing_inputs": [p for p in required if not (case_root / p).is_file()],
        "field_dense_present": (data_dir / "field_dense.csv").is_file(),
        "expected_output": str(data_dir / "k0_verdict.json"),
        "arithmetic_self_check": {
            "as_found_reproduces_audit": [r["rel_err_vs_audit"] for r in audit],
            "as_found_values": [r["as_found_exact_stokes_score"] for r in audit],
            "corrected_exact_solution_score": exact},
    }
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description="route2 K0 truth self-scoring gate")
    ap.add_argument("--case", default=DEFAULT_CASE)
    ap.add_argument("--level", default="hgrade", help="which S1 mesh level to score")
    ap.add_argument("--case-root", default=str(HERE.parents[1] / "cases" / "tbif_2d"))
    ap.add_argument("--sigma", type=float, default=SIGMA_PRIMARY)
    ap.add_argument("--plans", default="a,b")
    ap.add_argument("--dry-run", action="store_true", help="no torch, no training")
    args = ap.parse_args()

    case_root = Path(args.case_root)
    plans = [p.strip() for p in args.plans.split(",") if p.strip()]
    if any(p not in ("a", "b") for p in plans):
        raise SystemExit(f"--plans accepts a and/or b, got {plans}")
    if not (0.02 <= args.sigma <= 0.5):
        raise SystemExit(f"--sigma out of preregistered range: {args.sigma}")

    if args.dry_run:
        rep = dry_run_report(case_root, args.case, args.level, args.sigma)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        errs = rep["arithmetic_self_check"]["as_found_reproduces_audit"]
        if any(e > 5.0e-3 for e in errs) or rep["arithmetic_self_check"][
                "corrected_exact_solution_score"] != 0.0:
            print("DRY-RUN INVALID: scorer arithmetic disagrees with the 2026-09-25 audit")
            return 2
        if rep["missing_inputs"] or not rep["field_dense_present"]:
            print("DRY-RUN PARTIAL: S1 artifacts absent, so the gate would halt on input "
                  "loading. Arithmetic checks above are the only evidence.")
            return 0
        print("DRY-RUN OK: gate arithmetic reproducible without torch. The torch half "
              "(chain completeness, surrogate reading) is untested on this machine.")
        return 0

    verdict = run_gate(case_root, args.case, args.level, args.sigma, plans)
    out = case_root / "data" / args.case / "k0_verdict.json"
    print("K0 " + ("PASS" if verdict["pass"] else "FAIL") + " failed="
          + json.dumps(verdict["failed"], ensure_ascii=False))
    for k, v in verdict["ratio_truth_over_model"].items():
        print(f"  ratio(truth/model) plan {k}: {v:.4g}")
    for k in ("v4_style_detached_features_momentum_mse", "v4_style_detached_vs_fd_first",
              "plan_b_contract_vs_fd_first"):
        if k in verdict["extra"]:
            print(f"  {k}: {verdict['extra'][k]:.4g}")
    print(f"json={out} elapsed={verdict['elapsed_s']}s")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
