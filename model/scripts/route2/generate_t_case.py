#!/usr/bin/env python3
"""S1: symmetric T-bifurcation ground truth with FreeFEM++ + mesh-independence gate.

Artifacts of a real run (on the instance, inside the repository so the truth is
tracked and re-hashable):

    <root>/cfd/<case>_<level>/<case>_<level>.edp        the solve script
    <root>/cfd/<case>_<level>/..._raw.csv               mesh vertices + field
    <root>/cfd/<case>_<level>/..._summary.csv           solver-side boundary integrals
    <root>/cfd/<case>_<level>/..._samples_*.csv         structured FD grids (2 spacings)
    <root>/cfd/<case>_<level>/..._sections.csv         cross-section profiles
    <root>/data/<case>/field_dense.csv                  post-processed truth (finest level)
    <root>/data/<case>/mesh_independence.json          the <10% gate verdict
    <root>/data/<case>/s1_plan.json                    geometry, levels, quantities
    <root>/data/<case>/sha256sums.json                 manifest + ENV_LOCK

Only stdlib is imported, so `--dry-run` renders every .edp and the whole plan on a
machine without numpy/torch (this laptop).  Residual self-scoring on those grids is
K0's job (k0_truth_gate.py), not this script's.

Dimensionless steady Stokes in star units (W_stem = 1, inlet mean velocity = 1,
mu = 1):  lap(u) = grad(p), div(u) = 0, no-slip walls, p = 0 on both outlets.
Three global refinement levels plus one junction-graded level.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import artifacts as art                                # noqa: E402
import residual_scorers as rs                          # noqa: E402
import t_geometry as tg                                # noqa: E402

LABEL = {"inlet": 1, "outlet_up": 2, "outlet_down": 3, "wall": 4}
import tempfile as _tempfile                       # noqa: E402
# must be platform-neutral: a hard-coded "D:/..." default became a literal directory named
# "D:" on the Linux instance.  ROUTE2_OUT wins, else the system temp dir, else the local
# scratch tree the laptop uses.
SCRATCH_DEFAULT = (Path(__import__("os").environ["ROUTE2_OUT"]) / "s1_dryrun"
                   if __import__("os").environ.get("ROUTE2_OUT")
                   else (Path("D:/PINN-restart/.scratch/route2/s1_dryrun")
                         if _tempfile.gettempdir().startswith(("C:", "D:"))
                         else Path(_tempfile.gettempdir()) / "route2_s1_dryrun"))
REPO_CASE_ROOT = HERE.parents[1] / "cases" / "tbif_2d"
SAMPLE_HEADER = "x_star,y_star,u_star,v_star,p_star,xi,eta,branch"
SECTION_HEADER = "branch,xi,eta,x_star,y_star,u_star,v_star,p_star"
BRANCH_STATION_START = 0.25
STATION_STEP = 0.25


# ----------------------------------------------------------------- FreeFEM text
def render_edp(geom: tg.TGeometry, case: tg.TCase, level: dict, out_dir: Path,
               section_eta_step: float = 0.02) -> str:
    spacing = level["spacing"]
    counts = level["counts"]
    jx, jy = geom.j_point
    o: List[str] = []
    a = o.append
    a(f"// Auto-generated route-2 symmetric-T Stokes solve: {case.case_id} / {level['name']}")
    a(f"// theta_deg={case.theta_deg} L_stem={case.l_stem} L_branch={case.l_branch} "
      f"W_up={case.w_branch_up} W_down={case.w_branch_down} target h={spacing}")
    a("// star units: inlet mean velocity = 1, both outlet pressures = 0, mu = 1")
    a("real JX = %.12g; real JY = %.12g;" % (jx, jy))
    a("real CT = %.12g; real ST = %.12g;" % (geom.cos_t, geom.sin_t))
    a("real LSTEM = %.12g; real HSTEM = %.12g;" % (case.l_stem, geom.h_stem))
    a("real LBR = %.12g; real HBUP = %.12g; real HBDN = %.12g;"
  % (case.l_branch, geom.h_branch["up"], geom.h_branch["down"]))
    a("")
    a("func int inDomain(real xx, real yy) {")
    a("  if (xx >= -1e-12 && xx <= LSTEM + 1e-12 && abs(yy) <= HSTEM + 1e-12) return 1;")
    a("  real a1 = (xx - JX) * CT + (yy - JY) * ST;")
    a("  real b1 = -(xx - JX) * ST + (yy - JY) * CT;")
    a("  if (a1 >= -1e-12 && a1 <= LBR + 1e-12 && abs(b1) <= HBUP + 1e-12) return 1;")
    a("  real a2 = (xx - JX) * CT - (yy - JY) * ST;")
    a("  real b2 = -(xx - JX) * ST - (yy - JY) * CT;")
    a("  if (a2 >= -1e-12 && a2 <= LBR + 1e-12 && abs(b2) <= HBDN + 1e-12) return 1;")
    a("  return 0;")
    a("}")
    a("")
    for idx, ((ax, ay), (bx, by), label) in enumerate(
            zip(geom.polygon.verts, [e[1] for e in geom.polygon.edge_ends],
                geom.polygon.edge_labels)):
        a("border b%d(t=0., 1.) {" % idx)
        a("  x = %.12g + (%.12g - %.12g) * t;" % (ax, bx, ax))
        a("  y = %.12g + (%.12g - %.12g) * t;" % (ay, by, ay))
        a("  label = %d;" % LABEL[label])
        a("}")
    a("")
    a("mesh Th = buildmesh(" + "+".join(f"b{i}({counts[i]})" for i in range(len(counts))) + ");")
    a("")
    a("fespace Vh(Th, P2);")
    a("fespace Qh(Th, P1);")
    a("Vh u, v, ut, vt;")
    a("Qh p, qt;")
    a("")
    a("func real uIn(real yy) { return 1.5 * (1.0 - (2.0 * yy) * (2.0 * yy)); }")
    a("")
    a("solve Stokes([u,v,p], [ut,vt,qt], solver=UMFPACK) =")
    a("    int2d(Th)(")
    a("        dx(u)*dx(ut) + dy(u)*dy(ut)")
    a("      + dx(v)*dx(vt) + dy(v)*dy(vt)")
    a("      - p*(dx(ut) + dy(vt))")
    a("      - qt*(dx(u) + dy(v))")
    a("      - 1.0e-10*p*qt")
    a("    )")
    a("  + on(1, u=uIn(y), v=0)")
    a("  + on(4, u=0, v=0)")
    a("  + on(2, p=0)")
    a("  + on(3, p=0)")
    a("  ;")
    a("")
    a("int[int] vTag(Th.nv);")
    a("for (int i = 0; i < Th.nv; ++i) vTag[i] = 0;")
    a("for (int be = 0; be < Th.nbe; ++be) {")
    a("  int iv0 = Th.be(be)[0];")
    a("  int iv1 = Th.be(be)[1];")
    a("  int lab = Th.be(be).label;")
    a("  if (vTag[iv0] == 0 || (lab <= 3 && vTag[iv0] == 4)) vTag[iv0] = lab;")
    a("  if (vTag[iv1] == 0 || (lab <= 3 && vTag[iv1] == 4)) vTag[iv1] = lab;")
    a("}")
    a("")
    prefix = out_dir / f"{case.case_id}_{level['name']}"
    a(f'ofstream fr("{prefix.as_posix()}_raw.csv");')
    a('fr << "x_star,y_star,u_star,v_star,p_star,bc_tag" << endl;')
    a("for (int i = 0; i < Th.nv; ++i) {")
    a("  real xx = Th(i).x; real yy = Th(i).y;")
    a('  fr << xx << "," << yy << "," << u(xx,yy) << "," << v(xx,yy) << "," << p(xx,yy)'
      ' << "," << vTag[i] << endl;')
    a("}")
    a("")
    a(f'ofstream fs("{prefix.as_posix()}_summary.csv");')
    a('fs << "key,value" << endl;')
    a('fs << "nv," << Th.nv << endl;')
    a('fs << "nt," << Th.nt << endl;')
    a('fs << "domain_area," << int2d(Th)(1.0) << endl;')
    a('fs << "q_in_edp," << (-int1d(Th, 1)(u * N.x + v * N.y)) << endl;')
    a('fs << "q_up_edp," << (int1d(Th, 2)(u * N.x + v * N.y)) << endl;')
    a('fs << "q_down_edp," << (int1d(Th, 3)(u * N.x + v * N.y)) << endl;')
    a('fs << "p_in_mean_edp," << (int1d(Th, 1)(p) / int1d(Th, 1)(1.0)) << endl;')
    a('fs << "stokes_weak_grad_sq," << int2d(Th)(dx(u)*dx(u)+dy(u)*dy(u)'
      '+ dx(v)*dx(v)+dy(v)*dy(v)) << endl;')
    a('fs << "div_energy_edp," << int2d(Th)(pow(dx(u)+dy(v),2)) << endl;')
    a("")
    for key in (tg.STEM, tg.UP, tg.DOWN):
        for mult, tag in ((1.0, "h"), (2.0, "h2")):
            o.extend(_emit_branch_grid(geom, geom.branch_grid(key, spacing * mult),
                                       prefix, case, tag))
    for mult, tag in ((1.0, "h"), (2.0, "h2")):
        o.extend(_emit_junction_grid(geom, geom.junction_grid(spacing * mult), prefix, case, tag))
    o.extend(_emit_sections(geom, prefix, case, section_eta_step))
    a('cout << "done" << endl;')
    return "\n".join(o) + "\n"


def _emit_branch_grid(geom: tg.TGeometry, spec: dict, prefix: Path, case: tg.TCase,
                      tag: str) -> List[str]:
    path = Path(prefix.as_posix() + f"_samples_{spec['branch']}_{tag}.csv")
    n_xi, n_eta = spec["n_xi"], spec["n_eta"]
    d_xi = (spec["xi1"] - spec["xi0"]) / max(n_xi - 1, 1)
    d_eta = 2.0 * spec["eta_max"] / max(n_eta - 1, 1)
    ox, oy = spec["origin"]
    (dx_, dy_), (mx, my) = spec["d"], spec["m"]
    return [
        "{",
        f'  ofstream fo("{path.as_posix()}");',
        f'  fo << "{SAMPLE_HEADER}" << endl;',
        f"  int NX = {n_xi}; int NE = {n_eta};",
        f'  real XI0 = {spec["xi0"]:.12g}; real DXI = {d_xi:.12g};',
        f'  real ETA0 = {-spec["eta_max"]:.12g}; real DETA = {d_eta:.12g};',
        f"  real OX = {ox:.12g}; real OY = {oy:.12g};",
        f"  real DDX = {dx_:.12g}; real DDY = {dy_:.12g};",
        f"  real MDX = {mx:.12g}; real MDY = {my:.12g};",
        "  for (int i = 0; i < NX; i++) {",
        "    real xi = XI0 + i * DXI;",
        "    for (int j = 0; j < NE; j++) {",
        "      real eta = ETA0 + j * DETA;",
        "      real xx = OX + xi * DDX + eta * MDX;",
        "      real yy = OY + xi * DDY + eta * MDY;",
        '      fo << xx << "," << yy << "," << u(xx,yy) << "," << v(xx,yy) << "," << p(xx,yy)'
        f' << "," << xi << "," << eta << ",{spec["branch"]}" << endl;',
        "    }",
        "  }",
        "}",
        "",
    ]


def _emit_junction_grid(geom: tg.TGeometry, spec: dict, prefix: Path, case: tg.TCase,
                        tag: str) -> List[str]:
    path = Path(prefix.as_posix() + f"_samples_junction_{tag}.csv")
    n_x = max(3, int(round((spec["x1"] - spec["x0"]) / spec["spacing"])) + 1)
    n_y = max(3, int(round((spec["y1"] - spec["y0"]) / spec["spacing"])) + 1)
    return [
        "{",
        f'  ofstream fo("{path.as_posix()}");',
        f'  fo << "{SAMPLE_HEADER}" << endl;',
        f"  int NX = {n_x}; int NY = {n_y};",
        f'  real X0 = {spec["x0"]:.12g}; real DX = {(spec["x1"] - spec["x0"]) / (n_x - 1):.12g};',
        f'  real Y0 = {spec["y0"]:.12g}; real DY = {(spec["y1"] - spec["y0"]) / (n_y - 1):.12g};',
        f'  real CX = {spec["cx"]:.12g}; real CY = {spec["cy"]:.12g}; '
        f'real R = {spec["r_excl"]:.12g};',
        "  for (int i = 0; i < NX; i++) {",
        "    real xx = X0 + i * DX;",
        "    for (int j = 0; j < NY; j++) {",
        "      real yy = Y0 + j * DY;",
        "      if ((xx - CX) * (xx - CX) + (yy - CY) * (yy - CY) < R * R) continue;",
        "      if (!inDomain(xx, yy)) continue;",
        '      fo << xx << "," << yy << "," << u(xx,yy) << "," << v(xx,yy) << "," << p(xx,yy)'
        ' << "," << xx << "," << yy << ",junction" << endl;',
        "    }",
        "  }",
        "}",
        "",
    ]


def _emit_sections(geom: tg.TGeometry, prefix: Path, case: tg.TCase,
                   eta_step: float) -> List[str]:
    """Cross-section profiles -> Q(xi) per branch, centreline p(xi), mass closure."""
    path = Path(prefix.as_posix() + "_sections.csv")
    o: List[str] = ["{",
                    f'  ofstream fo("{path.as_posix()}");',
                    f'  fo << "{SECTION_HEADER}" << endl;',
                    f"  int NE = {int(round(2.0 * geom.max_half_width() / eta_step)) + 1};",
                    f"  real DETA = {eta_step:.12g};"]
    for key in (tg.STEM, tg.UP, tg.DOWN):
        fr = geom.frames[key]
        xi0 = 0.0 if fr.key == tg.STEM else BRANCH_STATION_START
        n_sec = int(round((fr.length - xi0) / STATION_STEP)) + 1
        ox, oy = fr.origin
        o += ["  {",
              f"    real OX = {ox:.12g}; real OY = {oy:.12g};",
              f"    real DDX = {fr.d[0]:.12g}; real DDY = {fr.d[1]:.12g};",
              f"    real MDX = {fr.m[0]:.12g}; real MDY = {fr.m[1]:.12g};",
              f"    real HW = {fr.half_width:.12g};",
              f"    for (int s = 0; s < {n_sec}; s++) {{",
              f"      real xi = {xi0:.12g} + s * {STATION_STEP:.12g};",
              "      for (int j = 0; j < NE; j++) {",
              "        real eta = -HW + j * DETA;",
              f"        if (eta > HW + 1e-12) break;",
              "        real xx = OX + xi * DDX + eta * MDX;",
              "        real yy = OY + xi * DDY + eta * MDY;",
              f'        fo << "{fr.name}," << xi << "," << eta << "," << xx << "," << yy << ","'
              f' << u(xx,yy) << "," << v(xx,yy) << "," << p(xx,yy) << endl;',  # one line!
              "      }",
              "    }",
              "  }"]
    o += ["}", ""]
    return o


# ------------------------------------------------------------------------ lint
OPERATOR_START = ("<<", ">>", "&&", "||", "++", "--")


def lint_edp(text: str) -> list[str]:
    """Static checks on a rendered .edp, run before it is ever handed to FreeFEM.

    FreeFEM's parser does not continue a statement across lines, so a stream statement
    split in two is a hard compile error at the second line's `<<` -- which is exactly
    how the first real S1 run died (line 293, code=1).  Catching it here costs a second
    and does not need FreeFEM installed.
    """
    bad: list[str] = []
    for n, line in enumerate(text.split("\n"), start=1):
        t = line.strip()
        if not t or t.startswith("//"):
            continue
        if t.startswith(OPERATOR_START):
            bad.append(f"line {n}: statement continuation starts with an operator: {t[:40]!r}")
        if "<<" in t and not t.endswith(";"):
            bad.append(f"line {n}: stream statement does not end with ';' on one line")
        if t.endswith((",", "<<")):
            bad.append(f"line {n}: line ends mid-expression: {t[-20:]!r}")
    if text.count("{") != text.count("}"):
        bad.append(f"unbalanced braces: {text.count('{')} '{{' vs {text.count('}')} '}}'")
    if "\r" in text:
        bad.append("rendered .edp contains CR bytes (would break FreeFEM on Linux)")
    for n, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if stripped.startswith("border b") and not stripped.endswith("{"):
            bad.append(f"line {n}: border header malformed: {stripped[:40]!r}")
    return bad


def assert_edp_clean(text: str, name: str) -> None:
    problems = lint_edp(text)
    if problems:
        raise ValueError(f"{name} failed the .edp lint:\n  " + "\n  ".join(problems[:12]))


# ------------------------------------------------------------------- processing
def build_field_dense(raw_path: Path, geom: tg.TGeometry) -> Tuple[List[dict], dict]:
    """Mesh-vertex CSV -> field_dense rows, boundary type classified geometrically."""
    header, rows = art.read_csv_rows(raw_path)
    idx = {name: i for i, name in enumerate(header)}
    out: List[dict] = []
    stats = {"n_vertices": len(rows), "n_kept": 0, "outside": 0, "bc_tag_disagree": 0}
    tol = 1.0e-6
    for row in rows:
        x = float(row[idx["x_star"]])
        y = float(row[idx["y_star"]])
        if not geom.contains(x, y, tol=1.0e-6):
            stats["outside"] += 1
            continue
        fr = geom.frames[geom.primary_frame(x, y)]
        xi, eta = fr.local(x, y)
        btype = "interior"
        if abs(abs(eta) - fr.half_width) <= tol:
            btype = "wall"
        if fr.key == tg.STEM and abs(xi) <= tol:
            btype = "inlet"
        if fr.key in (tg.UP, tg.DOWN) and abs(xi - fr.length) <= tol:
            btype = "outlet_up" if fr.key == tg.UP else "outlet_down"
        tag = int(float(row[idx["bc_tag"]]))
        if (tag in (1, 2, 3)) != (btype in ("inlet", "outlet_up", "outlet_down")):
            stats["bc_tag_disagree"] += 1
        dist = geom.wall_distance_exact(x, y)
        region = 0
        if math.hypot(x - geom.j_point[0], y - geom.j_point[1]) <= 1.5 * max(fr.half_width, 1e-9):
            region = 1
        if dist < 0.15 * fr.half_width:
            region = 2
        feats = geom.features(x, y, 0.15)
        rec = {"sample_id": len(out), "case_id": geom.case.case_id, "family": geom.case.family,
               "x_star": x, "y_star": y,
               "u_star": float(row[idx["u_star"]]), "v_star": float(row[idx["v_star"]]),
               "p_star": float(row[idx["p_star"]]),
               "speed_star": math.hypot(float(row[idx["u_star"]]), float(row[idx["v_star"]])),
               "branch": fr.name, "xi_star": xi, "eta_star": eta,
               "wall_distance_star": dist, "region_id": region,
               "is_boundary": int(btype != "interior"), "boundary_type": btype,
               "bc_tag": tag}
        for name in tg.GEOMETRY_FEATURES:
            rec["feat_" + name] = feats[name]
        out.append(rec)
    stats["n_kept"] = len(out)
    return out, stats


BRANCH_KEY = {"stem": tg.STEM, "branch_up": tg.UP, "branch_down": tg.DOWN}


def section_integrals(section_path: Path, geom: tg.TGeometry) -> dict:
    """Q(xi) per section (trapezoid across the section) and centreline p(xi)."""
    header, rows = art.read_csv_rows(section_path)
    idx = {name: i for i, name in enumerate(header)}
    per: Dict[str, Dict[float, List[Tuple[float, float, float]]]] = {}
    for row in rows:
        br = row[idx["branch"]]
        xi = round(float(row[idx["xi"]]), 9)
        eta = float(row[idx["eta"]])
        u = float(row[idx["u_star"]])
        v = float(row[idx["v_star"]])
        p = float(row[idx["p_star"]])
        fr = geom.frames[BRANCH_KEY[br]]
        per.setdefault(br, {}).setdefault(xi, []).append((eta, u * fr.d[0] + v * fr.d[1], p))
    q: Dict[str, Dict[float, float]] = {}
    pcl: Dict[str, Dict[float, float]] = {}
    for br, sections in per.items():
        q[br], pcl[br] = {}, {}
        for xi, samples in sections.items():
            samples.sort()
            q[br][xi] = _trapz([s[1] for s in samples], [s[0] for s in samples])
            pcl[br][xi] = min(samples, key=lambda s: abs(s[0]))[2]
    return {"q_per_section": q, "p_centreline": pcl}


def _trapz(vals: Sequence[float], coords: Sequence[float]) -> float:
    return sum(0.5 * (vals[i] + vals[i - 1]) * (coords[i] - coords[i - 1])
               for i in range(1, len(vals)))


def quantities(summary: dict, ints: dict, junction_ps: List[float]) -> dict:
    """The gated quantities plus the absolute closures.  All integral or averaged."""
    qs, qu, qd = ints["q_per_section"]["stem"], ints["q_per_section"]["branch_up"], \
        ints["q_per_section"]["branch_down"]
    ps, pu, pd = ints["p_centreline"]["stem"], ints["p_centreline"]["branch_up"], \
        ints["p_centreline"]["branch_down"]
    q_in = _nearest(qs, 0.0)
    q_up_out, q_down_out = _nearest(qu, max(qu)), _nearest(qd, max(qd))
    p_in = _nearest(ps, 0.0)
    p_out_up, p_out_down = _nearest(pu, max(pu)), _nearest(pd, max(pd))
    p_j = sum(junction_ps) / max(len(junction_ps), 1)
    conservation = 0.0
    for table in (qs, qu, qd):
        if not table:
            continue
        ref = abs(_nearest(table, max(table)))
        if ref <= 1.0e-12:
            continue
        conservation = max(conservation,
                           max(abs(q - ref) for q in table.values()) / ref)
    return {
        "q_stem": q_in,
        "q_up": q_up_out,
        "q_down": q_down_out,
        "dp_stem_to_up": p_in - p_out_up,
        "dp_stem_to_down": p_in - p_out_down,
        "p_junction_over_outlet": p_j - 0.5 * (p_out_up + p_out_down),
        "split_fraction_up": q_up_out / max(q_in, 1e-12),
        "mass_closure_residual": abs(q_in - q_up_out - q_down_out) / max(abs(q_in), 1e-12),
        "flux_conservation_max_rel": conservation,
        "q_in_edp_crosscheck": summary.get("q_in_edp"),
        "q_up_edp_crosscheck": summary.get("q_up_edp"),
        "q_down_edp_crosscheck": summary.get("q_down_edp"),
        "n_junction_samples": len(junction_ps),
        "nv": summary.get("nv"), "nt": summary.get("nt"),
        "domain_area_edp": summary.get("domain_area"),
    }


def _nearest(table: Dict[float, float], target: float) -> float:
    if not table:
        return float("nan")
    return table[min(table, key=lambda k: abs(k - target))]


# --------------------------------------------------------------------- driver
def freefem_executable() -> str:
    for cand in ("FreeFem++", "FreeFEM++", "freefem++"):
        path = shutil.which(cand)
        if path:
            return path
    raise FileNotFoundError("FreeFem++ not in PATH")


def run_case(case: tg.TCase, out_root: Path, levels: List[dict], execute: bool,
             sigma: float = 0.15) -> dict:
    geom = tg.TGeometry(case)
    data_dir = out_root / "data" / case.case_id
    cfd_root = out_root / "cfd"
    data_dir.mkdir(parents=True, exist_ok=True)
    plan: dict = {"case": case.to_metadata(),
                  "geometry": {"polygon_ccw": geom.polygon.is_ccw(), "area": geom.area(),
                               "j_point": geom.j_point, "crotch": geom.crotch,
                               "outer_corners": {str(k): v for k, v in geom.corners.items()},
                               "a_outer": {str(k): v for k, v in geom.a_outer.items()},
                               "blend_sigma_star": sigma,
                               "feature_names": list(tg.GEOMETRY_FEATURES)},
                  "level_order": [], "levels": [], "quantities_by_level": {}}
    last_dense: List[dict] = []
    for level in levels:
        counts = tg.border_counts(geom, level["spacing"], level["graded"])
        lvl = dict(level, counts=counts)
        lvl_dir = cfd_root / f"{case.case_id}_{lvl['name']}"
        lvl_dir.mkdir(parents=True, exist_ok=True)
        edp = lvl_dir / f"{case.case_id}_{lvl['name']}.edp"
        edp.write_text(render_edp(geom, case, lvl, lvl_dir), encoding="utf-8")
        entry = {"level": lvl["name"], "spacing_star": lvl["spacing"], "graded": lvl["graded"],
                 "border_counts": lvl["counts"], "edp": edp.name,
                 "edp_sha256": art.sha256_file(edp),
                 "expected": [f"{case.case_id}_{lvl['name']}{suf}.csv" for suf in
                              ["_raw", "_summary", "_sections"]
                              + [f"_samples_{n}_{t}" for n in
                                 ("stem", "branch_up", "branch_down", "junction")
                                 for t in ("h", "h2")]]}
        if not execute:
            entry["status"] = "rendered-only (dry-run: nothing solved, gate not evaluated)"
            plan["levels"].append(entry)
            continue
        exe = freefem_executable()
        # `-nw` only.  `-noplot` is NOT a switch in FreeFem++ v4.9: it is parsed as an
        # input file name and every solve dies with "lex: Error input opening file"
        # (measured on the instance 2026-09-25, after I added it on a freeglut hunch).
        subprocess.run([exe, "-nw", str(edp)], check=True, cwd=str(lvl_dir))
        raw = lvl_dir / f"{case.case_id}_{lvl['name']}_raw.csv"
        summary_p = lvl_dir / f"{case.case_id}_{lvl['name']}_summary.csv"
        sections = lvl_dir / f"{case.case_id}_{lvl['name']}_sections.csv"
        missing = [str(p) for p in (raw, summary_p, sections) if not p.is_file()]
        for name in entry["expected"][3:]:
            if not (lvl_dir / name).is_file():
                missing.append(str(lvl_dir / name))
        if missing:
            raise FileNotFoundError("FreeFEM did not produce: " + ", ".join(missing))
        dense, stats = build_field_dense(raw, geom)
        last_dense = dense
        ints = section_integrals(sections, geom)
        jun = _read_p(lvl_dir / f"{case.case_id}_{lvl['name']}_samples_junction_h.csv")
        q = quantities(_read_summary(summary_p), ints, jun)
        q["mesh_spacing_star"] = lvl["spacing"]
        plan["quantities_by_level"][lvl["name"]] = q
        plan["level_order"].append(lvl["name"])
        plan["levels"].append(dict(entry, status="solved", dense_stats=stats))
        art.write_csv(lvl_dir / f"{case.case_id}_{lvl['name']}_field_dense.csv",
                      list(dense[0].keys()), [list(d.values()) for d in dense])
        art.write_csv(lvl_dir / f"{case.case_id}_{lvl['name']}_sections_tidy.csv",
                      ["branch", "xi_star", "q_star"],
                      [[br, xi, val] for br, table in ints["q_per_section"].items()
                       for xi, val in sorted(table.items())])
    if execute and plan["level_order"]:
        order = plan["level_order"]
        table = {name: [plan["quantities_by_level"][lv][name] for lv in order]
                 for name in tg.MESH_INDEPENDENCE_QUANTITIES}
        gate = rs.mesh_independence_gate(table)
        expected_split = 0.5 if case.is_geometrically_symmetric else None
        abs_gate = rs.absolute_gates(
            plan["quantities_by_level"][order[-1]]["q_stem"],
            plan["quantities_by_level"][order[-1]]["q_up"],
            plan["quantities_by_level"][order[-1]]["q_down"],
            plan["quantities_by_level"][order[-1]]["flux_conservation_max_rel"],
            expected_split=expected_split)
        art.write_csv(data_dir / "field_dense.csv", list(last_dense[0].keys()),
                      [list(d.values()) for d in last_dense])
        art.write_json(data_dir / "mesh_independence.json", {
            "limit_rel_change": rs.MESH_INDEPENDENCE_REL_MAX,
            "level_order": order, "quantities_by_level": plan["quantities_by_level"],
            "relative_change_gate": gate, "absolute_gates": abs_gate,
            "pass": bool(gate["pass"] and abs_gate["pass"]),
            "verdict": ("PASS" if gate["pass"] and abs_gate["pass"]
                        else "FAIL -> halt: no credible truth for this geometry"),
        })
        plan["mesh_independence"] = {"relative": gate, "absolute": abs_gate}
    art.write_json(data_dir / "s1_plan.json", plan)
    plan["manifest_files"] = len(_manifest(out_root, case, data_dir)["files"])
    art.write_json(data_dir / "sha256sums.json", _manifest(out_root, case, data_dir))
    return {"plan": plan, "data_dir": data_dir}


def _read_p(path: Path) -> List[float]:
    header, rows = art.read_csv_rows(path)
    idx = {n: i for i, n in enumerate(header)}
    return [float(r[idx["p_star"]]) for r in rows]


def _read_summary(path: Path) -> dict:
    header, rows = art.read_csv_rows(path)
    return {r[0]: float(r[1]) for r in rows if len(r) >= 2}


def _manifest(out_root: Path, case: tg.TCase, skip_dir: Path) -> dict:
    paths = [p for p in out_root.rglob("*") if p.is_file() and skip_dir not in p.parents]
    env: dict = {}
    try:
        proc = subprocess.run([freefem_executable(), "-nw", "-e",
                               "cout<<version<<endl;"],
                              capture_output=True, text=True, timeout=120)
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        env["freefem_version"] = tail[-1] if tail else "ran (no banner)"
    except Exception as exc:
        env["freefem_version"] = f"unavailable: {type(exc).__name__}: {exc}"
    return {"case": case.to_metadata(),
            "env": art.env_lock(env),
            "files": {str(p).replace("\\", "/"): art.sha256_file(p) for p in sorted(paths)}}


def main() -> int:
    ap = argparse.ArgumentParser(description="route2 S1 T-bifurcation truth generator")
    ap.add_argument("--case", default="TB-base")
    ap.add_argument("--dry-run", action="store_true",
                    help="render .edp + plan only; refuses to write inside the repository")
    ap.add_argument("--out-root", default="",
                    help=f"artifact root (default: {SCRATCH_DEFAULT} when --dry-run, "
                         f"{REPO_CASE_ROOT} otherwise)")
    ap.add_argument("--levels", default="", help="comma list, e.g. h1,h2 (default all four)")
    ap.add_argument("--base-spacing", type=float, default=0.16)
    ap.add_argument("--blend-sigma", type=float, default=0.15)
    args = ap.parse_args()

    if not (0.02 <= args.blend_sigma <= 0.5):
        raise SystemExit(f"--blend-sigma out of range: {args.blend_sigma}")
    case = tg.case_by_id(args.case)
    levels = tg.mesh_levels(args.base_spacing)
    if args.levels:
        wanted = {s.strip() for s in args.levels.split(",") if s.strip()}
        levels = [lv for lv in levels if lv["name"] in wanted]
        unknown = wanted - {lv["name"] for lv in levels}
        if unknown:
            raise SystemExit(f"unknown level(s): {sorted(unknown)}")
    out_root = Path(args.out_root) if args.out_root else (
        SCRATCH_DEFAULT if args.dry_run else REPO_CASE_ROOT)
    repo_root = HERE.parents[1].resolve()
    if args.dry_run and repo_root in out_root.resolve().parents:
        raise SystemExit(f"--dry-run refuses to write inside the repository: {out_root}")
    execute = not args.dry_run
    if execute:
        try:
            exe = freefem_executable()
        except FileNotFoundError as exc:
            # overlay disk is not persistent: a restarted instance loses FreeFEM, and a
            # silent fall back to "not measured" would make the segment look completed
            print(f"[ABORT] {exc} -- container disk is not persistent, re-install first. "
                  f"Rerun with --dry-run only renders scripts.")
            return 3
        print(f"[gate] FreeFEM resolved to {exe}")
    res = run_case(case, out_root, levels, execute, sigma=args.blend_sigma)
    plan = res["plan"]
    print(f"out_root={out_root}")
    for entry in plan["levels"]:
        print(f"  level {entry['level']:8s} h={entry['spacing_star']:.4g} "
              f"borders={sum(entry['border_counts']):4d} graded={entry['graded']} "
              f"status={entry['status']}")
    if "mesh_independence" in plan:
        rel = plan["mesh_independence"]["relative"]
        print(f"mesh gate: relative pass={rel['pass']} worst={rel['worst_rel_change']:.4g} "
              f"({rel['worst_quantity']}) limit={rel['limit']}")
        for name, row in rel["quantities"].items():
            print(f"  {name:24s} rel(last pair)={row.get('last_rel_change', float('nan')):.4g} "
                  f"ok={row['ok']}")
        abs_g = plan["mesh_independence"]["absolute"]
        print(f"absolute gates pass={abs_g['pass']}")
    else:
        print("mesh gate: NOT EVALUATED (dry-run renders scripts only)")
    print("stop rule: gate fail => halt and report; no credible truth, no downstream run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
