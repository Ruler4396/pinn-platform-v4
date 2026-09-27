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
    <root>/data/<case>/env-probe.json                  solver/interpreter provenance, including the
                                                       FreeFEM start-up wall second (volatile by design)
    <root>/data/<case>/sha256sums.json                 content_sha256 over every artefact byte except
                                                       itself and env-probe.json + files_digest

Only stdlib is imported, so `--dry-run` renders every .edp and the whole plan on a
machine without numpy/torch (this laptop).  Residual self-scoring on those grids is
K0's job (k0_truth_gate.py), not this script's.

Dimensionless steady Stokes in star units (W_stem = 1, inlet mean velocity = 1,
mu = 1):  lap(u) = grad(p), div(u) = 0, no-slip walls, p = 0 on both outlets.
Three global refinement levels plus one junction-graded level.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
SECTION_ETA_STEP = 0.02
BRANCH_STATION_START = tg.TGeometry.BRANCH_STATION_START   # declared once, in t_geometry
STATION_STEP = tg.TGeometry.STATION_STEP


def section_eta_nodes(half_width: float, eta_step: float = SECTION_ETA_STEP) -> List[float]:
    """The eta nodes across one section, from wall to wall, and back to the emitter.

    n cells of exactly 2*HW/n so both walls are nodes: a fixed step with "break when
    past HW" stops 0.01 short on the 0.85W branch (42.5 cells) and that truncation
    biases the branch flux by 2.85e-4, independent of mesh refinement.
    """
    n = max(2, int(round(2.0 * half_width / eta_step)))
    d = 2.0 * half_width / n
    return [-half_width + j * d for j in range(n + 1)]


# ------------------------------------------- 12-digit truth reference (WP-A / K0b)
# Pre-registration: paper-route2/K0b-高精度真值参考-预注册-20260927.md.  Two measured facts set
# the shape: v4.9's `ofstream <<` prints six significant digits, and it rejects `setprecision`
# at compile time -- so a 12-digit truth has to be built on the writing side, in stages.
#
# WHY STAGES, AND THE NUMBER THAT MAKES THIS A FINDING RATHER THAN A TASTE.  One hi/lo split at
# lattice 1e-N must satisfy two inequalities: hi printable in six digits (N < 6 - log10(scale))
# and the residual small enough that its own six digits reach the target
# (N >= -log10(2 * scale * 10**-digits) - 6).  The window between them is
# `2*PRINT + log10(2) - digits` ~ 12.3 - digits integers, INDEPENDENT of the magnitude:
#     digits = 10 -> window ~2.3   (what T6-Re used: widths 4/6/2, floors 10.6-11.0 digits)
#     digits = 12 -> window ~0.3   (lucky magnitudes only; impossible for |p| of order 400)
# So a single split cannot carry K0b's reference for this geometry.  Each extra stage prints a
# chunk of the previous residual and adds six digits of resolution.  Nothing is emitted yet --
# see TRUTH_EMISSION_WIRED, which the self-check asserts rather than assumes.
TRUTH_DIGITS_REQUIRED = 12
TRUTH_STAGES_MAX = 4
TRUTH_EMISSION_WIRED = True         # the staged companion is rendered; see selfcheck_emission


def stage_widths(scale: float, stages: int) -> list:
    """Lattice decimals per stage: every stage's hi must survive a six-digit printer."""
    out = []
    room = scale
    for k in range(stages - 1):
        n = int(math.floor(tg.FREEFEM_PRINT_DIGITS - math.log10(room) - 1e-12))
        if n < 1:
            raise ValueError("scale %g: stage %d has no printable lattice" % (scale, k + 1))
        out.append(n)
        room = 10.0 ** -n
    return out


def stage_floor(widths: list) -> float:
    """Absolute error the staged stream still carries.

    The last column is the leftover residual, printed by the same six-digit printer, so its
    error is 5e-7 of ITS OWN size -- and that size is set by the last *lattice* stage (a
    residual is below 1e-N of the stage that produced it).  Using the 0 marker instead would
    claim 5e-7 for every plan, which is the mistake this function first shipped with.
    """
    lattice = [n for n in widths if n]
    if not lattice:
        return math.inf
    return 0.5 * 10.0 ** -(lattice[-1] + tg.FREEFEM_PRINT_DIGITS)


def split_plan(scale: float, digits: int = TRUTH_DIGITS_REQUIRED) -> list:
    """The fewest stages that reach `digits` relative to `scale`, or a refusal."""
    if not scale > 0:
        raise ValueError("no measurable scale (got %r)" % (scale,))
    for stages in range(2, TRUTH_STAGES_MAX + 1):
        try:
            widths = stage_widths(scale, stages)
        except ValueError:
            continue
        widths = widths + [0]                        # last column prints the remainder
        # `* (1 + 1e-9)`: the boundary case is exact equality by construction (a scale of
        # 0.5 with N=6 lands on 1e-12), and without the tolerance the answer would flip on
        # the last binary bit of two different machines.
        if stage_floor(widths) <= 10.0 ** -digits * scale * (1.0 + 1e-9):
            return widths
    raise ValueError(
        "scale %g: %d stages still miss %d digits. Do not lower the requirement and do not "
        "emit a coarser truth -- report it: the reference is not buildable for this magnitude "
        "with a six-digit printer." % (scale, TRUTH_STAGES_MAX, digits))


def measure_scales(case_dir: Path):
    """Magnitudes of x*, y*, u, v, p from an existing samples stream, or None.

    Six printed digits suffice to place a lattice (the staging needs log10(scale) to a
    fraction), so a prior run is a legitimate scale source and a guess is not.
    """
    found = sorted(case_dir.glob("*_samples_*.csv"))
    if not found:
        return None
    mx = {k: 0.0 for k in ("x", "y", "u", "v", "p")}
    for path in found:
        with path.open(encoding="utf-8", newline="") as fh:
            head = fh.readline().strip().split(",")
            want = ("x_star", "y_star", "u_star", "v_star", "p_star")
            if any(c not in head for c in want):
                continue
            idx = [head.index(c) for c in want]
            for ln in fh:
                parts = ln.strip().split(",")
                if len(parts) <= max(idx):
                    continue
                for name, k in zip(mx, idx):
                    try:
                        mx[name] = max(mx[name], abs(float(parts[k])))
                    except ValueError:
                        pass
    return {k: (v or 1.0) for k, v in mx.items()} or None


def assert_truth_digits_plan(digits: int, plans: dict) -> None:
    """Hard refusal before anything is written, when the truth cannot carry `digits`."""
    if digits < TRUTH_DIGITS_REQUIRED:
        raise SystemExit(
            "--truth-digits %d < %d: the 6-digit truth is what made the old K0 undecidable, so "
            "asking for it again is refused here rather than warned about (K0b pre-registration, "
            "2026-09-27)." % (digits, TRUTH_DIGITS_REQUIRED))
    for name, widths in plans.items():
        if not widths:
            raise SystemExit("%s: no staging plan -- refusing to write" % name)


# ------------------------------------------- the companion (staged) stream itself
# K0b pre-registration §一: the field values cannot be printed at 12 digits, so each sample
# grid gets a second stream written in the SAME loop, in the SAME row order, carrying the
# staged chunks.  Coordinates are deliberately not in it -- the lattice never goes through the
# printer (the .edp expands Python's `:.12g` parameters), so staging them would invent columns
# nothing reads.
STAGED_FIELDS = ("u_star", "v_star", "p_star")
STAGED_SCALE_KEY = {"u_star": "u", "v_star": "v", "p_star": "p"}
STAGED_VAR = {"u_star": "qU", "v_star": "qV", "p_star": "qP"}
STAGED_SUFFIX = "_staged"


def staged_plans(scales: dict, digits: int = TRUTH_DIGITS_REQUIRED) -> dict:
    """Per-field plans keyed by the CSV column name, from the measured scales."""
    return {f: split_plan(scales[STAGED_SCALE_KEY[f]], digits) for f in STAGED_FIELDS}


def staged_header(plans: dict) -> list:
    """Companion columns in print order: `<field>_s0..s<k-1>`, the last one is the leftover."""
    return ["%s_s%d" % (f, j) for f in STAGED_FIELDS for j in range(len(plans[f]))]


def staged_chunk_lines(field: str, plan: list) -> list:
    """FreeFEM statements splitting `qU` (already evaluated) into its staged chunks.

    chunk_j = floor(residual * 1e-N_j) / 1.0e-N_j, and the next stage works on what that left.
    Identifiers carry no underscore because v4.9 rejects `_` in code (`lint_edp` enforces it on
    the rendered text); the header is a string literal, so it keeps the `_`.
    """
    var = STAGED_VAR[field]
    out, rest = [], var
    for j, n in enumerate(plan[:-1]):
        out.append("      real %s%d = floor(%s*1.0e%d)/1.0e%d;" % (var, j, rest, n, n))
        out.append("      real %sr%d = %s - %s%d;" % (var, j, rest, var, j))
        rest = "%sr%d" % (var, j)
    return out


def staged_print_terms(field: str, plan: list) -> list:
    """The expressions printed for one field -- exactly `len(plan)` of them."""
    var = STAGED_VAR[field]
    terms = ["%s%d" % (var, j) for j in range(len(plan) - 1)]
    terms.append("%sr%d" % (var, len(plan) - 2) if len(plan) > 1 else var)
    return terms


def staged_value(row: dict, field: str) -> float:
    """Merge one staged row back: the sum of its chunks, read by name from the header.

    The reconstruction is the sum because each chunk is the floor of the residual at its own
    lattice, so the chunks partition the value down to the leftover's printing bound.
    """
    return sum(float(row[c]) for c in row if c.startswith(field + "_s"))


def assert_staged_render(text: str, plans: dict, n_grids: int, name: str) -> None:
    """The rendered post-condition: every companion is there, and every one prints as many
    columns as its plan says.  Written because the first splice landed after a `return`, i.e.
    it rendered nothing and still printed a plan."""
    expected = len(staged_header(plans))
    heads = re.findall(r'ofstream fst\("([^"]+)"\);', text)
    if len(heads) != n_grids:
        raise ValueError(f"{name}: {len(heads)} companion streams rendered, expected "
                         f"{n_grids} -- a staged plan that is not emitted is the same defect "
                         f"as an uncalled assertion")
    for path in heads:
        if not path.endswith(".csv") or (STAGED_SUFFIX + ".csv") not in path:
            raise ValueError(f"{name}: companion stream {path} is not named <grid>"
                             f"{STAGED_SUFFIX}.csv, so the reader will not find it")
    hdrs = re.findall(r'fst << "([^"]+)" << endl;', text)
    if len(hdrs) != n_grids:
        raise ValueError(f"{name}: {len(hdrs)} companion headers, expected {n_grids}")
    for h in hdrs:
        cols = h.split(",")
        if len(cols) != expected:
            raise ValueError(f"{name}: the companion header carries {len(cols)} columns "
                             f"({cols}) while the plan says {expected} "
                             f"({staged_header(plans)}) -- header and plan disagreeing is how "
                             f"a merge silently shifts by one column")
    body = [b for b in re.findall(r'fst << ([^\n]*?) << endl;', text)
            if not b.strip().startswith('"')]          # the header line is a bare literal
    if len(body) != n_grids:
        raise ValueError(f"{name}: {len(body)} companion row prints, expected {n_grids}")
    for b in body:
        terms = [t for t in (x.strip() for x in b.split("<<"))
                 if t and not (t.startswith('"') and t.endswith('"'))]
        if len(terms) != expected:
            raise ValueError(f"{name}: a companion row prints {len(terms)} values "
                             f"({terms}) against {expected} header columns -- a header and a "
                             f"row that disagree are how a merge shifts by one column silently")


def staged_widths_from_edp(text: str) -> dict:
    """Read the staged lattices back out of the .edp that wrote a companion.

    The rule lives in the artefact, not in a docstring: an earlier revision of the sibling
    T6 merger carried `hi = 1e-4` in prose while the .edp emitted something else, and the
    assertion passed on both.  Parsing the emitted `floor(qU*1.0eN)/1.0eN` makes the reader
    disagree with a stale .edp instead of agreeing with a wrong claim.

    Parsed per companion block, and all blocks must agree: the same eight grids each re-declare
    their chunks, so scanning the whole file at once would return eight copies of the plan and
    call that a lattice (which is exactly what the first version of this function did).
    """
    blocks = text.split('ofstream fst("')[1:]
    if not blocks:
        raise ValueError("the .edp carries no companion stream, so there is no staged lattice "
                         "to read -- this is a 6-digit emission, not a K0b reference")
    per_block = []
    for blk in blocks:
        found = {}
        for field in STAGED_FIELDS:
            var = STAGED_VAR[field]
            ns = [int(m.group(1)) for m in
                  re.finditer(r"floor\(%s\w*\*1\.0e(\d+)\)/1\.0e\1" % var, blk)]
            if not ns:
                raise ValueError(f"a companion block carries no staged chunks for {field} "
                                 f"(looked for floor({var}...)) -- cannot read a companion "
                                 f"whose lattice is not in the file that wrote it")
            found[field] = ns + [0]
        per_block.append(found)
    if any(b != per_block[0] for b in per_block[1:]):
        raise ValueError(f"the .edp's companion blocks disagree on the lattice: "
                         f"{ {k: [b[k] for b in per_block] for k in STAGED_FIELDS} } -- one "
                         f"grid staged differently from the others is not one reference")
    return per_block[0]


def staged_budget_scale(width: int) -> float:
    """Largest |value| whose 1e-`width` chunk still fits a six-digit printer."""
    return 10.0 ** (tg.FREEFEM_PRINT_DIGITS - width)


def selfcheck_emission() -> int:
    """Controls for the staging plan.  At least one of them is a refusal."""
    rc = 0
    scales = {"x": 5.0, "y": 0.5, "u": 2.5, "v": 2.5, "p": 415.058}
    plans = {k: split_plan(v, TRUTH_DIGITS_REQUIRED) for k, v in scales.items()}
    plans_f = staged_plans(scales, TRUTH_DIGITS_REQUIRED)
    try:
        assert_truth_digits_plan(6, plans)
        ok, why = False, "NO REFUSAL -- a 6-digit truth would have been emitted"
    except SystemExit as exc:
        ok, why = True, str(exc).splitlines()[0][:64]
    rc |= 0 if ok else 1
    print("[%s] control: --truth-digits 6 is refused, not warned (%s)"
          % ("PASS" if ok else "FAIL", why))
    worst = max(stage_floor(plans[k]) / scales[k] for k in scales)
    ok = worst <= 10.0 ** -TRUTH_DIGITS_REQUIRED * (1.0 + 1e-9)
    rc |= 0 if ok else 1
    print("[%s] staging reaches 1e-%d of every scale: worst %.2e; plan %s"
          % ("PASS" if ok else "FAIL", TRUTH_DIGITS_REQUIRED, worst, plans))
    try:
        split_plan(1.0e9, 12)
        ok, why = False, "accepted an unstaggeable magnitude"
    except ValueError as exc:
        ok, why = True, str(exc)[:56]
    rc |= 0 if ok else 1
    print("[%s] control: |value|=1e9 is refused, not quietly coarsened (%s)"
          % ("PASS" if ok else "FAIL", why))
    p10, p12 = split_plan(415.058, 10), split_plan(415.058, 12)
    ok = len(p10) == 2 and len(p12) == 3
    rc |= 0 if ok else 1
    print("[%s] the window claim as a number: |p|=415 needs 2 stages for 10 digits (T6's case)"
          " and 3 for 12 (K0b's) -> %s vs %s" % ("PASS" if ok else "FAIL", p10, p12))
    ok = bool(TRUTH_EMISSION_WIRED)
    rc |= 0 if ok else 1
    print("[%s] wiring control: TRUTH_EMISSION_WIRED=%s -- it must be True before a run claims "
          "12-digit truth, and the two checks below are what back the word 'wired'"
          % ("PASS" if ok else "FAIL", TRUTH_EMISSION_WIRED))
    # The splice is judged on a RENDERED file, not on the flag: the first attempt set the plan,
    # passed the flag, and put the companion after a `return`.
    case = tg.case_by_id("TB-base")
    geom = tg.TGeometry(case)
    lvl = {"name": "h1", "spacing": 0.1, "graded": False,
           "counts": tg.border_counts(geom, 0.1, False)}
    text = render_edp(geom, case, lvl, Path("/tmp/x"), truth_plans=plans_f)
    grids = text.count('ofstream fst("')
    ok = grids == 8 and text.count(STAGED_SUFFIX + '.csv') == 8
    rc |= 0 if ok else 1
    print("[%s] render control: a real .edp carries %d companion streams and %d staged file "
          "names (8 sample grids, coordinates excluded on purpose)"
          % ("PASS" if ok else "FAIL", grids, text.count(STAGED_SUFFIX + ".csv")))
    bad = {f: plans_f[f][:1] for f in STAGED_FIELDS}          # one column short, as if the
    try:                                                      # plan changed after rendering
        assert_staged_render(text, bad, 8, "synthetic.edp")
        ok, why = False, "NO REFUSAL -- a header/plan mismatch would ship"
    except ValueError as exc:
        ok, why = True, str(exc).splitlines()[0][:78]
    rc |= 0 if ok else 1
    print("[%s] control: a rendered companion whose columns disagree with the plan is refused "
          "(%s)" % ("PASS" if ok else "FAIL", why))
    return rc


# ----------------------------------------------------------------- FreeFEM text
def precision_prefix(var: str, digits: int) -> list[str]:
    """Always empty: kept only so the emitters stay readable.

    FreeFem++ v4.9 rejects `setprecision` at compile time ("The Identifier setprecision
    does not exist"), which killed every solve in a35a6ab.  Coordinates cannot be made
    more precise from the .edp, so acceptance matches the file's own ruler instead --
    see t_geometry.representation_bound and the measured FREEFEM_PRINT_DIGITS = 6.
    """
    return []


def render_edp(geom: tg.TGeometry, case: tg.TCase, level: dict, out_dir: Path,
               section_eta_step: float = SECTION_ETA_STEP,
               coord_digits: int = 17,
               truth_plans: Optional[dict] = None) -> str:
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
    o.extend(precision_prefix("fr", coord_digits))
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
                                       prefix, case, tag, coord_digits, truth_plans))
    for mult, tag in ((1.0, "h"), (2.0, "h2")):
        o.extend(_emit_junction_grid(geom, geom.junction_grid(spacing * mult), prefix, case,
                                  tag, coord_digits, truth_plans))
    o.extend(_emit_sections(geom, prefix, case, section_eta_step, coord_digits))
    a('cout << "done" << endl;')
    text = "\n".join(o) + "\n"
    if truth_plans:
        # 3 branches x (h, h2) + junction x (h, h2): the grids `sample_grid()` reads.  The
        # post-condition is on the RENDERED TEXT, not on the argument list, because the first
        # splice put the companion after a `return` and still printed a plan.
        assert_staged_render(text, truth_plans, 8,
                             f"{case.case_id}_{level['name']}.edp")
    return text


def _emit_branch_grid(geom: tg.TGeometry, spec: dict, prefix: Path, case: tg.TCase,
                      tag: str, digits: int = 17,
                      plans: Optional[dict] = None) -> List[str]:
    path = Path(prefix.as_posix() + f"_samples_{spec['branch']}_{tag}.csv")
    n_xi, n_eta = spec["n_xi"], spec["n_eta"]
    d_xi = (spec["xi1"] - spec["xi0"]) / max(n_xi - 1, 1)
    d_eta = 2.0 * spec["eta_max"] / max(n_eta - 1, 1)
    ox, oy = spec["origin"]
    (dx_, dy_), (mx, my) = spec["d"], spec["m"]
    o = [
        "{",
        f'  ofstream fo("{path.as_posix()}");',
        *precision_prefix("fo", digits),
        f'  fo << "{SAMPLE_HEADER}" << endl;',
    ]
    o += _staged_stream_lines(path, plans)
    o += [
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
        *(_field_value_lines() if plans else []),
        '      fo << xx << "," << yy << "," << u(xx,yy) << "," << v(xx,yy) << "," << p(xx,yy)'
        f' << "," << xi << "," << eta << ",{spec["branch"]}" << endl;',
        *_staged_print_lines(plans),
        "    }",
        "  }",
        "}",
        "",
    ]
    return o


def _field_value_lines() -> List[str]:
    """Evaluate each field once, in the order the primary stream prints them.

    Both streams have to read the SAME numbers, and printing `u(xx,yy)` twice in two `<<`
    chains is two interpolations of the same FE function at the same point: identical, but
    only this form lets the staged stream print the value the primary one printed rather than
    a second evaluation that could differ if the point changed.
    """
    return ["      real qU = u(xx,yy); real qV = v(xx,yy); real qP = p(xx,yy);"]


def _staged_stream_lines(path: Path, plans: Optional[dict]) -> List[str]:
    if not plans:
        return []
    staged = Path(str(path).replace(".csv", STAGED_SUFFIX + ".csv"))
    return [f'  ofstream fst("{staged.as_posix()}");',
            f'  fst << "{",".join(staged_header(plans))}" << endl;']


def _staged_print_lines(plans: Optional[dict]) -> List[str]:
    if not plans:
        return []
    o: List[str] = []
    for field in STAGED_FIELDS:
        o += staged_chunk_lines(field, plans[field])
    terms = []
    for field in STAGED_FIELDS:
        terms += staged_print_terms(field, plans[field])
    o.append("      fst << " + ' << "," << '.join(terms) + " << endl;")
    return o


def _emit_junction_grid(geom: tg.TGeometry, spec: dict, prefix: Path, case: tg.TCase,
                        tag: str, digits: int = 17,
                        plans: Optional[dict] = None) -> List[str]:
    path = Path(prefix.as_posix() + f"_samples_junction_{tag}.csv")
    n_x = max(3, int(round((spec["x1"] - spec["x0"]) / spec["spacing"])) + 1)
    n_y = max(3, int(round((spec["y1"] - spec["y0"]) / spec["spacing"])) + 1)
    o = [
        "{",
        f'  ofstream fo("{path.as_posix()}");',
        *precision_prefix("fo", digits),
        f'  fo << "{SAMPLE_HEADER}" << endl;',
    ]
    o += _staged_stream_lines(path, plans)
    o += [
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
        *(_field_value_lines() if plans else []),
        '      fo << xx << "," << yy << "," << u(xx,yy) << "," << v(xx,yy) << "," << p(xx,yy)'
        ' << "," << xx << "," << yy << ",junction" << endl;',
        *_staged_print_lines(plans),
        "    }",
        "  }",
        "}",
        "",
    ]
    return o


def _emit_sections(geom: tg.TGeometry, prefix: Path, case: tg.TCase,
                   eta_step: float, digits: int = 17) -> List[str]:
    """Cross-section profiles -> Q(xi) per branch, centreline p(xi), mass closure."""
    path = Path(prefix.as_posix() + "_sections.csv")
    o: List[str] = ["{",
                    f'  ofstream fo("{path.as_posix()}");',
                    *precision_prefix("fo", digits),
                    f'  fo << "{SECTION_HEADER}" << endl;']
    for key in (tg.STEM, tg.UP, tg.DOWN):
        fr = geom.frames[key]
        xis = geom.station_xis(key)          # the one declaration shared with the gate
        xi0 = xis[0]
        n_sec = len(xis)
        # One eta cell per frame, ending exactly on both walls.  A shared DETA = 0.02 with
        # "break when past HW" silently stops 0.01 short on the 0.85W branch (42.5 cells),
        # and that truncation biases the branch flux by 2.85e-4 -- refinement-independent,
        # and it lands only on the kill-test geometry.
        eta_nodes = section_eta_nodes(fr.half_width, eta_step)
        n_eta = len(eta_nodes) - 1
        ox, oy = fr.origin
        o += ["  {",
              f"    real OX = {ox:.12g}; real OY = {oy:.12g};",
              f"    real DDX = {fr.d[0]:.12g}; real DDY = {fr.d[1]:.12g};",
              f"    real MDX = {fr.m[0]:.12g}; real MDY = {fr.m[1]:.12g};",
              f"    real HW = {fr.half_width:.12g};",
              f"    int NE = {len(eta_nodes)}; real DETA = {eta_nodes[1] - eta_nodes[0]:.12g};",
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


def significant_digit_roundtrip(value: float,
                                digits: int = tg.FREEFEM_PRINT_DIGITS) -> float:
    """The nearest double to `value` written with `digits` significant digits.

    This is FreeFEM's own operation (`ofstream` ~ `%.*g`), and it must not be replaced by
    `round(v/10**(e-d+1))*10**(e-d+1)`: multiplying back is inexact, and measured on
    `contraction_2d/cfd/C-base/C-base_raw.csv` that variant calls only **58.5%** of the
    `x_star` values 6-digit fixed points when the true share is **100.0%**
    (max_dev 1.78e-15 vs 0.0 -- that artifact is the probe, not the file; 统括官 caught it
    with `toPrecision(6)` on 2026-09-25 and his number is the one that stands).
    """
    return float("%.*g" % (digits, value))


def coordinate_digits_used(rows: Sequence[Sequence[str]], cols: Sequence[str]) -> dict:
    """How many significant digits does a raw CSV actually carry?  Read, not assumed.

    Re-rounding a value to 6 significant digits and seeing no change is proof the file
    was written at 6 digits; that single number tells us whether the precision question is
    closed, so the next report cannot be argued about.
    """
    sig6 = significant_digit_roundtrip
    out: dict = {}
    for name in cols:
        try:
            i = list(rows[0]).index(name) if name in (rows[0] if rows else []) else cols.index(name)
        except ValueError:
            continue
        vals = []
        for r in rows[:4000]:
            try:
                vals.append(float(r[i]))
            except (ValueError, IndexError):
                continue
        if not vals:
            continue
        unchanged = sum(1 for v in vals if v == sig6(v))
        worst = max(abs(v - sig6(v)) for v in vals)
        out[name] = {"share_unchanged_by_6sig_roundtrip": unchanged / len(vals),
                     "max_abs_deviation_from_6sig": worst}
    return out


# ------------------------------------------------------------------- processing
def build_field_dense(raw_path: Path, geom: tg.TGeometry) -> Tuple[List[dict], dict]:
    """Mesh-vertex CSV -> field_dense rows, boundary type classified geometrically."""
    header, rows = art.read_csv_rows(raw_path)
    idx = {name: i for i, name in enumerate(header)}
    out: List[dict] = []
    stats = {"n_vertices": len(rows), "n_kept": 0, "bc_tag_disagree": 0,
             **{v: 0 for v in tg.MEMBERSHIP_VERDICTS},
             "absorb_tol_floor_star": tg.ABSORB_TOL,
             "print_digits_assumed": tg.FREEFEM_PRINT_DIGITS,
             "max_reject_frac": tg.MAX_REJECT_FRAC,
             "worst_rejected": [], "max_absorbed_gap_star": 0.0,
             "max_absorbed_gap_bound_star": 0.0, "bc_tag_disagree_examples": []}
    worst: List[Tuple[float, float, float]] = []
    for row in rows:
        x = float(row[idx["x_star"]])
        y = float(row[idx["y_star"]])
        verdict, key, gap = geom.membership(x, y)
        if key is None:
            # dropped, but never silently: counted, and the largest gaps are kept
            stats["outside"] += 1
            worst.append((gap, x, y))
            continue
        if verdict not in stats:
            # a verdict nobody declared must not surface as a KeyError halfway through a
            # 500-second solve; say what is unknown and where it came from.  The counters
            # are derived from tg.MEMBERSHIP_VERDICTS, so this guard cannot drift away
            # from the enum -- the drift is what defect 5 was.
            raise RuntimeError(f"undeclared membership verdict {verdict!r} at ({x}, {y}); "
                               f"known: {tg.MEMBERSHIP_VERDICTS}")
        stats[verdict] += 1
        if verdict in ("frame", "absorbed_print"):
            if not geom.absorbed_within_bound(x, y, gap):
                raise ValueError(
                    f"absorbed vertex ({x}, {y}) has gap {gap:.3e} above the printing "
                    f"bound {geom.representation_bound(x, y):.3e}: the band would be "
                    f"wider than the artefact can explain -> refusing to absorb")
            if gap > stats["max_absorbed_gap_star"]:
                stats["max_absorbed_gap_star"] = gap
                stats["max_absorbed_gap_bound_star"] = geom.representation_bound(x, y)
        fr = geom.frames[key]
        xi, eta = fr.local(x, y)
        # Same ruler as membership(), and for the same reason: xi/eta are linear
        # combinations of the two printed coordinates, so on a rotated boundary (the
        # 45-degree branch outlets) 6-digit printing displaces them by up to
        # 2 x half-ulp -- measured 3.0e-6 and 4.1e-6 at |x| ~ 8, i.e. beyond the fixed
        # 1e-6 band, which made the post-process call genuine outlet vertices "interior"
        # and disagree with the solver's own boundary labels on 18 of 40 Dirichlet nodes.
        boundary_tol = geom.representation_bound(x, y)
        btype = "interior"
        if abs(abs(eta) - fr.half_width) <= boundary_tol:
            btype = "wall"
        if fr.key == tg.STEM and abs(xi) <= boundary_tol:
            btype = "inlet"
        if fr.key in (tg.UP, tg.DOWN) and abs(xi - fr.length) <= boundary_tol:
            btype = "outlet_up" if fr.key == tg.UP else "outlet_down"
        tag = int(float(row[idx["bc_tag"]]))
        if (tag in (1, 2, 3)) != (btype in ("inlet", "outlet_up", "outlet_down")):
            stats["bc_tag_disagree"] += 1
            if len(stats["bc_tag_disagree_examples"]) < 5:
                stats["bc_tag_disagree_examples"].append(
                    {"x_star": x, "y_star": y, "bc_tag": tag, "classified": btype,
                     "gap_to_own_plane": min(abs(xi), abs(xi - fr.length)),
                     "boundary_tol_star": boundary_tol})
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
               "branch": fr.name, "membership": verdict, "rect_gap_star": gap,
               "representation_bound_star": geom.representation_bound(x, y),
               "xi_star": xi, "eta_star": eta,
               "wall_distance_star": dist, "region_id": region,
               "is_boundary": int(btype != "interior"), "boundary_type": btype,
               "bc_tag": tag}
        for name in tg.GEOMETRY_FEATURES:
            rec["feat_" + name] = feats[name]
        out.append(rec)
    stats["n_kept"] = len(out)
    worst.sort(reverse=True)
    stats["worst_rejected"] = [{"gap_star": g, "x_star": x, "y_star": y}
                               for g, x, y in worst[:10]]
    if stats["polygon_without_frame"]:
        # the contour and the frame union must coincide; if they don't, truth is not trusted
        raise ValueError(f"{stats['polygon_without_frame']} vertices inside the contour but "
                         "farther than ABSORB_TOL from every branch rect -- geometry and "
                         "mesh disagree, halting instead of guessing a branch")
    frac = stats["outside"] / max(stats["n_vertices"], 1)
    if frac > tg.MAX_REJECT_FRAC:
        raise ValueError(f"rejected {stats['outside']}/{stats['n_vertices']} vertices "
                         f"(> {tg.MAX_REJECT_FRAC:.1%}); see worst_rejected in the stats")
    return out, stats


def _material_flags(geom: tg.TGeometry, per: dict) -> Dict[str, Dict[float, bool]]:
    """Which stations are material cross-sections (see TGeometry.section_is_material)."""
    return {br: {xi: geom.section_is_material(BRANCH_KEY[br], xi) for xi in tab}
            for br, tab in per.items()}


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
    return {"q_per_section": q, "p_centreline": pcl,
            "material_station": _material_flags(geom, q)}


def _trapz(vals: Sequence[float], coords: Sequence[float]) -> float:
    return sum(0.5 * (vals[i] + vals[i - 1]) * (coords[i] - coords[i - 1])
               for i in range(1, len(vals)))


def _conservation(table: Dict[float, float], flags: Dict[float, bool]
                  ) -> Tuple[float, Optional[float]]:
    """max |Q(xi) - Q(outlet)| / |Q(outlet)| over the stations `flags` accepts.

    Too few material stations is an error, not a zero: a gate that silently reports 0 when
    it had nothing to compare would be the same class of mistake as the missing dict key.
    """
    keys = [xi for xi in sorted(table) if flags.get(xi)]
    if len(keys) < 2:
        raise ValueError(f"flux conservation needs >= 2 material cross-sections, got "
                         f"{len(keys)} of {len(table)} stations; refusing to report 0.0 "
                         f"for a comparison that was never made")
    ref = abs(table[keys[-1]])
    if ref <= 1.0e-12:
        return 0.0, None
    worst = max(keys, key=lambda xi: abs(table[xi] - ref) / ref)
    return abs(table[worst] - ref) / ref, worst


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
    # Along-branch flux conservation is a statement about *material* sections only: near
    # the junction the frame's xi=const line is open on one side (it crosses into the other
    # branch), so the flux through it legitimately differs from the outlet value by an
    # amount refinement does not remove.  That definition -- not the mesh, not the truth --
    # is what made the instance read 0.1454 (TB-base) / 0.2164 (TB-asym) at the 1e-3 gate.
    # The excluded stations are still reported, never dropped.
    mats = ints.get("material_station") or {b: {xi: True for xi in tab}
                                            for b, tab in ints["q_per_section"].items()}
    per_branch: Dict[str, dict] = {}
    conservation = conservation_all = 0.0
    n_gated = n_excluded = 0
    for br, table in (("stem", qs), ("branch_up", qu), ("branch_down", qd)):
        flags = mats.get(br, {})
        rel, xi_w = _conservation(table, flags)
        rel_all, xi_a = _conservation(table, {xi: True for xi in table})
        excl = sorted(xi for xi in table if not flags.get(xi))
        per_branch[br] = {"max_rel_material": rel, "worst_material_xi": xi_w,
                          "max_rel_all_stations": rel_all, "worst_xi_all_stations": xi_a,
                          "n_material": len(table) - len(excl), "n_excluded": len(excl),
                          "excluded_xi": excl}
        conservation = max(conservation, rel)
        conservation_all = max(conservation_all, rel_all)
        n_gated += len(table) - len(excl)
        n_excluded += len(excl)
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
        "flux_conservation_all_stations_rel": conservation_all,
        "flux_per_branch": per_branch,
        "flux_stations_gated": n_gated, "flux_stations_excluded": n_excluded,
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
    """The solver, with an explicit override for the instance-local recovery.

    On a replaced DSW instance FreeFEM comes back from `ffroot.tgz` into a directory under
    /mnt/workspace plus an LD_LIBRARY_PATH -- it is deliberately NOT installed into /usr, because
    that is the recovery that cost the previous instance.  So PATH alone cannot find it, and the
    runner exports FREEFEM_BIN=<absolute path>.  An override that is not executable is a refusal,
    not a silent fall back to PATH: silently dropping back would let "we solved it" mean a
    different binary than the one the trip verified.
    """
    override = os.environ.get("FREEFEM_BIN", "").strip()
    if override:
        if os.path.isfile(override) and os.access(override, os.X_OK):
            return override
        raise FileNotFoundError(f"FREEFEM_BIN={override!r} is not an executable file -- "
                               f"refusing to fall back to a different binary")
    for cand in ("FreeFem++", "FreeFEM++", "freefem++"):
        path = shutil.which(cand)
        if path:
            return path
    raise FileNotFoundError("FreeFem++ not in PATH and FREEFEM_BIN is unset")


def run_case(case: tg.TCase, out_root: Path, levels: List[dict], execute: bool,
             sigma: float = 0.15, coord_digits: int = 17,
             truth_plans: Optional[dict] = None) -> dict:
    geom = tg.TGeometry(case)
    data_dir = out_root / "data" / case.case_id
    cfd_root = out_root / "cfd"
    data_dir.mkdir(parents=True, exist_ok=True)
    timing: Dict[str, float] = {"render_s": 0.0, "solve_s": 0.0, "postprocess_s": 0.0,
                                "manifest_s": 0.0}
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
        _t0 = time.perf_counter()
        text = render_edp(geom, case, lvl, lvl_dir, coord_digits=coord_digits,
                          truth_plans=truth_plans)
        timing["render_s"] += time.perf_counter() - _t0
        assert_edp_clean(text, edp.name)      # refuse to ship an .edp FreeFEM cannot eat
        edp.write_text(text, encoding="utf-8")
        n_evals = _point_evaluations(geom, lvl["spacing"])
        sample_names = [f"{case.case_id}_{lvl['name']}_samples_{n}_{t}.csv"
                        for n in ("stem", "branch_up", "branch_down", "junction")
                        for t in ("h", "h2")]
        # A requested companion that FreeFEM did not write is a FAILURE, not an absence: the
        # plan printed in the log would otherwise look like a 12-digit truth over a 6-digit file.
        expected = ([f"{case.case_id}_{lvl['name']}{suf}.csv"
                     for suf in ["_raw", "_summary", "_sections"]] + sample_names)
        if truth_plans:
            expected += [n.replace(".csv", STAGED_SUFFIX + ".csv") for n in sample_names]
        entry = {"level": lvl["name"], "spacing_star": lvl["spacing"], "graded": lvl["graded"],
                 "n_fem_point_evaluations": n_evals,
                 "border_counts": lvl["counts"], "edp": edp.name,
                 "edp_sha256": art.sha256_file(edp),
                 "expected": expected}
        if not execute:
            entry["status"] = "rendered-only (dry-run: nothing solved, gate not evaluated)"
            plan["levels"].append(entry)
            continue
        exe = freefem_executable()
        _t1 = time.perf_counter()
        # `-nw` only.  `-noplot` is NOT a switch in FreeFem++ v4.9: it is parsed as an
        # input file name and every solve dies with "lex: Error input opening file"
        # (measured on the instance 2026-09-25, after I added it on a freeglut hunch).
        subprocess.run([exe, "-nw", str(edp)], check=True, cwd=str(lvl_dir))
        timing["solve_s"] += time.perf_counter() - _t1
        raw = lvl_dir / f"{case.case_id}_{lvl['name']}_raw.csv"
        summary_p = lvl_dir / f"{case.case_id}_{lvl['name']}_summary.csv"
        sections = lvl_dir / f"{case.case_id}_{lvl['name']}_sections.csv"
        missing = [str(p) for p in (raw, summary_p, sections) if not p.is_file()]
        for name in entry["expected"][3:]:
            if not (lvl_dir / name).is_file():
                missing.append(str(lvl_dir / name))
        if missing:
            raise FileNotFoundError("FreeFEM did not produce: " + ", ".join(missing))
        # Where the wall clock went, measured instead of guessed: the .edp writes its
        # artefacts in stage order (raw -> summary -> 8 sample grids -> sections), so the
        # mtimes of files FreeFEM itself created split startup+buildmesh+solve from the
        # point-evaluation loops -- without adding any FreeFEM verb I have not run.
        t0_art = edp.stat().st_mtime
        produced = [raw, summary_p] + [lvl_dir / n for n in entry["expected"][3:]] + [sections]
        entry["artifact_age_s"] = {p.name: round(p.stat().st_mtime - t0_art, 3)
                                   for p in produced}
        _t2 = time.perf_counter()
        dense, stats = build_field_dense(raw, geom)
        header_probe, probe_rows = art.read_csv_rows(raw)
        stats["coordinate_precision_probe"] = coordinate_digits_used(
            probe_rows, header_probe)
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
        timing["postprocess_s"] += time.perf_counter() - _t2
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
    _t3 = time.perf_counter()
    # 定档丙: the start-up probe carries a wall-clock second, so it gets its own file and stays out of
    # the hashed set.  This is the same information the old manifest carried under "env", relocated.
    art.write_json(data_dir / "env-probe.json",
                   {"case": case.to_metadata(), "env": art.env_lock(freefem_probe())})
    # 定档乙: every byte the manifest will hash is final before the manifest is built, and the manifest
    # is the last thing written.  The old order wrote the plan, hashed everything, then rewrote the plan
    # -- so once 甲 put the plan into the list, the list carried a hash of bytes already replaced.
    # (Getting this order wrong is what the determinism control caught: with the build above the plan
    # write, the manifest hashed a stale s1_plan.json and the digest moved between two identical runs.)
    plan["timing_s"] = {k: round(v, 2) for k, v in timing.items()}
    plan["total_solve_wall_s"] = round(timing["solve_s"], 2)
    art.write_json(data_dir / "s1_plan.json", plan)
    man = _manifest(out_root, case, data_dir / "sha256sums.json", data_dir / "env-probe.json")
    timing["manifest_s"] += time.perf_counter() - _t3     # env-probe + build; the write below is outside
    art.write_json(data_dir / "sha256sums.json", man)
    print(f"manifest n_files={man['n_files']} files_digest={man['files_digest'][:16]}")
    print("timing_s=" + ", ".join(f"{k}:{v}" for k, v in plan["timing_s"].items()))
    print("per-level FEM sample points (each costs 3 u/v/p lookups)="
          + ", ".join(f"{e['level']}:{e['n_fem_point_evaluations']['total']}"
                      f"=g{e['n_fem_point_evaluations']['grid_loops']}"
                      f"+s{e['n_fem_point_evaluations']['section_stations']}"
                      for e in plan["levels"]))
    return {"plan": plan, "data_dir": data_dir}


def _point_evaluations(geom: tg.TGeometry, spacing: float) -> dict:
    """How many `u(x,y)` lookups the .edp will perform, split by the loop that makes them.

    FreeFEM's evaluation of a FEM function at an arbitrary point is a mesh search, so it
    is the natural suspect for wall clock that the solver's own 0.094 s cannot explain.
    Splitting the count matters because the two loops write different files: the mtime of
    the sample grids vs the mtime of `sections.csv` says which of them ate the time, which
    turns my guess into a column the next report can check against its own timer.
    """
    grids = stations = 0
    for key in (tg.STEM, tg.UP, tg.DOWN):
        for mult in (1.0, 2.0):
            spec = geom.branch_grid(key, spacing * mult)
            grids += spec["n_xi"] * spec["n_eta"]
    for mult in (1.0, 2.0):
        jspec = geom.junction_grid(spacing * mult)
        nx = max(3, int(round((jspec["x1"] - jspec["x0"]) / jspec["spacing"])) + 1)
        ny = max(3, int(round((jspec["y1"] - jspec["y0"]) / jspec["spacing"])) + 1)
        grids += nx * ny
    for key in (tg.STEM, tg.UP, tg.DOWN):
        fr = geom.frames[key]
        n_sta = int(round((fr.length - (0.0 if fr.key == tg.STEM else 0.25)) / 0.25)) + 1
        stations += n_sta * (2 * int(round(fr.half_width / 0.02)) + 1)
    return {"grid_loops": grids, "section_stations": stations,
            "total": grids + stations}


def _read_p(path: Path) -> List[float]:
    header, rows = art.read_csv_rows(path)
    idx = {n: i for i, n in enumerate(header)}
    return [float(r[idx["p_star"]]) for r in rows]


def _read_summary(path: Path) -> dict:
    header, rows = art.read_csv_rows(path)
    return {r[0]: float(r[1]) for r in rows if len(r) >= 2}


_FF_PROBE: dict = {}


def freefem_probe() -> dict:
    """How long a bare FreeFem++ process costs in this container, measured once.

    The instance spent 497 s on an h1 step whose own solve timer read 0.094 s, so the
    question is whether the money went into process start-up or into the loops inside the
    .edp.  This probe answers the first half; `artifact_age_s` answers the second.
    """
    if not _FF_PROBE:
        _tp = time.perf_counter()
        try:
            proc = subprocess.run([freefem_executable(), "-nw", "-e",
                                   "cout<<version<<endl;"],
                                  capture_output=True, text=True, timeout=120)
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            _FF_PROBE["freefem_version"] = tail[-1] if tail else "ran (no banner)"
        except Exception as exc:
            _FF_PROBE["freefem_version"] = f"unavailable: {type(exc).__name__}: {exc}"
        _FF_PROBE["freefem_version_probe_wall_s"] = round(time.perf_counter() - _tp, 3)
    return dict(_FF_PROBE)


def _manifest(out_root: Path, case: tg.TCase, manifest_path: Path,
              env_probe_path: Path) -> dict:
    """content_sha256 over every artefact byte, plus a canonical digest of that map.

    Two files are excluded BY NAME, not by directory (定档甲): the manifest itself, which cannot hold
    its own hash, and env-probe.json.  The previous version excluded the whole `data/<case>/` directory,
    which is precisely why #39 found the verdict JSONs, the dense field and the plan missing from the
    list that everyone assumed covered them.

    env-probe.json is split out (定档丙) because freefem_probe() writes a wall-clock second into it, so
    a manifest containing it differs between two identical runs -- and this manifest is the reproduction
    anchor behind the public "truth is recomputable bit for bit" claim (#42).  The digest is computed
    over sorted `path:sha` lines so a comparison is one string equality, not a diff of JSON layout.
    """
    skip = {manifest_path.resolve(), env_probe_path.resolve()}
    paths = [p for p in out_root.rglob("*") if p.is_file() and p.resolve() not in skip]
    files = {str(p).replace("\\", "/"): art.sha256_file(p) for p in sorted(paths)}
    canonical = "\n".join(f"{key}:{value}" for key, value in sorted(files.items()))
    return {"case": case.to_metadata(),
            "n_files": len(files),
            "files": files,
            "files_digest": art.sha256_text(canonical),
            "digest_over": "sha256 of sorted 'path:content_sha256' lines joined by LF",
            "excluded_by_name": [str(manifest_path).replace("\\", "/"),
                                  str(env_probe_path).replace("\\", "/")]}


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
    ap.add_argument("--truth-digits", type=int, default=TRUTH_DIGITS_REQUIRED,
                    help="digits the truth must carry; below TRUTH_DIGITS_REQUIRED is refused")
    ap.add_argument("--truth-scales", default="",
                    help="x,y,u,v,p magnitudes; empty = measure from existing samples CSVs, "
                         "and refuse if there are none")
    ap.add_argument("--selfcheck-emission", action="store_true",
                    help="check the staging plan and the refusals, no FreeFEM")
    ap.add_argument("--coord-precision", type=int, default=0,
                    help="deprecated: v4.9 has no setprecision, so nothing is emitted "
                         "into the .edp; anything > 0 is refused (it broke every solve)")
    ap.add_argument("--selfcheck-raw", default="",
                    help="run one real FreeFEM *_raw.csv through build_field_dense and "
                         "print the membership accounting; no solve, no writes")
    args = ap.parse_args()
    if args.selfcheck_emission:
        return selfcheck_emission()

    if args.selfcheck_raw:
        case = tg.case_by_id(args.case)
        geom = tg.TGeometry(case)
        raw = Path(args.selfcheck_raw)
        dense, stats = build_field_dense(raw, geom)
        header, rows = art.read_csv_rows(raw)
        stats["coordinate_precision_probe"] = coordinate_digits_used(rows, header)
        print(json.dumps({"raw": str(raw), "stats": stats,
                          "n_dense_rows": len(dense),
                          "absorb_tol_star_unchanged": tg.ABSORB_TOL},
                         ensure_ascii=False, indent=2))
        return 0
    if args.coord_precision > 0:
        raise SystemExit("--coord-precision >0 is not available: FreeFem++ v4.9 refuses "
                         "`setprecision` at compile time. Precision is handled by "
                         "t_geometry.representation_bound(); leave it at 0.")
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
    if args.truth_scales.strip():
        scales = dict(zip(("x", "y", "u", "v", "p"),
                          (float(v) for v in args.truth_scales.split(","))))
    else:
        cfd = out_root / "cfd"
        scales = (measure_scales(cfd / ("%s_%s" % (case.case_id, levels[0]["name"])))
                  or measure_scales(cfd))
        if not scales:
            raise SystemExit(
                "no magnitude to stage against: no *_samples_*.csv under %s and "
                "--truth-scales is empty. Refusing to guess -- the staging, and hence whether "
                "the truth can carry %d digits, follows from the magnitude."
                % (cfd, args.truth_digits))
    truth_plans = {k: split_plan(scales[k], args.truth_digits) for k in scales}
    assert_truth_digits_plan(args.truth_digits, truth_plans)
    staged = staged_plans(scales, args.truth_digits)
    print("truth staging (K0b): "
          + ", ".join("%s=%s" % (k, truth_plans[k]) for k in sorted(truth_plans))
          + "; worst floor %.2e of scale (target 1e-%d)"
          % (max(stage_floor(truth_plans[k]) / scales[k] for k in truth_plans),
             args.truth_digits)
          + "; companion columns " + str(staged_header(staged)))
    if TRUTH_EMISSION_WIRED and not staged:
        raise SystemExit("TRUTH_EMISSION_WIRED is True but no staged plan was built -- the "
                         "emitter would render the old single-stream .edp while the log claims "
                         "a companion, so refusing to run either way")
    if not TRUTH_EMISSION_WIRED:
        print("[NOT WIRED] the staged companion stream is not spliced into the .edp yet, so "
              "these widths are a plan, not an artefact: do not read S1 output as 12-digit "
              "truth until TRUTH_EMISSION_WIRED is True and a companion column count has been"
              " checked on a rendered file.")

    res = run_case(case, out_root, levels, execute, sigma=args.blend_sigma,
                   coord_digits=args.coord_precision,
                   truth_plans=staged if TRUTH_EMISSION_WIRED else None)
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
