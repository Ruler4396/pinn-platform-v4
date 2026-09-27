#!/usr/bin/env python3
"""Second implementation of the C-base Stokes reference, in FEniCSx (登记七 / 登记九).

What is being reproduced is `model/cases/contraction_2d/cfd/C-base/C-base_stokes.edp`, and every
constant below is taken from that file with its line number, so a reviewer can put the two side by
side.  A different *mesh generator* is intended; a different *problem* is not.

  .edp :5-9    BETA 0.7, Lin 4, Lc 4, Lout 8, Ltot 16
  .edp :15-27  width(x) = 1 -> smoothstep5 contraction -> beta, walls at +-width/2
  .edp :11-13  labels inlet=1, outlet=2, wall=3
  .edp :53     buildmesh(bottomWall(180) + outletEdge(28) + topWall(180) + inletEdge(40))
  .edp :55-58  fespace Vh P2 (velocity), Qh P1 (pressure)  -- Taylor-Hood
  .edp :60-63  uIn(y) = 1.5(1 - (2y)^2), inlet average velocity = 1
  .edp :66-71  grad u : grad ut  - p div(ut) - qt div(u) - 1e-10 p qt   (SIGNS ARE THE REFERENCE'S)
  .edp :73-75  on(inlet, u=uIn(y), v=0); on(wall, u=0, v=0); on(outlet, p=0)
  .edp :84-85  vertex tag precedence: inlet/outlet override a wall tag

The sign of the pressure coupling is not cosmetic: flipping it negates p, hence dp and R, and the
cross-check would report ~200% disagreement between two correct solvers.  So the form above is
transcribed, not re-derived.

This file does NOT judge anything.  The three judged integrals (dp, Q, R) are computed once, in
crosscheck_second_impl.py, and applied to both sides -- pre-registration section 1.  Writing a
second definition of Q here would be the definitional drift that section sends you to avoid.

Venue: instance only, through ops/run_second_impl.sh (登记九: nothing is installed on the user's
machine).  Written against dolfinx 0.9, the version install_external_solver.sh pins; the call
shapes were read off upstream `python/demo/demo_stokes.py` at ref v0.9.0 (mixed_direct()), not from
memory.  `--selfcheck` runs with stdlib only -- it never imports dolfinx, so the geometry, the
tagging rule and the CSV contract are falsifiable before any machine time is spent.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):            # Windows consoles are often GBK, not UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# --- constants transcribed from C-base_stokes.edp, with the line they come from -----------------
BETA = 0.7        # :5   real beta
LIN = 4.0         # :6   real Lin
LC = 4.0          # :7   real Lc
LTOT = 16.0       # :9   real Ltot
# One tolerance for every "is this point on that boundary" test in this file: the BC predicates below
# use np.isclose (atol 1e-8) and vertex_tag uses this.  A coordinate that arrived as 1.91486939696e-18
# was tagged interior when the two disagreed, and the export then had no inlet row for the consumer to
# average -- which is the failure measured on the box at 02:28:10 (7,417 zeros, 4 walls, 0 inlet).
EDGE_TOL = 1e-9
TAG_IN, TAG_OUT, TAG_WALL = "1", "2", "3"      # :11-13
PRESSURE_PIN = 1.0e-10                          # :71  the - 1.0e-10*p*qt term
N_X, N_Y = 180, 40                              # :53  border segment counts (bottom/top 180, inlet 40)
HEADER = ["x_star", "y_star", "u_star", "v_star", "p_star", "bc_tag"]   # :89


def smoothstep5(s: float) -> float:
    """:15-17.  s=0 -> 0, s=1 -> 1, and both endpoints are flat (6s^5-15s^4+10s^3)."""
    return 6.0 * s ** 5 - 15.0 * s ** 4 + 10.0 * s ** 3


def channel_width(x: float) -> float:
    """:19-24.  Half-heights are +-channel_width(x)/2 (:26-27)."""
    if x <= LIN:
        return 1.0
    if x >= LIN + LC:
        return BETA
    return 1.0 - (1.0 - BETA) * smoothstep5((x - LIN) / LC)


def y_top(x: float) -> float:
    return 0.5 * channel_width(x)


def y_bot(x: float) -> float:
    return -0.5 * channel_width(x)


def u_inlet(y: float) -> float:
    """:60-63.  At the inlet the width is 1, so eta = 2y maps +-0.5 onto +-1."""
    eta = 2.0 * y
    return 1.5 * (1.0 - eta * eta)


def map_to_channel(xi: float, eta: float) -> tuple[float, float]:
    """(xi, eta) in [0, LTOT] x [0, 1] -> the physical point of the warped rectangle.

    Declared ONCE, used both by grid_vertices (which the selfcheck can and does test) and by the
    geometry warp inside solve() (which it cannot run).  Before this existed the two arithmetic
    copies could disagree, and a mutation of the solver's offset passed the selfcheck green.
    """
    return xi, (eta - 0.5) * channel_width(xi)


# Signs of the three pressure-bearing terms, as transcription rather than as characters in code the
# selfcheck cannot execute (.edp :69-71).  Flipping either of the first two negates p, hence dp and
# R, and two correct solvers would then be reported as disagreeing by ~200%.
SIGN_PRESSURE_ON_DIV_TEST = -1.0    # :69  - p*(dx(ut) + dy(vt))
SIGN_DIV_TRIAL_ON_P_TEST = -1.0     # :70  - qt*(dx(u) + dy(v))
SIGN_PRESSURE_PIN = -1.0            # :71  - 1.0e-10*p*qt


# --- the mesh, as pure arithmetic: this is the half that can be falsified on the laptop ----------

def grid_vertices(nx: int = N_X, ny: int = N_Y) -> list[tuple[float, float]]:
    """Warped rectangle: x = LTOT*i/nx, y = (j/ny - 1/2)*channel_width(x).

    The vertical coordinate is a scaled version of the width, so every wall vertex lands exactly on
    +-width/2 -- the same piecewise-linear wall that FreeFEM's border parametrisation produces.
    x is left untouched, which is what makes the outlet plane an exact x == LTOT section.
    """
    pts: list[tuple[float, float]] = []
    for i in range(nx + 1):
        x = LTOT * i / nx
        for j in range(ny + 1):
            pts.append(map_to_channel(x, j / ny))
    return pts


def vertex_tag(x: float, y: float) -> str:
    """Reproduces the vTag rule of .edp:84-85 geometrically: inlet/outlet beat wall, interior is 0.

    Only the inlet and outlet sets feed a judged quantity (the two pressure means); wall and
    interior tags are exported for readers and decide nothing.

    Measured on the box at 02:28:10, in the base-level export: the first two rows arrived with
    x = -1.91486939696e-18 and +1.91486939696e-18, not 0.0.  Comparing exactly therefore tagged
    nothing -- 7,417 zeros and 4 walls, no inlet and no outlet -- and the consumer correctly refused
    to average an empty set.  The BC predicates in solve() use np.isclose and found their facets
    (inlet=40 outlet=40 wall=356), so the two tests disagreed: one name for the tolerance now, and
    the export checks its own tag coverage before writing.
    """
    if abs(x) <= EDGE_TOL:
        return TAG_IN
    if abs(x - LTOT) <= EDGE_TOL:
        return TAG_OUT
    if abs(abs(y) - y_top(x)) <= EDGE_TOL:
        return TAG_WALL
    return "0"


def write_rows(path: Path, rows: list[tuple[float, float, float, float, float]]) -> int:
    """One row per node, same header as the reference export.  12 digits: this side is the newer
    implementation, so it must not be the one that loses precision (the K0 lesson runs the other
    way -- a 6-digit reference cannot separate bands).  The single place that formats a node."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(HEADER)
        for x, y, uu, vv, pp in rows:
            w.writerow([f"{x:.12g}", f"{y:.12g}", f"{uu:.12g}", f"{vv:.12g}", f"{pp:.12g}",
                        vertex_tag(x, y)])
    return len(rows)


def write_nodes(path: Path, pts: list[tuple[float, float]], u: list[float], v: list[float],
                p: list[float]) -> int:
    return write_rows(path, [(x, y, uu, vv, pp)
                             for (x, y), uu, vv, pp in zip(pts, u, v, p)])


def nodes_from_vertex_arrays(pts, u, v, p):
    """(x, y, u, v, p) rows from arrays that all index the SAME vertex list, sorted by (x, y).

    Measured on the box at 01:59:58, in the env this file is meant to run in: for a vector P1 space
    `tabulate_dof_coordinates()` returns one row PER VERTEX (9 rows on a 2x2 mesh) while `x.array`
    holds gdim values per vertex (18) -- so the coordinate list of the vector space cannot be paired
    with its own flat array by position, and grouping by coordinate (the first fix attempt) died with
    `vertex (8.0, -0.35) carries 1 velocity dofs`.  The component-wise interpolate that replaced it
    (`Function(scalar).interpolate(vector.sub(0))`, measured OK) puts every quantity in the scalar
    space, so there is exactly one vertex list and one index per row.  The two checks below are what
    would have caught the wrong pairing at the source: a length mismatch, or the same vertex twice.
    """
    n = len(pts)
    if not (len(u) == len(v) == len(p) == n):
        raise ValueError(f"length mismatch: {n} vertices but arrays are "
                         f"{len(u)}/{len(v)}/{len(p)} -- one of them is indexed by dof, not by vertex")
    seen = set()
    for k, (x, y) in enumerate(pts):
        key = (round(float(x), 9), round(float(y), 9))
        if key in seen:
            raise ValueError(f"vertex {key} appears twice in the export list -- "
                             f"the coordinate source does not match the arrays")
        seen.add(key)
    rows = [(float(pts[k][0]), float(pts[k][1]), float(u[k]), float(v[k]), float(p[k]))
            for k in range(n)]
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


# --- the solver: dolfinx lives only inside here, so importing this module needs no install ------

def solve(nx: int, ny: int, out_csv: Path) -> dict:
    from mpi4py import MPI                       # noqa: F401  (imported first, as upstream does)
    import numpy as np
    import ufl
    from basix.ufl import element, mixed_element
    from petsc4py import PETSc

    from dolfinx import default_real_type, fem
    from dolfinx.fem import (Function, dirichletbc, form, functionspace, locate_dofs_topological)
    from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector
    from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary

    print(f"[solve] mesh {nx}x{ny} triangles -> P2-P1 Taylor-Hood, Ltot={LTOT} beta={BETA}",
          flush=True)
    msh = create_rectangle(MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([LTOT, 1.0])],
                           (nx, ny), CellType.triangle)
    geo = msh.geometry.x                                   # (n_points, 3); P1 geometry -> one per vertex
    for k in range(geo.shape[0]):                          # same map the selfcheck tests, not a copy of it
        geo[k, 1] = map_to_channel(float(geo[k, 0]), float(geo[k, 1]))[1]
    geo[:, 2] = 0.0

    gdim = msh.geometry.dim
    p2 = element("Lagrange", msh.basix_cell(), 2, shape=(gdim,), dtype=default_real_type)   # .edp :55
    p1 = element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type)                  # .edp :56
    W = functionspace(msh, mixed_element([p2, p1]))
    W0, W1 = W.sub(0), W.sub(1)
    Vv, _ = W0.collapse()
    Vp, _ = W1.collapse()

    # Boundary marking.  locate_entities_boundary only ever looks at EXTERIOR facets, and it
    # evaluates its marker at facet midpoints -- which is exactly why the wall is defined here as
    # "exterior, and not one of the two end planes" rather than by a curve test: on a
    # piecewise-linear wall the chord midpoint is not on the curve, so a curve test would mark
    # nothing and would silently drop the no-slip condition.
    at_in = lambda x: np.isclose(x[0], 0.0)                                    # noqa: E731
    at_out = lambda x: np.isclose(x[0], LTOT)                                  # noqa: E731
    at_wall = lambda x: ~(np.isclose(x[0], 0.0) | np.isclose(x[0], LTOT))       # noqa: E731
    f_in = locate_entities_boundary(msh, 1, at_in)
    f_out = locate_entities_boundary(msh, 1, at_out)
    f_wall = locate_entities_boundary(msh, 1, at_wall)
    # Why this line exists: the base level measured an outlet flux of 0.531 against an inlet flux of
    # 0.999 (probe at 02:37:51), and the pressure rows make the NET boundary flux exactly zero, so
    # ~0.47 of the flow left through boundary facets no BC touched.  Constant q in P1 turns
    # "div u = 0 weakly" into global conservation, so a deficit is a marking hole, not a discretisation
    # difference.  Count it instead of reasoning about it: every exterior facet must be in exactly one
    # of the three sets.
    all_ext = np.asarray(msh.exterior_facets.indices)
    covered = np.concatenate([f_in, f_out, f_wall])
    uncovered = np.setdiff1d(all_ext, covered)
    doubled = len(covered) - len(np.unique(covered))
    print(f"[solve] exterior facets: total={len(all_ext)} claimed={len(covered)} "
          f"uncovered={len(uncovered)} in-two-sets={doubled}", flush=True)
    if len(uncovered) or doubled:
        raise RuntimeError(f"boundary marking is not a partition: {len(uncovered)} exterior facets "
                           f"carry no condition and {doubled} are claimed twice -- through those the "
                           f"solve leaks, and its three integrals would be reported as a second "
                           f"implementation's answer")
    print(f"[solve] exterior facets: inlet={len(f_in)} outlet={len(f_out)} wall={len(f_wall)} "
          f"(inlet+outlet+wall must cover the boundary)", flush=True)

    noslip = Function(Vv)                                                      # zero by construction
    inlet = Function(Vv)
    inlet.interpolate(lambda x: np.stack((np.array([u_inlet(float(yy)) for yy in x[1]]),
                                          np.zeros(x.shape[1], dtype=default_real_type))))
    bc = [
        dirichletbc(inlet, locate_dofs_topological((W0, Vv), 1, f_in), W0),     # .edp :73
        dirichletbc(noslip, locate_dofs_topological((W0, Vv), 1, f_wall), W0),  # .edp :74
        dirichletbc(Function(Vp), locate_dofs_topological((W1, Vp), 1, f_out), W1),  # .edp :75  p=0
    ]

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    # transcribed from .edp :66-72, signs included; dx(u)*dx(ut)+dy(u)*dy(ut) == inner(grad u, grad v)
    a = form((ufl.inner(ufl.grad(u), ufl.grad(v))
              + SIGN_PRESSURE_ON_DIV_TEST * p * ufl.div(v)
              + SIGN_DIV_TRIAL_ON_P_TEST * ufl.div(u) * q
              + SIGN_PRESSURE_PIN * PRESSURE_PIN * p * q) * ufl.dx)
    # All driving here is Dirichlet, so the load is zero.  The linear form carries only the velocity
    # test function and the pressure row stays implicitly zero -- the same shape upstream
    # demo_stokes.mixed_direct() uses (f = Function(Q); L = form(inner(f, v)*dx)), rather than a
    # ZeroForm whose constructor is the part of UFL that has actually moved between releases.
    zero_f = Function(Vv)
    L = form(ufl.inner(zero_f, v) * ufl.dx)

    A = assemble_matrix(a, bcs=bc)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], bcs=[bc])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    for one in bc:
        one.set(b.array_w)

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    # Which package does the factorization is a linear-algebra choice, not part of the registered form,
    # and it is not free here: measured in this env between 02:10 and 02:17 the default SuperLU path
    # converges at 2x2 / 4x4 / 10x10 / 20x10 / 45x20 / 90x40 / 120x40 / 150x40 / 180x30 and returns
    # reason=-11 with its=0 at the registered 180x40.  MUMPS is present (hasExternalPackage('mumps')
    # -> True, measured 02:21:27) and is a direct sparse solver like the UMFPACK the reference asks
    # for, so the driver asks for it BY NAME and the name lands in the meta file beside the numbers.
    pkg = os.environ.get("S1_MAT_SOLVER", "").strip()
    if pkg:
        pc.setFactorSolverType(pkg)
    print(f"[solve] assembled, solving with LU (the reference asks UMFPACK: a direct solve either way); "
          f"factorization package = {pkg or 'PETSc default (SuperLU)'}", flush=True)
    Usol = Function(W)
    ksp.solve(b, Usol.x.petsc_vec)
    reason, its = ksp.getConvergedReason(), ksp.getIterationNumber()
    if reason <= 0:
        try:
            rname = PETSc.KSP.ConvergedReasons(reason).name
        except Exception:
            rname = "name-lookup-failed"
        raise RuntimeError(f"KSP did not converge (reason={reason}/{rname}, its={its}) -- "
                           f"reporting an unconverged solve as a second implementation would be a lie")
    print(f"[solve] converged reason={reason} its={its}", flush=True)

    # nodal export at the mesh vertices, the same sampling the reference CSV uses (.edp :90-93)
    q1 = functionspace(msh, element("Lagrange", msh.basix_cell(), 1, shape=(gdim,),
                                    dtype=default_real_type))
    s1 = functionspace(msh, element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type))
    u1, p1f = Function(q1), Function(s1)
    u1.interpolate(Usol.sub(0).collapse())
    p1f.interpolate(Usol.sub(1).collapse())
    uc, vc = Function(s1), Function(s1)
    uc.interpolate(u1.sub(0))
    vc.interpolate(u1.sub(1))
    pts = [(float(c[0]), float(c[1])) for c in s1.tabulate_dof_coordinates()]
    rows = nodes_from_vertex_arrays(pts, uc.x.array, vc.x.array, p1f.x.array)
    # The two pressure means are the judged quantities, so an export without an inlet row or without an
    # outlet row is a marking failure, not a result.  Caught exactly this way at 02:26:37: the consumer
    # refused with "empty selection", because vertex_tag had compared x exactly.
    tags = [vertex_tag(r[0], r[1]) for r in rows]
    n_in, n_out = tags.count(TAG_IN), tags.count(TAG_OUT)
    if n_in == 0 or n_out == 0:
        raise RuntimeError(f"export tagged inlet={n_in} outlet={n_out} of {len(rows)} nodes -- "
                           f"the boundary marking and the BC facets disagree; refusing to write a "
                           f"table whose two pressure means are averages over an empty set")
    print(f"[solve] tag coverage: inlet={n_in} outlet={n_out} wall={tags.count(TAG_WALL)} "
          f"interior={tags.count('0')}", flush=True)
    n = write_rows(out_csv, rows)
    import dolfinx
    return {"nx": nx, "ny": ny, "nodes": n, "ksp_reason": int(reason), "ksp_its": int(its),
            "mat_solver": pkg or "PETSc default (SuperLU)",
            "dolfinx": dolfinx.__version__, "geometry": {"BETA": BETA, "LIN": LIN, "LC": LC,
                                                         "LTOT": LTOT, "pin": PRESSURE_PIN}}


# --- selfcheck: stdlib only, and it feeds the CONSUMER, not just this file -----------------------

def _plug_flow_csv(path: Path, nx: int = 16, ny: int = 8) -> None:
    """A CSV whose three integrals are computable by hand: u == 1 gives Q = channel_width(LTOT),
    p == LTOT - x gives dp = LTOT (inlet high, outlet zero, the physical direction the reference
    solves in).  The trapezoid of a constant over a section is exact for any spacing, so this
    checks the consumer's arithmetic against this file's geometry, not just its parsing."""
    pts = grid_vertices(nx, ny)
    return write_nodes(path, pts, [1.0] * len(pts), [0.0] * len(pts), [LTOT - x for x, _ in pts])


def selfcheck() -> int:
    import tempfile
    rc = 0
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import crosscheck_second_impl as xchk        # the consumer, in-repo, stdlib-only

    def ck(name: str, ok: bool, value: object = "") -> None:
        nonlocal rc
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {value}")
        if not ok:
            rc = 1

    # geometry anchors, each one straight off the reference
    ck("smoothstep5 endpoints and midpoint (6s^5-15s^4+10s^3)",
       smoothstep5(0.0) == 0.0 and smoothstep5(1.0) == 1.0 and abs(smoothstep5(0.5) - 0.5) < 1e-15,
       f"f(0.5)={smoothstep5(0.5):.6g}")
    ck("width: 1 upstream, 0.7 downstream, 0.85 at the contraction midpoint",
       channel_width(0.0) == 1.0 and channel_width(LTOT) == BETA
       and abs(channel_width(LIN + LC / 2) - 0.85) < 1e-12,
       f"w(0)={channel_width(0.0)} w(16)={channel_width(LTOT)} w(6)={channel_width(6.0):.6g}")
    ck("walls antisymmetric about y=0", all(abs(y_top(x) + y_bot(x)) < 1e-15 for x in (0.0, 6.0, 16.0)))
    ck("inlet profile: 1.5 on the axis, 0 at both walls (.edp :60-63)",
       abs(u_inlet(0.0) - 1.5) < 1e-15 and u_inlet(0.5) == 0.0 and u_inlet(-0.5) == 0.0)
    ys = [-0.5 + i / 4000 for i in range(4001)]
    mean_u = sum(0.5 * (u_inlet(a) + u_inlet(b)) * (1 / 4000) for a, b in zip(ys, ys[1:]))
    ck("inlet AVERAGE velocity = 1 (the normalisation .edp:3 states; a mis-scaled inlet would "
       "redden the cross-check for the wrong reason)", abs(mean_u - 1.0) < 1e-6, f"trapz mean={mean_u:.9g}")

    # The form's signs are a transcription of .edp:69-71, and the solver cannot be run here to test
    # them, so the transcription lives in named constants and is asserted on its own.
    ck("pressure coupling signs as transcribed from .edp:69-71 (both minus, and the pin minus)",
       SIGN_PRESSURE_ON_DIV_TEST == -1.0 and SIGN_DIV_TRIAL_ON_P_TEST == -1.0
       and SIGN_PRESSURE_PIN == -1.0 and PRESSURE_PIN == 1.0e-10,
       f"{SIGN_PRESSURE_ON_DIV_TEST} {SIGN_DIV_TRIAL_ON_P_TEST} {SIGN_PRESSURE_PIN} pin={PRESSURE_PIN}")

    nx, ny = 18, 8
    pts = grid_vertices(nx, ny)
    ck("vertex count (nx+1)(ny+1)", len(pts) == (nx + 1) * (ny + 1), f"{len(pts)}")
    inlet_pts = [pt for pt in pts if pt[0] == 0.0]
    outlet_pts = [pt for pt in pts if pt[0] == LTOT]
    ck("inlet column spans the full +-0.5 width, outlet column +-0.35",
       len(inlet_pts) == ny + 1 and len(outlet_pts) == ny + 1
       and abs(max(y for _, y in inlet_pts) - 0.5) < 1e-15
       and abs(max(y for _, y in outlet_pts) - 0.35) < 1e-15,
       f"inlet y=[{min(y for _, y in inlet_pts):.6g},{max(y for _, y in inlet_pts):.6g}] "
       f"outlet y=[{min(y for _, y in outlet_pts):.6g},{max(y for _, y in outlet_pts):.6g}]")
    wall_tagged = [(x, y) for x, y in pts if vertex_tag(x, y) == TAG_WALL]
    ck("wall vertices are exactly the top and bottom rows minus the two end columns",
       len(wall_tagged) == 2 * (nx - 1)
       and all(abs(abs(y) - y_top(x)) < 1e-15 for x, y in wall_tagged),
       f"{len(wall_tagged)} tagged, expected {2 * (nx - 1)}")
    interior = [(x, y) for x, y in pts if vertex_tag(x, y) == "0"]
    ck("interior vertices get tag 0 and are excluded from both pressure means",
       len(interior) == (nx - 1) * (ny - 1), f"{len(interior)} nodes, expected {(nx - 1) * (ny - 1)}")
    corners = [(vertex_tag(0.0, 0.5), vertex_tag(0.0, -0.5)), (vertex_tag(LTOT, 0.35), vertex_tag(LTOT, -0.35))]
    ck("corner tags follow the .edp:84-85 precedence (inlet/outlet override wall)",
       corners == [(TAG_IN, TAG_IN), (TAG_OUT, TAG_OUT)], f"{corners}")
    # The exact reading that made the base-level export unusable (02:28:10, first two rows of
    # second_impl_nodes.csv): x is not 0.0 there, it is -1.91486939696e-18.
    ck("the measured 1.9e-18 round-off at the inlet is still tagged inlet, and a slightly over "
       "outlet is still tagged outlet (this is what made the two pressure means an empty average)",
       vertex_tag(-1.91486939696e-18, 0.5) == TAG_IN and vertex_tag(1.91486939696e-18, -0.5) == TAG_IN
       and vertex_tag(LTOT + 1.5e-15, 0.35) == TAG_OUT
       and vertex_tag(8.0, y_top(8.0)) == TAG_WALL,
       f"x=-1.9e-18->{vertex_tag(-1.91486939696e-18, 0.5)} x=16+1.5e-15->{vertex_tag(LTOT + 1.5e-15, 0.35)}")
    root = Path(tempfile.mkdtemp(prefix="second_impl_selfcheck_"))
    good = root / "second_impl_nodes.csv"
    n = _plug_flow_csv(good)
    cols = {k: h for k, h in zip(("x", "y", "u", "v", "p", "tag"), HEADER)}
    nodes = xchk.read_nodes(good, cols)
    tri = xchk.integrals(nodes, TAG_IN, TAG_OUT)
    ck("the consumer parses this file's header and tags, and its Q equals the outlet width",
       n == len(nodes) and abs(tri["Q"] - BETA) < 1e-12 and abs(tri["dp"] - LTOT) < 1e-12,
       f"dp={tri['dp']:.12g} (want {LTOT}) Q={tri['Q']:.12g} (want {BETA}) R={tri['R']:.12g}")

    bad = root / "no_tag_col.csv"
    bad.write_text("\n".join([",".join(HEADER[:5])] + [",".join(f"{r:.12g}" for r in row[:5])
                                                        for row in [[0.0, 0.1, 1.0, 0.0, 1.0]]]) + "\n",
                   encoding="utf-8")
    try:
        xchk.read_nodes(bad, cols)
        ck("MUST-RED: a CSV without the bc_tag column is refused by the consumer", False, "it parsed")
    except ValueError as exc:
        ck("MUST-RED: a CSV without the bc_tag column is refused by the consumer", True,
           f"refused: {str(exc)[:60]}")

    one = root / "single_outlet_node.csv"
    pts_one = [(0.0, 0.0), (LTOT, 0.3), (LTOT - 1.0, 0.0)]
    write_nodes(one, pts_one, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [2.0, 1.0, 2.0])
    try:
        xchk.integrals(xchk.read_nodes(one, cols), TAG_IN, TAG_OUT)
        ck("MUST-RED: one node on the outlet plane cannot pass for a flux integral", False, "it computed")
    except ValueError as exc:
        ck("MUST-RED: one node on the outlet plane cannot pass for a flux integral", True,
           f"refused: {str(exc)[:60]}")

    # The export trap the smoke leg found on the real box, now asserted instead of assumed: the arrays
    # must be indexed by vertex, and the two checks that catch a dof-indexed array are length and
    # duplication.  A 2x2 mesh is 9 vertices and 18 velocity values -- pairing those by position is
    # exactly what wrote rows belonging to different vertices.
    verts = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    want_rows = sorted((x, y, 100.0 + k, 200.0 + k, 300.0 + k) for k, (x, y) in enumerate(verts))
    got = nodes_from_vertex_arrays(verts, [100.0 + k for k in range(4)],
                                   [200.0 + k for k in range(4)], [300.0 + k for k in range(4)])
    ck("nodes_from_vertex_arrays pairs one vertex with one value per quantity and sorts the table",
       got == want_rows, f"{len(got)} rows, want {len(want_rows)}")
    try:
        nodes_from_vertex_arrays(verts, [1.0] * 8, [1.0] * 4, [1.0] * 4)
        ck("MUST-RED: a dof-indexed velocity array (2 per vertex) is refused, not written", False, "it wrote")
    except ValueError as exc:
        ck("MUST-RED: a dof-indexed velocity array (2 per vertex) is refused, not written", True,
           f"refused: {str(exc)[:66]}")
    try:
        nodes_from_vertex_arrays(verts + [(0.0, 0.0)], [1.0] * 5, [1.0] * 5, [1.0] * 5)
        ck("MUST-RED: the same vertex listed twice is refused, not written", False, "it wrote")
    except ValueError as exc:
        ck("MUST-RED: the same vertex listed twice is refused, not written", True,
           f"refused: {str(exc)[:66]}")

    # the reference itself, read through the same code path: proves this file's tag convention
    # reproduces what the shipped .edp exported (inlet/outlet columns present and non-empty)
    ref = Path(__file__).resolve().parents[2] / "cases/contraction_2d/cfd/C-base/C-base_raw.csv"
    if ref.is_file():
        rn = xchk.read_nodes(ref, cols)
        ri = xchk.integrals(rn, TAG_IN, TAG_OUT)
        ck("reference CSV read through the consumer gives finite dp/Q/R (baseline side of the trip)",
           all(abs(ri[k]) > 0 for k in ("dp", "Q", "R")), f"{ri['dp']:.9g} {ri['Q']:.9g} {ri['R']:.9g}")
    else:
        print(f"[SKIP] reference CSV not in this tree: {ref}")

    print(f"json-free selfcheck rc={rc} tmp={root}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="FEniCSx second implementation of the C-base Stokes case")
    ap.add_argument("--nx", type=int, default=N_X, help="x cells; the .edp uses 180 (:53)")
    ap.add_argument("--ny", type=int, default=N_Y, help="y cells; the .edp inlet edge uses 40 (:53)")
    ap.add_argument("--out", type=Path,
                    help="instance-side directory (route2_out/<ts>/), never a repo write face; "
                         "not needed for --selfcheck, which writes to a temp dir")
    ap.add_argument("--selfcheck", action="store_true", help="stdlib-only controls, no dolfinx import")
    args = ap.parse_args()

    if args.selfcheck:
        return selfcheck()
    if args.out is None:
        print("[INDETERMINATE] need --out; nothing was solved (same convention as "
              "crosscheck_second_impl.py --selfcheck, which needs no inputs)")
        return 2
    out_csv = args.out / "second_impl_nodes.csv"
    if "pinn-platform-v4" in str(out_csv.resolve()):
        raise SystemExit(f"refusing to write the solve's output inside the repo: {out_csv}")
    import json
    meta = solve(args.nx, args.ny, out_csv)
    meta["csv"] = str(out_csv)
    meta["ref_edp"] = "model/cases/contraction_2d/cfd/C-base/C-base_stokes.edp"
    (args.out / "second_impl_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"[solve] wrote {out_csv} ({meta['nodes']} nodes) and "
          f"{args.out / 'second_impl_meta.json'} dolfinx={meta['dolfinx']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
