"""Symmetric T-bifurcation geometry in dimensionless star units (route 2, K0/S1).

Pure stdlib on purpose: the geometry, its analytic feature Jacobian and the
frame-blend weights must be checkable without numpy/torch/pandas.

Star units: stem width W* = 1 (half-width 0.5), inlet mean velocity = 1,
dynamic viscosity = 1, so steady Stokes reads  lap(u) = grad(p),  div(u) = 0
with both sides already of the same order -- the corrected residual therefore
carries no fitted per-side scale (see paper-route2/K0-S1设计与预注册-20260925.md).

Domain = union of three straight rectangles (stem + two branches at +/-theta);
its polygon boundary is solved analytically so mesh borders, membership tests,
wall distances and structured FD grids all share one source of truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
# membership tolerances, star units (W_stem = 1).  One number, used by every predicate.
ABSORB_TOL = 1.0e-6        # floor: what an exact-precision file needs; never the working band
MAX_REJECT_FRAC = 5.0e-3   # above this share of rejected vertices the level halts
FREEFEM_PRINT_DIGITS = 6   # measured: FreeFEM's ofstream writes 6 significant digits
# The only place verdict names are declared.  Anything that counts them must derive its
# keys from here -- defect 5 was a counter that kept a stale key and died on real meshes.
MEMBERSHIP_VERDICTS: Tuple[str, ...] = ("frame", "absorbed_print",
                                        "polygon_without_frame", "outside")
STEM, UP, DOWN = 0, 1, 2
FRAME_NAMES = {STEM: "stem", UP: "branch_up", DOWN: "branch_down"}


@dataclass(frozen=True)
class TCase:
    case_id: str = "TB-base"
    l_stem: float = 4.0
    theta_deg: float = 45.0
    l_branch: float = 4.0
    w_stem: float = 1.0
    w_branch_up: float = 1.0
    w_branch_down: float = 1.0
    family: str = "tbif_2d"
    note: str = "symmetric T: carries K0 self-scoring, mesh independence, conservation"

    @property
    def is_geometrically_symmetric(self) -> bool:
        return abs(self.w_branch_up - self.w_branch_down) < 1.0e-12

    def to_metadata(self) -> dict:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "l_stem_over_W": self.l_stem,
            "theta_deg": self.theta_deg,
            "l_branch_over_W": self.l_branch,
            "w_stem_over_W": self.w_stem,
            "w_branch_up_over_W": self.w_branch_up,
            "w_branch_down_over_W": self.w_branch_down,
            "geometrically_symmetric": self.is_geometrically_symmetric,
            "units": "star: W_stem=1, U_inlet_mean=1, mu=1, steady Stokes",
            "role": ("kill-test geometry (adversary table)"
                     if not self.is_geometrically_symmetric else "self-check geometry"),
            "note": self.note,
        }


ASYM_CASE = TCase(
    case_id="TB-asym",
    w_branch_up=0.85,
    note=("adversary-table geometry per ruling R2-1 (统括官, 2026-09-25): the upper branch is "
          "0.85W so the flow split is no longer pinned to 0.5 by symmetry and becomes a "
          "measurable, invertible observable"),
)


@dataclass(frozen=True)
class Frame:
    """Orthonormal branch-local frame: point = origin + xi * d + eta * m."""

    key: int
    name: str
    origin: Point
    d: Point
    m: Point
    half_width: float
    length: float
    kind: str  # inlet / outlet / wall handled by the polygon

    def local(self, x: float, y: float) -> Point:
        dx, dy = x - self.origin[0], y - self.origin[1]
        return (dx * self.d[0] + dy * self.d[1], dx * self.m[0] + dy * self.m[1])

    def global_xy(self, xi: float, eta: float) -> Point:
        return (
            self.origin[0] + xi * self.d[0] + eta * self.m[0],
            self.origin[1] + xi * self.d[1] + eta * self.m[1],
        )

    def inside(self, x: float, y: float, margin: float = 0.0, tol: float = 1.0e-9) -> bool:
        """Membership with tolerance: FreeFEM puts mesh vertices exactly on the wall."""
        xi, eta = self.local(x, y)
        return (margin - tol <= xi <= self.length - margin + tol
                and abs(eta) <= self.half_width - margin + tol)


@dataclass
class Polygon:
    verts: List[Point] = field(default_factory=list)
    edge_labels: List[str] = field(default_factory=list)  # inlet / outlet_up / outlet_down / wall
    edge_ends: List[Tuple[Point, Point]] = field(default_factory=list)

    def area(self) -> float:
        s = 0.0
        n = len(self.verts)
        for i in range(n):
            x0, y0 = self.verts[i]
            x1, y1 = self.verts[(i + 1) % n]
            s += x0 * y1 - x1 * y0
        return 0.5 * s

    def is_ccw(self) -> bool:
        return self.area() > 0.0


def _seg_intersect(p: Point, r: Point, q: Point, s: Point) -> Point:
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1.0e-14:
        raise ValueError("parallel lines in corner construction")
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / cross
    return (p[0] + t * r[0], p[1] + t * r[1])


class TGeometry:
    """Analytic geometry of a T with one stem and two (possibly unequal) branches."""

    def __init__(self, case: TCase = TCase()):
        self.case = case
        th = math.radians(case.theta_deg)
        self.cos_t, self.sin_t = math.cos(th), math.sin(th)
        self.h_stem = 0.5 * case.w_stem
        self.h_branch = {"up": 0.5 * case.w_branch_up, "down": 0.5 * case.w_branch_down}
        self.j_point: Point = (case.l_stem, 0.0)

        frames: Dict[int, Frame] = {}
        frames[STEM] = Frame(STEM, FRAME_NAMES[STEM], (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                             self.h_stem, case.l_stem, "stem")
        for key, sign, hb in ((UP, +1.0, self.h_branch["up"]), (DOWN, -1.0, self.h_branch["down"])):
            d = (self.cos_t, sign * self.sin_t)
            m = (-self.sin_t, sign * self.cos_t)  # away from the stem centreline; y-reflected only
            frames[key] = Frame(key, FRAME_NAMES[key], self.j_point, d, m,
                                hb, case.l_branch, "branch")
        self.frames = frames

        # Outward (away-from-axis) wall meeting point A_sigma on the stem wall line.
        s, c = self.sin_t, self.cos_t
        if s <= 1.0e-12:
            raise ValueError("theta too small to form a T")
        self.a_outer: Dict[int, Point] = {}
        self.corners: Dict[int, Tuple[Point, Point]] = {}
        for key in (UP, DOWN):
            fr = self.frames[key]
            u = (self.h_stem - fr.half_width * c) / s
            if u < -1.0e-9:
                raise ValueError(f"{fr.name}: h_branch*cos(theta) > h_stem leaves the branch "
                                 "outer wall meeting the stem wall line upstream of the inlet")
            if u > case.l_branch:
                raise ValueError(f"{fr.name}: wall meeting point lies beyond the branch outlet")
            self.a_outer[key] = fr.global_xy(u, fr.half_width)
            self.corners[key] = (fr.global_xy(case.l_branch, fr.half_width),
                                 fr.global_xy(case.l_branch, -fr.half_width))
        # crotch = where the two inner walls meet (off-axis once the branches differ)
        up, dn = self.frames[UP], self.frames[DOWN]
        p = up.global_xy(0.0, -up.half_width)
        q = dn.global_xy(0.0, -dn.half_width)
        self.crotch = _seg_intersect(p, up.d, q, dn.d)
        t_up = (self.crotch[0] - p[0]) * up.d[0] + (self.crotch[1] - p[1]) * up.d[1]
        t_dn = (self.crotch[0] - q[0]) * dn.d[0] + (self.crotch[1] - q[1]) * dn.d[1]
        if not (1.0e-9 < t_up <= up.length and 1.0e-9 < t_dn <= dn.length):
            raise ValueError("inner walls do not meet inside both branches: not a T junction")
        if self.crotch[0] <= case.l_stem:
            raise ValueError("crotch sits upstream of the junction plane")
        self.polygon = self._build_polygon()
        self._perimeter = sum(
            math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in self.polygon.edge_ends
        )

    # ---------------------------------------------------------------- polygon
    def _build_polygon(self) -> Polygon:
        case, hs = self.case, self.h_stem
        inlet_lo: Point = (0.0, -hs)
        inlet_hi: Point = (0.0, hs)
        # counter-clockwise (interior on the left) walk of the union boundary
        walk = [
            (inlet_hi, self.a_outer[UP], "wall"),          # stem top wall, running downstream
            (self.a_outer[UP], self.corners[UP][0], "wall"),   # upper branch outer wall
            (self.corners[UP][0], self.corners[UP][1], "outlet_up"),
            (self.corners[UP][1], self.crotch, "wall"),        # upper branch inner wall
            (self.crotch, self.corners[DOWN][1], "wall"),      # lower branch inner wall
            (self.corners[DOWN][1], self.corners[DOWN][0], "outlet_down"),
            (self.corners[DOWN][0], self.a_outer[DOWN], "wall"),
            (self.a_outer[DOWN], inlet_lo, "wall"),         # stem bottom wall
            (inlet_lo, inlet_hi, "inlet"),
        ]
        if _shoelace([e[0] for e in walk]) < 0.0:      # need interior on the left
            walk = [(b, a, lab) for a, b, lab in reversed(walk)]
        return Polygon(verts=[e[0] for e in walk],
                       edge_labels=[e[2] for e in walk],
                       edge_ends=[(e[0], e[1]) for e in walk])

    def rect_gap(self, x: float, y: float, fr: "Frame") -> float:
        """Distance from (x, y) to frame k's rectangle; 0.0 when it is inside it."""
        xi, eta = fr.local(x, y)
        g_xi = max(0.0, max(-xi, xi - fr.length))
        g_eta = max(0.0, abs(eta) - fr.half_width)
        return math.hypot(g_xi, g_eta)

    @staticmethod
    def half_ulp(value: float, digits: int = FREEFEM_PRINT_DIGITS) -> float:
        """Half the spacing of the printed decimals at this magnitude (0 if |v| < 10^(d-1))."""
        v = abs(value)
        if v == 0.0:
            return 0.0
        return 0.5 * 10.0 ** (math.floor(math.log10(v)) - (digits - 1))

    def representation_bound(self, x: float, y: float) -> float:
        """The largest gap that printing at FREEFEM_PRINT_DIGITS could possibly cause.

        2 x half-ulp because a wall's normal offset sums the rounding of both stored
        coordinates.  Anything whose gap exceeds this is NOT a printing artefact, which
        is what turns the reverse circuit-breaker below into a real test.
        """
        return max(ABSORB_TOL,
                   2.0 * max(self.half_ulp(x), self.half_ulp(y)))

    def membership(self, x: float, y: float,
                   absorb_tol: float = ABSORB_TOL) -> Tuple[str, Optional[int], float]:
        """The single source of truth for "is this point in the domain, and in which branch".

        Returns (verdict, frame_key, gap).  `contains`, `primary_frame` and the S1
        post-processor all go through here: the bug this method exists to kill was
        `contains()` accepting 1e-6 while frame membership accepted 1e-9, so a real
        FreeFEM vertex 1e-8 upstream of the inlet plane was inside the domain yet had no
        branch -- and died mid-postprocess.
        """
        bound = self.representation_bound(x, y)
        gaps = {k: self.rect_gap(x, y, fr) for k, fr in self.frames.items()}
        key = min(gaps, key=lambda k: (gaps[k],
                                       abs(self.frames[k].local(x, y)[1])
                                       / self.frames[k].half_width))
        if gaps[key] <= absorb_tol:          # inside the rect at exact geometry
            return ("frame", key, gaps[key])
        if gaps[key] <= bound:               # only the printed decimals can explain this
            return ("absorbed_print", key, gaps[key])
        if self._in_polygon(x, y, bound):
            # inside the contour yet farther than the printing bound from every rect:
            # not a representation artefact -> a second source, the caller must halt
            return ("polygon_without_frame", key, gaps[key])
        return ("outside", None, min(gaps.values()))

    def absorbed_within_bound(self, x: float, y: float, gap: float) -> bool:
        """Reverse breaker: an absorbed vertex may not exceed what printing can explain."""
        return gap <= self.representation_bound(x, y) * (1.0 + 1.0e-9)

    def contains(self, x: float, y: float, tol: float = ABSORB_TOL) -> bool:
        return self.membership(x, y, tol)[0] != "outside"

    def _in_polygon(self, x: float, y: float, tol: float) -> bool:
        verts = self.polygon.verts
        inside = False
        n = len(verts)
        for i in range(n):
            x0, y0 = verts[i]
            x1, y1 = verts[(i + 1) % n]
            if (y0 > y) != (y1 > y):
                x_int = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                if x < x_int - tol:
                    inside = not inside
        return inside

    # ------------------------------------------------------- wall distances
    def wall_distance_exact(self, x: float, y: float) -> float:
        """Distance to the no-slip boundary (min over wall segments of the polygon)."""
        best = math.inf
        for (a, b), label in zip(self.polygon.edge_ends, self.polygon.edge_labels):
            if label != "wall":
                continue
            best = min(best, _dist_point_seg(x, y, a, b))
        return 0.0 if best == math.inf else best

    def frames_at(self, x: float, y: float, tol: float = ABSORB_TOL) -> List[int]:
        return [k for k, f in self.frames.items() if f.inside(x, y, tol=tol)]

    # ------------------------------------------------- cross-section stations
    # The .edp's section walk and the flux gate must agree on where the stations are, so
    # the numbers are declared once, here, next to the test that decides which of them is
    # a material cross-section.
    STATION_STEP = 0.25
    BRANCH_STATION_START = 0.25

    def station_xis(self, key: int) -> List[float]:
        fr = self.frames[key]
        xi0 = 0.0 if fr.key == STEM else self.BRANCH_STATION_START
        n = int(round((fr.length - xi0) / self.STATION_STEP)) + 1
        return [xi0 + i * self.STATION_STEP for i in range(n)
                if xi0 + i * self.STATION_STEP <= fr.length + 1e-12]

    def section_is_material(self, key: int, xi: float, tol: float = 1.0e-9) -> bool:
        """Is the `xi = const` segment of this frame a MATERIAL cross-section?

        A section is material only when both its ends sit on a no-slip wall: then the
        strip between two such sections is bounded by walls, and incompressibility forces
        the flux through them to be equal.  Inside the junction the frame's rectangle is
        still part of the union, but its side is *open into the other branch* -- the line
        there is not a stream surface, so the flux through it legitimately differs from the
        outlet value by an amount that no amount of mesh refinement removes.
        """
        fr = self.frames[key]
        return (self.wall_distance_exact(*fr.global_xy(xi, fr.half_width)) <= tol
                and self.wall_distance_exact(*fr.global_xy(xi, -fr.half_width)) <= tol)

    def material_span_closed_form(self, key: int) -> Tuple[float, float]:
        """The analytic [xi_lo, xi_hi] of material stations, to cross-check the geometric
        test above: the stem ends where a branch's outer wall meets its wall line, and a
        branch begins at the crotch where the two inner walls meet."""
        fr = self.frames[key]
        if fr.key == STEM:
            return (0.0, min(self.a_outer[UP][0], self.a_outer[DOWN][0]))
        xi_crotch = (self.frames[UP].half_width + self.frames[DOWN].half_width) \
            * self.cos_t / (2.0 * self.sin_t)
        return (xi_crotch, fr.length)

    def primary_frame(self, x: float, y: float, tol: float = ABSORB_TOL) -> int:
        """Deterministic frame assignment: the frame whose centreline is nearest.

        The stem/branch rectangles overlap in a lens around the junction, so assignment
        is a *convention* and is recorded as such (see the design doc).  Never contradicts
        `contains`: whatever that accepts, this resolves (possibly as "absorbed").
        """
        verdict, key, gap = self.membership(x, y, tol)
        if key is None:
            raise ValueError(f"point ({x}, {y}) outside the domain (nearest rect {gap:.3e})")
        return key

    # --------------------------------------------------- blend weights (plan b)
    def chi(self, x: float, y: float, sigma: float) -> Dict[int, Tuple[float, Point]]:
        """Smooth membership chi_k and its gradient, C^1 in the whole plane.

        chi_k == 1 deep inside frame k and decays over sigma outside it, so the
        blended features are single-valued and differentiable across the
        stem/branch interfaces (the hard-assignment kinks are what made the 2026-03
        attempt's junction points ambiguous).
        """
        s2 = sigma * sigma
        out: Dict[int, Tuple[float, Point]] = {}
        for key, fr in self.frames.items():
            xi, eta = fr.local(x, y)
            g_xi_lo = max(0.0, -xi)
            g_xi_hi = max(0.0, xi - fr.length)
            g_eta = max(0.0, abs(eta) - fr.half_width)
            val = math.exp(-(g_xi_lo * g_xi_lo + g_xi_hi * g_xi_hi + g_eta * g_eta) / (2.0 * s2))
            gx = gy = 0.0
            if val > 0.0:
                for g, coef, vec in (
                    (g_xi_lo, -1.0, fr.d),
                    (g_xi_hi, 1.0, fr.d),
                    (g_eta, math.copysign(1.0, eta) if eta != 0.0 else 0.0, fr.m),
                ):
                    if g > 0.0:
                        k = -val * g / s2 * coef
                        gx += k * vec[0]
                        gy += k * vec[1]
            out[key] = (val, (gx, gy))
        return out

    def blend_weights(self, x: float, y: float, sigma: float) -> Dict[int, Tuple[float, Point]]:
        """w_k = chi_k / sum(chi) and its gradient (sum_k w_k = 1 identically)."""
        chi = self.chi(x, y, sigma)
        tot = sum(v for v, _ in chi.values())
        if tot <= 0.0:
            raise ValueError("blend weights vanished: point far outside the domain")
        g_tot = (sum(g[0] for _, g in chi.values()) / tot,
                 sum(g[1] for _, g in chi.values()) / tot)
        out: Dict[int, Tuple[float, Point]] = {}
        for k, (v, g) in chi.items():
            w = v / tot
            out[k] = (w, (g[0] / tot - w * g_tot[0], g[1] / tot - w * g_tot[1]))
        return out

    # ------------------------------------------------------------- 14-like feats
    def features(self, x: float, y: float, sigma: float) -> Dict[str, float]:
        """Geometry features, frame-blended (plan b inputs)."""
        acc: Dict[str, float] = {name: 0.0 for name in GEOMETRY_FEATURES}
        for key, (w, _) in self.blend_weights(x, y, sigma).items():
            if w <= 0.0:
                continue
            fr = self.frames[key]
            vals = self._frame_features(fr, x, y)
            for name, val in vals.items():
                acc[name] += w * val
        return acc

    def features_hard(self, x: float, y: float) -> Dict[str, float]:
        return self._frame_features(self.frames[self.primary_frame(x, y)], x, y)

    def _frame_features(self, fr: Frame, x: float, y: float) -> Dict[str, float]:
        """Network-facing features.

        Every one of them is C^1 in (x*, y*): |eta| and exp(-r) style kinks at the
        centreline / focal point would silently poison the second total derivative
        plan (b) needs, so clearance is parabolic and proximity is gaussian.
        """
        xi, eta = fr.local(x, y)
        eta_n = eta / fr.half_width
        clearance = max(0.0, 1.0 - eta_n * eta_n) ** 2
        return {
            "eta_norm_star": eta_n,
            "axial_frac_star": xi / fr.length,
            "wall_distance_frac": clearance,
            "half_width_ratio_star": fr.half_width / self.h_stem,
            "tangent_x_star": fr.d[0],
            "tangent_y_star": fr.d[1],
            "branch_stem_flag": 1.0 if fr.key == STEM else 0.0,
            "branch_up_flag": 1.0 if fr.key == UP else 0.0,
            "branch_down_flag": 1.0 if fr.key == DOWN else 0.0,
            "junction_proximity_star": self._junction_proximity(x, y),
        }

    def clearance(self, x: float, y: float, sigma: float) -> float:
        """Blended 1-(eta/h)^2: 0 at the wall, 1 on the centreline (plan (a) envelope)."""
        return self.features(x, y, sigma)["wall_distance_frac"]

    def _junction_proximity(self, x: float, y: float) -> float:
        r2 = (x - self.j_point[0]) ** 2 + (y - self.j_point[1]) ** 2
        return math.exp(-r2 / (2.0 * self.h_stem * self.h_stem))

    GEOM_FEAT_NAMES: Tuple[str, ...] = ()

    def features_jacobian(self, x: float, y: float, sigma: float) -> Dict[str, Point]:
        """Analytic d(feature)/d(x*, y*) for every blended feature (plan b chain)."""
        weights = self.blend_weights(x, y, sigma)
        vals = {k: self._frame_features(self.frames[k], x, y) for k in weights}
        grads = {k: self._frame_feature_grads(self.frames[k], x, y) for k in weights}
        out: Dict[str, Point] = {}
        for name in GEOMETRY_FEATURES:
            gx = sum(weights[k][1][0] * vals[k][name] + weights[k][0] * grads[k][name][0]
                     for k in weights)
            gy = sum(weights[k][1][1] * vals[k][name] + weights[k][0] * grads[k][name][1]
                     for k in weights)
            out[name] = (gx, gy)
        return out

    def _frame_feature_grads(self, fr: Frame, x: float, y: float) -> Dict[str, Point]:
        xi, eta = fr.local(x, y)
        # grad(xi) = d, grad(eta) = m  (the frame basis is orthonormal)
        ux, uy = fr.d
        mx, my = fr.m
        h = fr.half_width
        eta_n = eta / h
        g_eta_n = (mx / h, my / h)
        if abs(eta_n) < 1.0:
            g_clear = (-4.0 * eta_n * (1.0 - eta_n * eta_n) * g_eta_n[0],
                       -4.0 * eta_n * (1.0 - eta_n * eta_n) * g_eta_n[1])
        else:
            g_clear = (0.0, 0.0)
        jp = self._junction_proximity(x, y)
        g_jp = (-jp * (x - self.j_point[0]) / (self.h_stem ** 2),
                -jp * (y - self.j_point[1]) / (self.h_stem ** 2))
        return {
            "eta_norm_star": g_eta_n,
            "axial_frac_star": (ux / fr.length, uy / fr.length),
            "wall_distance_frac": g_clear,
            "half_width_ratio_star": (0.0, 0.0),
            "tangent_x_star": (0.0, 0.0),
            "tangent_y_star": (0.0, 0.0),
            "branch_stem_flag": (0.0, 0.0),
            "branch_up_flag": (0.0, 0.0),
            "branch_down_flag": (0.0, 0.0),
            "junction_proximity_star": g_jp,
        }

    # ------------------------------------------------------------ FD grids
    def branch_grid(self, key: int, spacing: float, wall_margin: float = 0.15,
                    xi_start: float = 0.35, xi_end_margin: float = 0.0) -> dict:
        fr = self.frames[key]
        eta_max = fr.half_width * (1.0 - wall_margin)
        xi0 = xi_start if key == STEM else xi_start
        xi1 = fr.length - xi_end_margin
        if key == STEM:
            xi1 = fr.length - 0.05
        n_eta = max(3, int(round(2.0 * eta_max / spacing)) | 1)  # odd -> centreline node
        n_xi = max(5, int(round((xi1 - xi0) / spacing)) + 1)
        return {
            "kind": "branch",
            "branch": fr.name,
            "branch_key": key,
            "origin": fr.origin,
            "d": fr.d,
            "m": fr.m,
            "half_width": fr.half_width,
            "xi0": xi0,
            "xi1": xi1,
            "eta_max": eta_max,
            "n_xi": n_xi,
            "n_eta": n_eta,
            "spacing": spacing,
        }

    def junction_grid(self, spacing: float, radius: float = 1.15,
                      crotch_exclude_radius: float = 0.75) -> dict:
        cx, cy = self.j_point
        return {
            "kind": "junction",
            "x0": cx - radius,
            "x1": cx + radius,
            "y0": -radius,
            "y1": radius,
            "cx": self.crotch[0],
            "cy": self.crotch[1],
            "r_excl": crotch_exclude_radius,
            "spacing": spacing,
        }

    def grid_points(self, spec: dict) -> List[Point]:
        """Python-side replica of the structured emitter (used for FD and checks)."""
        pts: List[Point] = []
        if spec["kind"] == "branch":
            fr = self.frames[spec["branch_key"]]
            n_xi, n_eta = spec["n_xi"], spec["n_eta"]
            d_xi = (spec["xi1"] - spec["xi0"]) / max(n_xi - 1, 1)
            d_eta = 2.0 * spec["eta_max"] / max(n_eta - 1, 1)
            for i in range(n_xi):
                xi = spec["xi0"] + i * d_xi
                for j in range(n_eta):
                    eta = -spec["eta_max"] + j * d_eta
                    pts.append(fr.global_xy(xi, eta))
        else:
            n_x = max(3, int(round((spec["x1"] - spec["x0"]) / spec["spacing"])) + 1)
            n_y = max(3, int(round((spec["y1"] - spec["y0"]) / spec["spacing"])) + 1)
            dx = (spec["x1"] - spec["x0"]) / max(n_x - 1, 1)
            dy = (spec["y1"] - spec["y0"]) / max(n_y - 1, 1)
            for i in range(n_x):
                x = spec["x0"] + i * dx
                for j in range(n_y):
                    y = spec["y0"] + j * dy
                    if math.hypot(x - spec["cx"], y - spec["cy"]) < spec["r_excl"]:
                        continue
                    if not self.contains(x, y):
                        continue
                    pts.append((x, y))
        return pts

    # ----------------------------------------------------- mesh-independence
    def report_quantities(self) -> List[str]:
        return MESH_INDEPENDENCE_QUANTITIES

    def perimeter(self) -> float:
        return self._perimeter

    def max_half_width(self) -> float:
        return max(fr.half_width for fr in self.frames.values())

    def area(self) -> float:
        return self.polygon.area()


GEOMETRY_FEATURES: Tuple[str, ...] = (
    "eta_norm_star",
    "axial_frac_star",
    "wall_distance_frac",
    "half_width_ratio_star",
    "tangent_x_star",
    "tangent_y_star",
    "branch_stem_flag",
    "branch_up_flag",
    "branch_down_flag",
    "junction_proximity_star",
)
TGeometry.GEOM_FEAT_NAMES = GEOMETRY_FEATURES

MESH_INDEPENDENCE_QUANTITIES: Tuple[str, ...] = (
    "q_stem",
    "q_up",
    "q_down",
    "dp_stem_to_up",
    "dp_stem_to_down",
    "p_junction_over_outlet",
)
"""Integral cross-section / volume-averaged quantities judged by relative change
(autopsy P1).  Mass closure and the symmetry check are near-zero quantities and get
absolute gates instead; point values and residual norms belong to K0."""


def _shoelace(verts: Sequence[Point]) -> float:
    s = 0.0
    n = len(verts)
    for i in range(n):
        x0, y0 = verts[i]
        x1, y1 = verts[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return 0.5 * s


def _dist_point(x: float, y: float, p: Point) -> float:
    return math.hypot(x - p[0], y - p[1])


def _dist_point_seg(px: float, py: float, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    len2 = vx * vx + vy * vy
    if len2 <= 0.0:
        return math.hypot(wx, wy)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / len2))
    return math.hypot(wx - t * vx, wy - t * vy)


def relative_change(coarse: float, fine: float) -> float:
    denom = max(abs(fine), 1.0e-12)
    return abs(coarse - fine) / denom


def case_by_id(case_id: str) -> TCase:
    table = {c.case_id: c for c in (TCase(), ASYM_CASE)}
    if case_id not in table:
        raise KeyError(f"unknown case {case_id!r}; known: {sorted(table)}")
    return table[case_id]


def mesh_levels(base_spacing: float = 0.16) -> List[dict]:
    """Three global refinement levels plus a junction-graded level (autopsy P1)."""
    out = []
    for idx, mult in enumerate((1.0, 0.5, 0.25), start=1):
        out.append({"level": idx, "name": f"h{idx}", "spacing": base_spacing * mult,
                    "graded": False})
    out.append({"level": 4, "name": "hgrade", "spacing": base_spacing * 0.5, "graded": True})
    return out


def border_counts(geom: TGeometry, spacing: float, graded: bool) -> List[int]:
    """Elements per contour border; graded mode densifies borders touching the junction."""
    counts = []
    for (a, b), label in zip(geom.polygon.edge_ends, geom.polygon.edge_labels):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
        n = max(4, int(round(length / spacing)))
        if graded:
            near = _dist_point(mid[0], mid[1], geom.j_point) < 1.6 * max(geom.h_stem, 1e-9) + length
            if near or label == "wall" and _dist_point(mid[0], mid[1], geom.crotch) < 1.2:
                n = int(round(n * 1.8))
        if label.startswith("inlet") or label.startswith("outlet"):
            n = max(8, int(round(n * 1.2)))
        counts.append(n)
    return counts


def spacing_tolerance(values: Sequence[float],
                      digits: int = FREEFEM_PRINT_DIGITS) -> float:
    """The largest spread a *uniform* axis can show once it has been printed at `digits`.

    Derivation, no free parameter: one printed coordinate carries at most
    `half_ulp(|value|) = 0.5*10^(floor(log10|value|)-(digits-1))` of rounding error; a
    spacing is the difference of two of them, so it carries at most 2*half_ulp; the spread
    is the largest spacing minus the smallest, so its error is at most **4*half_ulp**
    evaluated at the largest magnitude on the axis.  It is the same ruler as
    `TGeometry.representation_bound` (which is the 2*half_ulp for one point-to-line gap)
    times the one extra difference a spacing introduces -- hence floor at ABSORB_TOL too.

    Measured, the reason this exists: the instance's `TB-base_h3_samples_branch_up_h.csv`
    axis has min 0.0401 / max 0.040112 (spread **1.20e-05**) and the graded
    `stem_h2` 0.16363/0.163641 (**1.10e-05**), reproduced on the laptop from the same
    lattice; |xi|max = 4 gives 4*half_ulp = 2.0e-05 which covers both, while the test this
    replaces demanded 4.0e-08 -- exactness a 6-digit file cannot carry, so it was reading a
    representation artefact as "the mesh is not a lattice".
    """
    m = max((abs(float(v)) for v in values), default=0.0)
    return max(ABSORB_TOL, 4.0 * TGeometry.half_ulp(m, digits))
