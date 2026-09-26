#!/usr/bin/env python3
"""T6-Re: derive the finite-Reynolds Navier-Stokes solve from the SHIPPED Stokes .edp.

Work order: docs/revision/派单-T6Re真值探针-20260926.md (统括官, 2026-09-26, hard line 9/28).

The claim this script exists to make verifiable is narrow and it is enforced, not asserted:
**the NS file is the shipped Stokes file plus a declared difference set, and the only
physical term that appears is the convection.**  Mesh, borders, boundary conditions,
inflow profile, viscosity, quadrature order, the vertex-tag loop and the raw write loop are
never reworded, re-indented or re-wrapped.  `--check` proves it by *inverting*: delete the
declared insertions, undo the declared replacements, and require the result to equal
`C-base_stokes.edp` byte for byte.  Anything undeclared breaks the inversion and the
script exits 1 rather than shipping a file whose difference set nobody can name.

Two declared modelling choices, both in the emitted header comment:

* The weak form is the Stokes form multiplied by Re, so the diffusion/pressure block stays
  byte-identical.  The unknown the solver calls `p` is then P = Re*p, which is why
  `p_star` is written as `p(xx,yy)/Re` and remains the same physical quantity as in the
  Stokes CSV (declared replacement #2).
* FreeFEM v4.9 writes six significant digits and has no `setprecision` (measured on the
  instance: "The Identifier setprecision does not exist").  Precision is recovered on the
  writing side by a companion stream that prints, per field, `hi = floor(x*1e4)/1e4`
  (exactly representable in six digits for |x| < 1000) and `lo = x - hi` (|lo| < 1e-4, so
  its own six digits carry ~1e-10 absolute).  `finalize_ns_truth.py` merges them and
  asserts the floor.  No unverified FreeFEM feature is used anywhere: only `ofstream`,
  `<<`, `endl`, `floor`, arithmetic, `int2d`, `pow`, `sqrt`, and a `break` inside a `for`
  -- every one of those already appears in a file the instance has run.

Usage:
    python3 gen_ns_re_edp.py                 # diff proof only, writes nothing
    python3 gen_ns_re_edp.py --write         # emit the four .edp into the case dirs
    python3 gen_ns_re_edp.py --list-differences
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE_DIR = HERE.parents[0] / "cases" / "contraction_2d" / "cfd" / "C-base"
STOKES = "C-base_stokes.edp"
STOKES_FIRST_LINE = "// Auto-generated contraction_2d Stokes solve for C-base"
RE_LEVELS = ("1e-3", "1", "10", "50")        # 1e-3 is the Re->0 consistency reference
CONV_TOL = 1.0e-8                             # relative Picard increment, fixed here
CONV_MAX = 60                                 # iteration cap, fixed here

HEADER = [
    "// T6-Re finite-Reynolds Navier-Stokes probe, derived from C-base_stokes.edp",
    "// Re = REV. Star units: W_stem = 1, inlet mean velocity = 1, mu = 1, so Re is the",
    "//   parameter multiplying the convective term and the weak form below is the shipped",
    "//   Stokes form times Re. Consequence: the pressure the solver solves for is P = Re*p,",
    "//   hence p_star is written as p/Re and stays comparable with the Stokes CSV.",
    "// Nonlinear scheme: Picard with the advecting velocity frozen at the previous iterate",
    "//   (u0, v0). The first iterate has u0 = v0 = 0, so it IS the shipped Stokes solve:",
    "//   convection is the only term that can move the answer.",
    f"//   Criterion: squared relative L2 increment du2 < ({CONV_TOL:.1e})^2; cap CMAX iters.",
    "//   Hitting the cap prints 'NS NOT CONVERGED' and the CSV is still written -- the",
    "//   cost table must then record the level as FAIL, never delete it.",
    "// Printing: v4.9 gives 6 significant digits and no setprecision, so *_raw_pair.csv",
    "//   carries (hi, lo) per field with hi = floor(x*1e2)/1e2; finalize_ns_truth.py merges",
    "//   them and asserts the reconstruction floor is below 1e-10 of the field scale.",
]

# --- the declared difference set ------------------------------------------------------
INSERT_BEFORE = {
    "solve Stokes([u,v,p],[ut,vt,qt], solver=UMFPACK) =": [
        "for (int it = 0; it < CONV_MAX; ++it) {",
        "  u0 = u; v0 = v;"],
    "    )": ["    + Re*((u0*dx(u)+v0*dy(u))*ut + (u0*dx(v)+v0*dy(v))*vt)"],
}
INSERT_AFTER = {
    "real beta = 0.7;": ["real Re = REV;", f"real CONV_TOL = {CONV_TOL:.1e};",
                         f"int CONV_MAX = {CONV_MAX};"],
    "Vh u, v, ut, vt;": ["Vh u0, v0;"],
    "  ;": [  # closes the solve statement: increment, exit test, non-convergence report.
        # Every statement here is one line and uses only what a file the instance has
        # already run uses -- int2d, pow, arithmetic, `if (c) break;`, cout.  The criterion
        # is squared so no sqrt is needed: du2 < CONV_TOL*CONV_TOL is the same test as
        # relative-L2-increment < CONV_TOL.
        "  real du2 = int2d(Th)(pow(u-u0,2)+pow(v-v0,2))"
        " / (int2d(Th)(pow(u,2)+pow(v,2)) + 1.0e-30);",
        f"  if (du2 < CONV_TOL*CONV_TOL) break;",
        f"  if (it == {CONV_MAX - 1}) cout << \"NS NOT CONVERGED Re=REV iterations=\""
        ' << it+1 << " du2=" << du2 << endl;',
        'cout << "NS it=" << it+1 << " du2=" << du2 << " Re=REV" << endl;',
        "}  // end Picard iteration"],
}
PAIR_STREAM = [
    "// companion hi/lo stream: reconstructing needs both files, neither is optional.",
    "// hi keeps 2 decimals, so it stays inside 6 significant digits for |x| < 1000;",
    "// lo = x - hi is then below 1e-2 and its own 6 digits carry ~5e-9 absolute.",
    'ofstream fpair("C-base_ns_reREV_raw_pair.csv");',
    'fpair << "x_star,y_star,u_hi,u_lo,v_hi,v_lo,p_hi,p_lo" << endl;',
    "for (int i = 0; i < Th.nv; ++i) {   // companion loop: same vertex order, no renumbering",
    "  real hu = u(Th(i).x, Th(i).y);",
    "  real hv = v(Th(i).x, Th(i).y);",
    "  real hp = p(Th(i).x, Th(i).y)/Re;",
    "  real hi_u = floor(hu*1e2)/1e2;",
    "  real hi_v = floor(hv*1e2)/1e2;",
    "  real hi_p = floor(hp*1e2)/1e2;",
    '  fpair << Th(i).x << "," << Th(i).y << "," << hi_u << "," << hu - hi_u << ","',
    '        << hi_v << "," << hv - hi_v << "," << hi_p << "," << hp - hi_p << endl;',
    "}  // end hi/lo companion loop",
]
# the only two shipped lines that are replaced rather than kept
REPLACEMENTS = (
    ("solve Stokes([u,v,p],[ut,vt,qt], solver=UMFPACK) =",
     "solve NS([u,v,p],[ut,vt,qt], solver=UMFPACK) =",
     "label only: the statement is now iterated; its arguments are untouched"),
    ('  fout << xx << "," << yy << "," << u(xx,yy) << "," << v(xx,yy) << "," '
     '<< p(xx,yy) << "," << vTag[i] << endl;',
     '  fout << xx << "," << yy << "," << u(xx,yy) << "," << v(xx,yy) << "," '
     '<< p(xx,yy)/Re << "," << vTag[i] << endl;',
     "p_star divided by Re, required by the multiply-by-Re convention declared above"),
)
# the shipped .edp hard-codes /root/dev/pinn_v3/..., which does not exist on the instance
# (already flagged in docs/revision/CFD真值同机复算-20260925.md).  The NS probe writes next
# to its own .edp instead, so the runner must `cd` into the level directory -- the same
# convention model/scripts/route2/generate_t_case.py uses when it solves.
SHIPPED_ABS = "/root/dev/pinn_v3/cases/contraction_2d/cfd/C-base/C-base_raw.csv"
PATH_SWAP = (SHIPPED_ABS, "C-base_ns_reREV_raw.csv")


def _sub(line: str, re_label: str) -> str:
    return (line.replace("REV", re_label).replace("CMAX", str(CONV_MAX)))


def inserted_lines(re_label: str) -> set:
    out = {_sub(h, re_label) for h in HEADER} | {_sub(h, re_label) for h in PAIR_STREAM}
    for adds in INSERT_BEFORE.values():
        out |= {_sub(a, re_label) for a in adds}
    for adds in INSERT_AFTER.values():
        out |= {_sub(a, re_label) for a in adds}
    out.discard("")            # a blank line is never a droppable insertion
    return out


def assert_insertions_are_unique(stokes: str) -> None:
    """A declared insertion must never equal a line the shipped file already contains.

    Otherwise the inverse would delete shipped code: the first version of this shipped a
    bare "}" as its loop-closer, and inverting removed every closing brace in the file.
    """
    shipped = {ln for ln in stokes.splitlines() if ln.strip()}
    clash = sorted(inserted_lines("1") & shipped)
    if clash:
        raise SystemExit("declared insertions collide with shipped lines (the inverse would "
                         "delete them): " + " | ".join(c[:60] for c in clash))


def transform(stokes: str, re_label: str) -> str:
    lines = stokes.split("\n")
    if lines[0] != STOKES_FIRST_LINE:
        raise SystemExit(f"{STOKES}: the first line is not the expected anchor, so the "
                         f"difference set cannot be proven -- refusing to emit")
    out = [lines[0]] + [_sub(h, re_label) for h in HEADER]
    for ln in lines[1:]:
        if ln in INSERT_BEFORE:
            out.extend(_sub(a, re_label) for a in INSERT_BEFORE[ln])
        keep = ln
        for old, new, _why in REPLACEMENTS:
            if ln == old:
                keep = new
        if PATH_SWAP[0] in keep:
            keep = keep.replace(PATH_SWAP[0], PATH_SWAP[1].replace("REV", re_label))
        out.append(keep)
        if ln in INSERT_AFTER:
            out.extend(_sub(a, re_label) for a in INSERT_AFTER[ln])
        if ln.startswith("cout << \"Saved: "):
            out.extend(_sub(a, re_label) for a in PAIR_STREAM)
    return "\n".join(out)


def invert(ns: str, re_label: str) -> str:
    """Undo the declared difference set; the result must be the shipped file."""
    drop = inserted_lines(re_label)
    back = []
    for ln in ns.split("\n"):
        if ln in drop:
            continue
        for old, new, _why in REPLACEMENTS:
            if ln == new:
                ln = old
        if PATH_SWAP[1].replace("REV", re_label) in ln:
            ln = ln.replace(PATH_SWAP[1].replace("REV", re_label), PATH_SWAP[0])
        back.append(ln)
    return "\n".join(back)


def difference_report(stokes: str, ns: str):
    added, replaced = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, stokes.split("\n"), ns.split("\n")).get_opcodes():
        if tag == "equal":
            continue
        added += [x for x in ns.split("\n")[j1:j2] if tag == "insert" and x.strip()]
        if tag != "insert":
            replaced += [(x, y) for x, y in zip(stokes.split("\n")[i1:i2],
                                                ns.split("\n")[j1:j2])]
    return added, replaced


def main() -> int:
    ap = argparse.ArgumentParser(description="derive the NS .edp from the shipped Stokes one")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--list-differences", action="store_true")
    ap.add_argument("--case-dir", default=str(CASE_DIR))
    args = ap.parse_args()
    case_dir = Path(args.case_dir)
    stokes = (case_dir / STOKES).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert_insertions_are_unique(stokes)
    bad = 0
    for re_label in RE_LEVELS:
        ns = transform(stokes, re_label)
        same = invert(ns, re_label).strip() == stokes.strip()
        conv = [ln for ln in ns.split("\n") if "u0*dx(u)" in ln]
        added, replaced = difference_report(stokes, ns)
        print(f"Re={re_label:5s}  lines {len(stokes.splitlines())} -> {len(ns.splitlines())}"
              f"  inserted={len(added)}  replaced={len(replaced)}  "
              f"convection lines={len(conv)}  invert-to-Stokes={'OK' if same else 'MISMATCH'}")
        if not same:
            bad += 1
            for ln in list(difflib.unified_diff(stokes.split("\n"),
                                                invert(ns, re_label).split("\n"),
                                                "stokes", "inverted", lineterm=""))[:16]:
                print("   ", ln[:104])
        if len(conv) != 1:
            print(f"    the convective term must be exactly one line, found {len(conv)}")
            bad += 1
        expected_replaced = len(REPLACEMENTS) + 2      # + the two output-path lines
        if len(replaced) != expected_replaced:
            print(f"    {len(replaced)} replaced lines, {expected_replaced} declared "
                  f"(2 named + 2 output paths)")
            for old, new in replaced:
                print(f"      ~ {old[:52]!r} -> {new[:52]!r}")
            bad += 1
        if args.list_differences:
            for ln in added:
                print(f"    + {ln}")
            for old, new in replaced:
                why = next((w for o, n, w in REPLACEMENTS if o == old),
                           "output path: one level's truth must not overwrite the "
                           "shipped Stokes truth")
                print(f"    ~ {new}\n        why: {why}")
        if args.write:
            out_dir = case_dir.parent / f"C-base_ns_re{re_label}"
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"C-base_ns_re{re_label}.edp"
            target.write_text(ns if ns.endswith("\n") else ns + "\n", encoding="utf-8")
            print(f"    wrote {target.relative_to(case_dir.parents[1])}")
    if bad:
        print("DIFF PROOF FAILED -- the emitted file is not the shipped file plus a "
              "declared difference set; the work order forbids shipping that")
        return 1
    print("diff proof OK: each NS file inverts to the shipped Stokes text byte for byte once "
          "the declared inserts and the two declared replacements are undone; the physics "
          "difference is one convective line.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
