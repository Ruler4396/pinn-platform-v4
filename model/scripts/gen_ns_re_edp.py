#!/usr/bin/env python3
"""T6-Re: derive the finite-Reynolds Navier-Stokes solve from the SHIPPED Stokes .edp.

Work order: docs/revision/派单-T6Re真值探针-20260926.md (统括官, 2026-09-26, hard line 9/28).

The claim this script exists to make verifiable is narrow and it is enforced, not asserted:
**the NS file is the shipped Stokes file plus a declared difference set, and the only
physical term that appears is the convection.**  Mesh, borders, boundary conditions, inflow
profile, viscosity, quadrature order, the vertex-tag loop and the raw write loop are never
reworded, re-indented or re-wrapped.  The default run proves it by *inverting*: delete the
declared insertions, undo the declared replacements, and require the result to equal
`C-base_stokes.edp` byte for byte.  Anything undeclared breaks the inversion and the script
exits 1 rather than shipping a file whose difference set nobody can name.

WHAT v4.9 ACTUALLY ACCEPTS, AND WHERE THAT CAME FROM.  The first version of this file failed
to compile on the instance, at `real CONV_TOL = 1.0e-8;`, with
    Error line number 21 ... before  token _
so these two rules are hard-coded into what gets emitted, and `--check-syntax` enforces them
on every line of every file this script writes:

* **no underscore in an identifier.**  `_` occurs in the shipped .edp files only inside
  string literals and comments, and the four files this script wrote are the only place it
  ever appeared in code -- so the emitted code text keeps `_` out entirely.  Tolerances and
  the iteration cap are therefore *literals* (`1.0e-16`, `60`), and the companion fields are
  `uHi`, `vHi`, `pHi`, the same camelCase the shipped file already uses (`vTag`, `uIn`).
* **whole-line comments only.**  `[^/ ]+ +//` has zero hits across every shipped .edp, so no
  emitted line puts code and a comment on the same line.

Everything else in the emitted block is in one of three buckets, and the generated header
repeats them, because "it looks like FreeFEM" is not a category:

  (i) proven by the instance reproducing `C-base_stokes.edp` bit for bit: `solve`, `int2d`,
      products of two fields (`1.0e-10*p*qt`), `real`/`int` declarations,
      `for (int i = 0; ...; ++i)`, `ofstream`, `<<`, `endl`, scientific literals;
  (ii) proven by the route2 T-case .edp files that the instance compiled and solved (S1,
      four mesh levels): `if (c) break;` inside a `for` block;
  (iii) **not** yet run on this build, and the reason `probe_syntax.edp` exists: `floor`
      (only ever as `floor(x*1.0eN)/1.0eN`), the fespace assignments `u0 = u;` and
      `u = 0.0;`, and a `solve` statement inside a block.  The probe appends each of them to
      the shipped Stokes text as a pure append, one construct per line behind a
      `// PROBE <tag>` comment, and closes its blocks the same merged way the real file does;
      it costs about 1.5 s.  If it stops at a tagged line, the reported line number names
      exactly one construct and the difference set here has a documented replacement
      (integer truncation for `floor`; interpolate-into-buffer for `u0 = u`).  The fix
      belongs in this generator, not in a hand-edited .edp.

`pow` is deliberately absent: it appears in other shipped cases' .edp files but not in the
contraction file whose reproduction is what we verified, and `(u-u0)*(u-u0)` needs no
function at all, so keeping it costs nothing and removes one unknown.

Precision is recovered on the writing side: v4.9 prints six significant digits and has no
`setprecision` (measured: "The Identifier setprecision does not exist"), so each .edp also
writes a companion `<case>_raw_pair.csv` with `hi = floor(x*1.0eN)/1.0eN` and `lo = x - hi`.
`N` is per field and is *derived* here from the measured field magnitudes -- see
`finalize_ns_truth.hi_decimals_for`, which is the one place the two inequalities (hi must be
printable in six digits; lo must be small enough that its own six digits reach 1e-10 of the
field scale) are written down.  `finalize_ns_truth.py --selfcheck` then reads those widths
back out of the emitted text and re-derives them, so the two scripts cannot drift apart.

Usage:
    python3 gen_ns_re_edp.py                  # diff proof + syntax lint, writes nothing
    python3 gen_ns_re_edp.py --write          # emit the four .edp and the probe into case dirs
    python3 gen_ns_re_edp.py --list-differences
    python3 gen_ns_re_edp.py --check-syntax   # lint + its own controls, no case files needed
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import finalize_ns_truth as fz                                   # noqa: E402

CASE_DIR = HERE.parents[0] / "cases" / "contraction_2d" / "cfd" / "C-base"
STOKES = "C-base_stokes.edp"
STOKES_FIRST_LINE = "// Auto-generated contraction_2d Stokes solve for C-base"
RE_LEVELS = ("1e-3", "1", "10", "50")        # 1e-3 is the Re->0 consistency reference
CONV_TOL = 1.0e-8                            # relative Picard increment, fixed here
CONV_MAX = 60                                # iteration cap, fixed here
DU2_LIMIT = CONV_TOL * CONV_TOL              # emitted as a literal: the test is on du2
RE_PRIME = re.compile(r"real\s+\w*_\w*\s*=")          # an underscore identifier, declared
TRAILING_COMMENT = re.compile(r"^[^/\";]*[^/\" ]\s+//[a-zA-Z0-9 ]")

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
    return line.replace("REV", re_label)


def hi_lines(widths: dict[str, int]) -> list[str]:
    """The three floor() lines, derived from the widths and nothing else.  Both the emitted
    companion stream and the syntax probe take them from here, so the probe cannot test a
    line that differs from the one the real file writes."""
    return [f"  real {fz.LETTER[f]}Hi = floor(h{fz.LETTER[f]}*1.0e{widths[f]})"
            f"/1.0e{widths[f]};" for f in fz.FIELDS]


def build_spec(widths: dict[str, int]) -> dict:
    """The declared difference set, with the derived hi widths filled in.

    Returned as one object that transform/invert/inserted_lines all read, so a width can
    only ever reach the emitted file through this function -- there is no second copy of the
    number for a docstring or a comment to disagree with.
    """
    header = [
        "// T6-Re finite-Reynolds Navier-Stokes probe, derived from C-base_stokes.edp",
        "// Re = REV. Star units: W_stem = 1, inlet mean velocity = 1, mu = 1, so Re is the",
        "//   parameter multiplying the convective term and the weak form below is the shipped",
        "//   Stokes form times Re. Consequence: the pressure the solver solves for is P = Re*p,",
        "//   hence p_star is written as p/Re and stays comparable with the Stokes CSV.",
        "// Nonlinear scheme: Picard with the advecting velocity frozen at the previous iterate",
        "//   (u0, v0). u and v are set to zero before the loop, so the first iterate advects",
        "//   with u0 = v0 = 0 and IS the shipped Stokes solve: convection is the only term",
        "//   that can move the answer, and nothing here depends on initialisation garbage.",
        f"//   Criterion: squared relative L2 increment du2 < {DU2_LIMIT:.1e}, which is the",
        f"//   relative increment {CONV_TOL:.1e} squared (written as a literal because v4.9",
        "//   rejects underscore identifiers, so this file declares no named constants).",
        f"//   Cap {CONV_MAX} iterations; it == {CONV_MAX - 1} is the non-convergence report.",
        "//   Hitting the cap prints 'NS NOT CONVERGED' and the CSV is still written -- the",
        "//   cost table must then record the level as FAIL, never delete it.",
        "// Printing: v4.9 gives 6 significant digits and no setprecision, so *_raw_pair.csv",
        "//   carries (hi, lo) per field with hi = floor(x*1.0eN)/1.0eN and",
    ]
    header += [f"//     N = {widths[field]} for {field}: |{field}| must stay below "
               f"{fz.budget_scale(widths[field]):g} or hi itself loses digits."
               for field in fz.FIELDS]
    header += [
        "//   The widths are per field because the three fields span two decades of",
        "//   magnitude and no single width serves both ends of that range; finalize_ns_truth",
        "//   reads them back out of this file instead of trusting a copy of them, and",
        "//   finalize_ns_truth.py merges (hi, lo) and asserts the reconstruction floor.",
        "// Syntax: whole-line comments and underscore-free identifiers only, because",
        "//   v4.9's parser rejects the other two forms (measured on the instance).",
        "//   Four constructs here have not been run on this build yet: floor(), the field",
        "//   assignments u = 0.0 and u0 = u, and a solve statement inside a block.  The file",
        "//   model/cases/contraction_2d/cfd/C-base_ns_re1/probe_syntax.edp appends each of",
        "//   them, one per line and tagged with a '// PROBE' comment, to the shipped Stokes",
        "//   text; run that first, it costs about 1.5 s and no postprocessing.",
    ]
    insert_before = {
        "solve Stokes([u,v,p],[ut,vt,qt], solver=UMFPACK) =": [
            # Seed the advecting field with an explicit zero rather than relying on whatever
            # a freshly declared fespace happens to hold: with u0 = v0 = 0 the first solve
            # *is* the shipped Stokes solve, which is what makes "convection is the only
            # difference" a fact about the emitted file instead of an assumption about
            # initialisation.  The zero is overwritten by that solve.
            "u = 0.0; v = 0.0;",
            f"for (int it = 0; it < {CONV_MAX}; ++it) {{",
            "  u0 = u; v0 = v;"],
        "    )": ["    + Re*((u0*dx(u)+v0*dy(u))*ut + (u0*dx(v)+v0*dy(v))*vt)"],
    }
    insert_after = {
        "real beta = 0.7;": ["real Re = REV;"],
        "Vh u, v, ut, vt;": ["Vh u0, v0;"],
        "  ;": [  # closes the solve statement: increment, exit test, non-convergence report.
            # One construct per line on purpose: FreeFEM aborts at the first syntax error and
            # prints its line number, so a failure names exactly one construct.  The loop's
            # closing brace rides on the last line instead of standing alone because a bare
            # "}" also occurs in the shipped file, and the inverse deletes by line content --
            # `assert_insertions_are_unique` is what turns that into a checked fact.
            # No `pow` and no `sqrt`: (u-u0)*(u-u0) is a product of two fields, which the
            # shipped form already has (`1.0e-10*p*qt`), and squaring by multiplication keeps
            # the criterion's units identical without a function this build has not run.
            "  real du2 = int2d(Th)((u-u0)*(u-u0)+(v-v0)*(v-v0))"
            " / (int2d(Th)(u*u+v*v) + 1.0e-30);",
            '  cout << "NS it=" << it+1 << " du2=" << du2 << " Re=REV" << endl;',
            f"  if (du2 < {DU2_LIMIT:.1e}) break;",
            f"  if (it == {CONV_MAX - 1}) cout << \"NS NOT CONVERGED Re=REV iterations=\""
            ' << it+1 << " du2=" << du2 << endl; }'],
    }
    pair = [
        "// companion hi/lo stream: reconstructing needs both files, neither is optional.",
        'ofstream fpair("C-base_ns_reREV_raw_pair.csv");',
        'fpair << "x_star,y_star,u_hi,u_lo,v_hi,v_lo,p_hi,p_lo" << endl;',
        "for (int ii = 0; ii < Th.nv; ++ii) {",
        "  real hu = u(Th(ii).x, Th(ii).y);",
        "  real hv = v(Th(ii).x, Th(ii).y);",
        "  real hp = p(Th(ii).x, Th(ii).y)/Re;",
    ]
    pair += hi_lines(widths)
    pair += [
        '  fpair << Th(ii).x << "," << Th(ii).y << "," << uHi << "," << hu - uHi << ","',
        '        << vHi << "," << hv - vHi << "," << pHi << "," << hp - pHi << endl; }',
    ]
    return {"header": header, "before": insert_before, "after": insert_after,
            "pair": pair, "widths": dict(widths)}


def inserted_lines(re_label: str, spec: dict) -> set:
    out = {_sub(h, re_label) for h in spec["header"]}
    out |= {_sub(h, re_label) for h in spec["pair"]}
    for adds in spec["before"].values():
        out |= {_sub(a, re_label) for a in adds}
    for adds in spec["after"].values():
        out |= {_sub(a, re_label) for a in adds}
    out.discard("")            # a blank line is never a droppable insertion
    return out


def assert_insertions_are_unique(stokes: str, spec: dict) -> None:
    """A declared insertion must never equal a line the shipped file already contains.

    Otherwise the inverse would delete shipped code: the first version of this shipped a
    bare "}" as its loop-closer, and inverting removed every closing brace in the file.
    """
    shipped = {ln for ln in stokes.splitlines() if ln.strip()}
    clash = sorted(inserted_lines("1", spec) & shipped)
    if clash:
        raise SystemExit("declared insertions collide with shipped lines (the inverse would "
                         "delete them): " + " | ".join(c[:60] for c in clash))


def assert_anchors_are_unique(stokes: str, spec: dict) -> None:
    """Every anchor must match exactly one shipped line.

    `transform` inserts after *every* line that equals an anchor, so an anchor that occurs
    twice would silently duplicate the whole Picard block -- and the invert proof would still
    pass, because dropping the inserted lines removes both copies.  That is the one failure
    mode the content-matching scheme cannot catch by itself, so it is checked here.
    """
    lines = stokes.split("\n")
    want = {**spec["before"], **spec["after"]}
    for anchor in list(want) + [old for old, _new, _w in REPLACEMENTS]:
        n = lines.count(anchor)
        if n != 1:
            raise SystemExit(f"anchor {anchor[:52]!r} matches {n} shipped lines, not 1: "
                             f"the insertion would be duplicated (or vanish) and the diff "
                             f"proof would not notice")
    n = sum(1 for ln in lines if PATH_SWAP[0] in ln)
    if n != 2:
        raise SystemExit(f"the shipped output path appears {n} times, not 2 (an ofstream "
                         f"line and a cout line) -- the path swap is no longer understood")


def transform(stokes: str, re_label: str, spec: dict) -> str:
    lines = stokes.split("\n")
    if lines[0] != STOKES_FIRST_LINE:
        raise SystemExit(f"{STOKES}: the first line is not the expected anchor, so the "
                         f"difference set cannot be proven -- refusing to emit")
    out = [lines[0]] + [_sub(h, re_label) for h in spec["header"]]
    for ln in lines[1:]:
        if ln in spec["before"]:
            out.extend(_sub(a, re_label) for a in spec["before"][ln])
        keep = ln
        for old, new, _why in REPLACEMENTS:
            if ln == old:
                keep = new
        if PATH_SWAP[0] in keep:
            keep = keep.replace(PATH_SWAP[0], PATH_SWAP[1].replace("REV", re_label))
        out.append(keep)
        if ln in spec["after"]:
            out.extend(_sub(a, re_label) for a in spec["after"][ln])
        if ln.startswith('cout << "Saved: '):
            out.extend(_sub(a, re_label) for a in spec["pair"])
    return "\n".join(out)


def invert(ns: str, re_label: str, spec: dict) -> str:
    """Undo the declared difference set; the result must be the shipped file."""
    drop = inserted_lines(re_label, spec)
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


# --- what the instance taught us about v4.9's parser ----------------------------------
def code_only(text: str) -> str:
    """Strip string literals and whole-line comments; what is left must be code."""
    out = []
    for ln in text.split("\n"):
        if ln.lstrip().startswith("//"):
            out.append("")
            continue
        out.append(re.sub(r'"(?:[^"\\]|\\.)*"', '""', ln))
    return "\n".join(out)


def syntax_findings(text: str) -> list[tuple[int, str, str]]:
    """Every line that uses a construct v4.9 is known to reject, or that a shipped .edp
    never used.  Returns (line number, kind, line)."""
    hits = []
    for n, ln in enumerate(code_only(text).split("\n"), start=1):
        if "_" in ln:
            hits.append((n, "underscore in code (v4.9: 'before token _')", ln))
        elif TRAILING_COMMENT.search(ln):
            hits.append((n, "code and comment on one line (zero hits in shipped .edp)", ln))
    return hits


def probe_text(stokes: str, widths: dict[str, int]) -> str:
    """The shipped Stokes file with the *new* constructs appended -- a pure append, so the
    difference set is trivially the tail, and every construct the NS solve adds is compiled
    against a file the instance has already solved successfully.

    One construct per line, each preceded by a `// PROBE <tag>` line: FreeFEM aborts at the
    first syntax error and prints the line number, so a single 0.4 s run says which of
    `floor` / `u0 = u` / the iterated solve is the problem and which are already proven.
    """
    body = [
        "// --- T6-Re syntax probe. Appended to the shipped Stokes text unchanged; nothing",
        "//     below is a modelling choice, every line is a construct the NS file uses.",
        "// PROBE real-decl",
        "//   Re is set to 1 here, not to the level's value: the construct under test is a",
        "//   real declaration named Re (the name has no underscore), and at 1.0 the probe's",
        "//   own companion CSV stays interpretable.",
        "real Re = 1.0;",
        "// PROBE fespace-decl",
        "Vh u0, v0;",
        "// PROBE field-zero-assign",
        "u = 0.0; v = 0.0;",
        "// PROBE field-assign",
        "u0 = u; v0 = v;",
        "// PROBE increment-integral",
        "real du2 = int2d(Th)((u-u0)*(u-u0)+(v-v0)*(v-v0))"
        " / (int2d(Th)(u*u+v*v) + 1.0e-30);",
        "// PROBE iteration-loop",
        f"for (int it = 0; it < {CONV_MAX}; ++it) {{",
        "  u0 = u; v0 = v;",
        "// PROBE iterate-report",
        '  cout << "NS it=" << it+1 << " du2=" << du2 << " Re=1.0" << endl;',
        "// PROBE break-in-if",
        f"  if (du2 < {DU2_LIMIT:.1e}) break;",
        "// PROBE merged-closer",
        "//   the real file ends its blocks with 'statement; }' on one line, because a bare",
        "//   closing brace is not a droppable insertion for the inverse proof.",
        f"  if (it == {CONV_MAX - 1}) cout << \"NS NOT CONVERGED Re=1.0 iterations=\""
        ' << it+1 << " du2=" << du2 << endl; }',
        "// PROBE solve-inside-block",
        "//   The NS file wraps the shipped solve statement in a for block.  That is the one",
        "//   construct the whole Picard design rests on, so it gets tested on its own here:",
        "//   same form text as the shipped solve, two iterations, then the exit test.",
        "for (int it = 0; it < 2; ++it) {",
        "  u0 = u; v0 = v;",
        "  solve StokesAgain([u,v,p],[ut,vt,qt], solver=UMFPACK) =",
        "    int2d(Th)(",
        "      dx(u)*dx(ut) + dy(u)*dy(ut)",
        "    + dx(v)*dx(vt) + dy(v)*dy(vt)",
        "    - p*(dx(ut) + dy(vt))",
        "    - qt*(dx(u) + dy(v))",
        "    - 1.0e-10*p*qt",
        "    )",
        "  + on(labIn, u=uIn(y), v=0)",
        "  + on(labWall, u=0, v=0)",
        "  + on(labOut, p=0)",
        "  ;",
        "  real du2b = int2d(Th)((u-u0)*(u-u0)+(v-v0)*(v-v0))"
        " / (int2d(Th)(u*u+v*v) + 1.0e-30);",
        f"  if (du2b < {DU2_LIMIT:.1e}) break;",
        "}",
        "// PROBE companion-stream",
        'ofstream fpair("probe_syntax_pair.csv");',
        'fpair << "x_star,y_star,u_hi,u_lo,v_hi,v_lo,p_hi,p_lo" << endl;',
        "for (int ii = 0; ii < Th.nv; ++ii) {",
        "  real hu = u(Th(ii).x, Th(ii).y);",
        "  real hv = v(Th(ii).x, Th(ii).y);",
        "  real hp = p(Th(ii).x, Th(ii).y)/Re;",
    ]
    body += hi_lines(widths)
    body += [
        '  fpair << Th(ii).x << "," << Th(ii).y << "," << uHi << "," << hu - uHi << ","',
        '        << vHi << "," << hv - vHi << "," << pHi << "," << hp - pHi << endl; }',
        "// PROBE end",
        'cout << "PROBE OK du2=" << du2 << endl;',
    ]
    head = stokes if stokes.endswith("\n") else stokes + "\n"
    return head + "\n".join(body) + "\n"


def syntax_selfcheck() -> int:
    """The lint's own controls: a known-good file must be clean and the exact string that
    broke the instance must be caught.  A lint without a positive control is decoration."""
    rc = 0
    bad = syntax_findings('real CONV_TOL = 1.0e-8;\nint hi_p = 1;\n')
    ok = len(bad) == 2
    rc |= 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] control: the two lines that killed the last run "
          f"({len(bad)} flagged, want 2)")
    good = syntax_findings('real du2 = 1.0e-30;\n  fout << "x_star,y_star" << endl;\n'
                           '// vTag labels are mentioned in this comment with x_star\n')
    ok = not good
    rc |= 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] control: underscore inside a string literal or a "
          f"comment is not flagged ({len(good)} flagged, want 0)")
    tc = syntax_findings('int it = 0;   // the loop counter\n')
    ok = len(tc) == 1 and tc[0][1].startswith("code and comment")
    rc |= 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] control: a trailing comment is flagged as "
          f"unprecedented ({len(tc)}, want 1)")
    # the emitted widths must equal the derived ones, and a wrong one must be catchable
    try:
        widths = derived_widths()
    except Exception as exc:                                   # noqa: BLE001
        print(f"[INDETERMINATE] cannot derive widths here: {exc}")
        return 2
    parsed = fz.hi_decimals_from_edp("\n".join(hi_lines(widths)))
    ok = parsed == widths
    rc |= 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] control: the emitted hi lines parse back to the "
          f"widths they say ({parsed})")
    try:
        fz.hi_decimals_from_edp("  real uHi = floor(hu*1.0e4)/1.0e3;")
        ok = False
        why = "no error raised"
    except ValueError as exc:
        ok = "self-inconsistent" in str(exc)
        why = str(exc)[:60]
    rc |= 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] control: a hi line whose multiplier and divisor "
          f"disagree is refused ({why})")
    lines = hi_lines(widths)
    dup = "\n".join(lines + [lines[-1].replace(f"1.0e{widths['p_star']}",
                                               f"1.0e{widths['p_star'] + 1}")])
    try:
        fz.hi_decimals_from_edp(dup)
        ok = False
        why = "accepted two widths for p_star"
    except ValueError as exc:
        ok = "two different widths" in str(exc)
        why = str(exc)[:60]
    rc |= 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] control: one field split at two widths in one file "
          f"is refused ({why})")
    return rc


def derived_widths() -> dict[str, int]:
    """Per-field `N` from the shipped truth's own magnitudes -- the only place they come from."""
    mags = fz.field_magnitudes()
    return {f: fz.hi_decimals_for(max(abs(v) for v in mags[f])) for f in fz.FIELDS}


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
    ap.add_argument("--check-syntax", action="store_true",
                    help="run the v4.9 lint and its controls on a synthetic input only")
    ap.add_argument("--case-dir", default=str(CASE_DIR))
    args = ap.parse_args()
    if args.check_syntax:
        return syntax_selfcheck()
    case_dir = Path(args.case_dir)
    stokes = (case_dir / STOKES).read_text(encoding="utf-8").replace("\r\n", "\n")
    try:
        widths = derived_widths()
    except (ValueError, FileNotFoundError) as exc:
        print(f"[FAIL] cannot derive the hi widths: {exc}")
        return 1
    spec = build_spec(widths)
    print("derived hi widths from the shipped magnitudes: " +
          ", ".join(f"{f}:N={widths[f]} (<{fz.budget_scale(widths[f]):g})" for f in widths))
    assert_insertions_are_unique(stokes, spec)
    assert_anchors_are_unique(stokes, spec)
    bad = 0
    for re_label in RE_LEVELS:
        ns = transform(stokes, re_label, spec)
        same = invert(ns, re_label, spec).strip() == stokes.strip()
        conv = [ln for ln in ns.split("\n") if "u0*dx(u)" in ln]
        hits = syntax_findings(ns)
        added, replaced = difference_report(stokes, ns)
        print(f"Re={re_label:5s}  lines {len(stokes.splitlines())} -> {len(ns.splitlines())}"
              f"  inserted={len(added)}  replaced={len(replaced)}  convection lines="
              f"{len(conv)}  rejectable-tokens={len(hits)}  "
              f"invert-to-Stokes={'OK' if same else 'MISMATCH'}")
        for n, kind, ln in hits[:4]:
            print(f"    [{kind}] line {n}: {ln[:70]}")
        bad += len(hits)
        if not same:
            bad += 1
            for ln in list(difflib.unified_diff(stokes.split("\n"),
                                                invert(ns, re_label, spec).split("\n"),
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
    if args.write:
        probe = probe_text(stokes, widths)
        if not probe.startswith(stokes if stokes.endswith("\n") else stokes + "\n"):
            print("the probe is not a pure append of the shipped text -- refusing")
            return 1
        pfile = case_dir.parent / "C-base_ns_re1" / "probe_syntax.edp"
        pfile.parent.mkdir(parents=True, exist_ok=True)
        pfile.write_text(probe, encoding="utf-8")
        tail = probe[len(stokes):]
        phits = syntax_findings(probe)
        print(f"wrote {pfile.relative_to(case_dir.parents[1])}: the shipped text plus "
              f"{len(tail.splitlines())} appended lines, one construct each, rejectable "
              f"tokens={len(phits)} -- run this before the levels, it costs one Stokes solve")
        bad += len(phits)
    if bad:
        print("REFUSING TO SHIP: the emitted text is either not the shipped file plus a "
              "declared difference set, or it uses a construct v4.9 is known to reject")
        return 1
    print("diff proof OK: each NS file inverts to the shipped Stokes text byte for byte once "
          "the declared inserts and the two declared replacements are undone; the physics "
          "difference is one convective line.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
