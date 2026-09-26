#!/usr/bin/env python3
"""T6-Re deliverable 2: rebuild the NS truth CSV at >=10 significant digits, and prove the
printing floor with controls that must go red.

Why this exists: FreeFEM v4.9 writes six significant digits and has no `setprecision`
(measured on the instance: "The Identifier setprecision does not exist"), so precision has
to be recovered on the writing side.  Each .edp therefore also writes `<case>_raw_pair.csv`
carrying, per field, `hi = floor(x*1eN)/1eN` and `lo = x - hi` for a field-specific N, and
this file adds them back.

THE RULE, AND WHY N IS PER FIELD.  hi is printed by a six-significant-digit printer, so hi
survives exactly only while its integer numerator stays below 10**6:

    (budget)   N < PRINT_DIGITS - log10(scale)
    (accuracy) N >= -log10(2 * FLOOR_LIMIT_REL * scale) - PRINT_DIGITS

The first says "hi must be printable", the second says "lo must carry enough digits"; a
value of |x| < 1e-N is printed with relative error 5e-7, i.e. absolute 0.5e-(N+6).  The gap
between the two bounds is `2*PRINT_DIGITS + log10(2*FLOOR_LIMIT_REL)` ~ 2.3 integers and is
independent of the scale, so a feasible N always exists -- but *where* it sits moves with
the scale.  In this case the three fields span 0.0476 (v) to 415 (p), so a single N cannot
serve them all; that is not a preference, it is what the two inequalities say, and
`hi_decimals_for()` is the only place they are written down.

NO CONSTANT IS DUPLICATED IN THIS FILE.  An earlier revision of it carried `hi = 1e-4` in a
docstring while the .edp emitted something else, and the assertion passed on both.  So
`--selfcheck` now *parses the emitted .edp text*, takes N from there, replays exactly that
rule, and separately re-derives N from the measured field magnitudes; rule, artefact and
replay must agree.  `finalize --level L` refuses to merge a level whose .edp is missing,
because the meaning of a companion CSV is fixed by the file that wrote it.

WHAT THE CONTROLS ARE FOR.  `--selfcheck` deliberately breaks the rule in three ways (lo at
4 digits, N one step coarser, and the pre-fix single global-N scheme) and requires each to
exceed the bound.  An assertion that cannot fail is not a gate, and the third control is
this file's own bug kept as a fixture: with one N and a global denominator the old check
reported 1.2e-11 and passed, while v_star actually carried ~7.5 significant digits.

Usage:
    python3 finalize_ns_truth.py --selfcheck
    python3 finalize_ns_truth.py --level 1            # merge into *_raw_10dig.csv
    python3 finalize_ns_truth.py --level 1 --dry-run
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFD = HERE.parents[0] / "cases" / "contraction_2d" / "cfd"
STOKES_CSV = CFD / "C-base" / "C-base_raw.csv"
FIELDS = ("u_star", "v_star", "p_star")
LETTER = {"u_star": "u", "v_star": "v", "p_star": "p"}
PRINT_DIGITS = 6             # measured on v4.9: `ofstream << x` gives six significant digits
FLOOR_LIMIT_REL = 1.0e-10    # the work order's ">= 10 significant digits", per field
LEVELS = ("1e-3", "1", "10", "50")

# floor(hu*1.0e4)/1.0e4 as it appears in the .edp.  Group 2 is the multiplier and group 4
# the divisor, and they must be equal -- that is checked, not assumed.
HI_RE = re.compile(r"floor\(\s*h([uvp])\s*\*\s*(\d+(?:\.\d+)?)e(\d+)\s*\)\s*/\s*"
                   r"(\d+(?:\.\d+)?)e(\d+)")


def printed(value: float, digits: int = PRINT_DIGITS) -> float:
    """What FreeFEM's `ofstream << x` puts in the file, read back as a double."""
    return float("%.*g" % (digits, value))


def hi_decimals_for(scale: float) -> int:
    """Smallest N that meets the accuracy bound, kept honest by the budget bound.

    See the module docstring: budget N < 6 - log10(scale), accuracy
    N >= -log10(2 * FLOOR_LIMIT_REL * scale) - 6.  Raises if no integer satisfies both,
    which is the shape a field whose scale is not measurable would produce.
    """
    if not scale > 0.0 or not math.isfinite(scale):
        raise ValueError(f"no measurable scale for this field: {scale!r}")
    n_acc = -math.log10(2.0 * FLOOR_LIMIT_REL * scale) - PRINT_DIGITS
    n_budget = PRINT_DIGITS - math.log10(scale)
    n = math.ceil(n_acc - 1e-12)
    if not n < n_budget:
        raise ValueError(f"scale {scale:.6g}: needs N >= {n_acc:.2f} for the floor but "
                         f"N < {n_budget:.2f} to stay printable -- no feasible N")
    return n


def budget_scale(N: int) -> float:
    """|field| must stay below this or hi itself loses digits: 10**(6-N)."""
    return 10.0 ** (PRINT_DIGITS - N)


def split(value: float, decimals: int) -> tuple[float, float]:
    """The (hi, lo) pair the .edp writes, computed the way that file computes it."""
    m = 10.0 ** decimals
    hi = math.floor(value * m) / m
    return hi, value - hi


def reconstruct(hi: float, lo: float) -> float:
    return hi + lo


def roundtrip(value: float, decimals: int, lo_digits: int = PRINT_DIGITS,
              hi_digits: int = PRINT_DIGITS) -> float:
    hi, lo = split(value, decimals)
    return reconstruct(printed(hi, hi_digits), printed(lo, lo_digits))


def field_magnitudes(limit: int = 4000) -> dict[str, list[float]]:
    """Per-field values of the shipped Stokes truth -- the Re->0 limit, so the same orders
    of magnitude the NS levels will have."""
    if not STOKES_CSV.is_file():
        raise FileNotFoundError(str(STOKES_CSV))
    out: dict[str, list[float]] = {f: [] for f in FIELDS}
    with STOKES_CSV.open(encoding="utf-8", newline="") as fh:
        header = fh.readline().strip().split(",")
        idx = {n: header.index(n) for n in FIELDS}
        for n, line in enumerate(fh):
            if n >= limit:
                break
            parts = line.strip().split(",")
            if len(parts) <= max(idx.values()):
                continue
            for f, i in idx.items():
                try:
                    out[f].append(float(parts[i]))
                except ValueError:
                    pass
    for f in FIELDS:
        if not out[f]:
            raise ValueError(f"{STOKES_CSV.name}: column {f} has no readable values, so no "
                             f"N can be derived -- refusing to guess one")
    return out


def within_rounding_cell(values: list[float], draws: int = 3) -> list[float]:
    """Full-precision surrogates for what the true field values looked like.

    The shipped file has already lost everything past six digits, so replaying the rule on
    the printed values proves nothing: they round-trip exactly at almost any N.  A true
    value is anywhere in the half-open rounding cell of its printed form, so each printed
    value is expanded into `draws` deterministic samples across that cell.  The seed and the
    order are fixed, which is what makes the numbers in the cost table reproducible.
    """
    out: list[float] = []
    for k, v in enumerate(values):
        if v == 0.0:
            out.append(0.0)
            continue
        ulp = 10.0 ** (math.floor(math.log10(abs(v))) - (PRINT_DIGITS - 1))
        for j in range(draws):
            t = (((k * 2654435761 + j * 40503) % 1000003) / 1000003.0) - 0.5
            out.append(v + t * ulp)
    return out


def edp_path(level: str) -> Path:
    return CFD / f"C-base_ns_re{level}" / f"C-base_ns_re{level}.edp"


def _git_blob_id(path: Path) -> str:
    """The git blob id of a file, if this checkout is a git repo and git is available.

    Diagnostic only -- used to tell "stale working-tree artefact" from "wrong method".  It
    never gates: no git, or the file outside a repo, prints `not-a-git-object`.
    """
    import subprocess
    try:
        out = subprocess.run(["git", "hash-object", str(path)],
                             capture_output=True, text=True, timeout=10)
    except Exception:                                          # noqa: BLE001
        return "not-a-git-object"
    return out.stdout.strip()[:16] if out.returncode == 0 else "not-a-git-object"


def hi_decimals_from_edp(text: str) -> dict[str, int]:
    """Read the emitted rule back: field -> decimals, from the .edp text itself."""
    found: dict[str, int] = {}
    for letter, m1, e1, m2, e2 in HI_RE.findall(text):
        if float(m1) != float(m2) or e1 != e2:
            raise ValueError(f"hi/lo rule is self-inconsistent: floor(h{letter}*{m1}e{e1})"
                             f"/{m2}e{e2} -- the multiplier and the divisor must be equal")
        if float(m1) != 1.0:
            raise ValueError(f"unexpected mantissa {m1}e{e1}: the derived rule only emits "
                             f"1eN, and accepting anything else would let a silent edit "
                             f"through the gate")
        n = int(e1)
        field = next(f for f in FIELDS if LETTER[f] == letter)
        if field in found and found[field] != n:
            raise ValueError(f"{field} is split at two different widths in one file "
                             f"({found[field]} and {n})")
        found[field] = n
    missing = [f for f in FIELDS if f not in found]
    if missing:
        raise ValueError("no floor() hi line for " + ",".join(missing) +
                         " -- this .edp does not carry the companion stream this file reads")
    return found


def load_emitted_rule(levels=LEVELS) -> tuple[dict[str, int], list[str]]:
    """The rule as four files emit it, plus the levels that could not be read."""
    rule: dict[str, int] = {}
    unreadable: list[str] = []
    for level in levels:
        p = edp_path(level)
        if not p.is_file():
            unreadable.append(f"{p.name} absent")
            continue
        got = hi_decimals_from_edp(p.read_text(encoding="utf-8").replace("\r\n", "\n"))
        if rule and got != rule:
            raise ValueError(f"levels disagree on the hi width: {rule} vs {got} in "
                             f"{p.name}; four levels whose digits are not comparable cannot "
                             f"be put in one cost table")
        rule = got
    return rule, unreadable


def selfcheck(levels=LEVELS) -> int:
    rc = 0
    print(f"the assertion below is bound to the emitted .edp text, not to a constant in "
          f"this file (PRINT_DIGITS={PRINT_DIGITS}, FLOOR_LIMIT_REL={FLOOR_LIMIT_REL:.0e} "
          f"per field)")
    try:
        mags = field_magnitudes()
    except FileNotFoundError as exc:
        print(f"[INDETERMINATE] no magnitudes to bind the rule to: {exc}")
        return 2
    scale = {f: max(abs(v) for v in mags[f]) for f in FIELDS}
    rule, unreadable = load_emitted_rule(levels)
    if not rule:
        print("[FAIL] none of the four .edp files is present, so there is no emitted rule "
              "to check: " + "; ".join(unreadable))
        return 1
    for note in unreadable:
        print(f"  note: {note} (checked against the levels that are present)")
    print(f"emitted hi widths (read from the .edp): " +
          ", ".join(f"{f}=1e{rule[f]}" for f in FIELDS))
    print(f"measured |max| from {STOKES_CSV.name}: " +
          ", ".join(f"{f}={scale[f]:.6g}" for f in FIELDS))

    # 1. artefact vs rule: what the .edp emits must be what hi_decimals_for() derives.
    # A mismatch here has one known cause in practice: the scripts were re-pulled but the
    # .edp files in the working tree are still the previous pin's, so the widths on disk
    # predate the per-field derivation.  Say that, with the path and the blob id, because
    # the alternative reading -- "the tightened assertion caught a bad method" -- is what
    # this output looks like and it is wrong (the 2026-09-26 19:54 instance run was exactly
    # this case; it reproduced locally against 63a9309's .edp to the last digit).
    mismatch = [f for f in FIELDS if rule[f] != hi_decimals_for(scale[f])]
    if mismatch:
        print("  [HINT] the .edp on disk disagrees with the rule this script derives from "
              "the measured magnitudes. Before reading that as a method failure, check that "
              "the case files came from the same pin as the scripts:")
        for level in levels:
            p = edp_path(level)
            if p.is_file():
                print(f"    {p}  blob={_git_blob_id(p)}")
    for f in FIELDS:
        want = hi_decimals_for(scale[f])
        ok = rule[f] == want
        rc |= 0 if ok else 1
        head = budget_scale(want) / max(scale[f], 1e-300)
        print(f"[{'PASS' if ok else 'FAIL'}] {f}: emitted N={rule[f]}, derived N={want} "
              f"(with the derived N, |{f}| may grow to {budget_scale(want):.6g} before hi "
              f"loses digits = {head:.1f}x headroom over the Stokes scale)")

    # 2. the floor itself, replayed through the emitted rule on full-precision surrogates.
    # Two columns are published per field, per ruling (a) of the 2026-09-26 19:54 work-order
    # message: relative to that field's own max, AND absolute.  The gate is the relative one
    # at FLOOR_LIMIT_REL, which is the number the work order wrote; what changed on
    # 2026-09-26 is the DENOMINATOR (global max -> per-field max), and that change was
    # ordered, not chosen by me -- it makes the assertion strictly harder, and it is the
    # reason v_star went from "passing" to 1.05e-07.
    print("  floor per field (both columns reported; gate = relative <= "
          f"{FLOOR_LIMIT_REL:.0e}, denominator = that field's own max, per ruling (a) "
          "of 2026-09-26 19:54 by the orchestrator):")
    worst = {}
    for f in FIELDS:
        xs = within_rounding_cell(mags[f])
        err = max(abs(roundtrip(x, rule[f]) - x) for x in xs)
        worst[f] = err / max(scale[f], 1e-300)
        digits = -math.log10(worst[f]) if worst[f] > 0 else math.inf
        print(f"    {f:7s} |max|={scale[f]:9.5g}  N={rule[f]}  abs<={err:8.2e}  "
              f"rel<={worst[f]:8.2e}  = {digits:4.1f} significant digits")
    limit_ok = all(worst[f] <= FLOOR_LIMIT_REL for f in FIELDS)
    rc |= 0 if limit_ok else 1
    print(f"[{'PASS' if limit_ok else 'FAIL'}] per-field floor: " +
          ", ".join(f"{f} {worst[f]:.2e}" for f in FIELDS) +
          f" <= {FLOOR_LIMIT_REL:.0e} of each field's own scale")
    print(f"[INFO] same numbers under the OLD global denominator (|max| over all fields = "
          f"{max(scale.values()):.6g}): " +
          ", ".join(f"{f} {worst[f] * scale[f] / max(scale.values()):.2e}" for f in FIELDS)
          + " -- that is the figure that made a single width look adequate")

    # 3. controls: each one must be RED, or the assertion above proves nothing.
    ctrl_lo = {f: max(abs(roundtrip(x, rule[f], lo_digits=4) - x) for x in within_rounding_cell(mags[f]))
               / max(scale[f], 1e-300) for f in FIELDS}
    ctrl_n = {f: max(abs(roundtrip(x, rule[f] - 1) - x) for x in within_rounding_cell(mags[f]))
              / max(scale[f], 1e-300) for f in FIELDS}
    single_n = min(rule[f] for f in FIELDS)
    ctrl_glob = {f: max(abs(roundtrip(x, single_n) - x) for x in within_rounding_cell(mags[f]))
                 / max(scale[f], 1e-300) for f in FIELDS}
    old_style = {f: max(abs(roundtrip(x, single_n) - x) for x in within_rounding_cell(mags[f]))
                 / max(max(scale.values()), 1e-300) for f in FIELDS}
    for name, got, why in (
            ("lo printed at 4 digits", ctrl_lo, "one digit fewer of lo must break the floor"),
            ("N one step coarser", ctrl_n, "the emitted constant itself must be load-bearing"),
            ("one N for all three fields", ctrl_glob,
             "the scheme this file shipped before: per-field it misses the bound")):
        bad = max(got.values()) > FLOOR_LIMIT_REL
        rc |= 0 if bad else 1
        print(f"[{'PASS' if bad else 'FAIL'}] control ({name}) is red as required: worst "
              f"{max(got.values()):.2e} -- {why}")
    print(f"  what the old check saw: a single N={single_n} under a global denominator "
          f"reports {max(old_style.values()):.2e} and passes; per field the same scheme "
          f"gives " + ", ".join(f"{f} {ctrl_glob[f]:.2e}" for f in FIELDS) +
          f".  The denominator, not the data, is what made it green.")

    # Coverage guards.  `wide` is the premise of the whole per-field design: if the three
    # fields did not straddle 1.0, a single width would do and this file would be over-built.
    wide = max(scale.values()) > 100.0 and min(scale.values()) < 1.0
    enough = all(len(mags[f]) >= 500 for f in FIELDS)
    rc |= 0 if (wide and enough) else 1
    print(f"[{'PASS' if wide else 'FAIL'}] the fields straddle 1.0 "
          f"(|max| from {min(scale.values()):.6g} to {max(scale.values()):.6g}), which is "
          f"why one width cannot serve all three, and the largest is big enough that 1e-10 "
          f"absolute cannot masquerade as 10 digits")
    print(f"[{'PASS' if enough else 'FAIL'}] at least 500 values per field are replayed "
          f"({min(len(mags[f]) for f in FIELDS)})")
    return rc


def read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        header = fh.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    return header, rows


def finalize(level: str, dry_run: bool) -> int:
    d = CFD / f"C-base_ns_re{level}"
    raw, pair = d / f"C-base_ns_re{level}_raw.csv", d / f"C-base_ns_re{level}_raw_pair.csv"
    edp = edp_path(level)
    missing = [str(p) for p in (raw, pair, edp) if not p.is_file()]
    if missing:
        print("[AWAITING INSTANCE RUN] missing: " + ", ".join(missing))
        print("this is not a pass: the level's truth CSV does not exist yet.  The .edp is "
              "among the required files because the hi width it emitted is what fixes the "
              "meaning of the companion CSV -- guessing it here would merge two different "
              "conventions and print a number either way.")
        return 3
    rule = hi_decimals_from_edp(edp.read_text(encoding="utf-8").replace("\r\n", "\n"))
    h_raw, r_raw = read_rows(raw)
    h_pair, r_pair = read_rows(pair)
    # Rows must match; columns must NOT (the companion stream has six hi/lo columns and no
    # bc_tag).  An earlier revision compared len(h_raw) with len(h_pair) in the same
    # condition, so every well-formed level was refused with a "row-count mismatch" message
    # whose two numbers were equal -- selftest_ns_re.py is what caught it.
    if len(r_raw) != len(r_pair):
        print(f"row-count mismatch raw={len(r_raw)} pair={len(r_pair)} -- the two streams "
              f"did not come from the same vertex order; refusing to merge")
        return 1
    ip = {n: i for i, n in enumerate(h_pair)}
    ir = {n: i for i, n in enumerate(h_raw)}
    need_raw = ["x_star", "y_star", *FIELDS, "bc_tag"]
    need_pair = ["x_star", "y_star"] + [LETTER[f] + "_" + s for f in FIELDS for s in ("hi", "lo")]
    miss = ([f"raw:{c}" for c in need_raw if c not in ir]
            + [f"pair:{c}" for c in need_pair if c not in ip])
    if miss:
        print(f"the written headers do not carry the columns this merge reads: {miss}. "
              f"raw={h_raw} pair={h_pair} -- refusing rather than merging by position")
        return 1
    field_scale = {f: max(abs(float(r[ir[f]])) for r in r_raw) for f in FIELDS}

    # The precondition of a chosen N is a magnitude bound, and it is measured here, on the
    # artefact, rather than assumed from the Stokes file.  Note what is *not* checked: a
    # lattice test on the printed hi cannot detect truncation, because rounding a multiple of
    # 1e-N to six digits yields a value that is still a multiple of 1e-N -- an earlier
    # revision of this function had such a test and it could never have fired.  What does
    # detect it is the magnitude bound below and the per-field drift band further down.
    over = []
    for f in FIELDS:
        if field_scale[f] >= budget_scale(rule[f]):
            over.append(f"{f}: |{f}|_max = {field_scale[f]:.6g} is at or above the "
                        f"{budget_scale(rule[f]):.6g} a width of N={rule[f]} can still print "
                        f"in six digits, so hi lost digits in the .edp itself -> re-emit this "
                        f"level with N={hi_decimals_for(field_scale[f])} and re-solve")
    if over:
        print("[FAIL] the six-digit printer could not carry `hi` for this level, so merging "
              "cannot recover the digits:")
        for line in over:
            print("   " + line)
        return 1

    drift = {f: 0.0 for f in FIELDS}
    merged = []
    for row_r, row_p in zip(r_raw, r_pair):
        if row_r[ir["x_star"]] != row_p[ip["x_star"]] or row_r[ir["y_star"]] != row_p[ip["y_star"]]:
            print(f"join key mismatch at a row: raw=({row_r[0]},{row_r[1]}) "
                  f"pair=({row_p[0]},{row_p[1]}) -- refusing")
            return 1
        new = list(row_r)
        for f in FIELDS:
            hi, lo = LETTER[f] + "_hi", LETTER[f] + "_lo"
            val = reconstruct(float(row_p[ip[hi]]), float(row_p[ip[lo]]))
            if field_scale[f] > 0:
                drift[f] = max(drift[f], abs(val - float(row_r[ir[f]])) / field_scale[f])
            new[ir[f]] = repr(val)
        merged.append(new)
    worst = max(drift.values())
    print(f"level Re={level}: {len(merged)} rows merged with the widths emitted by "
          f"{edp.name} (" + ", ".join(f"{f}:N={rule[f]}" for f in FIELDS) + "); shift away "
          f"from the 6-digit file, per field: " +
          ", ".join(f"{f} {drift[f]:.2e}" for f in FIELDS) + " (expected ~5e-7)")
    # Per field, not pooled: a global denominator is what let a single-width scheme look
    # green in the first revision of this file while v_star carried 7 digits.
    bad = [f for f in FIELDS if field_scale[f] > 0 and not (1.0e-9 < drift[f] < 1.0e-4)]
    if bad:
        print(f"the merge moved {','.join(bad)} by an amount inconsistent with a 6-digit "
              f"print; check the hi/lo convention (widths read from the .edp: {rule}) before "
              f"trusting the file")
        return 1
    if dry_run:
        print("[dry-run] nothing written")
        return 0
    out = d / f"C-base_ns_re{level}_raw_10dig.csv"
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(h_raw) + "\n")
        for r in merged:
            fh.write(",".join(r) + "\n")
    print(f"wrote {out.relative_to(CFD.parents[2])}  "
          f"(the 6-digit *_raw.csv is kept: it is what the .edp wrote, and the two must be "
          f"reconcilable, not overwritten)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="merge the NS hi/lo stream and assert the floor")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--level", choices=LEVELS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.selfcheck or not args.level:
        rc = selfcheck()
        if not args.level:
            return rc
        return rc or finalize(args.level or "1", args.dry_run)
    return finalize(args.level, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
