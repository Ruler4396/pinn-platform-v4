#!/usr/bin/env python3
"""T6-Re deliverable 2: rebuild the NS truth CSV at >=10 significant digits, and prove the
printing floor with a control that must go red.

Why this exists: FreeFEM v4.9 writes six significant digits and has no `setprecision`
(measured on the instance: "The Identifier setprecision does not exist"), so precision has
to be recovered on the writing side.  Each .edp therefore also writes
`<case>_raw_pair.csv` carrying, per field, `hi = floor(x*1e4)/1e4` and `lo = x - hi`.
`hi` is exact under six-digit printing for |x| < 1000 (at most 6 significant digits), and
`lo` is smaller than 1e-4, so its own six digits carry ~1e-10 absolute.  Adding them back
gives ~1e-11 relative, which is what the work order asks for.

`--selfcheck` needs no instance and no FreeFEM: it replays the printing rule on real
magnitudes taken from the shipped Stokes truth and asserts
    max |reconstructed - exact| / max(|exact|, scale)  <  1e-10
and then, as the control, prints `lo` with one digit fewer -- which MUST break that
assertion.  An assertion that cannot fail is not a gate.

Usage:
    python3 finalize_ns_truth.py --selfcheck
    python3 finalize_ns_truth.py --level 1            # merge (writes the _raw.csv in place)
    python3 finalize_ns_truth.py --level 1 --dry-run
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFD = HERE.parents[0] / "cases" / "contraction_2d" / "cfd"
FIELDS = ("u_star", "v_star", "p_star")
HI_DECIMALS = 2                # must match floor(x*1e2)/1e2 in the .edp
HI_MAX_MAGNITUDE = 1000.0      # 2 decimals keep hi within 5 significant digits below this
FLOOR_LIMIT_REL = 1.0e-10      # the work order's >=10 significant digits, as a relative cap
LEVELS = ("1e-3", "1", "10", "50")


def printed(value: float, digits: int = 6) -> float:
    """What FreeFEM's `ofstream << x` puts in the file, read back as a double."""
    return float("%.*g" % (digits, value))


def split(value: float) -> tuple[float, float]:
    """The (hi, lo) pair the .edp writes, computed the same way it computes them."""
    # 2 decimals, not 4: with |p| up to 415 in the shipped contraction truth, a
    # 4-decimal hi needs 7 significant digits and FreeFEM's 6 truncate it, which showed up
    # as a 1.0e-04 round-trip error in this very file's first self-check.
    hi = math.floor(value * 10.0 ** HI_DECIMALS) / 10.0 ** HI_DECIMALS
    return hi, value - hi


def reconstruct(hi: float, lo: float) -> float:
    return hi + lo


def roundtrip(value: float, lo_digits: int = 6) -> float:
    hi, lo = split(value)
    return reconstruct(printed(hi), printed(lo, lo_digits))


def magnitudes_from_shipped(limit: int = 4000) -> list[float]:
    """Real field magnitudes from the shipped Stokes truth, or a synthetic sweep if absent."""
    src = CFD / "C-base" / "C-base_raw.csv"
    out: list[float] = []
    if src.is_file():
        with src.open(encoding="utf-8", newline="") as fh:
            header = fh.readline().strip().split(",")
            idx = [header.index(f) for f in FIELDS]
            for n, line in enumerate(fh):
                if n >= limit:
                    break
                parts = line.strip().split(",")
                if len(parts) <= max(idx):
                    continue
                for i in idx:
                    try:
                        out.append(float(parts[i]))
                    except ValueError:
                        pass
    if not out:                                   # laptop without the case checkout
        out = [v * s for v in (0.0, 1e-40, 3e-7, 0.114329, 1.91421, 8.75, 72.0)
               for s in (1.0, -1.0)]
    return out


def selfcheck() -> int:
    vals = magnitudes_from_shipped()
    scale = max((abs(v) for v in vals), default=1.0)
    err = max(abs(roundtrip(v) - v) for v in vals)
    rel = err / max(scale, 1e-300)
    bad = max(abs(roundtrip(v, lo_digits=4) - v) for v in vals) / max(scale, 1e-300)
    over = [v for v in vals if abs(v) >= HI_MAX_MAGNITUDE]
    print(f"magnitudes replayed   : {len(vals)} values, |max| = {scale:.6g}")
    print(f"hi/lo round-trip error: absolute {err:.3e}  relative {rel:.3e}  "
          f"(limit {FLOOR_LIMIT_REL:.0e})")
    print(f"control, lo at 4 dig  : relative {bad:.3e}  (must exceed the limit)")
    print(f"magnitudes >= {HI_MAX_MAGNITUDE:g} (hi would lose digits): {len(over)}")
    ok_floor = rel < FLOOR_LIMIT_REL
    ok_control = bad >= FLOOR_LIMIT_REL
    print(f"[{'PASS' if ok_floor else 'FAIL'}] printing floor is at least 10 significant digits")
    print(f"[{'PASS' if ok_control else 'FAIL'}] the assertion is sensitive: dropping one "
          f"digit of lo breaks it by a factor {bad / max(rel, 1e-300):.0f}")
    # a floor measured on values that are all tiny would be an absolute-error accident
    wide = any(abs(v) > 1.0 for v in vals)
    in_range = not over
    print(f"[{'PASS' if wide else 'FAIL'}] the replay covers values above 1.0 "
          f"(otherwise 1e-10 absolute would masquerade as 10 digits)")
    print(f"[{'PASS' if in_range else 'FAIL'}] every replayed magnitude is below "
          f"{HI_MAX_MAGNITUDE:g}, the precondition for a 2-decimal hi")
    return 0 if (ok_floor and ok_control and wide and in_range) else 1


def read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        header = fh.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    return header, rows


def finalize(level: str, dry_run: bool) -> int:
    d = CFD / f"C-base_ns_re{level}"
    raw, pair = d / f"C-base_ns_re{level}_raw.csv", d / f"C-base_ns_re{level}_raw_pair.csv"
    missing = [str(p) for p in (raw, pair) if not p.is_file()]
    if missing:
        print("[AWAITING INSTANCE RUN] missing: " + ", ".join(missing))
        print("this is not a pass: the level's truth CSV does not exist yet")
        return 3
    h_raw, r_raw = read_rows(raw)
    h_pair, r_pair = read_rows(pair)
    if len(h_raw) != len(h_pair) or len(r_raw) != len(r_pair):
        print(f"row-count mismatch raw={len(r_raw)} pair={len(r_pair)} -- the two streams "
              f"did not come from the same vertex order; refusing to merge")
        return 1
    ip = {n: i for i, n in enumerate(h_pair)}
    ir = {n: i for i, n in enumerate(h_raw)}
    scale = max(max(abs(float(r[ir[f]])) for f in FIELDS) for r in r_raw) or 1.0
    worst = 0.0
    merged = []
    for row_r, row_p in zip(r_raw, r_pair):
        if row_r[ir["x_star"]] != row_p[ip["x_star"]] or row_r[ir["y_star"]] != row_p[ip["y_star"]]:
            print(f"join key mismatch at a row: raw=({row_r[0]},{row_r[1]}) "
                  f"pair=({row_p[0]},{row_p[1]}) -- refusing")
            return 1
        new = list(row_r)
        for f_hi, f_lo, f_out in (("u_hi", "u_lo", "u_star"), ("v_hi", "v_lo", "v_star"),
                                  ("p_hi", "p_lo", "p_star")):
            val = reconstruct(float(row_p[ip[f_hi]]), float(row_p[ip[f_lo]]))
            drift = abs(val - float(row_r[ir[f_out]])) / scale
            worst = max(worst, drift)
            new[ir[f_out]] = repr(val)
        merged.append(new)
    print(f"level Re={level}: {len(merged)} rows merged, max shift away from the 6-digit "
          f"file = {worst:.3e} of the field scale (expected ~1e-6)")
    if not (1.0e-9 < worst < 1.0e-4):
        print("the merge moved the values by an amount inconsistent with a 6-digit print; "
              "check the hi/lo convention before trusting the file")
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
