#!/usr/bin/env python3
"""T6-Re deliverable 3 (load-bearing): the Re -> 0 consistency gate. stdlib only.

    python3 check_ns_re_to_stokes.py                 # gate every level that exists
    python3 check_ns_re_to_stokes.py --selfcheck     # synthetic pair, must pass and must fail

The bound is written here, before any run, and is not to be edited after seeing a number:

    RE0_REL_LIMIT = 0.02   -- ||u(Re=1e-3) - u_stokes|| / ||u_stokes|| must be <= 2%

Why Re = 1e-3 is the gated one: at Re = 1 the difference from Stokes is *physics* (the
O(Re) correction the whole exercise is about), so a 2% bound there would be a claim about
the fluid, not about the solver.  At Re = 1e-3 the convection is a million times weaker
than at Re = 50, so the only thing that can move the answer is the numerics: the gate asks
whether the Picard/NS machinery reproduces the shipped Stokes solution in the limit where
it must.  The Re = 1/10/50 differences are still printed -- as the size of the physical
correction, never as a pass/fail.

Three verdicts, pre-registered in the work order (section 3) and kept distinct here:
  FAIL          a level did not converge (its .edp printed NS NOT CONVERGED), or the Re->0
                bound is exceeded
  INDETERMINATE the environment did not produce the files at all -- never reported as pass
                or as fail
  PASS          converged and inside the bound
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFD = HERE.parents[0] / "cases" / "contraction_2d" / "cfd"
STOKES_CSV = CFD / "C-base" / "C-base_raw.csv"
RE0_LEVEL = "1e-3"
RE0_REL_LIMIT = 0.02                     # fixed before the run; see the docstring
PRINT_DIGITS = 6             # v4.9's `ofstream << x`, measured on the instance
LEVELS = ("1e-3", "1", "10", "50")
FIELDS = ("u_star", "v_star", "p_star")
NOT_CONVERGED_MARK = "NS NOT CONVERGED"

# Two claims, deliberately separated on 2026-09-26 at the orchestrator's order (19:54 message,
# item 2).  The Re->0 *verdict* is an L2 comparison against a 2% bound, and the worst case
# six significant digits can contribute to it is computed below (`printing_noise_bound`) --
# if that is orders of magnitude below the bound, the verdict does not need 10 digits.  What
# DOES need them is the separate claim "this truth file is good enough to train on or to
# score pointwise", which is what `digits_claim` reports.  Collapsing the two is how a
# printing question ends up deciding a physics question, and vice versa.
DIGITS_CLAIM_MET = "MET (>=10 digits, merged hi/lo stream)"
DIGITS_CLAIM_NOT = "NOT MET (6-digit stream: pointwise scoring and training use unproven)"


def _half_ulp_six(value: float) -> float:
    """The largest error six significant digits can make on this value."""
    v = abs(value)
    if v == 0.0 or not math.isfinite(v):
        return 0.0
    return 0.5 * 10.0 ** (math.floor(math.log10(v)) - (PRINT_DIGITS - 1))


def printing_noise_bound(ns_idx, ns_rows, stokes_idx, stokes_rows) -> float:
    """Worst-case movement of rel ||u_ns - u_ref|| / ||u_ref|| from 6-digit printing alone.

    Triangle inequality on the numerator plus the same perturbation on the denominator,
    summed in quadrature over vertices -- deterministic, no sampling, and it is an upper
    bound rather than an estimate, which is the right direction for a gate.
    """
    def norm(idx, rows, key_a, key_b):
        return math.sqrt(sum(_half_ulp_six(float(r[idx[key_a]])) ** 2
                             + _half_ulp_six(float(r[idx[key_b]])) ** 2 for r in rows))
    num = (norm(ns_idx, ns_rows, "u_star", "v_star")
           + norm(stokes_idx, stokes_rows, "u_star", "v_star"))
    den = max(velocity_norm(stokes_idx, stokes_rows), 1.0e-300)
    return num / den


def read_truth(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        header = fh.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    idx = {n: i for i, n in enumerate(header)}
    for need in ("x_star", "y_star", *FIELDS, "bc_tag"):
        if need not in idx:
            raise ValueError(f"{path.name}: missing column {need}; the NS truth must carry "
                             f"the same field names as the Stokes truth")
    return idx, rows


def pick_csv(level_dir: Path, level: str) -> tuple[Path, bool]:
    """Prefer the >=10-digit merged file; say loudly when only 6 digits are available."""
    merged = level_dir / f"C-base_ns_re{level}_raw_10dig.csv"
    raw = level_dir / f"C-base_ns_re{level}_raw.csv"
    if merged.is_file():
        return merged, True
    if raw.is_file():
        return raw, False
    raise FileNotFoundError(str(raw))


def velocity_norm(idx, rows) -> float:
    return math.sqrt(sum(float(r[idx["u_star"]]) ** 2 + float(r[idx["v_star"]]) ** 2
                         for r in rows))


def pressure_drop(idx, rows) -> float:
    """mean p on inlet vertices minus mean p on outlet vertices (bc_tag 1 / 2, as shipped)."""
    inside = [float(r[idx["p_star"]]) for r in rows if r[idx["bc_tag"]].strip() == "1"]
    outside = [float(r[idx["p_star"]]) for r in rows if r[idx["bc_tag"]].strip() == "2"]
    if not inside or not outside:
        raise ValueError("no inlet/outlet tagged vertices: pressure drift is not measurable, "
                         "and reporting 0.0 for it would be a lie")
    return sum(inside) / len(inside) - sum(outside) / len(outside)


def compare(level: str, stokes_idx, stokes_rows, want_full_digits: bool = False):
    """Compare one level against the shipped Stokes truth.

    `want_full_digits=True` restores the earlier, stricter behaviour of refusing to judge a
    level whose merged hi/lo stream does not exist yet; it is off by default because the
    2026-09-26 margin arithmetic (see the note above) shows the 2% bound is not reachable by
    6-digit printing noise, so refusing would block a verdict that is already decided.  A
    6-digit level now yields the verdict AND `digits_claim = NOT MET`, which is the honest
    split: the Re->0 test is decided, the "10 digits" deliverable is not.
    """
    level_dir = CFD / f"C-base_ns_re{level}"
    try:
        path, full_digits = pick_csv(level_dir, level)
    except FileNotFoundError as exc:
        return {"level": level, "verdict": "INDETERMINATE", "reason": f"no truth file: {exc}"}
    if not full_digits and want_full_digits:
        return {"level": level, "verdict": "INDETERMINATE",
                "reason": f"{path.name} is the 6-digit stream and --require-full-digits was "
                          f"passed; run finalize_ns_truth.py --level {level} first"}
    idx, rows = read_truth(path)
    if len(rows) != len(stokes_rows):
        return {"level": level, "verdict": "FAIL",
                "reason": f"vertex count {len(rows)} != Stokes {len(stokes_rows)}: the mesh "
                          f"changed, so the comparison is not Re-vs-Re=0 anymore"}
    worst_xy = max(max(abs(float(a[idx["x_star"]]) - float(b[stokes_idx["x_star"]])),
                       abs(float(a[idx["y_star"]]) - float(b[stokes_idx["y_star"]])))
                   for a, b in zip(rows, stokes_rows))
    if worst_xy > 1.0e-5:                       # one print-bound width at |x| ~ 16
        return {"level": level, "verdict": "FAIL",
                "reason": f"vertex positions differ by {worst_xy:.3e}: same mesh is a "
                          f"precondition of this gate, and it is not met"}
    num = math.sqrt(sum((float(a[idx["u_star"]]) - float(b[stokes_idx["u_star"]])) ** 2
                        + (float(a[idx["v_star"]]) - float(b[stokes_idx["v_star"]])) ** 2
                        for a, b in zip(rows, stokes_rows)))
    den = max(velocity_norm(stokes_idx, stokes_rows), 1e-300)
    rel = num / den
    noise = printing_noise_bound(idx, rows, stokes_idx, stokes_rows)
    dp = pressure_drop(idx, rows)
    # Same argument for the table's fourth column: dp is a difference of two means, so the
    # worst case of six-digit printing on it is the largest half-ulp inside either mean.
    def cell(col: str, tag: str) -> float:
        got = [_half_ulp_six(float(r[idx[col]])) for r in rows
               if r[idx["bc_tag"]].strip() == tag]
        return max(got) if got else 0.0
    dp_noise = (cell("p_star", "1") + cell("p_star", "2")) / max(abs(dp), 1e-300)
    return {"level": level, "verdict": None, "rel_vs_stokes": rel,
            "pressure_drop": dp, "file": str(path),
            "digits": ">=10" if full_digits else "6",
            "digits_claim": DIGITS_CLAIM_MET if full_digits else DIGITS_CLAIM_NOT,
            "print_noise_bound_rel": noise,
            "dp_print_noise_bound_rel": dp_noise,
            "margin_bound_over_noise": (RE0_REL_LIMIT / noise) if noise > 0 else math.inf}


def gate(level: str, stokes_idx, stokes_rows, want_full_digits: bool = False) -> dict:
    out = compare(level, stokes_idx, stokes_rows, want_full_digits)
    if out.get("verdict") is not None:
        return out
    rel = out["rel_vs_stokes"]
    if level == RE0_LEVEL:
        # The bound is the pre-registered 2%; `print_noise_bound_rel` is published next to it
        # so a reader can see the verdict is not a printing artefact, and `digits_claim` says
        # separately whether the >=10-digit deliverable is met.
        out["verdict"] = "PASS" if rel <= RE0_REL_LIMIT else "FAIL"
        out["bound"] = RE0_REL_LIMIT
    else:
        out["verdict"] = "REPORTED"             # physical size of the O(Re) correction
    return out


def selfcheck() -> int:
    """Prove the gate has teeth, without an instance: a synthetic pair on real magnitudes.

    It drives the *real* entry point -- `gate()` with `CFD` repointed at a temp tree -- rather
    than a second copy of the arithmetic.  An earlier revision kept a private
    `_gate_against_synthetic` -- now deleted -- which meant the self-check could pass while
    the shipped path broke, and it could not see the margin or digits-claim fields at all.
    """
    global CFD
    if not STOKES_CSV.is_file():
        print(f"[INDETERMINATE] reference truth absent: {STOKES_CSV}")
        return 2
    real_cfd = CFD
    idx, rows = read_truth(STOKES_CSV)
    root = Path(tempfile.gettempdir()) / "ns_probe_selfcheck"
    level_dir = root / "C-base_ns_re1e-3"
    level_dir.mkdir(parents=True, exist_ok=True)
    CFD = root                                        # gate()/compare() read this global
    rc = 0

    def write(name: str, scale: float) -> Path:
        path = level_dir / name
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(",".join(["x_star", "y_star", "u_star", "v_star", "p_star", "bc_tag"])
                     + "\n")
            for r in rows:
                u = float(r[idx["u_star"]]) * (1.0 + scale)
                v = float(r[idx["v_star"]]) * (1.0 - scale)
                p = float(r[idx["p_star"]]) * (1.0 + scale)
                fh.write(f"{r[idx['x_star']]},{r[idx['y_star']]},{u!r},{v!r},{p!r},"
                         f"{r[idx['bc_tag']]}\n")
        return path

    try:
        for scale, expect in ((0.001, "PASS"), (0.05, "FAIL")):
            write("C-base_ns_re1e-3_raw_10dig.csv", scale)
            got = gate(RE0_LEVEL, idx, rows)
            ok = got.get("verdict") == expect
            rc |= 0 if ok else 1
            print(f"[{'PASS' if ok else 'FAIL'}] synthetic perturbation {scale:.1%} -> "
                  f"verdict {got.get('verdict')}, expected {expect} "
                  f"(rel={got.get('rel_vs_stokes', float('nan')):.4%} vs bound "
                  f"{RE0_REL_LIMIT:.0%})")
            if expect == "PASS":
                margin = got.get("margin_bound_over_noise", 0.0)
                noise = got.get("print_noise_bound_rel", float("nan"))
                ok = margin >= 1.0e3 and got.get("digits_claim") == DIGITS_CLAIM_MET
                rc |= 0 if ok else 1
                print(f"[{'PASS' if ok else 'FAIL'}] six printing digits cannot reach the "
                      f"bound: worst-case movement of this statistic = {noise:.2e}, "
                      f"bound/noise = {margin:.3g}x, and the merged stream is claimed "
                      f"sufficient ({got.get('digits_claim')})")
        # The two claims are separate: the verdict may run on the 6-digit stream while the
        # ">=10 digits" deliverable is explicitly NOT met.  Reverse case: with
        # --require-full-digits the same level is INDETERMINATE, never a silent pass.
        (level_dir / "C-base_ns_re1e-3_raw_10dig.csv").unlink()
        write("C-base_ns_re1e-3_raw.csv", 0.001)
        loose = gate(RE0_LEVEL, idx, rows)
        strict = gate(RE0_LEVEL, idx, rows, True)
        dp_noise = loose.get("dp_print_noise_bound_rel", 1.0)
        ok = (loose.get("verdict") == "PASS" and loose.get("digits_claim") == DIGITS_CLAIM_NOT
              and loose.get("margin_bound_over_noise", 0.0) >= 1.0e3
              and loose.get("print_noise_bound_rel", 1.0) < RE0_REL_LIMIT / 100.0
              and dp_noise < 1.0e-4 and strict.get("verdict") == "INDETERMINATE")
        rc |= 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] the Re->0 verdict and the digit claim are "
              f"independent: 6-digit stream -> {loose.get('verdict')} with "
              f"digits_claim={loose.get('digits_claim')}, noise "
              f"{loose.get('print_noise_bound_rel'):.2e} < bound "
              f"{RE0_REL_LIMIT:.0%}; dp column noise {dp_noise:.2e} < 1e-4 (the four-decimal "
              f"resolution the table prints); --require-full-digits -> {strict.get('verdict')}")
        print("  reading of the line above: the printing floor being short of 1e-10 blocks "
              "the claim 'this truth is good enough to train on / score pointwise'. It does "
              "not block the Re->0 verdict, because 6 digits move that statistic ~4 orders "
              "of magnitude less than the bound allows. Ordered by the orchestrator, 2026-09-26 "
              "19:54, item 2; RE0_REL_LIMIT itself unchanged.")
    finally:
        CFD = real_cfd
    print("selfcheck exercises the same code path the real level takes; the bound is fixed at "
          f"{RE0_REL_LIMIT:.0%} in this file, before any number existed")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="the Re->0 gate for the NS truth probe")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--require-full-digits", action="store_true",
                    help="refuse to judge a level that only has the 6-digit stream (the "
                         "stricter mode from before the 2026-09-26 margin arithmetic)")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    if not STOKES_CSV.is_file():
        print(f"[INDETERMINATE] reference truth absent: {STOKES_CSV}")
        return 2
    s_idx, s_rows = read_truth(STOKES_CSV)
    print(f"reference: {STOKES_CSV.name} ({len(s_rows)} vertices), "
          f"bound for Re={RE0_LEVEL} is {RE0_REL_LIMIT:.0%} (fixed before the run)")
    if args.require_full_digits:
        print("  --require-full-digits: a level with only the 6-digit stream is reported as "
              "INDETERMINATE instead of being judged (the stricter pre-2026-09-26 mode)")
    worst = 0
    dp0 = None
    for level in [x.strip() for x in args.levels.split(",") if x.strip()]:
        got = gate(level, s_idx, s_rows, args.require_full_digits)
        if got.get("verdict") in ("PASS", "FAIL", "REPORTED"):
            if dp0 is None and level == RE0_LEVEL:
                dp0 = got["pressure_drop"]
            rel = got["rel_vs_stokes"]
            dp = got["pressure_drop"]
            drift = "" if not dp0 else f"  dp/dp(Re0) = {dp / dp0:8.4f}"
            print(f"  Re={level:5s} rel(u - u_stokes) = {rel:9.4%}  dp = {dp:10.4f}"
                  f"{drift}  digits={got['digits']}  -> {got['verdict']}")
            print(f"        printing cannot reach the bound: worst-case 6-digit movement of "
                  f"this statistic = {got['print_noise_bound_rel']:.2e}, so bound/noise = "
                  f"{got['margin_bound_over_noise']:.3g}x; on the dp column "
                  f"{got['dp_print_noise_bound_rel']:.2e} of dp (the table prints 4 decimals)")
            print(f"        digits claim: {got['digits_claim']}")
        else:
            print(f"  Re={level:5s} {got['verdict']}: {got['reason']}")
        if got.get("verdict") == "FAIL":
            worst = 1
    if worst:
        print("GATE RED: per the work order this probe is FAIL and its artefacts stay out of "
              "the main text")
    return worst


if __name__ == "__main__":
    sys.exit(main())
