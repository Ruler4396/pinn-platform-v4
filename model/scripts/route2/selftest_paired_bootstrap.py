#!/usr/bin/env python3
"""Self-test for paired_bootstrap.py -- the §4.2 paired-uncertainty gate.  stdlib only.

Run:  python model/scripts/route2/selftest_paired_bootstrap.py
Exit 0 only if every check passes.

The three fixtures the dispatch asked for, and what each one actually claims:

  1. delta-is-zero      e(B) == e(C) on every paired row  ->  no bound is produced,
                      delta_status reads 未验：Δ恒为0.  This doubles as route-2's own
                      "feed the predicate arms that are pointwise identical, it must go red"
                      control, so it has to be a refusal rather than a 0.0.
  2. known distribution 3 rows, deltas (2, 5, 8).  All 3^3 = 27 replicate means are
                      enumerated exactly, so the analytic quantile is known; the Monte Carlo
                      number must equal it.  The tolerance is 0, and it is *justified by two
                      binomial margins* printed by the check (the atom straddles the cut
                      index 100 by 3.1 SD from below and 12 SD from above), so "exact match"
                      is arithmetic, not luck.
  3. null effect        delta is pure noise around a true mean of 0.  The rejection rate
                      (how often the 95% lower bound clears 0) is MEASURED and printed
                      against the nominal 5%.  No check here says the method "passed": the
                      assertion only bounds the rate from above, because a method that
                      called everything significant is fatal while 12% vs 5% is a note.

Plus the shared floor guard: with mean(e(A)) = 0 the floor 0.20 * e(A) is satisfied by every
possible improvement, so the verdict must not be PASS -- that is why this line exists at all.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import paired_bootstrap as pb                            # noqa: E402

SRC = (HERE / "paired_bootstrap.py").read_text(encoding="utf-8")
_IS_WINDOWS = tempfile.gettempdir().startswith(("C:", "D:"))
DEFAULT_OUT = (Path("D:/PINN-restart/.scratch/route2/paired_bootstrap_selftest.json")
               if _IS_WINDOWS else Path(os.environ.get("ROUTE2_OUT", tempfile.gettempdir()))
               / "paired_bootstrap_selftest.json")
DELTA3: Tuple[float, ...] = (2.0, 5.0, 8.0)
NULL_EXPERIMENTS = 200
NULL_ROWS = 12
NULL_REPS = 1000
NULL_BASE_SEED = 20260928
MIN_SD_MARGIN = 2.0


class Check:
    def __init__(self) -> None:
        self.rows: List[dict] = []

    def add(self, name: str, ok: bool, value: object, limit: object = "") -> None:
        self.rows.append({"check": name, "pass": bool(ok), "value": value, "limit": limit})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: value={_fmt(value)} limit={_fmt(limit)}")

    def skip(self, name: str, value: object, limit: object = "") -> None:
        """Counted and printed, never folded into PASS: 'all green' and 'green minus the
        checks that had no input to run on' are different claims."""
        self.rows.append({"check": name, "pass": True, "skipped": True,
                          "value": value, "limit": limit})
        print(f"[SKIP] {name}: value={_fmt(value)} limit={_fmt(limit)}")

    def measure(self, name: str, value: object) -> None:
        print(f"[boot-measure] {name}={_fmt(value)}")

    @property
    def skipped(self) -> List[str]:
        return [r["check"] for r in self.rows if r.get("skipped")]

    @property
    def failed(self) -> List[str]:
        return [r["check"] for r in self.rows if not r["pass"]]


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)


def _raises(fn, exc) -> bool:
    try:
        fn()
        return False
    except exc:
        return True


def _table(deltas: Sequence[float], e_a: float = 1.0) -> List[dict]:
    """Rows whose paired difference is exactly the requested delta.

    e(B) = delta, e(C) = 0, and e(A) is a non-zero constant by default so the floor guard
    stays quiet unless a check is specifically about the floor.
    """
    return [{"obs_seed": i + 1, "A": e_a, "B": d, "C": 0.0} for i, d in enumerate(deltas)]


# --------------------------------------------------------------- fixture 1: delta == 0
def zero_delta_checks(ck: Check) -> None:
    same = [{"obs_seed": i + 1, "A": 1.0, "B": b, "C": b}
            for i, b in enumerate((3.0, 7.0, -2.0, 5.5))]
    res = pb.paired_lower_bound(same)
    ck.add("boot_red1_identical_arms_refuse_instead_of_bound",
           res["lower_bound"] is None and res["delta_status"] == pb.ZERO_DELTA
           and res["cell_verdict"] == pb.UNVERIFIED and bool(res["reasons"]),
           {"lower_bound": res["lower_bound"], "delta_status": res["delta_status"],
            "verdict": res["cell_verdict"]},
           "未验：Δ恒为0 and no number produced")
    line = pb.format_row(res)
    ck.add("boot_red1_printed_row_refuses_to_look_like_a_measurement",
           "lower95=n/a" in line and "verdict=未验" in line and pb.ZERO_DELTA in line,
           line, "printed row: lower95=n/a, verdict=未验, refusal named")
    ck.add("boot_red1_zero_is_not_used_as_the_answer",
           all(res[c] is None for c in ("lower_bound", "upper_check"))
           and res["delta_status"] != pb.OK,
           {"lower_bound": res["lower_bound"], "upper_check": res["upper_check"]},
           "0.0 would read as 'just barely failed'; here there is no answer at all")
    tiny = pb.paired_lower_bound(_table([0.0, 0.0, 1.0e-9]))
    ck.add("boot_red1_an_epsilon_delta_is_not_called_identical",
           tiny["delta_status"] == pb.OK and tiny["lower_bound"] is not None
           and tiny["cell_verdict"] != "PASS",
           {"delta_status": tiny["delta_status"], "lower_bound": tiny["lower_bound"],
            "verdict": tiny["cell_verdict"]},
           "refusal keys on exact zeros; a 1e-9 delta still gets a bound (which 3 rows "
           "cannot push past 0 anyway)")


# ------------------------------------------------------- fixture 2: enumerable distribution
def _enumerate_bound(deltas: Sequence[float], reps: int, alpha: float) -> dict:
    """Exact bootstrap quantile for a tiny table: every n^n replicate, with its weight.

    The Monte Carlo estimator is the (ceil(alpha*R))-th of R sorted draws, so the analytic
    counterpart is the smallest atom whose cumulative probability reaches alpha -- the same
    right-continuous inverse-ECDF, on exact weights instead of counts.  Two binomial
    margins are returned so the "tolerance 0" claim can be checked rather than asserted.
    """
    n = len(deltas)
    counts: Dict[float, int] = {}
    for picks in itertools.product(range(n), repeat=n):
        mean = round(sum(deltas[i] for i in picks) / n, 12)
        counts[mean] = counts.get(mean, 0) + 1
    total = n ** n
    cut = int(math.ceil(alpha * reps))
    cum = 0
    analytic: Optional[float] = None
    cum_below = 0
    for value in sorted(counts):
        if analytic is None and (cum + counts[value]) / total >= alpha:
            analytic = value
            cum_below = cum
            break
        cum += counts[value]
    assert analytic is not None
    at_prob = (cum_below + counts[analytic]) / total
    below_prob = cum_below / total
    sd = lambda p: math.sqrt(reps * p * (1.0 - p))
    margin_below = (cut - reps * below_prob) / sd(below_prob) if below_prob > 0 else float("inf")
    margin_at = (reps * at_prob - cut) / sd(at_prob)
    return {"analytic_bound": analytic, "atoms": sorted(counts),
            "n_resamples": total, "cut_index": cut,
            "expected_count_at_analytic": reps * at_prob,
            "expected_count_below_analytic": reps * below_prob,
            "sd_margin_below": margin_below, "sd_margin_at": margin_at}


def known_distribution_checks(ck: Check) -> None:
    table = _table(DELTA3)
    res = pb.paired_lower_bound(table, reps=pb.DEFAULT_REPS, seed=pb.DEFAULT_SEED)
    en = _enumerate_bound(DELTA3, pb.DEFAULT_REPS, pb.ALPHA)
    ck.add("boot_pos2_mc_bound_equals_enumerated_bound",
           res["lower_bound"] == en["analytic_bound"],
           {"mc": res["lower_bound"], "enumerated": en["analytic_bound"],
            "atoms": en["atoms"], "n_resamples": en["n_resamples"]},
           f"tolerance 0: MC = the {en['cut_index']}-th of 2000 sorted draws, analytic = "
           f"smallest atom whose exact CDF reaches 0.05")
    ck.add("boot_pos2_exactness_is_justified_by_two_margins",
           min(en["sd_margin_below"], en["sd_margin_at"]) >= MIN_SD_MARGIN,
           {"sd_margin_below": en["sd_margin_below"], "sd_margin_at": en["sd_margin_at"],
            "expected_at": en["expected_count_at_analytic"],
            "expected_below": en["expected_count_below_analytic"], "cut": en["cut_index"]},
           f"both >= {MIN_SD_MARGIN} binomial SD from the cut index, else equality is luck")
    ck.add("boot_pos2_observed_statistic_is_the_hand_mean",
           res["delta_obs"] == 5.0 and res["delta_min"] == 2.0 and res["delta_max"] == 8.0,
           {"delta_obs": res["delta_obs"], "min": res["delta_min"], "max": res["delta_max"]},
           "mean(2,5,8) = 5")
    a = pb.paired_lower_bound(table, seed=1234)["lower_bound"]
    b = pb.paired_lower_bound(table, seed=1234)["lower_bound"]
    c = pb.paired_lower_bound(table, seed=1235)["lower_bound"]
    ck.add("boot_pos2_same_seed_replays_bitwise",
           a == b, {"seed1234": a, "seed1234_again": b},
           "the recorded seed is what makes the bound replayable")
    # Stated rather than asserted: on a 3-row table the bootstrap distribution has only 8
    # atoms, so the 5% quantile is the same number for every seed -- seed-insensitivity here
    # is a property of an atomised distribution, not evidence that the seed is ignored.
    # A continuous table is where a different seed must move the bound; checked there.
    cont = _table([0.7, 2.1, -0.4, 1.3, 0.2, 1.9, 0.5, 1.1])
    d1 = pb.paired_lower_bound(cont, seed=11)["lower_bound"]
    d2 = pb.paired_lower_bound(cont, seed=12)["lower_bound"]
    ck.add("boot_pos2_seed_matters_once_the_distribution_is_not_atomised",
           d1 != d2, {"seed11": d1, "seed12": d2, "n_pairs": len(cont)},
           "8 continuous rows: two seeds give two different bounds")
    ck.add("boot_pos2_provenance_is_in_the_result",
           (res["seed"], res["reps"], res["n_pairs"], res["alpha"])
           == (pb.DEFAULT_SEED, pb.DEFAULT_REPS, 3, pb.ALPHA),
           {k: res[k] for k in ("n_pairs", "reps", "seed", "alpha")},
           "(3 pairs, 2000 reps, seed 20260928, alpha 0.05)")
    small = pb.paired_lower_bound(table)          # the 3-row table above
    big = pb.paired_lower_bound(_table([0.7, 2.1, -0.4, 1.3, 0.2, 1.9, 0.5, 1.1]))
    ck.add("boot_pos2_small_sample_atomisation_is_recorded_not_assumed",
           any("caveat" in r for r in small["reasons"])
           and not any("caveat" in r for r in big["reasons"])
           and small["distinct_draws"] == len(en["atoms"]) and big["distinct_draws"] > 100,
           {"n3_distinct": small["distinct_draws"], "n8_distinct": big["distinct_draws"],
            "reps": small["reps"]},
           "the Monte Carlo distinct-mean count must equal the enumerated atom count "
           "(7 at n=3), and the caveat fires only below 5 pairs")
    ck.measure("atomisation", {"distinct_draws_at_n3": small["distinct_draws"],
                               "distinct_draws_at_n8": big["distinct_draws"]})
    ck.measure("enumerated_atoms", {"bound": en["analytic_bound"],
                                    "cut": en["cut_index"],
                                    "exp_at": en["expected_count_at_analytic"],
                                    "exp_below": en["expected_count_below_analytic"]})


# ------------------------------------------------------------------ the floor guard (§4.1)
def floor_guard_checks(ck: Check) -> None:
    dead_floor = _table([9.0, 11.0, 8.0, 10.0, 12.0], e_a=0.0)
    res = pb.paired_lower_bound(dead_floor)
    ck.add("boot_floor_zero_ea_blocks_the_verdict_even_with_a_strong_signal",
           res["floor_status"] == pb.ZERO_FLOOR and res["cell_verdict"] == pb.UNVERIFIED
           and res["lower_bound"] is None,
           {"floor_status": res["floor_status"], "verdict": res["cell_verdict"],
            "lower_bound": res["lower_bound"], "delta_obs": res["delta_obs"]},
           "mean(e(A))=0 => 0.20*e(A)=0 is satisfied by everything => no bound, no PASS")
    ck.add("boot_floor_guard_line_is_printed_on_the_row",
           "floor_status=" in pb.format_row(res) and pb.ZERO_FLOOR in pb.format_row(res),
           pb.format_row(res)[-110:], "the shared guard shows up in the log line")
    live = pb.paired_lower_bound(_table([9.0, 11.0, 8.0, 10.0, 12.0], e_a=1.0))
    ck.add("boot_floor_nonzero_lets_a_real_signal_through",
           live["floor_status"] == pb.OK and live["cell_verdict"] == "PASS"
           and live["lower_bound"] > 0.0,
           {"lower_bound": live["lower_bound"], "verdict": live["cell_verdict"]},
           "live floor + 5 rows around +10 => bound clears 0")
    noisy = pb.paired_lower_bound(_table([9.0, -9.0, 9.0, -9.0, 9.0], e_a=1.0))
    ck.add("boot_no_steady_improvement_does_not_pass",
           noisy["cell_verdict"] == "未通过" and noisy["lower_bound"] <= 0.0,
           {"delta_obs": noisy["delta_obs"], "lower_bound": noisy["lower_bound"],
            "verdict": noisy["cell_verdict"]},
           "a positive mean with 2 of 5 rows negative must not clear a 95% lower bound")
    ck.add("boot_row_without_e_a_is_refused",
           _raises(lambda: pb.paired_lower_bound([{"A": None, "B": 1.0, "C": 0.0},
                                                  {"A": 1.0, "B": 1.0, "C": 0.0}]),
                   pb.BootstrapError),
           "BootstrapError", "a missing e(A) cannot feed the guard, so the row is rejected "
                             "rather than treated as 0")


# ------------------------------------------------------------- fixture 3: null calibration
def null_effect_checks(ck: Check) -> None:
    """Under a true delta of zero, how often does the 95% lower bound clear 0?"""
    outer = random.Random(NULL_BASE_SEED)
    bounds: List[float] = []
    for exp in range(NULL_EXPERIMENTS):
        deltas = [outer.gauss(0.0, 1.0) for _ in range(NULL_ROWS)]
        res = pb.paired_lower_bound(_table(deltas), reps=NULL_REPS,
                                    seed=NULL_BASE_SEED + exp)
        bounds.append(res["lower_bound"])
    rejects = sum(1 for x in bounds if x > 0.0)
    rate = rejects / NULL_EXPERIMENTS
    se = math.sqrt(rate * (1.0 - rate) / NULL_EXPERIMENTS)
    ck.measure(f"null_rejection_rate_M{NULL_EXPERIMENTS}_n{NULL_ROWS}_reps{NULL_REPS}",
               {"measured_rate": rate, "nominal_alpha": pb.ALPHA, "mc_se": se,
                "rejects": rejects, "median_lower_bound": sorted(bounds)[len(bounds) // 2],
                "min": min(bounds), "max": max(bounds)})
    ck.add("boot_pos3_null_rejection_rate_is_bounded_not_claimed",
           rate <= 0.20, {"measured_rate": rate, "nominal": pb.ALPHA, "mc_se": se},
           "upper bound 0.20 only: a method that always rejected would be fatal; the "
           "comparison with 0.05 is reported above as a measurement")
    ck.add("boot_pos3_null_runs_produce_bounds_not_refusals",
           all(x is not None for x in bounds) and len(bounds) == NULL_EXPERIMENTS,
           {"n_bounds": len(bounds), "nulls": sum(1 for x in bounds if x is None)},
           "noise around zero is not the same as no signal: those runs must still be judged")


# --------------------------------------------------------------------- input rejections
def guard_checks(ck: Check) -> None:
    cases = {
        "empty_table": lambda: pb.paired_lower_bound([]),
        "single_row": lambda: pb.paired_lower_bound(_table([3.0])),
        "nan_value": lambda: pb.paired_lower_bound(_table([float("nan"), 1.0, 2.0])),
        "short_row": lambda: pb.paired_lower_bound([[1.0, 2.0], [3.0, 4.0]]),
        "too_few_reps": lambda: pb.paired_lower_bound(_table([1.0, 2.0]), reps=10),
        "bad_alpha": lambda: pb.paired_lower_bound(_table([1.0, 2.0]), alpha=0.9),
    }
    for label, fn in cases.items():
        ck.add(f"boot_guard_{label}_raises", _raises(fn, pb.BootstrapError),
               "BootstrapError raised", "no numeric result on unusable input")
    ck.add("boot_guard_no_path_returns_a_bare_zero",
           all(_raises(fn, pb.BootstrapError) for fn in cases.values())
           and pb.ZERO_DELTA.startswith("未验") and pb.ZERO_FLOOR.startswith("未验"),
           "empty/<2 rows raise; the two soft states both start with 未验",
           "dispatch: 空输入或行数 <2 直接报错退出而非给出 0")


def cli_checks(ck: Check) -> None:
    empty = _write_csv([])
    one = _write_csv([(1.0, 2.0, 1.0)])
    five = _write_csv([(1.0, 3.0, 1.0), (1.0, 5.0, 1.0), (1.0, 4.0, 1.0),
                       (1.0, 6.0, 1.0), (1.0, 2.0, 1.0)])
    ck.add("boot_cli_empty_table_exits_1", pb.main(["--table", empty]) == 1, "rc=1",
           "loud failure, not an empty bound")
    ck.add("boot_cli_single_row_exits_1", pb.main(["--table", one]) == 1, "rc=1",
           "one pair cannot carry an uncertainty statement")
    api = pb.paired_lower_bound(_table([2.0, 4.0, 3.0, 5.0, 1.0]), reps=500, seed=7)
    rc = pb.main(["--table", five, "--reps", "500", "--seed", "7", "--cell", "shape_l2"])
    ck.add("boot_cli_rc_matches_the_api_verdict_on_the_same_rows",
           rc == (0 if api["cell_verdict"] == "PASS" else 1) and rc in (0, 1),
           {"cli_rc": rc, "api_verdict": api["cell_verdict"]},
           "rc = 0 only on PASS; the CSV path and the mapping path agree")
    refused = pb.main(["--table", _write_csv([(0.0, 3.0, 1.0), (0.0, 5.0, 3.0)])])
    ck.add("boot_cli_dead_floor_exits_nonzero", refused == 1, "rc=1",
           "a blocked verdict is not a successful run")
    ck.add("boot_cli_refuses_to_write_inside_the_repo",
           pb.main(["--table", five, "--json", str(HERE / "junk.json")]) == 2
           and not (HERE / "junk.json").exists(),
           "rc=2 and no file created", "artifact must land outside the repository")


def _write_csv(rows: Sequence[Tuple[float, float, float]]) -> str:
    path = Path(tempfile.mkdtemp(prefix="boot_")) / "table.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write("obs_seed,A,B,C\n")
        for i, (a, b, c) in enumerate(rows):
            fh.write(f"{i + 1},{a},{b},{c}\n")
    return str(path)


def structure_checks(ck: Check) -> None:
    roots = set()
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    ck.add("boot_stdlib_only_no_numpy_no_torch",
           not (roots & {"numpy", "scipy", "torch", "pandas"}), sorted(roots),
           "numpy/scipy/torch/pandas absent -- the dispatch requires stdlib only")
    ck.add("boot_uses_stdlib_random_and_records_the_draw_seed",
           "import random" in SRC and "random.Random(seed)" in SRC
           and '"seed": seed' in SRC, "stdlib random; seed stored in the output",
           "重抽种子写进产物")
    ck.add("boot_defaults_are_the_prereg_numbers",
           (pb.DEFAULT_REPS, pb.ALPHA, pb.MIN_ROWS) == (2000, 0.05, 2),
           {"reps": pb.DEFAULT_REPS, "alpha": pb.ALPHA, "min_rows": pb.MIN_ROWS},
           "(2000, 0.05, 2) -- §4.2's 2,000 and a one-sided 95% bound")
    ck.add("boot_resampling_is_paired_not_two_independent_draws",
           "picks = [deltas[rng.randrange(n)] for _ in range(n)]" in SRC
           and "for _ in range(reps)" in SRC,
           "one index set per replicate, applied to the paired delta column",
           "resampling B and C separately would answer a different question")
    ck.add("boot_pass_word_only_reachable_via_the_two_conditions",
           len([1 for line in SRC.splitlines() if '"cell_verdict"] = "PASS"' in line]) <= 1
           and 'if out["lower_bound"] > 0.0 and out["floor_status"] == OK' in SRC,
           "one PASS assignment, guarded by lower_bound>0 AND floor_status==OK",
           "so neither guard can be bypassed by a later edit that forgets it")



GATE_SRC = HERE / "inversion_gate.py"          # route-2's own gate; read, never imported
ROUTE2_REPORTED = {"boot_lower_95": -0.02417721, "e_b": 0.02375871, "e_c": 0.03914839,
                   "delta": -0.01538968, "e_k1": 0.05074815, "floor": 0.01014963,
                   "n_obs_seed": 8}


def _route2_index_expression() -> Optional[str]:
    """Pull route-2's order-statistic expression out of its source, if the file is present.

    The cross-check models their line `means[min(int((1.0 - level) * n), n - 1)]`; if that
    text ever changes, this returns the new expression and the check below goes red instead
    of quietly comparing against a stale strawman.
    """
    if not GATE_SRC.is_file():
        return None
    for line in GATE_SRC.read_text(encoding="utf-8").splitlines():
        if "means[" in line and "level" in line:
            return line.strip()
    return ""


def cross_impl_checks(ck: Check) -> None:
    """§4.2 now has two implementations on purpose.  Where they differ is stated as a number."""
    ck.add("boot_ximpl_index_conventions_differ_by_exactly_one",
           pb.quantile_index(2000, pb.ALPHA, "inverse_ecdf") == 99
           and pb.quantile_index(2000, pb.ALPHA, "route2_int") == 100,
           {"mine": pb.quantile_index(2000, pb.ALPHA, "inverse_ecdf"),
            "route2": pb.quantile_index(2000, pb.ALPHA, "route2_int")},
           "99 vs 100 (0-based) at R=2000, alpha=0.05")

    expr = _route2_index_expression()
    if expr is None:
        ck.skip("boot_ximpl_route2_expression_still_matches_my_model",
                "inversion_gate.py not present next to this file",
                "not attempted, so it is not counted as green")
    else:
        ck.add("boot_ximpl_route2_expression_still_matches_my_model",
               "int((1.0 - level) * n)" in expr, expr,
               "my route2_int convention models their live line; if they edit it this goes red")

    # Reproduce their estimator from scratch (same stream, their index) and compare with the
    # convention this module exposes -- an independent recalculation, not a call into their code.
    cont = _table([0.7, 2.1, -0.4, 1.3, 0.2, 1.9, 0.5, 1.1])
    deltas = pb._as_delta_rows(cont)
    import random as _r
    rng = _r.Random(pb.DEFAULT_SEED)
    k = len(deltas)
    means = sorted(sum(deltas[rng.randrange(k)] for _ in range(k)) / k
                   for _ in range(pb.DEFAULT_REPS))
    their_number = means[min(int(pb.ALPHA * pb.DEFAULT_REPS), pb.DEFAULT_REPS - 1)]
    rec = pb.reconcile(cont, reps=pb.DEFAULT_REPS, seed=pb.DEFAULT_SEED)
    ck.add("boot_ximpl_my_route2_convention_reproduces_their_estimator",
           rec["route2_int"] == their_number,
           {"from_their_formula": their_number, "from_my_convention": rec["route2_int"],
            "mine_inverse_ecdf": rec["mine_inverse_ecdf"]},
           "bitwise equal to their formula applied to the same replicate stream")
    ck.add("boot_ximpl_route2_bound_is_never_below_mine",
           rec["route2_int"] >= rec["mine_inverse_ecdf"],
           {"mine": rec["mine_inverse_ecdf"], "route2": rec["route2_int"]},
           "a higher order statistic can only be >=; reported, not averaged away")

    # Battery over several table sizes and replicate counts, because "the conventions differ by
    # one order statistic" is only worth stating with the rate it actually shows up at.  Two
    # kinds of difference are counted apart: a real one (>1e-9, the statistic moved) and a
    # last-bit one (<=1e-15, just summation order -- quoting these numbers past ~12 digits
    # is what would make that count matter).
    prng = random.Random(7)
    battery = [_table([0.7, 2.1, -0.4, 1.3, 0.2, 1.9, 0.5, 1.1])]
    battery += [_table([round(prng.gauss(0.0, 1.0), 6) for _ in range(k)])
                for k in (8, 8, 14, 30, 50)]
    battery.append(_table([2.0, 5.0, 8.0]))
    cells = [(t, r, pb.reconcile(t, reps=r, seed=pb.DEFAULT_SEED))
             for t in battery for r in (200, 500, 1000, 2000, 4000)]
    real = [c for c in cells
            if abs(c[2]["route2_int"] - c[2]["mine_inverse_ecdf"]) > 1.0e-9]
    ulp = [c for c in cells if 0.0 < abs(c[2]["route2_int"] - c[2]["mine_inverse_ecdf"])
           <= 1.0e-15]
    ck.add("boot_ximpl_route2_never_sits_below_mine_across_the_battery",
           all(c[2]["route2_int"] >= c[2]["mine_inverse_ecdf"] - 1.0e-12 for c in cells),
           {"cells": len(cells),
            "worst_violation": max((c[2]["mine_inverse_ecdf"] - c[2]["route2_int"])
                                   for c in cells)},
           "a higher order statistic cannot be lower, over every cell"),
    ck.add("boot_ximpl_convention_difference_is_observable_not_assumed",
           len(real) >= 1, {"cells": len(cells), "real_differences": len(real),
                             "last_bit_only": len(ulp),
                             "example": [round(x[2]["mine_inverse_ecdf"], 9)
                                         for x in real[:1]]
                             + [round(x[2]["route2_int"], 9) for x in real[:1]]},
           ">=1 cell must differ by >1e-9, else the named convention would be decoration")
    ck.measure("ximpl_battery", {"cells": len(cells), "differences_gt_1e-9": len(real),
                                 "differences_only_at_last_bit": len(ulp),
                                 "identical_bitwise": sum(1 for c in cells
                                                          if c[2]["identical"])})

    ck.add("boot_ximpl_degeneracy_differences_are_documented",
           _raises(lambda: pb.paired_lower_bound(_table([3.0])), pb.BootstrapError)
           and _raises(lambda: pb.paired_lower_bound([]), pb.BootstrapError),
           "1 row -> BootstrapError here; empty -> BootstrapError here",
           "their gate returns a number for 1 row and None for empty; the dispatch asked for "
           "a non-zero failure here, so this difference is intentional and named")

    ck.measure("route2_reported_first_run", dict(ROUTE2_REPORTED))
    ck.measure("reconciliation_input_available_on_this_disk",
               {"looked_for": "inversion_arms.json 5,405 B / sha256 43d825ba5e8fc57f",
                "found_instead": sorted(str(x.relative_to(HERE.parents[1] / ".scratch" /
                                                        "route2"))
                                        for x in (HERE.parents[1] / ".scratch" / "route2")
                                        .glob("armC_*/inversion_arms.json"))
                if (HERE.parents[1] / ".scratch" / "route2").is_dir() else "no .scratch",
                "per_seed_arrays_present": False})


def main() -> int:
    ap = argparse.ArgumentParser(description="paired_bootstrap self-test (stdlib, no machine time)")
    ap.add_argument("--json", default="", help="write the check log outside the repository")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ck = Check()
    t0 = time.perf_counter()
    zero_delta_checks(ck)
    known_distribution_checks(ck)
    floor_guard_checks(ck)
    null_effect_checks(ck)
    guard_checks(ck)
    cli_checks(ck)
    structure_checks(ck)
    cross_impl_checks(ck)

    out = Path(args.json) if args.json else DEFAULT_OUT
    repo_root = HERE.parents[2]
    if repo_root in out.resolve().parents or str(out).startswith(str(repo_root)):
        raise SystemExit(f"refusing to write self-test output inside the repo: {out}")
    if args.json:
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"checks": ck.rows, "failed": ck.failed}, ensure_ascii=False,
                         indent=2) + "\n"
        if out.is_file() and out.stat().st_size > 0 and not args.force:
            raise SystemExit(f"refusing to overwrite non-empty {out} (pass --force)")
        out.write_bytes(text.encode("utf-8"))
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"json={out}")
        print(f"[boot] artifact_selfcert: path={out} bytes={out.stat().st_size} "
              f"sha256={digest}")
    print(f"total={len(ck.rows)} failed={len(ck.failed)} skipped={len(ck.skipped)}")
    if ck.failed:
        print("FAILED: " + ", ".join(ck.failed))
        return 1
    print(f"ALL GREEN elapsed_s={time.perf_counter() - t0:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
