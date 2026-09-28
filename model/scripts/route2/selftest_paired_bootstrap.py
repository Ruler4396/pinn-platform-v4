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
import contextlib
import hashlib
import io
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import tempfile
import time
import pathlib
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


def _emit(line: str) -> None:
    """Print without ever letting a glyph take the suite down.

    Measured 12:57: a `ck.measure` value containing `=>`/`\u21d2` hit the GBK console, raised
    UnicodeEncodeError inside `print`, and killed 58 finished readings plus the summary line --
    the same failure family as an in-process product call throwing through `ck.add`'s argument.
    Evidence must degrade to a `?`, never to a missing `total=`.
    """
    try:
        print(line)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(enc, errors="replace").decode(enc, errors="replace"))


class Check:
    def __init__(self) -> None:
        self.rows: List[dict] = []

    def add(self, name: str, ok: bool, value: object, limit: object = "") -> None:
        self.rows.append({"check": name, "pass": bool(ok), "value": value, "limit": limit})
        _emit(f"[{'PASS' if ok else 'FAIL'}] {name}: value={_fmt(value)} limit={_fmt(limit)}")

    def skip(self, name: str, value: object, limit: object = "") -> None:
        """Counted and printed, never folded into PASS: 'all green' and 'green minus the
        checks that had no input to run on' are different claims."""
        self.rows.append({"check": name, "pass": True, "skipped": True,
                          "value": value, "limit": limit})
        _emit(f"[SKIP] {name}: value={_fmt(value)} limit={_fmt(limit)}")

    def measure(self, name: str, value: object) -> None:
        _emit(f"[boot-measure] {name}={_fmt(value)}")

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


def cli_rc(argv: Sequence[str]) -> Dict[str, object]:
    """Run the CLI in-process and turn a raise into DATA instead of a dead suite.

    The mutation test at 12:36 found the hole: with the identity line reading
    `res["quantile_index"]` (which the refusal paths never set), the KeyError propagated out of
    `ck.add`'s own argument, the suite died mid-phase and printed no `total=` -- one product
    defect destroying 55 other readings.  A raise is a finding: name it, count it, carry on.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return {"rc": pb.main(list(argv)), "raised": None, "printed": buf.getvalue()}
    except BaseException as exc:                      # noqa: BLE001 - the point is to catch it
        return {"rc": None, "raised": f"{type(exc).__name__}: {str(exc)[:80]}",
                "printed": buf.getvalue()}


def cli_checks(ck: Check) -> None:
    empty = _write_csv([])
    one = _write_csv([(1.0, 2.0, 1.0)])
    five = _write_csv([(1.0, 3.0, 1.0), (1.0, 5.0, 1.0), (1.0, 4.0, 1.0),
                       (1.0, 6.0, 1.0), (1.0, 2.0, 1.0)])
    a_empty, a_one = cli_rc(["--table", empty]), cli_rc(["--table", one])
    ck.add("boot_cli_empty_table_exits_1", a_empty["rc"] == 1 and a_empty["raised"] is None,
           {"rc": a_empty["rc"], "raised": a_empty["raised"]},
           "loud failure, not an empty bound")
    ck.add("boot_cli_single_row_exits_1", a_one["rc"] == 1 and a_one["raised"] is None,
           {"rc": a_one["rc"], "raised": a_one["raised"]},
           "one pair cannot carry an uncertainty statement")
    api = pb.paired_lower_bound(_table([2.0, 4.0, 3.0, 5.0, 1.0]), reps=500, seed=7)
    five_res = cli_rc(["--table", five, "--reps", "500", "--seed", "7", "--cell", "shape_l2"])
    want_rc = 0 if api["cell_verdict"] == "PASS" else 1
    ck.add("boot_cli_rc_matches_the_api_verdict_on_the_same_rows",
           five_res["rc"] == want_rc and five_res["raised"] is None
           and five_res["rc"] in (0, 1),
           {"cli_rc": five_res["rc"], "raised": five_res["raised"],
            "api_verdict": api["cell_verdict"]},
           "rc = 0 only on PASS; the CSV path and the mapping path agree")
    refused = cli_rc(["--table", _write_csv([(0.0, 3.0, 1.0), (0.0, 5.0, 3.0)])])
    ck.add("boot_cli_dead_floor_exits_nonzero",
           refused["rc"] == 1 and refused["raised"] is None,
           {"rc": refused["rc"], "raised": refused["raised"]},
           "a blocked verdict is not a successful run -- and must not crash the CLI either")
    if pb.repo_root() is None:
        ck.skip("boot_cli_refuses_to_write_inside_the_repo",
                "no .git above this copy, so the guard has nothing to refuse",
                "not attempted here; in the real repo it must return rc=2")
    else:
        junk = cli_rc(["--table", five, "--json", str(HERE / "junk.json")])
        ck.add("boot_cli_refuses_to_write_inside_the_repo",
               junk["rc"] == 2 and junk["raised"] is None and not (HERE / "junk.json").exists(),
               {"rc": junk["rc"], "raised": junk["raised"]},
               "rc=2 and no file created; artifact must land outside the repository")


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
    # 统括官 12:5x ④: the 1-ULP residue closes only when the instance product itself enters git
    # (`git ls-tree -r HEAD | grep inversion_arms` read 0 hits at 12:5x, so "一切走 git" covers the
    # code but not the bytes).  Naming the condition here means the label flips by measurement --
    # nobody has to remember which way it went last time, and nobody may loosen the tolerance.
    # Two scratch roots are searched, not one: this suite lives under the repository, but the
    # route-2 products land in the WORKSPACE scratch (`D:/PINN-restart/.scratch/route2`) -- the
    # 统括官's bad base64 frame is there.  Scanning only the repo root would keep reporting
    # "转写层" out of blindness even after the real bytes arrived on this disk.
    seen = set()
    roots, found_scratch = [], {}
    for cand in (pb.repo_root(), HERE.parents[2] if len(HERE.parents) > 2 else None,
                 Path("D:/PINN-restart")):
        if cand is not None and str(cand) not in seen:
            seen.add(str(cand))
            roots.append(Path(cand))
    for r in roots:
        sc = r / ".scratch" / "route2"
        if sc.is_dir():
            hits = sorted(str(x.relative_to(sc)) for x in sc.glob("**/inversion_arms.json"))
            found_scratch[str(sc)] = hits or "dir present, 0 arms files"
    if not found_scratch:
        found_scratch = {"(no .scratch/route2 under any root)": "none"}
    in_git = []
    if roots:
        done = subprocess.run(["git", "ls-tree", "-r", "HEAD", "--name-only"],
                              capture_output=True, cwd=str(roots[0]), timeout=120, env=child_env())
        if done.returncode == 0 and done.stdout is not None:
            in_git = [l for l in done.stdout.decode(CHILD_ENCODING, "replace").splitlines()
                      if "inversion_arms" in l]
    # The condition that closes the last ULP is the BYTES, not the location: 统括官 quoted
    # sha256 43d825ba... for the 5,405 B instance product, so every candidate on this disk is
    # hashed and compared.  Reporting each candidate's own digest also proves the comparison is
    # not a vacuous "nothing found, so nothing to check".
    cands = []
    for f in sorted({x for sc in found_scratch if isinstance(sc, str)
                     for x in (pathlib.Path(sc).glob("**/inversion_arms.json")
                               if pathlib.Path(sc).is_dir() else [])}):
        raw = f.read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        cands.append({"path": str(f), "bytes": len(raw), "sha16": h[:16],
                      "matches_quoted_43d825ba": h.startswith("43d825ba5e8fc57f")})
    found_scratch = {k: (len(v) if isinstance(v, list) else v) for k, v in found_scratch.items()}
    closed = any(c["matches_quoted_43d825ba"] for c in cands)
    ck.measure("reconciliation_input_available_on_this_disk",
               {"looked_for": "inversion_arms.json 5,405 B / sha256 43d825ba... (旧) "
                              "和 11,172 B (新, leg21 12:19)",
                "found_on_scratch": found_scratch, "found_in_git": in_git,
                "candidates_hashed": cands,
                "strength_label": "原件在这块盘上（sha 对上 43d825ba…）=> 请点一发“原件复算”帧；本测只报条件，"
                                  "不自动解析别人的 schema" if closed else
                                  "转写层：quoted sha 未在这块盘上任何 arms 件中出现 => "
                                  "1 ULP 维持未闭合，不许放宽容差（此测只报条件，不自动跑那一帧）"})


# ---- route-2's first-run product, transcribed by the 统括官 from the instance (11:3x).
# Source: /mnt/workspace/pinn-repro-2026/route2_out/armC_20260928/inversion_arms.json
#         5,405 B, sha256 43d825ba5e8fc57faf79f532f6b44effd4a3a742f3d050ea0197f923b1140332
# This is a labelled copy of someone else's reading, kept as a fixture so the two
# implementations can be compared on identical bytes.  It is NOT this module's output.
REAL8: List[Tuple[float, float, float]] = [
    (0.0508827031098853, 0.02172042622438079, 0.030818473494250005),
    (0.050383390457691954, 0.045548358138925174, 0.05249919052400318),
    (0.05226954559010526, 0.005012222557949382, 0.016903228810166357),
    (0.050250377641424614, 0.03283269641449873, 0.04998445277598644),
    (0.05349508986262563, 0.02899105894254597, 0.021914339177867108),
    (0.05013454953272896, 0.01582664475129503, 0.05395804245021143),
    (0.048640532585062686, 0.009826337424069189, 0.021710216595462444),
    (0.04992904508834397, 0.03031195998386412, 0.06539916848865751),
]
ROUTE2_BOOT = {"lower_bound": -0.024177214033532964, "delta_obs": -0.015389675984884511,
               "seed": 4242, "reps": 2000, "index": 100, "floor": 0.010149630846696709}


def identity_checks(ck: Check) -> None:
    """④ 统括官 12:2x: `reconcile()` must carry an estimator-identity line (convention + seed +
    input digest), so the next person quoting a bound cannot merge "same estimator" (①) with
    "same value" (②).

    The expected digest is rebuilt here from the row bytes rather than read back from
    `rows_digest` -- an assertion whose expectation comes from the function under test is a
    tautology wearing a name.
    """
    import hashlib

    def oracle(rows: Sequence[Tuple[float, float, float]]) -> str:
        lines = []
        for i, (a, b, c) in enumerate(rows, 1):
            lines.append(f"obs_seed={float(i)!r}||A={a!r}|B={b!r}|C={c!r}")
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

    table = [{"obs_seed": i + 1, "A": a, "B": b, "C": c}
             for i, (a, b, c) in enumerate(REAL8)]
    rec = pb.reconcile(table)
    ident = rec.get("identity", "")
    want = oracle(REAL8)
    ck.add("boot_identity_names_conventions_seed_and_input_digest",
           all(s in ident for s in ("seed=20260928", "inverse_ecdf(99)", "route2_int(100)",
                                    "reps=2000", "n_pairs=8", f"rows_sha256={want}"))
           and rec["rows_sha256"] == want,
           {"identity_tail": ident[-88:], "digest_matches_independent_rebuild":
            rec["rows_sha256"] == want},
           "identity carries both conventions, the seed and the digest rebuilt from the rows")
    perturbed = [{**table[0], "C": table[0]["C"] + 1e-9}] + table[1:]
    ck.add("boot_identity_survives_reps_and_breaks_on_one_changed_number",
           pb.reconcile(table)["identity"] == pb.reconcile(table)["identity"]
           and pb.reconcile(table)["rows_sha256"] != pb.reconcile(perturbed)["rows_sha256"],
           {"stable_across_calls": pb.reconcile(table)["identity"]
            == pb.reconcile(table)["identity"],
            "changed_one_C_digit": pb.reconcile(perturbed)["rows_sha256"][:16],
            "original": want[:16]},
           "same bytes -> same identity; one 1e-9 change in one arm -> different digest")
    ck.add("boot_identity_is_shared_by_two_bounds_that_differ",
           rec["identical"] is False
           and rec["mine_inverse_ecdf"] != rec["route2_int"]
           and rec["identity"].count(want) == 1,
           {"mine": rec["mine_inverse_ecdf"], "theirs": rec["route2_int"],
            "identity": ident[:60] + "..."},
           "one identity, two numbers -> ② (cross-convention) is written as sign-only")
    align = pb.paired_lower_bound(table, reps=2000, seed=ROUTE2_BOOT["seed"], index="route2_int")
    ck.add("boot_alignment_cell_names_its_input_too",
           align["rows_sha256"] == want and align["lower_bound"] == ROUTE2_BOOT["lower_bound"],
           {"seed": align["seed"], "index": align["index_convention"],
            "digest_16": align["rows_sha256"][:16],
            "bound": align["lower_bound"], "route2_reported": ROUTE2_BOOT["lower_bound"]},
           "① : bitwise equality is evidence exactly when convention+seed+rows all match")
    # Same numbers, two entry points: the CSV loader turns obs_seed into a float while an
    # in-memory table keeps it an int.  The first version hashed the raw reprs, so ONE input got
    # TWO digests depending on the door it came in through -- the opposite of what an identity
    # line is for.  This check runs wherever the suite runs; only the subprocess checks below
    # need a repository.
    staged = Path(tempfile.mkdtemp(prefix="boot_door_")) / "rows.csv"
    staged.write_bytes(("obs_seed,A,B,C\n" + "\n".join(
        f"{i},{a!r},{b!r},{c!r}" for i, (a, b, c) in enumerate(REAL8, 1)) + "\n").encode("utf-8"))
    via_csv = pb._load_table(str(staged))
    ck.add("boot_digest_is_the_same_through_both_doors",
           pb.rows_digest(table) == pb.rows_digest(via_csv) == want
           and isinstance(via_csv[0]["obs_seed"], float)
           and isinstance(table[0]["obs_seed"], int),
           {"csv_type": type(via_csv[0]["obs_seed"]).__name__,
            "memory_type": type(table[0]["obs_seed"]).__name__,
            "digest_memory_16": pb.rows_digest(table)[:16],
            "digest_csv_16": pb.rows_digest(via_csv)[:16]},
           "door-to-door, not door-to-oracle: the two entry points must hash one input alike")
    reap(staged.parent)

    box = Path(tempfile.mkdtemp(prefix="boot_ident_"))
    csv_path = box / "rows.csv"
    csv_path.write_bytes(("obs_seed,A,B,C\n" + "\n".join(
        f"{i},{a!r},{b!r},{c!r}" for i, (a, b, c) in enumerate(REAL8, 1))
        + "\n").encode("utf-8"))
    rc, out, err, missing = run_suite(Path(pb.__file__), box,
                                      ("--table", str(csv_path), "--cell", "shape_l2"),
                                      timeout=120)
    id_lines = [L for L in out.splitlines() if L.startswith("[boot] identity:")]
    ck.add("boot_cli_prints_exactly_one_identity_line",
           not missing and len(id_lines) == 1
           and f"rows_sha256={want}" in id_lines[0] and "cell=shape_l2" in id_lines[0],
           {"rc": rc, "identity_lines": len(id_lines),
            "line": id_lines[0][:100] if id_lines else ""},
           "the printed bound travels with its identity, on the real entry point")
    # The identity line reads `quantile_index`, which the refusal paths never set -- so a naive
    # implementation crashes the CLI exactly on the inputs it exists to document.  Caught here
    # rather than in someone else's terminal: dead floor -> rc=1, identity still printed, and no
    # traceback on stderr.
    dead = box / "dead_floor.csv"
    dead.write_bytes(b"obs_seed,A,B,C\n1,0.0,3.0,1.0\n2,0.0,5.0,3.0\n")
    rc_d, out_d, err_d, missing_d = run_suite(Path(pb.__file__), box,
                                              ("--table", str(dead), "--cell", "shape_l2"),
                                              timeout=120)
    ck.add("boot_cli_identity_line_survives_a_refused_floor",
           rc_d == 1 and not missing_d and "[boot] identity:" in out_d
           and "Traceback" not in err_d,
           {"rc": rc_d, "identity_present": "[boot] identity:" in out_d,
            "stderr_has_traceback": "Traceback" in err_d,
            "identity_line": next((L[:96] for L in out_d.splitlines()
                                   if L.startswith("[boot] identity:")), "")},
           "refused input must still exit 1 with the identity line, not with a KeyError")
    reap(box)


def real_input_checks(ck: Check) -> None:
    """The reconciliation on route-2's own bytes: four bounds, and TWO claims that must not be
    merged (统括官 12:2x withdrew the previous turn's blanket ban on "bit-for-bit equal").

    (1) Same order-statistic convention AND same resample seed (`route2_int @ 4242`) reproduces
        route-2's 17 digits bit for bit.  THAT cell is the evidence that the two implementations
        compute one estimator.  An over-broad prohibition here would throw away real evidence --
        which is as damaging as having no rule at all.
    (2) Across conventions and across seeds the four numbers take 3 distinct values and are all
        < 0: the *judgement* (lower bound < 0) is robust, the *value* is not, because 8 paired
        means give an atomised resampling distribution (2^8 index arrangements).
        `boot_real_the_four_bounds_are_not_one_number` catches writing (2) as if it were (1).
        `boot_real_our_seed_collision_is_atomicity_not_agreement` catches the other direction: at
        OUR seed one convention lands on their atom and the other does not -- that collision is
        still an atom, not alignment.
    """
    table = [{"obs_seed": i + 1, "A": a, "B": b, "C": c}
             for i, (a, b, c) in enumerate(REAL8)]
    four = {}
    for seed in (ROUTE2_BOOT["seed"], pb.DEFAULT_SEED):
        for idx in pb.INDEX_CONVENTIONS:
            r = pb.paired_lower_bound(table, reps=2000, seed=seed, index=idx)
            four[f"{idx}@{seed}"] = r["lower_bound"]
    bounds = list(four.values())
    ck.add("boot_real_their_seed_and_their_index_reproduce_their_number_bitwise",
           four["route2_int@4242"] == ROUTE2_BOOT["lower_bound"],
           {"from_this_module": four["route2_int@4242"], "route2_reported":
            ROUTE2_BOOT["lower_bound"]},
           "exact 17-digit match => same RNG consumption, same sort, same order statistic")
    ck.add("boot_real_all_four_bounds_are_negative", all(v < 0.0 for v in bounds),
           four, "sign is the invariant; this is the claimable sentence")
    ck.add("boot_real_the_four_bounds_are_not_one_number", len(set(bounds)) >= 2,
           {"distinct": len(set(bounds)), "bounds": four},
           "value is convention-and-seed dependent, so no receipt may quote one of them "
           "as 'the' lower bound")
    ck.add("boot_real_our_seed_collision_is_atomicity_not_agreement",
           four["inverse_ecdf@20260928"] == ROUTE2_BOOT["lower_bound"]
           and four["route2_int@20260928"] != ROUTE2_BOOT["lower_bound"],
           {"mine_at_our_seed": four["inverse_ecdf@20260928"],
            "their_index_at_our_seed": four["route2_int@20260928"]},
           "one of the four lands on their atom, the other does not => atoms, not methods")
    obs = pb.paired_lower_bound(table, reps=2000, seed=ROUTE2_BOOT["seed"],
                                index="route2_int")
    floor = 0.20 * obs["e_a_mean"]
    # delta_obs is bitwise equal; the floor differs by EXACTLY one ULP (measured:
    # 1.734723475976807e-18 = ulp of the value), which is what transcribing the instance JSON
    # into 17-digit CSV rows does.  The bound is therefore stated in ULPs -- the first version
    # guessed 1e-18 and the measurement went red on it, i.e. it was another un-measured
    # assertion.  Closing the last ULP needs the file itself, not a looser bound.
    ck.add("boot_real_aggregate_quantity_matches_their_verdict_fields",
           obs["delta_obs"] == ROUTE2_BOOT["delta_obs"]
           and abs(floor - ROUTE2_BOOT["floor"]) <= math.ulp(ROUTE2_BOOT["floor"]),
           {"delta_obs": obs["delta_obs"], "floor_mine": floor, "floor_theirs":
            ROUTE2_BOOT["floor"], "floor_diff_in_ulps": abs(floor - ROUTE2_BOOT["floor"])
            / math.ulp(ROUTE2_BOOT["floor"])},
           "delta_obs bitwise equal; floor within 1 ULP (CSV is a transcription of the "
           "instance JSON -- closing that ULP needs the file, not a retuned bound)")
    better = sum(1 for _, b, c in REAL8 if c < b)
    ck.add("boot_real_only_one_of_eight_rows_favours_C", better == 1
           and [i for i, (_, b, c) in enumerate(REAL8, 1) if c < b] == [5],
           {"rows_favouring_C": better, "obs_seeds": [i for i, (_, b, c) in enumerate(REAL8, 1)
                                                      if c < b]},
           "matches route-2's own 'only seed 5' line, read off the same 8 rows")
    ck.measure("four_lower_bounds_real_input", four)
    ck.measure("input_provenance", {"path_on_nas":
               "/mnt/workspace/pinn-repro-2026/route2_out/armC_20260928/inversion_arms.json",
               "bytes": 5405,
               "sha256": "43d825ba5e8fc57faf79f532f6b44effd4a3a742f3d050ea0197f923b1140332",
               "transcribed_by": "统括官 11:3x, from GET api/contents (not on this laptop)"})


CHILD_ENCODING = "utf-8"
SUITE_FILES = ("paired_bootstrap.py", "selftest_paired_bootstrap.py")
OLD_PUBLISHED_COMMIT = "114acd7baf7c0dcf0c1e6347df0d6d0a00467206"   # == origin/main at 11:53


def child_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = CHILD_ENCODING       # the child writes what we are going to decode
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        env.update(extra)
    return env


def capture(done: "subprocess.CompletedProcess") -> Tuple[int, str, str, bool]:
    """(rc, stdout, stderr, stdout_unavailable).

    `text=True` decodes with the locale default (cp936 on this machine) while the child encodes
    with whatever PYTHONIOENCODING it inherited, and when the two disagree the parent does not
    merely garble: `CompletedProcess.stdout` comes back None and the harness died on `.strip()`
    instead of reporting (统括官 11:5x; reproduced 11:53:53 on 114acd7 and on the working tree:
    PYTHONIOENCODING=utf-8 at a repository-free depth -> rc=1, no total= line).  Capture bytes,
    decode them here, and never call a method on a value that can be None.
    """
    missing = done.stdout is None or done.stderr is None
    out = "" if done.stdout is None else done.stdout.decode(CHILD_ENCODING, errors="replace")
    err = "" if done.stderr is None else done.stderr.decode(CHILD_ENCODING, errors="replace")
    return done.returncode, out, err, missing


def run_suite(script: Path, cwd: Path, args=(),
              extra: Optional[Dict[str, str]] = None,
              timeout: int = 900) -> Tuple[int, str, str, bool]:
    return capture(subprocess.run([sys.executable, str(script)] + list(args),
                                  capture_output=True, cwd=str(cwd), timeout=timeout,
                                  env=child_env(extra)))


def no_repo_above(path: Path) -> bool:
    for cand in (path, *path.parents):
        if (cand / ".git").exists():
            return False
    return True


def reap(*paths: Optional[Path]) -> None:
    """Remove only the boxes THIS run created, and prove they are gone.

    A skipped branch that returns without cleaning leaves an orphan under D:/Temp, and the next
    run's "residue must be empty" assertion goes red for nobody's fault but mine (this machine
    already carries boot_mirror_* orphans from interrupted runs -- those are left alone; glob
    deleting by prefix is the risky pattern, not the fix).
    """
    for p in paths:
        if p is None or not p.exists():
            continue
        shutil.rmtree(p, ignore_errors=True)
        if p.exists():
            print(f"[boot] oob_box_left_behind: {p} (not removed, deliberately not forced)")


def oob_verdict(rc: int, total_line: str, rc_allowed: Tuple[int, ...] = (0, 2)) -> Dict[str, object]:
    """The judgement as a pure function, so the fixture's tooth can be fed a degenerate child
    and shown to go red instead of being argued about.  rc=2 is a legitimate outcome (the
    artifact write was refused), rc=1 with no summary is not."""
    reasons = []
    if not total_line:
        reasons.append("summary_missing")
    if rc not in rc_allowed:
        reasons.append(f"rc_out_of_range:{rc}")
    return {"green": not reasons, "reasons": reasons, "rc": rc, "total_line": total_line[:70]}


def _git_show(rev: str, rel: str, cwd: Path) -> Optional[str]:
    done = subprocess.run(["git", "show", f"{rev}:{rel}"], capture_output=True,
                          cwd=str(cwd), timeout=120, env=child_env())
    if done.returncode != 0 or done.stdout is None:
        return None
    return done.stdout.decode(CHILD_ENCODING, errors="replace")


LEGACY_MARKER = "⇒ 未验"          # cp936 cannot round-trip this; that is exactly the point


def legacy_capture(script, cwd, extra=None):
    """The PRE-FIX pattern, kept only so the fixture can show it would have caught it.

    With `text=True` and no encoding, the parent decodes with the locale default (cp936 here)
    inside a reader thread; a child that prints a byte cp936 cannot decode kills that thread and
    `communicate()` then hands back `stdout=None`.  One mechanism, two faces -- which is why the
    crash reported at 11:5x showed a UnicodeDecodeError in the log AND an AttributeError on None.
    Nothing real is decided through this path.
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = CHILD_ENCODING
    if extra:
        env.update(extra)
    # The decode failure happens on CPython's reader THREAD, where it is reported through
    # threading.excepthook (i.e. straight to our stderr) and `communicate()` then returns None.
    # Capturing it here keeps the evidence in the check value and the suite's stderr empty.
    caught = []
    saved_hook = threading.excepthook

    def _hook(args):
        caught.append("%s: %s" % (args.exc_type.__name__, str(args.exc_value)[:70]))

    threading.excepthook = _hook
    try:
        done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                              timeout=120, cwd=str(cwd), env=env)
    except Exception as exc:                       # UnicodeDecodeError can surface right here
        return False, "%s: %s" % (type(exc).__name__, str(exc)[:70])
    finally:
        threading.excepthook = saved_hook
    if done.stdout is None:
        return False, (caught[0] if caught else "stdout is None (no thread error captured)")
    return True, done.stdout[:40]


def capture_teeth_checks(ck):
    """One child, two capture patterns: the fixed path must read the marker, the old pattern
    must be unable to.  Without this the out-of-repo fixture would only ever see a healthy
    child, and a harness that prints "green" for everything passes that test just as well.
    """
    box = Path(tempfile.mkdtemp(prefix="boot_leg_"))
    child = box / "child.py"
    child.write_bytes(('print("%s")\n' % LEGACY_MARKER).encode("utf-8"))
    rc_new, out_new, err_new, missing_new = run_suite(child, box, timeout=120)
    ok_old, note_old = legacy_capture(child, box)
    ck.add("boot_oob_teeth_old_capture_cannot_read_the_same_child",
           (not missing_new) and (LEGACY_MARKER in out_new) and (not ok_old),
           {"fixed_readable": not missing_new, "fixed_saw_marker": LEGACY_MARKER in out_new,
            "legacy_readable": ok_old, "legacy_note": note_old, "rc_fixed": rc_new},
           "fixed path reads what the legacy path cannot -> the fixture can actually bite")
    reap(box)


def oob_run_checks(ck: Check) -> None:
    """(b) 统括官 11:5x: "a suite run where no .git is above it must still print its summary
    line, with rc in {0,2}" -- as a fixture that executes, not as an inference.

    It carries its own tooth: the *published* bytes (114acd7) are fetched out of git history,
    copied to a repository-free box, and run under the same hostile encoding.  They crash, so
    `oob_verdict` must call them not-green.  Without that second half this fixture would be a
    check that only ever sees a healthy child.
    """
    if os.environ.get("RULER_OOB_PROBED") == "1":
        ck.skip("boot_oob_run_prints_the_summary_line", "we are the out-of-repo run itself",
                "not re-entered, so the fixture cannot recurse")
        return
    box = Path(tempfile.mkdtemp(prefix="boot_oob_"))
    if not no_repo_above(box):
        ck.skip("boot_oob_run_prints_the_summary_line", str(box),
                "the probe box sits under a repository, so it would not be an out-of-repo run")
        reap(box)
        return
    for name in SUITE_FILES:
        shutil.copy2(HERE / name, box / name)
    # RULER_OOB_PROBED stops *this* fixture from recursing; BOOT_MIRRORED is deliberately NOT
    # set, so the copy still executes its own mirror fixture -- that nested child is the thing
    # that used to come back None, and skipping it would make this fixture test only the easy half.
    rc, out, err, missing = run_suite(box / "selftest_paired_bootstrap.py", box,
                                      extra={"RULER_OOB_PROBED": "1"})
    totals = [L for L in out.splitlines() if L.startswith("total=")]
    now = oob_verdict(rc, totals[0] if totals else "")
    ck.add("boot_oob_run_prints_the_summary_line", now["green"],
           {"rc": rc, "total_line": now["total_line"], "reasons": now["reasons"],
            "stdout_unavailable": missing, "box": box.name,
            "last_line": out.splitlines()[-1][:70] if out.strip() else ""},
           "summary line present and rc in {0,2} with no .git above")

    root = pb.repo_root()
    if root is None:
        ck.measure("boot_oob_old_published_bytes_at_foreign_depth",
                   "unavailable: no repository above this copy, so the published bytes "
                   "cannot be fetched (the mechanism tooth does not depend on them)")
        reap(box)
        return
    old = Path(tempfile.mkdtemp(prefix="boot_oob_old_"))
    wrote = 0
    for name in SUITE_FILES:
        text = _git_show(OLD_PUBLISHED_COMMIT, f"model/scripts/route2/{name}", root)
        if text is None:
            break
        (old / name).write_bytes(text.encode("utf-8"))
        wrote += 1
    if wrote < len(SUITE_FILES) or not no_repo_above(old):
        ck.measure("boot_oob_old_published_bytes_at_foreign_depth",
                   {"staged_files": wrote, "expected": len(SUITE_FILES),
                    "box_ok": no_repo_above(old), "note": "old bytes not stageable"})
        reap(box, old)
        return
    rc_old, out_old, err_old, missing_old = run_suite(
        old / "selftest_paired_bootstrap.py", old, extra={"RULER_OOB_PROBED": "1"})
    old_totals = [L for L in out_old.splitlines() if L.startswith("total=")]
    ck.measure("boot_oob_old_published_bytes_at_foreign_depth",
               {"rc": rc_old, "summary_present": bool(old_totals),
                "stdout_lines": len(out_old.splitlines()),
                "crash_line": next((L.strip()[:96] for L in err_old.splitlines()
                                    if "AttributeError" in L), "")})
    reap(box, old)


def mirror_run_checks(ck: Check) -> None:
    """Copy the pair to a directory outside the repo and run the copy: the summary line must
    still print and rc must be 0.

    The first version located the repository by counting parent levels
    (`Path(__file__).parents[2]`), so a verifier who moved these two files elsewhere got a
    "repo root" of the drive letter, every --json target looked in-repo, and the refusal fired
    after all the checks but before the summary -> 42 green lines, no total=, rc=1, which reads
    exactly like a broken suite.
    """
    if os.environ.get("BOOT_MIRRORED") == "1":
        ck.skip("boot_mirror_run_at_foreign_depth_is_green", "we are the mirror run",
                "not re-entered, so the fixture cannot recurse")
        return
    mirror = Path(tempfile.mkdtemp(prefix="boot_mirror_"))
    for name in ("paired_bootstrap.py", "selftest_paired_bootstrap.py"):
        shutil.copy2(HERE / name, mirror / name)
    rc, out, err, missing = run_suite(mirror / "selftest_paired_bootstrap.py", mirror,
                                      extra={"BOOT_MIRRORED": "1"})
    if missing:
        # never `.strip()` a value that can be None, and never fold an unreadable child into
        # either verdict: it is counted, printed, and reported as not-judged.
        ck.skip("boot_mirror_run_at_foreign_depth_is_green",
                {"rc": rc, "stdout_unavailable": True, "stderr_tail": err[-80:]},
                "child output could not be read, so this run judges nothing (not a pass)")
        reap(mirror)
        return
    lines = out.strip().splitlines()
    ck.add("boot_mirror_run_at_foreign_depth_is_green",
           rc == 0 and "total=" in out and "ALL GREEN" in out,
           {"rc": rc, "summary_present": "total=" in out,
            "last_line": lines[-1][:70] if lines else "", "mirror": mirror.name},
           "rc=0 with a total= line, at a depth where counting parents gives a wrong root")
    reap(mirror)


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
    real_input_checks(ck)
    identity_checks(ck)
    mirror_run_checks(ck)
    oob_run_checks(ck)
    capture_teeth_checks(ck)

    # The verdict prints BEFORE anything touches the disk: a refused artifact write must not
    # be able to eat the summary line and leave a green-looking run exiting 1.
    _emit(f"total={len(ck.rows)} failed={len(ck.failed)} skipped={len(ck.skipped)}")
    skip_names = ", ".join(ck.skipped) if ck.skipped else "(none)"
    _emit(f"[boot] skip_context: skipped={len(ck.skipped)} 只在本次上下文成立——凡依赖 .git 或真数据的格子，"
          f"在仓外子箱里诚实 [SKIP]（是\u201c没跑\u201d，不是\u201c跑坏\u201d）；本轮名单={skip_names}")
    if ck.failed:
        _emit("FAILED: " + ", ".join(ck.failed))
    verdict_rc = 1 if ck.failed else 0
    _emit(f"{'FAILED' if ck.failed else 'ALL GREEN'} "
          f"elapsed_s={time.perf_counter() - t0:.3f}")

    out = Path(args.json) if args.json else DEFAULT_OUT
    root = pb.repo_root()
    if root is not None and (root in out.resolve().parents
                             or str(out.resolve()).startswith(str(root) + os.sep)):
        print(f"[boot] artifact_refused: 目标是仓内路径，不写盘: {out}")
        return verdict_rc or 2
    if args.json:
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"checks": ck.rows, "failed": ck.failed,
                           "skipped": ck.skipped}, ensure_ascii=False,
                          indent=2) + "\n"
        if out.is_file() and out.stat().st_size > 0 and not args.force:
            print(f"[boot] artifact_refused: 目标非空且未加 --force，不覆盖: {out}")
            return verdict_rc or 2
        out.write_bytes(text.encode("utf-8"))
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"json={out}")
        print(f"[boot] artifact_selfcert: path={out} bytes={out.stat().st_size} "
              f"sha256={digest}")
    return verdict_rc


if __name__ == "__main__":
    sys.exit(main())
