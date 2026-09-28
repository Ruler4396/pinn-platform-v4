#!/usr/bin/env python3
"""Self-test for shape_metrics.py -- the three scoring cells of the h(xi) inversion (§3).

Run:  python model/scripts/route2/selftest_shape_metrics.py
Exit 0 only if every check passes.  stdlib only, no instance time, nothing installed.

What this file is FOR: three of the checks below are refusals that must fire.  A scoring
module that returns 0.0 or True when it cannot measure is worse than one that crashes,
because 0.0 reads like a perfect score -- the pre-registration's §4 already had to void a
floor of exactly that shape (0.20 x e(A) with e(A)=0 is true for every candidate).  So the
red fixtures here are the load-bearing ones:

  red-1a  a branch whose |dp_branch| = 0        -> 未验：分母为 0, no number out
  red-1b  a reference error of 0 in a ratio      -> 未验：分母为 0, meets_floor stays None
  red-1c  a constant reference series, no scale  -> 未验：分母为 0
  red-2   two identical series                   -> shape_l2 is 0 but the row must say 平凡相等
                                                    and score_eligible must be False
  pos-3   known peak shift, known monotone break, hand-computed RMS -> each cell hits its
        the hand-computed number is written as a literal here, so the check is a comparison
        against arithmetic done outside the module, not a mirror of it.

Plus a structural check for §3's banned list: the entry point exposes exactly three cells,
so a pointwise second-derivative MSE or any "differentiate the reconstructed field again"
quantity has no way to leave this module.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import shape_metrics as sm                              # noqa: E402

SRC = (HERE / "shape_metrics.py").read_text(encoding="utf-8")
XI3 = (0.30, 0.60, 0.90)
REF3 = (100.0, 60.0, 20.0)
PRED3 = (98.0, 63.0, 17.0)
DP3 = 80.0
# hand-computed: diffs (-2,3,-3)/80 = (-0.025,0.0375,-0.0375); RMS = sqrt(0.0034375/3)
SHAPE_L2_EXPECTED = 0.033850160019420
PEAK_SHIFT_EXPECTED = 0.3                              # interior peak 0.60 -> 0.90
_IS_WINDOWS = tempfile.gettempdir().startswith(("C:", "D:"))
DEFAULT_OUT = (Path("D:/PINN-restart/.scratch/route2/shape_metrics_selftest.json")
               if _IS_WINDOWS else Path(os.environ.get("ROUTE2_OUT", tempfile.gettempdir()))
               / "shape_metrics_selftest.json")


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

    def measure(self, name: str, value: object) -> None:
        _emit(f"[shape-measure] {name}={_fmt(value)}")

    def skip(self, name: str, value: object, limit: object = "") -> None:
        """Counted and printed, never folded into PASS: a check with nothing to run on is not
        the same claim as one that ran and passed."""
        self.rows.append({"check": name, "pass": True, "skipped": True,
                          "value": value, "limit": limit})
        _emit(f"[SKIP] {name}: value={_fmt(value)} limit={_fmt(limit)}")

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


def repo_root() -> Optional[Path]:
    """Repository = the directory with a `.git` above this file; None if there is none.

    Counting parent levels (the first version: `parents[2]`) is only correct while these files
    sit at `model/scripts/route2/`.  A verifier who copies them elsewhere got a "repo root" of
    the drive letter, so every --json target looked in-repo and the refusal fired *after* all the
    checks and *before* the summary -> a wall of green lines, no total=, rc=1, i.e. a suite that
    looks broken.  Found no root -> refuse nothing, and say root=unknown out loud.
    """
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / ".git").exists():
            return cand
    return None



CHILD_ENCODING = "utf-8"
OLD_PUBLISHED_COMMIT = "114acd7baf7c0dcf0c1e6347df0d6d0a00467206"      # the byte pair that carried the crash-prone capture
SUITE_FILES = ("shape_metrics.py", "selftest_shape_metrics.py")


def child_env(extra=None):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = CHILD_ENCODING      # the child writes what we are going to decode
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        env.update(extra)
    return env


def capture(done):
    """(rc, stdout, stderr, stdout_unavailable) -- the decoding is ours, over bytes.

    `text=True` let the PARENT decode with the locale default (cp936 on this machine) while the
    CHILD encoded per whatever PYTHONIOENCODING it inherited.  When the two disagree the parent
    does not merely garble: `CompletedProcess.stdout` comes back None and the harness died on
    `.strip()` instead of reporting (统括官 11:5x; reproduced 11:53:53 on 114acd7 and on the
    working tree: hostile env + a directory with no .git above -> rc=1, no total= line).
    """
    missing = done.stdout is None or done.stderr is None
    out = "" if done.stdout is None else done.stdout.decode(CHILD_ENCODING, errors="replace")
    err = "" if done.stderr is None else done.stderr.decode(CHILD_ENCODING, errors="replace")
    return done.returncode, out, err, missing


def run_suite(script, cwd, args=(), extra=None, timeout=900):
    return capture(subprocess.run([sys.executable, str(script)] + list(args),
                                  capture_output=True, cwd=str(cwd), timeout=timeout,
                                  env=child_env(extra)))


def no_repo_above(path):
    for cand in (path, *path.parents):
        if (cand / ".git").exists():
            return False
    return True


def reap(*paths):
    """Remove only the boxes THIS run created, and prove they are gone.

    A skipped branch that leaks its box is how a later "residue must be empty" assertion turns
    red.  Orphans already under D:/Temp from interrupted runs are left alone -- sweeping by
    filename prefix is the risky pattern, not the fix.
    """
    for q in paths:
        if q is None or not q.exists():
            continue
        shutil.rmtree(q, ignore_errors=True)
        if q.exists():
            print(f"shape_box_left_behind: {q} (not removed, deliberately not forced)")


def oob_verdict(rc, total_line, rc_allowed=(0, 2)):
    """The judgement as a pure function, so the fixture's tooth can be fed a degenerate child
    and shown to go red instead of being argued about.  rc=2 (artifact write refused) is a
    legitimate outcome; rc=1 with no summary line is not.
    """
    reasons = []
    if not total_line:
        reasons.append("summary_missing")
    if rc not in rc_allowed:
        reasons.append(f"rc_out_of_range:{rc}")
    return {"green": not reasons, "reasons": reasons, "rc": rc,
            "total_line": total_line[:70]}


def _git_show(rev, rel, cwd):
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
    box = Path(tempfile.mkdtemp(prefix="shape_leg_"))
    child = box / "child.py"
    child.write_bytes(('print("%s")\n' % LEGACY_MARKER).encode("utf-8"))
    rc_new, out_new, err_new, missing_new = run_suite(child, box, timeout=120)
    ok_old, note_old = legacy_capture(child, box)
    ck.add("shape_oob_teeth_old_capture_cannot_read_the_same_child",
           (not missing_new) and (LEGACY_MARKER in out_new) and (not ok_old),
           {"fixed_readable": not missing_new, "fixed_saw_marker": LEGACY_MARKER in out_new,
            "legacy_readable": ok_old, "legacy_note": note_old, "rc_fixed": rc_new},
           "fixed path reads what the legacy path cannot -> the fixture can actually bite")
    reap(box)


def oob_run_checks(ck):
    """(b) 统括官 11:5x: "run the suite where no .git is above it -> the summary line must
    still print, with rc in {0,2}" -- as a fixture that executes, not as an inference.

    It carries its own tooth: the PUBLISHED bytes come out of git history, run in a
    repository-free box under the same hostile encoding, and really do lose their summary line
    (measured 12:01: rc=1, 52 stdout lines, AttributeError on the None stdout).  Without that
    second half this would be a check that only ever sees a healthy child.
    """
    if os.environ.get("RULER_OOB_PROBED") == "1":
        ck.skip("shape_oob_run_prints_the_summary_line", "we are the out-of-repo run itself",
                "not re-entered, so the fixture cannot recurse")
        return
    box = Path(tempfile.mkdtemp(prefix="shape_oob_"))
    if not no_repo_above(box):
        ck.skip("shape_oob_run_prints_the_summary_line", str(box),
                "the probe box sits under a repository, so it would not be an out-of-repo run")
        reap(box)
        return
    for name in SUITE_FILES:
        shutil.copy2(HERE / name, box / name)
    # RULER_OOB_PROBED stops THIS fixture from recursing; the *_MIRRORED guard is deliberately
    # NOT set, so the copy still executes its own mirror fixture -- that nested child is the
    # thing that used to come back None, and skipping it would test only the easy half.
    rc, out, err, missing = run_suite(box / "selftest_shape_metrics.py", box,
                                      extra={"RULER_OOB_PROBED": "1"})
    totals = [L for L in out.splitlines() if L.startswith("total=")]
    now = oob_verdict(rc, totals[0] if totals else "")
    ck.add("shape_oob_run_prints_the_summary_line", now["green"],
           {"rc": rc, "total_line": now["total_line"], "reasons": now["reasons"],
            "stdout_unavailable": missing, "box": box.name,
            "last_line": out.splitlines()[-1][:70] if out.strip() else ""},
           "summary line present and rc in {0,2} with no .git above")

    root = repo_root()
    if root is None:
        ck.measure("shape_oob_old_published_bytes_at_foreign_depth",
                   "unavailable: no repository above this copy, so the published bytes "
                   "cannot be fetched (the mechanism tooth does not depend on them)")
        reap(box)
        return
    old = Path(tempfile.mkdtemp(prefix="shape_oob_old_"))
    wrote = 0
    for name in SUITE_FILES:
        text = _git_show(OLD_PUBLISHED_COMMIT, f"model/scripts/route2/{name}", root)
        if text is None:
            break
        (old / name).write_bytes(text.encode("utf-8"))
        wrote += 1
    if wrote < len(SUITE_FILES) or not no_repo_above(old):
        ck.measure("shape_oob_old_published_bytes_at_foreign_depth",
                   {"staged_files": wrote, "expected": len(SUITE_FILES),
                    "box_ok": no_repo_above(old), "note": "old bytes not stageable"})
        reap(box, old)
        return
    rc_old, out_old, err_old, missing_old = run_suite(
        old / "selftest_shape_metrics.py", old, extra={"RULER_OOB_PROBED": "1"})
    old_totals = [L for L in out_old.splitlines() if L.startswith("total=")]
    ck.measure("shape_oob_old_published_bytes_at_foreign_depth",
               {"rc": rc_old, "summary_present": bool(old_totals),
                "stdout_lines": len(out_old.splitlines()),
                "crash_line": next((L.strip()[:96] for L in err_old.splitlines()
                                    if "AttributeError" in L), "")})
    reap(box, old)


def mirror_run_checks(ck: Check) -> None:
    """Copy the pair to a directory outside any repository and run the copy: rc must be 0 and
    the summary line must be printed.  Under the old depth-counting root this assertion fails."""
    if os.environ.get("SHAPE_MIRRORED") == "1":
        ck.skip("shape_mirror_run_at_foreign_depth_is_green", "we are the mirror run",
                "not re-entered, so the fixture cannot recurse")
        return
    mirror = Path(tempfile.mkdtemp(prefix="shape_mirror_"))
    for name in ("shape_metrics.py", "selftest_shape_metrics.py"):
        shutil.copy2(HERE / name, mirror / name)
    rc, out, err, missing = run_suite(mirror / "selftest_shape_metrics.py", mirror,
                                      extra={"SHAPE_MIRRORED": "1"})
    if missing:
        # never `.strip()` a value that can be None, and never fold an unreadable child into
        # either verdict: it is counted, printed, and reported as not-judged.
        ck.skip("shape_mirror_run_at_foreign_depth_is_green",
                {"rc": rc, "stdout_unavailable": True, "stderr_tail": err[-80:]},
                "child output could not be read, so this run judges nothing (not a pass)")
        reap(mirror)
        return
    lines = out.strip().splitlines()
    ck.add("shape_mirror_run_at_foreign_depth_is_green",
           rc == 0 and "total=" in out and "ALL GREEN" in out,
           {"rc": rc, "summary_present": "total=" in out,
            "last_line": lines[-1][:70] if lines else "", "mirror": mirror.name},
           "rc=0 with a total= line, at a depth where counting parents gives a wrong root")
    reap(mirror)


# ------------------------------------------------------------------ red 1: zero denominator
def zero_denominator_checks(ck: Check) -> None:
    """The measurement must be reported as missing, never as a number or as True."""
    r = sm.shape_errors(PRED3, REF3, XI3, dp_branch=0.0)
    ck.add("shape_red1a_branch_dp_zero_refuses_instead_of_scoring",
           r["status"] == sm.UNVERIFIED and sm.DENOM_ZERO in " ".join(r["reasons"])
           and r["shape_l2"] is None and r["peak_offset"] is None and r["monotone_sign"] is None
           and r["score_eligible"] is False and r["trivial"] is False,
           {"status": r["status"], "shape_l2": r["shape_l2"], "reasons": r["reasons"]},
           "status=UNVERIFIED, all three cells None, 未验：分母为 0 in reasons")
    ck.add("shape_red1a_refused_row_prints_no_number_and_no_pass",
           all(("value=n/a" in line and "status=" + sm.UNVERIFIED in line
                and "score_eligible=False" in line)
               for line in sm.format_all(PRED3, REF3, XI3, "stem", 0.0)),
           [sm.format_row(c, "stem", r) for c in sm.METRIC_NAMES][:1],
           "every printed cell line: value=n/a, score_eligible=False")
    ck.add("shape_red1a_zero_denominator_never_emits_zero_or_true",
           not any(isinstance(r[k], (int, float)) for k in sm.METRIC_NAMES)
           and True not in (r["shape_l2"], r["monotone_sign"]),
           {k: r[k] for k in sm.METRIC_NAMES}, "no numeric 0.0 and no True in refused cells")

    ref_flat = (50.0, 50.0, 50.0)                       # constant branch => derived scale is 0
    r2 = sm.shape_errors(PRED3, ref_flat, XI3)          # no dp_branch => derive from ref span
    ck.add("shape_red1c_derived_scale_zero_refuses",
           r2["status"] == sm.UNVERIFIED and sm.DENOM_ZERO in " ".join(r2["reasons"])
           and r2["scale"] is None and r2["scale_source"] == "derived:ref_span",
           {"scale": r2["scale"], "source": r2["scale_source"], "reasons": r2["reasons"]},
           "未验：分母为 0 and the refusal names which scale was used")

    imp = sm.relative_improvement(0.01, 0.0)            # e_reference = 0 => the voided floor
    ck.add("shape_red1b_zero_baseline_error_refuses_the_improvement",
           imp["status"] == sm.UNVERIFIED and imp["value"] is None
           and imp["meets_floor"] is None and sm.DENOM_ZERO in " ".join(imp["reasons"]),
           {"value": imp["value"], "meets_floor": imp["meets_floor"], "reasons": imp["reasons"]},
           "value/meets_floor stay None: a zero floor must not read as 'improvement proven'")
    ck.add("shape_red1b_zero_floor_is_not_reported_as_zero_improvement",
           imp["floor"] == 0.0 and imp["value"] is None and imp["meets_floor"] is None,
           {"floor": imp["floor"], "value": imp["value"], "meets_floor": imp["meets_floor"]},
           "floor is 0.0 exactly, and the verdict fields stay None")
    ok_imp = sm.relative_improvement(0.02, 0.05, floor_frac=0.20)
    ck.add("shape_refusal_is_specific_not_a_blanket_block",
           ok_imp["status"] == "OK" and abs(ok_imp["value"] - 0.6) < 1.0e-12
           and ok_imp["meets_floor"] is True and abs(ok_imp["floor"] - 0.01) < 1.0e-15,
           {"value": ok_imp["value"], "floor": ok_imp["floor"],
            "meets_floor": ok_imp["meets_floor"]},
           "a non-zero reference still yields a number (0.6) and the floor 0.2*0.05")
    small = sm.relative_improvement(0.0499, 0.05)
    gain, floor = 0.05 - 0.0499, 0.2 * 0.05
    ck.add("shape_tiny_gain_over_nonzero_floor_does_not_meet_floor",
           small["status"] == "OK" and small["meets_floor"] is False
           and gain < floor and abs(small["value"] - 0.002) < 1.0e-12,
           {"gain": gain, "floor": floor, "meets": small["meets_floor"]},
           "gain 1e-4 < floor 1e-2 rejects it; value = gain/reference = 0.002")


# ------------------------------------------------------- red 2: identical sequences
def trivial_equality_checks(ck: Check) -> None:
    r = sm.shape_errors(REF3, REF3, XI3, dp_branch=DP3)
    ck.add("shape_red2_identical_series_is_named_trivial_not_perfect",
           r["shape_l2"] == 0.0 and r["trivial"] is True and r["score_eligible"] is False
           and "平凡相等" in " ".join(r["notes"].get("shape_l2", [])),
           {"shape_l2": r["shape_l2"], "trivial": r["trivial"],
            "score_eligible": r["score_eligible"]},
           "shape_l2 == 0.0 AND trivial AND not score_eligible")
    ck.add("shape_red2_full_monotone_agreement_is_also_trivial",
           r["monotone_sign"] == r["steps"] == 2 and r["score_eligible"] is False
           and all("平凡相等" in " ".join(r["notes"][c]) for c in sm.METRIC_NAMES),
           {"agreement": r["agreement"], "steps": r["steps"],
            "score_eligible": r["score_eligible"]},
           "a perfect count on identical input carries the same warning")
    nt = sm.format_row("peak_offset", "no-peak", plat := sm.shape_errors(REF3, REF3, XI3, DP3))
    ck.add("shape_sentinel_prints_minus_one_literally",
           "value=-1" in nt, nt[:58], "the printed not-taken code is -1, not 0")
    line = sm.format_row("shape_l2", "identical", r)
    ck.add("shape_red2_verdict_line_says_so",
           "trivial=True" in line and "score_eligible=False" in line and "平凡相等" in line,
           line, "printed row must carry the trivial-equality wording")
    near = sm.shape_errors((100.0, 60.0, 20.0000001), REF3, XI3, dp_branch=DP3)
    ck.add("shape_near_identical_is_not_marked_trivial",
           near["trivial"] is False and near["score_eligible"] is True
           and near["shape_l2"] > 0.0,
           {"trivial": near["trivial"], "shape_l2": near["shape_l2"]},
           "trivial means bitwise equal inputs, not a small number")


# ------------------------------------------------------------ positive control: 3 cells hit
def positive_control_checks(ck: Check) -> None:
    r = sm.shape_errors(PRED3, REF3, XI3, dp_branch=DP3)
    ck.add("shape_pos3a_shape_l2_matches_hand_computed_rms",
           abs(r["shape_l2"] - SHAPE_L2_EXPECTED) / SHAPE_L2_EXPECTED < 1.0e-9,
           r["shape_l2"], f"{SHAPE_L2_EXPECTED:.12f} rel 1e-9")

    # A 4-station branch, because on the declared 3 stations (0.30/0.60/0.90) the ONLY
    # interior index is 0.60 -- see shape_pos3b_declared_stations_... below.
    xi4 = (0.2, 0.4, 0.7, 0.9)
    ref_peak = (10.0, 30.0, 20.0, 5.0)                  # interior max at xi=0.40
    pred_peak = (10.0, 5.0, 30.0, 2.0)                 # interior max at xi=0.70
    rp = sm.shape_errors(pred_peak, ref_peak, xi4, dp_branch=25.0)
    ck.add("shape_pos3b_peak_offset_hits_the_known_shift",
           rp["peak_offset_taken"] is True and abs(rp["peak_offset"] - PEAK_SHIFT_EXPECTED)
           < 1.0e-12,
           {"peak_offset": rp["peak_offset"], "peak_ref": rp["peak_ref"],
            "peak_pred": rp["peak_pred"]}, f"== {PEAK_SHIFT_EXPECTED}")
    ck.add("shape_pos3b_a_real_shift_is_never_the_not_taken_sentinel",
           rp["peak_offset"] != -1.0 and rp["peak_offset"] > 0.0,
           {"value": rp["peak_offset"], "sentinel_expected": -1.0},
           "0.3 is not -1: the sentinel cannot be read as a large error either way")
    # The cell's discriminating power on the §1 grid, stated rather than assumed: with 3
    # stations every interior maximum sits at 0.60, so a taken peak_offset is always 0.0.
    perms = [(a, b, c) for a in (1.0, 2.0, 3.0) for b in (1.0, 2.0, 3.0)
             for c in (1.0, 2.0, 3.0) if len({a, b, c}) == 3]
    on_grid = [sm.shape_errors(q, (1.0, 3.0, 2.0), XI3, dp_branch=2.0)["peak_offset"]
               for q in perms]
    ck.add("shape_pos3b_declared_stations_peak_offset_is_only_zero_or_not_taken",
           all(v in (0.0, -1.0) for v in on_grid)
           and any(v == 0.0 for v in on_grid) and any(v == -1.0 for v in on_grid),
           {"distinct_values": sorted(set(on_grid)), "n_permutations": len(perms)},
           "{0.0, -1}: on 3 stations this cell cannot rank arms by peak shift")
    # How many distinct peak shifts this cell can even express, as a function of station
    # count: only interior indices can be a peak, so n stations give n-2 positions and
    # therefore at most n-2 distinct offsets.  Measured here rather than argued.
    def offsets(n: int):
        grid = tuple((i + 1.0) / (n + 1.0) for i in range(n))
        series = lambda at: tuple(3.0 if i == at else 1.0 for i in range(n))
        got = set()
        for a in range(1, n - 1):
            for b in range(1, n - 1):
                r = sm.shape_errors(series(b), series(a), grid, dp_branch=2.0)
                if r["peak_offset_taken"]:
                    got.add(round(r["peak_offset"], 9))
        return sorted(got)
    ck.add("shape_pos3b_expressible_offsets_grow_as_stations_minus_two",
           len(offsets(3)) == 1 and len(offsets(4)) == 2 and len(offsets(5)) == 3
           and offsets(3) == [0.0],
           {"n3": offsets(3), "n4": offsets(4), "n5": offsets(5)},
           "n=3 -> [0.0] only; the cell starts ranking at 4 stations")

    ref_mono = (30.0, 20.0, 10.0)                       # signs [-1,-1]
    pred_break = (30.0, 20.0, 25.0)                    # signs [-1,+1] => one agreement
    rm = sm.shape_errors(pred_break, ref_mono, XI3, dp_branch=20.0)
    ck.add("shape_pos3c_monotone_break_hits_the_known_count",
           rm["monotone_sign"] == 1 and rm["steps"] == 2 and rm["sign_pred"] == [-1, 1]
           and rm["sign_ref"] == [-1, -1],
           {"agreement": rm["agreement"], "steps": rm["steps"],
            "sign_pred": rm["sign_pred"]}, "1 of 2, discrete, no tolerance involved")
    ck.add("shape_pos3c_exact_match_scores_full_and_is_not_trivial",
           sm.shape_errors(ref_mono, ref_mono, XI3, dp_branch=20.0)["monotone_sign"] == 2,
           sm.shape_errors(ref_mono, ref_mono, XI3, dp_branch=20.0)["monotone_sign"],
           "steps=2")


# --------------------------------------------------------------- refusal paths and guards
def input_guard_checks(ck: Check) -> None:
    cases = {
        "length_mismatch": sm.shape_errors(PRED3, REF3, (0.3, 0.6), dp_branch=DP3),
        "too_few_points": sm.shape_errors((1.0, 2.0), (1.0, 3.0), (0.3, 0.6), dp_branch=DP3),
        "xi_not_increasing": sm.shape_errors(PRED3, REF3, (0.9, 0.6, 0.3), dp_branch=DP3),
        "xi_with_duplicate": sm.shape_errors(PRED3, REF3, (0.3, 0.6, 0.6), dp_branch=DP3),
        "nan_input": sm.shape_errors((float("nan"), 63.0, 17.0), REF3, XI3, dp_branch=DP3),
        "inf_denominator": sm.shape_errors(PRED3, REF3, XI3, dp_branch=float("inf")),
    }
    for label, res in cases.items():
        ck.add(f"shape_guard_{label}_is_unverified",
               res["status"] == sm.UNVERIFIED and res["reasons"]
               and all(res[c] is None for c in sm.METRIC_NAMES),
               {"status": res["status"], "reason": res["reasons"][0]},
               "all three cells None and a stated reason")
    plat = sm.shape_errors((10.0, 30.0, 30.0, 10.0), (10.0, 30.0, 30.0, 5.0),
                           (0.2, 0.4, 0.7, 0.9), dp_branch=25.0)
    ck.add("shape_guard_plateau_peak_is_not_taken",
           plat["peak_offset"] == -1.0 and plat["peak_offset_taken"] is False
           and plat["peak_pred"] is None,
           {"peak_offset": plat["peak_offset"], "peak_ref": plat["peak_ref"],
            "peak_pred": plat["peak_pred"]},
           "-1, not 0: a plateau peak has no position")
    ck.add("shape_guard_minus_one_never_appears_as_a_taken_value",
           not any(res["peak_offset"] == -1.0 and res["peak_offset_taken"]
                   for res in list(cases.values()) + [plat,
                                                      sm.shape_errors(PRED3, REF3, XI3, DP3)]),
           "no row pairs -1 with taken=True", "sentinel cannot double as a measurement")


def structure_checks(ck: Check) -> None:
    """§3's banned list, enforced as reachability rather than as a comment."""
    ck.add("shape_registry_is_exactly_the_three_cells",
           sm.METRIC_NAMES == ("shape_l2", "peak_offset", "monotone_sign"),
           list(sm.METRIC_NAMES), "shape_l2, peak_offset, monotone_sign")
    ck.add("shape_registry_appears_on_every_row",
           all(res["metrics"] == sm.METRIC_NAMES
               for res in (sm.shape_errors(PRED3, REF3, XI3, DP3),
                           sm.shape_errors(PRED3, REF3, XI3, 0.0))),
           sm.METRIC_NAMES, "the emitted key set cannot grow silently")
    ck.add("shape_banned_fourth_cell_cannot_be_printed",
           _raises(lambda: sm.format_row("curvature_mse", "stem",
                                         sm.shape_errors(PRED3, REF3, XI3, DP3)), KeyError),
           "KeyError", "no entry point for a pointwise second-derivative MSE")
    roots = set()
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    ck.add("shape_imports_are_stdlib_only",
           roots <= {"__future__", "math", "typing"} and not (roots & {"numpy", "scipy",
                                                                       "torch", "pandas"}),
           sorted(roots), "{'__future__','math','typing'}")
    banned = re.findall(r"\b(d2|second_deriv|curvature|laplacian|gradient_mse|np\.diff)\b", SRC)
    ck.add("shape_no_banned_quantity_is_named", banned == [], banned,
           "empty: §3 bans pointwise 2nd-derivative MSE and re-differentiating a field")
    ck.add("shape_no_nested_difference_is_computed",
           not re.search(r"diff\w*\([^)]*diff\w*\(", SRC), "no diff-of-diff call in the source",
           "monotone_sign uses exactly one difference; a second one is the banned object")


def format_checks(ck: Check) -> None:
    rows = sm.format_all(PRED3, REF3, XI3, "stem", DP3)
    ck.add("shape_format_all_emits_one_line_per_cell",
           len(rows) == 3 and [r.split("cell=")[1].split()[0] for r in rows]
           == list(sm.METRIC_NAMES), rows[:1], "3 lines, in registry order")
    ck.add("shape_line_carries_branch_and_status_and_reason_slot",
           all("branch=stem" in r and "status=" in r and "reason=" in r for r in rows),
           [r.split("cell=")[1].split()[0] for r in rows],
           "each line names its own cell when pasted alone")
    refused = sm.format_all(PRED3, REF3, XI3, "stem", 0.0)
    ck.add("shape_refused_lines_carry_the_chinese_refusal_wording",
           all(sm.DENOM_ZERO in r for r in refused), refused[0][:60],
           "未验：分母为 0 appears on the line, not only in the dict")


def _raises(fn, exc) -> bool:
    try:
        fn()
        return False
    except exc:
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description="shape_metrics self-test (stdlib, 0 instance time)")
    ap.add_argument("--json", default="", help="write the check log outside the repository")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ck = Check()
    t0 = time.perf_counter()
    zero_denominator_checks(ck)
    trivial_equality_checks(ck)
    positive_control_checks(ck)
    input_guard_checks(ck)
    structure_checks(ck)
    format_checks(ck)
    rc_mod, out_mod, err_mod, missing_mod = run_suite(HERE / "shape_metrics.py", HERE,
                                                     timeout=120)
    ck.add("shape_module_main_runs", rc_mod == 0 and not missing_mod,
           {"rc": rc_mod, "stdout_unavailable": missing_mod,
            "stderr_tail": err_mod[-60:] if err_mod else ""},
           "the demo entry point is reachable")

    mirror_run_checks(ck)
    oob_run_checks(ck)
    capture_teeth_checks(ck)

    # The verdict prints before anything touches the disk, so a refused artifact write can
    # never leave "a wall of green lines, no summary, rc=1" -- the shape a healthy suite was
    # wearing when a verifier ran it from another directory (统括官 11:0x).
    _emit(f"total={len(ck.rows)} failed={len(ck.failed)} skipped={len(ck.skipped)}")
    skip_names = ", ".join(ck.skipped) if ck.skipped else "(none)"
    _emit(f"[shape] skip_context: skipped={len(ck.skipped)} 只在本次上下文成立——凡依赖 .git 或真数据的格子，"
          f"在仓外子箱里诚实 [SKIP]（是\u201c没跑\u201d，不是\u201c跑坏\u201d）；本轮名单={skip_names}")
    if ck.failed:
        _emit("FAILED: " + ", ".join(ck.failed))
    verdict_rc = 1 if ck.failed else 0
    print(f"{'FAILED' if ck.failed else 'ALL GREEN'} elapsed_s={time.perf_counter() - t0:.3f}")

    out = Path(args.json) if args.json else DEFAULT_OUT
    root = repo_root()
    if root is not None and (root in out.resolve().parents
                             or str(out.resolve()).startswith(str(root) + os.sep)):
        print(f"[shape] artifact_refused: 目标是仓内路径，不写盘: {out}")
        return verdict_rc or 2
    if args.json:
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"checks": ck.rows, "failed": ck.failed, "skipped": ck.skipped},
                          ensure_ascii=False, indent=2) + "\n"
        if out.is_file() and out.stat().st_size > 0 and not args.force:
            print(f"[shape] artifact_refused: 目标非空且未加 --force，不覆盖: {out}")
            return verdict_rc or 2
        out.write_bytes(text.encode("utf-8"))
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"json={out}")
        print(f"[shape] artifact_selfcert: path={out} bytes={out.stat().st_size} sha256={digest}")
    return verdict_rc


if __name__ == "__main__":
    sys.exit(main())
