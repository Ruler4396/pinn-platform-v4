#!/usr/bin/env python3
"""The §4 discriminability gate, as code -- with the tautology that killed the paper draft's version
made structurally impossible.

Why this file exists: the pre-registered floor was written `e(B) - e(C) >= 0.20 * e(A)`, but arm A
(the zero-training impedance network) has NO shape DOF, so its shape error is the constant field and
`e(A)` was filled with 0.0000 -- which makes the floor 0 and the gate incapable of ever going red.
The lead's ruling (10:1x) is option (a): the denominator is `e(K1)`, the constant-width fit measured
through the SAME metric pipeline as every other arm.

The two refusals are part of the design, not commentary:
  * floor <= 0            -> status "未验", and the caller cannot get a verdict out of it;
  * arm C identical to B  -> delta == 0, so 归宿② is the only legal output.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import roundtrip_h_test as rt                      # one forward model, one metric -- no second copy to drift
import impedance_baseline as ib                    # rank_by_elimination

ALPHA = 0.20                                        # pre-registered effect-size floor fraction
BOOT_N = 2000                                       # pre-registered bootstrap draws
BOOT_SEED = 4242                                    # fixed, so a re-run reproduces the interval
GRID = 100                                          # samples per branch for the shape metric


def h_from_fn(branch, frac, fn):
    return fn(branch, frac)


def shape_error(hat, truth) -> dict:
    """hat/truth: callables (branch, frac) -> h. Same grid for every arm, knots or MLP alike."""
    num = den = 0.0
    grid = [(i + 0.5) / GRID for i in range(GRID)]
    for b in rt.KNOTS:
        for f in grid:
            a, t = hat(b, f), truth(b, f)
            num += (a - t) ** 2
            den += t * t
    throat = min(((b, f, hat(b, f)) for b in rt.KNOTS for f in grid), key=lambda x: x[2])
    throat_true = min(((b, f, truth(b, f)) for b in rt.KNOTS for f in grid), key=lambda x: x[2])
    # the two discrete cells, so `decide()` stops being handed only one of the three it requires
    mono = 0
    tot = 0
    for b in rt.KNOTS:
        ds_hat = [hat(b, (i + 1) / 20.0) - hat(b, i / 20.0) for i in range(20)]
        ds_true = [truth(b, (i + 1) / 20.0) - truth(b, i / 20.0) for i in range(20)]
        for x, y in zip(ds_hat, ds_true):
            if abs(y) < 1.0e-4:
                continue
            tot += 1
            mono += 1 if (x > 0) == (y > 0) else 0
    monotone_frac = (mono / tot) if tot else None
    return {"rel_l2": math.sqrt(num / max(den, 1.0e-30)),
            "throat": {"branch": throat[0], "frac": round(throat[1], 4), "h": round(throat[2], 6)},
            "throat_true": {"branch": throat_true[0], "frac": round(throat_true[1], 4)},
            "throat_offset_frac": round(abs(throat[1] - throat_true[1]), 4),
            "monotone_sign_agreement": monotone_frac,
            "monotone_cells": tot}


def dof_fn(dof):
    return lambda branch, frac: rt.h_at(branch, frac, dof)


def constant_fn(value=1.0):
    return lambda branch, frac: value


def paired_bootstrap_lower(deltas, n=BOOT_N, seed=BOOT_SEED, level=0.95):
    """Resample the PAIRED deltas (obs_seed is the pairing unit) and return the one-sided lower bound."""
    if not deltas:
        return None
    rng = random.Random(seed)
    means = []
    k = len(deltas)
    for _ in range(n):
        pick = [deltas[rng.randrange(k)] for _ in range(k)]
        means.append(sum(pick) / k)
    means.sort()
    return means[min(int((1.0 - level) * n), n - 1)]


def decide(per_seed_e_k1, per_seed_e_b, per_seed_e_c, monotone_ok=None, peak_ok=None):
    """The three-cell rule from §4, with the refusal branches that make the floor non-vacuous.

    Inputs are lists indexed by obs_seed: shape errors of the K=1 baseline, the same-DOF NLS arm (B)
    and the trained arm (C); plus the two discrete cells (monotonicity, peak position) if supplied.
    """
    e_k1 = sum(per_seed_e_k1) / len(per_seed_e_k1)
    e_b = sum(per_seed_e_b) / len(per_seed_e_b)
    e_c = sum(per_seed_e_c) / len(per_seed_e_c)
    out = {"floor_alpha": ALPHA, "e_k1": e_k1, "e_b": e_b, "e_c": e_c,
           "delta": e_b - e_c, "cells": {}}
    if e_k1 <= 0.0:
        return {**out, "status": "未验", "refused": "floor_denominator_zero",
                "why": "e(K1)=0 -- the floor 0.20*e(K1) is 0, so ANY improvement would count as "
                       "discriminable. A gate that cannot go red is not a gate; no verdict is issued.",
                "verdict": None}
    floor = ALPHA * e_k1
    deltas = [b - c for b, c in zip(per_seed_e_b, per_seed_e_c)]
    lb = paired_bootstrap_lower(deltas)
    effect_ok = (e_b - e_c) >= floor
    stat_ok = lb is not None and lb > 0.0
    cells = {"shape_l2": effect_ok and stat_ok,
             "peak_position": None if peak_ok is None else bool(peak_ok),
             "monotone_sign": None if monotone_ok is None else bool(monotone_ok)}
    out.update({"floor_value": floor, "boot_lower_95": lb, "cells": cells,
                "effect_ok": effect_ok, "stat_ok": stat_ok})
    decided = [v for v in cells.values() if v is not None]
    out["cells_supplied"] = len(decided)
    if len(decided) < 2:
        # §4 needs >=2 cells, so a runner that hands over one cell cannot produce a verdict at all --
        # reporting 归宿② there would look like a judgement and actually be the gate measuring nothing.
        return {**out, "status": "未验", "refused": "insufficient_cells", "verdict": None,
                "why": f"cells_supplied={len(decided)} < 2: the rule is '>=2 of 3 cells', so one cell "
                       "can neither pass nor fail it. Fix the runner, not the threshold."}
    n_ok = sum(1 for v in decided if v)
    # §4 says >= 2 of the three cells.  A one-cell shortcut here would quietly weaken the rule that
    # the pre-registration exists to freeze, so there is deliberately no `n_ok == 1` branch.
    verdict = ("反演成立（在本观测集下）" if n_ok >= 2
               else "在本装置可得的观测下，PINN 不优于零训练阻抗基线")
    return {**out, "status": "判定", "verdict": verdict, "cells_ok": n_ok, "refused": None}


def shape_gate(e_base: float, e_train: float, factor: float = 0.5) -> dict:
    """The round-trip G3 as a standalone gate, with the zero-denominator refusal attached.

    Same disease as the vacuous §4 floor: `e_train <= factor * e_base` is meaningless when `e_base`
    is 0 -- it passes for any value of `e_train`, including a worse one. So a zero denominator is
    reported as "未验" and refuses to yield a PASS; it is never silently treated as a fail or a pass.
    """
    if e_base <= 0.0:
        return {"status": "未验", "refused": "denominator_zero",
                "why": f"e_base={e_base} -> the threshold {factor}*e_base is 0, so this comparison "
                       "cannot go red for any trained value. No PASS/FAIL is issued.",
                "pass": None, "threshold": None, "e_base": e_base, "e_train": e_train}
    thr = factor * e_base
    ok = e_train <= thr
    return {"status": "判定", "refused": None, "pass": bool(ok), "threshold": thr,
            "factor": factor, "e_base": e_base, "e_train": e_train}


def training_gate(loss_first: float, loss_last: float, grad_norm_last: float,
                  orders: float = 3.0, initialized_from_baseline: bool = False) -> dict:
    """G2, with the degenerate-initialisation refusal: an arm that STARTS at the baseline's own
    optimum has no information in its loss drop, so a >=3-order fall must not be read as a pass."""
    drop = loss_first / max(loss_last, 1.0e-300)
    got = math.log10(max(drop, 1.0e-300))
    base = {"loss_first": loss_first, "loss_last": loss_last, "orders": round(got, 3),
            "need_orders": orders, "grad_norm_last": grad_norm_last,
            "initialized_from_baseline": bool(initialized_from_baseline)}
    if initialized_from_baseline:
        return {**base, "status": "未验", "refused": "degenerate_initialisation", "pass": None,
                "why": "the trainable arm was seeded with the baseline's own solution: its loss fall "
                       "measures nothing, so no PASS may be issued from this cell"}
    ok = got >= orders and grad_norm_last > 0.0
    return {**base, "status": "判定", "refused": None, "pass": bool(ok)}


def rank_is_deficient_for_node_only(dof, tol=1.0e-9):
    """归宿③ test: the node-only observables carry no information about the shape DOF."""
    node = ["q_in", "q_up", "q_down", "p_junction"]
    J = rt.jacobian(dof)
    idx = [i for i, k in enumerate(rt.OBS_ORDER) if k in node]
    sub = [[row[i] for i in idx] for row in J]
    r = ib.rank_by_elimination(sub, tol=tol)
    return {"rank_node_only": r, "n_dof": len(rt.PARAMS), "rows": len(sub),
            "deficient": r < len(rt.PARAMS),
            "meaning": "with only node quantities, ANY method lands in 归宿③ -- this is an "
                       "identifiability statement about the observation set, not about the optimizer"}


def selfcheck() -> int:
    fails = []
    htrue = dof_fn(rt.H_TRUE)

    # F1 positive: a K=1 constant baseline is genuinely wrong about the taper, so e(K1) > 0 and a
    # trained arm that halves the error is discriminable.
    e_k1 = shape_error(constant_fn(1.0), htrue)["rel_l2"]
    e_b = shape_error(dof_fn({"stem": (1.0, 0.98, 0.85), "up": (1.02, 1.0), "down": (1.05,)}), htrue)["rel_l2"]
    e_c = shape_error(dof_fn({"stem": (1.0, 0.96, 0.78), "up": (1.04, 0.96), "down": (1.09,)}), htrue)["rel_l2"]
    d1 = decide([e_k1], [e_b], [e_c], monotone_ok=True, peak_ok=True)
    if not (e_k1 > 0 and d1["status"] == "判定" and "反演成立" in d1["verdict"]):
        fails.append(("F1", d1))

    # F2 negative: arm C is a copy of arm B -> delta must be exactly 0 and the verdict must be the
    # honest "not better than the baseline", never a pass by tie.
    d2 = decide([e_k1], [e_b], [e_b], monotone_ok=False, peak_ok=False)
    if not (abs(d2["delta"]) < 1.0e-15 and "不优于" in (d2["verdict"] or "")):
        fails.append(("F2", d2))

    # F3 MUST-RED (the old denominator): feed the arm-A metric, which is 0 by construction, and the
    # gate must REFUSE -- while the old formula would have passed any improvement.
    old_gate = (e_b - e_c) >= ALPHA * 0.0            # what the vacuous floor actually evaluated to
    d3 = decide([0.0], [e_b], [e_c], monotone_ok=True)
    if not (old_gate is True and d3["status"] == "未验"
            and d3["refused"] == "floor_denominator_zero" and d3["verdict"] is None):
        fails.append(("F3", {"old_gate_said_pass": old_gate, "new_gate": d3}))

    # F4 rank/归宿③: node-only observables cannot support 6 shape DOF -- measured, not asserted.
    r4 = rank_is_deficient_for_node_only(rt.H_TRUE)
    if not r4["deficient"]:
        fails.append(("F4", r4))

    # F5 the metric itself must separate the blind fit from the seeing one (positive control).
    e_blind = shape_error(constant_fn(1.0), htrue)["rel_l2"]
    e_seen = shape_error(dof_fn({"stem": (1.0, 0.96, 0.77), "up": (1.04, 0.95), "down": (1.08,)}), htrue)["rel_l2"]
    if not (e_blind > e_seen and e_blind > 0.05):
        fails.append(("F5", {"blind": e_blind, "seen": e_seen}))

    print(json.dumps({"checks": 5, "fails": [f[0] for f in fails],
                      "e_k1_constant_baseline": round(e_k1, 6),
                      "e_b": round(e_b, 6), "e_c": round(e_c, 6),
                      "F3_old_gate_would_pass": old_gate,
                      "F3_new_gate": {"status": d3["status"], "refused": d3["refused"]},
                      "F4_node_only_rank": {"rank": r4["rank_node_only"], "n_dof": r4["n_dof"]},
                      "boot_lower_95_F1": round(d1["boot_lower_95"], 8) if d1.get("boot_lower_95") else None,
                      "verdict_F1": d1["verdict"], "verdict_F2": d2["verdict"]},
                     ensure_ascii=False, indent=2))
    if fails:
        for tag, info in fails:
            print(f"[FAIL] {tag}: {json.dumps(info, ensure_ascii=False)[:400]}")
        return 1
    print("GATE SELF-CHECK ALL GREEN (5 fixtures: 2 正/反例对, 1 必红-旧分母, 1 秩归宿③, 1 度量正对照)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="the §4 discriminability gate and its fixtures")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--json", default="", help="per-seed errors file to judge: {e_k1:[], e_b:[], e_c:[]}")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    if args.json:
        d = json.loads(Path(args.json).read_text(encoding="utf-8"))
        r = decide(d["e_k1"], d["e_b"], d["e_c"], d.get("monotone_ok"), d.get("peak_ok"))
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0 if r.get("verdict") else 2
    print("nothing to do (use --selfcheck or --json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
