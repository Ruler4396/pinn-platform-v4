#!/usr/bin/env python3
"""Paired bootstrap of e(B) - e(C) for the h(xi) inversion's "resolvable improvement" (§4.2).

Scope discipline: §4 needs two conditions at once -- (1) an effect-size floor
0.20 * e(A) and (2) a paired-uncertainty lower bound.  Condition (1) belongs to route-2,
so this module does NOT implement it.  It does emit one shared guard line, `floor_status`,
because with the A arm having no shape DOF, e(A) = 0 and a floor of 0.20 * 0 is satisfied
by every imaginable improvement: a downstream PASS computed while that denominator is zero
would be a tautology wearing a gate.  So:

    floor_status = "未验：地板分母为0"  whenever mean(e(A)) == 0
    cell_verdict  is never PASS in that case, no matter what the lower bound says.

Inputs are paired readings: one row per obs_seed holding the SAME cell's error for each arm
(a mapping like {"obs_seed": 1, "A": .., "B": .., "C": ..} or a plain (A, B, C) sequence).
The training seeds of arm C do not create pairs -- §4 pairs on obs_seed only, so the caller
must hand one row per obs_seed, already reduced.

Method: statistic = mean(B - C) over rows; resample ROWS with replacement (the pair is the
unit, so B and C always move together), `reps` draws (default 2000) from stdlib `random`
with the draw seed written into the output; the 95% lower bound is the inverse empirical
CDF at alpha=0.05 of those draws, i.e. the smallest draw whose empirical CDF >= alpha.

Failure policy, because a bootstrap happily returns a number for garbage:
  * empty table or fewer than 2 rows -> raises (no 0.0, no bound)
  * non-finite / missing arm value    -> raises
  * delta == 0 on every row           -> status "未验：Δ恒为0", lower_bound is None
  * everything refused                -> cell_verdict is 未验/未通过, never PASS
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import random
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

ARMS: Tuple[str, ...] = ("A", "B", "C")
DEFAULT_REPS = 2000                    # §4.2's 2,000
DEFAULT_SEED = 20260928                # written into every artifact, so a bound is replayable
ALPHA = 0.05                           # one-sided 95% lower bound
ZERO_DELTA = "未验：Δ恒为0"
ZERO_FLOOR = "未验：地板分母为0"
MISSING_FLOOR = "未验：缺 A 臂读数，地板无从算"
OK = "OK"
UNVERIFIED = "未验"
MIN_ROWS = 2
ATOMISED_ROWS = 5


class BootstrapError(ValueError):
    """The table cannot be bootstrapped at all.  Distinct from a refused *verdict*."""


def _as_delta_rows(table: Sequence[object]) -> List[float]:
    """Extract B - C per row, in the declared arm order, and reject unusable tables."""
    if table is None or len(table) == 0:
        raise BootstrapError("空输入：配对读数表为 0 行，不给下界")
    if len(table) < MIN_ROWS:
        raise BootstrapError(f"行数 {len(table)} < {MIN_ROWS}：配对 bootstrap 无定义，"
                             f"按 §4 记未验而不是给 0")
    deltas: List[float] = []
    for i, row in enumerate(table):
        if isinstance(row, Mapping):
            missing = [a for a in ARMS if row.get(a) is None]
            vals = [row.get(a) for a in ARMS]
        else:
            seq = list(row)
            if len(seq) < len(ARMS):
                raise BootstrapError(f"第 {i} 行只有 {len(seq)} 列，需要 A/B/C 三臂")
            missing, vals = [], seq[: len(ARMS)]
        if missing:
            raise BootstrapError(f"第 {i} 行缺臂 {missing}：缺一臂的配对行不进重抽")
        try:
            a, b, c = (float(v) for v in vals)
        except (TypeError, ValueError) as exc:
            raise BootstrapError(f"第 {i} 行的读数不是数：{vals}") from exc
        if not all(math.isfinite(v) for v in (a, b, c)):
            raise BootstrapError(f"第 {i} 行含非有限值 {(a, b, c)}")
        deltas.append(b - c)
    return deltas


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _inverse_cdf(sorted_draws: Sequence[float], alpha: float) -> float:
    """Smallest draw whose empirical CDF reaches alpha (right-continuous inverse).

    Stated as a formula so the self-test can reproduce it by enumeration with the SAME
    definition -- comparing a Monte-Carlo quantile against a differently-defined analytic
    one would be a tolerance fight, not a check.
    """
    r = len(sorted_draws)
    idx = min(max(int(math.ceil(alpha * r)) - 1, 0), r - 1)
    return sorted_draws[idx]


def resample_deltas(deltas: Sequence[float], reps: int, rng: random.Random
                    ) -> List[float]:
    """Row-wise paired resampling: indices are drawn once per replicate and applied to both
    arms, which is what makes this *paired* rather than two independent bootstraps."""
    n = len(deltas)
    out = []
    for _ in range(reps):
        picks = [deltas[rng.randrange(n)] for _ in range(n)]
        out.append(_mean(picks))
    return out


def paired_lower_bound(table: Sequence[object], reps: int = DEFAULT_REPS,
                       seed: int = DEFAULT_SEED, alpha: float = ALPHA) -> dict:
    """The whole computation, including the two refusal paths and the shared floor guard."""
    if reps < 100:
        raise BootstrapError(f"reps={reps} 太少（<100）：分位数由重抽次数决定，别拿它当读数")
    if not (0.0 < alpha < 0.5):
        raise BootstrapError(f"alpha={alpha} 不在 (0, 0.5)：这里要的是单侧下界")
    deltas = _as_delta_rows(table)
    e_a = [float(row["A"] if isinstance(row, Mapping) else list(row)[0]) for row in table]
    n = len(deltas)
    obs = _mean(deltas)
    floor_mean = _mean(e_a)

    out: dict = {
        "n_pairs": n, "reps": reps, "seed": seed, "alpha": alpha,
        "arms": list(ARMS), "statistic": "mean(e_B - e_C) over paired obs_seed rows",
        "delta_obs": obs, "delta_min": min(deltas), "delta_max": max(deltas),
        "deltas": deltas, "e_a_mean": floor_mean,
        "lower_bound": None, "upper_check": None,
        "delta_status": OK, "floor_status": OK, "cell_verdict": "未验",
        "reasons": [],
    }

    # guard 1 (shared with §4.1): a zero floor denominator makes "improvement" unfalsifiable
    if floor_mean == 0.0:
        out["floor_status"] = ZERO_FLOOR
        out["reasons"].append("e(A) 均值 = 0（A 臂没有形状自由度）=> 0.20*e(A) 恒被满足，"
                              "地板不bind；总判决不许 PASS")
    # guard 2: no signal at all
    if all(d == 0.0 for d in deltas):
        out["delta_status"] = ZERO_DELTA
        out["reasons"].append("所有配对行 Δ=e(B)-e(C) 逐位为 0 => 无改进可判，"
                              "下界不产出（不得读成 0 或 PASS）")
    if out["delta_status"] != OK or out["floor_status"] != OK:
        # the verdict vocabulary stays small (PASS / 未通过 / 未验); which guard fired is
        # readable from delta_status / floor_status / reasons, so it is not encoded twice
        out["cell_verdict"] = UNVERIFIED
        return out

    rng = random.Random(seed)
    draws = sorted(resample_deltas(deltas, reps, rng))
    out["lower_bound"] = _inverse_cdf(draws, alpha)
    out["upper_check"] = _inverse_cdf(draws, 1.0 - alpha)      # printed, not part of the verdict
    out["resample_min"], out["resample_max"] = draws[0], draws[-1]
    out["distinct_draws"] = len(set(draws))
    if n < ATOMISED_ROWS:
        # same lesson as the peak_offset cell: with few pairs the replicate means come from
        # n^n arrangements, so the "bound" takes a handful of discrete values no matter how
        # many replicates are drawn.  Recording it stops a reader treating it as continuous.
        out["reasons"].append(f"caveat: n_pairs={n} < {ATOMISED_ROWS}，重抽均值是原子的"
                              f"（本次 {out['distinct_draws']} 个不同取值），"
                              f"增大 reps 不会让它变连续")
    if out["lower_bound"] > 0.0 and out["floor_status"] == OK:
        out["cell_verdict"] = "PASS"
    else:
        out["cell_verdict"] = "未通过" if out["floor_status"] == OK else "未验"
    return out


def format_row(res: dict, cell: str = "cell") -> str:
    lb = res.get("lower_bound")
    return (f"[boot] cell={cell} n_pairs={res.get('n_pairs')} reps={res.get('reps')} "
            f"seed={res.get('seed')} alpha={res.get('alpha')} "
            f"delta_obs={_num(res.get('delta_obs'))} "
            f"lower95={_num(lb)} upper95={_num(res.get('upper_check'))} "
            f"e_a_mean={_num(res.get('e_a_mean'))} delta_status={res.get('delta_status')} "
            f"floor_status={res.get('floor_status')} verdict={res.get('cell_verdict')} "
            f"reason={'; '.join(res.get('reasons', []))}")


def _num(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.6g}"


def _load_table(path: str) -> List[dict]:
    """CSV with a header naming the arms (extra columns like obs_seed are carried along)."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if any(k.strip() in ARMS for k in r):
            out.append({k.strip(): (None if (v is None or v == "") else float(v))
                        for k, v in r.items()})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="paired bootstrap of e(B)-e(C) (§4.2 only)")
    ap.add_argument("--table", default="", help="CSV with columns A,B,C (one row per obs_seed)")
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--cell", default="shape_l2", help="label printed back for traceability")
    ap.add_argument("--json", default="", help="write the result outside the repository")
    args = ap.parse_args(argv)
    if args.json:
        root = _repo_root()
        out = pathlib.Path(args.json)
        if root in out.resolve().parents or str(out).startswith(str(root)):
            print(f"[boot] REFUSED: 不许写进仓内: {out}")
            return 2
    try:
        table = _load_table(args.table) if args.table else []
        res = paired_lower_bound(table, reps=args.reps, seed=args.seed)
    except BootstrapError as exc:
        print(f"[boot] ERROR: {exc}")
        return 1
    print(format_row(res, args.cell))
    if args.json:
        from pathlib import Path
        text = json.dumps(res, ensure_ascii=False, indent=2) + "\n"
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_bytes(text.encode("utf-8"))
        import hashlib
        digest = hashlib.sha256(Path(args.json).read_bytes()).hexdigest()
        print(f"[boot] artifact_selfcert: path={args.json} "
              f"bytes={Path(args.json).stat().st_size} sha256={digest}")
    return 0 if res["cell_verdict"] == "PASS" else 1


def _repo_root():
    return pathlib.Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    sys.exit(main())
