#!/usr/bin/env python3
"""Shape-error metrics for the route-2 h(xi) inversion (pre-registration §3, 三格).

Scoring only -- no arm runs here and no verdict is invented here.  The three cells come out
of ONE entry point so two arms can never be scored with three different conventions:

    shape_errors(pred, ref, xi, dp_branch=None) -> dict

with exactly the three names in METRIC_NAMES.  A fourth quantity cannot leak in, and the
banned list of §3 (pointwise MSE of a numerical second derivative, anything that
differentiates a reconstructed field again) is therefore not merely unimplemented, it is
not reachable from the entry point.  selftest pins that.

Definitions are the pre-registration's, verbatim:
  shape_l2      -- divide each branch's pressure points by that branch's |dp_branch|, then
                  take the RMS of the difference.  Refuses (None + 未验) on a zero
                  denominator; a zero denominator would make every error "infinite" or
                  every comparison "0/0", and the §4 floor was already voided once for
                  exactly that shape (0.20 x e(A) with e(A)=0 is always true).
  peak_offset   -- |xi(argmax pred) - xi(argmax ref)|, and -1.0 -- never 0 -- when either
                  series has no INTERIOR maximum.  -1 means "not taken", so a caller must
                  read peak_offset_taken before treating the number as an error.
  monotone_sign -- how many of the n-1 successive-difference signs agree between the two
                  series.  A count, not a fraction: 2/2 and 2/8 are different statements,
                  so `steps` travels with it.  Discrete and precision-independent by
                  construction (no tolerance, no threshold).

Identical inputs are a trap this module has to name out loud: shape_l2 is then exactly 0
and monotone_sign is perfect, which reads like a result but is only the statement "these
are the same sequence".  Hence `trivial` / `score_eligible` in every return.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

METRIC_NAMES: Tuple[str, ...] = ("shape_l2", "peak_offset", "monotone_sign")
PEAK_NOT_TAKEN = -1.0                 # §3: 记 -1 = 未取, 不记 0
DENOM_ZERO = "未验：分母为 0"
MIN_POINTS = 3                        # §3: 每支 >= 3 点; below that no cell is measurable
UNVERIFIED = "未验"


def _bad(value: float) -> bool:
    return not math.isfinite(value)


def _preflight(pred: Sequence[float], ref: Sequence[float], xi: Sequence[float]) -> str:
    if len(pred) != len(ref) or len(pred) != len(xi):
        return f"未验：长度不一致 pred={len(pred)} ref={len(ref)} xi={len(xi)}"
    if len(pred) < MIN_POINTS:
        return f"未验：点数 {len(pred)} < {MIN_POINTS}（§3 要求每支 >=3 点）"
    if any(_bad(v) for v in list(pred) + list(ref) + list(xi)):
        return "未验：输入含非有限值"
    if any(b <= a for a, b in zip(xi, xi[1:])):
        return "未验：xi 网格不严格递增，差分符号与峰位无定义"
    return ""


def _denominator(ref: Sequence[float], dp_branch: Optional[float]) -> Tuple[float, str, str]:
    """The branch pressure-drop scale, or an explicit refusal.

    Prefer the caller's |dp_branch| (it is one of the three integral quantities, so all arms
    share it).  Deriving it from the reference span is allowed but must say so, because a
    derived scale makes shape_l2 a *relative-to-that-branch's-own-span* number and is not
    interchangeable with the declared one.
    """
    if dp_branch is not None:
        if _bad(dp_branch):
            return 0.0, "", UNVERIFIED + "：dp_branch 非有限值"
        scale = abs(float(dp_branch))
        source = "declared:|dp_branch|"
    else:
        scale = abs(max(ref) - min(ref))
        source = "derived:ref_span"
    if scale == 0.0:
        return 0.0, source, DENOM_ZERO
    return scale, source, ""


def _interior_argmax(series: Sequence[float], xi: Sequence[float]) -> Optional[float]:
    """xi of the maximum, only when the maximum is at an interior index.

    An endpoint maximum means the series is monotone through its window, so there is no
    peak position to compare -- §3 says record -1 (not taken), and 0 would be a lie
    because 0 is also the value of "both peaks in the same place".
    """
    best = max(range(len(series)), key=lambda i: series[i])
    if best == 0 or best == len(series) - 1:
        return None
    if sum(1 for v in series if v == series[best]) > 1:
        return None                                 # a plateau: the peak position is not unique
    return xi[best]


def _signs(series: Sequence[float]) -> List[int]:
    out = []
    for a, b in zip(series, series[1:]):
        out.append(0 if a == b else (1 if b > a else -1))
    return out


def shape_errors(pred: Sequence[float], ref: Sequence[float], xi: Sequence[float],
                 dp_branch: Optional[float] = None) -> dict:
    """The single entry point for all three cells.  Never raises on bad data: it returns
    status=未验 with a reason, and every refused metric is None rather than 0.0, because a
    refused measurement that reads as a perfect score is the failure mode §4 was written
    against."""
    p = [float(v) for v in pred]
    r = [float(v) for v in ref]
    g = [float(v) for v in xi]
    bad = _preflight(p, r, g)
    out: dict = {
        "xi": g, "n": len(g), "steps": max(len(g) - 1, 0),
        "shape_l2": None, "peak_offset": None, "monotone_sign": None,
        "peak_offset_taken": False, "peak_pred": None, "peak_ref": None,
        "agreement": None, "scale": None, "scale_source": None,
        "trivial": False, "score_eligible": False,
        "status": UNVERIFIED, "reasons": [], "notes": {}, "metrics": METRIC_NAMES,
    }
    if bad:
        out["reasons"] = [bad]     # entry-level: kills all three cells at once
        return out

    scale, source, refuse = _denominator(r, dp_branch)
    out["scale"], out["scale_source"] = scale or None, source or None
    if refuse:
        out["reasons"] = [refuse]
        return out

    # cell 1
    diffs = [(a - b) / scale for a, b in zip(p, r)]
    out["shape_l2"] = math.sqrt(sum(d * d for d in diffs) / len(diffs))

    # cell 2
    xp, xr = _interior_argmax(p, g), _interior_argmax(r, g)
    out["peak_pred"], out["peak_ref"] = xp, xr
    if xp is None or xr is None:
        out["peak_offset"] = PEAK_NOT_TAKEN
        out["peak_offset_taken"] = False
        missing = [n for n, v in (("pred", xp), ("ref", xr)) if v is None]
        out["notes"]["peak_offset"] = [
            "peak_offset 未取：分布无内部极大值（或为平台），记 -1 不记 0；缺的一侧="
            + "+".join(missing)]
    else:
        out["peak_offset"] = abs(xp - xr)
        out["peak_offset_taken"] = True

    # cell 3
    sp, sr = _signs(p), _signs(r)
    out["agreement"] = sum(1 for a, b in zip(sp, sr) if a == b)
    out["monotone_sign"] = out["agreement"]

    out["trivial"] = all(a == b for a, b in zip(p, r))
    out["status"] = "OK"
    out["score_eligible"] = not out["trivial"]
    if out["trivial"]:
        note = ("平凡相等：pred 与 ref 逐点相同 => shape_l2=0 与 monotone_sign 满格都不是精度，"
                "只是同一个序列被自己减了一遍；不得当得分")
        for cell in METRIC_NAMES:
            out["notes"][cell] = [note]
    out["sign_pred"] = sp
    out["sign_ref"] = sr
    return out


def relative_improvement(e_candidate: Optional[float], e_reference: Optional[float],
                         floor_frac: float = 0.20) -> dict:
    """Paired comparison helper for §4's effect-size floor: (e_ref - e_cand) / e_ref.

    The reference error is the denominator, so a zero it must refuse rather than pass --
    that is the e(A)=0 defect in the pre-registration's own words: with a zero floor every
    infinitesimal change counts as "resolvable".  Returns value=None + 未验 in that case.
    """
    out = {"value": None, "floor": None, "meets_floor": None, "status": UNVERIFIED,
           "reasons": [], "floor_frac": floor_frac}
    if e_candidate is None or e_reference is None:
        out["reasons"] = [f"未验：任一臂误差为 None（cand={e_candidate} ref={e_reference}）"]
        return out
    c, ref = float(e_candidate), float(e_reference)
    if _bad(c) or _bad(ref):
        out["reasons"] = ["未验：输入含非有限值"]
        return out
    out["floor"] = floor_frac * abs(ref)
    if ref == 0.0:
        out["reasons"] = [DENOM_ZERO + "（基线误差 e_reference=0 => 地板恒真，不可判改进）"]
        return out
    out["value"] = (ref - c) / abs(ref)
    out["meets_floor"] = bool((ref - c) >= out["floor"])
    out["status"] = "OK"
    return out


def format_row(cell: str, branch: str, res: dict) -> str:
    """One printed line per (cell, branch), always naming status and reason together so a
    refused measurement cannot be pasted as a number."""
    if cell not in METRIC_NAMES:
        raise KeyError(f"unknown cell {cell!r}; this module exposes only {METRIC_NAMES}")
    v = res.get(cell)
    shown = "n/a" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))
    extra = ""
    if cell == "monotone_sign":
        extra = f" agreement={res.get('agreement')}/{res.get('steps')}"
    if cell == "peak_offset":
        extra = f" taken={res.get('peak_offset_taken')}"
    if cell == "shape_l2":
        extra = (f" scale={res.get('scale'):.6g}" if res.get("scale") is not None else " scale=n/a")
    why = res.get("reasons") or (res.get("notes") or {}).get(cell) or [""]
    return (f"[shape] cell={cell} branch={branch} value={shown}{extra} "
            f"status={res.get('status')} trivial={res.get('trivial')} "
            f"score_eligible={res.get('score_eligible')} reason={why[-1]!r}")


def format_all(pred: Sequence[float], ref: Sequence[float], xi: Sequence[float],
               branch: str = "branch", dp_branch: Optional[float] = None) -> List[str]:
    res = shape_errors(pred, ref, xi, dp_branch)
    return [format_row(c, branch, res) for c in METRIC_NAMES]


def main() -> int:
    """Demonstration on the pre-registration's own station layout: xi = 0.30/0.60/0.90."""
    xi = (0.30, 0.60, 0.90)
    ref = (100.0, 60.0, 20.0)                       # monotone decreasing, endpoint max => no interior peak
    pred = (98.0, 63.0, 17.0)
    lines = format_all(pred, ref, xi, branch="stem", dp_branch=80.0)
    for line in lines:
        print(line)
    print(format_row("shape_l2", "trivial-equal", shape_errors(ref, ref, xi, dp_branch=80.0)))
    print(format_row("shape_l2", "zero-denom", shape_errors(pred, ref, xi, dp_branch=0.0)))
    print(f"[shape] relative_improvement ref=0 -> {relative_improvement(0.01, 0.0)}")
    print("[shape] cells are exactly " + ",".join(METRIC_NAMES) +
          "; combined e() and the three-arm predicate belong to the predicate owner, not here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
