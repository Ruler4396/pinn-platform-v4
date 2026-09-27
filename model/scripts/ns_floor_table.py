#!/usr/bin/env python3
"""T6-Re: the 4 x 3 floor table -- per level (Re) x per field, the printing floor as a
repo artefact, so "12/12 过 1e-10" is either backed by a file or explicitly not claimed.

Why this file exists: the cost table says the NS truth reaches >=10 significant digits, but
until now the only published figure was one row of three fields computed at the Stokes
magnitudes.  Twelve cells (4 levels x 3 fields) were never written down anywhere, so the
sentence "12/12 pass" had no artefact behind it and the 统括官 correctly refused to cite it.

What each cell is.  The six-digit printer's information content is fixed by two things, and
this script uses one measured and one read-from-the-artefact:
  * the emitted width N, parsed out of that level's own `.edp` (`finalize.hi_decimals_from_edp`)
    -- absolute bound = 0.5 * 10**-(N+6), because `lo < 1e-N` and `lo`'s own six digits are a
    relative 5e-7 of that;
  * the level's measured |max| of that field, taken from `<case>_raw.csv` if that file exists.
    A cell that has no product yet is marked ESTIMATED and scored on the shipped Stokes
    magnitude instead, and the headline refuses to claim 12/12.  A bound computed at an
    unmeasured magnitude is a rule statement, not a reading.

`--selfcheck` carries the controls: the same-magnitude replay must agree with
`finalize_ns_truth.py --selfcheck` to the last digit (one ruler, two scripts), and a
deliberately coarsened width must push a cell over the bound (so the gate can redden).

Usage:
    python3 ns_floor_table.py                       # print the table
    python3 ns_floor_table.py --write PATH           # same, as markdown
    python3 ns_floor_table.py --json PATH            # machine-readable
    python3 ns_floor_table.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):          # Windows console is GBK: `=>` survives, `⇒` kills it
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import finalize_ns_truth as fz                                    # noqa: E402

LEVELS = fz.LEVELS                                                 # 1e-3, 1, 10, 50
LIMIT_REL = fz.FLOOR_LIMIT_REL                                     # 1e-10, unchanged


def measured_field_maxima(level_dir: Path, level: str) -> dict[str, float] | None:
    """|max| per field from that level's own raw stream, or None when it has not run."""
    raw = level_dir / f"C-base_ns_re{level}_raw.csv"
    if not raw.is_file():
        return None
    with raw.open(encoding="utf-8", newline="") as fh:
        head = fh.readline().strip().split(",")
        idx = {f: head.index(f) for f in fz.FIELDS}
        maxima = {f: 0.0 for f in fz.FIELDS}
        n = 0
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) <= max(idx.values()):
                continue
            n += 1
            for f, i in idx.items():
                maxima[f] = max(maxima[f], abs(float(parts[i])))
    if n == 0:
        return None
    return maxima


def stokes_field_maxima() -> dict[str, float]:
    return {f: max(abs(v) for v in fz.field_magnitudes()[f]) for f in fz.FIELDS}


def absolute_bound(decimals: int) -> float:
    """The largest error the hi/lo split can carry, given a six-digit printer."""
    return 0.5 * 10.0 ** -(decimals + fz.PRINT_DIGITS)


def build_table() -> list[dict]:
    fallback = stokes_field_maxima()
    cells: list[dict] = []
    for level in LEVELS:
        level_dir = fz.CFD / f"C-base_ns_re{level}"
        edp = fz.edp_path(level)
        widths = (fz.hi_decimals_from_edp(edp.read_text(encoding="utf-8").replace("\r\n", "\n"))
                  if edp.is_file() else None)
        measured = measured_field_maxima(level_dir, level)
        for f in fz.FIELDS:
            n = widths[f] if widths else None
            scale = (measured or fallback)[f]
            cell = {"level": level, "field": f, "edp_present": widths is not None,
                    "hi_decimals": n, "scale": scale,
                    "scale_source": "measured" if measured else "ESTIMATED(stokes)"}
            if n is None:
                cell.update(absolute=None, relative=None, digits=None,
                            verdict="NO_EDP", claim="not claimed")
            else:
                absb = absolute_bound(n)
                rel = absb / max(scale, 1e-300)
                cell.update(absolute=absb, relative=rel,
                            digits=(-math.log10(rel) if rel > 0 else math.inf),
                            verdict="PASS" if rel <= LIMIT_REL else "FAIL",
                            claim=("citable" if cell["scale_source"] == "measured"
                                   else "rule-only, magnitude not measured"))
            cells.append(cell)
    return cells


def headline(cells: list[dict]) -> dict:
    twelve = len(cells) == 4 * 3
    passed = [c for c in cells if c["verdict"] == "PASS"]
    measured = [c for c in cells if c["scale_source"] == "measured"]
    return {"cells": len(cells), "twelve_by_construction": twelve,
            "pass": len(passed), "measured_cells": len(measured),
            "claim_12_of_12": bool(twelve and len(passed) == 12 and len(measured) == 12),
            "why_not": ("all 12 cells PASS at measured magnitudes"
                        if twelve and len(passed) == 12 and len(measured) == 12 else
                        f"{12 - len(measured)} cell(s) scored at ESTIMATED magnitudes "
                        f"(their level's _raw.csv is not in this checkout)")}


def render_markdown(cells: list[dict], head: dict) -> str:
    out = ["# T6-Re 逐场逐档打印地板表（4 档 × 3 场 = 12 格）", "",
           f"限 `rel <= {LIMIT_REL:.0e}`（派单口径，数值未动）。宽度 N 从每档自己的 `.edp` 读出，"
           f"幅度优先从该档 `_raw.csv` 实测；缺产物时退回入库 Stokes 量级并标 "
           f"`ESTIMATED`，此时**不得声称 12/12**。", "",
           "| Re | 场 | N (from .edp) | \\|max\\| | 绝对地板 | 相对地板 | 有效数字 | 幅度来源 | 判决 |",
           "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for c in cells:
        out.append(f"| {c['level']} | {c['field']} | {c['hi_decimals'] if c['hi_decimals'] else '—'} "
                   f"| {c['scale']:.6g} | "
                   f"{('%.2e' % c['absolute']) if c['absolute'] else '—'} | "
                   f"{('%.2e' % c['relative']) if c['relative'] else '—'} | "
                   f"{('%.1f' % c['digits']) if c['digits'] else '—'} | {c['scale_source']} "
                   f"| {c['verdict']} |")
    out += ["", f"**12/12 是否可声称：{'是' if head['claim_12_of_12'] else '否'}** —— "
            f"PASS {head['pass']}/{head['cells']}，实测幅度格 {head['measured_cells']}/12；"
            f"理由：{head['why_not']}。"]
    return "\n".join(out) + "\n"


def selfcheck() -> int:
    """Controls for the ruler itself: agreement with finalize, and a cell that must go red."""
    rc = 0
    fallback = stokes_field_maxima()
    # 1. the same ruler, two scripts: this file's bound at the Stokes magnitudes must equal
    #    what finalize_ns_truth measures by replaying the printing rule on real values.
    mags = fz.field_magnitudes()
    widths = fz.hi_decimals_from_edp(fz.edp_path("1").read_text(encoding="utf-8")
                                     .replace("\r\n", "\n"))
    for f in fz.FIELDS:
        replay = max(abs(fz.roundtrip(x, widths[f]) - x)
                     for x in fz.within_rounding_cell(mags[f]))
        bound = absolute_bound(widths[f])
        ok = abs(replay - bound) / bound <= 0.05       # the bound is an envelope, replay the max
        rc |= 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {f}: 规则上界 {bound:.3e} vs 实测回放 "
              f"{replay:.3e}（same ruler, two scripts，比值 {replay / bound:.3f}）")
    # 2. positive control: coarsen one width by two digits, that cell must go red
    bad = absolute_bound(widths["u_star"] - 2) / max(fallback["u_star"], 1e-300)
    ok = bad > LIMIT_REL
    rc |= 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] control: N 降两档后 u_star 相对地板 {bad:.2e} > "
          f"{LIMIT_REL:.0e} -> 判决会红（不是恒绿）")
    # 3. negative control: a level with no product must NOT be counted as measured
    head = headline(build_table())
    ok = head["measured_cells"] < 12 and not head["claim_12_of_12"]
    rc |= 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] control: 本机无档产物时 measured_cells="
          f"{head['measured_cells']}/12 且 claim_12_of_12=False（缺证据就不许声称）")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="T6-Re 4x3 打印地板表")
    ap.add_argument("--write", metavar="PATH")
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    cells = build_table()
    head = headline(cells)
    md = render_markdown(cells, head)
    if args.write:
        out = Path(args.write)
        if (HERE.parents[1] in out.resolve().parents) and out.suffix not in (".md", ".csv", ".json"):
            print(f"[FAIL] 拒绝把自测产物写成非文档扩展名: {out}")
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"wrote {out}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"cells": cells, "headline": head},
                                              ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        print(f"wrote {args.json}")
    if not args.write and not args.json:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
