#!/usr/bin/env python3
"""Second-implementation cross-check: three INTEGRAL quantities, computed once and applied
to two solvers' outputs, so the comparison cannot drift by definition.

Pre-registration: paper-route2/第二实现交叉验证-预注册-20260927.md (统括官 2026-09-27 09:0x).
Judgement frozen there and echoed here: |rel diff| <= 1% on (1) pressure drop dp,
(2) outlet flux Q, (3) flow resistance R = dp/Q.  Nodal fields are reported, never judged.
Different implementations cannot agree digit-for-digit -- the repo's "5/5 diff=0" statement
holds only for the SAME solver re-run, and must not be extrapolated to this script's output.

Why the quantities live in this file rather than in each solver's postprocessing: if FreeFEM
computes Q by quadrature and dolfinx by a trapezoid over its own node list, a >1% disagreement
would be unattributable -- §2 of the pre-registration sends you to "definitions first", and the
cheapest way to honour that is one definition, two inputs.  Both sides are therefore given as
CSVs of nodal data with a boundary marker, and both go through `integrals()`.

Column/tag names are arguments, not assumptions, because the two meshes will not agree on
them: `--inlet-tag`, `--outlet-tag`, `--wall-tag`, and the column names default to the shipped
FreeFEM truth (`x_star,y_star,u_star,v_star,p_star,bc_tag`, tags 1/2/3).

Self-controls (both required by project rules):
  * positive: perturb ONE quantity of one side by 3% -- the run must FAIL;
  * negative: compare a file against itself -- every relative difference must be exactly 0.000%;
  * ruler: `--selfcheck` must say out loud that the same code on both sides cannot manufacture
    a difference, so a nonzero dp/Q/R gap means the two solutions really differ.

Usage:
    python3 crosscheck_second_impl.py --freefem A.csv --other B.csv
    python3 crosscheck_second_impl.py --freefem A.csv --other B.csv --json out.json
    python3 crosscheck_second_impl.py --selfcheck
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):        # Windows consoles are often GBK, not UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REL_LIMIT = 0.01                              # frozen in the pre-registration, section 1
MESH_REL_LIMIT = 0.02                         # section 1's mesh-refinement gate
QUANTITIES = ("dp", "Q", "R")


def read_nodes(path: Path, cols: dict) -> list[dict]:
    """Load a nodal CSV into [{'x','y','u','v','p','tag'}...] with the names mapped once."""
    with path.open(encoding="utf-8", newline="") as fh:
        rdr = csv.DictReader(fh)
        missing = [c for c in cols.values() if c not in (rdr.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name}: missing column(s) {missing}; "
                             f"present={rdr.fieldnames}. Passing wrong column names would "
                             f"silently compute a different quantity on each side.")
        out = []
        for row in rdr:
            out.append({"x": float(row[cols["x"]]), "y": float(row[cols["y"]]),
                        "u": float(row[cols["u"]]), "v": float(row[cols["v"]]),
                        "p": float(row[cols["p"]]), "tag": row[cols["tag"]].strip()})
    if not out:
        raise ValueError(f"{path.name}: no rows")
    return out


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty selection: that boundary does not exist under these tags, "
                         "and averaging nothing would report 0.0 as if it were a measurement")
    return sum(values) / len(values)


def pressure_drop(nodes: list[dict], inlet_tag: str, outlet_tag: str) -> float:
    """mean p on the inlet boundary minus mean p on the outlet boundary."""
    return (_mean([n["p"] for n in nodes if n["tag"] == inlet_tag])
            - _mean([n["p"] for n in nodes if n["tag"] == outlet_tag]))


def outlet_flux(nodes: list[dict], outlet_tag: str) -> float:
    """int u.n ds over the outlet section, trapezoid in y, for the plane x = x_max.

    The section is taken at the maximum x of the whole mesh so it is the same geometric
    plane whichever tool produced the file; the normal is +x, hence u.n = u. Sorted by y so
    the trapezoid is along the section, not along file order.
    """
    x_exit = max(n["x"] for n in nodes)
    on_exit = sorted((n for n in nodes if n["x"] == x_exit), key=lambda n: n["y"])
    if len(on_exit) < 2:
        raise ValueError(f"only {len(on_exit)} node(s) on the outlet plane x={x_exit}: "
                         f"a flux integral needs at least two, and reporting 0 would be a lie")
    q = 0.0
    for a, b in zip(on_exit, on_exit[1:]):
        q += 0.5 * (a["u"] + b["u"]) * (b["y"] - a["y"])
    return abs(q)


def integrals(nodes: list[dict], inlet_tag: str, outlet_tag: str) -> dict:
    """The three judged quantities, from one definition."""
    dp = pressure_drop(nodes, inlet_tag, outlet_tag)
    q = outlet_flux(nodes, outlet_tag)
    if q == 0.0:
        raise ValueError("outlet flux is exactly 0 -- R = dp/Q is undefined, and 0 flux from "
                         "a driven flow means the wrong nodes were selected")
    return {"dp": dp, "Q": q, "R": dp / q}


def compare(a: dict, b: dict) -> tuple[dict, bool]:
    rows, ok = {}, True
    for k in QUANTITIES:
        denom = max(abs(a[k]), 1e-300)
        rel = abs(b[k] - a[k]) / denom
        rows[k] = {"freefem": a[k], "other": b[k], "rel_diff": rel,
                   "pass": rel <= REL_LIMIT}
        ok = ok and rel <= REL_LIMIT
    return rows, ok


def nodal_field_report(a_nodes: list[dict], b_nodes: list[dict]) -> dict:
    """Diagnostic only (the pre-registration says nodal fields never decide the verdict)."""
    if len(a_nodes) != len(b_nodes):
        return {"applicable": False,
                "reason": f"different node counts ({len(a_nodes)} vs {len(b_nodes)}): the two "
                          f"meshes are not the same, so a pointwise gap would be a "
                          f"resampling artefact. Integral comparison is unaffected."}
    gaps = [max(abs(x["u"] - y["u"]), abs(x["v"] - y["v"]), abs(x["p"] - y["p"]))
            for x, y in zip(a_nodes, b_nodes)]
    scale = max(max(abs(n["u"]), abs(n["v"]), abs(n["p"])) for n in a_nodes) or 1.0
    gaps.sort()
    return {"applicable": True, "max_rel": max(gaps) / scale,
            "median_rel": gaps[len(gaps) // 2] / scale,
            "note": "same node order assumed; reported, never judged"}


def run(freefem: Path, other: Path, tags: dict, cols: dict) -> tuple[int, dict]:
    a_nodes = read_nodes(freefem, cols)
    b_nodes = read_nodes(other, cols)
    a = integrals(a_nodes, tags["inlet"], tags["outlet"])
    b = integrals(b_nodes, tags["inlet"], tags["outlet"])
    rows, ok = compare(a, b)
    print(f"side A (reference) = {freefem.name}\nside B (second impl) = {other.name}")
    print(f"tags: inlet={tags['inlet']} outlet={tags['outlet']} "
          f"(both sides read with the SAME code path -- a nonzero gap below is a real "
          f"difference between the two solutions, not a definitional drift)")
    for k in QUANTITIES:
        r = rows[k]
        print(f"  {k:2s}  A={r['freefem']:+.9g}  B={r['other']:+.9g}  "
              f"rel={r['rel_diff']:.4%}  {'PASS' if r['pass'] else 'FAIL'} (limit {REL_LIMIT:.0%})")
    nodal = nodal_field_report(a_nodes, b_nodes)
    if nodal.get("applicable"):
        print(f"  nodal (diagnostic, not judged): max {nodal['max_rel']:.3e}  "
              f"median {nodal['median_rel']:.3e}")
    else:
        print(f"  nodal: NOT APPLICABLE -- {nodal['reason']}")
    print(f"[{'PASS' if ok else 'FAIL'}] cross-check verdict (three integrals, each <= "
          f"{REL_LIMIT:.0%})")
    return (0 if ok else 1), {"quantities": rows, "nodal": nodal, "pass": ok,
                              "rel_limit": REL_LIMIT, "tags": tags,
                              "files": {"freefem": str(freefem), "other": str(other)}}


def _synthetic_rows(n: int = 41) -> list[dict]:
    """A toy outlet plane + tagged inlet/outlet, used only by --selfcheck."""
    rows = []
    for i in range(n):
        y = -0.5 + i / (n - 1)
        rows.append({"x": 0.0, "y": y, "u": 1.5 * (1 - 4 * y * y), "v": 0.0,
                     "p": 1.0, "tag": "1"})
        rows.append({"x": 4.0, "y": y, "u": 1.5 * (1 - 4 * y * y), "v": 0.0,
                     "p": -3.0, "tag": "2"})
    return rows


def _write_csv(path: Path, rows: list[dict], scale: dict | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("x_star,y_star,u_star,v_star,p_star,bc_tag\n")
        for r in rows:
            s = scale or {}
            fh.write(f"{r['x']},{r['y']},{r['u'] * s.get('u', 1.0)},{r['v']},{r['p'] * s.get('p', 1.0)},{r['tag']}\n")


def selfcheck() -> int:
    import tempfile
    rc = 0
    root = Path(tempfile.gettempdir()) / "crosscheck_selfcheck"
    root.mkdir(parents=True, exist_ok=True)
    base = root / "A.csv"
    same = root / "A_copy.csv"
    bad = root / "B_3pct.csv"
    rows = _synthetic_rows()
    _write_csv(base, rows)
    _write_csv(same, rows)
    _write_csv(bad, rows, scale={"p": 1.03})          # one quantity moved by 3%
    tags = {"inlet": "1", "outlet": "2"}
    cols = {"x": "x_star", "y": "y_star", "u": "u_star", "v": "v_star",
            "p": "p_star", "tag": "bc_tag"}        # the shipped FreeFEM header names

    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc_zero, _ = run(base, same, tags, cols)
    ok = rc_zero == 0 and all(s == 0.0 for s in
                              [abs(integrals(read_nodes(same, cols), "1", "2")[k]
                                   - integrals(read_nodes(base, cols), "1", "2")[k])
                               for k in QUANTITIES])
    rc |= 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] negative control: a file against itself gives "
          f"rel=0.000% on all three (rc={rc_zero})")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc_bad, det = run(base, bad, tags, cols)
    ok = rc_bad == 1 and det["quantities"]["dp"]["rel_diff"] > REL_LIMIT
    rc |= 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] positive control: nudging p by 3% reddens the gate "
          f"(dp rel={det['quantities']['dp']['rel_diff']:.3%}, rc={rc_bad}) -- the limit has "
          f"teeth")
    ok2 = det["quantities"]["Q"]["rel_diff"] == 0.0
    rc |= 0 if ok2 else 1
    print(f"[{'PASS' if ok2 else 'FAIL'}] ruler control: the untouched quantity stayed at "
          f"exactly 0.000% => the script applies one definition to both sides")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="three-integral cross-check between two solvers")
    ap.add_argument("--freefem", type=Path, help="the reference CSV (repo FreeFEM truth)")
    ap.add_argument("--other", type=Path, help="the second implementation's CSV")
    ap.add_argument("--inlet-tag", default="1")
    ap.add_argument("--outlet-tag", default="2")
    ap.add_argument("--col-x", default="x_star"); ap.add_argument("--col-y", default="y_star")
    ap.add_argument("--col-u", default="u_star"); ap.add_argument("--col-v", default="v_star")
    ap.add_argument("--col-p", default="p_star"); ap.add_argument("--col-tag", default="bc_tag")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    if not args.freefem or not args.other:
        print("[INDETERMINATE] need --freefem and --other; nothing was compared")
        return 2
    cols = {"x": args.col_x, "y": args.col_y, "u": args.col_u, "v": args.col_v,
            "p": args.col_p, "tag": args.col_tag}
    rc, detail = run(args.freefem, args.other, {"inlet": args.inlet_tag,
                                                "outlet": args.outlet_tag}, cols)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(detail, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        print(f"json={args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
