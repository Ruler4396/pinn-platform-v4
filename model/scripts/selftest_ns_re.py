#!/usr/bin/env python3
"""T6-Re: run every self-gate of the three delivery scripts, and drive the merge path
end to end against a stand-in solver.  stdlib only, no FreeFEM, no instance.

Why this file exists: the other three scripts each check themselves, but until something
writes a `_raw.csv` and a matching `_raw_pair.csv` and pushes them through
`finalize_ns_truth.finalize()`, the code path that runs *after* a successful solve has never
executed anywhere.  That gap has already cost this project one round trip ("device all
green, real artefact shatters on first contact"), and closing it caught two more: `finalize`
compared the two header *widths* inside its row-count guard, so every well-formed level was
refused with a message whose two numbers were equal; and the hi-lattice precondition it
claimed to check could never have fired, because rounding a multiple of 1e-N to six
significant digits yields another multiple of 1e-N.  What is checked instead is the
magnitude bound, which is measurable on the artefact.

The stand-in is driven by the artefact, not by a description of it: it reads the emitted
.edp's own `ofstream` statements and its own header lines, and writes exactly those files
with those columns, quantising `hi` at the widths parsed out of the same text.  A header typo
or a width change in the .edp therefore propagates into the fixture, and the merge either
follows it or says why it cannot.

What it runs:
  1. in-process: the Re=1 text inverts to the shipped Stokes text; one convective line; the
     v4.9 lint (no `_`, no trailing comments) on the emitted text and on the probe
  2. subprocess: gen_ns_re_edp.py --check-syntax, finalize_ns_truth.py --selfcheck,
     check_ns_re_to_stokes.py --selfcheck
  3. finalize --dry-run on a well-formed stand-in level                       -> want rc 0
  4. control A: the same level with |v_star| 100x larger                      -> want rc 1
     (the widths come from the Stokes magnitudes and a finite-Re solve is allowed to
     outgrow them; hi then loses digits inside FreeFEM and the refusal must name the width
     that would work)
  5. control B: `hi` rewritten onto a coarser lattice, `lo` untouched          -> want rc 1
     (a convention break; the per-field drift band is what sees it)

Usage:
    python3 selftest_ns_re.py
    python3 selftest_ns_re.py --scratch /tmp/ns_re_selftest
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import finalize_ns_truth as fz                                    # noqa: E402
import gen_ns_re_edp as gen                                       # noqa: E402

LEVEL = "1"


def run_script(script: str, *args: str) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, str(HERE / script), *args],
                          capture_output=True, text=True, cwd=str(HERE))
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def stand_in_solver(edp: Path, out_dir: Path, stokes_csv: Path) -> dict[str, Path]:
    """Write the files the .edp says it writes, at the widths the .edp says it uses.

    Nothing about the output is hard-coded here: the file names and the column lists come
    from the .edp's own `ofstream` and first `<< "..." << endl;` statements, and the hi
    widths come from its `floor()` lines.
    """
    text = edp.read_text(encoding="utf-8").replace("\r\n", "\n")
    rule = fz.hi_decimals_from_edp(text)
    streams = re.findall(r'ofstream\s+(\w+)\("([^"]+)"\)', text)
    if len(streams) != 2:
        raise SystemExit(f"expected two ofstream streams in {edp.name}, found {len(streams)}")
    headers: dict[str, list[str]] = {}
    for var, name in streams:
        m = re.search(re.escape(var) + r'\s*<<\s*"([^"]+)"', text)
        if not m:
            raise SystemExit(f"{name}: no header line is written to {var} -- the .edp is not "
                             f"the shape this stand-in knows how to imitate")
        headers[name] = m.group(1).split(",")
    with stokes_csv.open(encoding="utf-8", newline="") as fh:
        head = fh.readline().strip().split(",")
        rows = [dict(zip(head, ln.strip().split(","))) for ln in fh if ln.strip()]
    written: dict[str, Path] = {}
    for name, cols in headers.items():
        out = [",".join(cols)]
        for row in rows:
            cells = []
            for col in cols:
                if col == "bc_tag":
                    cells.append(str(row["bc_tag"]))
                    continue
                if col in ("x_star", "y_star"):
                    cells.append(repr(fz.printed(float(row[col]))))
                    continue
                field = next(f for f in fz.FIELDS if fz.LETTER[f] == col.split("_")[0])
                # the node's true value, including the digits the shipped file lost
                exact = fz.within_rounding_cell([float(row[field])], draws=1)[0]
                if col.endswith("_hi") or col.endswith("_lo"):
                    hi, lo = fz.split(exact, rule[field])
                    cells.append(repr(fz.printed(hi if col.endswith("_hi") else lo)))
                else:
                    cells.append(repr(fz.printed(exact)))
            out.append(",".join(cells))
        path = out_dir / name
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        written["pair" if name.endswith("_pair.csv") else "raw"] = path
    return written


def rescale_columns(path: Path, cols: list[str], factor: float) -> float:
    lines = path.read_text(encoding="utf-8").split("\n")
    head = lines[0].split(",")
    idx = [head.index(c) for c in cols]
    biggest = 0.0
    out = [lines[0]]
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split(",")
        for i in idx:
            v = float(parts[i]) * factor
            biggest = max(biggest, abs(v))
            parts[i] = repr(fz.printed(v))
        out.append(",".join(parts))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return biggest


def bump_magnitude(raw: Path, pair: Path, field: str, factor: float) -> float:
    """Pretend this level's solution came out `factor` times larger, in both streams.

    That is the failure a derived width actually risks on the instance: the widths come from
    the Stokes magnitudes, and a finite-Re solve need not respect them.  hi then loses digits
    inside FreeFEM and no merge recovers them, so `finalize` must refuse and name a width
    that would work.
    """
    rescale_columns(raw, [field], factor)
    return rescale_columns(pair, [fz.LETTER[field] + "_hi", fz.LETTER[field] + "_lo"], factor)


def coarsen_hi(pair: Path, field: str, decimals: int) -> int:
    """Move one field's `hi` onto a coarser lattice, leaving its `lo` alone.

    A convention break of the kind the merge must not be talked past: hi + lo is then no
    longer the value the raw stream printed, so the per-field drift has to leave its band.
    """
    key = fz.LETTER[field] + "_hi"
    m = 10.0 ** decimals
    lines = pair.read_text(encoding="utf-8").split("\n")
    j = lines[0].split(",").index(key)
    moved = 0
    out = [lines[0]]
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = ln.split(",")
        was = float(parts[j])
        coarsened = math.floor(was * m) / m
        moved += 1 if coarsened != was else 0
        parts[j] = repr(fz.printed(coarsened))
        out.append(",".join(parts))
    pair.write_text("\n".join(out) + "\n", encoding="utf-8")
    return moved


def capture(fn, *a, **kw) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a, **kw)
    return rc, buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description="T6-Re self-gates plus an end-to-end merge test")
    ap.add_argument("--scratch",
                    default=str(Path(tempfile.gettempdir()) / "pinn_ns_re_selftest"))
    args = ap.parse_args()
    scratch = Path(args.scratch).resolve()
    if REPO in scratch.parents or scratch == REPO:
        print(f"[FAIL] refusing to write selftest output inside the repository: {scratch}")
        return 1
    stokes = gen.CASE_DIR / gen.STOKES
    if not stokes.is_file():
        print(f"[INDETERMINATE] {stokes} is not in this checkout; nothing to derive from")
        return 2
    shipped = stokes.read_text(encoding="utf-8").replace("\r\n", "\n")
    widths = gen.derived_widths()
    spec = gen.build_spec(widths)
    ns = gen.transform(shipped, LEVEL, spec)

    fails = 0
    n_pass = n_fail = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails, n_pass, n_fail
        n_pass += 1 if ok else 0
        n_fail += 0 if ok else 1
        fails |= 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n        {detail}" if detail else ""))

    def last(text: str, n: int = 220) -> str:
        keep = [x.strip() for x in text.strip().splitlines() if x.strip()]
        return keep[-1][:n] if keep else ""

    check(f"diff proof (widths {widths}): the Re={LEVEL} text inverts to the shipped Stokes "
          f"text byte for byte", gen.invert(ns, LEVEL, spec).strip() == shipped.strip())
    conv = [x for x in ns.split("\n") if "u0*dx(u)" in x]
    check("diff proof: the physics difference is exactly one convective line", len(conv) == 1,
          conv[0].strip() if conv else "none")
    hits = gen.syntax_findings(ns)
    check("v4.9 lint: no emitted line uses a construct the instance rejected", not hits,
          "; ".join(f"line {n}: {k}" for n, k, _ in hits[:4]))
    probe = gen.probe_text(shipped, widths)
    base = gen.probe_base(shipped)
    base_nl = base if base.endswith("\n") else base + "\n"
    diff_lines = [i for i, (a, b) in enumerate(zip(shipped.split("\n"), base.split("\n")))
                  if a != b]
    appended = len(probe.split("\n")) - len(base_nl.split("\n"))
    check("probe = shipped + exactly the 2 output-path edits + an appended tail, and it is "
          "lint-clean",
          probe.startswith(base_nl) and len(diff_lines) == 2
          and gen.PROBE_PATH_SWAP[0] in shipped and gen.PROBE_PATH_SWAP[0] not in probe
          and not gen.syntax_findings(probe),
          f"paths at lines {diff_lines}, {appended} appended lines, "
          f"absolute path present: {gen.PROBE_PATH_SWAP[0] in probe}")
    check("control: the reason for that edit is still true (the shipped text does carry the "
          "absolute path the instance has no /root/dev for)",
          gen.SHIPPED_ABS in shipped and gen.SHIPPED_ABS not in probe,
          "2026-09-26 21:12: parsed 158 lines cleanly, then rc=8 on that path")
    for script, flag in (("gen_ns_re_edp.py", "--check-syntax"),
                         ("finalize_ns_truth.py", "--selfcheck"),
                         ("check_ns_re_to_stokes.py", "--selfcheck")):
        rc, tail = run_script(script, flag)
        check(f"{script} {flag}", rc == 0, "" if rc == 0 else tail[-500:])

    level_dir = scratch / "cfd" / f"C-base_ns_re{LEVEL}"
    base_dir = scratch / "cfd" / "C-base"
    for d in (level_dir, base_dir):
        d.mkdir(parents=True, exist_ok=True)
    edp = level_dir / f"C-base_ns_re{LEVEL}.edp"
    edp.write_text(ns, encoding="utf-8")
    shutil.copyfile(gen.CASE_DIR / "C-base_raw.csv", base_dir / "C-base_raw.csv")
    fz.CFD = scratch / "cfd"

    made = stand_in_solver(edp, level_dir, base_dir / "C-base_raw.csv")
    check("stand-in wrote both streams the .edp names",
          len(made) == 2 and all(p.is_file() and p.stat().st_size > 1000
                                 for p in made.values()),
          ", ".join(f"{p.name} ({p.stat().st_size} B)" for p in made.values()))
    rc, out = capture(fz.finalize, LEVEL, True)
    check("finalize --dry-run on a well-formed level", rc == 0, last(out))

    big = bump_magnitude(made["raw"], made["pair"], "v_star", 100.0)
    rc_a, out_a = capture(fz.finalize, LEVEL, True)
    named = [ln.strip() for ln in out_a.splitlines() if "N=" in ln]
    check("control A: a level that outgrew its derived width is refused, not merged",
          rc_a == 1 and big > 1.0, f"|v_star| pushed to {big:.3g}, rc={rc_a}")
    check("control A: the refusal names the width to re-emit at",
          any("re-emit" in ln for ln in named), named[0][:220] if named else last(out_a))

    made = stand_in_solver(edp, level_dir, base_dir / "C-base_raw.csv")
    moved = coarsen_hi(made["pair"], "v_star", 2)
    rc_b, out_b = capture(fz.finalize, LEVEL, True)
    check("control B: hi on the wrong lattice is refused as a convention break",
          rc_b == 1 and moved > 100, f"{moved} rows moved off their hi value, rc={rc_b}")
    check("control B: the drift is reported per field, not pooled",
          "v_star" in out_b and "p_star" in out_b, last(out_b))

    # control C: the failure mode that actually stopped the 2026-09-26 instance run -- the
    # four scripts re-pulled from the new pin while the .edp files in the working tree were
    # still the previous pin's, so the widths on disk were a single 1e2 for all three fields.
    stale = scratch / "cfd_stale"
    for lvl in fz.LEVELS:
        d = stale / f"C-base_ns_re{lvl}"
        d.mkdir(parents=True, exist_ok=True)
        txt = re.sub(r"floor\((h[uvp])\*1\.0e\d+\)/1\.0e\d+", r"floor(\1*1.0e2)/1.0e2", ns)
        (d / f"C-base_ns_re{lvl}.edp").write_text(txt, encoding="utf-8")
    keep = fz.CFD
    fz.CFD = stale
    rc_c, out_c = capture(fz.selfcheck, fz.LEVELS)
    fz.CFD = keep
    check("control C: .edp files that predate the per-field widths are caught",
          rc_c == 1 and "emitted N=2, derived N=4" in out_c and "emitted N=2, derived N=6" in out_c,
          last(out_c, 120))
    check("control C: the refusal says which file to re-pull, so it is not read as a bad method",
          "[HINT]" in out_c and "blob=" in out_c,
          next((ln.strip() for ln in out_c.splitlines() if "blob=" in ln), "")[:150])

    print(f"\ntotal={n_pass + n_fail} failed={n_fail} "
          f"{'ALL GREEN' if not fails else 'SEE RED'}   (scratch: {scratch})")
    return fails


if __name__ == "__main__":
    sys.exit(main())
