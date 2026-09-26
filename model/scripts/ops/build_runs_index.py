#!/usr/bin/env python3
"""Generate the coordinate index for the 95-run T5 matrix.

Reads out/progress.jsonl (the ledger) and checks, per training row, whether the
run's metrics.json exists under model/results/pinn/. Emits a pipe-separated index
that the thesis work-order cites instead of quoting aggregate tables.
"""
import json
import os
import sys

ROOT = "/mnt/workspace/pinn-repro-2026"
LEDGER = os.path.join(ROOT, "out", "progress.jsonl")
OUT = os.path.join(ROOT, "out", "runs_index.psv")

HEADER = ["run", "cell", "train_seed", "obs_seed", "rc", "rel_l2_u",
          "rel_l2_speed", "rel_l2_p", "wall_ms", "metrics_present", "metrics_root"]


def fmt(v):
    if isinstance(v, float):
        return "%.6g" % v
    return "NA"


def main():
    if not os.path.isfile(LEDGER):
        sys.stderr.write("[ABORT] ledger missing: %s\n" % LEDGER)
        return 3
    rows = []
    with open(LEDGER, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("phase") != "train":
                continue
            run = str(d.get("run"))
            # Two artifact roots: dual-model / joint-PINN write results/pinn, the pure-data
            # MLP baseline (arm B) writes results/supervised. Probing only the first would
            # make a healthy arm-B run read as "artifact missing".
            root = ""
            for cand in ("pinn", "supervised"):
                if os.path.isfile(os.path.join(ROOT, "model", "results", cand, run, "metrics.json")):
                    root = cand
                    break
            rows.append([
                run, str(d.get("cell")), str(d.get("train_seed")), str(d.get("obs_seed")),
                str(d.get("rc")), fmt(d.get("rel_l2_u")), fmt(d.get("rel_l2_speed")),
                fmt(d.get("rel_l2_p")), str(d.get("wall_ms")),
                "YES" if root else "NO", root or "none",
            ])

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("|".join(HEADER) + "\n")
        for r in rows:
            fh.write("|".join(r) + "\n")

    n_ok = sum(1 for r in rows if r[4] == "0")
    n_metrics = sum(1 for r in rows if r[9] == "YES")
    roots = {}
    for r in rows:
        roots[r[10]] = roots.get(r[10], 0) + 1
    print("rows=%d rc0=%d metrics_present=%d roots=%s out=%s"
          % (len(rows), n_ok, n_metrics,
             ",".join("%s:%d" % kv for kv in sorted(roots.items())), OUT))
    if len(rows) == 0:
        print("INVALID: ledger parsed but zero train rows -> nothing to index")
        return 4
    if n_metrics == 0:
        print("INVALID: no run directory has metrics.json -> coordinate backflow incomplete")
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
