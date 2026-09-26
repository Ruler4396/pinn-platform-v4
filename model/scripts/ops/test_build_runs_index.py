#!/usr/bin/env python3
"""Regression fixture for ops/build_runs_index.py (stdlib only, no instance needed).

Why this exists: the coordinate index is what every thesis reading cites. If the
artifact-existence probe degrades, a healthy run reads as missing (or, worse, a
missing run reads as present), so the gate must be able to go red on purpose.
Run:  python3 model/scripts/ops/test_build_runs_index.py
"""
import importlib.util
import json
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location(
    "build_runs_index", os.path.join(HERE, "build_runs_index.py"))
bri = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bri)


def build(train_rows, out_name="runs_index.psv"):
    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, "ws")
    out = os.path.join(root, "out")
    os.makedirs(out)
    for run, where in train_rows:
        if where:
            d = os.path.join(root, "model", "results", where, run)
            os.makedirs(d)
            with open(os.path.join(d, "metrics.json"), "w") as fh:
                fh.write("{}")
    with open(os.path.join(out, "progress.jsonl"), "w") as fh:
        for run, _where in train_rows:
            fh.write(json.dumps({
                "phase": "train", "run": run, "cell": "t5c22", "train_seed": 42,
                "obs_seed": 0, "rc": 0, "rel_l2_u": 0.02, "rel_l2_speed": 0.021,
                "rel_l2_p": 0.1, "wall_ms": 1234}) + "\n")
        fh.write(json.dumps({"phase": "eval", "run": "must_be_ignored"}) + "\n")
    bri.ROOT = root
    bri.LEDGER = os.path.join(out, "progress.jsonl")
    bri.OUT = os.path.join(out, out_name)
    rc = bri.main()
    with open(bri.OUT, encoding="utf-8") as fh:
        lines = fh.read().strip().split("\n")
    return rc, lines


def main():
    rc, lines = build([("r_pinn", "pinn"), ("r_mlp", "supervised"), ("r_missing", "")])
    assert rc == 0, "case1 must pass, rc=%s" % rc
    assert lines[0].split("|")[-1] == "metrics_root", "header column"
    assert lines[1].split("|")[9:11] == ["YES", "pinn"], lines[1]
    assert lines[2].split("|")[9:11] == ["YES", "supervised"], lines[2]
    assert lines[3].split("|")[9:11] == ["NO", "none"], lines[3]
    assert all(len(l.split("|")) == 11 for l in lines), "column count drift"
    assert len(lines) == 4, "eval-phase row must not be indexed"
    print("case1 two-root probe + eval-row filter: OK")

    rc, _ = build([("r_gone", ""), ("r_gone2", "")])
    assert rc == 5, "all-artifacts-absent must return 5, got %s" % rc
    print("case2 positive control (no artifact anywhere) -> rc=5 INVALID: OK")

    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, "ws")
    out = os.path.join(root, "out")
    os.makedirs(out)
    with open(os.path.join(out, "progress.jsonl"), "w") as fh:
        fh.write(json.dumps({"phase": "eval", "run": "x"}) + "\n")
    bri.ROOT = root
    bri.LEDGER = os.path.join(out, "progress.jsonl")
    bri.OUT = os.path.join(out, "i.psv")
    rc = bri.main()
    assert rc == 4, "zero train rows must return 4, got %s" % rc
    print("case3 positive control (ledger with zero train rows) -> rc=4 INVALID: OK")
    print("ALL_FIXTURES_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
