#!/usr/bin/env python3
"""Two-run manifest comparison -- the acceptance half of #39 定档(丙), written so it can be
falsified BEFORE a machine-time trip.

The ruling (paper-route2/裁决-R2-10…md §四L) asks for: 同一 case 跑两遍, `files` 规范化摘要**逐字符相同**
(env 块允许不同), and to report **which lines the manifest gained or lost, by file name** -- not just a
total.  Both halves live here so nobody improvises them under segment-time pressure, and so the
"只报总数" degradation is impossible: the verdict is one string equality plus three printed lists.

Why the two passes must share ONE out-root: `_manifest` keys are absolute paths, so two different roots
give two different canonical strings for identical bytes.  Comparing across roots would report a
reproduction failure that is only a directory name.  This script therefore also refuses, out loud, when
the key sets differ only by prefix -- that shape means the runs were done in different trees, which is a
procedure error, not a physics result.

Not judged here: the manifest's own bytes.  This compares the digests the generator produced; the digest
definition is in generate_t_case._manifest and is declared once there.

Usage:
    python3 compare_manifests.py A/sha256sums.json B/sha256sums.json
    python3 compare_manifests.py --selfcheck
Exit 0 only if the digests match character for character.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("files", "files_digest", "n_files"):
        if key not in data:
            raise SystemExit(f"{path}: no '{key}' -- this is not a post-#39 manifest, so the "
                             f"comparison cannot be made; refuse instead of silently passing")
    return data


def compare(a: dict, b: dict, name_a: str, name_b: str) -> int:
    ka, kb = set(a["files"]), set(b["files"])
    added, removed = sorted(kb - ka), sorted(ka - kb)
    changed = sorted(k for k in ka & kb if a["files"][k] != b["files"][k])
    same = a["files_digest"] == b["files_digest"]

    print(f"pass A = {name_a}  n_files={a['n_files']}  digest={a['files_digest']}")
    print(f"pass B = {name_b}  n_files={b['n_files']}  digest={b['files_digest']}")
    print(f"files added in B   ({len(added)}):")
    for k in added:
        print(f"    + {k}")
    print(f"files removed in B ({len(removed)}):")
    for k in removed:
        print(f"    - {k}")
    print(f"content changed    ({len(changed)}):")
    for k in changed:
        print(f"    * {k}  {a['files'][k][:16]} -> {b['files'][k][:16]}")

    if ka != kb and {Path(k).name for k in added} == {Path(k).name for k in removed}:
        print("[PROCEDURE-ERROR] the two key sets differ ONLY by name -> these runs were done in "
              "different out-roots. The manifest keys are absolute, so this is not a reproducibility "
              "result; rerun both passes into the same --out-root.")
        return 2
    if not same:
        print("[FAIL] files_digest differs between the two runs -- 丙 is undone, or a real "
              "content difference exists; look at the lists above before touching any threshold")
        return 1
    print("[PASS] files_digest identical character for character across the two runs "
          "(the env block is allowed to differ -- it lives outside files by design)")
    return 0


def selfcheck() -> int:
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="compare_manifests_"))
    base = {"files": {"d/x.csv": "aa", "d/p.json": "bb"}, "files_digest": "DIGEST-1", "n_files": 2}
    # the case 丙 exists for: the wall clock moved, nothing else did -> must PASS
    moved_wall = dict(base)
    rc = 0
    pa, pb = root / "A.json", root / "B_same_digest.json"
    pa.write_text(json.dumps(base), encoding="utf-8")
    pb.write_text(json.dumps(moved_wall), encoding="utf-8")
    print("--- control 1: two runs, same bytes, digest equal (this is the acceptance's green shape)")
    if compare(base, moved_wall, "A", "B") != 0:
        print("[SELFCHK-FAIL] the equal case did not pass"); rc = 1
    # 必红 (a): one truth file changed by one byte
    real_change = {"files": {"d/x.csv": "aa", "d/p.json": "bc"}, "files_digest": "DIGEST-2", "n_files": 2}
    print("\n--- MUST-RED (a): a truth artefact actually changed -> must redden")
    if compare(base, real_change, "A", "changed") != 1:
        print("[SELFCHK-FAIL] the content change was not caught"); rc = 1
    # 必红 (b): the volatile block folded back into files, so the digest moved with the stopwatch
    print("\n--- MUST-RED (b): digest differs (what an unsplit 丙 looks like) -> must redden")
    if compare(base, dict(base, files_digest="DIGEST-3"), "A", "unsplit") != 1:
        print("[SELFCHK-FAIL] a moved digest went unnoticed"); rc = 1
    # 必红 (c): different out-roots must be labelled a procedure error, never a physics result
    other_root = {"files": {"other/d/x.csv": "aa", "other/d/p.json": "bb"},
                  "files_digest": "DIGEST-9", "n_files": 2}
    print("\n--- MUST-RED (c): same file NAMES under a different root -> procedure error, not a verdict")
    if compare(base, other_root, "A", "other-root") != 2:
        print("[SELFCHK-FAIL] the cross-root comparison was not flagged"); rc = 1
    # a pre-#39 manifest has no digest at all: refuse rather than report a pass
    old = root / "old_shape.json"
    old.write_text(json.dumps({"files": {}, "env": {}}), encoding="utf-8")
    print("\n--- MUST-RED (d): a manifest without files_digest must be refused, not passed")
    try:
        load(old)
        print("[SELFCHK-FAIL] an old-shape manifest was accepted"); rc = 1
    except SystemExit as exc:
        print(f"    refused as designed: {str(exc)[:70]}")
    print(f"\nselfcheck rc={rc}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="compare two S1 manifests from the same out-root")
    ap.add_argument("a", type=Path, nargs="?")
    ap.add_argument("b", type=Path, nargs="?")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        return selfcheck()
    if not args.a or not args.b:
        print("[INDETERMINATE] need two manifest paths; nothing was compared")
        return 3
    return compare(load(args.a), load(args.b), str(args.a), str(args.b))


if __name__ == "__main__":
    sys.exit(main())
