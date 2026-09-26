#!/usr/bin/env python3
"""Rewrite the pinned blob-hash table inside an ops runner, so nobody hand-types 40-char shas.

Usage: python3 model/scripts/ops/bump_runner_pin.py <commit> <runner-file>
Writes nothing unless every listed path exists at <commit> and the runner's existing
PIN/EXPECT block parses; prints the diff of what changed.
"""
import io
import os
import re
import subprocess
import sys

EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def blob_sha256(commit, path):
    out = subprocess.run(["git", "show", "%s:%s" % (commit, path)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if out.returncode != 0:
        raise SystemExit("[FAIL] path missing at %s: %s\n%s" % (commit, path, out.stderr.decode("utf-8", "replace")))
    digest = __import__("hashlib").sha256(out.stdout).hexdigest()
    if digest == EMPTY_SHA:
        raise SystemExit("[FAIL] empty blob for %s at %s (that is a failed read, not a hash)" % (path, commit))
    return digest


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    commit, runner = sys.argv[1], sys.argv[2]
    commit = subprocess.run(["git", "rev-parse", commit], stdout=subprocess.PIPE).stdout.decode().strip()
    text = io.open(runner, encoding="utf-8").read()
    nl = "\n" if "\r\n" not in text else "\r\n"
    pm = re.search(r"^PIN=([0-9a-f]{40})$", text, re.M)
    if not pm:
        raise SystemExit("[FAIL] no PIN=<40 hex> line in %s" % runner)
    keys = re.findall(r"^  \[(model/[^\]]+)\]=([0-9a-f]{64})$", text, re.M)
    if not keys:
        raise SystemExit("[FAIL] no EXPECT entries of the form '  [model/path]=<64 hex>' in %s" % runner)
    lines = text.split(nl)
    changed = []
    for path, old in keys:
        new = blob_sha256(commit, path)
        if new != old:
            changed.append((path, old, new))
        for i, l in enumerate(lines):
            if l == "  [%s]=%s" % (path, old):
                lines[i] = "  [%s]=%s" % (path, new)
    old_pin = pm.group(1)
    for i, l in enumerate(lines):
        if l == "PIN=%s" % old_pin:
            lines[i] = "PIN=%s" % commit
    out = nl.join(lines)
    # guard: never leave a runner half-bumped
    if re.findall(r"\[(model/[^\]]+)\]=([0-9a-f]{64})", out).__len__() != len(keys):
        raise SystemExit("[FAIL] entry count changed after rewrite")
    for path, old, new in changed:
        print("CHANGED %s\n   old %s\n   new %s" % (path, old[:16] + "…", new[:16] + "…"))
    print("PIN %s… -> %s…   entries=%d changed=%d" % (old_pin[:8], commit[:8], len(keys), len(changed) + 1))
    io.open(runner, "w", encoding="utf-8", newline="").write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
