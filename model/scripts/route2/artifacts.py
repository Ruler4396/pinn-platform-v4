"""Shared artifact helpers for route 2 (pure stdlib: hash / json / csv / env lock)."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> tuple[List[str], List[List[str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [row for row in reader if row and any(c.strip() for c in row)]
    return [c.strip() for c in header], rows


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
            n += 1
    return n


def env_lock(extra: dict | None = None) -> dict:
    """Autopsy P1 requires solver + interpreter provenance to travel with the truth."""
    lock = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }
    for mod in ("numpy", "pandas", "torch", "matplotlib"):
        try:
            __import__(mod)
            lock[mod] = getattr(__import__(mod), "__version__", "present")
        except Exception:
            lock[mod] = "absent"
    if extra:
        lock.update(extra)
    return lock


def hash_tree(paths: Iterable[Path]) -> dict:
    return {str(p).replace("\\", "/"): sha256_file(p) for p in sorted(paths) if p.is_file()}
