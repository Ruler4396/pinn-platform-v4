#!/usr/bin/env python3
"""Generate modality-reduced observation CSVs from existing sparse/noisy observations.

Low-impact usage:
- only rewrites derived CSV views under existing case data directories
- no CFD regeneration
- explicit max_retries guard is not needed because there is no iterative training
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAMILY_SUBDIR = {
    "contraction_2d": "contraction_2d",
    "bend_2d": "bend_2d",
}


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def normalize_source_name(source: str) -> str:
    source_name = source.strip()
    if not source_name:
        raise ValueError("Empty source name is not allowed")
    return source_name if source_name.endswith(".csv") else f"{source_name}.csv"


def normalize_components(text: str) -> tuple[str, ...]:
    mapping = {"u", "v", "p"}
    comps = tuple(part.strip() for part in text.split(",") if part.strip())
    if not comps:
        raise ValueError("At least one observed component is required")
    unknown = set(comps) - mapping
    if unknown:
        raise ValueError(f"Unknown observed components: {sorted(unknown)}")
    return comps


def default_suffix(components: tuple[str, ...]) -> str:
    if components == ("p",):
        return "pressure_only"
    if components == ("u", "v"):
        return "velocity_only"
    return "partial_" + "".join(components)


def iter_case_dirs(family: str) -> list[Path]:
    data_root = PROJECT_ROOT / "cases" / FAMILY_SUBDIR[family] / "data"
    return sorted([path for path in data_root.iterdir() if path.is_dir()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate partial-observation CSVs")
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_SUBDIR))
    parser.add_argument("--sources", required=True, help="comma-separated source names, e.g. obs_sparse_5pct,obs_sparse_5pct_noise_3pct")
    parser.add_argument("--observed-components", required=True, help="comma-separated subset of u,v,p; examples: p or u,v")
    parser.add_argument("--output-suffix", default="", help="optional suffix without .csv; default derived from observed-components")
    args = parser.parse_args()

    components = normalize_components(args.observed_components)
    suffix = args.output_suffix.strip() or default_suffix(components)
    sources = [normalize_source_name(item) for item in parse_csv_list(args.sources)]
    case_dirs = iter_case_dirs(args.family)

    for case_dir in case_dirs:
        for source_name in sources:
            source_path = case_dir / source_name
            if not source_path.exists():
                continue
            df = pd.read_csv(source_path)
            required = {"u_obs", "v_obs", "p_obs"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{source_path} missing columns: {sorted(missing)}")
            out = df.copy()
            for component in ("u", "v", "p"):
                keep = component in components
                out[f"{component}_obs_mask"] = np.full((len(out),), 1 if keep else 0, dtype=np.int64)
                if not keep:
                    out[f"{component}_obs"] = np.nan
            stem = source_name[:-4] if source_name.endswith(".csv") else source_name
            output_path = case_dir / f"{stem}_{suffix}.csv"
            out.to_csv(output_path, index=False)
            print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
