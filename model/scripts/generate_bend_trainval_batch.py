#!/usr/bin/env python3
"""Batch-generate bend_2d training/validation CFD datasets in serial mode."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bend_cases import SUPPORTED_INLET_PROFILES, build_variant_case_id, get_case

TRAINVAL_CASES = [
    "B-train-1",
    "B-train-2",
    "B-train-3",
    "B-val",
]



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch generate bend_2d train/val CFD datasets")
    parser.add_argument("--field-source", default="freefem_stokes_cfd", choices=["freefem_stokes_cfd", "synthetic_streamfunction_smoke"])
    parser.add_argument("--inlet-profile", default="parabolic", choices=SUPPORTED_INLET_PROFILES)
    parser.add_argument("--boundary-samples", type=int, default=101)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=1, help="hard cap to avoid runaway retries")
    parser.add_argument("--include-base", action="store_true", help="also regenerate B-base before train/val cases")
    return parser



def run_case(base_case_id: str, args: argparse.Namespace) -> dict:
    case_id = build_variant_case_id(base_case_id, args.inlet_profile)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_bend_case.py"),
        "--case",
        base_case_id,
        "--inlet-profile",
        args.inlet_profile,
        "--field-source",
        args.field_source,
        "--boundary-samples",
        str(args.boundary_samples),
        "--seed",
        str(args.seed),
        "--max-retries",
        str(args.max_retries),
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    data_dir = PROJECT_ROOT / "cases" / "bend_2d" / "data" / case_id
    cfd_dir = PROJECT_ROOT / "cases" / "bend_2d" / "cfd" / case_id
    dense_csv = data_dir / "field_dense.csv"
    meta_json = data_dir / "meta.json"
    raw_csv = cfd_dir / f"{case_id}_raw.csv"

    dense_rows = sum(1 for _ in dense_csv.open("r", encoding="utf-8")) - 1
    raw_rows = sum(1 for _ in raw_csv.open("r", encoding="utf-8")) - 1 if raw_csv.exists() else 0
    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    case = get_case(case_id)
    return {
        "case_id": case_id,
        "base_case_id": base_case_id,
        "rc_over_w": case.rc_over_w,
        "theta_deg": case.theta_deg,
        "inlet_profile_name": case.inlet_profile_name,
        "role": "validation" if base_case_id == "B-val" else "training",
        "field_source": meta.get("field_source", ""),
        "dense_rows": dense_rows,
        "raw_rows": raw_rows,
        "data_dir": str(data_dir),
        "cfd_dir": str(cfd_dir),
    }



def write_manifest(records: list[dict], args: argparse.Namespace) -> Path:
    out_dir = PROJECT_ROOT / "cases" / "bend_2d" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.inlet_profile == "parabolic" else f"__ip_{args.inlet_profile}"
    csv_path = out_dir / f"trainval_manifest{suffix}.csv"
    json_path = out_dir / f"trainval_manifest{suffix}.json"

    fields = [
        "case_id",
        "base_case_id",
        "rc_over_w",
        "theta_deg",
        "inlet_profile_name",
        "role",
        "field_source",
        "dense_rows",
        "raw_rows",
        "data_dir",
        "cfd_dir",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "serial_low_impact",
        "field_source": args.field_source,
        "inlet_profile": args.inlet_profile,
        "boundary_samples": args.boundary_samples,
        "seed": args.seed,
        "max_retries": args.max_retries,
        "cases": records,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path



def main() -> None:
    args = build_parser().parse_args()
    case_ids = list(TRAINVAL_CASES)
    if args.include_base:
        case_ids = ["B-base"] + case_ids

    records = []
    for case_id in case_ids:
        print(f"[batch] generating {case_id} ...")
        records.append(run_case(case_id, args))

    manifest = write_manifest(records, args)
    print(f"[done] wrote manifest: {manifest}")
    print(f"cases={','.join(case_ids)}")


if __name__ == "__main__":
    main()
