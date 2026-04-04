#!/usr/bin/env python3
"""Generate bend_2d geometry/data artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bend_cases import SUPPORTED_INLET_PROFILES, build_variant_case_id, get_case, list_cases
from src.data.bend_freefem import run_freefem
from src.data.bend_geometry import BendGeometry, GridSpec
from src.data.sparse_sampling import (
    SparseSamplingSpec,
    add_gaussian_noise,
    sample_region_aware,
    sample_uniform,
    to_observation_frame,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate bend_2d dataset artifacts")
    parser.add_argument("--case", default="B-base", help="case id, e.g. B-base")
    parser.add_argument(
        "--inlet-profile",
        default="parabolic",
        choices=SUPPORTED_INLET_PROFILES,
        help="inlet profile family; non-parabolic profiles auto-generate a suffixed case id unless --output-case-id is provided",
    )
    parser.add_argument("--output-case-id", default="", help="optional explicit output case id")
    parser.add_argument("--list-cases", action="store_true", help="list known bend cases")
    parser.add_argument("--nx", type=int, default=241, help="structured x resolution for synthetic smoke")
    parser.add_argument("--ny", type=int, default=241, help="structured y resolution for synthetic smoke")
    parser.add_argument("--boundary-samples", type=int, default=401, help="samples per boundary branch")
    parser.add_argument("--seed", type=int, default=42, help="seed for sparse sampling")
    parser.add_argument(
        "--field-source",
        default="freefem_stokes_cfd",
        choices=["synthetic_streamfunction_smoke", "freefem_stokes_cfd"],
        help="dense field source",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="hard cap for external solver retries; kept to avoid runaway consumption",
    )
    return parser



def finalize_freefem_dense_field(raw_df: pd.DataFrame, geometry: BendGeometry, case_id: str) -> pd.DataFrame:
    bc_map = {0: "interior", 1: "inlet", 2: "outlet", 3: "wall"}
    df = raw_df.copy()
    df["sample_id"] = range(len(df))
    df["case_id"] = case_id
    df["family"] = "bend"
    df["boundary_type"] = df["bc_tag"].map(bc_map).fillna("interior")
    df["is_boundary"] = (df["bc_tag"] > 0).astype(int)
    df["wall_distance_star"] = geometry.wall_distance(df["x_star"].to_numpy(), df["y_star"].to_numpy())
    df["region_id"] = geometry.region_id(df["x_star"].to_numpy(), df["y_star"].to_numpy())
    df = df[
        [
            "sample_id",
            "case_id",
            "family",
            "x_star",
            "y_star",
            "u_star",
            "v_star",
            "p_star",
            "region_id",
            "wall_distance_star",
            "is_boundary",
            "boundary_type",
        ]
    ]
    return df.assign(speed_star=lambda frame: (frame["u_star"] ** 2 + frame["v_star"] ** 2) ** 0.5)[
        [
            "sample_id",
            "case_id",
            "family",
            "x_star",
            "y_star",
            "u_star",
            "v_star",
            "p_star",
            "speed_star",
            "region_id",
            "wall_distance_star",
            "is_boundary",
            "boundary_type",
        ]
    ]



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_cases:
        for case in list_cases():
            print(
                f"{case.case_id}: Rc/W={case.rc_over_w} theta={case.theta_deg} "
                f"inlet_profile={case.inlet_profile_name} note={case.note}"
            )
        return

    root = PROJECT_ROOT
    resolved_case_id = args.output_case_id.strip() or build_variant_case_id(args.case, args.inlet_profile)
    case = get_case(resolved_case_id)
    geometry = BendGeometry(case)
    grid = GridSpec(nx=args.nx, ny=args.ny, boundary_samples=args.boundary_samples)
    sampling = SparseSamplingSpec(seed=args.seed)

    case_data_dir = root / "cases" / "bend_2d" / "data" / case.case_id
    case_fig_dir = root / "cases" / "bend_2d" / "figures" / case.case_id
    case_cfd_dir = root / "cases" / "bend_2d" / "cfd" / case.case_id
    for directory in [case_data_dir, case_fig_dir, case_cfd_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if args.field_source == "synthetic_streamfunction_smoke":
        dense_base = geometry.interior_grid(grid)
        dense_field = geometry.synthetic_reference_field(dense_base)
    else:
        _, raw_csv = run_freefem(case, case_cfd_dir, max_retries=args.max_retries)
        raw_df = pd.read_csv(raw_csv)
        dense_field = finalize_freefem_dense_field(raw_df, geometry=geometry, case_id=case.case_id)
    dense_field.to_csv(case_data_dir / "field_dense.csv", index=False)

    boundary_points = geometry.boundary_points(grid)
    boundary_points.insert(1, "case_id", case.case_id)
    boundary_points.to_csv(case_data_dir / "boundary_points.csv", index=False)

    for rate in sampling.sample_rates:
        pct = int(round(rate * 100))
        region_obs = to_observation_frame(
            sample_region_aware(dense_field, rate=rate, seed=args.seed + pct, family=case.family),
            sampling_tag=f"region_aware_{pct}pct",
        )
        region_obs.to_csv(case_data_dir / f"obs_sparse_{pct}pct.csv", index=False)

        uniform_obs = to_observation_frame(
            sample_uniform(dense_field, rate=rate, seed=args.seed + 100 + pct),
            sampling_tag=f"uniform_{pct}pct",
        )
        uniform_obs.to_csv(case_data_dir / f"obs_uniform_{pct}pct.csv", index=False)

    base_region_obs = to_observation_frame(
        sample_region_aware(dense_field, rate=0.05, seed=args.seed + 5, family=case.family),
        sampling_tag="region_aware_5pct",
    )
    for noise_rate in sampling.default_noise_rates:
        noise_pct = int(round(noise_rate * 100))
        noisy = add_gaussian_noise(base_region_obs, noise_rate=noise_rate, seed=args.seed + 300 + noise_pct)
        noisy["noise_tag"] = f"noise_{noise_pct}pct"
        noisy.to_csv(case_data_dir / f"obs_sparse_5pct_noise_{noise_pct}pct.csv", index=False)

    geometry.write_geometry_manifest(case_data_dir / "geometry.json", field_source=args.field_source)

    meta = case.to_metadata()
    meta["base_case_id"] = args.case
    meta["field_source"] = args.field_source
    if args.field_source == "synthetic_streamfunction_smoke":
        meta["warning"] = (
            "Synthetic smoke field only. Replace with CFD truth before formal training, evaluation, or thesis figures."
        )
    else:
        meta["warning"] = ""
        meta["cfd"] = {
            "solver": "FreeFEM++",
            "equation": "dimensionless steady Stokes",
            "raw_export": f"{case.case_id}_raw.csv",
            "script": f"{case.case_id}_stokes.edp",
        }
    meta["grid"] = {"nx": args.nx, "ny": args.ny, "boundary_samples": args.boundary_samples}
    meta["sampling"] = {
        "sample_rates": list(sampling.sample_rates),
        "default_noise_rates": list(sampling.default_noise_rates),
        "seed": args.seed,
    }
    meta["runtime_guard"] = {"max_retries": args.max_retries}
    (case_data_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    cfd_readme = case_cfd_dir / "README.md"
    if args.field_source == "freefem_stokes_cfd":
        cfd_readme.write_text(
            f"# {case.case_id} CFD artifacts\n\n"
            f"本目录已生成 `{case.case_id}` 的 FreeFEM++ CFD 数据：\n\n"
            f"- `{case.case_id}_stokes.edp`：自动生成的稳态 Stokes 求解脚本\n"
            f"- `{case.case_id}_raw.csv`：FreeFEM++ 原始导出结果\n\n"
            "说明：\n\n"
            "- 当前解为 dimensionless steady Stokes baseline\n"
            f"- 入口剖面：`{case.inlet_profile_name}`\n"
            "- 后处理结果已同步写入 data 目录的 `field_dense.csv`\n",
            encoding="utf-8",
        )
    elif not cfd_readme.exists():
        cfd_readme.write_text(
            "# CFD placeholder\n\n"
            "本目录预留给正式 FreeFEM++/CFD 真值导出结果。\n"
            "当前 data 目录中的 dense field 仅为 synthetic smoke 数据，用于验证数据链路。\n",
            encoding="utf-8",
        )

    print(f"[done] generated bend dataset artifacts for {case.case_id}")
    print(f"data_dir={case_data_dir}")
    print(f"dense_points={len(dense_field)} boundary_points={len(boundary_points)}")
    print(f"field_source={args.field_source}")
    print(f"inlet_profile={case.inlet_profile_name}")
    if args.field_source == "synthetic_streamfunction_smoke":
        print("warning=synthetic smoke field only; replace with CFD truth for formal experiments")


if __name__ == "__main__":
    main()
