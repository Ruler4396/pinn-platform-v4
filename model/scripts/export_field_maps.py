#!/usr/bin/env python3
"""导出耦合后速度场/压力场的真值-预测-误差图。

设计约束：
- 仅做文件读取与绘图，不训练模型；
- 默认低冲击执行；
- 默认对流道外区域留白，不进行着色；
- 相对误差默认以百分比显示。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bend_cases import get_case as get_bend_case
from src.data.bend_geometry import BendGeometry
from src.data.contraction_cases import get_case as get_contraction_case
from src.data.contraction_geometry import ContractionGeometry
from src.utils.plotting import 配置中文绘图


def parse_csv_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出速度/压力真值-预测-误差图")
    parser.add_argument("--run-name", default="", help="结果目录名（位于 results/pinn/ 下）")
    parser.add_argument("--split-name", default="test", help="预测子目录后缀，例如 val/test/target")
    parser.add_argument("--cases", default="", help="可选，逗号分隔的工况名；为空时导出全部 csv")
    parser.add_argument("--predictions-subdir", default="evaluations", help="run 目录下的预测父目录")
    parser.add_argument("--predictions-dir", default="", help="直接指定 predictions 目录，优先级高于 run-name")
    parser.add_argument("--output-dirname", default="", help="输出目录名；为空时自动命名")
    parser.add_argument("--output-dir", default="", help="直接指定输出目录，优先级高于 output-dirname")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--panel-layout",
        default="row",
        choices=["row", "stack"],
        help="子图排布方式：row 为横向排布，stack 为纵向堆叠排布",
    )
    parser.add_argument(
        "--colorbar-orientation",
        default="vertical",
        choices=["vertical", "horizontal"],
        help="色标方向；对细长流道论文插图推荐 horizontal",
    )
    parser.add_argument(
        "--error-mode",
        default="rel",
        choices=["rel", "abs"],
        help="误差图模式：rel 为相对误差，abs 为绝对误差",
    )
    parser.add_argument(
        "--relative-floor-ratio",
        type=float,
        default=0.01,
        help="相对误差分母下限比例，避免真值接近 0 时误差发散",
    )
    parser.add_argument(
        "--percent-scale",
        action="store_true",
        default=True,
        help="相对误差按百分比显示（默认开启）",
    )
    parser.add_argument(
        "--mask-outside-geometry",
        action="store_true",
        default=True,
        help="按流道几何屏蔽域外三角形（默认开启）",
    )
    parser.add_argument("--max-retries", type=int, default=1, help="最大重试次数，避免无限重试")
    return parser


def resolve_case_files(predictions_dir: Path, cases: list[str]) -> list[Path]:
    if cases:
        files = [predictions_dir / f"{case_id}_predictions.csv" for case_id in cases]
    else:
        files = sorted(predictions_dir.glob("*_predictions.csv"))
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少预测文件：{missing}")
    if not files:
        raise FileNotFoundError(f"目录中未找到预测文件：{predictions_dir}")
    return files


def 计算相对误差(预测值: np.ndarray, 真值: np.ndarray, floor_value: float) -> np.ndarray:
    分母 = np.maximum(np.abs(真值), max(float(floor_value), 1.0e-12))
    return np.abs(预测值 - 真值) / 分母


def 构造几何对象(case_id: str):
    if case_id.startswith("B-"):
        return BendGeometry(get_bend_case(case_id))
    if case_id.startswith("C-"):
        return ContractionGeometry(get_contraction_case(case_id))
    raise ValueError(f"无法从工况名推断流道类型：{case_id}")


def 计算三角形掩码(triangulation, geometry) -> np.ndarray:
    triangles = triangulation.triangles
    x = triangulation.x
    y = triangulation.y
    tri_x = x[triangles]
    tri_y = y[triangles]

    centroid_x = np.mean(tri_x, axis=1)
    centroid_y = np.mean(tri_y, axis=1)
    mid01_x = 0.5 * (tri_x[:, 0] + tri_x[:, 1])
    mid01_y = 0.5 * (tri_y[:, 0] + tri_y[:, 1])
    mid12_x = 0.5 * (tri_x[:, 1] + tri_x[:, 2])
    mid12_y = 0.5 * (tri_y[:, 1] + tri_y[:, 2])
    mid20_x = 0.5 * (tri_x[:, 2] + tri_x[:, 0])
    mid20_y = 0.5 * (tri_y[:, 2] + tri_y[:, 0])

    keep = (
        geometry.contains(centroid_x, centroid_y)
        & geometry.contains(mid01_x, mid01_y)
        & geometry.contains(mid12_x, mid12_y)
        & geometry.contains(mid20_x, mid20_y)
    )
    return ~keep


def 构造三角剖分(x: np.ndarray, y: np.ndarray, case_id: str, mask_outside_geometry: bool):
    import matplotlib.tri as mtri  # type: ignore

    if len(x) < 3:
        return None
    triang = mtri.Triangulation(x, y)
    if mask_outside_geometry:
        geometry = 构造几何对象(case_id)
        triang.set_mask(计算三角形掩码(triang, geometry))
    return triang


def 估计绘图网格(shape_ratio: float) -> tuple[int, int]:
    ratio = max(float(shape_ratio), 1.0)
    short_side = 260
    long_side = int(min(max(short_side * ratio, 520), 1400))
    if ratio >= 1.0:
        return long_side, short_side
    return short_side, long_side


def 构造规则渲染网格(geometry, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if hasattr(geometry, "x_min"):
        x_min = float(getattr(geometry, "x_min"))
    if hasattr(geometry, "x_max"):
        x_max = float(getattr(geometry, "x_max"))
    if hasattr(geometry, "y_min"):
        y_min = float(getattr(geometry, "y_min"))
    if hasattr(geometry, "y_max"):
        y_max = float(getattr(geometry, "y_max"))
    span_x = max(x_max - x_min, 1.0e-6)
    span_y = max(y_max - y_min, 1.0e-6)
    nx, ny = 估计绘图网格(span_x / span_y)
    gx = np.linspace(x_min, x_max, nx)
    gy = np.linspace(y_min, y_max, ny)
    return np.meshgrid(gx, gy)


def 规则网格插值(triang, values: np.ndarray, geometry, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    import matplotlib.tri as mtri  # type: ignore
    from scipy.spatial import cKDTree  # type: ignore

    grid_x, grid_y = 构造规则渲染网格(geometry, x, y)
    if triang is not None and len(values) >= 3:
        interpolator = mtri.LinearTriInterpolator(triang, values)
        grid_values = interpolator(grid_x, grid_y)
        if np.ma.isMaskedArray(grid_values):
            grid_values = grid_values.filled(np.nan)
    else:
        grid_values = np.full_like(grid_x, np.nan, dtype=float)
    inside = geometry.contains(grid_x, grid_y)
    grid_values = np.asarray(grid_values, dtype=float)
    missing_inside = inside & ~np.isfinite(grid_values)
    if np.any(missing_inside):
        points = np.column_stack([x, y])
        tree = cKDTree(points)
        query_points = np.column_stack([grid_x[missing_inside], grid_y[missing_inside]])
        _, idx = tree.query(query_points, k=1)
        grid_values[missing_inside] = np.asarray(values, dtype=float)[idx]
    extent = (
        float(grid_x.min()),
        float(grid_x.max()),
        float(grid_y.min()),
        float(grid_y.max()),
    )
    grid_values[~inside] = np.nan
    return grid_values, extent


def 绘制几何边界(ax, geometry) -> None:
    if isinstance(geometry, ContractionGeometry):
        x_line = np.linspace(0.0, geometry.total_length, 1200)
        ax.plot(x_line, geometry.top_wall(x_line), color="#333333", linewidth=1.1)
        ax.plot(x_line, geometry.bottom_wall(x_line), color="#333333", linewidth=1.1)
        ax.plot([0.0, 0.0], [-0.5, 0.5], color="#333333", linewidth=1.1)
        ax.plot(
            [geometry.total_length, geometry.total_length],
            [-0.5 * geometry.beta, 0.5 * geometry.beta],
            color="#333333",
            linewidth=1.1,
        )
        return
    if isinstance(geometry, BendGeometry):
        x_in = np.linspace(0.0, geometry.l_in, 400)
        ax.plot(x_in, np.full_like(x_in, geometry.half_width), color="#333333", linewidth=1.1)
        ax.plot(x_in, np.full_like(x_in, -geometry.half_width), color="#333333", linewidth=1.1)

        theta = np.linspace(geometry.theta0, geometry.theta1, 800)
        top_arc = geometry.arc_point(theta, geometry.half_width)
        bottom_arc = geometry.arc_point(theta, -geometry.half_width)
        ax.plot(top_arc[:, 0], top_arc[:, 1], color="#333333", linewidth=1.1)
        ax.plot(bottom_arc[:, 0], bottom_arc[:, 1], color="#333333", linewidth=1.1)

        outlet_top_start = geometry.arc_end_center + geometry.half_width * geometry.n_out
        outlet_top_end = geometry.outlet_center + geometry.half_width * geometry.n_out
        outlet_bottom_start = geometry.arc_end_center - geometry.half_width * geometry.n_out
        outlet_bottom_end = geometry.outlet_center - geometry.half_width * geometry.n_out
        ax.plot(
            [outlet_top_start[0], outlet_top_end[0]],
            [outlet_top_start[1], outlet_top_end[1]],
            color="#333333",
            linewidth=1.1,
        )
        ax.plot(
            [outlet_bottom_start[0], outlet_bottom_end[0]],
            [outlet_bottom_start[1], outlet_bottom_end[1]],
            color="#333333",
            linewidth=1.1,
        )
        ax.plot([0.0, 0.0], [-geometry.half_width, geometry.half_width], color="#333333", linewidth=1.1)
        ax.plot(
            [outlet_top_end[0], outlet_bottom_end[0]],
            [outlet_top_end[1], outlet_bottom_end[1]],
            color="#333333",
            linewidth=1.1,
        )


def 构造几何裁剪路径(ax, geometry):
    from matplotlib.path import Path  # type: ignore
    from matplotlib.patches import PathPatch  # type: ignore

    if isinstance(geometry, ContractionGeometry):
        x_line = np.linspace(0.0, geometry.total_length, 2400)
        top = np.column_stack([x_line, geometry.top_wall(x_line)])
        bottom = np.column_stack([x_line[::-1], geometry.bottom_wall(x_line[::-1])])
        polygon = np.vstack([top, bottom, top[:1]])
    elif isinstance(geometry, BendGeometry):
        x_in = np.linspace(0.0, geometry.l_in, 1200)
        inlet_top = np.column_stack([x_in, np.full_like(x_in, geometry.half_width)])
        theta = np.linspace(geometry.theta0, geometry.theta1, 2400)
        top_arc = geometry.arc_point(theta, geometry.half_width)
        outlet_s = np.linspace(0.0, geometry.l_out, 1200)
        outlet_centerline = geometry.arc_end_center[None, :] + outlet_s[:, None] * geometry.t_out[None, :]
        outlet_top = outlet_centerline + geometry.half_width * geometry.n_out[None, :]
        outlet_bottom = outlet_centerline - geometry.half_width * geometry.n_out[None, :]
        bottom_arc = geometry.arc_point(theta[::-1], -geometry.half_width)
        inlet_bottom = np.column_stack([x_in[::-1], np.full_like(x_in, -geometry.half_width)])
        polygon = np.vstack([inlet_top, top_arc, outlet_top, outlet_bottom[::-1], bottom_arc, inlet_bottom, inlet_top[:1]])
    else:  # pragma: no cover
        raise TypeError(f"暂不支持的几何类型：{type(geometry)}")

    codes = [Path.MOVETO] + [Path.LINETO] * (len(polygon) - 2) + [Path.CLOSEPOLY]
    patch = PathPatch(Path(polygon, codes), facecolor="none", edgecolor="none", transform=ax.transData)
    ax.add_patch(patch)
    return patch


def export_case_maps(
    case_path: Path,
    output_dir: Path,
    dpi: int,
    panel_layout: str,
    colorbar_orientation: str,
    error_mode: str,
    relative_floor_ratio: float,
    percent_scale: bool,
    mask_outside_geometry: bool,
) -> list[Path]:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.ticker import MaxNLocator  # type: ignore

    配置中文绘图(prefer_serif=False)

    frame = pd.read_csv(case_path)
    case_id = str(frame["case_id"].iloc[0])
    x = frame["x_star"].to_numpy(dtype=float)
    y = frame["y_star"].to_numpy(dtype=float)
    geometry = 构造几何对象(case_id)
    triang = 构造三角剖分(x, y, case_id, mask_outside_geometry=mask_outside_geometry)

    speed_true = np.sqrt(frame["u_true"].to_numpy(dtype=float) ** 2 + frame["v_true"].to_numpy(dtype=float) ** 2)
    speed_pred = np.sqrt(frame["u_pred"].to_numpy(dtype=float) ** 2 + frame["v_pred"].to_numpy(dtype=float) ** 2)
    speed_abs_err = np.abs(speed_pred - speed_true)

    p_true = frame["p_true"].to_numpy(dtype=float)
    p_pred = frame["p_pred"].to_numpy(dtype=float)
    p_abs_err = np.abs(p_pred - p_true)

    speed_scale = max(float(np.nanmax(speed_true)), 1.0e-12)
    p_range = max(float(np.nanmax(p_true) - np.nanmin(p_true)), 1.0e-12)
    speed_rel_err = 计算相对误差(speed_pred, speed_true, relative_floor_ratio * speed_scale)
    p_rel_err = 计算相对误差(p_pred, p_true, relative_floor_ratio * p_range)

    if error_mode == "rel":
        speed_err = speed_rel_err * (100.0 if percent_scale else 1.0)
        p_err = p_rel_err * (100.0 if percent_scale else 1.0)
        error_name = "相对误差（%）" if percent_scale else "相对误差"
        error_suffix = "rel_error_pct" if percent_scale else "rel_error"
    else:
        speed_err = speed_abs_err
        p_err = p_abs_err
        error_name = "绝对误差"
        error_suffix = "abs_error"

    speed_vmin = float(min(speed_true.min(), speed_pred.min()))
    speed_vmax = float(max(speed_true.max(), speed_pred.max()))
    p_vmin = float(min(p_true.min(), p_pred.min()))
    p_vmax = float(max(p_true.max(), p_pred.max()))
    speed_err_vmax = max(float(np.nanpercentile(speed_err, 99.0)), 1.0e-12)
    p_err_vmax = max(float(np.nanpercentile(p_err, 99.0)), 1.0e-12)

    saved: list[Path] = []

    def 应用坐标轴样式(ax) -> None:
        ax.set_xlabel("x*")
        ax.set_ylabel("y*")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    def 添加色标(fig, ax, sc, label: str):
        if colorbar_orientation == "horizontal":
            cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.08, pad=0.10)
        else:
            cbar = fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.046, pad=0.03)
        cbar.set_label(label)
        cbar.ax.tick_params(labelsize=8)
        return cbar

    def draw_smooth_field(ax, values, cmap: str, vmin: float | None, vmax: float | None, colorbar_label: str):
        import matplotlib.pyplot as plt  # type: ignore

        ax.set_facecolor("white")
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(alpha=0.0)
        grid_values, extent = 规则网格插值(triang, values, geometry, x, y)
        artist = ax.imshow(
            np.ma.masked_invalid(grid_values),
            extent=extent,
            origin="lower",
            interpolation="bicubic",
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        artist.set_clip_path(构造几何裁剪路径(ax, geometry))
        绘制几何边界(ax, geometry)
        return artist, colorbar_label

    def draw_triplet(
        values_true,
        values_pred,
        values_err,
        title_prefix: str,
        cmap_main: str,
        cmap_err: str,
        filename: str,
        vmin: float,
        vmax: float,
        err_vmax: float,
    ) -> Path:
        if panel_layout == "stack":
            fig, axes = plt.subplots(3, 1, figsize=(11.8, 9.2))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        axes_list = np.atleast_1d(axes).reshape(-1)
        panels = [
            (values_true, f"{case_id} {title_prefix}真值", cmap_main, vmin, vmax, title_prefix),
            (values_pred, f"{case_id} {title_prefix}预测", cmap_main, vmin, vmax, title_prefix),
            (values_err, f"{case_id} {title_prefix}{error_name}", cmap_err, 0.0, err_vmax, error_name),
        ]
        for ax, (vals, title, cmap, cmin, cmax, cbar_label) in zip(axes_list, panels):
            sc, label = draw_smooth_field(ax, vals, cmap, cmin, cmax, cbar_label)
            ax.set_title(title)
            应用坐标轴样式(ax)
            添加色标(fig, ax, sc, label)
        fig.tight_layout()
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    saved.append(
        draw_triplet(
            speed_true,
            speed_pred,
            speed_err,
            title_prefix="速度模值",
            cmap_main="turbo",
            cmap_err="magma",
            filename=f"{case_id}_speed_truth_pred_{error_suffix}.png",
            vmin=speed_vmin,
            vmax=speed_vmax,
            err_vmax=speed_err_vmax,
        )
    )
    saved.append(
        draw_triplet(
            p_true,
            p_pred,
            p_err,
            title_prefix="压力",
            cmap_main="viridis",
            cmap_err="plasma",
            filename=f"{case_id}_pressure_truth_pred_{error_suffix}.png",
            vmin=p_vmin,
            vmax=p_vmax,
            err_vmax=p_err_vmax,
        )
    )

    if panel_layout == "stack":
        combined_fig, axes = plt.subplots(6, 1, figsize=(11.8, 17.2))
    else:
        combined_fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    combined_panels = [
        (speed_true, f"{case_id} 速度模值真值", "turbo", speed_vmin, speed_vmax, "速度模值"),
        (speed_pred, f"{case_id} 速度模值预测", "turbo", speed_vmin, speed_vmax, "速度模值"),
        (speed_err, f"{case_id} 速度模值{error_name}", "magma", 0.0, speed_err_vmax, error_name),
        (p_true, f"{case_id} 压力真值", "viridis", p_vmin, p_vmax, "压力"),
        (p_pred, f"{case_id} 压力预测", "viridis", p_vmin, p_vmax, "压力"),
        (p_err, f"{case_id} 压力{error_name}", "plasma", 0.0, p_err_vmax, error_name),
    ]
    for ax, (vals, title, cmap, cmin, cmax, cbar_label) in zip(np.atleast_1d(axes).reshape(-1), combined_panels):
        sc, label = draw_smooth_field(ax, vals, cmap, cmin, cmax, cbar_label)
        ax.set_title(title)
        应用坐标轴样式(ax)
        添加色标(combined_fig, ax, sc, label)
    combined_fig.tight_layout()
    combined_path = output_dir / f"{case_id}_speed_pressure_truth_pred_{error_suffix}.png"
    combined_fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
    plt.close(combined_fig)
    saved.append(combined_path)

    return saved


def resolve_predictions_dir(args: argparse.Namespace) -> Path:
    if args.predictions_dir:
        return Path(args.predictions_dir).expanduser().resolve()
    if not args.run_name:
        raise ValueError("未提供 run-name，也未提供 predictions-dir")
    run_dir = PROJECT_ROOT / "results" / "pinn" / args.run_name
    eval_dir = run_dir / args.predictions_subdir
    return eval_dir / f"predictions_{args.split_name}"


def resolve_output_dir(args: argparse.Namespace, predictions_dir: Path) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    if args.predictions_dir:
        output_dirname = args.output_dirname or "field_maps"
        return predictions_dir.parent / output_dirname
    run_dir = PROJECT_ROOT / "results" / "pinn" / args.run_name
    eval_dir = run_dir / args.predictions_subdir
    output_dirname = args.output_dirname or f"field_maps_{args.split_name}"
    return eval_dir / output_dirname


def export_once(args: argparse.Namespace) -> None:
    predictions_dir = resolve_predictions_dir(args)
    if not predictions_dir.exists():
        raise FileNotFoundError(f"缺少 predictions 目录：{predictions_dir}")

    cases = parse_csv_list(args.cases)
    case_files = resolve_case_files(predictions_dir, cases)
    output_dir = resolve_output_dir(args, predictions_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for case_path in case_files:
        saved_paths.extend(
            export_case_maps(
                case_path=case_path,
                output_dir=output_dir,
                dpi=args.dpi,
                panel_layout=args.panel_layout,
                colorbar_orientation=args.colorbar_orientation,
                error_mode=args.error_mode,
                relative_floor_ratio=args.relative_floor_ratio,
                percent_scale=args.percent_scale,
                mask_outside_geometry=args.mask_outside_geometry,
            )
        )

    manifest = {
        "run_name": args.run_name,
        "split_name": args.split_name,
        "predictions_dir": str(predictions_dir),
        "output_dir": str(output_dir),
        "cases": [path.stem.replace("_predictions", "") for path in case_files],
        "files": [str(path) for path in saved_paths],
        "panel_layout": args.panel_layout,
        "colorbar_orientation": args.colorbar_orientation,
        "error_mode": args.error_mode,
        "relative_floor_ratio": args.relative_floor_ratio,
        "percent_scale": args.percent_scale,
        "mask_outside_geometry": args.mask_outside_geometry,
        "max_retries": args.max_retries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[完成] output_dir={output_dir}")
    for path in saved_paths:
        print(path)


def main() -> None:
    args = build_parser().parse_args()
    attempts = max(1, int(args.max_retries))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            export_once(args)
            return
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"[第 {attempt}/{attempts} 次尝试失败] {exc}")
            if attempt >= attempts:
                raise
    if last_error is not None:
        raise last_error


if __name__ == "__main__":
    main()
