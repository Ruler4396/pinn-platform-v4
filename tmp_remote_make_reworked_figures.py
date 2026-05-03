from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("model").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.contraction_cases import get_case
from src.data.contraction_geometry import ContractionGeometry
from src.utils.plotting import 配置中文绘图


def main() -> None:
    配置中文绘图(prefer_serif=False)
    outdir = PROJECT_ROOT / "results" / "field_map_checks" / "chapter5_materials" / "reworked_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    base = PROJECT_ROOT / "cases" / "contraction_2d" / "data" / "C-val"
    dense = pd.read_csv(base / "field_dense.csv")
    sparse = pd.read_csv(base / "obs_sparse_5pct.csv")
    uniform = pd.read_csv(base / "obs_uniform_5pct.csv")
    region_summary = pd.read_csv(
        PROJECT_ROOT
        / "results"
        / "field_map_checks"
        / "chapter5_materials"
        / "sampling_region_error_analysis"
        / "contraction_sampling_region_error_summary.csv"
    )

    case = get_case("C-val")
    geom = ContractionGeometry(case)
    xs = np.linspace(0.0, case.total_length_over_w, 500)
    half_width = geom.half_width(xs)

    region_names = {0: "主体区", 1: "收缩段", 2: "近壁区"}
    colors = {0: "#2f6f9f", 1: "#d86c1f", 2: "#4f9d55"}

    fig = plt.figure(figsize=(12.6, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], hspace=0.34, wspace=0.28)

    layout_specs = [
        (fig.add_subplot(gs[0, 0]), sparse, "分层采样：收缩段配额更高"),
        (fig.add_subplot(gs[0, 1]), uniform, "均匀采样：全域覆盖更均衡"),
    ]
    for ax, df, title in layout_specs:
        ax.fill_between(xs, -half_width, half_width, color="#f6f7f4", zorder=0)
        ax.plot(xs, half_width, color="#444444", lw=1.15)
        ax.plot(xs, -half_width, color="#444444", lw=1.15)
        ax.scatter(dense["x_star"], dense["y_star"], s=3, color="#c9c9c9", alpha=0.12, linewidths=0, zorder=1)
        for region_id in (0, 1, 2):
            part = df[df["region_id"] == region_id]
            ax.scatter(
                part["x_star"],
                part["y_star"],
                s=33,
                color=colors[region_id],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
                label=f"{region_names[region_id]} n={len(part)}",
            )
        ax.set_title(title, fontsize=14, pad=8)
        ax.set_xlim(-0.25, case.total_length_over_w + 0.25)
        ax.set_ylim(-0.56, 0.56)
        ax.set_xlabel("x*")
        ax.set_ylabel("y*")
        ax.grid(alpha=0.22, linestyle=":")
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    ax = fig.add_subplot(gs[1, :])
    metrics = region_summary[region_summary["split"] == "val"].copy()
    regions = [0, 1, 2]
    x = np.arange(len(regions))
    width = 0.18
    bar_specs = [
        ("region-aware", "分层", "#0f766e", ""),
        ("uniform", "均匀", "#f59e0b", "//"),
    ]
    metric_specs = [("mean_rel_l2_speed", "速度Rel-L2", 0.92), ("mean_rel_l2_p", "压力Rel-L2", 0.55)]
    for sample_index, (sampling, sampling_label, color, hatch) in enumerate(bar_specs):
        for metric_index, (metric, metric_label, alpha) in enumerate(metric_specs):
            values = []
            for region_id in regions:
                row = metrics[
                    (metrics["strategy"] == sampling)
                    & (metrics["region_id"] == region_id)
                ]
                values.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            offset = (sample_index * 2 + metric_index - 1.5) * width
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=color,
                alpha=alpha,
                hatch=hatch,
                edgecolor=color,
                label=f"{sampling_label}-{metric_label}",
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.004,
                    f"{value * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([region_names[region_id] for region_id in regions])
    ax.set_ylabel("验证集分区相对误差")
    ax.set_title("同一5%预算下，分层采样降低收缩段速度误差，但主体区与近壁区收益不足", fontsize=14, pad=8)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_ylim(0, float(metrics[["mean_rel_l2_speed", "mean_rel_l2_p"]].to_numpy().max()) * 1.32)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper right")

    fig.suptitle("5%观测预算下采样点布局与分区误差对照", fontsize=18, y=0.985)
    fig.text(
        0.5,
        0.018,
        "注：区域0为主体区，区域1为收缩段，区域2为近壁区；误差按C-val验证工况统计。",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.95))
    out = outdir / "fig5_10_sampling_layout_region_error_reworked.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
