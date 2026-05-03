from pathlib import Path
import json
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


OUT = PROJECT_ROOT / "results" / "field_map_checks" / "chapter5_materials" / "reworked_figures_v2"


def setup_plot() -> None:
    配置中文绘图(prefer_serif=False)
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.unicode_minus": False,
        }
    )


def make_sampling_layout() -> Path:
    base = PROJECT_ROOT / "cases" / "contraction_2d" / "data" / "C-val"
    dense = pd.read_csv(base / "field_dense.csv")
    sparse = pd.read_csv(base / "obs_sparse_5pct.csv")
    uniform = pd.read_csv(base / "obs_uniform_5pct.csv")

    case = get_case("C-val")
    geom = ContractionGeometry(case)
    xs = np.linspace(0.0, case.total_length_over_w, 500)
    half_width = geom.half_width(xs)

    region_names = {0: "主体区", 1: "收缩段", 2: "近壁区"}
    colors = {0: "#2f6f9f", 1: "#d86c1f", 2: "#4f9d55"}

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True)
    specs = [
        (axes[0], sparse, "分层采样：收缩段与近壁区点位更多"),
        (axes[1], uniform, "均匀采样：全域覆盖更均衡"),
    ]
    for ax, df, title in specs:
        ax.fill_between(xs, -half_width, half_width, color="#f7f7f4", zorder=0)
        ax.plot(xs, half_width, color="#444444", lw=1.25)
        ax.plot(xs, -half_width, color="#444444", lw=1.25)
        ax.scatter(dense["x_star"], dense["y_star"], s=3, color="#cfcfcf", alpha=0.12, linewidths=0, zorder=1)
        for rid in (0, 1, 2):
            part = df[df["region_id"] == rid]
            ax.scatter(
                part["x_star"],
                part["y_star"],
                s=34,
                color=colors[rid],
                edgecolor="white",
                linewidth=0.45,
                label=f"{region_names[rid]} n={len(part)}",
                zorder=3,
            )
        ax.set_title(title, pad=8)
        ax.set_ylabel("y*")
        ax.set_xlim(-0.25, case.total_length_over_w + 0.25)
        ax.set_ylim(-0.56, 0.56)
        ax.grid(alpha=0.22, linestyle=":")
        ax.legend(frameon=False, loc="upper right", ncol=3)
    axes[-1].set_xlabel("x*")
    fig.suptitle("5%观测预算下的采样点空间分布", fontsize=17, y=0.99)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    out = OUT / "fig5_8_sampling_layout_split.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out


def make_sampling_region_error() -> Path:
    region_summary = pd.read_csv(
        PROJECT_ROOT
        / "results"
        / "field_map_checks"
        / "chapter5_materials"
        / "sampling_region_error_analysis"
        / "contraction_sampling_region_error_summary.csv"
    )
    metrics = region_summary[region_summary["split"] == "val"].copy()
    region_names = {0: "主体区", 1: "收缩段", 2: "近壁区"}
    regions = [0, 1, 2]
    x = np.arange(len(regions))
    width = 0.18

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    specs = [
        ("region-aware", "分层", "#0f766e", ""),
        ("uniform", "均匀", "#f59e0b", "//"),
    ]
    metric_specs = [("mean_rel_l2_speed", "速度Rel-L2", 0.92), ("mean_rel_l2_p", "压力Rel-L2", 0.55)]
    for s_idx, (sampling, sampling_label, color, hatch) in enumerate(specs):
        for m_idx, (metric, metric_label, alpha) in enumerate(metric_specs):
            vals = []
            for rid in regions:
                row = metrics[(metrics["strategy"] == sampling) & (metrics["region_id"] == rid)]
                vals.append(float(row[metric].iloc[0]))
            offset = (s_idx * 2 + m_idx - 1.5) * width
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=color,
                alpha=alpha,
                hatch=hatch,
                edgecolor=color,
                label=f"{sampling_label}-{metric_label}",
            )
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.004, f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels([region_names[rid] for rid in regions])
    ax.set_ylabel("验证集分区相对误差")
    ax.set_title("分区误差显示：分层采样只在收缩段速度误差上占优", pad=8)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_ylim(0.0, float(metrics[["mean_rel_l2_speed", "mean_rel_l2_p"]].to_numpy().max()) * 1.35)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    fig.tight_layout()
    out = OUT / "fig5_9_sampling_region_error_split.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out


def make_benchmark_time() -> Path:
    labels = ["收缩流道", "弯曲流道"]
    pinn_full = [0.111, 0.117]
    pinn_sparse = [0.242, 0.394]
    cfd = [0.481, 3.552]
    x = np.arange(len(labels))
    width = 0.23

    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    bars1 = ax.bar(x - width, pinn_full, width=width, color="#4c78a8", label="PINN完整推理")
    bars2 = ax.bar(x, pinn_sparse, width=width, color="#72b7b2", label="PINN稀疏重建")
    bars3 = ax.bar(x + width, cfd, width=width, color="#bab0ac", label="CFD求解")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("中位耗时（秒，对数坐标）")
    ax.set_title("同一环境下PINN与CFD中位耗时对比", pad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.12, f"{val:.3f}s", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    out = OUT / "fig5_17_benchmark_time_split.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out


def make_benchmark_speedup() -> Path:
    labels = ["收缩完整推理", "收缩稀疏重建", "弯曲完整推理", "弯曲稀疏重建"]
    vals = [4.34, 1.98, 30.44, 9.01]
    colors = ["#4c78a8", "#72b7b2", "#4c78a8", "#72b7b2"]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.barh(y, vals, color=colors, height=0.58)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("CFD / PINN加速倍数")
    ax.set_title("相对加速倍数", pad=8)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.set_xlim(0, max(vals) * 1.18)
    for bar, val in zip(bars, vals):
        ax.text(val + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f}x", va="center", fontsize=11)
    fig.tight_layout()
    out = OUT / "fig5_18_benchmark_speedup_split.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_coupling_vertical() -> Path:
    dual_sparse = load_json(
        PROJECT_ROOT / "results" / "pinn" / "contraction_independent_geometry_sparse5clean_stagepde_v4" / "metrics.json"
    )
    single_sparse = load_json(
        PROJECT_ROOT / "results" / "supervised" / "contraction_single_mlp_geometry_sparse5_20260502" / "metrics.json"
    )
    history = pd.read_csv(
        PROJECT_ROOT / "results" / "pinn" / "contraction_independent_geometry_notemplate_stagepde_mainline_v4" / "history.csv"
    )
    stage2_last = history[history["阶段"] == "第二阶段_压力模型"].iloc[-1]
    stage3_last = history[history["阶段"] == "第三阶段_交替耦合"].iloc[-1]

    dual_case = dual_sparse["验证工况指标"][0]
    single_case = single_sparse["val_case_metrics"][0]
    dual_final = dual_sparse["最终验证指标"]
    single_extreme = single_sparse["val_extreme_metrics"]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.3), gridspec_kw={"height_ratios": [1.02, 1.0]})
    ax = axes[0]
    labels = ["速度Rel-L2", "压力Rel-L2", "最大速度误差", "壁面u残余"]
    single_values = [
        single_case["rel_l2_speed"],
        single_case["rel_l2_p"],
        single_extreme["max_speed_err_over_speed_max"],
        single_case["wall_max_abs_u_pred"],
    ]
    dual_values = [
        dual_final["rel_l2_speed"],
        dual_final["rel_l2_p"],
        dual_final["max_speed_err_over_speed_max"],
        dual_case["wall_max_abs_u_pred"],
    ]
    x = np.arange(len(labels))
    width = 0.35
    b1 = ax.bar(x - width / 2, single_values, width=width, color="#f59e0b", label="单网络MLP")
    b2 = ax.bar(x + width / 2, dual_values, width=width, color="#0f766e", label="双模型PDE耦合")
    ymax = max(single_values + dual_values)
    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            label = f"{val * 100:.2f}%" if val >= 0.001 else f"{val:.1e}"
            ax.text(bar.get_x() + bar.get_width() / 2, val + ymax * 0.025, label, ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("C-val验证指标")
    ax.set_title("（a）5%稀疏监督下单网络与双模型对比", pad=8, fontsize=14)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_ylim(0, ymax * 1.30)
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=10.5)

    ax = axes[1]
    labels2 = ["速度Rel-L2", "压力Rel-L2", "最大速度误差", "最大压力误差"]
    uncoupled = [
        float(stage2_last["验证_rel_l2_speed"]),
        float(stage2_last["验证_rel_l2_p"]),
        float(stage2_last["验证_max_speed_err_over_speed_max"]),
        float(stage2_last["验证_max_p_err_over_p_range"]),
    ]
    coupled = [
        float(stage3_last["验证_rel_l2_speed"]),
        float(stage3_last["验证_rel_l2_p"]),
        float(stage3_last["验证_max_speed_err_over_speed_max"]),
        float(stage3_last["验证_max_p_err_over_p_range"]),
    ]
    y = np.arange(len(labels2))
    height = 0.32
    u = ax.barh(y - height / 2, uncoupled, height=height, color="#fb923c", label="耦合前")
    c = ax.barh(y + height / 2, coupled, height=height, color="#2563eb", label="耦合后")
    xmax = max(uncoupled + coupled)
    for bars in (u, c):
        for bar in bars:
            val = bar.get_width()
            ax.text(val + xmax * 0.025, bar.get_y() + bar.get_height() / 2, f"{val * 100:.2f}%", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(labels2)
    ax.invert_yaxis()
    ax.set_xlabel("验证误差百分比")
    ax.set_title("（b）同一双模型主线内的耦合前后对比", pad=8, fontsize=14)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0, xmax * 1.30)

    fig.suptitle("单网络基线与双模型耦合效果对照", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.955))
    out = OUT / "fig5_19_coupling_vertical.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_plot()
    outputs = [
        make_sampling_layout(),
        make_sampling_region_error(),
        make_benchmark_time(),
        make_benchmark_speedup(),
        make_coupling_vertical(),
    ]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
