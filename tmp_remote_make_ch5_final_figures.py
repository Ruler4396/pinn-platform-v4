from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("model").resolve()
OUT = PROJECT_ROOT / "results" / "field_map_checks" / "chapter5_materials" / "reworked_figures_v3"


def setup_plot() -> None:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.plotting import 配置中文绘图

    配置中文绘图(prefer_serif=False)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.unicode_minus": False,
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_summary(run_name: str, split: str) -> dict:
    if split == "val":
        path = PROJECT_ROOT / "results" / "pinn" / run_name / "evaluations" / "metrics_val.json"
    else:
        path = PROJECT_ROOT / "results" / "pinn" / run_name / "evaluations" / "metrics_test.json"
    return load_json(path)["summary"]


def run_name(strategy: str, rate: int) -> str:
    if rate == 5:
        if strategy == "region-aware":
            return "contraction_independent_geometry_sparse5clean_stagepde_v4"
        return "contraction_independent_geometry_uniform5clean_stagepde_v4"
    tag = "sparse" if strategy == "region-aware" else "uniform"
    return f"contraction_independent_geometry_{tag}{rate}clean_stagepde_v4_ratecheck_20260502"


def make_sampling_rate_check() -> Path:
    rates = [1, 5, 10, 15]
    rows = []
    for rate in rates:
        for strategy in ["region-aware", "uniform"]:
            rn = run_name(strategy, rate)
            for split in ["val", "test"]:
                summary = metric_summary(rn, split)
                rows.append(
                    {
                        "rate": rate,
                        "strategy": strategy,
                        "split": split,
                        "speed": summary["mean_rel_l2_speed"],
                        "pressure": summary["mean_rel_l2_p"],
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "contraction_sampling_rate_strategy_check.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    metric_specs = [("speed", "速度Rel-L2"), ("pressure", "压力Rel-L2")]
    style = {
        ("region-aware", "val"): ("#0f766e", "o", "-", "分层-验证"),
        ("uniform", "val"): ("#f59e0b", "s", "-", "均匀-验证"),
        ("region-aware", "test"): ("#0f766e", "o", "--", "分层-测试"),
        ("uniform", "test"): ("#f59e0b", "s", "--", "均匀-测试"),
    }
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        for key, (color, marker, linestyle, label) in style.items():
            strategy, split = key
            part = df[(df["strategy"] == strategy) & (df["split"] == split)].sort_values("rate")
            ax.plot(
                part["rate"],
                part[metric] * 100.0,
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=1.6,
                markersize=4.5,
                label=label,
            )
        ax.set_ylabel(ylabel + "（%）")
        ax.grid(alpha=0.25, linestyle=":")
    axes[0].set_title("不同采样率下采样策略对重建误差的影响", pad=6)
    axes[-1].set_xlabel("观测采样率（%）")
    axes[-1].set_xticks(rates)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.98))
    out = OUT / "fig5_8_sampling_rate_strategy_check.png"
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

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    specs = [
        ("region-aware", "分层", "#0f766e", ""),
        ("uniform", "均匀", "#f59e0b", "//"),
    ]
    metric_specs = [("mean_rel_l2_speed", "速度", 0.92), ("mean_rel_l2_p", "压力", 0.56)]
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
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.005,
                    f"{val * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([region_names[rid] for rid in regions])
    ax.set_ylabel("验证集分区相对误差")
    ax.set_title("5%采样率下两种策略的分区误差", pad=6)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_ylim(0.0, float(metrics[["mean_rel_l2_speed", "mean_rel_l2_p"]].to_numpy().max()) * 1.42)
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.98))
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

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars1 = ax.bar(x - width, pinn_full, width=width, color="#4c78a8", label="PINN完整推理")
    bars2 = ax.bar(x, pinn_sparse, width=width, color="#72b7b2", label="PINN稀疏重建")
    bars3 = ax.bar(x + width, cfd, width=width, color="#bab0ac", label="CFD求解")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("中位耗时（秒，对数坐标）")
    ax.set_title("同一环境下PINN与CFD中位耗时对比", pad=6)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    ax.set_ylim(0.06, 7.5)
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.08, f"{val:.3f}s", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = OUT / "fig5_17_benchmark_time_split.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out


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

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.9), gridspec_kw={"height_ratios": [1.05, 1.0]})

    ax = axes[0]
    labels = ["速度Rel-L2", "压力Rel-L2", "最大速度误差", "壁面u残余"]
    single_values = [
        single_case["rel_l2_speed"] * 100.0,
        single_case["rel_l2_p"] * 100.0,
        single_extreme["max_speed_err_over_speed_max"] * 100.0,
        single_case["wall_max_abs_u_pred"] * 100.0,
    ]
    dual_values = [
        dual_final["rel_l2_speed"] * 100.0,
        dual_final["rel_l2_p"] * 100.0,
        dual_final["max_speed_err_over_speed_max"] * 100.0,
        dual_case["wall_max_abs_u_pred"] * 100.0,
    ]
    y = np.arange(len(labels))
    height = 0.32
    b1 = ax.barh(y - height / 2, single_values, height=height, color="#f59e0b", label="单网络MLP")
    b2 = ax.barh(y + height / 2, dual_values, height=height, color="#0f766e", label="双模型PDE耦合")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("验证指标（%，对数坐标）")
    ax.set_title("（a）单网络（橙）与双模型（绿）", pad=6)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.set_xlim(0.002, 45)
    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_width()
            label = f"{val:.4f}%" if val < 0.01 else f"{val:.2f}%"
            ax.text(val * 1.12, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)

    ax = axes[1]
    labels2 = ["速度Rel-L2", "压力Rel-L2", "最大速度误差", "最大压力误差"]
    uncoupled = [
        float(stage2_last["验证_rel_l2_speed"]) * 100.0,
        float(stage2_last["验证_rel_l2_p"]) * 100.0,
        float(stage2_last["验证_max_speed_err_over_speed_max"]) * 100.0,
        float(stage2_last["验证_max_p_err_over_p_range"]) * 100.0,
    ]
    coupled = [
        float(stage3_last["验证_rel_l2_speed"]) * 100.0,
        float(stage3_last["验证_rel_l2_p"]) * 100.0,
        float(stage3_last["验证_max_speed_err_over_speed_max"]) * 100.0,
        float(stage3_last["验证_max_p_err_over_p_range"]) * 100.0,
    ]
    y = np.arange(len(labels2))
    height = 0.32
    u = ax.barh(y - height / 2, uncoupled, height=height, color="#fb923c", label="耦合前")
    c = ax.barh(y + height / 2, coupled, height=height, color="#2563eb", label="耦合后")
    xmax = max(uncoupled + coupled)
    for bars in (u, c):
        for bar in bars:
            val = bar.get_width()
            ax.text(val + xmax * 0.025, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels2)
    ax.invert_yaxis()
    ax.set_xlabel("验证误差（%）")
    ax.set_title("（b）同一双模型主线内的耦合前后对比", pad=6)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0, xmax * 1.34)

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.985))
    out = OUT / "fig5_19_coupling_vertical.png"
    fig.savefig(out, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_plot()
    outputs = [
        make_sampling_rate_check(),
        make_sampling_region_error(),
        make_benchmark_time(),
        make_coupling_vertical(),
    ]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
