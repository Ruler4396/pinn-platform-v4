#!/usr/bin/env python3
"""重导出并整理第 5 章论文插图。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting import 配置中文绘图


THESIS_ROOT = PROJECT_ROOT / "results" / "thesis_assets" / "chapter5"
CH5_MATERIALS = PROJECT_ROOT / "results" / "field_map_checks" / "chapter5_materials"
V3_ROOT = Path(os.environ.get("PINN_PLATFORM_V3_ROOT", PROJECT_ROOT.parent / "legacy" / "pinn_v3"))
V4_ROOT = PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="重导出并整理第 5 章论文插图")
    parser.add_argument("--max-retries", type=int, default=1, help="最大重试次数，避免无限重试")
    return parser


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_field_maps(predictions_dir: Path, cases: list[str], output_dir: Path, max_retries: int) -> None:
    reset_dir(output_dir)
    command = [
        "python3",
        str(PROJECT_ROOT / "scripts" / "export_field_maps.py"),
        "--predictions-dir",
        str(predictions_dir),
        "--cases",
        ",".join(cases),
        "--output-dir",
        str(output_dir),
        "--panel-layout",
        "stack",
        "--colorbar-orientation",
        "horizontal",
        "--error-mode",
        "rel",
        "--percent-scale",
        "--mask-outside-geometry",
        "--max-retries",
        str(max_retries),
    ]
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def rename_case_exports(section_dir: Path, case_id: str, prefix: str, delete_manifest: bool = False) -> list[str]:
    rename_pairs = {
        f"{case_id}_speed_truth_pred_rel_error_pct.png": f"{prefix}_speed_map.png",
        f"{case_id}_pressure_truth_pred_rel_error_pct.png": f"{prefix}_pressure_map.png",
        f"{case_id}_speed_pressure_truth_pred_rel_error_pct.png": f"{prefix}_combined_map.png",
    }
    created: list[str] = []
    for old_name, new_name in rename_pairs.items():
        old_path = section_dir / old_name
        new_path = section_dir / new_name
        if not old_path.exists():
            raise FileNotFoundError(f"缺少导图文件：{old_path}")
        old_path.rename(new_path)
        created.append(new_name)
    manifest_path = section_dir / "manifest.json"
    if manifest_path.exists():
        if delete_manifest:
            manifest_path.unlink()
        else:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw["renamed_files"] = created
            manifest_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return created


def plot_baseline_convergence(section_dir: Path) -> list[str]:
    配置中文绘图(prefer_serif=False)
    history = pd.read_csv(V4_ROOT / "results" / "pinn" / "contraction_independent_geometry_notemplate_stagepde_mainline_v4" / "history.csv")
    history["训练步"] = np.arange(1, len(history) + 1)
    stage_boundaries = []
    count = 0
    for _, group in history.groupby("阶段", sort=False):
        count += len(group)
        stage_boundaries.append(count)

    outputs = []
    specs = [
        ("验证_rel_l2_speed", "velocity_convergence.png", "验证集速度相对二范数误差", "速度相对二范数误差"),
        ("验证_rel_l2_p", "pressure_convergence.png", "验证集压力相对二范数误差", "压力相对二范数误差"),
    ]
    for col, filename, title, ylabel in specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(history["训练步"], history[col], color="#1f77b4", linewidth=2.2)
        for boundary in stage_boundaries[:-1]:
            ax.axvline(boundary, color="#999999", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(f"收缩流道基线模型{title}收敛曲线")
        ax.set_xlabel("训练步")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        out = section_dir / filename
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        outputs.append(filename)
    return outputs


def plot_bend_sparse_rate(section_dir: Path) -> list[str]:
    配置中文绘图(prefer_serif=False)
    summary = pd.read_csv(CH5_MATERIALS / "bend_sparse_rate_summary.csv")
    order = ["dense", "1%", "5%", "10%", "15%"]
    summary["setting"] = pd.Categorical(summary["setting"], categories=order, ordered=True)
    summary = summary.sort_values("setting")

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(summary["setting"].astype(str), summary["rel_l2_speed"], marker="o", linewidth=2.2, color="#1f77b4", label="速度误差")
    ax1.plot(summary["setting"].astype(str), summary["rel_l2_p"], marker="s", linewidth=2.2, color="#ff7f0e", label="压力误差")
    ax1.set_title("弯曲流道不同稀疏采样率下的平均相对二范数误差")
    ax1.set_xlabel("观测采样率")
    ax1.set_ylabel("平均相对二范数误差")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)
    fig.tight_layout()
    summary_path = section_dir / "sparse_rate_summary.png"
    fig.savefig(summary_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    runs = {
        "dense": "bend_independent_blunted_dense_nobc_hardwall_clean_v1_20260401",
        "1%": "bend_independent_blunted_sparse1_nobc_hardwall_clean_v1_20260401",
        "5%": "bend_independent_blunted_sparse5_nobc_hardwall_clean_v1_20260401",
        "10%": "bend_independent_blunted_sparse10_nobc_hardwall_clean_v1_20260401",
        "15%": "bend_independent_blunted_sparse15_nobc_hardwall_clean_v1_20260401",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
    for color, (label, run_name) in zip(palette, runs.items()):
        history = pd.read_csv(V3_ROOT / "results" / "pinn" / run_name / "history.csv")
        x = np.arange(1, len(history) + 1)
        y = history["验证_rel_l2_speed"]
        ax.plot(x, y, label=label, linewidth=2.0, color=color)
    ax.set_title("弯曲流道不同采样率下的验证集速度误差收敛曲线")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("验证集速度相对二范数误差")
    ax.grid(alpha=0.25)
    ax.legend(title="采样率", frameon=False)
    fig.tight_layout()
    convergence_path = section_dir / "sparse_rate_convergence.png"
    fig.savefig(convergence_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [summary_path.name, convergence_path.name]


def plot_region_vs_uniform(section_dir: Path) -> list[str]:
    配置中文绘图(prefer_serif=False)
    summary = pd.read_csv(CH5_MATERIALS / "contraction_regionaware_vs_uniform5_summary.csv")
    outputs: list[str] = []

    fig, axes = plt.subplots(3, 1, figsize=(7.8, 10.2), constrained_layout=True)
    metrics = [
        ("mean_rel_l2_speed", "平均速度相对二范数误差"),
        ("mean_rel_l2_p", "平均压力相对二范数误差"),
        ("mean_pressure_drop_rel_error", "平均压降相对误差"),
    ]
    colors = {"region-aware": "#1f77b4", "uniform": "#ff7f0e"}
    splits = ["val", "test"]
    x = np.arange(len(splits))
    width = 0.32
    for ax, (col, title) in zip(axes, metrics):
        for idx, strategy in enumerate(["region-aware", "uniform"]):
            values = [
                float(summary[(summary["strategy"] == strategy) & (summary["split"] == split)][col].iloc[0])
                for split in splits
            ]
            bars = ax.bar(x + (-0.5 + idx) * width, values, width=width, color=colors[strategy], label="分区感知" if strategy == "region-aware" else "均匀采样")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["验证集", "测试集"])
        ax.set_title(title)
        ax.set_ylabel("误差")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    metrics_path = section_dir / "metrics_comparison.png"
    fig.savefig(metrics_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    outputs.append(metrics_path.name)

    fig, axes = plt.subplots(2, 1, figsize=(7.8, 7.8), constrained_layout=True)
    history_map = {
        "分区感知": V4_ROOT / "results" / "pinn" / "contraction_independent_geometry_sparse5clean_stagepde_v4" / "history.csv",
        "均匀采样": V4_ROOT / "results" / "pinn" / "contraction_independent_geometry_uniform5clean_stagepde_v4" / "history.csv",
    }
    for label, path in history_map.items():
        history = pd.read_csv(path)
        x = np.arange(1, len(history) + 1)
        axes[0].plot(x, history["验证_rel_l2_speed"], linewidth=2.1, label=label)
        axes[1].plot(x, history["验证_rel_l2_p"], linewidth=2.1, label=label)
    axes[0].set_title("验证集速度误差收敛对比")
    axes[1].set_title("验证集压力误差收敛对比")
    for ax in axes:
        ax.set_xlabel("训练步")
        ax.set_ylabel("相对二范数误差")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    convergence_path = section_dir / "convergence_comparison.png"
    fig.savefig(convergence_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    outputs.append(convergence_path.name)

    field = pd.read_csv(V4_ROOT / "cases" / "contraction_2d" / "data" / "C-val" / "field_dense.csv")
    region = pd.read_csv(V4_ROOT / "cases" / "contraction_2d" / "data" / "C-val" / "obs_sparse_5pct.csv")
    uniform = pd.read_csv(V4_ROOT / "cases" / "contraction_2d" / "data" / "C-val" / "obs_uniform_5pct.csv")
    palette = {0: "#4C78A8", 1: "#F58518", 2: "#54A24B", 3: "#E45756"}
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 8.6), sharex=True, sharey=True, constrained_layout=True)
    all_regions = sorted(set(region["region_id"]).union(set(uniform["region_id"])))
    for ax, obs, title in [
        (axes[0], region, "分区感知 5% 观测点布局"),
        (axes[1], uniform, "均匀 5% 观测点布局"),
    ]:
        ax.scatter(field["x_star"], field["y_star"], s=4, color="#d9d9d9", alpha=0.35, linewidths=0)
        for rid, grp in obs.groupby("region_id"):
            rid_int = int(rid)
            ax.scatter(
                grp["x_star"],
                grp["y_star"],
                s=28,
                color=palette.get(rid_int, "#333333"),
                edgecolors="white",
                linewidths=0.3,
                label=f"区域 {rid_int}",
            )
        ax.set_title(title)
        ax.set_xlabel("x*")
        ax.set_ylabel("y*")
        ax.grid(alpha=0.18)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"区域 {rid}",
            markerfacecolor=palette.get(int(rid), "#333333"),
            markeredgecolor="white",
            markeredgewidth=0.3,
            markersize=7,
        )
        for rid in all_regions
    ]
    axes[1].legend(handles=handles, frameon=False, loc="upper right", ncol=len(handles))
    layout_path = section_dir / "sampling_layout_comparison.png"
    fig.savefig(layout_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    outputs.append(layout_path.name)
    return outputs


def plot_noise_summary(section_dir: Path) -> list[str]:
    配置中文绘图(prefer_serif=False)
    summary = pd.read_csv(CH5_MATERIALS / "contraction_dense_vs_noise3_summary.csv")
    labels = ["稠密观测", "5%含噪观测"]
    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(2, 1, figsize=(7.8, 7.8), constrained_layout=True)
    speed_val = summary["val_rel_l2_speed"].to_numpy(dtype=float)
    speed_test = summary["test_rel_l2_speed"].to_numpy(dtype=float)
    pressure_val = summary["val_rel_l2_p"].to_numpy(dtype=float)
    pressure_test = summary["test_rel_l2_p"].to_numpy(dtype=float)

    speed_bars_1 = axes[0].bar(x - width / 2, speed_val, width=width, label="验证集速度", color="#1f77b4")
    speed_bars_2 = axes[0].bar(x + width / 2, speed_test, width=width, label="测试集速度", color="#ff7f0e")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("速度重建误差对比")
    axes[0].set_ylabel("相对 L2 误差")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper left")
    for bars in (speed_bars_1, speed_bars_2):
        for bar in bars:
            height = float(bar.get_height())
            axes[0].text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    pressure_bars_1 = axes[1].bar(x - width / 2, pressure_val, width=width, label="验证集压力", color="#2ca02c")
    pressure_bars_2 = axes[1].bar(x + width / 2, pressure_test, width=width, label="测试集压力", color="#d62728")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("压力重建误差对比")
    axes[1].set_ylabel("相对 L2 误差")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")
    for bars in (pressure_bars_1, pressure_bars_2):
        for bar in bars:
            height = float(bar.get_height())
            axes[1].text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    out = section_dir / "dense_vs_noise_summary.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [out.name]


def plot_stagepde_comparison(section_dir: Path) -> list[str]:
    配置中文绘图(prefer_serif=False)
    summary = pd.read_csv(CH5_MATERIALS / "contraction_stagepde_comparison.csv")
    labels = ["不启用阶段 PDE", "启用阶段 PDE"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(x - width, summary["pressure_stage_rel_l2_p"], width=width, label="第二阶段结束", color="#ffbb78")
    ax.bar(x, summary["final_rel_l2_p"], width=width, label="最终压力误差", color="#1f77b4")
    ax.bar(x + width, summary["final_max_p_err"], width=width, label="最终最大压力误差", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("误差")
    ax.set_title("阶段内 PDE 约束对压力重建稳定性的影响")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = section_dir / "stagepde_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [out.name]


def plot_dual_model_coupled_training_loss(section_dir: Path) -> list[str]:
    配置中文绘图(prefer_serif=False)
    history = pd.read_csv(
        V4_ROOT / "results" / "pinn" / "contraction_independent_geometry_notemplate_stagepde_mainline_v4" / "history.csv"
    )
    history["训练步"] = np.arange(1, len(history) + 1)

    stage_spans: list[tuple[str, int, int]] = []
    start = 1
    for stage_name, group in history.groupby("阶段", sort=False):
        end = start + len(group) - 1
        stage_spans.append((str(stage_name), start, end))
        start = end + 1

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    bg_colors = ["#f6efe7", "#eef4fb", "#edf7ed"]
    stage_labels = {
        "第一阶段_速度模型": "阶段 1\n速度模型预训练",
        "第二阶段_压力模型": "阶段 2\n压力模型预训练",
        "第三阶段_交替耦合": "阶段 3\n交替耦合微调",
    }

    y_speed = history["验证_rel_l2_speed"].to_numpy()
    y_pressure = history["验证_rel_l2_p"].to_numpy()
    y_max = float(max(np.nanmax(y_speed), np.nanmax(y_pressure)))
    y_min = float(min(np.nanmin(y_speed), np.nanmin(y_pressure)))
    y_span = max(y_max - y_min, 1.0e-6)
    stage_centers: list[float] = []
    stage_texts: list[str] = []

    for idx, (stage_name, span_start, span_end) in enumerate(stage_spans):
        ax.axvspan(span_start, span_end, color=bg_colors[idx % len(bg_colors)], alpha=0.65, zorder=0)
        stage_centers.append((span_start + span_end) / 2.0)
        stage_texts.append(stage_labels.get(stage_name, stage_name))
        if idx > 0:
            ax.axvline(span_start - 0.5, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)

    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(stage_centers)
    top_ax.set_xticklabels(stage_texts, fontsize=10, color="#444444")
    top_ax.tick_params(axis="x", length=0, pad=8)
    for spine in top_ax.spines.values():
        spine.set_visible(False)

    ax.plot(
        history["训练步"],
        history["验证_rel_l2_speed"],
        color="#1f77b4",
        linewidth=2.4,
        label="速度损失",
        zorder=3,
    )
    ax.plot(
        history["训练步"],
        history["验证_rel_l2_p"],
        color="#d62728",
        linewidth=2.4,
        label="压力损失",
        zorder=3,
    )

    final_row = history.iloc[-1]
    ax.scatter([final_row["训练步"]], [final_row["验证_rel_l2_speed"]], color="#1f77b4", s=28, zorder=4)
    ax.scatter([final_row["训练步"]], [final_row["验证_rel_l2_p"]], color="#d62728", s=28, zorder=4)
    ax.annotate(
        f"{float(final_row['验证_rel_l2_speed']):.4f}",
        xy=(float(final_row["训练步"]), float(final_row["验证_rel_l2_speed"])),
        xytext=(-10, -12),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=9,
        color="#1f77b4",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
    )
    ax.annotate(
        f"{float(final_row['验证_rel_l2_p']):.4f}",
        xy=(float(final_row["训练步"]), float(final_row["验证_rel_l2_p"])),
        xytext=(-10, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#d62728",
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
    )

    ax.set_xlim(1, len(history))
    ax.set_ylim(y_min - y_span * 0.06, y_max + y_span * 0.06)
    ax.set_xlabel("累计训练轮次")
    ax.set_ylabel("验证集相对二范数误差")
    ax.set_title("双模型耦合训练中速度与压力误差的阶段性下降")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))

    out = section_dir / "dual_model_coupled_loss.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [out.name]


def plot_single_vs_dual_coupling_comparison(section_dir: Path) -> list[str]:
    配置中文绘图(prefer_serif=False)
    history = pd.read_csv(
        V4_ROOT / "results" / "pinn" / "contraction_independent_geometry_notemplate_stagepde_mainline_v4" / "history.csv"
    )

    stage2_last = history[history["阶段"] == "第二阶段_压力模型"].iloc[-1]
    stage3_last = history[history["阶段"] == "第三阶段_交替耦合"].iloc[-1]

    metrics = [
        ("验证_rel_l2_speed", "速度 Rel-L2"),
        ("验证_rel_l2_p", "压力 Rel-L2"),
        ("验证_max_speed_err_over_speed_max", "最大速度误差"),
        ("验证_max_p_err_over_p_range", "最大压力误差"),
    ]
    single_vals = [float(stage2_last[col]) for col, _ in metrics]
    dual_vals = [float(stage3_last[col]) for col, _ in metrics]
    labels = [label for _, label in metrics]

    y = np.arange(len(labels))
    height = 0.32
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    single_color = "#ff8a5b"
    dual_color = "#0f766e"
    single_bars = ax.barh(y - height / 2, single_vals, height=height, color=single_color, label="单模型独立阶段")
    dual_bars = ax.barh(y + height / 2, dual_vals, height=height, color=dual_color, label="双模型耦合阶段")

    max_val = max(max(single_vals), max(dual_vals))
    x_pad = max(max_val * 0.18, 0.01)

    for bars in (single_bars, dual_bars):
        for bar in bars:
            width = float(bar.get_width())
            ax.text(
                width + x_pad * 0.12,
                bar.get_y() + bar.get_height() / 2,
                f"{width * 100:.2f}%",
                ha="left",
                va="center",
                fontsize=9,
                color="#333333",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, max_val + x_pad)
    ax.set_xlabel("验证误差百分比")
    ax.set_title("单模型独立阶段与双模型耦合阶段误差对比")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(rect=(0.06, 0.06, 0.98, 0.98))

    out = section_dir / "single_vs_dual_coupling_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [out.name]


def build_manifest(section_files: dict[str, list[str]]) -> None:
    manifest = {
        "root": str(THESIS_ROOT),
        "sections": {
            section: [str((THESIS_ROOT / section / name).relative_to(THESIS_ROOT)) for name in files]
            for section, files in section_files.items()
        },
    }
    (THESIS_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 第五章论文插图目录",
        "",
        "本目录存放已整理为论文可直接取用的第 5 章图片，所有图片均已显式指定中文字体重导出。",
        "",
        "## 命名约定",
        "- 先按章节小节分目录；",
        "- 同一目录下使用语义化文件名，优先表达工况、对比类型与图像用途；",
        "- 后续新增第 5 章图片继续沿用同样规则。",
        "",
    ]
    for section, files in section_files.items():
        lines.append(f"## {section}")
        for name in files:
            lines.append(f"- `{section}/{name}`")
        lines.append("")
    (THESIS_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_once(max_retries: int) -> None:
    reset_dir(THESIS_ROOT)
    section_files: dict[str, list[str]] = {}

    sec = THESIS_ROOT / "5_1_baseline_convergence"
    reset_dir(sec)
    section_files[sec.name] = plot_baseline_convergence(sec)

    sec = THESIS_ROOT / "5_2_baseline_reconstruction"
    export_field_maps(
        predictions_dir=V4_ROOT / "results" / "pinn" / "contraction_independent_geometry_notemplate_stagepde_mainline_v4" / "evaluations" / "predictions_val",
        cases=["C-val"],
        output_dir=sec,
        max_retries=max_retries,
    )
    files = rename_case_exports(sec, "C-val", "contraction_val", delete_manifest=True)
    export_field_maps(
        predictions_dir=V4_ROOT / "results" / "pinn" / "contraction_independent_geometry_notemplate_stagepde_mainline_v4" / "evaluations" / "predictions_test",
        cases=["C-test-1", "C-test-2"],
        output_dir=sec / "_tmp_test",
        max_retries=max_retries,
    )
    for case_id, prefix in [("C-test-1", "contraction_test_1"), ("C-test-2", "contraction_test_2")]:
        files.extend(rename_case_exports(sec / "_tmp_test", case_id, prefix, delete_manifest=True))
    for path in (sec / "_tmp_test").glob("*.png"):
        shutil.move(str(path), str(sec / path.name))
    shutil.rmtree(sec / "_tmp_test")
    section_files[sec.name] = sorted(set(files))

    sec = THESIS_ROOT / "5_3_sparse_rate"
    reset_dir(sec)
    files = plot_bend_sparse_rate(sec)
    export_field_maps(
        predictions_dir=V3_ROOT / "results" / "pinn" / "bend_independent_blunted_dense_nobc_hardwall_clean_v1_20260401" / "predictions",
        cases=["B-val__ip_blunted"],
        output_dir=sec / "_tmp_dense",
        max_retries=max_retries,
    )
    files.extend(rename_case_exports(sec / "_tmp_dense", "B-val__ip_blunted", "bend_dense_val", delete_manifest=True))
    for path in (sec / "_tmp_dense").glob("*.png"):
        shutil.move(str(path), str(sec / path.name))
    shutil.rmtree(sec / "_tmp_dense")
    export_field_maps(
        predictions_dir=V3_ROOT / "results" / "pinn" / "bend_independent_blunted_sparse5_nobc_hardwall_clean_v1_20260401" / "predictions",
        cases=["B-val__ip_blunted"],
        output_dir=sec / "_tmp_sparse5",
        max_retries=max_retries,
    )
    files.extend(rename_case_exports(sec / "_tmp_sparse5", "B-val__ip_blunted", "bend_sparse5_val", delete_manifest=True))
    for path in (sec / "_tmp_sparse5").glob("*.png"):
        shutil.move(str(path), str(sec / path.name))
    shutil.rmtree(sec / "_tmp_sparse5")
    section_files[sec.name] = sorted(set(files))

    sec = THESIS_ROOT / "5_4_region_vs_uniform"
    reset_dir(sec)
    files = plot_region_vs_uniform(sec)
    export_field_maps(
        predictions_dir=V4_ROOT / "results" / "pinn" / "contraction_independent_geometry_sparse5clean_stagepde_v4" / "evaluations" / "predictions_val",
        cases=["C-val"],
        output_dir=sec / "_tmp_region_val",
        max_retries=max_retries,
    )
    files.extend(rename_case_exports(sec / "_tmp_region_val", "C-val", "regionaware_val", delete_manifest=True))
    for path in (sec / "_tmp_region_val").glob("*.png"):
        shutil.move(str(path), str(sec / path.name))
    shutil.rmtree(sec / "_tmp_region_val")
    export_field_maps(
        predictions_dir=V4_ROOT / "results" / "pinn" / "contraction_independent_geometry_uniform5clean_stagepde_v4" / "evaluations" / "predictions_val",
        cases=["C-val"],
        output_dir=sec / "_tmp_uniform_val",
        max_retries=max_retries,
    )
    files.extend(rename_case_exports(sec / "_tmp_uniform_val", "C-val", "uniform_val", delete_manifest=True))
    for path in (sec / "_tmp_uniform_val").glob("*.png"):
        shutil.move(str(path), str(sec / path.name))
    shutil.rmtree(sec / "_tmp_uniform_val")
    export_field_maps(
        predictions_dir=V4_ROOT / "results" / "pinn" / "contraction_independent_geometry_sparse5clean_stagepde_v4" / "evaluations" / "predictions_test",
        cases=["C-test-1", "C-test-2"],
        output_dir=sec / "_tmp_region_test",
        max_retries=max_retries,
    )
    for case_id, prefix in [("C-test-1", "regionaware_test_1"), ("C-test-2", "regionaware_test_2")]:
        files.extend(rename_case_exports(sec / "_tmp_region_test", case_id, prefix, delete_manifest=True))
    for path in (sec / "_tmp_region_test").glob("*.png"):
        shutil.move(str(path), str(sec / path.name))
    shutil.rmtree(sec / "_tmp_region_test")
    export_field_maps(
        predictions_dir=V4_ROOT / "results" / "pinn" / "contraction_independent_geometry_uniform5clean_stagepde_v4" / "evaluations" / "predictions_test",
        cases=["C-test-1", "C-test-2"],
        output_dir=sec / "_tmp_uniform_test",
        max_retries=max_retries,
    )
    for case_id, prefix in [("C-test-1", "uniform_test_1"), ("C-test-2", "uniform_test_2")]:
        files.extend(rename_case_exports(sec / "_tmp_uniform_test", case_id, prefix, delete_manifest=True))
    for path in (sec / "_tmp_uniform_test").glob("*.png"):
        shutil.move(str(path), str(sec / path.name))
    shutil.rmtree(sec / "_tmp_uniform_test")
    section_files[sec.name] = sorted(set(files))

    sec = THESIS_ROOT / "5_5_noise_robustness"
    reset_dir(sec)
    files = plot_noise_summary(sec)
    export_field_maps(
        predictions_dir=V4_ROOT / "results" / "pinn" / "contraction_independent_geometry_sparse5noise3_stagepde_v4" / "evaluations" / "predictions_val",
        cases=["C-val"],
        output_dir=sec / "_tmp_noise_val",
        max_retries=max_retries,
    )
    files.extend(rename_case_exports(sec / "_tmp_noise_val", "C-val", "noise3_val", delete_manifest=True))
    for path in (sec / "_tmp_noise_val").glob("*.png"):
        shutil.move(str(path), str(sec / path.name))
    shutil.rmtree(sec / "_tmp_noise_val")
    section_files[sec.name] = sorted(set(files))

    sec = THESIS_ROOT / "5_6_generalization"
    reset_dir(sec)
    export_field_maps(
        predictions_dir=V3_ROOT / "results" / "pinn" / "bend_targetrecon_independent_btest1_sparse5_nobc_hardwall_stagepde_v1_20260401" / "evaluations" / "predictions_target",
        cases=["B-test-1__ip_blunted"],
        output_dir=sec,
        max_retries=max_retries,
    )
    files = rename_case_exports(sec, "B-test-1__ip_blunted", "bend_target_test_1", delete_manifest=True)
    section_files[sec.name] = sorted(set(files))

    sec = THESIS_ROOT / "5_7_training_stability"
    reset_dir(sec)
    files = plot_stagepde_comparison(sec)
    files.extend(plot_dual_model_coupled_training_loss(sec))
    files.extend(plot_single_vs_dual_coupling_comparison(sec))
    section_files[sec.name] = sorted(set(files))

    build_manifest(section_files)


def main() -> None:
    args = build_parser().parse_args()
    attempts = max(1, int(args.max_retries))
    last_error: Exception | None = None
    for _attempt in range(1, attempts + 1):
        try:
            prepare_once(max_retries=args.max_retries)
            print(f"[完成] 输出目录：{THESIS_ROOT}")
            return
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if _attempt >= attempts:
                raise
            print(f"[第 {_attempt}/{attempts} 次尝试失败] {exc}")
    if last_error is not None:
        raise last_error


if __name__ == "__main__":
    main()
