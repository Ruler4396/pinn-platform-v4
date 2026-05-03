#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting import 配置中文绘图
from scripts.export_field_maps import (
    构造三角剖分,
    构造几何对象,
    构造几何裁剪路径,
    绘制几何边界,
    规则网格插值,
    计算相对误差,
)

OUT = PROJECT_ROOT / "results" / "thesis_assets" / "chapter5_strict_sparse_20260503"


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(run: str, split: str = "val_dense", kind: str = "pinn") -> dict:
    base = PROJECT_ROOT / "results" / kind / run / "evaluations" / f"metrics_{split}.json"
    return load_json(base)


def export_maps(predictions_dir: Path, cases: list[str], output_dir: Path, layout: str = "stack") -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "export_field_maps.py"),
        "--predictions-dir",
        str(predictions_dir),
        "--cases",
        ",".join(cases),
        "--output-dir",
        str(output_dir),
        "--panel-layout",
        layout,
        "--colorbar-orientation",
        "horizontal",
        "--dpi",
        "220",
        "--max-retries",
        "1",
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    return sorted(p.name for p in output_dir.glob("*.png"))


def export_focused_speed_map(prediction_csv: Path, output_path: Path, *, title_prefix: str) -> str:
    from matplotlib.ticker import MaxNLocator

    配置中文绘图(prefer_serif=False)
    frame = pd.read_csv(prediction_csv)
    case_id = str(frame["case_id"].iloc[0])
    x = frame["x_star"].to_numpy(dtype=float)
    y = frame["y_star"].to_numpy(dtype=float)
    geometry = 构造几何对象(case_id)
    triang = 构造三角剖分(x, y, case_id, mask_outside_geometry=True)

    speed_true = np.sqrt(frame["u_true"].to_numpy(dtype=float) ** 2 + frame["v_true"].to_numpy(dtype=float) ** 2)
    speed_pred = np.sqrt(frame["u_pred"].to_numpy(dtype=float) ** 2 + frame["v_pred"].to_numpy(dtype=float) ** 2)
    speed_scale = max(float(np.nanmax(speed_true)), 1.0e-12)
    speed_err = 计算相对误差(speed_pred, speed_true, 0.01 * speed_scale) * 100.0

    vmin = float(min(speed_true.min(), speed_pred.min()))
    vmax = float(max(speed_true.max(), speed_pred.max()))
    # Near-wall low-speed cells can inflate relative error by the denominator;
    # cap the display range so the map shows spatial structure rather than a few outliers.
    err_vmax = min(max(float(np.nanpercentile(speed_err, 99.0)), 1.0e-12), 12.0)

    fig, axes = plt.subplots(3, 1, figsize=(5.8, 10.4), constrained_layout=True)
    panels = [
        (speed_true, "真值", "turbo", vmin, vmax, "速度模值"),
        (speed_pred, "预测", "turbo", vmin, vmax, "速度模值"),
        (speed_err, "相对误差", "magma", 0.0, err_vmax, "相对误差（%）"),
    ]
    for ax, (values, title, cmap, cmin, cmax, cbar_label) in zip(axes, panels):
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(alpha=0.0)
        grid_values, extent = 规则网格插值(triang, values, geometry, x, y)
        artist = ax.imshow(
            np.ma.masked_invalid(grid_values),
            extent=extent,
            origin="lower",
            interpolation="bicubic",
            cmap=cmap_obj,
            vmin=cmin,
            vmax=cmax,
            aspect="auto",
        )
        artist.set_clip_path(构造几何裁剪路径(ax, geometry))
        绘制几何边界(ax, geometry)
        ax.set_title(title, fontsize=11, pad=2)
        ax.set_xlabel("x*", fontsize=9)
        ax.set_ylabel("y*", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", labelsize=8, pad=1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        cbar = fig.colorbar(artist, ax=ax, orientation="vertical", fraction=0.028, pad=0.012)
        cbar.set_label(cbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle(title_prefix, fontsize=12, y=1.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path.name


def savefig(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path.name


def pct_label(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def rel_label(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def add_point_labels(ax, xs, ys, *, fmt=pct_label, dy: float | None = None, fontsize: int = 8) -> None:
    ys = list(ys)
    if not ys:
        return
    offset = dy if dy is not None else max(ys) * 0.035
    for x, y in zip(xs, ys):
        ax.text(x, y + offset, fmt(float(y)), ha="center", va="bottom", fontsize=fontsize)


def add_bar_labels(ax, bars, *, fmt=pct_label, log_scale: bool = False, fontsize: int = 8) -> None:
    heights = [bar.get_height() for bar in bars]
    if not heights:
        return
    ymax = max(heights)
    for bar in bars:
        value = float(bar.get_height())
        y = value * 1.18 if log_scale else value + ymax * 0.035
        ax.text(bar.get_x() + bar.get_width() / 2, y, fmt(value), ha="center", va="bottom", fontsize=fontsize)


def add_final_label(ax, xs, ys, *, fmt=pct_label, color: str | None = None, fontsize: int = 8) -> None:
    xs = list(xs)
    ys = list(ys)
    if not xs:
        return
    ax.annotate(
        fmt(float(ys[-1])),
        xy=(xs[-1], ys[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=fontsize,
        color=color,
    )


def plot_bend_sparse_rate(sec: Path) -> list[str]:
    rates = [1, 5, 10, 15]
    rows = []
    for rate in rates:
        run = f"bend_strict_blunted_sparse{rate}_stagepde_20260503"
        data = metric(run, "val_dense")
        s = data["summary"]
        rows.append(
            {
                "rate": rate,
                "speed": float(s["mean_rel_l2_speed"]),
                "pressure": float(s["mean_rel_l2_p"]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(sec / "bend_strict_sparse_rate_summary.csv", index=False)

    files = []
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(df["rate"], df["speed"], marker="o", label="速度Rel-L2", color="#0f766e")
    ax.plot(df["rate"], df["pressure"], marker="s", label="压力Rel-L2", color="#c2410c")
    add_point_labels(ax, df["rate"], df["speed"], fmt=pct_label)
    add_point_labels(ax, df["rate"], df["pressure"], fmt=pct_label)
    ax.set_xlabel("采样率/%")
    ax.set_ylabel("验证集Rel-L2")
    ax.set_title("稀疏观测下不同采样率的重建误差")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(df["speed"].max(), df["pressure"].max()) * 1.32)
    files.append(savefig(fig, sec / "strict_sparse_rate_summary.png"))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for rate in rates:
        hist = pd.read_csv(PROJECT_ROOT / "results" / "pinn" / f"bend_strict_blunted_sparse{rate}_stagepde_20260503" / "history.csv")
        stage = hist[hist["阶段"] == "第三阶段_交替耦合"].copy()
        x = np.arange(len(stage)) + 1
        ax.plot(x, stage["验证_rel_l2_speed"], label=f"{rate}%")
    ax.set_xlabel("耦合阶段记录点")
    ax.set_ylabel("验证速度Rel-L2")
    ax.set_title("稀疏观测下速度误差收敛过程")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False, ncol=4)
    files.append(savefig(fig, sec / "strict_sparse_rate_convergence.png"))

    files.extend(
        export_maps(
            PROJECT_ROOT / "results" / "pinn" / "bend_strict_blunted_sparse5_stagepde_20260503" / "evaluations" / "predictions_val_dense",
            ["B-val__ip_blunted"],
            sec / "bend_sparse5_val_maps",
        )
    )
    files.append(
        export_focused_speed_map(
            PROJECT_ROOT
            / "results"
            / "pinn"
            / "bend_independent_blunted_geometry_notemplate_medium_v1_20260401"
            / "predictions"
            / "B-val__ip_blunted_predictions.csv",
            sec / "bend_sparse5_val_maps" / "B-val__ip_blunted_speed_dense_focused.png",
            title_prefix="弯曲流道B-val稠密配置速度场重建",
        )
    )
    files.append(
        export_focused_speed_map(
            PROJECT_ROOT
            / "results"
            / "pinn"
            / "bend_strict_blunted_sparse5_stagepde_20260503"
            / "evaluations"
            / "predictions_val_dense"
            / "B-val__ip_blunted_predictions.csv",
            sec / "bend_sparse5_val_maps" / "B-val__ip_blunted_speed_focused.png",
            title_prefix="弯曲流道B-val 5%稀疏观测速度场重建",
        )
    )
    return files


def plot_region_uniform(sec: Path) -> list[str]:
    rates = [1, 5, 10, 15]
    rows = []
    for rate in rates:
        for strategy, prefix in [("分区感知", "sparse"), ("均匀采样", "uniform")]:
            run = f"contraction_strict_geometry_{prefix}{rate}_stagepde_20260503"
            data = metric(run, "val_dense")
            s = data["summary"]
            rows.append(
                {
                    "rate": rate,
                    "strategy": strategy,
                    "speed": float(s["mean_rel_l2_speed"]),
                    "pressure": float(s["mean_rel_l2_p"]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(sec / "contraction_strict_region_uniform_rate_summary.csv", index=False)
    files = []

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True)
    for ax, col, title in [(axes[0], "speed", "速度Rel-L2"), (axes[1], "pressure", "压力Rel-L2")]:
        for strategy, color in [("分区感知", "#0f766e"), ("均匀采样", "#f59e0b")]:
            sub = df[df["strategy"] == strategy]
            ax.plot(sub["rate"], sub[col], marker="o", label=strategy, color=color)
            add_point_labels(ax, sub["rate"], sub[col], fmt=pct_label)
        ax.set_title(title)
        ax.set_xlabel("采样率/%")
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_ylim(0, df[col].max() * 1.35)
    axes[0].set_ylabel("验证集Rel-L2")
    axes[1].legend(frameon=False)
    fig.suptitle("稀疏观测下采样策略与采样率的影响", y=1.02)
    files.append(savefig(fig, sec / "strict_region_uniform_rate_comparison.png"))

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = ["验证速度", "验证压力", "测试速度", "测试压力"]
    vals = {}
    for strategy, prefix in [("分区感知", "sparse"), ("均匀采样", "uniform")]:
        run = f"contraction_strict_geometry_{prefix}5_stagepde_20260503"
        v = metric(run, "val_dense")["summary"]
        t = metric(run, "test_dense")["summary"]
        vals[strategy] = [
            float(v["mean_rel_l2_speed"]),
            float(v["mean_rel_l2_p"]),
            float(t["mean_rel_l2_speed"]),
            float(t["mean_rel_l2_p"]),
        ]
    x = np.arange(len(labels))
    width = 0.34
    b1 = ax.bar(x - width / 2, vals["分区感知"], width=width, label="分区感知", color="#0f766e")
    b2 = ax.bar(x + width / 2, vals["均匀采样"], width=width, label="均匀采样", color="#f59e0b")
    add_bar_labels(ax, b1, fmt=pct_label)
    add_bar_labels(ax, b2, fmt=pct_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rel-L2")
    ax.set_title("5%观测预算下采样策略对比")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(vals["分区感知"] + vals["均匀采样"]) * 1.28)
    files.append(savefig(fig, sec / "strict_region_uniform_5pct_metrics.png"))

    for run, case, subdir in [
        ("contraction_strict_geometry_sparse5_stagepde_20260503", "C-val", "regionaware_val_maps"),
        ("contraction_strict_geometry_uniform5_stagepde_20260503", "C-val", "uniform_val_maps"),
        ("contraction_strict_geometry_sparse5_stagepde_20260503", "C-test-1", "regionaware_test1_maps"),
        ("contraction_strict_geometry_uniform5_stagepde_20260503", "C-test-1", "uniform_test1_maps"),
    ]:
        split = "val_dense" if case == "C-val" else "test_dense"
        files.extend(
            export_maps(
                PROJECT_ROOT / "results" / "pinn" / run / "evaluations" / f"predictions_{split}",
                [case],
                sec / subdir,
            )
        )
    return files


def plot_noise(sec: Path) -> list[str]:
    clean = metric("contraction_strict_geometry_sparse5_stagepde_20260503", "val_dense")["summary"]
    noise = metric("contraction_strict_geometry_sparse5noise3_stagepde_20260503", "val_dense")["summary"]
    labels = ["速度Rel-L2", "压力Rel-L2", "压降相对误差"]
    clean_vals = [clean["mean_rel_l2_speed"], clean["mean_rel_l2_p"], clean["mean_pressure_drop_rel_error"]]
    noise_vals = [noise["mean_rel_l2_speed"], noise["mean_rel_l2_p"], noise["mean_pressure_drop_rel_error"]]
    df = pd.DataFrame({"metric": labels, "clean": clean_vals, "noise3": noise_vals})
    df.to_csv(sec / "contraction_strict_noise3_summary.csv", index=False)
    files = []
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    b1 = ax.bar(x - width / 2, clean_vals, width=width, label="5%无噪声", color="#0f766e")
    b2 = ax.bar(x + width / 2, noise_vals, width=width, label="5%+3%噪声", color="#c2410c")
    add_bar_labels(ax, b1, fmt=pct_label)
    add_bar_labels(ax, b2, fmt=pct_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("指标值")
    ax.set_title("5%稀疏观测下3%噪声影响")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(clean_vals + noise_vals) * 1.28)
    files.append(savefig(fig, sec / "strict_noise3_summary.png"))
    files.extend(
        export_maps(
            PROJECT_ROOT / "results" / "pinn" / "contraction_strict_geometry_sparse5noise3_stagepde_20260503" / "evaluations" / "predictions_val_dense",
            ["C-val"],
            sec / "noise3_val_maps",
        )
    )
    return files


def plot_generalization_and_ablation(sec: Path) -> list[str]:
    files = []
    files.extend(
        export_maps(
            PROJECT_ROOT / "results" / "pinn" / "bend_strict_blunted_sparse5_stagepde_20260503" / "evaluations" / "predictions_test_dense",
            ["B-test-1__ip_blunted"],
            sec / "bend_target_test_maps",
        )
    )
    files.append(
        export_focused_speed_map(
            PROJECT_ROOT
            / "results"
            / "pinn"
            / "bend_strict_blunted_sparse5_stagepde_20260503"
            / "evaluations"
            / "predictions_test_dense"
            / "B-test-1__ip_blunted_predictions.csv",
            sec / "bend_target_test_maps" / "B-test-1__ip_blunted_speed_focused.png",
            title_prefix="弯曲流道B-test-1速度场重建",
        )
    )

    basic = metric("contraction_strict_basic_sparse5_stagepde_20260503", "test_dense")["summary"]
    geom = metric("contraction_strict_geometry_sparse5_stagepde_20260503", "test_dense")["summary"]
    labels = ["测试速度Rel-L2", "测试压力Rel-L2"]
    bvals = [basic["mean_rel_l2_speed"], basic["mean_rel_l2_p"]]
    gvals = [geom["mean_rel_l2_speed"], geom["mean_rel_l2_p"]]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(labels))
    width = 0.34
    b1 = ax.bar(x - width / 2, bvals, width=width, label="基础输入", color="#f59e0b")
    b2 = ax.bar(x + width / 2, gvals, width=width, label="几何增强", color="#0f766e")
    add_bar_labels(ax, b1, fmt=rel_label, log_scale=True)
    add_bar_labels(ax, b2, fmt=rel_label, log_scale=True)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rel-L2")
    ax.set_title("稀疏观测下几何增强编码消融")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False)
    ax.set_ylim(min(gvals + bvals) * 0.45, max(gvals + bvals) * 2.2)
    files.append(savefig(fig, sec / "strict_geometry_encoding_ablation.png"))
    return files


def plot_model_structure(sec: Path) -> list[str]:
    files = []
    dual = metric("contraction_strict_geometry_sparse5_stagepde_20260503", "val_dense")["summary"]
    dual_case = metric("contraction_strict_geometry_sparse5_stagepde_20260503", "val_dense")["case_metrics"][0]
    single = metric("contraction_single_mlp_geometry_sparse5_strict_20260503", "val_dense", kind="supervised")["summary"]
    single_case = metric("contraction_single_mlp_geometry_sparse5_strict_20260503", "val_dense", kind="supervised")["case_metrics"][0]

    err_labels = ["速度Rel-L2", "压力Rel-L2", "压降误差"]
    single_err = [
        single["mean_rel_l2_speed"],
        single["mean_rel_l2_p"],
        single["mean_pressure_drop_rel_error"],
    ]
    dual_err = [
        dual["mean_rel_l2_speed"],
        dual["mean_rel_l2_p"],
        dual["mean_pressure_drop_rel_error"],
    ]
    wall_labels = ["壁面u残余"]
    single_wall = [single_case["wall_max_abs_u_pred"]]
    dual_wall = [dual_case["wall_max_abs_u_pred"]]

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.0), gridspec_kw={"height_ratios": [2.2, 1.0]})
    width = 0.34
    x = np.arange(len(err_labels))
    b1 = axes[0].bar(x - width / 2, single_err, width=width, label="单网络MLP", color="#f59e0b")
    b2 = axes[0].bar(x + width / 2, dual_err, width=width, label="双模型PDE耦合", color="#0f766e")
    ymax = max(single_err + dual_err)
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, v + ymax * 0.035, f"{v * 100:.2f}%", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(err_labels)
    axes[0].set_ylabel("验证误差")
    axes[0].grid(axis="y", alpha=0.25, linestyle=":")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0].set_ylim(0, ymax * 1.30)

    wall_y = np.arange(2)
    wall_vals = [single_wall[0], dual_wall[0]]
    wall_names = ["单网络MLP", "双模型PDE耦合"]
    wall_colors = ["#f59e0b", "#0f766e"]
    b_wall = axes[1].barh(wall_y, wall_vals, height=0.48, color=wall_colors)
    wall_max = max(single_wall + dual_wall)
    for bar, value in zip(b_wall, wall_vals):
        label = "<0.01%" if 0 < value * 100 < 0.01 else f"{value * 100:.2f}%"
        axes[1].text(value + wall_max * 0.035, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=8)
    axes[1].set_yticks(wall_y)
    axes[1].set_yticklabels(wall_names)
    axes[1].set_xlabel("壁面u残余")
    axes[1].grid(axis="x", alpha=0.25, linestyle=":")
    axes[1].set_xlim(0, wall_max * 1.22)
    fig.suptitle("5%稀疏观测下单网络与双模型对比", y=1.01)
    files.append(savefig(fig, sec / "strict_single_vs_dual_comparison.png"))

    hist = pd.read_csv(PROJECT_ROOT / "results" / "pinn" / "contraction_strict_geometry_sparse5_stagepde_20260503" / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    for stage, group in hist.groupby("阶段", sort=False):
        axes[0].plot(group.index, group["验证_rel_l2_speed"], label=stage)
        axes[1].plot(group.index, group["验证_rel_l2_p"], label=stage)
    axes[0].set_title("速度Rel-L2")
    axes[1].set_title("压力Rel-L2")
    for ax in axes:
        ax.set_xlabel("记录序号")
        ax.grid(alpha=0.25, linestyle=":")
    axes[0].set_ylabel("验证Rel-L2")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("双模型训练过程", y=1.02)
    files.append(savefig(fig, sec / "strict_dual_model_training_curve.png"))
    return files


def build_manifest(section_files: dict[str, list[str]]) -> None:
    payload = {
        "root": str(OUT),
        "sections": {
            section: [str((OUT / section / name).relative_to(OUT)) for name in files]
            for section, files in section_files.items()
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 第五章严格稀疏实验图片",
        "",
        "本目录基于 strict sparse 训练结果生成。训练时输入/输出标准化只由观测点拟合，稀疏训练关闭压降、出口压力和入口流量等 dense 真值损失。",
        "",
    ]
    for section, files in section_files.items():
        lines.append(f"## {section}")
        for name in files:
            lines.append(f"- `{section}/{name}`")
        lines.append("")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    配置中文绘图(prefer_serif=False)
    reset_dir(OUT)
    section_files: dict[str, list[str]] = {}
    for name, func in [
        ("5_3_sparse_rate", plot_bend_sparse_rate),
        ("5_4_region_vs_uniform", plot_region_uniform),
        ("5_5_noise_robustness", plot_noise),
        ("5_6_generalization_and_ablation", plot_generalization_and_ablation),
        ("5_8_model_structure", plot_model_structure),
    ]:
        sec = OUT / name
        sec.mkdir(parents=True, exist_ok=True)
        section_files[name] = func(sec)
    build_manifest(section_files)
    print(OUT)


if __name__ == "__main__":
    main()
