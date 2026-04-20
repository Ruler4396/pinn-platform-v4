#!/usr/bin/env python3
"""弯曲流道不同观测采样率对照实验与收敛图导出。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting import 配置中文绘图


DENSE_REFERENCE_RUN = "bend_independent_blunted_geometry_notemplate_medium_v1_20260401"
OUTPUT_DIR = REPO_ROOT / "docs" / "ablations" / "bend_zero_supervision_20260420"
RUN_SPECS = [
    ("0%", "obs_sparse_0pct", "bend_independent_blunted_sparse0clean_softwall_v1_20260420"),
    ("1%", "obs_sparse_1pct", "bend_independent_blunted_sparse1clean_softwall_v1_20260420"),
    ("5%", "obs_sparse_5pct", "bend_independent_blunted_sparse5clean_softwall_v1_20260420"),
    ("10%", "obs_sparse_10pct", "bend_independent_blunted_sparse10clean_softwall_v1_20260420"),
    ("15%", "obs_sparse_15pct", "bend_independent_blunted_sparse15clean_softwall_v1_20260420"),
]

TRAIN_CASES = "B-base__ip_blunted,B-train-1__ip_blunted,B-train-2__ip_blunted,B-train-3__ip_blunted"
VAL_CASES = "B-val__ip_blunted"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

BASE_TRAIN_ARGS = [
    "--family",
    "bend_2d",
    "--train-cases",
    TRAIN_CASES,
    "--val-cases",
    VAL_CASES,
    "--feature-mode",
    "geometry",
    "--drop-features",
    "inlet_profile_star",
    "--device",
    "cpu",
    "--velocity-hidden-layers",
    "128,128,128",
    "--pressure-hidden-layers",
    "128,128,128",
    "--activation",
    "silu",
    "--velocity-epochs",
    "200",
    "--pressure-epochs",
    "200",
    "--coupling-epochs",
    "40",
    "--velocity-lr",
    "8e-4",
    "--pressure-lr",
    "8e-4",
    "--coupling-velocity-lr",
    "2e-4",
    "--coupling-pressure-lr",
    "2e-4",
    "--wall-weight",
    "1.0",
    "--inlet-flux-weight",
    "0.5",
    "--continuity-weight",
    "0.1",
    "--velocity-stage-continuity-weight",
    "0.1",
    "--velocity-stage-momentum-weight",
    "0.0",
    "--outlet-pressure-weight",
    "1e-4",
    "--pressure-drop-weight",
    "1.0",
    "--pressure-stage-momentum-weight",
    "0.0",
    "--velocity-wall-mode",
    "soft",
    "--coupling-momentum-weight",
    "10.0",
    "--coupling-continuity-weight",
    "0.1",
    "--coupling-velocity-supervision-weight",
    "1.0",
    "--coupling-pressure-supervision-weight",
    "1.0",
    "--max-physics-points",
    "512",
    "--print-every",
    "20",
    "--max-retries",
    "1",
]

PALETTE = {
    "dense": "#4C78A8",
    "0%": "#D64550",
    "1%": "#7B61FF",
    "5%": "#2A9D8F",
    "10%": "#F4A261",
    "15%": "#7A8899",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bend sparse-rate controls and export convergence figure")
    parser.add_argument("--skip-train", action="store_true", help="若 run 已存在，仅重绘图片")
    parser.add_argument("--max-retries", type=int, default=1)
    return parser


def build_train_args(run_name: str, source_name: str) -> list[str]:
    return [
        "--run-name",
        run_name,
        "--train-velocity-source",
        source_name,
        "--val-velocity-source",
        source_name,
        "--train-pressure-source",
        source_name,
        "--val-pressure-source",
        source_name,
        *BASE_TRAIN_ARGS,
    ]


def run_command(command: list[str], log_path: Path, max_retries: int) -> None:
    env = os.environ.copy()
    env.update(THREAD_ENV)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, max(1, max_retries) + 1):
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n$ {' '.join(command)}\n")
            handle.flush()
            try:
                subprocess.run(
                    ["nice", "-n", "10", *command],
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                return
            except subprocess.CalledProcessError as exc:
                last_error = exc
                handle.write(f"[attempt {attempt}] failed with exit code {exc.returncode}\n")
                handle.flush()
    if last_error is not None:
        raise last_error


def ensure_run(run_name: str, source_name: str, args: argparse.Namespace) -> Path:
    run_dir = PROJECT_ROOT / "results" / "pinn" / run_name
    if (run_dir / "history.csv").exists() or args.skip_train:
        return run_dir
    log_path = PROJECT_ROOT / "results" / "pinn" / f"{run_name}.log"
    command = [
        "python3",
        str(PROJECT_ROOT / "scripts" / "train_velocity_pressure_independent.py"),
        *build_train_args(run_name, source_name),
    ]
    run_command(command, log_path, args.max_retries)
    return run_dir


def plot_curve(output_path: Path, histories: dict[str, pd.DataFrame], dense_history: pd.DataFrame) -> None:
    配置中文绘图(prefer_serif=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_dense = np.arange(1, len(dense_history) + 1)
    ax.plot(
        x_dense,
        dense_history["验证_rel_l2_speed"].to_numpy(dtype=float),
        linewidth=2.2,
        color=PALETTE["dense"],
        label="dense 基线",
    )

    for label, _, _ in RUN_SPECS:
        history = histories[label]
        x = np.arange(1, len(history) + 1)
        legend_label = f"{label}观测"
        if label == "0%":
            legend_label += "（无点位监督）"
        ax.plot(
            x,
            history["验证_rel_l2_speed"].to_numpy(dtype=float),
            linewidth=2.2 if label == "0%" else 1.9,
            color=PALETTE[label],
            label=legend_label,
        )

    ax.set_title("弯曲流道不同观测采样率下的验证集速度误差收敛曲线")
    ax.set_xlabel("迭代轮次")
    ax.set_ylabel("验证集速度相对二范数误差")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    final_rows = [("dense", float(dense_history["验证_rel_l2_speed"].iloc[-1]))]
    for label, _, _ in RUN_SPECS:
        final_rows.append((label, float(histories[label]["验证_rel_l2_speed"].iloc[-1])))
    summary_lines = ["末轮误差："]
    for label, value in final_rows:
        summary_lines.append(f"{label:<5} {value:.3f}")
    ax.text(
        0.035,
        0.34,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#22324A",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#D5DDE8",
            "linewidth": 0.8,
            "alpha": 0.92,
        },
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_summary(output_dir: Path, histories: dict[str, pd.DataFrame], dense_history: pd.DataFrame) -> None:
    final_dense = float(dense_history["验证_rel_l2_speed"].iloc[-1])
    sparse_metrics = {}
    for label, _, run_name in RUN_SPECS:
        history = histories[label]
        final_value = float(history["验证_rel_l2_speed"].iloc[-1])
        sparse_metrics[label] = {
            "run_name": run_name,
            "final_val_rel_l2_speed": final_value,
            "gap_vs_dense": final_value - final_dense,
            "ratio_vs_dense": final_value / max(final_dense, 1.0e-12),
        }

    payload = {
        "experiment": "bend_sparse_rate_curve_with_zero",
        "note": "0% 观测表示无点位监督，但仍保留物理方程、入口流量、出口压力/压降等约束；1%/5%/10%/15% 为对应比例的内部观测监督。",
        "dense_reference_run": DENSE_REFERENCE_RUN,
        "dense_final_val_rel_l2_speed": final_dense,
        "sparse_sampling_results": sparse_metrics,
    }
    (output_dir / "bend_zero_supervision_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 弯曲流道不同观测采样率对照实验",
        "",
        "- `0%观测` 表示没有任何内部观测点监督，但训练仍保留物理方程、入口流量、出口压力/压降等约束。",
        f"- dense 基线最终验证集速度 Rel-L2：`{final_dense:.4f}`",
        "",
        "| 采样率 | 最终验证集速度 Rel-L2 | 相对 dense 倍率 |",
        "| --- | ---: | ---: |",
    ]
    for label, _, _ in RUN_SPECS:
        item = sparse_metrics[label]
        lines.append(f"| {label} | `{item['final_val_rel_l2_speed']:.4f}` | `{item['ratio_vs_dense']:.2f}x` |")
    lines.append("")
    (output_dir / "bend_zero_supervision_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    histories: dict[str, pd.DataFrame] = {}
    run_dirs: dict[str, Path] = {}
    for label, source_name, run_name in RUN_SPECS:
        run_dir = ensure_run(run_name, source_name, args)
        run_dirs[label] = run_dir
        histories[label] = pd.read_csv(run_dir / "history.csv")

    dense_history = pd.read_csv(PROJECT_ROOT / "results" / "pinn" / DENSE_REFERENCE_RUN / "history.csv")

    plot_path = OUTPUT_DIR / "bend_sparse_rate_convergence_with_zero.png"
    plot_curve(plot_path, histories, dense_history)
    write_summary(OUTPUT_DIR, histories, dense_history)

    print(f"[done] plot={plot_path}")
    print(f"[done] final_dense={float(dense_history['验证_rel_l2_speed'].iloc[-1]):.6f}")
    for label, _, _ in RUN_SPECS:
        print(f"[done] final_{label}={float(histories[label]['验证_rel_l2_speed'].iloc[-1]):.6f}")
        print(f"[done] run_{label}={run_dirs[label]}")


if __name__ == "__main__":
    main()
