#!/usr/bin/env python3
"""几何增强编码消融实验。

对比两组输入特征：
1. 仅坐标 + 全局参数（basic）
2. 坐标 + 几何增强编码（geometry，沿用主线并去掉 inlet_profile_star）

统一在 soft-wall 设置下做公平对照，分别统计：
- 未见几何上的速度泛化误差（dense test mean_rel_l2_speed）
- 未见几何上的压力误差（dense test mean_rel_l2_p）
- 稀疏观测监督下的全场重建误差（sparse-trained model -> dense test mean_rel_l2_speed）
"""

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


DOCS_DIR = REPO_ROOT / "docs" / "ablations"
ABALATION_TAG = "geometry_encoding_ablation_20260420"
OUTPUT_DIR = DOCS_DIR / ABALATION_TAG

TRAIN_CASES = "C-base,C-train-1,C-train-2,C-train-3,C-train-4,C-train-5"
VAL_CASES = "C-val"
TEST_CASES = "C-test-1,C-test-2"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

COMMON_TRAIN_ARGS = [
    "--family",
    "contraction_2d",
    "--train-cases",
    TRAIN_CASES,
    "--val-cases",
    VAL_CASES,
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
    "80",
    "--velocity-lr",
    "6e-4",
    "--pressure-lr",
    "6e-4",
    "--coupling-velocity-lr",
    "1e-4",
    "--coupling-pressure-lr",
    "1e-4",
    "--wall-weight",
    "0.0",
    "--inlet-flux-weight",
    "0.5",
    "--continuity-weight",
    "0.1",
    "--velocity-stage-continuity-weight",
    "0.3",
    "--velocity-stage-momentum-weight",
    "0.0",
    "--outlet-pressure-weight",
    "1e-4",
    "--pressure-drop-weight",
    "1.0",
    "--pressure-stage-momentum-weight",
    "0.5",
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

RUN_SPECS = {
    "basic_dense": {
        "label": "仅坐标+全局参数",
        "run_name": "contraction_independent_basic_dense_softwall_ablation_v1_20260420",
        "feature_mode": "basic",
        "drop_features": "",
        "train_velocity_source": "dense",
        "val_velocity_source": "dense",
        "train_pressure_source": "dense",
        "val_pressure_source": "dense",
    },
    "geometry_dense": {
        "label": "坐标+几何增强编码",
        "run_name": "contraction_independent_geometry_dense_softwall_ablation_v1_20260420",
        "feature_mode": "geometry",
        "drop_features": "inlet_profile_star",
        "train_velocity_source": "dense",
        "val_velocity_source": "dense",
        "train_pressure_source": "dense",
        "val_pressure_source": "dense",
    },
    "basic_sparse": {
        "label": "仅坐标+全局参数",
        "run_name": "contraction_independent_basic_sparse5clean_softwall_ablation_v1_20260420",
        "feature_mode": "basic",
        "drop_features": "",
        "train_velocity_source": "obs_sparse_5pct",
        "val_velocity_source": "obs_sparse_5pct",
        "train_pressure_source": "obs_sparse_5pct",
        "val_pressure_source": "obs_sparse_5pct",
    },
    "geometry_sparse": {
        "label": "坐标+几何增强编码",
        "run_name": "contraction_independent_geometry_sparse5clean_softwall_ablation_v1_20260420",
        "feature_mode": "geometry",
        "drop_features": "inlet_profile_star",
        "train_velocity_source": "obs_sparse_5pct",
        "val_velocity_source": "obs_sparse_5pct",
        "train_pressure_source": "obs_sparse_5pct",
        "val_pressure_source": "obs_sparse_5pct",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run geometry-encoding ablation with low CPU pressure")
    parser.add_argument("--skip-train", action="store_true", help="仅复用已有 run 与评估结果")
    parser.add_argument("--force-eval", action="store_true", help="重新执行评估")
    parser.add_argument("--max-retries", type=int, default=1, help="命令级重试次数")
    return parser


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


def ensure_training(run_key: str, spec: dict[str, str], args: argparse.Namespace) -> None:
    run_dir = PROJECT_ROOT / "results" / "pinn" / spec["run_name"]
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() or args.skip_train:
        return
    command = [
        "python3",
        str(PROJECT_ROOT / "scripts" / "train_velocity_pressure_independent.py"),
        "--run-name",
        spec["run_name"],
        "--feature-mode",
        spec["feature_mode"],
        "--train-velocity-source",
        spec["train_velocity_source"],
        "--val-velocity-source",
        spec["val_velocity_source"],
        "--train-pressure-source",
        spec["train_pressure_source"],
        "--val-pressure-source",
        spec["val_pressure_source"],
        *COMMON_TRAIN_ARGS,
    ]
    if spec["drop_features"]:
        command.extend(["--drop-features", spec["drop_features"]])
    log_path = PROJECT_ROOT / "results" / "pinn" / f"{spec['run_name']}.log"
    run_command(command, log_path, args.max_retries)


def ensure_evaluation(run_key: str, spec: dict[str, str], split_name: str, eval_source: str, args: argparse.Namespace) -> dict:
    run_dir = PROJECT_ROOT / "results" / "pinn" / spec["run_name"]
    eval_dir = run_dir / "evaluations"
    metrics_path = eval_dir / f"metrics_{split_name}.json"
    if not metrics_path.exists() or args.force_eval:
        command = [
            "python3",
            str(PROJECT_ROOT / "scripts" / "evaluate_velocity_pressure_independent.py"),
            "--family",
            "contraction_2d",
            "--run-name",
            spec["run_name"],
            "--eval-cases",
            TEST_CASES,
            "--split-name",
            split_name,
            "--eval-source",
            eval_source,
            "--max-retries",
            "1",
        ]
        log_path = PROJECT_ROOT / "results" / "pinn" / f"{spec['run_name']}_eval_{split_name}.log"
        if args.force_eval and metrics_path.exists():
            metrics_path.unlink()
        run_command(command, log_path, args.max_retries)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def build_summary(dense_basic: dict, dense_geometry: dict, sparse_basic: dict, sparse_geometry: dict) -> pd.DataFrame:
    records = [
        {
            "metric_key": "geometry_generalization_error",
            "metric_label": "几何泛化误差",
            "definition": "dense 监督训练后，在未见几何 C-test-1/C-test-2 上的速度 mean Rel-L2",
            "basic": float(dense_basic["summary"]["mean_rel_l2_speed"]),
            "geometry": float(dense_geometry["summary"]["mean_rel_l2_speed"]),
        },
        {
            "metric_key": "pressure_error",
            "metric_label": "压力误差",
            "definition": "dense 监督训练后，在未见几何 C-test-1/C-test-2 上的压力 mean Rel-L2",
            "basic": float(dense_basic["summary"]["mean_rel_l2_p"]),
            "geometry": float(dense_geometry["summary"]["mean_rel_l2_p"]),
        },
        {
            "metric_key": "sparse_reconstruction_error",
            "metric_label": "稀疏重建误差",
            "definition": "5% 稀疏观测监督训练后，在 dense test 上恢复全场的速度 mean Rel-L2",
            "basic": float(sparse_basic["summary"]["mean_rel_l2_speed"]),
            "geometry": float(sparse_geometry["summary"]["mean_rel_l2_speed"]),
        },
    ]
    frame = pd.DataFrame(records)
    frame["absolute_gain"] = frame["basic"] - frame["geometry"]
    frame["relative_gain_pct"] = np.where(frame["basic"] > 1.0e-12, frame["absolute_gain"] / frame["basic"] * 100.0, 0.0)
    return frame


def write_markdown(summary: pd.DataFrame, payload: dict, output_path: Path) -> None:
    lines = [
        "# 几何增强编码消融实验",
        "",
        "## 实验设置",
        "",
        "- 数据族：收缩流道 `contraction_2d`",
        "- 训练工况：`C-base, C-train-1..5`；验证工况：`C-val`；测试工况：`C-test-1, C-test-2`",
        "- 对比项 A：仅坐标 + 全局参数（`basic`）",
        "- 对比项 B：坐标 + 几何增强编码（`geometry`，与主线一致，去掉 `inlet_profile_star`）",
        "- 公平性约束：四组消融均使用 `soft wall`，其余主要训练超参与主线保持一致",
        "- 稀疏重建口径：使用 `obs_sparse_5pct` 训练，在 dense test 上评估全场恢复误差",
        "",
        "## 指标汇总",
        "",
        "| 指标 | 仅坐标+全局参数 | 坐标+几何增强编码 | 绝对改善 | 相对改善 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['metric_label']} | {row['basic']:.4f} | {row['geometry']:.4f} | {row['absolute_gain']:.4f} | {row['relative_gain_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- 几何泛化误差越低，说明模型面对未见收缩比/长度比几何时，速度场外推更稳定。",
            "- 压力误差越低，说明几何编码不仅改善速度场，也改善压力恢复。",
            "- 稀疏重建误差越低，说明在只给少量观测点时，几何增强编码更有利于恢复全场。",
            "",
            "## 对应运行",
            "",
        ]
    )
    for key, info in payload["runs"].items():
        lines.append(f"- `{key}`: `{info['run_name']}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    配置中文绘图(prefer_serif=True)
    label_map = {
        "几何泛化误差": "未见几何\n速度误差",
        "压力误差": "未见几何\n压力误差",
        "稀疏重建误差": "5%稀疏监督\n重建误差",
    }
    labels = [label_map.get(item, item) for item in summary["metric_label"].tolist()]
    basic_vals = summary["basic"].to_numpy(dtype=float)
    geometry_vals = summary["geometry"].to_numpy(dtype=float)
    gains = summary["relative_gain_pct"].to_numpy(dtype=float)

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    basic_color = "#B7ACA0"
    geometry_color = "#4C78A8"

    bars_basic = ax.bar(x - width / 2, basic_vals, width=width, color=basic_color, edgecolor="#8D847B", linewidth=0.8, label="仅坐标+全局参数")
    bars_geometry = ax.bar(x + width / 2, geometry_vals, width=width, color=geometry_color, edgecolor="#365A82", linewidth=0.8, label="坐标+几何增强编码")

    ymax = float(max(np.max(basic_vals), np.max(geometry_vals)) * 1.24)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("平均相对 L2 误差")
    ax.set_title("收缩流道几何增强编码消融实验", pad=16)
    ax.grid(axis="y", alpha=0.22, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)

    for bars in (bars_basic, bars_geometry):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.018,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#243142",
            )

    for idx, gain in enumerate(gains):
        best_height = max(basic_vals[idx], geometry_vals[idx])
        ax.text(
            x[idx],
            best_height + ymax * 0.06,
            f"误差下降 {gain:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#365A82",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
        )

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dense_payloads: dict[str, dict] = {}
    sparse_payloads: dict[str, dict] = {}

    for key, spec in RUN_SPECS.items():
        ensure_training(key, spec, args)
        split_name = "ablation_test_dense" if "dense" in key else "ablation_test_sparse"
        payload = ensure_evaluation(key, spec, split_name=split_name, eval_source="dense", args=args)
        if "dense" in key:
            dense_payloads[key] = payload
        else:
            sparse_payloads[key] = payload

    summary = build_summary(
        dense_basic=dense_payloads["basic_dense"],
        dense_geometry=dense_payloads["geometry_dense"],
        sparse_basic=sparse_payloads["basic_sparse"],
        sparse_geometry=sparse_payloads["geometry_sparse"],
    )

    csv_path = OUTPUT_DIR / "geometry_encoding_ablation_summary.csv"
    json_path = OUTPUT_DIR / "geometry_encoding_ablation_summary.json"
    md_path = OUTPUT_DIR / "geometry_encoding_ablation_summary.md"
    png_path = OUTPUT_DIR / "geometry_encoding_ablation_summary.png"

    summary.to_csv(csv_path, index=False)
    payload = {
        "experiment": "geometry_encoding_ablation",
        "family": "contraction_2d",
        "test_cases": TEST_CASES.split(","),
        "runs": {key: spec for key, spec in RUN_SPECS.items()},
        "dense_runs": dense_payloads,
        "sparse_runs": sparse_payloads,
        "summary": summary.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, payload, md_path)
    plot_summary(summary, png_path)

    print(f"[done] csv={csv_path}")
    print(f"[done] json={json_path}")
    print(f"[done] md={md_path}")
    print(f"[done] png={png_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
