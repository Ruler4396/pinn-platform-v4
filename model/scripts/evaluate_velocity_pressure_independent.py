#!/usr/bin/env python3
"""对独立速度模型 + 独立压力模型 + 控制方程耦合 checkpoint 做低冲击离线评估。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def 动态加载模块(script_name: str, module_name: str):
    module_path = PROJECT_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 解析列表(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def 汇总(case_metrics: list[dict]) -> dict[str, float | int]:
    frame = pd.DataFrame(case_metrics)
    if frame.empty:
        return {"num_cases": 0}
    summary: dict[str, float | int] = {"num_cases": int(len(frame))}
    for column in [
        "rel_l2_u",
        "rel_l2_v",
        "rel_l2_p",
        "rel_l2_speed",
        "mae_u",
        "mae_v",
        "mae_p",
        "pressure_drop_rel_error",
        "wall_max_abs_u_pred",
        "wall_max_abs_v_pred",
    ]:
        if column in frame:
            summary[f"mean_{column}"] = float(frame[column].mean())
            summary[f"max_{column}"] = float(frame[column].max())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate independent velocity/pressure checkpoint on arbitrary case sets")
    parser.add_argument("--family", required=True, choices=["contraction_2d", "bend_2d"])
    parser.add_argument("--run-name", required=True, help="training run directory under results/pinn/")
    parser.add_argument("--eval-cases", required=True, help="comma-separated case ids to evaluate")
    parser.add_argument("--eval-source", default="dense", help="evaluation source, default dense")
    parser.add_argument("--split-name", default="test", help="logical split name, e.g. val/test")
    parser.add_argument("--output-subdir", default="evaluations", help="subdirectory under run dir")
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--max-retries", type=int, default=1)
    return parser


def evaluate_once(args: argparse.Namespace) -> dict:
    mod = 动态加载模块("train_velocity_pressure_independent.py", "train_velocity_pressure_independent_runtime")
    sup = 动态加载模块("train_supervised.py", "train_supervised_runtime_for_independent_eval")

    run_dir = PROJECT_ROOT / "results" / "pinn" / args.run_name
    ckpt_path = run_dir / "best.ckpt"
    config_path = run_dir / "config.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    eval_cases = 解析列表(args.eval_cases)
    spec = sup.get_family_spec(args.family)
    feature_cols = config.get("feature_cols") or list(sup.resolve_feature_cols(spec, config.get("feature_mode", spec.default_feature_mode)))

    input_scaler_cfg = config["输入标准化"]
    vel_scaler_cfg = config["速度标准化"]
    p_scaler_cfg = config["压力标准化"]
    input_scaler = sup.StandardScaler(mean=np.array(input_scaler_cfg["x_mean"], dtype=np.float32), std=np.array(input_scaler_cfg["x_std"], dtype=np.float32))
    vel_scaler = mod.输出标准化器(mean=np.array(vel_scaler_cfg["mean"], dtype=np.float32), std=np.array(vel_scaler_cfg["std"], dtype=np.float32))
    p_scaler = mod.输出标准化器(mean=np.array(p_scaler_cfg["mean"], dtype=np.float32), std=np.array(p_scaler_cfg["std"], dtype=np.float32))

    eval_split, _ = mod.构建数据切分(spec, eval_cases, feature_cols=feature_cols, source=args.eval_source, input_scaler=input_scaler)
    device = torch.device(args.device)

    velocity_hidden_layers = config.get("velocity_hidden_layers", [128, 128, 128])
    pressure_hidden_layers = config.get("pressure_hidden_layers", [128, 128, 128])
    activation = config.get("activation", "silu")
    velocity_model = mod.多层感知机(len(feature_cols), 2, velocity_hidden_layers, activation=activation).to(device)
    pressure_model = mod.多层感知机(len(feature_cols), 1, pressure_hidden_layers, activation=activation).to(device)
    velocity_model.load_state_dict(checkpoint["速度模型参数"])
    pressure_model.load_state_dict(checkpoint["压力模型参数"])

    velocity_wall_cfg = config.get("速度壁面约束", {"mode": "soft", "hard_wall_sharpness": 12.0})
    constraint_info = mod.构建壁面硬约束信息(feature_cols, velocity_wall_cfg.get("mode", "soft"), float(velocity_wall_cfg.get("hard_wall_sharpness", 12.0)))

    pred_raw, metrics = mod.评估联合场(velocity_model, pressure_model, eval_split, vel_scaler, p_scaler, device, constraint_info)
    case_metrics = [
        asdict(sup.compute_case_metrics(case_id, eval_split, pred_raw))
        for case_id in sorted(set(eval_split.case_ids.tolist()))
    ]
    summary = 汇总(case_metrics)

    eval_dir = run_dir / args.output_subdir
    eval_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = eval_dir / f"predictions_{args.split_name}"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for case_id in sorted(set(eval_split.case_ids.tolist())):
        mask = eval_split.case_ids == case_id
        frame = pd.DataFrame(
            {
                "case_id": eval_split.case_ids[mask],
                "x_star": eval_split.x_star[mask],
                "y_star": eval_split.y_star[mask],
                "boundary_type": eval_split.boundary_type[mask],
                "u_true": eval_split.targets_raw[mask, 0],
                "v_true": eval_split.targets_raw[mask, 1],
                "p_true": eval_split.targets_raw[mask, 2],
                "u_pred": pred_raw[mask, 0],
                "v_pred": pred_raw[mask, 1],
                "p_pred": pred_raw[mask, 2],
            }
        )
        frame.to_csv(predictions_dir / f"{case_id}_predictions.csv", index=False)

    payload = {
        "family": args.family,
        "run_name": args.run_name,
        "split_name": args.split_name,
        "eval_cases": eval_cases,
        "eval_source": args.eval_source,
        "feature_mode": config.get("feature_mode"),
        "feature_cols": feature_cols,
        "summary": summary,
        "global_metrics": metrics,
        "case_metrics": case_metrics,
        "source_run_config": str(config_path),
    }
    metrics_path = eval_dir / f"metrics_{args.split_name}.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "family": args.family,
        "run_name": args.run_name,
        "split_name": args.split_name,
        "eval_cases": eval_cases,
        "eval_source": args.eval_source,
        "device": args.device,
        "max_retries": args.max_retries,
        "metrics_file": str(metrics_path),
        "predictions_dir": str(predictions_dir),
    }
    (eval_dir / f"manifest_{args.split_name}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metrics_path": str(metrics_path),
        "predictions_dir": str(predictions_dir),
        "summary": summary,
        "global_metrics": metrics,
    }


def main() -> None:
    args = build_parser().parse_args()
    attempts = max(1, int(args.max_retries))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = evaluate_once(args)
            print(f"[done] metrics={result['metrics_path']}")
            print(f"predictions_dir={result['predictions_dir']}")
            print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
            print(json.dumps(result["global_metrics"], ensure_ascii=False, indent=2))
            return
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"[attempt {attempt}/{attempts}] failed: {exc}", file=sys.stderr)
            if attempt >= attempts:
                raise
    if last_error is not None:
        raise last_error


if __name__ == "__main__":
    main()
