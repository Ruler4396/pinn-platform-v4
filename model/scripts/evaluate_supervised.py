#!/usr/bin/env python3
"""Evaluate a saved supervised baseline checkpoint on arbitrary case sets.

This script is designed for low-impact, thesis-traceable evaluation runs:
- CPU only
- explicit max_retries guard
- can evaluate validation or test splits without retraining
"""

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


def load_train_supervised_module():
    module_path = PROJECT_ROOT / "scripts" / "train_supervised.py"
    spec = importlib.util.spec_from_file_location("train_supervised_runtime", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load train_supervised module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def summarize_metrics(case_metrics: list[dict]) -> dict[str, float | int]:
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
    parser = argparse.ArgumentParser(description="Evaluate supervised checkpoint on arbitrary case sets")
    parser.add_argument("--family", required=True, choices=["contraction_2d", "bend_2d"])
    parser.add_argument("--run-name", required=True, help="training run directory under results/supervised/")
    parser.add_argument("--eval-cases", required=True, help="comma-separated case ids to evaluate")
    parser.add_argument(
        "--split-name",
        default="test",
        help="logical split name used in output files, e.g. val, test, test_interp",
    )
    parser.add_argument(
        "--output-subdir",
        default="evaluations",
        help="subdirectory under run dir for evaluation artifacts",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu"], help="cpu only by default to keep impact low")
    parser.add_argument("--max-retries", type=int, default=1, help="hard cap to avoid infinite retries")
    return parser


def evaluate_once(args: argparse.Namespace) -> dict:
    mod = load_train_supervised_module()
    run_dir = PROJECT_ROOT / "results" / "supervised" / args.run_name
    ckpt_path = run_dir / "best.ckpt"
    config_path = run_dir / "config.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    train_config = json.loads(config_path.read_text(encoding="utf-8"))
    eval_cases = parse_csv_list(args.eval_cases)
    family_spec = mod.get_family_spec(args.family)
    checkpoint = torch.load(ckpt_path, map_location=args.device)

    hidden_layers = train_config.get("hidden_layers", [128, 128, 128, 128])
    activation = train_config.get("activation", "silu")
    pressure_weight = float(train_config.get("pressure_weight", 1.0))
    feature_mode = train_config.get("feature_mode", family_spec.default_feature_mode)
    feature_cols = train_config.get("feature_cols") or list(mod.resolve_feature_cols(family_spec, feature_mode))

    input_scaler = mod.StandardScaler(mean=np.array(checkpoint["input_mean"]), std=np.array(checkpoint["input_std"]))
    output_scaler = mod.StandardScaler(mean=np.array(checkpoint["output_mean"]), std=np.array(checkpoint["output_std"]))
    eval_split, _, _ = mod.build_split(
        family_spec,
        eval_cases,
        feature_cols=feature_cols,
        input_scaler=input_scaler,
        output_scaler=output_scaler,
    )

    device = torch.device(args.device)
    eval_x, eval_y = mod.tensorize(eval_split, device)
    model = mod.SupervisedMLP(
        in_dim=eval_x.shape[1],
        out_dim=eval_y.shape[1],
        hidden_layers=hidden_layers,
        activation=activation,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pred = model(eval_x)
        losses = mod.compute_component_losses(pred, eval_y, pressure_weight=pressure_weight)
    pred_raw = output_scaler.inverse_transform(pred.cpu().numpy())

    case_metrics = [
        asdict(mod.compute_case_metrics(case_id, eval_split, pred_raw))
        for case_id in sorted(set(eval_split.case_ids.tolist()))
    ]
    summary = summarize_metrics(case_metrics)

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

    metrics_payload = {
        "family": args.family,
        "run_name": args.run_name,
        "split_name": args.split_name,
        "eval_cases": eval_cases,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "feature_mode": feature_mode,
        "feature_cols": feature_cols,
        "eval_total_loss": float(losses["total"].item()),
        "eval_loss_u": float(losses["loss_u"].item()),
        "eval_loss_v": float(losses["loss_v"].item()),
        "eval_loss_p": float(losses["loss_p"].item()),
        "summary": summary,
        "case_metrics": case_metrics,
    }
    metrics_path = eval_dir / f"metrics_{args.split_name}.json"
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "family": args.family,
        "run_name": args.run_name,
        "split_name": args.split_name,
        "eval_cases": eval_cases,
        "device": args.device,
        "max_retries": args.max_retries,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "feature_mode": feature_mode,
        "feature_cols": feature_cols,
        "source_run_config": str(config_path),
        "metrics_file": str(metrics_path),
        "predictions_dir": str(predictions_dir),
    }
    (eval_dir / f"manifest_{args.split_name}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metrics_path": str(metrics_path),
        "predictions_dir": str(predictions_dir),
        "summary": summary,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
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
            print(f"checkpoint_epoch={result['checkpoint_epoch']}")
            print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
            return
        except Exception as exc:  # pragma: no cover - runtime guard path
            last_error = exc
            print(f"[attempt {attempt}/{attempts}] failed: {exc}", file=sys.stderr)
            if attempt >= attempts:
                raise
    if last_error is not None:
        raise last_error


if __name__ == "__main__":
    main()
