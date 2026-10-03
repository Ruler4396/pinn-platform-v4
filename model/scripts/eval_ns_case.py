#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 NS 档工况跑双模型评估器：先注册工况名，再原样交给 `evaluate_velocity_pressure_independent.py`。

为什么要单独一枚：评估器是**另一个入口**，训练器里那套"运行期把 `<base>_ns_re<lvl>` 挂进收缩族工况表"
的做法不会跟着过去——直接给它 NS 工况名会 `KeyError: Unknown contraction case`。
注册函数不复制第二份：本件 import 训练器那枚模块、调它的 `注册NS工况`，
所以"挂表"这件事全仓只有一处实现（两处实现必然漂移，这是本线自己踩过的）。

用法：
  python3 eval_ns_case.py --run-name ns3p_10_ns_s42 --eval-cases C-val_ns_re50 --split-name ext50
其余参数透传给评估器。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATOR = "evaluate_velocity_pressure_independent.py"


def load(name: str):
    path = SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(path.stem, module)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ns = load("train_velocity_pressure_independent_ns")
    argv = sys.argv[1:]
    ids: list[str] = []
    for i, a in enumerate(argv):
        for flag in ("--eval-cases", "--train-cases", "--val-cases"):
            if a == flag and i + 1 < len(argv):
                ids += [s.strip() for s in argv[i + 1].split(",") if s.strip()]
            elif a.startswith(flag + "="):
                ids += [s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()]
    registered = ns.注册NS工况(ids)
    for cid in registered:
        print(f"[NS-case] {cid} 已挂表（评估入口，复用训练器里那一份实现）", flush=True)
    ev = load(EVALUATOR)
    sys.argv = [str(SCRIPT_DIR / EVALUATOR)] + argv
    return ev.main()


if __name__ == "__main__":
    raise SystemExit(main())
