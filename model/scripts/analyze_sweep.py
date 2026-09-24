#!/usr/bin/env python3
"""T5/T6 扫描结果统计（修订方案 §5 的执行体）。

三件事，每件都拒绝假装成功：
1. 每格 mean±std；**同时**给 mean-of-cases 与 pooled 两套口径并标明哪套进表
   （论文现用的是 mean-of-cases，仓库 JSON 顶层字段是 pooled，两者在 test 上差到 1.2 倍）。
2. 采样策略配对检验：配对单位 = (obs_seed, train_seed)。Wilcoxon 符号秩（双侧、精确）
   + 配对 t，并打印"本组最小可达双侧 p"—— n=8 ⇒ 2/2^8 = 0.0078；n=5 ⇒ 0.0625 > 0.05，
   所以主矩阵只做 mean±std，显著性只由 T6 承担。
3. 自动判"不可判"：两臂差小于组内离散度时输出该字样，供正文直接引用。

scipy 缺失 ⇒ 明确报错退出，不降级手算（手算Wilcoxon的并列秩/单双侧极易错，错了没人发现）。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
PINN_DIR = RESULTS_ROOT / "pinn"

METRICS = ["rel_l2_u", "rel_l2_v", "rel_l2_p", "rel_l2_speed",
           "pressure_drop_rel_error", "wall_max_abs_u_pred", "wall_max_abs_v_pred"]
EVAL_FILES = {"val": "metrics_val_dense.json", "test": "metrics_test_dense.json"}


def _make_stdout_utf8() -> None:
    """脚本会打印中文与 ⇒/±；在非 UTF-8 控制台上这会 UnicodeEncodeError 中途崩掉，
    表现为'闸门自己死了'而不是结论。统一切成 utf-8 并对不可编码字符降级替换。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def require_scipy():
    try:
        from scipy import stats  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "[FAIL] scipy 不可用：%s\n"
            "       本脚本不做手算降级。手算 Wilcoxon 的并列秩与双侧 p 极易写错，而这里的 p 值要进论文回复表。\n"
            "       注意 numpy/scipy 的版本耦合：代码需要 numpy>=2（train_supervised.py:588 用 np.trapezoid），\n"
            "       镜像自带 scipy 1.12.0 会警告需 numpy<1.29 ⇒ 请装支持 numpy 2.x 的 scipy（>=1.13）。" % exc)
    from scipy import stats
    return stats


def parse_run_name(name: str):
    """rev2609_t5c04__s43__o1  →  (t5c04, 43, 1)；t6 臂名带 __region/__uniform 后缀。"""
    if not name.startswith("rev"):
        return None
    parts = name.split("__")
    if len(parts) < 3:
        return None
    cell = parts[0].split("_", 1)[-1]
    try:
        train_seed = int(parts[1][1:])
    except ValueError:
        return None
    obs_field = parts[2]
    arm = ""
    if "_" in obs_field:
        obs_field, arm = obs_field.split("_", 1)
    try:
        obs_seed = int(obs_field[1:])
    except ValueError:
        return None
    if arm:
        cell = "%s|%s" % (cell, arm)
    return cell, train_seed, obs_seed


def read_case_metrics(run_dir: Path, split: str):
    """返回 (pooled, mean_of_cases, n_cases)；缺文件返回 None。"""
    path = run_dir / "evaluations" / EVAL_FILES[split]
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    pooled = {k: float(v) for k, v in (payload.get("global_metrics") or {}).items() if k in METRICS}
    cases = payload.get("case_metrics") or []
    if isinstance(cases, dict):
        cases = list(cases.values())
    means = {}
    for metric in METRICS:
        vals = [float(c[metric]) for c in cases if isinstance(c, dict) and metric in c]
        if vals:
            means[metric] = sum(vals) / len(vals)
    return {"pooled": pooled, "mean_of_cases": means, "n_cases": len(cases),
            "cases": [c.get("case_id") for c in cases if isinstance(c, dict)]}


def collect(prefix: str, split: str, pinn_dir: Path = PINN_DIR):
    rows = []
    missing = []
    for run_dir in sorted(Path(pinn_dir).glob(prefix + "*")):
        parsed = parse_run_name(run_dir.name)
        if not parsed:
            continue
        cell, train_seed, obs_seed = parsed
        payload = read_case_metrics(run_dir, split)
        if payload is None:
            missing.append(run_dir.name)
            continue
        rows.append({"run": run_dir.name, "cell": cell, "train_seed": train_seed,
                     "obs_seed": obs_seed, **payload})
    return rows, missing


def group_stats(rows, metric, basis):
    """basis ∈ {mean_of_cases, pooled} → {cell: {n, mean, std, values}}"""
    buckets = defaultdict(list)
    for row in rows:
        value = row[basis].get(metric)
        if value is not None:
            buckets[row["cell"]].append(value)
    out = {}
    for cell, values in buckets.items():
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else float("nan")
        out[cell] = {"n": len(values), "mean": mean, "std": std,
                     "cv": (std / abs(mean)) if (mean and not math.isnan(std)) else float("nan"),
                     "values": values,
                     "warn": ("n=1 无任何统计强度" if len(values) == 1 else "")}
    return out


def paired(rows, cell_a, cell_b, metric, basis):
    """按 (obs_seed, train_seed) 取交集配对。"""
    def index(cell):
        found = {}
        for row in rows:
            if row["cell"] != cell:
                continue
            value = row[basis].get(metric)
            if value is not None:
                found[(row["obs_seed"], row["train_seed"])] = value
        return found
    da, db = index(cell_a), index(cell_b)
    keys = sorted(set(da) & set(db))
    only_a = sorted(set(da) - set(db))
    only_b = sorted(set(db) - set(da))
    return [(da[k], db[k]) for k in keys], keys, only_a, only_b


def min_two_sided_p(n: int) -> float:
    """n 对全非零差值、无并列时，双侧 Wilcoxon 符号秩的最小可达 p = 2/2^n。"""
    return 2.0 / (2 ** n) if n > 0 else float("nan")


def judge(cell_a, cell_b, a, b, metric):
    """a/b 是 group_stats() 里**单个格**的统计字典（不是按 cell 索引的那层）。"""
    if a is None or b is None:
        return "不可判：对照两臂里有一格没有读数"
    if a["n"] < 2 or b["n"] < 2:
        return "不可判：n<2，没有组内离散度可比"
    diff = abs(a["mean"] - b["mean"])
    band = max(a["std"], b["std"])
    direction = cell_a if a["mean"] < b["mean"] else cell_b
    if diff < band:
        return ("不可判：%s 两臂差 %.5f < 组内 std %.5f ⇒ 落在换一次种子的波动带内，正文不得写谁占优"
                % (metric, diff, band))
    return "%s 在该指标上占优（差 %.5f ≥ 组内 std %.5f；仍需配对检验给 p）" % (direction, diff, band)


def self_test():
    """正对照：合成数据必须能被抓出显著；零位移必须被判不可判。不跑训练。"""
    stats = require_scipy()
    print("[self-test] 目的：证明检验装置会红也会绿，而不是恒绿/恒红")
    ok = True
    for n in (5, 6, 8, 16):
        print("  n=%2d 最小可达双侧 p = %.6f %s"
              % (n, min_two_sided_p(n), "← 数学上不可能 <0.05" if min_two_sided_p(n) >= 0.05 else ""))
    if min_two_sided_p(5) >= 0.05 and min_two_sided_p(6) < 0.05 and min_two_sided_p(8) < 0.03:
        print("  [OK] 最小可达 p 的计算与 Wilcoxon 的理论下限一致（n=5 不可能显著）")
    else:
        print("  [FAIL] 最小可达 p 算错"); ok = False

    base = [0.0310, 0.0295, 0.0330, 0.0302, 0.0298, 0.0321, 0.0305, 0.0299]
    shifted = [v + 0.020 for v in base]           # 大位移 ⇒ 必须显著
    jitter = [v + (0.0004 if i % 2 else -0.0004) for i, v in enumerate(base)]  # 小于离散度 ⇒ 必须不显著
    for tag, other, want_sig in (("大位移", shifted, True), ("微小位移", jitter, False)):
        res = stats.wilcoxon(base, other, alternative="two-sided")
        got_sig = res.pvalue < 0.05
        flag = "OK" if got_sig == want_sig else "FAIL"
        print("  [%s] %-6s W=%.1f p=%.6g（期望显著=%s）" % (flag, tag, res.statistic, res.pvalue, want_sig))
        if flag == "FAIL":
            ok = False
    rows = [
        {"cell": "cA", "obs_seed": 0, "train_seed": 42, "mean_of_cases": {"rel_l2_speed": a}, "pooled": {}}
        for a in base] + [
        {"cell": "cB", "obs_seed": 0, "train_seed": 42, "mean_of_cases": {"rel_l2_speed": b}, "pooled": {}}
        for b in shifted]
    pairs, keys, oa, ob = paired(rows, "cA", "cB", "rel_l2_speed", "mean_of_cases")
    if len(pairs) != 8 or oa or ob:
        print("  [FAIL] 配对交集逻辑不对：%d 对，孤儿 %s/%s" % (len(pairs), oa, ob)); ok = False
    else:
        print("  [OK] 配对交集 8 对、无孤儿")
    sa = group_stats(rows, "rel_l2_speed", "mean_of_cases")
    small = group_stats(rows, "rel_l2_speed", "mean_of_cases")
    big_rows = [dict(r) for r in rows]
    for r in big_rows:
        if r["cell"] == "cB":
            r["mean_of_cases"] = {"rel_l2_speed": r["mean_of_cases"]["rel_l2_speed"] * 0.1}
    big = group_stats(big_rows, "rel_l2_speed", "mean_of_cases")
    v_small = judge("cA", "cB", small.get("cA"), small.get("cB"), "rel_l2_speed")
    v_big = judge("cA", "cB", big.get("cA"), big.get("cB"), "rel_l2_speed")
    print("  [self-test] 小差异判词：%s" % v_small)
    print("  [self-test] 大差异判词：%s" % v_big)
    ok = ok and ("不可判" in v_small) and ("占优" in v_big)
    print("  [%s] judge 这道闸两个方向都能变（小差异→不可判，大差异→占优）"
          % ("OK" if ("不可判" in v_small and "占优" in v_big) else "FAIL"))
    print("[self-test] %s" % ("全部通过" if ok else "有失败项"))
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="T5/T6 扫描统计")
    ap.add_argument("--prefix", default="rev2609")
    ap.add_argument("--results-root", default=str(RESULTS_ROOT), help="含 pinn/ 的结果根目录（默认为 model/results）")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--metric", default="rel_l2_speed", choices=METRICS)
    ap.add_argument("--basis", default="both", choices=["both", "mean_of_cases", "pooled"])
    ap.add_argument("--paired", default="", help="如 t5c04,t5c08（分层 vs 均匀）或 t6_region|region,t6_uniform|uniform")
    ap.add_argument("--out", default="", help="JSON 输出路径")
    ap.add_argument("--self-test", action="store_true", help="只跑合成数据正对照，不读产物")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

    rows, missing = collect(args.prefix, args.split, Path(args.results_root) / "pinn")
    if not rows:
        print("INVALID：前缀 %s 在 %s/pinn 下没有可解析的 run（不是'没有差异'，是根本没数据）"
              % (args.prefix, args.results_root))
        raise SystemExit(1)
    # scipy 只在配对检验处要求；聚合/判"不可判"用 stdlib 就能算，便于在没装 scipy 的机器上先看数
    bases = ["mean_of_cases", "pooled"] if args.basis == "both" else [args.basis]
    report: dict = {"prefix": args.prefix, "split": args.split, "n_runs": len(rows),
                    "unparsable_or_missing_eval": missing,
                    "basis_note": {
                        "mean_of_cases": "逐工况先算 rel-L2/压降，再对工况取算术均值 = 论文表5-1/5-4/5-5/5-6/5-7 现用口径",
                        "pooled": "该 split 所有点合并算一次 = evaluations/metrics_*.json#global_metrics，数值与上者不同"},
                    "cells": {}, "paired": [], "verdicts": []}
    print("[load] 可解析 run=%d，缺评估文件=%d" % (len(rows), len(missing)))

    for basis in bases:
        grouped = group_stats(rows, args.metric, basis)
        report["cells"][basis] = grouped
        print("\n=== %s / %s / %s ===" % (args.metric, args.split, basis))
        print("%-22s %-4s %-26s %-10s %s" % ("cell", "n", "mean ± std", "cv", "备注"))
        for cell in sorted(grouped):
            g = grouped[cell]
            std = ("%.6f" % g["std"]) if not math.isnan(g["std"]) else "NA"
            cv = ("%.1f%%" % (100 * g["cv"])) if not math.isnan(g["cv"]) else "NA"
            print("%-22s %-4d %.6f ± %s            %-10s %s" % (cell, g["n"], g["mean"], std, cv, g["warn"]))

    if args.paired:
        stats = require_scipy()
        cell_a, cell_b = [x.strip() for x in args.paired.split(",")]
        if len(cell_a) == 0 or len(cell_b) == 0:
            raise SystemExit("--paired 需要两个格名")
        for basis in bases:
            pairs, keys, oa, ob = paired(rows, cell_a, cell_b, args.metric, basis)
            n = len(pairs)
            entry = {"cell_a": cell_a, "cell_b": cell_b, "basis": basis, "metric": args.metric,
                     "n_pairs": n, "pair_keys": ["o%d_s%d" % k for k in keys],
                     "orphan_a": ["o%d_s%d" % k for k in oa], "orphan_b": ["o%d_s%d" % k for k in ob],
                     "min_reachable_two_sided_p": min_two_sided_p(n)}
            if n < 2:
                entry["error"] = "配对数<2，无法检验（这是装置问题，不是结论）"
                print("\n=== 配对检验 %s vs %s（%s）===\n  [FAIL] %s" % (cell_a, cell_b, basis, entry["error"]))
                report["paired"].append(entry)
                continue
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            wil = stats.wilcoxon(xs, ys, alternative="two-sided")
            ttest = stats.ttest_rel(xs, ys)
            entry["wilcoxon"] = {"W": float(wil.statistic), "p": float(wil.pvalue)}
            entry["paired_t"] = {"t": float(ttest.statistic), "p": float(ttest.pvalue)}
            entry["mean_diff"] = statistics.fmean([x - y for x, y in pairs])
            report["paired"].append(entry)
            print("\n=== 配对检验 %s vs %s / %s / %s ===" % (cell_a, cell_b, args.metric, basis))
            print("  配对单位 n=%d（单位=(obs_seed,train_seed)）；孤儿 A=%d B=%d" % (n, len(oa), len(ob)))
            print("  **本组最小可达双侧 p = %.6f**%s" % (entry["min_reachable_two_sided_p"],
                  "  ⇒ 该 n 下不可能得到 p<0.05，只能报 mean±std"
                  if entry["min_reachable_two_sided_p"] >= 0.05 else ""))
            print("  Wilcoxon W=%.1f p=%.6g | 配对 t=%.3f p=%.6g | 平均差 %.6f"
                  % (wil.statistic, wil.pvalue, ttest.statistic, ttest.pvalue, entry["mean_diff"]))
            if n < 8:
                print("  [WARN] 配对单位不足 8：只用了 %d 个，功效低于方案 §5 的设定" % n)

    # 判"不可判"只用组内离散度，不需要 scipy ⇒ 单独一段，保证没装 scipy 也能给正文用文字
    pairs_to_judge = []
    if args.paired:
        cell_a, cell_b = [x.strip() for x in args.paired.split(",")]
        pairs_to_judge.append((cell_a, cell_b, "用户指定对照"))
    pairs_to_judge += [("t5c04", "t5c08", "分层 vs 均匀 @5%（表5-5）"),
                       ("t5c13", "t5c14", "basic vs geometry，两臂同 soft（表5-1 消融行）")]
    grouped = group_stats(rows, args.metric, "mean_of_cases")
    seen = set()
    for cell_a, cell_b, label in pairs_to_judge:
        if (cell_a, cell_b) in seen or cell_a not in grouped or cell_b not in grouped:
            continue
        seen.add((cell_a, cell_b))
        text = judge(cell_a, cell_b, grouped.get(cell_a), grouped.get(cell_b), args.metric)
        report["verdicts"].append({"A": cell_a, "B": cell_b, "label": label,
                                   "metric": args.metric, "text": text})
        print("\n=== 判词：%s（%s vs %s / %s）===\n  %s" % (label, cell_a, cell_b, args.metric, text))

    print("\n[accounting] 解析 run=%d 用了 metric=%s split=%s 格数=%d 判词=%d 配对检验=%d"
          % (len(rows), args.metric, args.split, len(grouped),
             len(report.get("verdicts", [])), len(report.get("paired", []))))
    if args.paired and not report.get("paired"):
        print("INVALID：请求了配对检验但一条配对记录都没产出 ⇒ 装置问题，不许把空结果当结论")
        raise SystemExit(1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        json.loads(Path(args.out).read_text(encoding="utf-8"))   # 落盘即读回
        print("\n[out] %s 已写并读回校验（cells=%d paired=%d）"
              % (args.out, len(report["cells"]), len(report["paired"])))


if __name__ == "__main__":
    _make_stdout_utf8()
    main()
