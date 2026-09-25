#!/usr/bin/env python3
"""按 obs_seed 重新生成稀疏观测点位 CSV（T4/T5/T6 的两层种子入口）。

为什么需要这个脚本
------------------
仓库里的 obs_sparse_*pct.csv / obs_uniform_*pct.csv 全部是用固定 base seed=42 生成的**一份点位实现**
（model/cases/<family>/data/<case>/meta.json#sampling.seed = 42）。训练脚本的 --seed 只影响初始化、
批序与物理配点（train_velocity_pressure_independent_strict_sparse.py:105-108），**不影响点位**。
而"分层采样 vs 均匀采样"这条结论的主方差源恰恰是点位，所以多种子必须同时换 obs_seed。

种子推导规则与 model/scripts/generate_contraction_case.py:137-156 完全一致：
    base   = OBS_BASE_SEED + obs_seed * OBS_SEED_STRIDE        （默认 42 + k*1000，k=0 即复现已落盘文件）
    region_aware : seed = base + pct
    uniform      : seed = base + 100 + pct
    含噪基底     : seed = base + 5      （generate_contraction_case.py:150-152 的 region 5%）
    含噪扰动     : seed = base + 300 + noise_pct
抽样函数直接 import src/data/sparse_sampling.py，不改一行抽样语义。

安全约束
--------
1. 只读 field_dense.csv，绝不重跑 CFD、绝不改 field_dense.csv。
2. obs_seed=0 不写进 cases/ 目录（那会覆盖已入库文件），只在 --verify-committed 模式下
   生成到内存/临时目录并与已落盘 CSV 逐 sample_id 比对。obs_seed>0 一律带 __s{k} 后缀。
3. 断言"同一 (case, rate, obs_seed) 下分层臂与均匀臂的**总点数**相同"——这是配对比较的前提
   （观测预算必须对齐）。**不断言区域配额相同**：配额正是被试变量本身，若两臂配额相同
   则两臂成了同一策略，比较失效（实测 C-val 5%：分层 13/28/22，均匀 21/18/24）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FAMILY_SUBDIR = {"contraction_2d": "contraction_2d", "bend_2d": "bend_2d"}
FAMILY_SAMPLING_KEY = {"contraction_2d": "contraction", "bend_2d": "bend"}
DEFAULT_CASES = {
    "contraction_2d": "C-base,C-train-1,C-train-2,C-train-3,C-train-4,C-train-5,C-val,C-test-1,C-test-2",
    "bend_2d": (
        "B-base__ip_blunted,B-train-1__ip_blunted,B-train-2__ip_blunted,"
        "B-train-3__ip_blunted,B-val__ip_blunted,B-test-1__ip_blunted"
    ),
}
def _make_stdout_utf8() -> None:
    """脚本会打印中文与 ⇒/±；在非 UTF-8 控制台上这会 UnicodeEncodeError 中途崩掉，
    表现为'闸门自己死了'而不是结论。统一切成 utf-8 并对不可编码字符降级替换。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def parse_list(text: str) -> list[str]:
    """逗号/分号/空白任一种分隔 ⇒ 列表。

    只认逗号时，`--obs-seeds "42 43"` 会被当成**一个**值，然后在 int() 上抛裸 traceback；
    反过来 `--seeds 42,43` 在 shell 侧也会被当成一个种子（9/25 实例上真踩过）。
    两侧统一：先规范化分隔符，再逐项校验类型，坏值把原值打出来。
    """
    return [tok for tok in re.split(r"[,;\s]+", (text or "").strip()) if tok]


def parse_int_list(text: str, flag: str) -> list[int]:
    items = parse_list(text)
    if not items:
        raise SystemExit(f"[FAIL] {flag} 解析出 0 项（收到的原值={text!r}）")
    out: list[int] = []
    for item in items:
        if not re.fullmatch(r"[0-9]+", item):
            raise SystemExit(f"[FAIL] {flag} 含非整数项 {item!r}（收到的原值={text!r}）")
        out.append(int(item))
    return out


def parse_rate_list(text: str, flag: str, *, allow_empty: bool = False) -> list[float]:
    items = parse_list(text)
    if not items:
        if allow_empty:
            return []
        raise SystemExit(f"[FAIL] {flag} 解析出 0 项（收到的原值={text!r}）")
    out: list[float] = []
    for item in items:
        try:
            value = float(item)
        except ValueError:
            raise SystemExit(f"[FAIL] {flag} 含非数值项 {item!r}（收到的原值={text!r}）") from None
        if not 0.0 < value <= 1.0:
            raise SystemExit(f"[FAIL] {flag}={value} 不在 (0,1]（采样率写的是比例，5% 要写 0.05）"
                             f"（收到的原值={text!r}）")
        out.append(value)
    return out


def parse_csv_list(text: str) -> list[str]:  # 旧名保留给读代码的人，行为=parse_list
    return parse_list(text)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def obs_seed_base(obs_seed: int, base: int, stride: int) -> int:
    return int(base) + int(obs_seed) * int(stride)


def seed_for(strategy: str, obs_seed: int, pct: int, base: int, stride: int) -> int:
    root = obs_seed_base(obs_seed, base, stride)
    if strategy == "region":
        return root + pct
    if strategy == "uniform":
        return root + 100 + pct
    if strategy == "region_noise_base":
        return root + pct  # 含噪基底沿用 region 规则（噪声档基底固定 5%，见 build_one）
    raise ValueError(f"Unknown strategy: {strategy}")


def file_stem(strategy: str, pct: int, obs_seed: int, noise_pct: int | None) -> str:
    """obs_seed=0 用已入库的名字；>0 加 __s{k}。含噪档与 generate_contraction_case.py 同命名。"""
    kind = "obs_sparse" if strategy.startswith("region") else "obs_uniform"
    stem = f"{kind}_{pct}pct"
    if noise_pct is not None:
        stem = f"{kind}_5pct_noise_{noise_pct}pct"
    if obs_seed != 0:
        stem = f"{stem}__s{obs_seed}"
    return stem


def committed_path(job: dict) -> Path:
    """该 job 对应的**已入库**观测文件名（obs_seed=0 的命名，不带 __s 后缀）。"""
    stem = file_stem(job["strategy"], job["pct"], 0, job["noise_pct"])
    return (PROJECT_ROOT / "cases" / FAMILY_SUBDIR[job["family"]] / "data"
            / job["case_id"] / f"{stem}.csv")


def disp(path: Path) -> str:
    """打印用相对路径；--write-root 指到仓库外时不能因为 relative_to 抛错而中断正在写的作业。"""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def committed_stats(path: Path) -> dict:
    """纯 stdlib 读已入库 CSV ⇒ 行数、区域直方图、sample_id 序列。

    预算对齐断言的**权威依据就是这里**：它量的是"实际被训练吃到的那份 CSV"，
    不是"我以为抽样会产生的数"。verify 与 --budget-only 两条路都用它填 job["n_points"]。
    """
    rows = 0
    hist: dict[str, int] = {}
    ids: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "sample_id" not in reader.fieldnames:
            raise SystemExit(f"已入库观测文件缺 sample_id 列：{path}")
        for row in reader:
            rows += 1
            ids.append(row["sample_id"])
            if "region_id" in row and row["region_id"] not in ("", None):
                key = str(int(float(row["region_id"])))
                hist[key] = hist.get(key, 0) + 1
    return {"n_points": rows, "region_hist": dict(sorted(hist.items())), "sample_ids": ids}



def build_one(
    dense_field,
    family_key: str,
    strategy: str,
    rate: float,
    pct: int,
    seed: int,
    noise_pct: int | None,
    noise_seed: int | None = None,
):
    """单次抽样，返回 (obs_frame, 区域直方图, 总点数)。抽样函数零改动。"""
    from src.data.sparse_sampling import (  # 延迟 import：--dry-run 不需要 numpy/pandas
        add_gaussian_noise,
        sample_region_aware,
        sample_uniform,
        to_observation_frame,
    )

    if strategy == "uniform":
        sampled = sample_uniform(dense_field, rate=rate, seed=seed)
        tag = f"uniform_{pct}pct"
    else:
        sampled = sample_region_aware(dense_field, rate=rate, seed=seed, family=family_key)
        tag = f"region_aware_{pct}pct"

    frame = to_observation_frame(sampled, sampling_tag=tag, noise_tag="clean")
    if noise_pct is not None:
        if noise_seed is None:
            raise ValueError("含噪档必须显式给 noise_seed，禁止从抽样种子上再加（会与已入库文件不同式）")
        frame = add_gaussian_noise(frame, noise_rate=noise_pct / 100.0, seed=noise_seed)
        frame["noise_tag"] = f"noise_{noise_pct}pct"

    hist: dict[str, int] = {}
    for value in frame["region_id"].tolist():
        hist[str(int(value))] = hist.get(str(int(value)), 0) + 1
    return frame, {"n_points": int(len(frame)), "region_hist": hist, "sampling_tag": str(frame["sampling_tag"].iloc[0]),
                   "noise_tag": str(frame["noise_tag"].iloc[0])}


def plan_jobs(args: argparse.Namespace) -> list[dict]:
    """把 (family, case, strategy, rate, obs_seed, noise) 展开成作业表；dry-run 与执行共用。"""
    rates = parse_rate_list(args.rates, "--rates")
    obs_seeds = parse_int_list(args.obs_seeds, "--obs-seeds")
    strategies = parse_list(args.strategies)
    bad_strat = [s for s in strategies if s not in ("region", "uniform")]
    if bad_strat or not strategies:
        # 否则 "regio" 会被 else 分支静默当成 region 臂跑掉
        raise SystemExit(f"[FAIL] --strategies 只许 region/uniform，收到 {bad_strat or '空列表'}"
                         f"（原值={args.strategies!r}）")
    noise_rates = parse_int_list(args.noise_rates, "--noise-rates") if args.noise_rates.strip() else []
    jobs: list[dict] = []
    for family in parse_csv_list(args.family):
        if family not in FAMILY_SUBDIR:
            raise ValueError(f"Unknown family: {family}")
        cases = parse_csv_list(args.cases) if args.cases else parse_csv_list(DEFAULT_CASES[family])
        for case_id in cases:
            for obs_seed in obs_seeds:
                for strategy in strategies:
                    for rate in rates:
                        pct = int(round(rate * 100))
                        jobs.append({
                            "family": family,
                            "case_id": case_id,
                            "strategy": strategy,
                            "rate": rate,
                            "pct": pct,
                            "obs_seed": obs_seed,
                            "noise_pct": None,
                            "seed": seed_for(strategy, obs_seed, pct, args.base_seed, args.seed_stride),
                            "stem": file_stem(strategy, pct, obs_seed, None),
                        })
                # 含噪档只在 5% region 基底上加，命名与已入库文件同构
                if "region" in strategies and 0.05 in rates and noise_rates:
                    for noise_pct in noise_rates:
                        jobs.append({
                            "family": family,
                            "case_id": case_id,
                            "strategy": "region",
                            "rate": 0.05,
                            "pct": 5,
                            "obs_seed": obs_seed,
                            "noise_pct": noise_pct,
                            "seed": seed_for("region", obs_seed, 5, args.base_seed, args.seed_stride),
                            # 与 generate_contraction_case.py:156 同式：base + 300 + noise_pct
                            "noise_seed": obs_seed_base(obs_seed, args.base_seed, args.seed_stride) + 300 + noise_pct,
                            "stem": file_stem("region", 5, obs_seed, noise_pct),
                        })
    return jobs


def resolve_path(job: dict, *, write_root: Path | None = None) -> Path:
    subdir = FAMILY_SUBDIR[job["family"]]
    root = write_root if write_root is not None else PROJECT_ROOT / "cases" / subdir / "data"
    return root / job["case_id"] / f"{job['stem']}.csv"


def budget_check(jobs: list[dict]) -> tuple[list[str], list[dict]]:
    """同 (family, case, rate, obs_seed) 下两臂**总点数**必须相同。

    不断言区域配额相同：配额正是分层 vs 均匀的被试变量（实测 C-val 5%
    分层 13/28/22、均匀 21/18/24），若配额也相同则两臂变成同一策略，比较失效。
    """
    buckets: dict[tuple, dict[str, int]] = {}
    problems: list[str] = []
    for job in jobs:
        if job["noise_pct"] is not None:
            continue
        count = job.get("n_points")
        if count is None:                      # 计数没算出来 = 闸门没跑到，不是"通过"
            problems.append("作业 %s/%s %s%% 没有 n_points ⇒ 预算断言没测到东西"
                            % (job["family"], job["case_id"], job["pct"]))
            continue
        key = (job["family"], job["case_id"], job["pct"], job["obs_seed"])
        buckets.setdefault(key, {})[job["strategy"]] = int(count)
    table: list[dict] = []
    for key, per_strategy in sorted(buckets.items()):
        row = {"family": key[0], "case_id": key[1], "rate_pct": key[2], "obs_seed": key[3], **per_strategy}
        table.append(row)
        if len(per_strategy) < 2:
            problems.append("预算对齐没法判 %s：只有一臂有读数 %s" % (key, per_strategy))
        elif len(set(per_strategy.values())) != 1:
            problems.append("观测预算不对齐 %s: %s" % (key, per_strategy))
    return problems, table


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate sparse observations per obs_seed (no CFD, no sampling change)")
    parser.add_argument("--family", default="contraction_2d", help="comma list of contraction_2d,bend_2d")
    parser.add_argument("--cases", default="", help="comma list of case ids; default = 该族正式工况")
    parser.add_argument("--strategies", default="region,uniform", help="region=分层(region_aware), uniform=均匀")
    parser.add_argument("--rates", default="0.01,0.05,0.10,0.15", help="采样率，逗号分隔")
    parser.add_argument("--obs-seeds", dest="obs_seeds", default="1,2,3", help="观测种子编号；0=已入库点位，不重写")
    parser.add_argument("--noise-rates", dest="noise_rates", default="", help="如 1,3,5；留空不生成含噪档")
    parser.add_argument("--base-seed", dest="base_seed", type=int, default=42, help="obs_seed=0 时的 base seed（须与 meta.json#sampling.seed 一致）")
    parser.add_argument("--seed-stride", dest="seed_stride", type=int, default=1000, help="相邻 obs_seed 的 base seed 间隔")
    parser.add_argument("--dry-run", action="store_true", help="只打印作业表与将要写出的路径，不读数据不写文件")
    parser.add_argument("--verify-committed", action="store_true", help="用 obs_seed=0 重算点位并与已入库 CSV 比对 sample_id 序列与取值（需 numpy/pandas）")
    parser.add_argument("--budget-only", action="store_true",
                        help="只读已入库 CSV 算点数/区域直方图并跑两臂预算断言：纯 stdlib，本地与实例都可跑，不 import numpy/pandas、不写文件")
    parser.add_argument("--write-root", dest="write_root", default="", help="改写输出根目录（默认 cases/<family>/data）")
    parser.add_argument("--manifest", default="", help="把作业表+点数+区域直方图+sha256 写到此 JSON")
    parser.add_argument("--allow-existing", action="store_true", help="目标文件已存在时跳过而非报错")
    args = parser.parse_args()

    requested_seeds = parse_int_list(args.obs_seeds, "--obs-seeds")
    if (args.verify_committed or args.budget_only) and requested_seeds != [0]:
        # 否则会把 obs_seed=1 的抽样结果拿去和 obs_seed=0 的已入库文件比，必然"不一致"——假红
        print("[note] --verify-committed/--budget-only 只校验已入库点位，obs_seeds 已被强制为 0（原请求 %s）"
              % args.obs_seeds)
        args.obs_seeds = "0"
    if 0 in requested_seeds and not (args.verify_committed or args.budget_only or args.dry_run):
        raise SystemExit("obs_seed=0 不允许写入 cases/（会覆盖已入库点位）。要用 --verify-committed/--budget-only 校验，或只传 --obs-seeds 1,2,3")

    jobs = plan_jobs(args)

    if args.dry_run:
        print(json.dumps({"mode": "dry-run", "n_jobs": len(jobs), "jobs": [
            {k: j[k] for k in ("family", "case_id", "strategy", "pct", "obs_seed",
                              "noise_pct", "seed", "noise_seed") if k in j}
            | {"out": str(resolve_path(j))} for j in jobs]}, ensure_ascii=False, indent=1))
        print(f"[dry-run] {len(jobs)} 个观测文件将被生成；抽样函数与种子推导规则不改")
        return

    if args.budget_only:
        # 只读已入库 CSV ⇒ 用 stdlib 数出行数与区域直方图，喂给同一个 budget_check。
        # 这条路的断言对象是"训练真正吃到的那份数据"，所以它也是 verify 的预算依据来源。
        for job in jobs:
            path = committed_path(job)
            if not path.exists():
                job["budget_problem"] = f"缺已入库文件 {path}"
                continue
            stats = committed_stats(path)
            job["n_points"] = stats["n_points"]
            job["region_hist"] = stats["region_hist"]
            job["n_points_source"] = "committed-csv"
        missing = [j["budget_problem"] for j in jobs if "budget_problem" in j]
        problems, table = budget_check(jobs)
        print(json.dumps({"mode": "budget-only", "n_jobs": len(jobs),
                          "n_points_source": "committed-csv（stdlib 数行 + region_id 直方图）",
                          "budget_table": table[:6], "table_rows": len(table),
                          "missing_files": missing[:5]}, ensure_ascii=False, indent=1))
        if missing:
            print("[FAIL] %d 个已入库观测文件读不到 ⇒ 预算断言没测到东西" % len(missing), file=sys.stderr)
            raise SystemExit(1)
        if problems:
            print("[FAIL] 观测预算不对齐：", file=sys.stderr)
            for item in problems:
                print("  " + item, file=sys.stderr)
            raise SystemExit(1)
        print("[OK] budget-only：%d 个 (工况×采样率) 组合的两臂点数全部相同；"
              "区域配额按设计不同（那是被试变量），例：%s" % (len(table), table[0] if table else {}))
        if args.manifest:
            Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
            Path(args.manifest).write_text(json.dumps({"jobs": jobs, "budget_table": table},
                                                      ensure_ascii=False, indent=1), encoding="utf-8")
            json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            print("[manifest] %s 已写并读回校验" % args.manifest)
        return

    import pandas as pd  # 真正执行才需要 numpy/pandas

    cache: dict[tuple, object] = {}
    for job in jobs:
        case_dir = PROJECT_ROOT / "cases" / FAMILY_SUBDIR[job["family"]] / "data" / job["case_id"]
        field_path = case_dir / "field_dense.csv"
        if not field_path.exists():
            raise SystemExit(f"缺少真值场，无法生成观测：{field_path}")
        key = (job["family"], job["case_id"])
        if key not in cache:
            cache[key] = pd.read_csv(field_path)
        dense = cache[key]

        frame, meta = build_one(
            dense, FAMILY_SAMPLING_KEY[job["family"]], job["strategy"], job["rate"],
            job["pct"], job["seed"], job["noise_pct"], job.get("noise_seed"),
        )
        job.update(meta)

        if args.verify_committed:
            committed = committed_path(job)
            if not committed.exists():
                job["verify"] = "NO-COMMITTED-FILE"
                continue
            # 预算读数一律以**已入库 CSV** 为准（stdlib 数行 + region_id 直方图）。
            # 用我自己重算的数去断言"两臂预算相同"是循环论证：抽样语义若被动过，
            # 两臂会一起错，断言照样通过。所以这里把 job 的计数覆盖回真实数据。
            real = committed_stats(committed)
            job["n_points"] = real["n_points"]
            job["region_hist"] = real["region_hist"]
            job["n_points_source"] = "committed-csv"
            job["regen_n_points"] = int(meta["n_points"])
            old_ids = pd.read_csv(committed)["sample_id"].tolist()
            new_ids = frame["sample_id"].tolist()
            same_order = old_ids == new_ids
            same_vals = same_order and all(
                abs(a - b) <= 1e-9 for a, b in zip(
                    pd.read_csv(committed)["u_obs"].fillna(-1).tolist(), frame["u_obs"].fillna(-1).tolist()))
            job["verify"] = "IDENTICAL" if (same_order and same_vals) else (
                "SAME-SET-DIFF-ORDER" if sorted(old_ids) == sorted(new_ids) else "DIFFERENT")
            continue

        out_path = resolve_path(job, write_root=Path(args.write_root) if args.write_root else None)
        if out_path.exists():
            if not args.allow_existing:
                raise SystemExit(f"目标已存在，拒绝覆盖（要跳过请加 --allow-existing）：{out_path}")
            # 真跳过：不重写文件；点数从磁盘上的文件读，保证续跑后两臂都有读数、断言不缺臂
            stats = committed_stats(out_path)
            job["n_points"] = stats["n_points"]
            job["region_hist"] = stats["region_hist"]
            job["n_points_source"] = "existing-file"
            job["sha256"] = sha256_of(out_path)
            print(f"[skip-obs] {disp(out_path)} 已存在 n={stats['n_points']}（不重写）")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        job["sha256"] = sha256_of(out_path)
        job["n_points_source"] = "regenerated"
        print(f"[obs] {disp(out_path)} n={job['n_points']} seed={job['seed']} 区域={job['region_hist']}")

    problems, table = budget_check(jobs)
    if args.verify_committed:
        bad = [j for j in jobs if j.get("verify") not in ("IDENTICAL",)]
        print(json.dumps({"mode": "verify-committed", "n_jobs": len(jobs),
                          "identical": len(jobs) - len(bad),
                          "n_points_source": "committed-csv（stdlib 数行 + region_id 直方图）",
                          "budget_table": table[:6], "table_rows": len(table),
                          "not_identical": [{k: j[k] for k in ("family", "case_id", "stem", "verify")} for j in bad]},
                         ensure_ascii=False, indent=1))
        rc = 0
        if bad:
            print("[FAIL] obs_seed=0 未能逐位复现已入库点位 ⇒ 抽样语义已被改动，停止使用本脚本产物",
                  file=sys.stderr)
            rc = 1
        if problems:
            print("[FAIL] 观测预算不对齐（按已入库 CSV 判）：", file=sys.stderr)
            for item in problems:
                print("  " + item, file=sys.stderr)
            rc = 1
        if rc:
            raise SystemExit(rc)
        print("[OK] obs_seed=0 逐位复现已入库点位；两臂预算按真实 CSV 判定对齐（区域配额不同属被试变量）")
    else:
        if problems:
            print("[FAIL] 观测预算不对齐：\n  " + "\n  ".join(problems))
            raise SystemExit(1)
        print(f"[OK] {len(jobs)} 个观测文件已生成；两臂预算对齐断言通过（区域配额按设计不同，那是被试变量）")

    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps({"jobs": jobs, "budget_table": table}, ensure_ascii=False, indent=1), encoding="utf-8")
        # 读回自检：声称是 JSON 的文件必须能被解析器加载
        json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        print(f"[manifest] {args.manifest} 顶层作业数={len(jobs)}，已读回校验")


if __name__ == "__main__":
    _make_stdout_utf8()
    main()
