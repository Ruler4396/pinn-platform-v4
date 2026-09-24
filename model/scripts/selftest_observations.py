#!/usr/bin/env python3
"""generate_observations_seeded.py 的本地自测：不需要 numpy/pandas，也不需要实例。

为什么要有这个文件：实例上崩过的 `KeyError: '_n_points'` 出在 `budget_check()`，
而 `--dry-run` 在那之前就 return —— 用"绕过坏分支的命令"当自测，等于没测。
这里补的是那两条真正吃数据的分支（写入 / `--verify-committed`），办法是桩化 pandas
并喂一个受控的假抽样函数，然后要求闸门**能红也能绿**（含 3 条负对照）。

分工（不许越界声称）：
  本文件证明 = 记账走得通、断言有读数、错了会红。
  本文件不证明 = obs_seed=0 能逐位复现已入库点位 —— 那要在实例上用真 pandas 跑
                 `--verify-committed`（真抽样函数在这里被假函数替换掉了）。

用法：python3 model/scripts/selftest_observations.py     # 全绿 rc=0
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent                      # model/
GENERATOR = SCRIPT_DIR / "generate_observations_seeded.py"


# ------------------------------------------------------------------ 假 pandas
class MiniSeries:
    def __init__(self, values):
        self._v = list(values)

    def tolist(self):
        return list(self._v)

    def fillna(self, filler):
        out = []
        for v in self._v:
            try:
                out.append(filler if v in ("", None) else float(v))
            except (TypeError, ValueError):
                out.append(v)
        return MiniSeries(out)


class MiniFrame:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, col):
        return MiniSeries([r.get(col, "") for r in self.rows])

    def to_csv(self, path, index=False):
        cols = list(self.rows[0].keys()) if self.rows else ["sample_id"]
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(self.rows)


def _as_num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def fake_read_csv(path, *args, **kwargs):
    with open(path, encoding="utf-8", newline="") as fh:
        return MiniFrame([{k: _as_num(v) for k, v in row.items()} for row in csv.DictReader(fh)])


def install_fake_pandas() -> None:
    module = types.ModuleType("pandas")
    module.read_csv = fake_read_csv          # type: ignore[attr-defined]
    module.DataFrame = MiniFrame             # type: ignore[attr-defined]
    sys.modules["pandas"] = module


def load_generator():
    spec = importlib.util.spec_from_file_location("genobs_under_test", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def region_hist(rows):
    out: dict[str, int] = {}
    for row in rows:
        if "region_id" in row and row["region_id"] not in ("", None):
            key = str(int(float(row["region_id"])))
            out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


# ------------------------------------------------------------------ 受控假抽样
def make_build_one(mod, family: str, case: str, *, rotate: int = 0, drop_arm: str | None = None):
    """假 `build_one`：把该 job 的已入库 CSV 当"重算结果"返回，可选注入两种污染。"""

    def build(dense_field, family_key, strategy, rate, pct, seed, noise_pct, noise_seed=None):
        path = mod.committed_path({"family": family, "case_id": case, "strategy": strategy,
                                   "pct": pct, "noise_pct": noise_pct})
        rows = [dict(r) for r in fake_read_csv(path).rows]
        if rotate:
            rows = rows[rotate:] + rows[:rotate]        # 点位换序 ⇒ 同集合不同顺序
        if drop_arm and strategy == drop_arm:
            rows = rows[:-1]                            # 少一个点 ⇒ 两臂预算不等 + 集合不同
        return MiniFrame(rows), {"n_points": len(rows), "region_hist": region_hist(rows)}

    return build


# ------------------------------------------------------------------ 断言对象
def run_main(mod, argv, build_one) -> tuple[int, str]:
    mod.build_one = build_one
    old_argv = sys.argv
    sys.argv = [str(GENERATOR.name)] + argv
    buf = io.StringIO()
    rc = 0
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            mod.main()
    except SystemExit as exc:
        rc = int(exc.code or 0)
    finally:
        sys.argv = old_argv
    return rc, buf.getvalue()


def check_gate(mod) -> int:
    """budget_check 的正/负对照：不许出现"永远绿"的闸门。返回不符例数。"""

    def job(strategy, n, pct=5, case="C-val", family="contraction_2d", obs=1, noise=None):
        d = {"family": family, "case_id": case, "strategy": strategy, "pct": pct,
             "obs_seed": obs, "noise_pct": noise}
        if n is not None:
            d["n_points"] = n
        return d

    cases = [
        ("两臂点数相同", [job("region", 63), job("uniform", 63)], 0),
        ("两臂点数不同", [job("region", 63), job("uniform", 64)], 1),
        ("只有一臂有读数", [job("region", 63)], 1),
        ("两臂都没读数(=闸没跑到)", [job("region", None), job("uniform", None)], 2),
        ("含噪档不参与对齐", [job("region", 63), job("uniform", 63), job("region", 999, noise=1)], 0),
        ("不同 obs_seed 不互相比", [job("region", 63, obs=1), job("uniform", 99, obs=2)], 2),
    ]
    bad = 0
    for name, jobs, want in cases:
        problems, _table = mod.budget_check(jobs)
        ok = len(problems) == want
        if not ok:
            bad += 1
        print("  [%s] %-26s 期望 problem=%d 实得=%d %s" % (
            "OK" if ok else "BAD", name, want, len(problems), "" if ok else "<< 不符"))
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="contraction_2d")
    parser.add_argument("--case", default="C-val", help="用哪个已入库工况当样本（只读，不改 cases/）")
    parser.add_argument("--rate", default="0.05", help="采样率档，须该工况两臂都已入库")
    parser.add_argument("--work-root", default=str(Path(tempfile.gettempdir()) / "obs_selftest_write"),
                        help="写入分支的 --write-root（系统临时目录，绝不落在 cases/ 里）")
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    install_fake_pandas()
    mod = load_generator()
    pct = int(round(float(args.rate) * 100))
    work = Path(args.work_root)
    base = ["--family", args.family, "--cases", args.case, "--rates", args.rate]
    write_args = base + ["--obs-seeds", "1", "--write-root", str(work)]
    verify_args = base + ["--obs-seeds", "0", "--verify-committed"]

    print("== A. budget_check 闸门正/负对照 ==")
    bad = check_gate(mod)

    print("\n== B. 两条吃数据的分支端到端（桩化 pandas + 假抽样）==")
    scenarios = [
        ("写入·首次", write_args, make_build_one(mod, args.family, args.case), 0, ["[obs]", "[OK] 2"]),
        ("写入·续跑(--allow-existing)", write_args + ["--allow-existing"],
         make_build_one(mod, args.family, args.case), 0, ["[skip-obs]", "[OK] 2"]),
        ("verify-committed·重放相同", verify_args,
         make_build_one(mod, args.family, args.case), 0, ["[OK]", '"identical": 2']),
        ("负对照·点位换序要报红", verify_args,
         make_build_one(mod, args.family, args.case, rotate=1), 1, ["SAME-SET-DIFF-ORDER", "[FAIL]"]),
        ("负对照·抽样缺点位要报红", verify_args,
         make_build_one(mod, args.family, args.case, drop_arm="uniform"), 1, ["DIFFERENT", "[FAIL]"]),
        ("负对照·两臂预算不等要报红", write_args,
         make_build_one(mod, args.family, args.case, drop_arm="uniform"), 1, ["观测预算不对齐"]),
    ]
    for label, argv, build, want_rc, want_subs in scenarios:
        if "--allow-existing" not in argv:
            shutil.rmtree(work, ignore_errors=True)
        rc, out = run_main(mod, argv, build)
        hits = [s for s in want_subs if s in out]
        ok = rc == want_rc and len(hits) == len(want_subs)
        if not ok:
            bad += 1
        tail = [l for l in out.splitlines() if l.strip()][-1:]
        print("  [%s] %-30s rc=%d(期望%d) 命中=%s/%s  %s" % (
            "OK" if ok else "BAD", label, rc, want_rc, len(hits), len(want_subs),
            (tail[0][:96] if tail else "")))

    n_csv = sum(len(files) for _d, _sub, files in os.walk(work)) if work.is_dir() else 0
    print("  写入分支产出 CSV 数=%d（期望 2）" % n_csv)
    if n_csv != 2:
        bad += 1
    shutil.rmtree(work, ignore_errors=True)

    print("\n== C. 不依赖实例的真数据闸（--budget-only，纯 stdlib）==")
    rc, out = run_main(mod, ["--family", "contraction_2d,bend_2d", "--budget-only"],
                       make_build_one(mod, args.family, args.case))
    ok = rc == 0 and "[OK] budget-only" in out
    if not ok:
        bad += 1
    line = [l for l in out.splitlines() if l.strip()][-1] if out.strip() else ""
    print("  [%s] 两族全部已入库 CSV rc=%d(期望0)  %s" % ("OK" if ok else "BAD", rc, line[:110]))

    print("\n结论：%s" % ("全部相符 ⇒ 记账与闸门可信（抽样数值等价仍须实例 --verify-committed）"
                          if bad == 0 else "有 %d 例不符 ⇒ 闸门不可信，别拿它的绿当结论" % bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
