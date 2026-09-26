#!/usr/bin/env python3
"""E5：同机 CFD 单价（表 5-9 的 A 列）。同一台机器上把仓库自带 `.edp` 每工况跑 ≥7 次，
逐次打印壁钟，中位数/区间由**程序自己算**（不许事后手填），产物 JSON 落仓内并自报 sha256+字节数。

用法（实例上）：
    python3 model/scripts/e5_cfd_price.py --repeats 7
本机自测（没有 FreeFem++ 也能验装置本身）：
    python3 model/scripts/e5_cfd_price.py --self-test
退出码：0=全成功；1=有用例 rc!=0 或产物自证不符（数据不可信）；3=求解器不可用（装置问题，不是读数问题）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
CASES_ROOT = REPO / "model" / "cases"
DEFAULT_CASES = "C-base,C-train-1,C-test-2,B-base,B-train-1"
# 入库 .edp 把原作者机器的绝对路径写死在 ofstream 里（32 份）；换机器必 rc=8，故跑前先改这一行。
# **匹整个字面量、不匹前缀**（9/26 23:4x 统括官抓到：只匹前缀会把 `…_2d/cfd/<case>/` 三层留下，
# 目标父目录不存在 ⇒ 35 次求解全 rc=8，机时白烧）。主机前缀单独剥，其余层级原样保留。
ABS_PATH_RE = re.compile(r'"/root/dev/[^"\n]*"')
HOST_PREFIX_RE = re.compile(r"^/root/dev/[A-Za-z0-9._-]+/")


def find_edp(case: str) -> Path:
    hits = sorted(CASES_ROOT.glob(f"*/cfd/{case}/{case}_stokes.edp"))
    if not hits:
        hits = sorted(CASES_ROOT.glob(f"*/cfd/{case}/{case}.edp"))
    if not hits:
        raise FileNotFoundError(f"找不到 {case} 的 .edp（在 {CASES_ROOT}/*/cfd/{case}/ 下）")
    return hits[0]


def family_of(case: str) -> str:
    return "contraction" if case.startswith("C") else ("bend" if case.startswith("B") else "other")


def literal_targets(text: str) -> list[str]:
    """取出一串 `"..."` 字面量里所有被写死的绝对路径（ofstream 与 cout 回显都算，两者必须一起改）。"""
    return [m.group(0)[1:-1] for m in ABS_PATH_RE.finditer(text)]


OFS_RE = re.compile(r'ofstream\s+\w+\s*\(\s*"([^"]+)"')


def ofstream_targets(text: str) -> list[str]:
    """改写后的真正写点。用 ABS_PATH_RE 找它会漏：改写成功后字面量已经不以 /root/dev/ 开头了。"""
    return OFS_RE.findall(text)


def _descends(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def verify_targets(case: str, targets: list[Path]) -> None:
    """控制打在**目标目录能不能落文件**上，不是打在"文本有没有变"上。
    只查"改写不是空操作"放过的正是 23:4x 那个形状：改得不完整。"""
    bad = []
    for t in targets:
        par = t.parent
        probe = par / ".e5_probe"
        try:
            if not par.is_dir():
                raise FileNotFoundError("父目录不存在")
            probe.write_bytes(b"")
            probe.unlink()
        except Exception as exc:
            bad.append("%s ⇒ %s" % (par.as_posix(), exc))
    if bad:
        raise SystemExit("[INVALID] 工况 %s：改写后仍落不下文件（缺层或不可写）：%s "
                         "⇒ 计时一次都没开始，别上机" % (case, "; ".join(bad)))


def prepare(edp: Path, work: Path, case: str, rewrite_root: Path) -> Path:
    """把绝对写路径整体改到本次工作区外（层级原样保留），并按仓库字节的 CRLF 写回**临时**副本。
    仓内 `.edp` 一个字节都不动 —— 那是"5/5 逐位 diff=0"凭据的所由。"""
    if _descends(rewrite_root, REPO) or _descends(work, REPO):
        raise SystemExit("[INVALID] 改写件/产物目录不许落在仓库内（%s / %s）⇒ "
                         "会污染那批按旧字节取证的 `.edp`/`*_raw.csv`" % (work, rewrite_root))
    raw = edp.read_bytes()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8", errors="replace")
    originals = literal_targets(text)
    if not originals:
        raise SystemExit("[INVALID] %s 里没找到被写死的绝对路径 ⇒ 本次改写是空操作（该件本就可移植，"
                         "或正则失配）" % edp.name)
    fam = family_of(case)

    def _sub(m):
        orig = m.group(0)[1:-1]
        rel = HOST_PREFIX_RE.sub("", orig)              # 例：cases/contraction_2d/cfd/C-base/C-base_raw.csv
        target = rewrite_root / fam / case / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        return '"%s"' % target.as_posix()

    rewritten = ABS_PATH_RE.sub(_sub, text)
    residual = literal_targets(rewritten)
    if residual:
        raise SystemExit("[INVALID] 工况 %s：改写后仍残留 %d 个绝对写路径：%s ⇒ 会有样本 rc=8、机时白烧"
                         % (case, len(residual), "; ".join(residual[:3])))
    targets = [Path(x) for x in ofstream_targets(rewritten)]
    if not targets:
        raise SystemExit("[INVALID] 工况 %s：改写后找不到 ofstream 目标 ⇒ 解析没对上，别跑" % case)
    verify_targets(case, targets)
    dst = work / ("%s_run.edp" % case)
    dst.write_bytes(rewritten.replace("\n", "\r\n").encode("utf-8") if crlf else rewritten.encode("utf-8"))
    return dst


def time_case(case: str, runner: Path, argv_prefix: list[str], repeats: int, timeout_s: float,
            prog: str = "") -> dict:
    samples: list[float] = []
    rcs: list[int] = []
    for k in range(1, repeats + 1):
        t0 = time.perf_counter()
        try:
            proc = subprocess.run([*argv_prefix, "-nw", str(runner)], capture_output=True, timeout=timeout_s)
            rc = proc.returncode
        except FileNotFoundError:
            print(f"[FAIL] 求解器不可用：`{prog or argv_prefix[0]}` 找不到（注意真实二进制只有大写 F，"
                  f"`command -v freefem++` 会假报未装）", file=sys.stderr)
            raise SystemExit(3)
        except subprocess.TimeoutExpired:
            rc, proc = 124, None
        dt = time.perf_counter() - t0
        samples.append(dt)
        rcs.append(rc)
        print(f"  {case}  run {k}/{repeats}  wall={dt * 1000:.0f} ms  rc={rc}")
        if rc != 0:
            tail = (proc.stderr.decode("utf-8", "replace") if proc is not None else "<超时，无 stderr>")[-300:]
            print(f"    [stderr 尾部] {tail.strip()}", file=sys.stderr)
    ok = all(r == 0 for r in rcs)
    return {
        "case": case,
        "family": family_of(case),
        "repeats": repeats,
        "rc_all": rcs,
        "all_zero_rc": ok,
        "samples_ms": [round(s * 1000, 1) for s in samples],
        "min_ms": round(min(samples) * 1000, 1),
        "median_ms": round(statistics.median(samples) * 1000, 1),
        "max_ms": round(max(samples) * 1000, 1),
        "median_s": round(statistics.median(samples), 4),
    }


def attest(path: Path) -> tuple[str, int]:
    b = path.read_bytes()
    return hashlib.sha256(b).hexdigest(), len(b)


def self_test() -> int:
    """没有 FreeFem++ 也能验：桩求解器 + 一条必定红的失败用例。"""
    print("== E5 装置自测（桩求解器；真实求解器在实例上）==")
    with tempfile.TemporaryDirectory(prefix="e5_selftest_") as td:
        root = Path(td)
        stub = root / "stub_solver"
        stub.write_text("#!/usr/bin/env python3\nimport sys, time\n"
                        "time.sleep(0.05)\nprint('stub ok')\nsys.exit(0)\n", encoding="utf-8")
        bad = root / "stub_fail"
        bad.write_text("#!/usr/bin/env python3\nimport sys\nprint('boom', file=sys.stderr)\nsys.exit(8)\n", encoding="utf-8")
        ok = time_case("C-selftest", root / "x.edp", [sys.executable, str(stub)], 7, 30.0)
        assert len(ok["samples_ms"]) == 7 and ok["all_zero_rc"], ok
        assert ok["median_ms"] > 0, "中位数没算出来 ⇒ 聚合逻辑是摆设"
        # 产物自证：写出去再读回来算，sha256 必须与文件字节一致
        out = root / "product.json"
        out.write_text(json.dumps(ok, ensure_ascii=False, indent=2), encoding="utf-8")
        sha, size = attest(out)
        assert sha == hashlib.sha256(out.read_bytes()).hexdigest() and size > 50, (sha, size)
        print(f"  [OK] 7 次全部计时、中位数由程序算、产物自证 sha256={sha[:12]}… bytes={size}")
        # 必定红：桩返回 rc=8 ⇒ all_zero_rc 必须为 False（否则"成功"是装出来的）
        bad_res = time_case("C-fail", root / "z.edp", [sys.executable, str(bad)], 2, 30.0)
        assert bad_res["all_zero_rc"] is False and set(bad_res["rc_all"]) == {8}, bad_res
        print(f"  [OK] 失败正对照：求解器 rc=8 的样本使 all_zero_rc=False（rc_all={bad_res['rc_all']}）"
              f" ⇒ 主循环会走 INVALID 分支，不会把失败读成单价")
        # ⑤ 真件演练（零求解）：对仓库里真实 .edp 跑 prepare()，父目录必须可落文件
        # 统括官 §五 的反对照要求是"五个工况全过"，不是抽两个 ⇒ 直接吃 DEFAULT_CASES
        for real_case in [c.strip() for c in DEFAULT_CASES.split(",")]:
            rwork = root / ("rw_" + real_case)
            rwork.mkdir(parents=True, exist_ok=True)
            dst = prepare(find_edp(real_case), rwork, real_case, rwork / "gen")
            txt = dst.read_bytes().decode("utf-8", "replace")
            assert not literal_targets(txt), "%s 改写后仍有残留绝对路径" % real_case
            tg = [Path(x) for x in ofstream_targets(txt)]
            assert len(tg) >= 1, real_case
            for t in tg:
                assert t.parent.is_dir() and (t.parent / ".x").parent.exists(), t
                assert _descends(t, rwork / "gen"), "改写后的目标跑到了工作区外：%s" % t
            print("  [OK] 真件 prepare（%s）：%d 个 ofstream 目标全部父目录存在、可落文件、且都在本次工作区内" % (real_case, len(tg)))
        # ⑥ 必定红：故意给一个不存在的目标目录 ⇒ verify_targets 必须点名，而不是继续跑
        try:
            verify_targets("C-boom", [root / "nope" / "deep" / "x.csv"])
        except SystemExit as exc:
            assert "缺层或不可写" in str(exc), str(exc)
            print("  [OK] 必定红正对照：目标缺层时 verify_targets 直接 SystemExit（旧版只查\u201c是否空操作\u201d放过这一形状）")
        else:
            raise AssertionError("verify_targets 对缺层目标没报错 ⇒ 这道控制是空的")
        # ⑦ 仓内落点必须被拒（保护 5/5 逐位 diff=0 的凭据）
        try:
            prepare(find_edp("C-base"), REPO / "_e5_should_not_exist", "C-base", REPO / "_e5_gen")
        except SystemExit as exc:
            assert "不许落在仓库内" in str(exc), str(exc)
            assert not (REPO / "_e5_gen").exists() and not (REPO / "_e5_should_not_exist").exists()
            print("  [OK] 仓内落点被拒且未建任何目录")
        else:
            raise AssertionError("仓内落点没被拒")
    print("总体：E5 装置的计时/聚合/自证/失败识别/真件改写/缺层必红/仓内必拒 七项全符合（含\u201c改写后不得残留绝对路径\u201d）")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="E5：同机 CFD 单价（逐次打印，程序算中位数）")
    ap.add_argument("--cases", default=DEFAULT_CASES)
    ap.add_argument("--repeats", type=int, default=7, help="每工况重复次数（<7 会打 WARNING 并标注为不足）")
    ap.add_argument("--bin", default="FreeFem++", help="求解器二进制名（真实名只有大写 F）")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--out", default=str(REPO / "docs" / "benchmarks" / "e5_cfd_price.json"))
    ap.add_argument("--work-dir", default="", help="改写后的 .edp 与产物 CSV 的落点；默认系统临时目录（仓外）")
    ap.add_argument("--self-test", action="store_true", dest="selftest")
    args = ap.parse_args()

    if args.selftest:
        return self_test()

    if args.repeats < 7:
        print(f"[WARN] repeats={args.repeats} < 7 ⇒ 中位数不稳定，本产物只能当下限用，表 5-9 不许删\u201c混合口径\u201d限定句")
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="e5_cfd_"))
    work.mkdir(parents=True, exist_ok=True)
    rewrite_root = work / "gen"
    cases = [c.strip() for c in re.split(r"[,\s]+", args.cases) if c.strip()]
    rows: list[dict] = []
    failed: list[str] = []
    for case in cases:
        edp = find_edp(case)
        runner = prepare(edp, work, case, rewrite_root)
        r = time_case(case, runner, [args.bin], args.repeats, args.timeout_s, prog=args.bin)
        r["edp_src"] = str(edp.relative_to(REPO))
        rows.append(r)
        if not r["all_zero_rc"]:
            failed.append(case)

    fam: dict[str, dict] = {}
    for row in rows:
        fam.setdefault(row["family"], []).extend([row["median_s"]] * 1)
    summary = {k: {"cases": len(v), "median_of_medians_s": round(statistics.median(v), 4),
                   "min_s": round(min(v), 4), "max_s": round(max(v), 4)} for k, v in fam.items()}

    payload = {
        "purpose": "表 5-9 的 A 列（CFD 单工况耗时）——与 B/C 同机同次",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "solver": args.bin, "repeats_per_case": args.repeats,
        "cases": rows, "family_summary": summary,
        "caveats": [
            "计时含进程启动与 IO；口径 = `FreeFem++ -nw <改路径后的 .edp>` 单次全过程",
            "入库 .edp 的绝对写路径已按正则改写后才计时（原样跑必 rc=8）",
            "本产物落仓内 ⇒ 表 5-9 的\u201c混合口径\u201d限定句可在两族都补齐后删除；只补一族则必须保留",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    json.loads(out.read_text(encoding="utf-8"))          # 读回校验，坏了别怪下游
    sha, size = attest(out)

    print("\n[E5] 同机 CFD 单价（程序自算，逐次读数见上）")
    for row in rows:
        print(f"  {row['case']:12s} {row['family']:12s} median={row['median_s']:.4f}s  "
              f"区间 {row['min_s']/1000:.4f}–{row['max_s']/1000:.4f}s  rc_all={row['rc_all']}")
    for k, v in summary.items():
        print(f"  [族] {k:12s} 工况数={v['cases']}  中位数之中位数={v['median_of_medians_s']:.4f}s  "
              f"区间 {v['min_s']:.4f}–{v['max_s']:.4f}s")
    print(f"  产物: {out.relative_to(REPO) if out.is_relative_to(REPO) else out}  bytes={size}  sha256={sha}")
    if failed:
        print(f"INVALID（数据不可信）：{len(failed)} 个工况有 rc!=0 的样本：{failed} ⇒ 不要引用本表任何中位数", file=sys.stderr)
        return 1
    if args.repeats < 7 or len({r['family'] for r in rows}) < 2:
        print("[NOTE] 重复数不足 7 或族数不足 2 ⇒ 表 5-9 的\u201c混合口径\u201d限定句**不许删**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
