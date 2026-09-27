#!/usr/bin/env python3
"""E5：同机 CFD 单价（表 5-9 的 A 列）。同一台机器上把仓库自带 `.edp` 每工况跑 ≥7 次，
逐次打印壁钟，中位数/区间由**程序自己算**（不许事后手填），产物 JSON 落仓内并自报 sha256+字节数。

用法（实例上）：
    python3 model/scripts/e5_cfd_price.py --repeats 7
本机自测（没有 FreeFem++ 也能验装置本身）：
    python3 model/scripts/e5_cfd_price.py --self-test
退出码：0=全成功；1=有 rc!=0 样本、或入统计的成功次数 < --min-ok（⇒ **拒写产物**）、或产物自证不符；3=求解器不可用/探测不通过（装置问题，不是读数问题——9/27 实例上 rc=127 曾被记成 ~1 ms 并入统计，静默把单价算小，现已三处拦死）。
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


def probe_solver(argv_prefix: list[str], work: Path, timeout_s: float, prog: str) -> None:
    """计时之前先证明求解器**起得来**：跑一行 .edp。起不来 ⇒ 整体退出 3（装置问题），
    而不是让 42 次调用各记一个 ~1 ms 的 rc=127 读数、把单价静默算小（9/27 实例上真发生过）。"""
    probe = work / "probe_ok.edp"
    probe.write_text('cout << "E5PROBE\\n";\n', encoding="utf-8")
    try:
        proc = subprocess.run([*argv_prefix, "-nw", str(probe)], capture_output=True, timeout=timeout_s)
    except FileNotFoundError:
        raise SystemExit(f"[FAIL] 求解器不可执行：`{prog or argv_prefix[0]}` 找不到 ⇒ 不计时、不写产物。"
                         f"注意真实二进制只有大写 F（`command -v freefem++` 会假报未装）")
    except subprocess.TimeoutExpired:
        raise SystemExit(f"[FAIL] 求解器探测超时 ⇒ 装置问题，不产出单价")
    out = (proc.stdout or b"").decode("utf-8", "replace")
    if proc.returncode != 0 or "E5PROBE" not in out:
        raise SystemExit(
            "[FAIL] 求解器探测不通过：rc=%s，stdout 里没有 E5PROBE ⇒ 不计时、不写产物。\n"
            "  stderr 尾部：%s\n"
            "  已知成因（9/27 实例实测）：冷启动容器里 `tar xzf ffroot.tgz -C /` 只回二进制，"
            "还缺 libumfpack.so.5 / libcholmod.so.3 / libarpack.so.2 / libhdf5_serial.so.103 "
            "四个 so ⇒ rc=127。补法：apt-get install -y libsuitesparse-dev（前两个）"
            "+ 从 NAS 的 ffstage/lib/x86_64-linux-gnu/ 拷（后两个）。" %
            (proc.returncode, (proc.stderr or b"").decode("utf-8", "replace")[-200:]))
    print("[probe] 求解器可执行性通过（E5PROBE 回显命中）⇒ 开始计时")


def time_case(case: str, runner: Path, argv_prefix: list[str], repeats: int, timeout_s: float,
            prog: str = "", min_ok: int = 5) -> dict:
    samples: list[float] = []          # 只收 rc==0 的读数：失败调用的墙钟不是"单价"，收进来会静默把单价算小
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
        rcs.append(rc)
        if rc == 0:
            samples.append(dt)
            print(f"  {case}  run {k}/{repeats}  wall={dt * 1000:.0f} ms  rc={rc}  → 入统计")
        else:
            print(f"  {case}  run {k}/{repeats}  wall={dt * 1000:.0f} ms  rc={rc}  → **剔除，不入统计**")
            tail = (proc.stderr.decode("utf-8", "replace") if proc is not None else "<超时，无 stderr>")[-300:]
            print(f"    [stderr 尾部] {tail.strip()}", file=sys.stderr)
    n_ok, n_bad = len(samples), len(rcs) - len(samples)
    if n_ok == 0:
        return {"case": case, "family": family_of(case), "repeats": repeats, "rc_all": rcs,
                "n_ok": 0, "n_excluded": n_bad, "min_ok": min_ok, "all_zero_rc": False,
                "citable": False, "samples_ms": [], "min_ms": None, "median_ms": None,
                "max_ms": None, "median_s": None}
    ok = all(r == 0 for r in rcs)
    return {
        "case": case,
        "family": family_of(case),
        "repeats": repeats,
        "rc_all": rcs,
        "n_ok": n_ok,
        "n_excluded": n_bad,
        "min_ok": min_ok,
        "all_zero_rc": ok,
        "citable": n_ok >= min_ok,
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
        assert bad_res["n_ok"] == 0 and bad_res["citable"] is False and bad_res["median_s"] is None, bad_res
        print(f"  [OK] 失败正对照：求解器 rc=8 的样本使 all_zero_rc=False（rc_all={bad_res['rc_all']}）"
              f" ⇒ 全失败时 n_ok=0、citable=False、median_s=None（旧版在这里会拿 2 条失败读数算出假的百毫秒级单价）")
        # ⑧ 9/27 实例真踩过的形状：rc=127（求解器起不来）的读数不得进统计
        miss = root / "stub_127"
        miss.write_text("#!/usr/bin/env python3\nimport sys\nprint('no libumfpack', file=sys.stderr)\nsys.exit(127)\n",
                        encoding="utf-8")
        r127 = time_case("C-127", root / "y.edp", [sys.executable, str(miss)], 7, 30.0)
        assert r127["n_ok"] == 0 and r127["n_excluded"] == 7 and r127["samples_ms"] == [], r127
        assert r127["citable"] is False, r127
        print("  [OK] rc=127 正对照：7 次调用全被剔除（samples_ms 为空、citable=False）"
              " ⇒ 旧版会把 42 次 rc=127 记成 ~1 ms 并入统计，静默把单价算小")
        # ⑨ 剔除后不足最低次数 ⇒ citable 必须为 False（这是"拒写产物"的判据）
        mix = root / "stub_mix"
        mix.write_text("#!/usr/bin/env python3\nimport sys, time\n"
                       "n = int(open(sys.argv[-1]).read().strip() or 0)\n"
                       "open(sys.argv[-1], 'w').write(str(n + 1))\n"
                       "time.sleep(0.02)\n"
                       "sys.exit(0 if n < 4 else 9)\n", encoding="utf-8")
        # ⑨ 剔除后不足最低次数 ⇒ citable 必须为 False（这是"拒写产物"的判据）
        def make_mix(path: Path, limit: int) -> Path:
            path.write_text("#!/usr/bin/env python3\nimport sys, time\n"
                            "n = int(open(sys.argv[-1]).read().strip() or 0)\n"
                            "open(sys.argv[-1], 'w').write(str(n + 1))\n"
                            "time.sleep(0.02)\n"
                            "sys.exit(0 if n < %d else 9)\n" % limit, encoding="utf-8")
            return path

        cnt = root / "counter"
        cnt.write_text("0", encoding="utf-8")
        mixed = time_case("C-mix", cnt, [sys.executable, make_mix(root / "stub_mix4", 4)], 7, 30.0, min_ok=5)
        assert mixed["n_ok"] == 4 and mixed["n_excluded"] == 3 and mixed["citable"] is False, mixed
        cnt.write_text("0", encoding="utf-8")
        ok6 = time_case("C-mix6", cnt, [sys.executable, make_mix(root / "stub_mix6", 6)], 7, 30.0, min_ok=5)
        assert ok6["n_ok"] == 6 and ok6["n_excluded"] == 1 and ok6["citable"] is True, ok6
        print(f"  [OK] 门槛正对照：4/7 成功 ⇒ citable=False；6/7 成功 ⇒ citable=True 且 n_excluded=1"
              f"（剔除条数随产物一起落盘，读者看得见）")
        # ⑩ 求解器探测：二进制不存在 ⇒ SystemExit，而不是开一圈 1 ms
        try:
            probe_solver([str(root / "no_such_bin")], root, 10.0, prog="no_such_bin")
        except SystemExit as exc:
            assert "找不到" in str(exc), str(exc)
            print("  [OK] 探测正对照：二进制不存在时直接退出，不进计时循环")
        else:
            raise AssertionError("求解器不存在却没被探测拦住 ⇒ 探测是摆设")
        # ⑪ 探测存在但起不来（rc!=0）⇒ 同样必须拦住，并带出缺 so 的提示
        brk = root / "stub_broken"
        brk.write_text("#!/usr/bin/env python3\nimport sys\nprint('error while loading shared libraries: "
                       "libumfpack.so.5', file=sys.stderr)\nsys.exit(127)\n", encoding="utf-8")
        try:
            probe_solver([sys.executable, str(brk)], root, 10.0, prog="FreeFem++")
        except SystemExit as exc:
            assert "libumfpack" in str(exc) and "ffstage/lib" in str(exc), str(exc)
            print("  [OK] 探测正对照：rc=127 时起不来就被判装置问题，且把缺 so 的补法一起报出来")
        else:
            raise AssertionError("求解器起不来却没被探测拦住")
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
    print("总体：E5 装置 11 项控制全符合——计时 / 聚合 / 产物自证 / 失败识别 / 全失败不产出单价 / rc=127 剔除 /"
          " 4 比 7 拒写与 6 比 7 放行 / 探测缺二进制必退 / 探测 rc!=0 必退并报缺 so / 真件改写 / 缺层必红 / 仓内必拒")
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
    ap.add_argument("--min-ok", type=int, default=5,
                    help="每工况至少这么多次 rc==0 才允许写产物（默认 5/7）；不够则拒写并退 1")
    ap.add_argument("--self-test", action="store_true", dest="selftest")
    args = ap.parse_args()

    if args.selftest:
        return self_test()

    if args.repeats < 7:
        print(f"[WARN] repeats={args.repeats} < 7 ⇒ 中位数不稳定，本产物只能当下限用，表 5-9 不许删\u201c混合口径\u201d限定句")
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="e5_cfd_"))
    work.mkdir(parents=True, exist_ok=True)
    probe_solver([args.bin], work, min(60.0, args.timeout_s), prog=args.bin)      # 起不来就整体退 3，别记 1 ms
    rewrite_root = work / "gen"
    cases = [c.strip() for c in re.split(r"[,\s]+", args.cases) if c.strip()]
    rows: list[dict] = []
    failed: list[str] = []
    for case in cases:
        edp = find_edp(case)
        runner = prepare(edp, work, case, rewrite_root)
        r = time_case(case, runner, [args.bin], args.repeats, args.timeout_s, prog=args.bin,
                      min_ok=args.min_ok)
        r["edp_src"] = str(edp.relative_to(REPO))
        rows.append(r)
        if not r["all_zero_rc"]:
            failed.append(case)

    uncitable = [r["case"] for r in rows if not r["citable"]]
    if uncitable:
        print(f"[INVALID] 有 {len(uncitable)} 个工况入统计的成功读数不足 {args.min_ok} 次：{uncitable}"
              f" ⇒ **拒写产物**（不产出可引用 JSON）。逐次 rc 与剔除条数见上；"
              f"先修装置（缺 so／路径不可写）再重跑，那一次才叫同机同次。", file=sys.stderr)
        return 1

    fam: dict[str, dict] = {}
    for row in rows:
        fam.setdefault(row["family"], []).append(row["median_s"])
    summary = {k: {"cases": len(v), "median_of_medians_s": round(statistics.median(v), 4),
                   "min_s": round(min(v), 4), "max_s": round(max(v), 4)} for k, v in fam.items()}

    payload = {
        "purpose": "表 5-9 的 A 列（CFD 单工况耗时）——与 B/C 同机同次",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "solver": args.bin, "repeats_per_case": args.repeats, "min_ok_runs": args.min_ok,
        "cases": rows, "family_summary": summary,
        "caveats": [
            "计时含进程启动与 IO；口径 = `FreeFem++ -nw <改路径后的 .edp>` 单次全过程",
            "入库 .edp 的绝对写路径已按正则改写后才计时（原样跑必 rc=8）",
            "**只有 rc==0 的调用进统计**；失败调用的墙钟（常常是 ~1 ms 的 rc=127）会把单价静默算小 ⇒ 已排除，"
            "每工况的 n_ok/n_excluded 随产物一起落盘",
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
              f"区间 {row['min_ms'] / 1000:.4f}–{row['max_ms'] / 1000:.4f}s  "
              f"n_ok={row['n_ok']}/{row['repeats']}（剔除 {row['n_excluded']}）  rc_all={row['rc_all']}")
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
