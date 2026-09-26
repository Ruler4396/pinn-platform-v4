#!/usr/bin/env python3
r"""check_md_tables.py — Markdown 表格列数一致性闸（入库版，取代"会话里跑过一次"的临时自检）。

GFM 的规矩：**反引号内的裸 `|` 也算列分隔符**，只有 `\|` 才是字面竖线。
2026-09-27 00:0x 统括官在同一批文件上扫出 4 类错位，而我此前那句"表块列数自检坏行 0"是假绿 ——
因为旧写法数的是 `|` 的**个数**，"把注直接接在末格 `|` 后面"这种**末尾多出一段**的形态个数不变。
本闸按"切出来的格数"判，且带必红正对照 + 合法反对照，见 `--self-test`。

跑法：
    python3 model/scripts/check_md_tables.py                      # 默认扫 docs/
    python3 model/scripts/check_md_tables.py --paths docs paper-route2/../..      # 追加路径
    python3 model/scripts/check_md_tables.py --self-test
退出码：0=无错位；1=有错位/孤儿子表头（逐条打印 file:line）；3=给定的路径不存在（空闸门=测不到，不判绿）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
SENTINEL = "\x01"
DELIM_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def cells(line: str) -> list[str]:
    """先剔掉转义的 `\\|`，再按未转义的 `|` 切；去掉首尾边界管道产生的空段。"""
    s = line.replace("\\|", SENTINEL).rstrip("\n").strip()
    parts = s.split("|")
    if parts and not parts[0].strip():
        parts = parts[1:]
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return [p.replace(SENTINEL, "|") for p in parts]


def in_fence(lineno: int, fences: list[int]) -> bool:
    """行号落在成对 ``` 之间（含围栏本身）时，视为示例代码，不参与表格判定。"""
    inside = False
    for f in fences:
        if f > lineno:
            break
        inside = not inside
    return inside


def is_delim(line: str) -> bool:
    return bool(DELIM_RE.match(line)) and "-" in line


def scan(paths: list[Path]) -> tuple[int, int, int, list[str]]:
    """返回 (文件数, 表块数, 坏行数, 明细)。"""
    files = 0
    blocks = 0
    problems: list[str] = []
    for root in paths:
        md_files = [root] if root.is_file() else sorted(root.rglob("*.md"))
        for f in md_files:
            files += 1
            lines = f.read_text(encoding="utf-8", errors="replace").split("\n")
            fences = [k for k, l in enumerate(lines, 1) if l.lstrip().startswith("```")]
            i = 0
            while i < len(lines):
                if not lines[i].lstrip().startswith("|"):
                    i += 1
                    continue
                start = i
                block = []
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    block.append((i + 1, lines[i]))
                    i += 1
                blocks += 1
                rel = f.relative_to(REPO) if f.is_relative_to(REPO) else f
                if in_fence(start, fences):        # ``` 里的示例表不是表，不判
                    continue
                if len(block) < 2:
                    problems.append("%s:%d 表块只有 1 行（无分隔行/无表体）" % (rel, block[0][0]))
                    continue
                if not is_delim(block[1][1]):
                    problems.append("%s:%d 表头下一行不是分隔行 ⇒ 整块不会被当成表格" % (rel, block[1][0]))
                n = len(cells(block[0][1]))
                body = [b for b in block[2:] if not is_delim(b[1])]
                if not body:
                    problems.append("%s:%d 孤儿子表头（只有表头+分隔行，无表体）" % (rel, block[0][0]))
                for ln, line in block[1:]:
                    if is_delim(line):
                        continue
                    m = len(cells(line))
                    if m != n:
                        problems.append("%s:%d 列数 %d ≠ 表头 %d ⇒ %s"
                                        % (rel, ln, m, n, line.strip()[:70]))
    return files, blocks, len(problems), problems


def self_test() -> int:
    import tempfile
    print("== check_md_tables 自测（必红正对照 + 合法反对照 + 孤儿表头）==")
    with tempfile.TemporaryDirectory(prefix="tblchk_") as td:
        root = Path(td)
        bad = root / "bad.md"
        bad.write_text(
            "| # | 甲 | 乙 |\n|---|---|---|\n| 1 | ok | ok |\n| 2 | 少一格 |\n",
            encoding="utf-8")
        files, blocks, n, det = scan([root])
        assert n == 1, (n, det)
        assert "列数 2 ≠ 表头 3" in det[0], det
        print("  [OK] 必红正对照：故意 3 列里塞一行 2 列 ⇒ 报出 1 行（%s）" % det[0].split(" ")[-1][:34])

        good = root / "good.md"
        good.write_text(
            "| 键 | 值 |\n|---|---|\n"
            "| 打印 `%s` | 合法：反引号里的竖线已转义 |\n| 另一格 | `<a> \\| <b>` |\n" % "obs_sparse.csv \\| C-val",
            encoding="utf-8")
        files, blocks, n2, det2 = scan([good.parent])
        assert n2 == 1, (n2, det2)          # 只应报 bad.md 那一行
        print("  [OK] 合法反对照：反引号内含 `\\|` 的两行**没被误报**（说明剔除转义这步有效）")

        orphan = root / "orphan.md"
        orphan.write_text("| 甲 | 乙 |\n|---|---|\n\n正文\n", encoding="utf-8")
        files, blocks, n3, det3 = scan([orphan])
        assert any("孤儿子表头" in d for d in det3), det3
        print("  [OK] 孤儿表头必报：%d 条明细里含“孤儿子表头”" % n3)

        missing = root / "does-not-exist"
        try:
            resolve_paths([str(missing)])
        except SystemExit as exc:
            print("  [OK] 路径不存在时直接退出（不静默变成空闸门）：%s" % str(exc)[:60])
        else:
            raise AssertionError("缺路径却被静默放过 ⇒ 空闸门会假绿")
        fenced = root / "fenced.md"
        fenced.write_text("示例：\n\n```\n| 甲 | 乙 |\n|---|---|\n```\n\n"
                          "| 真 | 表 |\n|---|---|\n| a | b |\n", encoding="utf-8")
        files, blocks, n4, det4 = scan([fenced])
        assert n4 == 0, (n4, det4)
        print("  [OK] 反对照二：``` 围栏里的示例表不判（示例行不数 ⇒ 否则正本里贴的替换行会全被误报）")
    print("总体：五类控制全符合 ⇒ 这闸能红能绿、不误吃围栏示例，且不会因路径打错而静默空跑")
    return 0


def resolve_paths(raw: list[str]) -> list[Path]:
    out = []
    for r in raw:
        p = Path(r)
        if not p.is_absolute():
            p = (REPO / r).resolve()
        if not p.exists():
            raise SystemExit("[INVALID] 扫描路径不存在：%s ⇒ 不许当“全仓已核”" % p)
        out.append(p)
    return out


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Markdown 表格列数一致性闸")
    ap.add_argument("--paths", nargs="*", default=["docs"], help="要扫的目录或文件（相对仓库根亦可）；可多给几份仓外正本")
    ap.add_argument("--self-test", action="store_true", dest="selftest")
    args = ap.parse_args()
    if args.selftest:
        return self_test()
    paths = resolve_paths(args.paths)
    files, blocks, n, det = scan(paths)
    for d in det:
        print("  RAGGED %s" % d)
    print("files=%d table_blocks=%d ragged_rows=%d" % (files, blocks, n))
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
