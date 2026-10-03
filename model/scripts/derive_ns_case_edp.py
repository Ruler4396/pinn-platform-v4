#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 NS 档 .edp 从**各工况自己的** Stokes 文本派生出来（C-base 之外八份 Stokes 也能出真值）。

为什么要有这个件：`gen_ns_re_edp.py` 的派生装置是按 C-base 写死的——它要求首行等于
`STOKES_FIRST_LINE`，并把绝对导出路径按 C-base 那一串做替换；其余八份 Stokes 的首行与导出路径
都带各自工况名，直接喂给它会 `SystemExit`。本件不改动那一枚（它是路线二/统括官线的在途件，
且自带证明），而是：先把工况名 token 归一化成 C-base 的形状 → 调它自己的
`transform / invert / syntax_findings / unit_findings` → 证完再把名字换回来。
于是"只差对流项"这一条对每个工况都是**它自己的证明**在担保，不是我另写一套插块。

内置对照（承重）：对 C-base 跑本件，产出的文本必须与仓里已入库的
`C-base_ns_re<档>.edp` 逐字节相同。对不上就说明这条归一化—换回的路子在某一处改了别的行。

宽度 N 用该工况自己的 Stokes 真值量级现推（`finalize_ns_truth.hi_decimals_for`），不抄 C-base 的
4/6/2——三条量的范围随几何参数 beta 平移，抄来会让 hi 自己丢位。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import finalize_ns_truth as fz                                    # noqa: E402

_spec = importlib.util.spec_from_file_location("gen_ns_re_edp", HERE / "gen_ns_re_edp.py")
g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g)                                        # type: ignore

CFD_DEFAULT = HERE.parents[0] / "cases" / "contraction_2d" / "cfd"
ABS_RE = re.compile(r'ofstream fout\("([^"]+)"\)')
# 插块锚点里有一条是几何常数行（`real beta = 0.7;`），它对 C-base 唯一；其余八份按各自 beta
# 写，所以归一化时必须连它一起换成 C-base 的那一行，证完再换回。
BETA_RE = re.compile(r"^real beta = [0-9.]+;$", re.M)
C_BASE_BETA = "real beta = 0.7;"


def sha_lf(text_or_bytes) -> str:
    """按 LF 比字节：仓里入库的是 LF，Windows 工作树被 git 检出成 CRLF，直接比文件字节会假红。"""
    b = text_or_bytes if isinstance(text_or_bytes, bytes) else text_or_bytes.encode("utf-8")
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()[:16]


def mag_of(csv_path: Path, field: str, limit: int = 4000) -> float:
    """该工况 Stokes 真值里 |field| 的最大值——N 的唯一来源。"""
    with csv_path.open(encoding="utf-8", newline="") as fh:
        header = fh.readline().strip().split(",")
        if field not in header:
            raise SystemExit(f"{csv_path.name} 没有列 {field}：{header}")
        i = header.index(field)
        best = 0.0
        n = 0
        for line in fh:
            if n >= limit:
                break
            parts = line.strip().split(",")
            if len(parts) <= i:
                continue
            try:
                best = max(best, abs(float(parts[i])))
            except ValueError:
                pass
            n += 1
        if best == 0.0:
            raise SystemExit(f"{csv_path.name}:{field} 全零或读不出数，不能推 N")
        return best


def normalize(stokes: str, case: str):
    """把该工况的文本换成 C-base 的形状；返回 (归一文本, 换回所需的三件事)。"""
    lines = stokes.split("\n")
    hits = [ln for ln in lines if "/root/dev" in ln]
    if len(hits) != 2:
        raise SystemExit(f"{case}: 绝对导出路径出现 {len(hits)} 行，不是 2 行（ofstream + cout）"
                         f"——换路径的规矩不再成立，拒绝派生")
    m = ABS_RE.search(stokes)
    if not m:
        raise SystemExit(f"{case}: 找不到 ofstream fout(\"...\")，无法定位导出路径")
    abs_x = m.group(1)
    if abs_x not in stokes:
        raise SystemExit(f"{case}: 解析出的路径没在文本里，自相矛盾")
    first = lines[0]
    betas = set(BETA_RE.findall(stokes))
    if len(betas) > 1:
        raise SystemExit(f"{case}: 出现两条 beta 声明 {betas}，锚点不唯一，拒绝派生")
    beta_line = betas.pop() if betas else None
    norm = stokes.replace(abs_x, g.SHIPPED_ABS)
    norm = norm.replace(first, g.STOKES_FIRST_LINE, 1)
    if beta_line and beta_line != C_BASE_BETA:
        if norm.count(beta_line) != 1:
            raise SystemExit(f"{case}: beta 行出现 {norm.count(beta_line)} 次，换不动")
        norm = norm.replace(beta_line, C_BASE_BETA, 1)
    if norm.split("\n")[0] != g.STOKES_FIRST_LINE:
        raise SystemExit(f"{case}: 首行换不上（{first!r}），装置不适用")
    if C_BASE_BETA not in norm.split("\n"):
        raise SystemExit(f"{case}: 插块锚点 `{C_BASE_BETA}` 在归一后的文本里不存在——"
                         f"这一份 Stokes 的形状与装置假设不同，拒绝派生")
    return norm, {"abs_x": abs_x, "first": first, "beta": beta_line}


def renorm_ns(out: str, case: str, meta: dict) -> str:
    """`denormalize` 的逆。写成独立函数而不是复用 normalize：NS 文本里那条绝对路径已经被
    装置换成相对名，normalize 的"两行 /root/dev"前提在这份文本上不成立（跑一次就红给我看）。"""
    t = out.replace(f"{case}_ns_re", "C-base_ns_re")
    t = t.replace(f"derived from {case}_stokes.edp", "derived from C-base_stokes.edp")
    t = t.replace(meta["first"], g.STOKES_FIRST_LINE, 1)
    if meta["beta"] and meta["beta"] != C_BASE_BETA:
        t = t.replace(meta["beta"], C_BASE_BETA, 1)
    return t


def denormalize(ns: str, case: str, meta: dict) -> str:
    out = ns.replace("C-base_ns_re", f"{case}_ns_re")
    out = out.replace(f"{case}_ns_re1/probe_syntax.edp", "C-base_ns_re1/probe_syntax.edp")
    out = out.replace("derived from C-base_stokes.edp", f"derived from {case}_stokes.edp")
    out = out.replace(g.SHIPPED_ABS, meta["abs_x"]) if g.SHIPPED_ABS in out else out
    if meta["beta"] and meta["beta"] != C_BASE_BETA:
        out = out.replace(C_BASE_BETA, meta["beta"], 1)
    lines = out.split("\n")
    lines[0] = meta["first"]
    return "\n".join(lines)


def derive_case(cfd: Path, case: str, levels, write: bool, shipped_check: bool):
    sdir = cfd / case
    stokes_path = sdir / f"{case}_stokes.edp"
    raw_path = sdir / f"{case}_raw.csv"
    if not stokes_path.is_file() or not raw_path.is_file():
        return [(case, "SKIP", "缺 Stokes 件")]
    stokes = stokes_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    mags = {f: mag_of(raw_path, f) for f in fz.FIELDS}
    widths = {f: fz.hi_decimals_for(mags[f]) for f in fz.FIELDS}
    norm, meta = normalize(stokes, case)
    spec = g.build_spec(widths)
    g.assert_insertions_are_unique(norm, spec)
    g.assert_anchors_are_unique(norm, spec)
    rows = []
    for label in levels:
        ns = g.transform(norm, label, spec)
        checks = []
        if g.invert(ns, label, spec).strip() != norm.strip():
            checks.append("invert!=Stokes")
        if g.syntax_findings(ns):
            checks.append("lint")
        if g.unit_findings(ns):
            checks.append("re-division")
        out = denormalize(ns, case, meta)
        # 换回工况名只许动那几个 token：把换回的文本再归一化一次，必须逐字节回到证过的 ns
        if renorm_ns(out, case, meta) != ns:
            checks.append("roundtrip")
        if case != "C-base":
            if g.STOKES_FIRST_LINE in out:
                checks.append("leftover-C-base-token")
            # 唯一允许留下的 C-base 字样是那句头部注释指的探针件——它确实只在 C-base 档目录下，
            # 把它换成 `{case}_ns_re1/probe_syntax.edp` 会让文件对着一份不存在的件说话。
            strays = [ln for ln in out.split("\n")
                      if "C-base_ns_re" in ln and "probe_syntax.edp" not in ln]
            if strays:
                checks.append(f"leftover-ns-name:{len(strays)}")
        target = cfd / f"{case}_ns_re{label}" / f"{case}_ns_re{label}.edp"
        if shipped_check and target.is_file():
            have = sha_lf(target.read_bytes())
            mine = sha_lf(out)
            checks.append("shipped-same" if have == mine else f"shipped-DIFFER({have}!={mine})")
        status = "OK" if not [c for c in checks if "same" not in c] else "FAIL"
        if write and status == "OK":
            target.mkdir(parents=True, exist_ok=True)
            target.write_text(out if out.endswith("\n") else out + "\n", encoding="utf-8")
            (target.parent / "ns_derivation.json").write_text(json.dumps({
                "derived_by": "model/scripts/derive_ns_case_edp.py",
                "base_case": case, "reynolds_label": label,
                "stokes_edp": str(stokes_path.name), "stokes_raw_for_widths": str(raw_path.name),
                "magnitudes": {f: mags[f] for f in fz.FIELDS},
                "hi_decimals": widths,
                "budget_scale": {f: fz.budget_scale(widths[f]) for f in fz.FIELDS},
                "proof": "invert-to-Stokes / lint / re-division 三条在归一化文本上跑通后才换回工况名",
                "note": "头部注释里 probe_syntax.edp 那条路径指的是 C-base 档的同名件，本工况目录里没有它",
            }, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append((f"Re={label}", status, f"N={widths['u_star']}/{widths['v_star']}/{widths['p_star']}"
                     f" |max|={mags['u_star']:.6g}/{mags['v_star']:.6g}/{mags['p_star']:.6g}",
                     ";".join(checks) or "-"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfd", default=str(CFD_DEFAULT))
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--levels", nargs="*", default=list(g.RE_LEVELS))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-shipped-check", action="store_true")
    a = ap.parse_args()
    cfd = Path(a.cfd)
    cases = a.cases or sorted(p.name for p in cfd.iterdir()
                              if (p / f"{p.name}_stokes.edp").is_file())
    rc = 0
    for case in cases:
        rows = derive_case(cfd, case, a.levels, a.write, not a.no_shipped_check)
        print(f"== {case}")
        for r in rows:
            if r[1] == "SKIP":
                print(f"   {r[0]} SKIP {r[2]}")
                continue
            print(f"   {r[0]:8s} {r[1]:4s} {r[2]}  checks={r[3]}")
            if r[1] == "FAIL":
                rc = 1
    print(f"[{'OK' if rc == 0 else 'FAIL'}] cases={len(cases)} write={a.write}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
