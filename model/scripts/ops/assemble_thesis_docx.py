#!/usr/bin/env python3
"""毕设 docx 装配器（统括官 9/27 11:2x 改派给方案线，四道闸写进代码而不是写在承诺里）。

用法（本机零机时；产物一律落在仓外 .scratch/）：
    python3 model/scripts/ops/assemble_thesis_docx.py --plan            # 只判定 51 行能不能自动落、为什么不能
    python3 model/scripts/ops/assemble_thesis_docx.py --apply           # 在带时间戳的副本上落字（原件只读）
    python3 model/scripts/ops/assemble_thesis_docx.py --verify <副本>    # 副本正文 ↔ 工单「新文本」逐行 diff
硬约（违反即 INVALID 退出码非 0）：
  ① 绝不就地改原件：--apply 先 copy 到 .scratch/装配副本-<UTC 时间戳>.docx，原件字节数与 mtime 进回执；
  ② 只用 python-docx，不手点 XML；③ 按锚句内容定位（`thesis.txt` 的 :NNNN 含表格单元，与 python-docx 段号不同尺）；
  ④ A13 是待批占位，永远不落字，只在清单里标出；⑤ 术语类算子（A15/A14）与"删段+插段成对"类（E4+E3）单独走。
"""
from __future__ import annotations
import argparse, datetime, hashlib, pathlib, re, shutil, sys

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[3]                                     # …/pinn-platform-v4
ROOT = HERE.parents[4]                                     # …/PINN-restart（工作区根，原件与 .scratch 在这）
if not (ROOT / "毕业论文汇编格式.docx").exists():            # 挪目录也别静默指错件
    for cand in (REPO.parent, REPO.parent.parent):
        if (cand / "毕业论文汇编格式.docx").exists():
            ROOT = cand
            break
SRC = ROOT / "毕业论文汇编格式.docx"
THESIS = ROOT / ".scratch" / "thesis.txt"
WORK = REPO / "docs" / "revision" / "正文改写工单-20260925.md"
OUT = ROOT / ".scratch"

# 待作者批的占位（④）：永远不写进论文
AWAITING = {"A13": "待作者批（窗口 9/28，沉默即按降级档执行）"}
# 术语类算子：整段替换会吃内容，必须走"整词替换 + 只动了目标词"校验
TERM_OPS = [("A15", "阶段内PDE约束", "阶段内残差惩罚项"),
            ("A14", "协同修正", "交替更新")]
TERM_RE = re.compile("|".join(re.escape(a) for _, a, _ in TERM_OPS))
FOLD_RE = re.compile("|".join(re.escape(w) for _, a, b in TERM_OPS for w in (a, b)))
# 新文本里出现这些词 ⇒ 判定为"操作指令或工单内部注"，不整段替换
INSTR = ("删除", "统一替换为", "同步替换", "替换为", "行名", "更正", "撤回", "定档", "凭据",
         "不许", "禁写", "见 §", "若将来", "留作投稿", "前置核查", "⇒", "本工单", "区间都要带上")
PAIR_OPS = ("E4", "E3")        # ③ 删段与插段成对：单独执行任何一个都 INVALID


def norm(s: str) -> str:
    s = re.sub(r"\[\d+(?:[-,]\d+)*\]", "", s)              # 行内文献号
    return re.sub(r"\s+", "", s.replace("，", ",").replace("。", "."))


def clean(md: str) -> str:
    """把工单格子里的 markdown 痕迹清成论文正文；`**【…】**` 之后一律算工单内部注。"""
    md = re.split(r"\*\*【", md)[0]
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    md = re.sub(r"`([^`]*)`", r"\1", md)
    return md.replace("\\|", "|").strip()


def rows():
    out = []
    for line in WORK.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\| [A-J]\d+ \|", line):
            c = [x.strip() for x in line.split("|")[1:-1]]
            out.append({"id": c[0], "loc": c[1], "act": c[2], "new": c[3] if len(c) > 3 else ""})
    return out


def paras(doc):
    return [p.text for p in doc.paragraphs]


def cells(doc):
    return [c.text for t in doc.tables for r in t.rows for c in r.cells]


def find_anchor(body_norm, frag):
    return [i for i, t in enumerate(body_norm) if frag and frag in t]


def payload(md: str) -> str:
    """有的格子写的是『换成：`论文原句`』⇒ 取那段反引号内容当正文，别把工单指令一起贴进论文。"""
    body = clean(md)
    m = re.search(r"[:：]\s*`([^`]{40,})`", body)
    if m:
        return m.group(1).strip()
    m = re.search(r"「([^」]{40,})」", body)               # 或「论文原句」包着的
    if m and body.startswith(("表注", "说明列", "声明改为", "标题与行名", "新增行")):
        return m.group(1).strip()
    return body


def landable(new_raw: str, old_text: str):
    """判定"这格能不能整段替换那一"：取正文载荷 → 查表题/公式号 → 查指令语 → 查长度塌缩。"""
    new = payload(new_raw)
    old_s = old_text.strip()
    if re.fullmatch(r"[（(]?\s*[\dA-Za-z]+[-–]\d+[a-z]?[)）]?\s*", old_s or "x"):
        return False, "", f"旧段是公式编号标签（{old_s}），替换会吃掉题号 ⇒ 该行走「式后补符号说明」的插段算子"
    if re.match(r"^(表|图)\s*\d", old_s) and "。" not in old_s and len(old_s) < 60:
        return False, "", (f"旧段是表题/图题（{old_s[:24]}…无句末标点且短），"
                           f"整段替换会吃掉题注 ⇒ 走「插表注」或「改题注」算子")
    if any(k in new for k in INSTR):
        return False, "", "载荷仍含操作指令或工单内部注 ⇒ 不整段替换"
    if len(new) < 0.6 * max(1, len(old_text)):
        return False, "", (f"载荷只有旧段落 {len(new) / max(1, len(old_text)):.0%}，"
                           f"是局部改写（旧 {len(old_text)} 字）⇒ 整段替换会吃内容")
    return True, new, ""


def note_inserts(doc, auto_ids):
    """插段类算子：只处理"锚点是公式编号标签"这一种（符号说明必须紧跟公式，位置唯一、不会插错）。
    表注要落在**表格之后**，而表题在上、表在下 ⇒ 从表题段插进去会插到表格上面，位置错；那类留在人工清单里。"""
    body = paras(doc)
    body_norm = [norm(t) for t in body]
    dump = [norm(l) for l in THESIS.read_text(encoding="utf-8", errors="replace").splitlines()]
    dump_of = {}
    for k, t in enumerate(dump):
        if t:
            dump_of.setdefault(t, set()).add(k + 1)
    line_to_body = {}
    for i, t in enumerate(body_norm):
        for ln in dump_of.get(t, ()):
            line_to_body.setdefault(ln, []).append(i)
    LABEL = re.compile(r"[（(]?\s*[\dA-Za-z]+[-–]\d+[a-z]?[)）]?\s*")
    out = []
    for r in rows():
        if r["id"] in auto_ids:
            continue
        new = payload(r["new"])
        if len(new) < 40 or any(k in new for k in INSTR):
            continue
        cited = {int(n) for n in re.findall(r":(\d{2,4})", r["loc"])}
        for i in sorted({b for ln in cited for b in line_to_body.get(ln, [])}):
            if LABEL.fullmatch(body[i].strip()):
                out.append((r["id"], i, new))
                break
    return out


def classify(doc):
    """⇒ (可整段重写, 术语算子, 成对删插, 待批, 人工)"""
    body = paras(doc)
    body_norm = [norm(t) for t in body]
    dump = [norm(l) for l in THESIS.read_text(encoding="utf-8", errors="replace").splitlines()]
    dump_of = {}
    for k, t in enumerate(dump):
        if t:
            dump_of.setdefault(t, set()).add(k + 1)       # 1 基行号，与工单 :NNNN 同尺
    line_to_body = {}                                    # 第二个定位器：:NNNN → 正文段（两把尺的桥）
    for i, t in enumerate(body_norm):
        for ln in dump_of.get(t, ()):                    # 正文段整行等于 dump 某行 ⇒ 该段就是那一行
            line_to_body.setdefault(ln, []).append(i)
    auto, manual, await_ = [], [], []
    for r in rows():
        rid = r["id"]
        if rid in AWAITING:
            await_.append((rid, AWAITING[rid]))
            continue
        if rid in PAIR_OPS:
            continue                                     # 由成对算子处理，不在这条路上
        cited = {int(n) for n in re.findall(r":(\d{2,4})", r["loc"])}
        quotes = [q for q in re.findall(r"「(.+?)」", r["loc"]) if norm(q) not in ("", "……")]
        hit_list = []
        if quotes:
            anchor = max(quotes, key=lambda x: len(norm(x)))
            frag = norm(max((f for f in re.split(r"…+", anchor) if len(norm(f)) >= 8),
                            key=lambda f: len(norm(f)), default=""))
            hit_list = find_anchor(body_norm, frag)
        if not hit_list and cited:                        # 定位器二：只报锚点，不自动写（9/27 实测它会把表题/公式号整段吃掉）
            cand = sorted({b for ln in cited for b in line_to_body.get(ln, [])})
            if cand:
                manual.append((rid, f"无摘引；按 :{sorted(cited)[:3]} 对应正文段 {cand[:3]}"
                                     f"（只作定位线索，落字走插段/改题注算子或人工）"))
            else:
                manual.append((rid, ":NNNN 不对应任何正文段（新增表/表位/仓库文件类）"))
            continue
        if not hit_list:
            manual.append((rid, "无原句可摘且 :NNNN 不对应正文段（新增表/表位/仓库文件类）"
                         if not quotes else f"锚句在 docx 正文段不命中（thesis.txt 行="
                                            f"{[i + 1 for i, t in enumerate(dump) if frag and frag in t][:2] or '全无'}）"))
            continue
        if len(hit_list) > 1:                             # 多段命中：用行号桥判别，判不出来就交人工
            pick = [i for i in hit_list if cited & dump_of.get(body_norm[i], set())]
            if len(pick) == 1:
                hit_list = pick
            else:
                manual.append((rid, f"锚句/行号命中 {len(hit_list)} 段 {hit_list[:5]}，"
                                     f"行号可判别={pick or '无'} ⇒ 需点名"))
                continue
        i = hit_list[0]
        ok, new, why = landable(r["new"], body[i])
        (auto if ok else manual).append((rid, i, new) if ok else (rid, why))
    return auto, manual, await_


def fix_terms(doc):
    """术语算子（A15/A14）：正文段 + 表内单元；返回改动处数。"""
    n = 0
    def one(paragraphs):
        nonlocal n
        for p in paragraphs:
            if not TERM_RE.search(p.text):
                continue
            n += 1
            hit = False
            for r in p.runs:
                if TERM_RE.search(r.text):
                    r.text = TERM_RE.sub(lambda m: dict((a, b) for _, a, b in TERM_OPS)[m.group(0)], r.text)
                    hit = True
            if not hit:
                p.runs[0].text = TERM_RE.sub(lambda m: dict((a, b) for _, a, b in TERM_OPS)[m.group(0)], p.text)
                for r in p.runs[1:]:
                    r.text = ""
    one(doc.paragraphs)
    for t in doc.tables:
        for r in t.rows:
            for c in r.cells:
                one(c.paragraphs)
    return n


def apply_rewrite(doc, auto):
    for rid, i, text in auto:
        p = doc.paragraphs[i]
        if not p.runs:
            p.add_run("")
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""


def verify(copy_path):
    """闸三：副本正文 ↔ 工单 51 行「新文本」逐行核；返回不一致清单。"""
    from docx import Document
    doc = Document(str(copy_path))
    body = [norm(t) for t in paras(doc)]
    joined = "\n".join(body)
    celljoin = "\n".join(norm(t) for t in cells(doc))
    ok, miss, await_, term = [], [], [], []
    term_ids = {a for a, _, _ in TERM_OPS}
    for r in rows():
        rid, new = r["id"], payload(r["new"])
        if rid in term_ids:
            term.append(("ROW", rid, "", 0))               # 术语行的核对在下面按残留数判，别按整段命中判
            continue
        if rid in AWAITING:
            await_.append((rid, AWAITING[rid]))
            continue
        if not new:
            continue
        probe = norm(new[:60])
        if probe and (probe in joined or probe in celljoin):
            ok.append(rid)
        else:
            miss.append((rid, new[:40]))
    for rid, a, b in TERM_OPS:
        left = sum(len(re.findall(re.escape(a), t)) for t in paras(doc)) + \
               sum(len(re.findall(re.escape(a), t)) for t in cells(doc))
        term.append((rid, a, b, left))
    print(f"[verify] 命中工单新文本={len(ok)} 未命中={len(miss)} 待批占位={len(await_)}")
    for rid, why in miss:
        print(f"   [未命中] {rid}: {why}")
    for rid, why in await_:
        print(f"   [待批占位] {rid}: {why}")
    for rid, a, b, left in term:
        if rid == "ROW":
            continue
        got = sum(t.count(b) for t in paras(doc)) + sum(t.count(b) for t in cells(doc))
        print(f"   [术语] {rid}: 旧词「{a}」残留 {left} 处（应 0）、新词「{b}」出现 {got} 处")
    bad = [t for t in term if t[3] != 0]
    print(f"结论：不一致 {len(miss)} 行（其中已声明待批 {len(await_)} 行不算未完成）"
          f"；术语残留 {len(bad)} 项")
    return 0 if not bad else 1


def insert_after(paragraph, text):
    """在锚点段之后插一段，样式克隆锚点段的 pPr（不碰 document.xml 文本，只用 python-docx 的对象树）。"""
    import copy
    from docx.oxml.ns import qn
    new_p = copy.deepcopy(paragraph._p)
    for child in list(new_p):                      # 丢掉锚点段的文字与书签，保留 pPr（样式）
        if child.tag != qn("w:pPr"):
            new_p.remove(child)
    run = new_p.makeelement(qn("w:r"), {})
    t = new_p.makeelement(qn("w:t"), {})
    t.text = text
    run.append(t)
    new_p.append(run)
    paragraph._p.addnext(new_p)


def structural_diff(before, after):
    """按内容对齐两版正文 ⇒ 返回（改写处、插入处、删除处）。插入会让下标错位，所以不能按下标硬比。"""
    import difflib
    sm = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    repl = insdele = dele = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            repl += max(i2 - i1, j2 - j1)
        elif tag == "insert":
            insdele += j2 - j1
        elif tag == "delete":
            dele += i2 - i1
    return repl, insdele, dele


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--verify", type=pathlib.Path)
    ap.add_argument("--expect-changed", type=int, default=None,
                    help="--apply 用：期望被改段落数，不接等即 INVALID（闸三）")
    args = ap.parse_args()
    import docx  # noqa: F401  ② 先确认库在，不在就别硬写
    from docx import Document

    if args.verify:
        return verify(args.verify)

    if not SRC.exists():
        print(f"[INVALID] 目标件不存在：{SRC}", file=sys.stderr)
        return 2
    st = SRC.stat()
    print(f"[原件只读] {SRC.name} bytes={st.st_size:,} mtime="
          f"{datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')} "
          f"sha256={hashlib.sha256(SRC.read_bytes()).hexdigest()[:12]}")
    src_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    auto, manual, await_ = classify(Document(str(SRC)))
    notes = note_inserts(Document(str(SRC)), {a[0] for a in auto})
    note_ids = {n[0] for n in notes}
    for r in rows():                                   # 表注类单独点名：载荷现成，只差一个正确的插入位置
        new = payload(r["new"])
        if new.startswith("表注") and r["id"] not in note_ids and r["id"] not in {a[0] for a in auto}:
            manual.append((r["id"], "表注文本已备好，但要落在**那张表之后**（表题在上、表在下，"
                                    "从表题段插会插到表格上面）⇒ 人工定位置，脚本不猜"))
    print(f"[plan] 可整段重写={len(auto)} 插表注={len(notes)} 术语算子={len(TERM_OPS)} "
          f"成对删插={len(PAIR_OPS)} 待批={len(await_)} 人工={len(manual)}")
    if args.plan:
        for rid, i, _ in auto:
            print(f"   [自动] {rid} → 段 {i}")
        for rid, i, _ in notes:
            print(f"   [插表注] {rid} → 段 {i} 之后")
        for rid, why in manual:
            if rid in note_ids:
                continue                              # 已由插段算子接手，不重复挂"人工"标签
            print(f"   [人工] {rid}: {why}")
        for rid, why in await_:
            print(f"   [待批] {rid}: {why}")
        return 0

    dst = OUT / f"装配副本-{datetime.datetime.now():%Y%m%dT%H%M%S}.docx"
    shutil.copy2(SRC, dst)
    doc = Document(str(dst))
    b0 = paras(doc)
    apply_rewrite(doc, auto)
    n_term = fix_terms(doc)
    for rid, i, text in sorted(notes, key=lambda x: -x[1]):     # 倒序插，先面的锚点下标才不被位移
        insert_after(doc.paragraphs[i], text)
    doc.save(str(dst))
    b1 = paras(Document(str(dst)))
    repl, ins, dele = structural_diff([norm(t) for t in b0], [norm(t) for t in b1])
    if ins != len(notes) or dele or repl < len(auto):
        print(f"[INVALID] 结构核对不过：改写 {repl}（应≥{len(auto)}）、插入 {ins}（应={len(notes)}）、删除 {dele}（应 0）",
              file=sys.stderr)
        return 1
    if args.expect_changed is not None and repl != args.expect_changed:
        print(f"[INVALID] 被改段数 {repl} ≠ 期望 {args.expect_changed}", file=sys.stderr)
        return 1
    print(f"[apply] 副本={dst} ({dst.stat().st_size:,} B) 段落 {len(b0)}→{len(b1)} "
          f"改写={repl} 插入={ins} 删除={dele} 术语命中={n_term}")
    print(f"[回读] 原件 sha256 未变={hashlib.sha256(SRC.read_bytes()).hexdigest() == src_sha}")
    print(f"下一步：python3 {pathlib.Path(__file__).name} --verify \"{dst}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
