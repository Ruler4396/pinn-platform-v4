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
PAIR_OPS = ("E2", "E3", "E4")   # ③ 换数+插段+删句三者同进同退，见 e_block()


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
    if not is_body_prose(new):
        return False, "", "载荷是给表格加列/加行/写表注那类指令，不是正文句 ⇒ 走插表/改列算子"
    lost = sorted({x.replace(" ", "") for x in re.findall(r"[图表]\s*\d+(?:-\d+)?", old_text)} -
                  {x.replace(" ", "") for x in re.findall(r"[图表]\s*\d+(?:-\d+)?", new)})
    if lost:
        return False, "", (f"替换会丢掉 {lost} 的正文引用（原件里它是该图/表唯一的引用点时会变成孤图）"
                           f"⇒ 先在新文本里补回图/表号再落")
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


TABLE_WORDS = ("说明列", "新增行", "表注", "表题", "列名", "行名", "增设", "逐行填实", "改为")


def is_body_prose(new: str) -> bool:
    """载荷得是一句正文，不能是"给表格加列/加行/写表注"那类指令性内容。"""
    return not any(k in new[:24] for k in TABLE_WORDS) and not new.startswith("表注")


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
        if not hit_list and cited:                        # 定位器二：按 :NNNN 反查正文段，但只在"本行处置写明确实是替换"时才允许写
            cand = sorted({b for ln in cited for b in line_to_body.get(ln, [])})
            act = r["act"]
            repl_like = re.search(r"换数|改写|替换|降级|撤|换词|换句", act) and not re.search(r"新增|插", act)
            if len(cand) == 1 and repl_like:
                hit_list = cand
                print(f"   [定位器二→自动] {rid} 处置={act[:14]} 认定替换正文段 {cand[0]}")
            elif cand:
                manual.append((rid, f"无摘引；按 :{sorted(cited)[:3]} 对应正文段 {cand[:3]}"
                                     f"（处置不是替换类{'' if repl_like else '，或本行是新增/插段'}"
                                     f"⇒ 只作定位线索）"))
                continue
            else:
                manual.append((rid, ":NNNN 不对应任何正文段（新增表/表位/仓库文件类）"))
                continue
        if not hit_list:
            manual.append((rid, "无摘引且 :NNNN 不对应正文段（新增表/表位/仓库文件类）"
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
        if rid == "E4":                                    # 成对块里的删句：判据是"旧句不在副本里了"
            q = [norm(x) for x in re.findall(r"「(.+?)」", r["loc"]) if len(norm(x)) >= 10]
            gone = q and not any(x and x in joined for x in q)
            (ok if gone else miss).append((rid, "旧结论句已删除 ✓" if gone else "旧结论句仍在 ⇒ 成对块没做完"))
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


def e_block(doc, auto):
    """5.7 效率节那一块的成对算子：E2 换数 → E3 在 E2 之后插段 → E4 删旧结论句。
    **三者同进同退**：只删 E4 会让 5.7 失去结论句，只插 E3 会与新数并排留着旧四组数。"""
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
    by_id = {r["id"]: r for r in rows()}

    def num_tokens(s):
        return {t for t in re.findall(r"\d+\.\d+|\d+", s) if len(t) >= 3}

    cand = []
    r2 = by_id.get("E2")
    if r2:
        cited = {int(n) for n in re.findall(r":(\d{2,4})", r2["loc"])}
        want = num_tokens(r2["loc"])
        pool = sorted({b for ln in cited for b in line_to_body.get(ln, [])})
        scored = [(len(want & num_tokens(body[i])), i) for i in pool]     # 旧数命中数
        scored = sorted([(s, i) for s, i in scored if s >= 2], reverse=True)
        if len(scored) == 1:
            cand.append(("E2", scored[0][1], payload(r2["new"])))
        elif len(scored) >= 2 and scored[0][0] > scored[1][0]:
            # E2 的 :1193 与 :1195 是两个段（前者是四组旧数，后者是结论句所在）⇒ 取旧数命中更多的那一段，
            # 另一段由 E4 删除，两块不重叠才不会把同一句既换又删
            print(f"   [E2 择一] 命中数 {scored[:2]} ⇒ 取段 {scored[0][1]}（另一段归 E4 删）")
            cand.append(("E2", scored[0][1], payload(r2["new"])))
        elif scored:
            print(f"   [E2 不唯一] 候选 {scored} ⇒ 整块不推")
            return None
    r3, r4 = by_id.get("E3"), by_id.get("E4")
    if not (cand and r3 and r4):
        return None
    q4 = [q for q in re.findall(r"「(.+?)」", r4["loc"]) if norm(q) not in ("", "……")]
    if not q4:
        return None
    frag = norm(max(q4, key=lambda x: len(norm(x))))
    hits4 = [i for i, t in enumerate(body_norm) if frag[:40] and frag[:40] in t]
    if len(hits4) != 1:
        print(f"   [E4 不唯一] 旧结论句命中 {hits4} ⇒ 整块不推")
        return None
    new3 = payload(r3["new"])
    if len(new3) < 80 or any(k in new3 for k in INSTR):
        print("   [E3 载荷不合格] 太短或含工单内部注 ⇒ 整块不推")
        return None
    # 删段的硬闸：被删段里提到的每一个图/表号，删完之后正文里还必须至少还剩一处引用
    doomed_text = body[hits4[0]]
    orphans = []
    for tag in sorted(set(re.findall(r"[图表]\s*\d+(?:-\d+)?", doomed_text))):
        t = norm(tag)
        # 题注段自己不算引用：只数"不以该号开头"的段落，删完才不会留下没人引用的孤图
        others = sum(x.count(t) for i, x in enumerate(body_norm)
                     if i != hits4[0] and not body[i].strip().startswith(tag.replace(" ", "")))
        if others == 0:
            orphans.append(tag)
    if orphans:
        print(f"   [E4 不能删] 被删段是 {orphans} 在正文里唯一的引用点 ⇒ 删完这些图/表就没人引用了。"
              f"要么先补一句带图号的过渡句（措辞归作者/统括官），要么整块不推")
        return None
    return {"rewrite": cand, "insert_after": cand[0][1], "insert_text": new3, "delete": hits4[0],
            "delete_text": body[hits4[0]]}


NEW_TABLES = {
    "表4-4b": {
        "anchor": ("after-table", "表4-4"),
        "caption": "表4-4b  严格稀疏观测批相对表 4-4 另置零的损失权重项",
        "header": ["权重项", "该批取值", "说明"],
        "rows": [["入口流量损失权重 inlet_flux_weight", "0.0", "在表 4-4 基础上另置零"],
                 ["出口压力损失权重 outlet_pressure_weight", "0.0", "同上"],
                 ["压降损失权重 pressure_drop_weight", "0.0", "同上"],
                 ["壁面损失权重 wall_weight", "0.0", "同上"]],
        "note": ("适用范围声明：表 4-4 与表 4-4b 分属两批训练，其权重不可互相代入；"
                 "表 4-4b 的四项为整批共同配置，不是双模型与单网络之间的差异项。"),
    },
    "表5-9": {
        "anchor": ("before-para", "5.8 PDE约束与双模型耦合作用分析"),
        "caption": "表5-9  PINN 与 CFD 的单次成本、训练入账与盈亏平衡工况数（混合口径）",
        "from_markdown": "**表 5-9（",   # 必须钉到标题行：只写"表 5-9"会先命中 §0 里提到这四个字的那一行
        "note": ("本表 A、C 两列取自 2026-04-20 旧主机（`iZ7xv19l7qsogyq3hzyhydZ`）的一次计时，"
                 "B 列取自本轮 8 核实例的五种子实测中位 ⇒ A、C 与 B 不同机、不同次，为混合口径；"
                 "同机补测（E5）本轮未做，故本限定不可删。K*=B/(A−C) 逐格向上取整，单价一换必整列重算。"),
    },
    "表5-10": {
        "anchor": ("before-para", "5.9 本章小结"),   # 不能用"但当前模型仍有部分不足"：那段正是 B2 的落字目标，落字后原文已不在
        "caption": "表5-10  三臂与零训练基线的对照（test / mean-of-cases / rel_l2_speed）",
        "header": ["方法与口径", "稠密", "分层 5%", "n", "备注"],
        "rows": [["双模型（本文）", "0.028617 ± 0.002162", "0.031202 ± 0.004515", "5", "均值与配对差同为 obs_seed=0 五粒"],
                 ["臂A 单网络联合 PINN", "0.039161 ± 0.009313", "0.023144 ± 0.004258", "5", "稠密档实测更慢：139.6 s 对 81.4 s"],
                 ["臂B 纯数据 MLP（无物理）", "0.019436 ± 0.006923", "0.051029 ± 0.008870", "5", "训练墙钟区间 120.9–212.9 s；pooled 口径无配对"],
                 ["臂C POD+观测点最小二乘", "—（本次只跑收缩族）",
                  "C-val 0.828487；C-test-1 0.531695；C-test-2 0.611451", "单次读数",
                  "零训练；阶数由 99.97% 能量判据算出（r=1），非人工挑选；敏感性检验未做"]],
        "note": ("① 全表 n=5，Wilcoxon 最小可达双侧 p=0.0625 ⇒ 只报符号与幅度，不写显著性；② 未做多重比较校正；"
                 "③ 臂C 为零训练、与三臂不同机时口径，其墙钟不可与本表 PINN 行直比；"
                 "④ 同一行并排的均值与配对差来自同一 obs_seed 子集；⑤ 本表只有 mean_of_cases 一个口径"
                 "（臂B 的评估件顶层无 global_metrics ⇒ pooled 键数为 0）。"),
    },
}


def md_block(md_text: str, title_marker: str):
    """从工单里把某张表的 markdown 块抠成 (header, rows)。"""
    lines = md_text.splitlines()
    start = next((i for i, l in enumerate(lines) if title_marker in l), None)
    if start is None:
        return None, None
    tbl = []
    for l in lines[start:]:                        # 只吃这一块：碰到第一行非表格就停（否则会吞掉后文所有表）
        if l.strip().startswith("|"):
            tbl.append(l.strip())
        elif tbl:
            break
    rows, header = [], None
    for l in tbl:
        cells = [c.strip() for c in l.replace("\\|", "@@PIPE@@").strip("|").split("|")]   # 转义竖线不当分隔（与 check_md_tables 同法）
        cells = [c.replace("@@PIPE@@", "|") for c in cells]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        if header is None:
            header = cells
        else:
            if len(cells) != len(header):
                print(f"   [截块 {title_marker}] 第 {len(rows) + 1} 行格数 {len(cells)} ≠ 表头 {len(header)} ⇒ 到此为止，后面的行不归这张表")
                break
            rows.append(cells)
    return header, rows


def insert_tables(copy_path):
    """③ 新建表框：表 4-4b / 5-9 / 5-10。逐张插到锚点后，并回读结构差。"""
    from docx import Document
    import copy as _copy
    from docx.oxml.ns import qn
    doc = Document(str(copy_path))
    n_tbl0, n_par0 = len(doc.tables), len(doc.paragraphs)
    md = WORK.read_text(encoding="utf-8")
    made = []
    for name, spec in NEW_TABLES.items():
        header, rows = spec.get("header"), spec.get("rows")
        if spec.get("from_markdown"):
            header, rows = md_block(md, spec["from_markdown"])
            if not header:
                print(f"   [跳过 {name}] 工单里找不到该 markdown 块")
                continue
        width = len(header)
        rows = [r + [""] * (width - len(r)) for r in rows]
        kind, key = spec["anchor"]
        if kind == "before-para":
            tgt = next((p for p in doc.paragraphs if p.text.strip().startswith(key)), None)
        else:
            tag = ("表" + key.replace("表", "")).strip()
            cap_idx = next((i for i, p in enumerate(doc.paragraphs)
                            if p.text.strip().startswith(tag) and not p.text.strip().startswith(tag + "b")), None)
            tgt = None
            if cap_idx is not None:
                el = doc.paragraphs[cap_idx]._p
                while el is not None:
                    el = el.getnext()
                    if el is not None and el.tag == qn("w:tbl"):
                        tgt = el
                        break
        if tgt is None:
            print(f"   [跳过 {name}] 锚点段落找不到：{key}"
                  f"（若该段本身是某行的落字目标，落字后原文就不在了 ⇒ 换用节标题当锚点）")
            continue
        cap = doc.add_paragraph(spec["caption"])
        tab = doc.add_table(rows=len(rows) + 1, cols=width)
        try:
            tab.style = "Table Grid"
        except KeyError:
            pass
        for j, h in enumerate(header):
            tab.cell(0, j).text = str(h)
        for i, r in enumerate(rows, start=1):
            for j in range(width):
                tab.cell(i, j).text = clean(str(r[j]))
        note = doc.add_paragraph(spec["note"])
        # 把刚建的三段搬到锚点前/后（add_* 只会追加到文末）
        els = [cap._p, tab._tbl, note._p]
        ref = tgt._p if hasattr(tgt, "_p") else tgt        # before-para 给的是 Paragraph，after-table 给的是 w:tbl 元素
        if kind == "before-para":                          # 插在锚点段之前 ⇒ 正序 addprevious
            for e in els:
                e.getparent().remove(e)
                ref.addprevious(e)
        else:                                              # 插在锚点表之后 ⇒ 反序 addnext 才不互相顶位
            for e in reversed(els):
                e.getparent().remove(e)
                ref.addnext(e)
        made.append((name, len(rows), width))
    doc.save(str(copy_path))
    after = Document(str(copy_path))
    print("[tables] 新建表框 " + str(len(made)) + " 张：" +
          "；".join(f"{n}({r}行×{c}列)" for n, r, c in made))
    print(f"[tables] 表数 {n_tbl0}→{len(after.tables)}  段落 {n_par0}→{len(after.paragraphs)}")
    if len(after.tables) != n_tbl0 + len(made):
        print("[INVALID] 表数增量对不上 ⇒ 别交")
        return 0
    same = all(a == b for a, b in zip(tgt_texts(Document(str(copy_path))), tgt_texts(Document(str(copy_path)))))
    return len(made)


def tgt_texts(doc):
    return [c.text for t in doc.tables for r in t.rows for c in r.cells]


def _unused_main() -> int:
    return made


T57_BATCH = {                       # 只填仓内说得出处的；说不出来就写"需人工"，不手填
    "B-test-1__ip_blunted": "bend_strict_blunted_sparse5_stagepde_20260503（削平入口·5% 稀疏）｜来源：工单 G4/I4",
    "B-test-2": "bend_independent_geometry_notemplate_parabolic_mainline_v1_20260419｜来源：一手评估件 "
                 "model/results/pinn/bend_independent_geometry_notemplate_parabolic_mainline_v1_20260419/evaluations/metrics_test.json",
    "C-test-1": "需人工（仓内未指明 C 行的模型批次；只知走 `evaluations/metrics_test_dense.json` 的 `case_metrics[0]`）",
    "C-test-2": "需人工（同上，`case_metrics[1]`）",
}


def update_t57(copy_path):
    """表 5-7：加「模型批次」列 + 追加 B-test-2 行，四格数字**从仓内一手评估件现取**（不手填）。"""
    import json
    from docx import Document
    from docx.shared import Pt
    doc = Document(str(copy_path))
    tgt = next((t for t in doc.tables if t.rows[0].cells[0].text.strip() == "工况"
                and any(c.text.strip() == "B-test-1__ip_blunted" for c in t.rows[-1].cells)
                and len(t.columns) == 6), None)
    if tgt is None:
        print("[t57] 没定位到表 5-7（工况/观测条件/5%稀疏 B 行三特征要同时命中）⇒ 不动")
        return 0
    jp = REPO / "model/results/pinn/bend_independent_geometry_notemplate_parabolic_mainline_v1_20260419/evaluations/metrics_test.json"
    d = json.loads(jp.read_text(encoding="utf-8"))
    row2 = next(c for c in d["case_metrics"] if c.get("case_id") == "B-test-2")
    vals = [row2["rel_l2_speed"], row2["rel_l2_p"], row2["pressure_drop_rel_error"]]
    print(f"[t57] 现取 B-test-2：speed={vals[0]:.6f} p={vals[1]:.6f} 压降={vals[2]:.6f}"
          f"（split={d['split_name']}、eval_source={d['eval_source']}，件 sha256 前缀 "
          f"{hashlib.sha256(jp.read_bytes()).hexdigest()[:12]}）")
    tgt.add_column(Pt(90))
    last = len(tgt.columns) - 1
    tgt.cell(0, last).text = "模型批次"
    for r in range(1, len(tgt.rows)):
        key = tgt.cell(r, 0).text.strip()
        tgt.cell(r, last).text = T57_BATCH.get(key, "需人工")
    new = tgt.add_row()
    for i, v in enumerate(["B-test-2", "弯曲流道", "dense（θ=60°，转角外推）",
                           f"{vals[0]:.4f}", f"{vals[1]:.4f}", f"{vals[2]:.4f}"]):
        new.cells[i].text = v
    new.cells[last].text = T57_BATCH["B-test-2"]
    for c in new.cells:
        for p in c.paragraphs:
            for r in p.runs:
                r.font.size = Pt(9)
    doc.save(str(copy_path))
    after = Document(str(copy_path))
    t2 = next(t for t in after.tables if t.rows[0].cells[0].text.strip() == "工况"
              and len(t.columns) == 7)
    print(f"[t57] 表 5-7 现 {len(t2.rows)} 行 × {len(t2.columns)} 列（改前 4×6）")
    return len(t2.rows)


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
    g.add_argument("--tables", type=pathlib.Path, help="在给定副本上插三张新表（表4-4b/5-9/5-10）")
    g.add_argument("--t57", type=pathlib.Path, help="在给定副本上改表 5-7：加「模型批次」列 + 追加 B-test-2 行（数字现取）")
    ap.add_argument("--expect-changed", type=int, default=None,
                    help="--apply 用：期望被改段落数，不接等即 INVALID（闸三）")
    args = ap.parse_args()
    import docx  # noqa: F401  ② 先确认库在，不在就别硬写
    from docx import Document

    if args.verify:
        return verify(args.verify)
    if args.t57:
        if not args.t57.exists():
            print(f"[INVALID] 副本不存在：{args.t57}", file=sys.stderr)
            return 2
        return 0 if update_t57(args.t57) >= 5 else 1

    if args.tables:
        if not args.tables.exists():
            print(f"[INVALID] 副本不存在：{args.tables}", file=sys.stderr)
            return 2
        n = insert_tables(args.tables)
        return 0 if n == len(NEW_TABLES) else 1

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
    eblk = e_block(Document(str(SRC)), auto)
    note_ids = {n[0] for n in notes}
    for r in rows():                                   # 表注类单独点名：载荷现成，只差一个正确的插入位置
        new = payload(r["new"])
        if new.startswith("表注") and r["id"] not in note_ids and r["id"] not in {a[0] for a in auto}:
            manual.append((r["id"], "表注文本已备好，但要落在**那张表之后**（表题在上、表在下，"
                                    "从表题段插会插到表格上面）⇒ 人工定位置，脚本不猜"))
    print(f"[plan] 可整段重写={len(auto)} 插表注={len(notes)} 术语算子={len(TERM_OPS)} "
          f"5.7 成对块={'E2+E3+E4 齐' if eblk else '不齐 ⇒ 整块不推'} 待批={len(await_)} 人工={len(manual)}")
    if args.plan:
        for rid, i, _ in auto:
            print(f"   [自动] {rid} → 段 {i}")
        for rid, i, _ in notes:
            print(f"   [插表注] {rid} → 段 {i} 之后")
        if eblk:
            print(f"   [5.7 成对块] E2 换数→段 {eblk['rewrite'][0][1]}；E3 插在其后；E4 删段 {eblk['delete']}"
                  f"（被删原文：{eblk['delete_text'][:40]}…）")
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
    n_del = 0
    if eblk:
        # 锚点下标是在**原件**上算的；前面每次插表注都会让后面的下标 +1 ⇒ 先补位移再动手
        shift = sum(1 for _, i, _ in notes if i < min(eblk["rewrite"][0][1], eblk["delete"]))
        ei = eblk["rewrite"][0][1] + shift
        erid, _, etext = eblk["rewrite"][0]
        p2 = doc.paragraphs[ei]
        p2.runs[0].text = etext
        for r in p2.runs[1:]:
            r.text = ""
        insert_after(doc.paragraphs[ei], eblk["insert_text"])        # E3 紧跟 E2
        d_idx = eblk["delete"] + shift + (1 if eblk["delete"] > eblk["rewrite"][0][1] else 0)
        doomed = doc.paragraphs[d_idx]
        doomed._element.getparent().remove(doomed._element)          # E4 删除，原文留证
        n_del = 1
    doc.save(str(dst))
    b1 = paras(Document(str(dst)))
    repl, ins, dele = structural_diff([norm(t) for t in b0], [norm(t) for t in b1])
    want_repl = len(auto) + (1 if eblk else 0)
    want_ins = len(notes) + (1 if eblk else 0)
    if ins != want_ins or dele != n_del or repl < want_repl:
        print(f"[INVALID] 结构核对不过：改写 {repl}（应≥{want_repl}）、插入 {ins}（应={want_ins}）、"
              f"删除 {dele}（应={n_del}）", file=sys.stderr)
        return 1
    if args.expect_changed is not None and repl != args.expect_changed:
        print(f"[INVALID] 被改段数 {repl} ≠ 期望 {args.expect_changed}", file=sys.stderr)
        return 1
    if eblk:
        drop = dst.parent / f"被删原文-{dst.stem}.txt"
        drop.write_text(eblk["delete_text"] + "\n", encoding="utf-8")
        print(f"[留证] E4 被删的原文写入 {drop}")
    print(f"[apply] 副本={dst} ({dst.stat().st_size:,} B) 段落 {len(b0)}→{len(b1)} "
          f"改写={repl} 插入={ins} 删除={dele} 术语命中={n_term}")
    print(f"[回读] 原件 sha256 未变={hashlib.sha256(SRC.read_bytes()).hexdigest() == src_sha}")
    print(f"下一步：python3 {pathlib.Path(__file__).name} --verify \"{dst}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
