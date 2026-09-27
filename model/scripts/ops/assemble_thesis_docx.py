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


def is_caption(text: str) -> bool:
    """唯一的题注判据（统括官 12:4x 指出我此前有两把尺：粗判"以号开头"会把"图5-15中，…""表5-8显示，…"
    这类正文句误归为题注 ⇒ 既造出假告警，又会在反向放行真删）。题注 = 以号开头 **且** 无句末标点 **且** 短。"""
    s = text.strip()
    return bool(re.match(r"^(表|图)\s*\d", s)) and "。" not in s and len(s) < 60


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
                     if i != hits4[0] and not is_caption(body[i]))          # 共用同一把题注判据
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
        "sources": ["config.json `run_strict_sparse_experiments.sh` 批次；工单 F1 行",
                    "同上（`outlet_pressure_weight`）", "同上（`pressure_drop_weight`）", "同上（`wall_weight`）"],
        "note": ("适用范围声明：表 4-4 与表 4-4b 分属两批训练，其权重不可互相代入；"
                 "表 4-4b 的四项为整批共同配置，不是双模型与单网络之间的差异项。"),
    },
    "表5-9": {
        "anchor": ("before-para", "5.8 PDE约束与双模型耦合作用分析"),
        "caption": "表5-9  PINN 与 CFD 的单次成本、训练入账与盈亏平衡工况数（混合口径）",
        "sources_md": {           # 表 5-9 每一行的来源（行号取自当前工单/读数，写前先核件在不在）
            "CFD 单工况": "docs/benchmarks/pinn_vs_cfd_speed_benchmark_20260420.md:14-17（旧主机 `iZ7xv…`，`metadata.cfd_runs=3`）",
            "PINN 单工况推理": "同上 md:14-15（`metadata.pinn_runs=7`）",
            "PINN 稀疏观测在线重建": "同上 md:15（同上）",
            "PINN 一次训练": "docs/revision/T5矩阵配对读数-20260925.md 表下注 + `T5矩阵run坐标索引-20260926.psv` 的 wall_ms 列",
            "盈亏平衡": "本表按 K 星=⌈B/(A−C)⌉ 现算（工单 §12 第 3 条规则）",
            "K→∞ 渐近加速比": "benchmark JSON `comparison` 原值 6.042225/2.193853/30.723651/9.335324",
            "臂 A": "docs/revision/SWEEP_DRYRUN.md:339 与 368（五粒中位与区间）",
            "臂 B": "docs/revision/SWEEP_DRYRUN.md:339（区间 120.9–212.9 s ⇒ 只引区间）",
            "评估单价": "`T5矩阵run坐标索引-20260926.psv` 的 eval 行（1.533 s）"},
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
        "sources": ["工单 B5/D2 行；`T5矩阵test口径读数-20260926.md:79` 与 §四B 配对表 :85-88",
                    "同上 :78-79；臂A 机时 `SWEEP_DRYRUN.md:339`",
                    "同上 :79；臂B 账本整列 NA ⇒ 只能取 analyze（§11.5）",
                    "E0 一手产物 `dsw-2213486:…/pod_baseline_contraction_a0d1375.json`（2,948 B、`da0dba2fef66318c`）；本表三数取 09-26 同框那次，见工单 D2"],
        "note": ("表 5-10 的读法（五条缺一不可）：① 全表 n=5，Wilcoxon 最小可达双侧 p=0.0625 ⇒ 只报符号与幅度，不写显著性；② 未做多重比较校正；"
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
        srcs = spec.get("sources") or []
        if spec.get("sources_md"):
            def pick(cell):
                for k, v in spec["sources_md"].items():
                    if k in cell:
                        return v
                return "需人工（该行来源未登记）"
            srcs = [pick(r[0]) for r in rows]
        if srcs:
            header = header + ["来源件 + 行号"]
            rows = [r + [s] for r, s in zip(rows, srcs)]
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


TRANSITION_517 = "图 5-17 展示了按上一节绝对耗时换算得到的相对加速倍数。"
# 统括官 9/27 12:4x 授权的四条约束：同块同进同退／不得引入新数字或新比较对象／强度评价词全禁／
# 句式只准"图 5-17 展示 <已核对象>"。旧句里的"十分显著"与被换掉的四个旧倍数一起在此消失，图号仍在。
FORBID_IN_TRANSITION = ("十分", "显著", "大幅", "明显", "优异", "领先", "远远", "4.34", "1.98", "30.44", "9.01")
# 注：不锁"倍"字——图 5-17 的本名就叫"PINN相对CFD的加速倍数对比图"，锁了会把图名本身判违规（我先就栽在这儿）


def pair57(copy_path):
    """5.7 成对块：E2 换数 → E3 插 K* 段 → E4 的结论句改成受约束的过渡句（图5-17 引用保住）。"""
    from docx import Document
    doc = Document(str(copy_path))
    body = paras(doc)
    bn = [norm(x) for x in body]
    if any(k in TRANSITION_517 for k in FORBID_IN_TRANSITION) or re.search(r"\d", TRANSITION_517[TRANSITION_517.find("展示") if "展示" in TRANSITION_517 else 0:]):
        print(f"[INVALID] 过渡句含被禁强度词/旧倍数/任何数字（{TRANSITION_517}）⇒ 不推")
        return 0
    rows = {r["id"]: r for r in rows_iter()}
    i2 = next((i for i, x in enumerate(bn) if norm("0.111") in x and norm("0.394") in x), None)
    i4 = next((i for i, x in enumerate(bn) if norm("十分显著") in x), None)
    if i2 is None or i4 is None:
        print(f"[pair57] 锚点缺失：E2 段 {i2}、E4 段 {i4} ⇒ 整块不推")
        return 0
    new2, new3 = payload(rows["E2"]["new"]), payload(rows["E3"]["new"])
    for lo, hi in ((i2, i4), (i4, i2)):
        pass
    p2, p4 = doc.paragraphs[i2], doc.paragraphs[i4]
    p2.runs[0].text = new2
    for r in p2.runs[1:]:
        r.text = ""
    p4.runs[0].text = TRANSITION_517
    for r in p4.runs[1:]:
        r.text = ""
    insert_after(p2, new3)                                   # E3 紧跟 E2
    doc.save(str(copy_path))
    aft = Document(str(copy_path))
    ap = paras(aft)
    ref517 = sum(1 for x in ap if "图5-17" in x.replace(" ", "") and not is_caption(x))
    print(f"[pair57] 三处同进同退完成：E2→段{i2} 换数、E3 插在其后、E4→段{i4} 改为受约束过渡句；"
          f"段落 {len(body)}→{len(ap)}；图5-17 正文引用数（排除题注）= {ref517}（应 ≥1）")
    return 1 if ref517 >= 1 else 0


def rows_iter():
    return rows()


def selfcheck_caption():
    """两类夹具（统括官 12:4x 要求）：真正文句必须算正文、真题注必须算题注。任一不满足 ⇒ 尺不可信。"""
    # 夹具逐字取自原件：正文句 = 段340/352/353，题注 = 段139/336/344/351/355
    prose = ["图5-15中，弯道转角附近的速度高值区域、近壁低速带和整体流动路径与参考真值基本一致。结合表5-7中的各项指标，降低观测成本并不必然导致重建失效。",
             "表5-8显示，阶段内PDE约束对速度场最终误差影响较小，对压力场影响更明显。不启用阶段内PDE约束时，最终速度场L2误差为0.022100。",
             "图5-18比较5%稀疏监督下的单网络MLP和双模型PDE耦合。双模型的速度Rel-L2由4.11%降至3.21%，壁面速度残余由23.33%降至0.01%以下。"]
    caps = ["图5-16  PINN与CFD在同一环境下的中位耗时对比图",
            "表5-8  阶段内PDE约束对模型最终性能的影响",
            "图5-14  收缩流道几何增强编码消融结果",
            "表3-4  数据预处理流程",
            "图5-18  单网络基线与双模型耦合效果对照图"]
    bad = [x for x in prose if is_caption(x)] + [x for x in caps if not is_caption(x)]
    print(f"[题注夹具] 正文句 {len(prose)} 条、题注 {len(caps)} 条 ⇒ 判错 {len(bad)} 条"
          + ("（**尺不可信，先修尺再谈装配**）" if bad else "，全对 ✓"))
    for x in bad:
        print("   判错：", x[:60])
    return 1 if bad else 0


# ==================== 图 5-14 / 5-16 / 5-17：数从仓内正本现取，重画后替换副本内图片 ====================
# 为什么必须重画：这三张图里印着的是被 §10 禁掉的旧数（0.561/11.760 与 0.111/0.242/0.481、4.34/1.98/30.44/9.01），
# 正文换数之后图与文不一致 ⇒ 只改文字等于交付一份自相矛盾的稿子。
BENCH = REPO / "docs" / "benchmarks" / "pinn_vs_cfd_speed_benchmark_20260420.json"
T5MD = REPO / "docs" / "revision" / "T5矩阵test口径读数-20260926.md"
FIG_IDS = {"图5-14": "fig514", "图5-16": "fig516", "图5-17": "fig517"}
# 图5-14 的指向句（原件里它是"只有题注、正文零引用"的孤图）。授权口径与 TRANSITION_517 同一把：
# 只把已有对象指过去，不带新数字、不带强度词、不写结论；句子本身仍要作者点头。
FIG514_POINTER = "图 5-14 展示了 basic 与 geometry 两种输入特征集在收缩流道上的速度 Rel-L2 对照。"


def fig_data():
    """三张图的数一律现取：图5-14 = test 口径正本里 t5c13/t5c14 两行；图5-16/5-17 = 同一枚计时 JSON
    的中位数与它自带的 comparison 字段（倍数不自己相除，免得又造一把尺）。"""
    import json
    if not BENCH.exists() or not T5MD.exists():
        print(f"[图] 正本缺件：BENCH={BENCH.exists()} T5MD={T5MD.exists()} ⇒ 不重画")
        return None
    ab = {}
    for ln in T5MD.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(t5c1[34])\s*\|(.+?)\|\s*(\d+)\s*\|\s*\*\*([\d.]+)\s*±\s*([\d.]+)\*\*", ln)
        if m:
            ab[m.group(1)] = {"label": m.group(2).strip(), "n": int(m.group(3)),
                              "mean": float(m.group(4)), "sd": float(m.group(5))}
    if len(ab) != 2:
        print(f"[图] 从 {T5MD.name} 现取 t5c13/t5c14 只拿到 {len(ab)} 行 ⇒ 图5-14 不重画（不硬编数字）")
        return None
    b = json.loads(BENCH.read_text(encoding="utf-8"))
    return {"ab": ab, "med": {k: v["median_s"] for k, v in b["benchmarks"].items()},
            "runs": {k: v.get("runs") for k, v in b["benchmarks"].items()},
            "sp": b["comparison"], "meta": b["metadata"],
            "src": {k: (f"{T5MD.name} 第二节 t5c13/t5c14" if k == "fig514"
                        else f"{BENCH.name}（benchmarks[*].median_s / comparison）") for k in FIG_IDS.values()}}


def render_figures(data, out_dir):
    """画三张 PNG 到仓外 out_dir；像素尺寸对齐原件（1277×781 / 1364×807 / 1373×843），换图后版式不跳。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    made = {}

    def save(fig, key, name):
        p = out_dir / name
        fig.savefig(str(p), dpi=100)
        plt.close(fig)
        made[key] = p
        print(f"   [画] {name} {p.stat().st_size:,} B")

    def footnote(fig, ax, text):
        """注放在坐标区下方（图内会压到柱子），并给底部留位。"""
        assert "⇒" not in text and "×" not in text, "中文字体没有这两个字形，图上会出豆腐块 ⇒ 换文字表达"
        fig.subplots_adjust(left=.10, right=.97, top=.90, bottom=.20)
        fig.text(.01, .02, text, fontsize=9, color="#555")

    a = data["ab"]
    fig, ax = plt.subplots(figsize=(12.77, 7.81))
    names = [a["t5c13"]["label"], a["t5c14"]["label"]]
    vals = [a["t5c13"]["mean"], a["t5c14"]["mean"]]
    errs = [a["t5c13"]["sd"], a["t5c14"]["sd"]]
    ax.bar(names, vals, yerr=errs, capsize=8, width=.45, color=["#F0A020", "#17726A"])
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v * 1.35, f"{v:.6f}±{e:.6f}", ha="center", fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(top=vals[0] * 4)
    ax.set_ylabel("速度 Rel-L2（test / mean-of-cases）")
    ax.set_title(f"收缩流道几何增强编码消融（各 {a['t5c13']['n']} 种子，mean±sd，ddof=1；比值 {vals[0] / vals[1]:.1f}）")
    footnote(fig, ax, "压力面板已撤：仓内正本没有与本次消融同源的配对压力数"
                      "（旧图 11.7598/0.1516 出自同时改特征集与壁面模式的两次运行，不能单独归因于几何增强编码）")
    save(fig, "fig514", "重画图5-14.png")

    m, r = data["med"], data["runs"]
    fig, ax = plt.subplots(figsize=(13.64, 8.07))
    groups = ["收缩流道", "弯曲流道"]
    series = [("PINN完整推理", ["contraction_full_inference", "bend_full_inference"], "#4C72B0"),
              ("PINN稀疏重建", ["contraction_sparse_reconstruction", "bend_sparse_reconstruction"], "#72B2AC"),
              ("CFD求解", ["contraction_cfd", "bend_cfd"], "#B8A9A4")]
    w = .26
    for j, (lab, keys, col) in enumerate(series):
        xs = [i + (j - 1) * w for i in range(2)]
        vv = [m[k] for k in keys]
        ax.bar(xs, vv, width=w, label=lab, color=col)
        for x, v, k in zip(xs, vv, keys):
            ax.text(x, v * 1.09, f"{v:.3f}s (n={r[k]})", ha="center", fontsize=10)
    ax.set_xticks(range(2))
    ax.set_xticklabels(groups)
    ax.set_yscale("log")
    ax.set_ylim(top=max(m.values()) * 3)
    ax.set_ylabel("中位耗时（秒，对数坐标）")
    ax.set_title("同一环境下 PINN 与 CFD 中位耗时对比")
    ax.legend(loc="upper left")
    footnote(fig, ax, f"来源 {data['meta']['created_at'][:10]} 主机 {data['meta']['hostname']}：PINN n=7、CFD n=3 中位数；"
                      "CFD 列为该旧主机值（同机单价 E5 未做），本图是混合口径，不得写成同机同次")
    save(fig, "fig516", "重画图5-16.png")

    sp = data["sp"]
    labels = ["收缩完整推理", "收缩稀疏重建", "弯曲完整推理", "弯曲稀疏重建"]
    keys = ["contraction_full_vs_cfd_speedup", "contraction_sparse_vs_cfd_speedup",
            "bend_full_vs_cfd_speedup", "bend_sparse_vs_cfd_speedup"]
    fig, ax = plt.subplots(figsize=(13.73, 8.43))
    vals = [sp[k] for k in keys]
    cols = ["#4C72B0", "#72B2AC", "#4C72B0", "#72B2AC"]
    ax.barh(range(4)[::-1], vals, color=cols, height=.55)
    for y, v in zip(range(4)[::-1], vals):
        ax.text(v + max(vals) * .015, y, f"{v:.2f}x", va="center", fontsize=12)
    ax.set_yticks(range(4)[::-1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("CFD / PINN 加速倍数")
    ax.set_title("相对加速倍数（由上一节同一枚计时产物的 comparison 字段现取）")
    ax.set_xlim(0, max(vals) * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=.6)
    footnote(fig, ax, "四个倍数取自上一节同一枚计时产物的 comparison 字段（与图 5-16 同源，不另算一遍）；"
                      "该件为旧主机混合口径，倍数只用于说明量级差别")
    save(fig, "fig517", "重画图5-17.png")
    return made


def replace_figure(doc, fig_id, png_path):
    """按题注严判定位（题注前 3 段内找图），**原地覆盖那张图的字节**：
    显示尺寸、图片段、关系表全不动 ⇒ 包内不会留下没人引用的旧 PNG（先前用 add_picture 换图，副本反而胖了 130 KB）。"""
    from docx.oxml.ns import qn
    paras = doc.paragraphs
    ci = next((i for i, p in enumerate(paras) if is_caption(p.text) and p.text.strip().startswith(fig_id)), None)
    if ci is None:
        return f"{fig_id} 题注没找到 ⇒ 未替换"
    pi = next((i for i in range(ci - 1, max(-1, ci - 4), -1) if paras[i]._p.findall(".//" + qn("a:blip"))), None)
    if pi is None:
        return f"{fig_id} 题注（段{ci}）前 3 段内没有图 ⇒ 未替换（不猜位置）"
    para = paras[pi]
    ext = para._p.findall(".//" + qn("wp:extent"))[0]
    cx, cy = int(ext.get("cx")), int(ext.get("cy"))
    part = para.part.rels[para._p.findall(".//" + qn("a:blip"))[0].get(qn("r:embed"))].target_part
    old_bytes = len(part.blob)
    new = png_path.read_bytes()
    part._blob = new                                    # python-docx 1.2 的 Part.blob 读的是 _blob；写成别的名字就是静默无效（回读会当场抓到）
    assert part.blob == new, f"{fig_id} 写入后立刻读回不一致 ⇒ 属性名或库版本变了，别继续"
    return (f"{fig_id}：段{pi} 原地换图（{str(part.partname)}），旧 {old_bytes:,} B → 新 {len(new):,} B；"
            f"显示尺寸 EMU 不变（{cx}×{cy}）")


def add_fig514_pointer(doc):
    """图5-14 在原件里是"只有题注、正文零引用"的孤图（9/27 census 现算）。这里只补**指向句**：
    同一个已核对象换个图号指过去，不带数字、不带强度词、不写结论——与 5.7 过渡句同一把授权。"""
    from docx.oxml.ns import qn
    txts = [p.text for p in doc.paragraphs]
    cited = sum(1 for t in txts if "图5-14" in t.replace(" ", "") and not is_caption(t))
    if cited:
        return f"图5-14 正文引用已有 {cited} 处 ⇒ 不补（幂等）"
    host = next((i for i, t in enumerate(txts) if "0.539923" in t and "0.148723" in t), None)
    if host is None:
        # D1 至今是「需人工」行（新文本里混着写作闸门与"两列都进表"这类指令，landable() 拒收），
        # 所以锚点退回题注定位：图5-14 题注 → 前一段是图片段 → 再前一段就是该节正文。
        ci = next((i for i, t in enumerate(txts) if is_caption(t) and t.strip().startswith("图5-14")), None)
        if ci is None or ci - 2 < 0:
            return "既没找到 D1 落字段、也没找到 图5-14 题注 ⇒ 不补句，不猜位置"
        cand = ci - 2                                   # 题注前两段：一般就是引出这张图的那句正文
        if len(txts[cand].strip()) < 40 or txts[cand].strip().startswith(("5.", "第")):
            return f"题注前第 2 段（段{cand}）不像正文 ⇒ 不补句，交人工放位置"
        host = cand
    sent = FIG514_POINTER
    for w in FORBID_IN_TRANSITION:
        assert w not in sent, f"指向句混进强度词 {w}"
    # 数字闸：先剔掉图号本身与指标名，剩下的任何数字都算"新数字"
    residue = re.sub(r"图\s*5-14|Rel-L2", "", sent)
    assert not re.search(r"\d", residue), f"指向句带进新数字：{residue}"
    assert "倍" not in residue and "%" not in residue, "指向句不许出现倍数或百分比"
    para = doc.paragraphs[host]
    tail = para._p.makeelement(qn("w:p"), {})            # 新建空段插在本段之后
    para._p.addnext(tail)
    from docx.text.paragraph import Paragraph
    Paragraph(tail, para._parent).add_run(sent)
    return f"图5-14 指向句已插在段{host}之后：{sent}"


def figs(copy_path):
    data = fig_data()
    if not data:
        return 0
    try:
        made = render_figures(data, OUT)
    except ImportError as e:
        print(f"[图] 画图库不在（{e}）⇒ 本轮跳过重画，不动副本")
        return 0
    from docx import Document
    doc = Document(str(copy_path))
    for fid, key in FIG_IDS.items():
        print("   [图]", replace_figure(doc, fid, made[key]))
    print("   [图]", add_fig514_pointer(doc))
    stale = [i for i, p in enumerate(doc.paragraphs)
             if any(x in p.text for x in ("0.5612", "11.7598"))]
    if stale:
        print(f"   [图][不一致告警] 图5-14 已换成 test 口径新数，但正文段 {stale} 仍印着旧数 0.5612/11.7598"
              f"（D1 行未落）⇒ 图文不一致只有「D1 落字」这一个封法，本副本不得当冻结版")
    doc.save(str(copy_path))
    chk = Document(str(copy_path))
    from docx.oxml.ns import qn
    blips = [(i, p) for i, p in enumerate(chk.paragraphs) if p._p.findall(".//" + qn("a:blip"))]
    print(f"[figs] 回读：段落 {len(chk.paragraphs)}、表 {len(chk.tables)}、图片位 {len(blips)}")
    for fid, key in FIG_IDS.items():
        ci = next(i for i, p in enumerate(chk.paragraphs) if is_caption(p.text) and p.text.strip().startswith(fid))
        pi = next(i for i in range(ci - 1, max(-1, ci - 4), -1) if chk.paragraphs[i]._p.findall(".//" + qn("a:blip")))
        part = chk.paragraphs[pi].part.rels[
            chk.paragraphs[pi]._p.findall(".//" + qn("a:blip"))[0].get(qn("r:embed"))].target_part
        same = hashlib.sha256(part.blob).hexdigest() == hashlib.sha256(made[key].read_bytes()).hexdigest()
        print(f"   [回读] {fid} 包内 {str(part.partname)} sha256 与渲染件相同 = {same}")
        if not same:
            print(f"[INVALID] {fid} 换图没落到包内 ⇒ 副本按半品标")
            return 0
    return len(made)


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
    g.add_argument("--pair57", type=pathlib.Path, help="5.7 成对块：E2 换数 + E3 插段 + E4 结论句改过渡句（同进同退）")
    g.add_argument("--figs", type=pathlib.Path, help="在给定副本上重画并替换 图5-14/5-16/5-17（数从仓内正本现取）")
    g.add_argument("--selfcheck", action="store_true", help="只跑题注夹具（必过 + 必红各一条）")
    g.add_argument("--all", action="store_true",
                   help="一把跑完整链：新建副本 → 整写/术语/插段 → 三张新表（含来源列）→ 表5-7 → 5.7 成对块。顺序固定，防每轮手接不同次序")
    ap.add_argument("--into", type=pathlib.Path, default=None,
                    help="--apply 用：写进指定副本（--all 串链用），不给就新建一个时间戳文件")
    ap.add_argument("--expect-changed", type=int, default=None,
                    help="--apply 用：期望被改段落数，不接等即 INVALID（闸三）")
    args = ap.parse_args()
    import docx  # noqa: F401  ② 先确认库在，不在就别硬写
    from docx import Document

    if args.selfcheck:
        return selfcheck_caption()
    if args.verify:
        if selfcheck_caption():                 # 同上：尺先自证，再谈终检结论
            return 3
        return verify(args.verify)
    if args.all:
        import subprocess
        if selfcheck_caption():                 # 题注夹具不过 ⇒ 整条链不开跑（统括官 13:0x：不能靠每次手看）
            return 3
        st = SRC.stat()
        print(f"[原件只读] {st.st_size:,} B mtime={datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')} "
              f"sha256={hashlib.sha256(SRC.read_bytes()).hexdigest()[:12]}")
        dst = OUT / f"装配副本-{datetime.datetime.now():%Y%m%dT%H%M%S}.docx"
        shutil.copy2(SRC, dst)
        here = str(pathlib.Path(__file__).resolve())
        for step in (["--apply-into", str(dst)], ["--tables", str(dst)], ["--t57", str(dst)],
                     ["--pair57", str(dst)], ["--figs", str(dst)]):
            if step[0] == "--apply-into":
                r = subprocess.run([sys.executable, here, "--apply", "--into", str(dst)],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
            else:
                r = subprocess.run([sys.executable, here, *step],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(f"--- {step[0]} rc={r.returncode} ---")
            print((r.stdout or "").strip()[-600:] or (r.stderr or "").strip()[-400:])
            if r.returncode:
                print(f"[INVALID] 链在 {step[0]} 断掉 ⇒ 交付止步，本副本标为半品")
                return r.returncode
        print(f"[all] 候选正本 = {dst}")
        return 0
    if args.t57:
        if not args.t57.exists():
            print(f"[INVALID] 副本不存在：{args.t57}", file=sys.stderr)
            return 2
        return 0 if update_t57(args.t57) >= 5 else 1

    if args.pair57:
        if not args.pair57.exists():
            print(f"[INVALID] 副本不存在：{args.pair57}", file=sys.stderr)
            return 2
        return 0 if pair57(args.pair57) else 1

    if args.figs:
        if not args.figs.exists():
            print(f"[INVALID] 副本不存在：{args.figs}", file=sys.stderr)
            return 2
        n = figs(args.figs)
        if n != len(FIG_IDS):
            print(f"[INVALID] 三张图只换了 {n} 张 ⇒ 正本缺件或库不在，本轮副本按半品标，别当交付")
            return 1
        return 0

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

    dst = args.into or (OUT / f"装配副本-{datetime.datetime.now():%Y%m%dT%H%M%S}.docx")
    if not dst.exists():
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
