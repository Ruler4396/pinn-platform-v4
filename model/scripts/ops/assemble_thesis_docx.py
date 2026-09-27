#!/usr/bin/env python3
"""毕设 docx 装配器（统括官 9/27 11:2x 改派给方案线，四道闸写进代码而不是写在承诺里）。

用法（本机零机时；产物一律落在仓外 .scratch/）：
    python3 model/scripts/ops/assemble_thesis_docx.py --plan            # 只判定工单数据行能不能自动落、为什么不能
    python3 model/scripts/ops/assemble_thesis_docx.py --apply           # 在带时间戳的副本上落字（原件只读）
    python3 model/scripts/ops/assemble_thesis_docx.py --verify <副本>    # 副本正文 ↔ 工单「新文本」逐行 diff
硬约（违反即 INVALID 退出码非 0）：
  ① 绝不就地改原件：--apply 先 copy 到 .scratch/装配副本-<UTC 时间戳>.docx，原件字节数与 mtime 进回执；
  ② 只用 python-docx，不手点 XML；③ 按锚句内容定位（`thesis.txt` 的 :NNNN 含表格单元，与 python-docx 段号不同尺）；
  ④ A13 是待批占位，永远不落字，只在清单里标出；⑤ 术语类算子（A15/A14）与"删段+插段成对"类（E4+E3）单独走。
"""
from __future__ import annotations
import argparse, datetime, hashlib, os, pathlib, re, shutil, sys, tempfile

# 顶层就改编码：夹具常被 `python3 -c "import assemble_thesis_docx as A; A.selfcheck_caption()"` 裸调用，
# 只放在 main() 里等于"从命令行跑没事、从别人手里跑就崩"（统括官 15:0x 在默认 GBK 终端实测崩在 '⇒'）。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
WORKLIST_REL = WORK.relative_to(REPO).as_posix()   # 给 git 用的路径必须正斜杠：str(WindowsPath) 给反斜杠，
                                                   # git 会把 `\` 当转义 ⇒ `git show HEAD:…` 静默返回空，
                                                   # 于是"基线 0 行"让丢行断言永远为空——一条恒真闸（15:5x 自抓）
OUT = ROOT / ".scratch"

# 待作者批的占位（④）：永远不写进论文
AWAITING = {"A13": "已授权（§12 第 13 条②：作者放弃否决权，降级档即为已授权文本）⇒ 不再算「待批」；"
                    "但本工具不机械落它的字：目标是长段落里的**句子**（摘要段3 的 26 个 run 中占 17–23，跨 'PDE'/'PINN' 等拉丁片段），"
                    "整段替换的兜底会把 26 个 run 压平成 run0 一种字体 ⇒ 版式风险。成品句已进对照表，由作者粘"}
# 术语类算子：整段替换会吃内容，必须走"整词替换 + 只动了目标词"校验
TERM_OPS = [("A15", "阶段内PDE约束", "阶段内残差惩罚项"),
            ("A15b", "阶段内PDE", "阶段内残差惩罚项"),   # 表内截短形（"启用阶段内PDE"）——旧算子只认全称，漏了一格
            ("A14", "协同修正", "交替更新")]
TERM_RE = re.compile("|".join(re.escape(a) for _, a, _ in TERM_OPS))
FOLD_RE = re.compile("|".join(re.escape(w) for _, a, b in TERM_OPS for w in (a, b)))
# 新文本里出现这些词 ⇒ 判定为"操作指令或工单内部注"，不整段替换
INSTR = ("删除", "统一替换为", "同步替换", "替换为", "行名", "更正", "撤回", "定档", "凭据",
         "不许", "禁写", "见 §", "若将来", "留作投稿", "前置核查", "⇒", "本工单", "区间都要带上")
PAIR_OPS = ("E2", "E3", "E4")   # ③ 换数+插段+删句三者同进同退，见 e_block()
# 表内一格换标签（工单里"换标签"类）：只允许"旧标签是新标签的子串"那种纯扩展，且整格唯一命中才做。
CELL_OPS = {"A16": ("表5-1", "阶段内残差惩罚项", "阶段内残差惩罚项（启用）")}
# D1b：判决＝(a) 改数（统括官 16:4x 第三次量到旧值仍在表里）。新值不手打——每条都带正本行号，
# 落字前现读那一行、里面没有这个数就拒做。同行 0.0274（B-test-1）不在表内 ⇒ 不会被碰。
CELL_VALUES = {
    "D1b-basic": ("表5-1", "0.5612", "0.539923", "docs/revision/T5矩阵test口径读数-20260926.md", 38),
    "D1b-geometry": ("表5-1", "0.0390", "0.148723", "docs/revision/T5矩阵test口径读数-20260926.md", 39),
}
POINTER_ROWS = ("A18", "A19")     # 孤表指向句，凭据＝§12 第 13 条①（作者拍定"补句、不删表"）；句子只从工单载荷取
LIMIT_ROWS = ("J4",)              # 新增一句局限，凭据＝§12 第 13 条③／作者 9/27 午后当回合选择（时刻以 git log %cI 为准）；句子与锚点都只从工单那一行取
# 混合格子的正文切片：有些行的「新文本」把正文句和写作指令写在同一格（D1 就是），整格落字会把
# "两列都进表""（样本口径；格13 sd …）""前置核查（写作闸门）"这类话印进论文。
# 白名单里**每条都必须逐字是该行新文本的子串**（下面断言），所以这条路不许造新句；被排除的句子留在工单里由作者自己粘。
BODY_SLICE = {
    "D1": ("为检查几何增强编码的作用，本文固定壁面构造模式与训练预算，只更换输入特征集：basic 集（4 列，只能用软壁面惩罚）"
           "与 geometry 集（14 列，同一软壁面惩罚），各训练五个种子。",
           "速度 Rel-L2（test / mean-of-cases）由 0.539923±0.014799 降至 0.148723±0.011416，即 3.6 倍；"
           "同一对照的 val 口径为 0.51955±0.01322 降至 0.13622±0.01143（3.8 倍）",
           "这是全矩阵中效应明显超出种子噪声的两条结果之一",
           "仅坐标输入无法直接表达点位与壁面、收缩段或入口剖面之间的关系，壁距、区域标签和轴向比例等特征补上了这部分信息。"),
}


def sliced_payload(rid: str, new_raw: str):
    """逐字校验切片后拼成正文载荷；任一切片不是该行新文本的子串 ⇒ 拒绝落字。"""
    base = payload(new_raw)
    parts = BODY_SLICE[rid]
    bad = [s for s in parts if s not in base]
    if bad:
        return "", f"切片不是 {rid} 行「新文本」的逐字子串 ⇒ 拒绝（这条路不许造新句）：{bad[0][:40]}…"
    return "".join(parts), ""



def norm(s: str) -> str:
    s = re.sub(r"\[\d+(?:[-,]\d+)*\]", "", s)              # 行内文献号
    return re.sub(r"\s+", "", s.replace("，", ",").replace("。", "."))


DASH_MAP = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
            "\u2015": "-", "\u2212": "-", "\uff0d": "-", "\u301c": "-", "\uff5e": "-"}


def norm_id(s: str) -> str:
    """**号判定的唯一归一尺**（统括官 9/27 17:5x 第③条）：凡"某号有没有被引用/是不是同一张表"
    这类判定，先过这里再比。落字那句写的是「表 3-4」（带空格），题注写的是「表3-4」——
    不归一就会把"已引用"读成"孤号"，也会让 `startswith('表3-4')` 只数到题注那一次。
    NFKC 统一全角／半角，短横族统一成 ASCII '-'，再剔掉全部空白。"""
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    for a, b in DASH_MAP.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s)


def clean(md: str) -> str:
    """把工单格子里的 markdown 痕迹清成论文正文；`**【…】**` 之后一律算工单内部注。"""
    md = re.split(r"\*\*【", md)[0]
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    md = re.sub(r"`([^`]*)`", r"\1", md)
    return md.replace("\\|", "|").strip()


def is_authorized(rid: str) -> bool:
    """裁决标了"已授权"（作者放弃否决权）＝**不是待批**，是"需作者粘贴"。判据只声明这一处。"""
    return AWAITING.get(rid, "").startswith("已授权")


def n_await(await_) -> int:
    """**"待批"这把尺只声明一处**：`AWAITING` 里标了"已授权"的那一类（A13：作者放弃否决权）不算待批，
    算"需作者粘贴"。三处打印共用它，免得 `--plan` 报 1 而对照表报 0——同一量两把尺今天已自报过两次。"""
    return sum(1 for rid, _ in await_ if not is_authorized(rid))


# **三张新表的形状声明**（统括官 19:0x 第③条）：丢**一整列**对 `check_md_tables` 的 ragged 闸天然不可见
# （每行列数一致 ⇒ 永远不 ragged），所以列数与表头名必须单独钉成声明值。
# 这里的值是 **15:0x 裁定之后**的形状：「来源件 + 行号」列**不进论文表**（仓内路径不可出版），
# 由对照表的「三张新表的来源」一节承载（`table_sources()` 是唯一声明处）。
# ⇒ 以后谁把它加回论文表、或再丢任何一列，第 11 条子检查当场红。
TABLE_SHAPES = {
    "表4-4b": {"cols": 3, "rows": 5, "last_col": "说明", "header": ("权重项", "该批取值", "说明")},
    "表5-9": {"cols": 4, "rows": 13, "last_col": "备注",
              "header": ("方法与口径", "收缩 C-base (s)", "弯曲 B-base (s)", "备注")},
    "表5-10": {"cols": 5, "rows": 5, "last_col": "备注",
                "header": ("方法与口径", "稠密", "分层 5%", "n", "备注")},
    # 两张被裁定改过结构的表也进声明（表5-7＝G4 加「模型批次」列 + B-test-2 行；表5-8＝A11 丙改三格文字）
    "表5-7": {"cols": 7, "rows": 5, "last_col": "模型批次",
              "header": ("工况", "几何类型", "观测条件", "速度场L2误差", "压力场L2误差", "压降相对误差", "模型批次")},
    "表5-8": {"cols": 5, "rows": 3, "last_col": "最终最大压力误差",
              "header": ("阶段内残差惩罚项", "压力阶段结束时压力场L2误差", "最终速度场L2误差",
                          "最终压力场L2误差", "最终最大压力误差")},
}


BANNED_HEADER_TOKENS = ("来源",)   # 15:0x 裁定：**论文表里不得出现「来源件+行号」列**（仓内路径不可出版，来源由对照表承载）。
# 把它写成禁令而不只是"表头不等"的附带后果：这样下一轮谁（包括我）把列加回来，报错信息说的是**为什么**不许加。


def shape_violations(header, ncols, nrows, spec):
    """纯函数：**闸二＝逐表形状签名**（期望行数, 期望列数, 末列表头名, 全表头），任一不符即报并指名。
    这道专管"丢一整列而终检照报 0"与"结构被等量替换"——它们对 ragged 判据天然不可见。
    行数是**等值**而非下限：给表加一行也得同批改声明，这是刻意的摩擦。"""
    v = [f"表头出现被禁的列名 {b!r} ⇒ 违反 15:0x 裁定（来源列不进论文表，改由对照表承载；要加请先否决 §12 第 19 条）"
         for b in BANNED_HEADER_TOKENS if any(b in h for h in header)]
    if ncols != spec["cols"]:
        v.append(f"列数 {ncols} ≠ 声明 {spec['cols']}")
    if header and header[-1] != spec["last_col"]:
        v.append(f"末列表头 {header[-1]!r} ≠ 声明 {spec['last_col']!r}")
    if nrows != spec["rows"]:
        v.append(f"行数 {nrows} ≠ 声明 {spec['rows']}（要加行就同批改声明）")
    if tuple(header) != spec["header"]:
        v.append(f"表头与声明不符：现 {list(header)} ｜声明 {list(spec['header'])}")
    return v


def selfcheck_table_shapes(copy_path=None):
    """第 11 条子检查：三张新表的**列数与表头名必须等于声明值**，含一条自带必红。"""
    from docx import Document
    p = pathlib.Path(str(copy_path)) if copy_path else candidate()
    if p is None or not p.exists():
        print("[子检查·新表形状] 候选正本不在场 ⇒ **未验**（不许拿原件或猜一份副本充数）")
        return 1
    doc = Document(str(p))
    body = list(doc.element.body)
    bad = []
    for tid, spec in TABLE_SHAPES.items():
        tbls = tables_by_id(doc, body, tid)
        if not tbls:
            bad.append(f"{tid}：题注整段相等找不到表 ⇒ 声明的表没进副本")
            continue
        for cap, tb in tbls:
            hdr = [c.text.strip() for c in tb.rows[0].cells]
            v = shape_violations(hdr, len(hdr), len(tb.rows), spec)
            bad += [f"{tid}（{len(tb.rows)}×{len(hdr)}）：{x}" for x in v]
    # **必红夹具**：拿"少一列 + 末列不是备注"的形状喂 shape_violations，它必须报
    spec = TABLE_SHAPES["表5-10"]
    dropped = shape_violations(list(spec["header"][:-1]), spec["cols"] - 1, spec["rows"], spec)
    renamed = shape_violations(list(spec["header"][:-1]) + ["来源件 + 行号"], spec["cols"], spec["rows"], spec)
    shrunk = shape_violations(list(spec["header"]), spec["cols"], spec["rows"] - 1, spec)
    added = shape_violations(list(spec["header"]) + ["来源件 + 行号"], spec["cols"] + 1, spec["rows"], spec)
    if not any("被禁" in x for x in added):
        bad.append("禁令夹具失效：把「来源件 + 行号」列加回来，shape_violations 不报被禁 ⇒ 裁定没进闸")
    if not (dropped and renamed and shrunk and added):
        bad.append("必红夹具失效：删一整列／改末列名／少一行三种形状 shape_violations 仍不报 ⇒ 这道闸是空的")
    print(f"[闸二·逐表形状签名] {len(TABLE_SHAPES)} 张表的（行数,列数,末列名,全表头）等于声明 = "
          f"{'是' if not [b for b in bad if '必红' not in b] else '否'}"
          f"；**必红四发**（删一列报 {len(dropped)} 条／改末列名报 {len(renamed)} 条／少一行报 {len(shrunk)} 条／**把来源列加回来报 {len(added)} 条**，均应 ≥1）⇒ "
          + ("全过 ✓" if not bad else f"**{len(bad)} 处不符**"))
    for x in bad:
        print("   ", x)
    return 1 if bad else 0


def count_needle(needle: str, quiet: bool = False):
    r"""**三数同框**（统括官 18:5x 把这条族规矩升级）：一个"某串出现几次"的断言一次报三个数，
    并写死哪个数拿来判不一致。作用域＝`rows()` 的 A-J 数据行——**用工具自己的解析，不再另写一条 grep**：
    §10 原文里那条 `'^\| [A-J]'` 在 GNU BRE 下 `\|` 是"或"，**匹配的是整本**（帧号必带：现读工单总行 413、以 `| ` 开头 157、A-J 数据行 56；
    统括官那帧 332；本注释旧版写的 328 是更早一帧——**“整本有多少行”这类数不带帧，就是下一次拼错的种子**），
    所谓"作用域"从来就没生效过（裸跑与"作用域"同样得 7，谁照它读都会以为判二失败了）。
    `quiet=True` 时只返回三数不打字——**子检查要喂的是这个函数本身**，不是重抄一遍正则。"""
    in_rows, outside, raw = needle_counts(needle)
    if not quiet:
        print(f"[三数同框] needle={needle!r}：数据行内 = **{in_rows}** ｜条文自身（非数据行）= {outside} ｜全文件裸跑 = {raw}")
        print("    判不一致**只用「数据行内」这个数**；「条文自身」那一档随“谁在断言行里多写一句”漂，"
              "裸跑值 = 两者之和，拿它判必假红。")
    return in_rows, outside, raw


def needle_counts(needle: str, text: str | None = None):
    r"""作用域**只这一处**：`^\|\s*[A-J]\d+[a-z]?\s*\|`＝工单数据行的形状（与 `rows()` 同一判据）。
    返回**三数**（数据行内，非数据行，全文件）。**needle 一律 `re.escape`**——不转义时 `.` 是通配符，
    实测同一枚 `0.539923` 会被数成 14/14/14，那把“尺”就什么都命中。
    （参数原名 `txt` 时函数体里还留着一句 `txt = WORK.read_text(...)`，**默认值永远取不到、实参形同虚设**——
    夹具喂不进去＝只能测默认路径，这种“能测但测不到注入”的形状要当场拆掉。）"""
    txt = WORK.read_text(encoding="utf-8") if text is None else text
    pat = re.compile(r"^\|\s*[A-J]\d+[a-z]?\s*\|")
    lines = txt.splitlines()
    in_rows = sum(len(re.findall(re.escape(needle), ln)) for ln in lines if pat.match(ln))
    raw = len(re.findall(re.escape(needle), txt))
    return in_rows, raw - in_rows, raw


IMPERATIVE = ("请", "不得", "必须", "不许", "禁止", "应当", "要")


def s12_directives(text: str | None = None, width: int = 46):
    """§12 是**散文条目** ⇒ `rows()` 的表格行正则看不见它：统括官写在里面的"请/不得/必须"
    不会变成任何一道闸的待办（9/27 15:44:40 那条 A13 指令就这么漏过一整轮）。
    **只匹配条目首行会把缺陷搬到下一层**——祈使句大多在缩进子行里（本函数第一版就犯过，
    统括官 19:1x 判出来），所以这里按**整条**扫：一条 = 从 `^N. ` 到下一条或下一节为止的全部行。
    返回 [(条号, 命中在第几行, 摘句)]；命中在第 2 行以后即"子行"。不判定，只逐条打印逼本轮读。"""
    if text is not None:                              # 夹具喂的就是条目正文本身，不再去读工单
        body = text
    else:
        txt = WORK.read_text(encoding="utf-8")
        if "## 12." not in txt:
            return []
        body = txt.split("## 12.", 1)[1]
    body = re.split(r"\n## ", body, 1)[0]
    items = re.split(r"(?m)^(\d+)\.\s+", body)          # [前言, 号, 体, 号, 体, …]
    out = []
    for k in range(1, len(items) - 1, 2):
        num, blob = int(items[k]), items[k + 1]
        lines = [l for l in blob.split("\n")]
        for li, line in enumerate(lines):
            if any(w in line for w in IMPERATIVE):
                out.append((num, li + 1, re.sub(r"\s+", " ", line)[:width]))
                break
    return out


def selfcheck_directives():
    """必红夹具（统括官 9/27 判：`s12_directives()` 第一版只读条目首行＝把原缺陷搬到下一层）。**三发**：
    ① 只有缩进子行含祈使词的条目**必须被打印**；② 整条不含词的条目**必须不被打印**（缺这一发，"什么都打印"的尺也算过＝空对照）；
    ③ **旧尺（只读条目首行）在这枚合成件上必须命中 0 条**，非 0 就说明夹具空转、抓不到它声称要抓的缺陷。"""
    synth = ("12. 标题句没有关键词\n    - **必须**：这条只在缩进子行里出现\n"
             "13. 另一条\n    - 这一条整条都不含祈使词\n")
    got = s12_directives(synth)
    hit = [g for g in got if g[0] == 12 and g[1] > 1]                      # ① 必打印
    leaked = [g for g in got if g[0] == 13]                                # ② 必不打印
    first_only = any("必须" in g[2] for g in got if g[1] == 1)
    old_ruler = sum(1 for l in synth.splitlines()
                    if re.match(r"^\d+\.\s", l) and any(w in l for w in IMPERATIVE))    # ③ 必为 0
    ok = bool(hit) and not first_only and not leaked and old_ruler == 0
    print(f"[必红夹具·§12 指令扫整条] 三发：① 子行祈使句被打印＝{bool(hit)}（应 True，命中 {len(hit)} 条）；"
          f"② 无词条目误报＝{len(leaked)} 条（应 0）、首行误报＝{first_only}（应 False）；"
          f"③ 旧尺（只读首行）命中＝{old_ruler}（**必须 0**，非 0＝夹具空转）⇒ "
          + ("三发全中：子行里的祈使句跑得出来，且这把尺不是什么都报 ✓" if ok else "**有一发不对：只读到首行＝把原缺陷搬下一层**"))
    return 0 if ok else 1


def rows():
    out = []
    for line in WORK.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\| [A-J]\d+[a-z]? \|", line):
            c = [x.strip() for x in line.split("|")[1:-1]]
            out.append({"id": c[0], "loc": c[1], "act": c[2], "new": c[3] if len(c) > 3 else "",
                        "why": c[4] if len(c) > 4 else ""})
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
    lost = sorted({norm_id(x) for x in re.findall(r"[图表]\s*\d+(?:-\d+)?", old_text)} -
                  {norm_id(x) for x in re.findall(r"[图表]\s*\d+(?:-\d+)?", new)})
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


def candidate() -> pathlib.Path | None:
    """**候选正本的唯一指针**（统括官 18:0x 定的规矩：一串时间戳交付物，"取最新"必拿错）。
    读 `.scratch/候选正本.txt`（第 1 行＝文件名，第 2 行＝sha256 前缀，可选）；
    指针缺失/指向不存在的件/sha 不吻合 ⇒ 返回 None，**由调用方判红**——绝不退回到"按 mtime 猜一份"。"""
    ptr = OUT / "候选正本.txt"
    if not ptr.exists():
        print(f"[INVALID] 候选正本指针缺失：{ptr} ⇒ 不许按 mtime 猜交付件（今天已被一枚 部件级 diff=0 的重打包件坑过）")
        return None
    lines = [x.strip() for x in ptr.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        print("[INVALID] 候选正本指针是空件")
        return None
    name = lines[0]
    if "/" in name or "\\" in name or name != (OUT / name).name:
        print(f"[INVALID] 候选正本指针里出现路径分隔：{name!r} ⇒ 指针只许写 `.scratch` 下的**文件名**，拒取"
              f"（否则一枚写歪的指针就能把原件或仓外件当交付物喂给夹具）")
        return None
    if name == SRC.name:
        print("[INVALID] 候选正本指针写的是**原件**名 ⇒ 停下（本条线全程不碰原件；夹具跑在原件上必假过）")
        return None
    p = OUT / name
    if not p.exists():
        print(f"[INVALID] 候选正本指针指向不存在的件：{name}")
        return None
    if len(lines) > 1 and not hashlib.sha256(p.read_bytes()).hexdigest().startswith(lines[1]):
        print(f"[INVALID] 候选正本指针的 sha 与件不符：{name}（指针 {lines[1]}，现算 "
              f"{hashlib.sha256(p.read_bytes()).hexdigest()[:12]}）⇒ 件被改过或指针写错，停下")
        return None
    # **自拒（统括官 16:2x 第②③条）**：取件只认指针（`glob` 在这儿**只当扫描器、不当选择器**——
    # 一件都不会靠名字或 mtime 取），但顶层只要还躺着未登记的副本就说明"声明"与"盘上"不一致：
    # 更新的没登记成候选正本 ⇒ 人和任何残留 glob 的工具都会拿错件；更旧的没挪走 ⇒ 与"顶层只剩交付件"的口径不符。
    # 两类一律判未验（他点名的必红夹具正是"扔一枚**旧**副本"，只挡更新的那类＝挡不住）。
    stray = sorted(f for f in OUT.glob("装配副本-*.docx") if f.name != name)
    if stray:
        n_new = sum(1 for f in stray if f.stat().st_mtime > p.stat().st_mtime + 1)
        print(f"[INVALID] `.scratch` 顶层另有 **未登记**的副本 {len(stray)} 枚（其中比候选正本更新 {n_new} 枚）："
              f"{[f.name for f in stray][:3]} ⇒ 判未验。新交付件＝改指针并 supersede 旧的；试验件＝挪进 superseded/"
              f"并留同名 `.作废` 旁标记（只追加、不删）。")
        return None
    return p


# **「改一格文字」那类工单行的现读坐标**（统括官 9/27 16:2x 之后的作者包收口：位置可算就不该占"需人工"的措辞，
# 但**落点 ≠ 编辑决定**——G1 要往表里引两枚跨工况读数、C2 还要挪行，脚本一律不落字，只把格子点名给作者）。
# 旧格文本必须**逐字声明**（不许猜），命中数 ≠ 1 即判"歧义"，由 `cell_coords()` 报红而不是硬给一个坐标。
CELL_TEXT_ROWS = {"G1": ("表3-3", "转角泛化测试工况"),
                  "C2": ("表5-1", "壁面u残余")}


ARTIFACT_GLOBS = ("model/results/pinn/**/*.json",)     # 扫描范围要写死并在输出里点名（CSV 不在内，另说）


def artifact_values(repo=None):
    """把仓内结果件的**所有数值**收成 `⇒ {四舍五入到 k 位小数的字符串: [(件, 路径), …]}`，k=2..8。
    这不是"证明某个数对"，只回答一个问题：**论文表里这个数，仓里有没有任何一个产物能供上。**"""
    import json
    repo = pathlib.Path(str(repo)) if repo else REPO
    idx = {}
    n_files = 0
    for pat in ARTIFACT_GLOBS:
        for f in repo.glob(pat):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            n_files += 1
            stack = [(d, "")]
            while stack:
                o, path = stack.pop()
                if isinstance(o, dict):
                    stack += [(v, f"{path}.{k}") for k, v in o.items()]
                elif isinstance(o, list):
                    stack += [(v, f"{path}[{i}]") for i, v in enumerate(o)]
                elif isinstance(o, bool):
                    continue
                elif isinstance(o, (int, float)):
                    for k in range(2, 9):
                        idx.setdefault(f"{o:.{k}f}", []).append((f.relative_to(repo).as_posix(), path))
    return idx, n_files


def history_values(repo=None):
    """A2 级＝**训练历史**（`model/results/pinn/**/history.csv`）。表5-2 那种「阶段起始值」是逐 epoch 的读数，
    本来就不可能原样躺在评估 JSON 里——不把它单独成层，就会被算进"两层都无"的假清单（这正是拿错尺的样子）。
    只按 `k=2..6` 位小数收，够论文表用；再多位就是自找麻烦。"""
    repo = pathlib.Path(str(repo)) if repo else REPO
    idx = set()
    n = 0
    for f in repo.glob("model/results/pinn/**/history.csv"):
        n += 1
        head = f.read_text(encoding="utf-8", errors="replace").splitlines()
        if not head:
            continue
        for line in head[1:]:
            for cell in line.split(","):
                s = cell.strip()
                if not s or len(s) > 24 or s[0] not in "0123456789-":
                    continue
                try:
                    v = float(s)
                except ValueError:
                    continue
                for k in range(2, 7):
                    idx.add(f"{v:.{k}f}")
    return idx, n


def doc_tokens():
    """B 级=**登记件**里出现过的数（`docs/revision/*.md`）。它不证明数对，只说明"这个数已被人登记过、
    有口径出处"，与 A 级（结果件里原样存在的数）必须分开报，不能合成一个命中率糊过去。"""
    out, skipped = set(), []
    for f in sorted((REPO / "docs" / "revision").glob("*.md")):
        # **必须排除过程日志（`回执-*`／含「对照表」的件）**——本回合是被**夹具自己**抓出来的：
        # 我在回执里写「合成数 0.987654 三层都不在场」这句话，于是 B 级把 `0.987654` 数成"有登记"
        # ⇒ **必红那一发被自己的日志拓灭**（假绿）。登记件那一层只收**读数正本／工单**，不收日志。
        if f.name.startswith("回执") or "对照表" in f.name:
            skipped.append(f.name)
            continue
        out |= set(re.findall(r"\d+\.\d{2,}|\d\.\d+[eE][-+]?\d+", f.read_text(encoding="utf-8", errors="replace")))
    return out, skipped


def value_prov_report() -> int:
    """`--value-prov`：论文表格里每个读数 token 对一遍 **A 级=结果件 / B 级=登记件** 两层；
    两发夹具当场跑（必过＝刚落字的读数至少在 B 级在场；必红＝合成数两层都不在，否则这把尺恒真）。"""
    from docx import Document
    p = candidate()
    if p is None:
        print("[未验] 候选正本指针缺失/失效 ⇒ 不做数值溯源（不猜一份副本）")
        return 1
    idx, nf = artifact_values()
    hist, nh = history_values()
    bset, bskip = doc_tokens()
    nd = len(list((REPO / "docs" / "revision").glob("*.md"))) - len(bskip)
    # **碰撞地板**：A2 把 37 枚 history.csv 的每个数都按 2..6 位收进索引 ⇒ 位数为 2~3 的 token 会撞上别的 run。
    # 所以先量这把尺的假阳性率，否则"两层都没有 = 0"会被读成"全部有源"——那是把筛查当证明。
    import random
    rnd = random.Random(20260927)
    floor = {}
    for d in (2, 3, 4, 5, 6):
        probes = [f"{rnd.uniform(0.0, 1.0):.{d}f}" for _ in range(300)]
        floor[d] = (sum(1 for q in probes if q in idx) / len(probes),
                    sum(1 for q in probes if q in hist) / len(probes),
                    sum(1 for q in probes if q in bset) / len(probes))
    tok = re.compile(r"\d+\.\d{2,}|\d\.\d+[eE][-+]?\d+")
    doc = Document(str(p))
    sci = []
    tiers = {"A": [], "A2": [], "WEAK": [], "B": [], "NONE": []}
    by_dec = {}
    total = 0
    for ti, tb in enumerate(doc.tables):
        for ri, row in enumerate(tb.rows):
            for ci, cell in enumerate(row.cells):
                for m in tok.finditer(cell.text):
                    s = m.group(0)
                    if cell.text[m.end():m.end() + 1] in ("e", "E"):
                        sci.append((ti, ri, ci, s + cell.text[m.end():m.end() + 3]))
                        continue
                    total += 1
                    d = len(s.split(".")[1]) if "." in s and "e" not in s.lower() else 6
                    tier = "A" if s in idx else ("A2" if s in hist else ("B" if s in bset else "NONE"))
                    if tier == "A2" and floor[d][1] >= 0.20:
                        tier = "WEAK"        # 该位数上 A2 的假命中率 ≥20% ⇒ 这个"命中"不构成来源
                    tiers[tier].append((ti, ri, ci, s))
                    by_dec.setdefault(d, dict.fromkeys(("A", "A2", "WEAK", "B", "NONE"), 0))[tier] += 1
    print(f"[数值溯源·范围] A 级＝评估/度量 JSON {nf} 枚（{'、'.join(ARTIFACT_GLOBS)}）；"
          f"A2 级＝训练历史 {nh} 枚（`model/results/pinn/**/history.csv`）；B 级＝登记件 {nd} 枚（`docs/revision/*.md`，**已排除过程日志 {len(bskip)} 枚**：{chr(12289).join(bskip)}）｜"
          f"**不在范围内**：`predictions/*.csv`、`.npz`、`out/**`（仓内 0 枚 ⇒ E5 那格的一手读数在**实例侧**、不在仓）｜"
          f"副本 {p.name} 的 {len(doc.tables)} 张表共 {total} 个读数")
    print("    碰撞地板（随机造 300 枚同位数、[0,1) 的数看它「假装命中」的比例）："
          + "；".join(f"{d} 位 A={floor[d][0]:.0%}/A2={floor[d][1]:.0%}/B={floor[d][2]:.0%}" for d in sorted(floor)))
    print(f"    ⇒ **A 原样命中 {len(tiers['A'])}（{len(tiers['A']) / max(total, 1):.1%}）** ｜ "
          f"A2＝训练历史 {len(tiers['A2'])} ｜ **弱命中 {len(tiers['WEAK'])}**（该位数上 A2 地板 ≥20%，不算来源） ｜ "
          f"仅 B 级命中 {len(tiers['B'])}（＝派生量或口径合成分，如均值／比值／加速比／sd） ｜ "
          f"**两层都没有 {len(tiers['NONE'])}**（这才是待归因清单）")
    n5 = sum(sum(v.values()) for d, v in by_dec.items() if d >= 5)
    print("    按位数分档（**可引用的只有这张**：位数 ≤4 时字符串命中不构成来源证明）")
    for d in sorted(by_dec):
        v = by_dec[d]
        print(f"       {d} 位：共 {sum(v.values()):3d} 枚 ⇒ A {v['A']:3d}｜A2 {v['A2']:2d}｜弱 {v['WEAK']:2d}"
              f"｜仅登记 {v['B']:2d}｜无 {v['NONE']:2d} ｜A 级碰撞地板 {floor[d][0]:.0%} ⇒ "
              + ("命中可信" if floor[d][0] <= 0.05 else "**不足为证，要 per-cell 归属**"))
    print(f"       ⇒ 全表 {total} 枚里 **≥5 位的只有 {n5} 枚**，其余 {total - n5} 枚是 2~4 位读数"
          f"（论文四舍五入到 4 位是常规写法）｜另 {len(sci)} 枚科学计数法（`3.84e-05` 这类）不进位数分档。"
          f"**结论：这道闸做不到逐格自动定源**——缺的是一张 per-cell 归属表（哪张表哪一行来自哪个 run 的哪个字段），"
          f"它不在仓里，靠字符串索引是造不出来的")
    for u in tiers["A2"][:3]:
        print(f"   A2 训练历史：表序{u[0]} (行{u[1]},列{u[2]}) = {u[3]}")
    for u in tiers["WEAK"][:8]:
        print(f"   弱命中（要人工指 run 才能定源）：表序{u[0]} (行{u[1]},列{u[2]}) = {u[3]}")
    for u in tiers["B"][:6]:
        print(f"   仅 B 级：表序{u[0]} (行{u[1]},列{u[2]}) = {u[3]}")
    for u in tiers["NONE"]:
        print(f"   ⚠两层都无：表序{u[0]} (行{u[1]},列{u[2]}) = {u[3]}")
    # **探针值绝不能再写成字面量**（本回合两次踩）：第一次它躺在 `回执-*`（我把排除做了），
    # 第二次它躺在**工单第 24 条那条"禁止把探针值写进介质"的规则里**——规则文本本身在被这把尺扫。
    # ⇒ 机制：探针**运行时导出**（固定种子 ⇒ 同一条命令可重放），并且**当场自证它不在任何一层**。
    probe = f"{random.Random(20260927 ^ 59).uniform(0.0, 1.0):.6f}"
    must_pass = ("0.539923" in idx or "0.539923" in bset) and ("0.026912" in idx or "0.026912" in hist)
    must_fail = (probe not in idx) and (probe not in bset) and (probe not in hist)
    print(f"[夹具] 刚落字的 `0.539923` 至少在一层在场={must_pass}（应 True）；"
          f"**运行时导出探针** `{probe}`（种子 20260927^59，同一条命令可重放）三层都不在场={must_fail}"
          f"（必须 True，否则这把尺恒真）⇒ " + ("两发都对 ✓" if must_pass and must_fail else "**夹具失效，本模式本轮不给结论**"))
    if not must_fail:
        print("   成因排查：探针值出现在——"
              + ("A 结果件 " if probe in idx else "") + ("A2 训练历史 " if probe in hist else "")
              + ("B 登记件（含工单正文！规则文本也被扫）" if probe in bset else ""))
    return 0 if (must_pass and must_fail) else 1


def status_line() -> int:
    """**交付用的那一段指纹，由工具印、不由我手打**（21:5x 自纠：我在投递消息里凭记忆写了两个号——
    一枚不存在的对照表名与一枚旧装配器 sha——被自己复查抓到 ⇒ 这不是粗心能治的，只能让号只有一个来源）。
    打印：指针四行（并现算两枚件的 sha 验相符）、本包写面各件 bytes/sha、tip 与两台远端、两系列顶层枚数。"""
    import subprocess
    def g(*a):
        r = subprocess.run(["git"] + list(a), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=str(REPO))
        return (r.stdout or r.stderr).strip()
    lines = [x.strip() for x in (OUT / "候选正本.txt").read_text(encoding="utf-8").splitlines() if x.strip()]         if (OUT / "候选正本.txt").exists() else []
    print("[指针·现算]")
    if len(lines) >= 4:
        for i, role in ((0, "候选正本"), (2, "当前对照表")):
            f = OUT / lines[i]
            ok = f.exists() and hashlib.sha256(f.read_bytes()).hexdigest().startswith(lines[i + 1])
            print(f"    第{1 if i == 0 else 3}、{2 if i == 0 else 4}行 ＝ {role} {lines[i]}（{lines[i+1]}）"
                  f"⇒ 件存在且 sha 相符 {'✓' if ok else '**✗ 指针与盘上不符**'}｜{f.stat().st_size if f.exists() else 0:,} B")
    else:
        print(f"    **指针不足四行（现 {len(lines)} 行）⇒ 对照表那一半没登记，别投递**")
    for f in (REPO / "docs/revision").glob("*.md"):
        if f.name.startswith(("正文改写工单", "回执-格3")):
            bb = f.read_bytes()
            print(f"[件] {f.name} {len(bb):,} B / sha256 {hashlib.sha256(bb).hexdigest()[:12]}")
    for rel in ("model/scripts/ops/assemble_thesis_docx.py",):
        bb = (REPO / rel).read_bytes()
        print(f"[件] {pathlib.PurePosixPath(rel).name} {len(bb):,} B / sha256 {hashlib.sha256(bb).hexdigest()[:12]}"
              f" ｜blob(套autocrlf) {g('hash-object', rel)[:12]}")
    ml = OUT / "make_ledger.py"
    if ml.exists():
        bb = ml.read_bytes()
        print(f"[件] make_ledger.py {len(bb):,} B / sha256 {hashlib.sha256(bb).hexdigest()[:12]}（仓外件，无 blob 号）")
    print(f"[git] HEAD {g('rev-parse', '--short', 'HEAD')} ｜ origin {g('rev-parse', '--short', 'origin/main')}"
          f" ｜ revision {g('rev-parse', '--short', 'revision/main')} ｜ 工作树脏行数 {len(g('status', '--porcelain').splitlines())}")
    print(f"[顶层枚数] 副本 {len(list(OUT.glob('装配副本-*.docx')))} ｜ 对照表 {len(list(OUT.glob('装配对照表-*.md')))}"
          f" ｜ superseded {len(list((OUT / 'superseded').iterdir())) if (OUT / 'superseded').is_dir() else 0} 条目")
    src_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()[:12]
    print(f"[原件只读] {SRC.name} {SRC.stat().st_size:,} B / {src_sha} / mtime "
          f"{datetime.datetime.fromtimestamp(SRC.stat().st_mtime).isoformat(timespec='seconds')}")
    return 0


def md_gate(target: str) -> int:
    """**把表格闸叫到自己这层来跑，并且先把路径变成绝对路径**（统括官与我对同一句
    `--paths ../` 拿到过 524 与 552 两个 files 数 ⇒ 根因嫌疑就是这个**相对路径**：它随 cwd 变。
    机制＝工具自己打印它扫的是哪个绝对根，报数不带根号的那一行从此不可比较。"""
    import subprocess
    tool = (REPO / "model" / "scripts" / "check_md_tables.py")
    # **相对路径一律按仓根解析，不按 cwd**——本回合我自己就踩了：在 `model/scripts/ops/` 里跑 `--md-gate ../`
    # 会解析成 `model/scripts`（files=0），而统括官在仓根跑同一句得到的是整棵 `D:/PINN-restart`（files=552）。
    # 这就是那 28 个件的全部来历嫌疑，所以机制是：**同一个参数串在任何 cwd 下必须指向同一个根**。
    tp = pathlib.Path(str(target))
    root = (tp if tp.is_absolute() else REPO / tp).resolve()
    if not root.exists():
        print(f"[未验] 表格闸的目标根不存在：{root}")
        return 1
    print(f"[表格闸·带根号] 命令＝python {tool.relative_to(REPO).as_posix()} --paths {root}｜"
          f"跑于 cwd＝{pathlib.Path.cwd()}（**相对参数已按仓根解析，与 cwd 无关**）｜"
          f"闸本体自 4a37251 起未被本次改动碰过")
    r = subprocess.run([sys.executable, str(tool), "--paths", str(root)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(REPO))
    out = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    for line in out[-3:]:
        print("   ", line)
    tail_files = int(out[-1].split("files=")[1].split()[0]) if out and "files=" in out[-1] else -1
    # **报数自带结构**：再把每个顶层目录单独跑一遍，差额就无处可藏
    # （统括官与我的 524／552 之争缺的就是一份逐件分解——总数对了也不知道是谁涨的）。
    parts = []
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        rr = subprocess.run([sys.executable, str(tool), "--paths", str(sub)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(REPO))
        line = ((rr.stdout or "") + (rr.stderr or "")).strip().splitlines()
        tail = next((l for l in reversed(line) if "files=" in l), "")
        f = int("".join(c if c.isdigit() else " " for c in tail.split("files=")[1].split()[0]) or 0)
        blk = int(tail.split("table_blocks=")[1].split()[0]) if "table_blocks=" in tail else -1
        rag = int(tail.split("ragged_rows=")[1].split()[0]) if "ragged_rows=" in tail else -1
        parts.append((sub.name, f, blk, rag))
    sfiles = sum(x[1] for x in parts)
    print("    分目录（files／table_blocks／ragged_rows）：" +
          "；".join(f"{n} {f}/{tk}/{rg}" for n, f, tk, rg in parts))
    nf = sum(1 for p in root.iterdir() if p.is_file() and p.suffix == ".md")
    print(f"    ⇒ 分目录合计 {sfiles} ＋ 根上散件 {nf} 枚＝**{sfiles + nf}**，与整根的 "
          f"{tail_files} 相比：{'一致 ✓（差额已归到具体目录，下次没人再问那 28 个件是谁）' if sfiles + nf == tail_files else f'**不一致，差 {tail_files - sfiles - nf} ⇒ 扫描有重叠或漏，报出来不掩盖**'}")
    print(f"    ⇒ **root={root}** 的数只能与同一根的数比；`../` 这类相对写法作废")
    return r.returncode


def value_prov_boundary():
    """**核对边界（统括官 21:0x 第⑥件：做半可以，但边界要写死）** ⇒ ⇒ 返回
    `(已核表清单, 未核表清单, 每档碰撞地板)`。**已核**＝该表**每一枚**读数都是"位数 d 满足 A 级地板 ≤5%"
    且**原样命中结果件/训练历史**；只要有一枚 ≤4 位或只能靠登记件背书，整张表就落进**未核**。
    这把尺只回答"能不能主张"，不回答"数对不对"——**建了闸 ≠ 全表已校**，所以两清单都必须印出来。"""
    from docx import Document
    p = candidate()
    if p is None:
        return None
    idx, nf = artifact_values()
    hist, nh = history_values()
    import random
    rnd = random.Random(20260927)
    floor = {}
    for d in range(2, 7):
        probes = [f"{rnd.uniform(0.0, 1.0):.{d}f}" for _ in range(300)]
        floor[d] = (sum(1 for q in probes if q in idx) / len(probes),
                    sum(1 for q in probes if q in hist) / len(probes))
    provable = {d for d in floor if floor[d][0] <= 0.05}
    doc = Document(str(p))
    tok = re.compile(r"\d+\.\d{2,}|\d\.\d+[eE][-+]?\d+")
    per = {}
    for ti, tb in enumerate(doc.tables):
        cap = ""
        at = list(doc.element.body).index(tb._tbl)
        for k in range(at - 1, max(-1, at - 8), -1):
            el = doc.element.body[k]
            if el.tag.endswith("}p"):
                s = "".join(x.text or "" for x in el.iter() if x.tag.endswith("}t")).strip()
                if s:
                    cap = s
                    break
        bad = []
        n = 0
        for ri, row in enumerate(tb.rows):
            for ci, cell in enumerate(row.cells):
                for m in tok.finditer(cell.text):
                    s = m.group(0)
                    if cell.text[m.end():m.end() + 1] in ("e", "E"):
                        continue
                    n += 1
                    d = len(s.split(".")[1]) if "." in s and "e" not in s.lower() else 6
                    hit = s in idx or (s in hist and d in provable)
                    if not (hit and d in provable):
                        bad.append((ri, ci, s, d))
        per[ti] = (cap[:24], n, bad)
    ok = [f"表序{ti}（{per[ti][0]}，{per[ti][1]} 枚全为 ≥5 位且原样命中）" for ti in per if per[ti][1] and not per[ti][2]]
    no = [f"表序{ti}（{per[ti][0]}）：{len(per[ti][2])}/{per[ti][1]} 枚不可证" for ti in per if per[ti][2]]
    empty = [f"表序{ti}（{per[ti][0]}）：表内无 ≥2 位小数读数" for ti in per if not per[ti][1]]
    return ok, no, empty, floor, provable


def cell_coords(doc=None):
    """只读：⇒ {行号: (表号, 旧格文本, [(全份表序, 行, 列), …])}。
    一把尺＝**整格文本归一后相等**（与 `--cells` 落字用的同一判据，不留第二份）。"""
    from docx import Document
    if doc is None:
        p = candidate()
        if p is None:
            print("[未验] 候选正本指针缺失/失效 ⇒ 不算坐标（不猜一份副本）")
            return None
        doc = Document(str(p))
    body = list(doc.element.body)
    idx = {id(tb._tbl): ti for ti, tb in enumerate(doc.tables)}
    out = {}
    for rid, (tid, old) in CELL_TEXT_ROWS.items():
        blocks = tables_by_id(doc, body, tid)
        hits = [(idx[id(tb._tbl)], ri, ci) for _, tb in blocks
                for ri, row in enumerate(tb.rows) for ci, c in enumerate(row.cells)
                if norm(c.text) == norm(old)]
        out[rid] = (tid, old, hits, len(blocks))
    return out


def cell_coords_report() -> int:
    """`--cell-coords`：把坐标打成作者可粘的一行，并对**歧义**退 1（0 命中＝表或格变了；≥2 命中＝硬给会粘错）。"""
    cc = cell_coords()
    if cc is None:
        return 1
    bad = []
    for rid, (tid, old, hits, nblk) in sorted(cc.items()):
        if len(hits) != 1:
            bad.append(rid)
        print(f"[格坐标] {rid}｜{tid}（全份共 {nblk} 块，含续表）旧格逐字 {old!r} ⇒ 命中 {len(hits)} 处 "
              f"{hits}（全份表序, 行, 列）⇒ " + ("可粘 ✓" if len(hits) == 1 else "**歧义／已变，交作者点名**"))
    # **必红夹具**：同一把尺喂一枚"两处都有"的串，必须报 ≥2 命中（否则"唯一命中"这句话是空的）
    from docx import Document
    p = candidate()
    if p is not None:
        d = Document(str(p))
        body = list(d.element.body)
        idx = {id(tb._tbl): ti for ti, tb in enumerate(d.tables)}
        dupes = {}
        for tb in d.tables:
            for ri, row in enumerate(tb.rows):
                for ci, c in enumerate(row.cells):
                    k = norm(c.text)
                    if len(k) >= 6:
                        dupes.setdefault(k, []).append((idx[id(tb._tbl)], ri, ci))
        amb = [v for v in dupes.values() if len(v) >= 2]
        probe = next((k for k, v in dupes.items() if len(v) >= 2), None)
        print(f"[必红夹具·唯一命中] 副本里整格文本重复（≥6 字且命中 ≥2）的有 {len(amb)} 组；"
              f"探针取「{(probe or '')[:16]}…」⇒ 命中 {len(dupes.get(probe, []))} 处（必须 ≥2，否则这把尺没有鉴别力）")
        if probe is None or len(dupes[probe]) < 2:
            bad.append("夹具")
    print(f"[格坐标] 唯一命中 {len(cc) - len([b for b in bad if b != '夹具'])}/{len(cc)} 行；"
          + ("全部可粘 ✓" if not bad else f"**{len(bad)} 项不唯一：{bad}**"))
    return 1 if bad else 0


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
        if rid in BODY_SLICE:                             # 混合格子：只落逐字校验过的正文切片
            new, err = sliced_payload(rid, r["new"])
            lost = sorted({norm_id(x) for x in re.findall(r"[图表]\s*\d+(?:-\d+)?", body[i])} -
                          {norm_id(x) for x in re.findall(r"[图表]\s*\d+(?:-\d+)?", new)})
            if err or lost:
                manual.append((rid, err or f"切片会丢掉 {lost} 的正文引用"))
            else:
                auto.append((rid, i, new))
            continue
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
    """闸三：副本正文 ↔ 工单各行「新文本」逐行核；返回不一致清单。"""
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
    print(f"[verify] 命中工单新文本={len(ok)} 未命中={len(miss)} 待批占位={n_await(await_)}")
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
    print(f"结论：不一致 {len(miss)} 行（其中已声明待批 {n_await(await_)} 行不算未完成）"
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
        "extra_rows": [["同机 CFD 单次 Stokes 解（`dsw-2213920`，pin `70317a59…`）", "0.618（=618 ms）",
                        "未测（该趟只跑一具几何）",
                        "n=7、rc≠0 剔除 0 次、min 0.587 / max 0.730；计时前先一次不计时真解（2113 数据行）；"
                        "**新增不替换**：上一行旧主机值与本行不同机，四个加速比仍按旧主机、跨机比值标注"]],
        "note_suffix": ("同机（`dsw-2213920`）单次 Stokes 解 median 618 ms，n=7，min 587 / max 730，pin `70317a59…`；"
                        "一手件 `out/coldtrip_20260927/e5_runs.tsv.summary.json`（928 B、sha256 前缀 `52c717434c098541`），"
                        "它替代 `e5_cfd_price_a0d1375.INVALID_rc127_do-not-cite.json`（旧件仍不得引用）。"
                        "本行只新增、不改上面任何比值；弯曲几何未测 ⇒ 不许由本行外推。"),
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
                 "B 列取自本轮 8 核实例的五种子实测中位 ⇒ A、C 与 B 不同机、不同次，为混合口径，本限定句保留"
                 "（9/27 同机补测已到账，见本表末行；末行是**新增**，不改上面任何比值）。"
                 "K*=B/(A−C) 逐格向上取整，单价一换必整列重算。"),
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


def table_sources():
    """每张新表逐行的来源（件 + 行号）。统括官 15:0x 裁定：**来源列不进论文表**（仓内路径不可出版），
    改由对照表与回复稿承载 ⇒ 这里只有一份声明源，脚本与对照表都读它，不许两处各写一遍。"""
    md = WORK.read_text(encoding="utf-8")
    out = {}
    for name, spec in NEW_TABLES.items():
        header, rows = spec.get("header"), spec.get("rows")
        if spec.get("from_markdown"):
            header, rows = md_block(md, spec["from_markdown"])
        srcs = spec.get("sources") or []
        if spec.get("sources_md"):
            def pick(cell):
                for k, v in spec["sources_md"].items():
                    if k in cell:
                        return v
                return "需人工（该行来源未登记）"
            srcs = [pick(r[0]) for r in rows]
        for er in spec.get("extra_rows") or []:
            srcs = list(srcs) + ["统括官 15:2x 一手读回：`out/coldtrip_20260927/e5_runs.tsv.summary.json`"
                                 "（928 B、sha256 前缀 `52c717434c098541`、instance `dsw-2213920`、pin `70317a59…`）"]
            rows = list(rows) + [er]
        out[name] = [(rows[i][0], srcs[i] if i < len(srcs) else "需人工（该行来源未登记）") for i in range(len(rows))]
    return out


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
        for er in spec.get("extra_rows") or []:          # 一手读数到账：新增行，不替换任何旧行
            rows.append((er + [""] * (width - len(er)))[:width])
        rows = [r + [""] * (width - len(r)) for r in rows]
        note = spec["note"] + (" " + spec["note_suffix"] if spec.get("note_suffix") else "")
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
        note_p = doc.add_paragraph(note)
        # 把刚建的三段搬到锚点前/后（add_* 只会追加到文末）
        els = [cap._p, tab._tbl, note_p._p]
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
    add_t57_note(doc)
    return len(t2.rows)


T57_NOTE = "表内数值由 metrics_test.json 的原始未取整值四舍五入到 6 位。"


def add_t57_note(doc):
    """统括官 15:0x 裁定②：拿展示值反推派生数今天已经算出过一次假错（加速比 30.83 vs 30.7237）
    ⇒ 一句表注堵掉下一次。紧跟表块之后，已有就不重复插（幂等）。"""
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
    if any(T57_NOTE in p.text for p in doc.paragraphs):
        print("   [表注] 表 5-7 的取整注已在 ⇒ 不重复插")
        return
    t2 = next((t for t in doc.tables if len(t.columns) >= 6 and "观测条件" in t.rows[0].cells[2].text
               and any("B-test-2" in c.text for c in t.rows[-1].cells)), None)
    if t2 is None:
        print("   [表注] 没定位到表 5-7 的表块 ⇒ 不猜位置，交人工")
        return
    el = t2._tbl.makeelement(qn("w:p"), {})
    t2._tbl.addnext(el)
    Paragraph(el, t2._parent).add_run(T57_NOTE)
    print(f"   [表注] 已插在表 5-7 之后：{T57_NOTE}")


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
    ref517 = sum(1 for x in ap if "图5-17" in norm_id(x) and not is_caption(x))
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
# 两张孤表（表3-4、表4-1）的指向句走工单行 A18/A19（见 POINTER_ROWS）：9/27 13:5x 那条裁决起初只在项目记忆里、
# 盘上无凭据 ⇒ 我只起草不落字并要求"落到工单条目"；统括官随后写入 §12 第 13 条①＋A18/A19 两行，本轮才落。
# 留下的规矩：**署名用户的裁决必须先归因到件**，且句子只从工单载荷逐字取，脚本里不另存一份文本。



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
    cited = sum(1 for t in txts if "图5-14" in norm_id(t) and not is_caption(t))
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


def id_census(doc):
    """每个号的**正文引用数**（题注严判排除在外）。与 `.scratch/census_refs.py` 同一把尺。"""
    out = {}
    for p in doc.paragraphs:
        t = p.text
        if is_caption(t):
            continue
        for m in re.finditer(r"(表|图)\s*(\d+(?:-\d+)?[a-z]?)", t):
            k = norm_id(m.group(0))                       # 号尺只声明一处：`norm_id`（17:5x 第③条）
            out[k] = out.get(k, 0) + 1
    return out


def add_row_pointers(doc):
    """A18/A19：孤表的指向句。**句子逐字取自工单载荷**（这里不重复声明一份文本），落点＝该号题注**之前**、
    正文段之后（作者拍定的三件见 §12 第 13 条①，删表分支已关闭）。号按整段相等选，防 表5-1 串到 表5-10。"""
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
    n = 0
    for r in rows():
        if r["id"] not in POINTER_ROWS:
            continue
        sent = payload(r["new"]).strip()
        m = re.search(r"(表|图)\s*(\d+(?:-\d+)?[a-z]?)", r["loc"])
        if not m or not sent:
            print(f"   [指向句] {r['id']}：工单行里没解析出表号或载荷为空 ⇒ 不做")
            continue
        tid = m.group(1) + m.group(2)
        for w in FORBID_IN_TRANSITION:
            assert w not in sent, f"{r['id']} 的指向句混进强度词 {w}"
        assert not re.search(r"\d", re.sub(re.escape(tid), "", norm_id(sent))), \
            f"{r['id']} 的指向句带进新数字：{sent}"      # 先归一空格再剔号，否则"表 3-4"躲过剔除、被自己的闸拦下
        ci = None
        for i, p in enumerate(doc.paragraphs):
            mm = re.match(r"^(表|图)(\d+(?:-\d+)?[a-z]?)", norm_id(p.text))
            if mm and mm.group(0) == tid and is_caption(p.text):
                ci = i
                break
        if ci is None:
            print(f"   [指向句] {r['id']}：副本里没有整段等于 {tid} 的题注 ⇒ 不做")
            continue
        if any(tid in norm_id(p.text) and not is_caption(p.text) for p in doc.paragraphs):
            print(f"   [指向句] {r['id']}：{tid} 已有正文引用 ⇒ 不重复插（幂等）")
            continue
        para = doc.paragraphs[ci]
        el = para._p.makeelement(qn("w:p"), {})
        para._p.addprevious(el)                       # 题注之前、正文段之后
        Paragraph(el, para._parent).add_run(sent)
        n += 1
        print(f"   [指向句] {r['id']}：插在 {tid} 题注（原段{ci}）之前 ⇒ {sent}")
    return n


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
    pre = id_census(doc)                      # 插句**之前**的逐号 census：只隔离指针句这一步的影响
    print("   [图]", add_fig514_pointer(doc))
    np_ = add_row_pointers(doc)
    post = id_census(doc)
    want = {"图5-14"} | {norm_id(re.search(r"(表|图)\s*(\d+(?:-\d+)?[a-z]?)", r["loc"]).group(0))
                         for r in rows() if r["id"] in POINTER_ROWS and re.search(r"(表|图)\s*\d", r["loc"])}
    changed = {k for k in set(pre) | set(post) if pre.get(k, 0) != post.get(k, 0)}
    drift = {k: (pre.get(k, 0), post.get(k, 0)) for k in changed if k not in want}
    notplus = {k: (pre.get(k, 0), post.get(k, 0)) for k in changed & want if post.get(k, 0) != pre.get(k, 0) + 1}
    print(f"   [census 前后并排] 变化号 = {sorted(changed)}（期望 {sorted(want)}）；"
          f"每个恰好 +1 = {not notplus}{'（异常 ' + str(notplus) + '）' if notplus else ''}；"
          f"其余号被带动 = {drift or '无'}")
    print("   [census 口径] 本表用**题注严判**（题注不算引用）⇒ 三号读作 0→1；"
          "若把题注算上（作者口径）同三号为 1→2，是同一事实两把尺，不是不吻合")
    if drift or notplus or changed != want:
        print("[INVALID] 插句带动了别的号或没恰好 +1 ⇒ 交人工核对，别当已闭合")
        return 0
    if np_ != len(POINTER_ROWS):
        print(f"[注] 孤表指向句本轮插了 {np_}/{len(POINTER_ROWS)}（其余为「该号已有正文引用」⇒ 幂等跳过，不是失败）")
    stale, stale_cells = stale_hits(doc)          # 与 selfcheck_stale() 同一把尺，不留两份扫描逻辑
    if stale_cells:
        print(f"   [告警·表内旧数] 正文已换新数，但**表格单元里仍有** {stale_cells}（＝工单 D1b，判决＝改数）"
              f"⇒ 扫描面已扩到表内：旧数只藏在表格里也会响")
    if stale:
        print(f"   [告警·正文旧数] 正文段 {stale} 仍印着被禁的消融结论数")
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


def add_limit_rows(doc):
    """J4 这类"新增一句局限"：**句子与锚点都只从工单那一行逐字取**（锚＝「loc」里第一对「…」内的原文），
    脚本里不留第二份文本。约束沿用 E4/图5-17 那档：无数字、无强度词、不许"未来将…"式承诺。"""
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
    n = 0
    for r in rows():
        if r["id"] not in LIMIT_ROWS:
            continue
        sent = payload(r["new"]).strip()
        m = re.search(r"[「『](.{6,}?)[」』]", r["loc"])
        if not m or not sent:
            print(f"   [局限] {r['id']}：载荷或锚句取不到 ⇒ 不做")
            continue
        anchor = m.group(1)[:24]
        for w in FORBID_IN_TRANSITION:
            assert w not in sent, f"{r['id']} 的局限句混进强度词 {w}"
        assert not re.search(r"\d", sent), f"{r['id']} 的局限句带数字：{sent}"
        assert not re.search(r"(未来|后续|下一步).{0,4}(将|会|计划)", sent), f"{r['id']} 写成承诺式：{sent}"
        hi = next((i for i, p in enumerate(doc.paragraphs) if p.text.strip().startswith(anchor)), None)
        if hi is None:
            print(f"   [局限] {r['id']}：副本里找不到以「{anchor}」开头的段 ⇒ 不猜位置")
            continue
        if any(sent in p.text for p in doc.paragraphs):
            print(f"   [局限] {r['id']}：该句已在 ⇒ 幂等跳过")
            continue
        para = doc.paragraphs[hi]
        el = para._p.makeelement(qn("w:p"), {})
        para._p.addnext(el)
        Paragraph(el, para._parent).add_run(sent)
        n += 1
        print(f"   [局限] {r['id']}：插在段{hi}（「{anchor}…」）之后 ⇒ {sent}")
    return n


def cell_values_op(doc):
    """D1b 这类**换承重格里的数**（不是换标签）：三重约束——① 表按题注号整段相等选（含续表一起找完再判唯一）；
    ② 整格相等才认，命中数 != 1 就拒做；③ **新值必须在所引正本那一行里现读得到**（谓词级，不接受我手打的数）。"""
    n = 0
    for rid, (tid, old, new, srcrel, srcline) in CELL_VALUES.items():
        line = (REPO / srcrel).read_text(encoding="utf-8").splitlines()[srcline - 1]
        if new not in line:
            print(f"   [数值] {rid}：正本 {srcrel}:{srcline} 里没有 {new} ⇒ 拒做（新值拿不到出处就是编的）")
            continue
        tbls = []
        for t in doc.tables:
            ids = caption_ids(_caption_above(t, doc))
            if ids == tid:
                tbls.append(t)
        if not tbls:
            print(f"   [数值] {rid}：没有题注恰为 {tid} 的表 ⇒ 不做")
            continue
        hits = [(k, ri, ci, c) for k, t in enumerate(tbls) for ri, r in enumerate(t.rows)
                for ci, c in enumerate(r.cells) if c.text.strip() == old]
        if len(hits) != 1:
            print(f"   [数值] {rid}：{tid}（{len(tbls)} 张，含续表）里整格等于「{old}」的有 {len(hits)} 格 ⇒ 不唯一，交人工")
            continue
        k, ri, ci, cell = hits[0]
        cell.text = new
        n += 1
        print(f"   [数值] {rid}：{tid} 第{k + 1}张 行{ri}列{ci}「{old}」→「{new}」出处 {srcrel}:{srcline}")
    return n


def _caption_above(tbl, doc):
    body = list(doc.element.body)
    at = body.index(tbl._tbl)
    from docx.oxml.ns import qn
    for k in range(at - 1, max(-1, at - 8), -1):
        if body[k].tag.endswith("}p"):
            s = "".join(x.text or "" for x in body[k].findall(f".//{qn('w:t')}")).strip()
            if s:
                return s
    return ""


def tables_by_id(doc, body, tid):
    """按**题注整段相等**选出某号的全部表块（含"（续表）"那张）——一把尺，谁要用谁调。
    题注必须整段等于号（`is_caption` + `norm_id`），前缀法会把 表5-1 串到 表5-10、表4-4 串到 表4-4b。"""
    from docx.oxml.ns import qn
    out = []
    for t in doc.tables:
        at = body.index(t._tbl)
        cap = ""
        for k in range(at - 1, max(-1, at - 8), -1):
            if body[k].tag.endswith("}p"):
                s = "".join(x.text or "" for x in body[k].findall(f".//{qn('w:t')}")).strip()
                if s:
                    cap = s
                    break
        m = re.match(r"^(表|图)(\d+(?:-\d+)?[a-z]?)", norm_id(cap))
        if m and m.group(1) + m.group(2) == tid:
            out.append((cap, t))
    return out


# A11（表 5-8）＝作者 9/27 拍定**丙**：只改三格文字，数值一格不动（§12 第 18 条是正本）。
# 这里的硬保障不是"新标签含旧标签"（丙恰恰不含），而是**改动面只允许是文字格**：
# 任何旧值/新值含数字 ⇒ 立刻抛错，防止手滑把 0.174478 那类读数卷进来。
SPLIT_OF = {"D1c": "I1"}      # 拆分行 → 母行：D1c 的每个句子必须是 I1 载荷里的**逐字子串**

A11_TABLE, A11_CELLS = "表5-8", {"配置": "阶段内残差惩罚项",
                                 "不启用阶段内残差惩罚项": "不启用",
                                 "启用阶段内残差惩罚项": "启用"}


def a11_op(doc, body):
    """A11 丙：表5-8 表头第一格 + 两行行名，三格文字；**其余格逐字不动**（含数值列）。"""
    for o, n in A11_CELLS.items():
        assert not re.search(r"\d", o + n), f"A11 只许改文字格，这格里有数字：{o}→{n}"
    tbls = tables_by_id(doc, body, A11_TABLE)
    if not tbls:
        print(f"   [格] A11：没找到题注恰为 {A11_TABLE} 的表 ⇒ 不做")
        return 0
    cells = {(i, r, c): cell for i, (_cap, tb) in enumerate(tbls)
             for r, row in enumerate(tb.rows) for c, cell in enumerate(row.cells)}
    before = {k: c.text for k, c in cells.items()}
    hits = [k for k, c in cells.items() if c.text.strip() in A11_CELLS]
    if len(hits) != len(A11_CELLS):
        print(f"   [格] A11：{A11_TABLE}（{len(tbls)} 张，含续表）里命中 {len(hits)} 格，应为 {len(A11_CELLS)} 格"
              f" ⇒ 整表不做（哪一列是「列名」要靠命中数判，不靠猜）")
        return 0
    for k in hits:
        cells[k].text = A11_CELLS[before[k].strip()]
    drift = [k for k in cells if k not in hits and cells[k].text != before[k]]
    assert not drift, f"A11 只该动 {len(A11_CELLS)} 格文字，却有 {len(drift)} 个非目标格变了：{drift[:3]}"
    wrong = [k for k in hits if cells[k].text != A11_CELLS[before[k].strip()]]
    if wrong:
        print(f"   [格] A11：{len(wrong)} 格回读不符 ⇒ 不算已落 {wrong[:3]}")
        return 0
    print(f"   [格] A11（丙）：{A11_TABLE} 改了 {len(hits)} 格文字（表头 1 ＋ 行名 2），"
          f"**非目标格 {len(cells) - len(hits)} 格逐字未动**（含全部数值列，回读比对）")
    return len(hits)


def cells_op(copy_path):
    """表内一格换标签（A16 这类）：按**号整段相等**选表、按**整格相等**选格，命中数不是 1 就拒做。
    A11 故意不在这里：它说"列名改为…"，而表 5-8 现在的表头是「配置」——哪一列是"列名"是编辑判断，不猜。"""
    from docx import Document
    from docx.oxml.ns import qn
    doc = Document(str(copy_path))
    body = list(doc.element.body)
    done = 0
    for rid, (tid, old, new) in CELL_OPS.items():
        tbls = tables_by_id(doc, body, tid)
        if not tbls:
            print(f"   [格] {rid}：没找到题注恰为 {tid} 的表 ⇒ 不做")
            continue
        # 同一号可能跨页分几张（"（续表）"），目标格可能在任一张 ⇒ 全部找完再判唯一
        hits = [(cap, ri, ci, c) for cap, t in tbls for ri, row in enumerate(t.rows)
                for ci, c in enumerate(row.cells) if c.text.strip() == old]
        if len(hits) != 1:
            print(f"   [格] {rid}：{tid}（{len(tbls)} 张，含续表）里整格等于「{old}」的有 {len(hits)} 格 ⇒ 不唯一，交人工")
            continue
        cap, ri, ci, cell = hits[0]
        assert old in new, f"{rid} 的新标签不含旧标签 ⇒ 这是改内容不是换标签，拒做"
        cell.text = new
        done += 1
        print(f"   [格] {rid}：{cap[:12]} 行{ri}列{ci}「{old}」→「{new}」（同行其余格未动）")
    nl = add_limit_rows(doc)
    nv = cell_values_op(doc)
    na = a11_op(doc, body)
    doc.save(str(copy_path))
    if nv != len(CELL_VALUES):
        print(f"[INVALID] 承重格数值只换了 {nv}/{len(CELL_VALUES)} ⇒ 出处对不上或命中不唯一，D1b 未闭合")
        return -1
    if na != len(A11_CELLS):
        print(f"[INVALID] A11（丙）只做了 {na}/{len(A11_CELLS)} 格 ⇒ 表5-8 的三格文字没改满，别当已落")
        return -1
    if nl != len(LIMIT_ROWS):
        print(f"[INVALID] 局限句只落了 {nl}/{len(LIMIT_ROWS)} ⇒ 锚点或句子取不到，别当已落")
        return -1
    return done


def selfcheck_guards():
    """两条**豁免夹具**（统括官 15:0x：判得对，不许放开限制；改的是判据粒度，并把两次误拦做成夹具）。
    这两条防的是"尺太粗把自己拦死"，与 selfcheck_caption() 防的"尺太粗把正文误判成题注"成对。"""
    bad = []
    if any(w in TRANSITION_517 for w in FORBID_IN_TRANSITION):
        bad.append(f"TRANSITION_517 被强度词黑名单拦了（图名本身含'倍' ⇒ 黑名单不许锁单字）：{TRANSITION_517}")
    if re.search(r"\d", re.sub(r"图\s*5-17", "", TRANSITION_517.replace("图 5-17", "图5-17"))):
        bad.append("TRANSITION_517 剔掉图号后仍有数字 ⇒ 号位豁免失效")
    r14 = re.sub(re.escape("图5-14"), "", norm_id(FIG514_POINTER))
    if re.search(r"\d", r14.replace("Rel-L2", "").replace("L2", "")):
        bad.append(f"FIG514_POINTER 剔号与指标名后仍有数字：{r14}")
    rowmap = {r["id"]: payload(r["new"]) for r in rows()}
    for kid, mother in SPLIT_OF.items():
        if kid not in rowmap or mother not in rowmap:
            bad.append(f"拆分行 {kid}←{mother}：有一行不在工单里（{kid in rowmap}/{mother in rowmap}）")
            continue
        sents = [s.strip() for s in re.split(r"[。；]", rowmap[kid]) if len(s.strip()) >= 8]
        fake = [s[:24] for s in sents if s not in rowmap[mother]]
        if fake:
            bad.append(f"拆分行 {kid} 造了新句（不在母行 {mother} 载荷里的逐字子串）：{fake}")
        if re.findall(r"\d\.\d{3,}", rowmap[kid]):      # **两条检查各自独立跑**：写成 elif 时"带读数"这一支永远测不到
            bad.append(f"拆分行 {kid} 带着读数（应留在母行那一半）："
                       f"{re.findall(r'\d\.\d{3,}', rowmap[kid])[:4]}")
    if "（启用）" not in CELL_OPS["A16"][2] or CELL_OPS["A16"][2] not in CELL_OPS["A16"][2]:
        bad.append("CELL_OPS 的'新标签必含旧标签'断言被绕开")
    print(f"[豁免夹具] 5 条（倍字界／两处号位豁免／格标签必含旧标签／**拆分行不许造新句也不许带读数**）⇒ " + ("全过 ✓" if not bad else f"**{len(bad)} 条失效**"))
    for x in bad:
        print("   失效：", x)
    return 1 if bad else 0


def selfcheck_fold():
    """必红子检查：拿"只折叠旧词"的坏尺去判一次合法替换，它必须判红。
    9/27 15:2x 从仓外 `.scratch/audit_copy.py` 追进这枚**已跟踪**件——统括官指出：只在临时区就随时会消失，
    而 §12 第 9/10 条引用的正是它的结论 ⇒ 不可复现的控制在盘上等于没有。"""
    old, new = "耦合阶段做低学习率协同修正。", "耦合阶段做低学习率交替更新。"
    good = FOLD_RE.sub("@@", norm(old)) == FOLD_RE.sub("@@", norm(new))
    only_old = re.compile("|".join(re.escape(a) for _, a, _ in TERM_OPS))
    bad = only_old.sub("@@", norm(old)) == only_old.sub("@@", norm(new))
    print(f"[必红子检查·折叠] 好尺(新旧一起折叠)判一致:{good}（应 True）；坏尺(只折旧词)判一致:{bad}（必须 False）")
    if not (good and not bad):
        print("[INVALID] 折叠控制失效 ⇒ 终检的 rc 不可信，先修尺再谈交付")
        return 1
    return 0


DECLARED_ROWS = 56        # 工单数据行的**声明值**（统括官 19:1x 闸一）。增/删行必须同批改这里：
# 这是刻意的摩擦——J3 被 J4 顶掉那次，行数守恒、结构变了，`--plan` 与表格闸全绿。



DECL_FILE = OUT / "本轮声明.txt"          # 每行一个**本轮允许变动**的工单行号（# 开头算注释）


def row_fingerprint(text: str):
    """把工单里的 A-J 数据行做成 {行号: (行内容 sha1, 第几行)}。**只看数据行**：§10/§12 的散文不在闸三的管辖内。"""
    out = {}
    for n, line in enumerate(text.splitlines(), 1):
        m = re.match(r"^\|\s*([A-J]\d+[a-z]?)\s*\|", line)
        if m:
            out[m.group(1)] = (hashlib.sha1(line.encode("utf-8")).hexdigest()[:12], n)
    return out


def selfcheck_undeclared(head_text: str | None = None, work_text: str | None = None, decl: set[str] | None = None):
    """闸三（统括官 19:2x 补）：**未声明的行不得变**。逐行哈希把本轮工作树与 `git show HEAD:` 比，
    任何一行内容变了、却没被写进 `.scratch/本轮声明.txt` ⇒ 红并指名行号与行号。
    这一道看的是"**变没变**"，不是"数对不对" ⇒ 同时拓住我 15:5x 的 J3（整行被顶掉、行数不变）
    与 19:0x 那次（前两格被替）——"行数不丢"那道闸对这两类天然盲。
    参数可注入（`head_text`/`work_text`/`decl`）是为了**夹具能喂谓词本身**，不靠改 git 历史、不靠真工单。"""
    import subprocess
    if head_text is None:
        head_text = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{WORKLIST_REL}"],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    if not head_text:
        print("[闸三·未声明行不得变] HEAD 侧读到空 ⇒ **本条未验**（不拿「没变化」糊过去）")
        return 1
    cur = row_fingerprint(work_text if work_text is not None else WORK.read_text(encoding="utf-8"))
    prev = row_fingerprint(head_text)
    if decl is None:
        decl = {l.strip() for l in DECL_FILE.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.strip().startswith("#")} if DECL_FILE.exists() else set()
    changed = sorted(k for k in set(cur) | set(prev)
                     if cur.get(k, ("", 0))[0] != prev.get(k, ("", 0))[0])
    silent = [k for k in changed if k not in decl]
    unused = sorted(decl - set(changed))
    ok = not silent
    print(f"[闸三·未声明行不得变] 声明 {len(decl)} 个可变动行｜本轮实际变动 {len(changed)} 行 {changed or ''}"
          f"｜**未声明却变了的 {len(silent)} 行：{silent or '无'}** ⇒ " + ("过 ✓" if ok else "**红：这就是「结构变了但数没变」**"))
    for k in silent:
        print(f"    行号：现 {cur.get(k, ('—', 0))[1]}／HEAD {prev.get(k, ('—', 0))[1]}（改它的人请在 {DECL_FILE.name} 里声明，或还原这一行）")
    if unused and ok:
        print(f"    注：声明了却没动的 {unused}（不清不楚只会让下一轮误以为还能随便改）")
    return 0 if ok else 1


def selfcheck_rows():
    """必红子检查（9/27 15:5x，我自己把 J3 整行顶掉之后加的）：**工单少一行就红**。
    那次事故里 `--plan` 与表格闸全绿——因为行数没变、只是 J3 被 J4 换了位置，
    所以"有没有丢行"必须由一条独立计数断言来管，不能指望渲染层发现。"""
    import subprocess
    cur = [r["id"] for r in rows()]
    dup = sorted({i for i in cur if cur.count(i) > 1})
    try:
        head = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{WORKLIST_REL}"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        prev = re.findall(r"^\| ([A-J]\d+[a-z]?) \|", head.stdout, flags=re.M)
    except Exception as e:                                    # 拿不到 HEAD 就明说"没测到"，不假装绿
        print(f"[必红子检查·行数] 读不到 HEAD 工单（{e}）⇒ **本条未验**，不算通过")
        return 1
    if not prev:                       # 恒真闸的形态：基线读到 0 行 ⇒ "丢行"永远为空，必须自己红
        print("[必红子检查·行数] HEAD 侧读到 **0 行** ⇒ 基线取不到，本条判未验（不是通过）")
        return 1
    lost = sorted(set(prev) - set(cur))
    extra = sorted(set(cur) - set(prev))
    # **闸一**：与声明值不等即红（HEAD 对比只防"净丢行"，防不了"等量替换"——J3/J4 就是那样漏过去的）
    declared = len(cur) == DECLARED_ROWS
    ok = declared and not lost and not dup and len(cur) >= len(prev)
    print(f"[闸一·工单行数] 数据行 声明 {DECLARED_ROWS} ｜HEAD {len(prev)} → 工作树 {len(cur)}"          f"；丢行 {lost or '无'}｜相对 HEAD 新增 {extra or '无'}｜重号 {dup or '无'} ⇒ "              + ("等于声明且无丢行无重号 ✓" if ok else f"**不成立（改行数必须同批改 DECLARED_ROWS）**"))
    if not ok:
        print(f"[INVALID] 行数断言不成立：丢行 {lost or '无'}｜相对 HEAD 新增 {extra or '无'}｜重号 {dup or '无'}"
              f"｜声明值 {DECLARED_ROWS} vs 现 {len(cur)} ⇒ 渲染层看不出来，必须在这里红")
        return 1
    return 0


def caption_ids(text: str):
    """从一行文本里抽**整段相等**的表/图号；抽不出返回 None。所有工具共用这一份抽号规则，
    免得每个工具各写一遍 `startswith` —— 今天同族已经犯两次（表5-1 吞 表5-10、表4-4 吞 表4-4b）。"""
    m = re.match(r"^(表|图)(\d+(?:-\d+)?[a-z]?)", norm_id(text))
    return (m.group(1) + m.group(2)) if m else None


def resolve_caption(paras, tid: str):
    """按号取题注段号：只认**整段相等**，且用题注严判（防把正文句当题注）。"""
    return [i for i, t in enumerate(paras) if is_caption(t) and caption_ids(t) == tid]


def selfcheck_ids(copy_path=None):
    """必过夹具（统括官 16:4x：把"整段相等"从教训升级成尺）：两对**必须不同**的号，解析结果不许相交。
    **必须跑在副本上**——`表5-10 / 表4-4b` 是本轮新建的表，原件里根本没有它们；
    而副本里 表4-4 的题注在段 228、表4-4b 在段 229（紧挨着）⇒ 这正是前缀法必撞的位置，夹具才咬得住。
    找不到副本 ⇒ 判**未验**并返回 1，不假装绿。"""
    from docx import Document
    if copy_path:
        p = pathlib.Path(str(copy_path))
        if not p.exists():
            print(f"[必过夹具·号整段相等] 指定的副本不在场：{p.name} ⇒ **未验**")
            return 1
    else:
        # **不猜"最新一份"**：一串时间戳交付物里 mtime 最大的往往是那枚 部件级 diff=0 的重打包件，
        # 不是候选正本（统括官 18:0x 的规矩）⇒ 只认 `.scratch/候选正本.txt` 指针，缺失/失效即判未验。
        p = candidate()
        if p is None:
            print("[必过夹具·号整段相等] 既没传副本、候选正本指针也缺失/失效 ⇒ **未验**（新表号只存在于副本，跑原件＝空门）")
            return 1
    paras = [x.text for x in Document(str(p)).paragraphs]
    bad = []
    for a, b in (("表5-1", "表5-10"), ("表4-4", "表4-4b")):
        ra, rb = resolve_caption(paras, a), resolve_caption(paras, b)
        if not ra or not rb:
            bad.append(f"{a}/{b} 有一号解析为空（{ra}/{rb}）⇒ 副本不对或抽号规则变了")
        elif set(ra) & set(rb):
            bad.append(f"{a} 与 {b} 解析到同一段 {sorted(set(ra) & set(rb))}")
        naive = [i for i, t in enumerate(paras) if norm_id(t).startswith(norm_id(a)) and is_caption(t)]
        if set(naive) == set(ra):
            bad.append(f"{a}：前缀法与整段法同解 ⇒ 这个副本里夹具没有鉴别力（换一枚含兄弟号的副本）")
    print(f"[必过夹具·号整段相等] 跑在 {p.name} 上，两对必不同 ⇒ " + ("全过 ✓" if not bad else f"**{len(bad)} 条失效**"))
    for x in bad:
        print("   失效：", x)
    return 1 if bad else 0


BANNED_VALUES = ("0.5612", "11.7598")


def stale_hits(doc):
    """被禁旧值的命中，**正文段与表格单元分开数**（统括官 16:4x 第③条：只扫正文会盖住表里的旧读数）。"""
    paras = [i for i, p in enumerate(doc.paragraphs) if any(v in p.text for v in BANNED_VALUES)]
    cells = [(ti, ri, ci) for ti, t in enumerate(doc.tables) for ri, r in enumerate(t.rows)
             for ci, c in enumerate(r.cells) if any(v in c.text for v in BANNED_VALUES)]
    return paras, cells


def selfcheck_stale():
    """正对照：闸必须在一件"旧值只活在表格里"的实物上响过，否则"表格单元也扫了"这句话没有凭据。
    天然正对照就是**原件**——表5-1 那格 `0.5612` 在原件里就是表格单元、正文没有。"""
    from docx import Document
    op, oc = stale_hits(Document(str(SRC)))
    cp = [p] if (p := candidate()) else []          # 不猜"最新"：读候选正本指针（统括官 18:0x）
    rp, rc = stale_hits(Document(str(cp[0]))) if cp else ([], [])
    # 正对照只要求一件事：**旧值躺在表格单元里时必须被扫到**（原件正文里本来也有那处旧句，那是 D1 未改前的原文，
    # 不是这条夹具要判的东西——我第一版误加了"正文必须 0 命中"，把夹具写成了永远红）
    ok = len(oc) >= 1
    # **口径（统括官 9/27 20:5x 第①条）**：禁令只有 `BANNED_VALUES` 那两枚。**别把"旧值清零"写成"全文 0 命中"**——
    # `0.0390` 在**表5-5 行3（test 行）列2** 是另一量的合法读数，拿它判残留会把这道闸写成永远红、或逼人把它拆掉。
    print(f"[正对照·作用域点名] 禁令清单＝{list(BANNED_VALUES)}（**只有这两枚**）；"
          f"`0.0390` **不在禁令内**（表5-5 行3 的 test 读数是合法另一量）⇒「清零」这句只许限定到表5-1 那两格")
    print(f"[正对照·旧值只在表格里必须响] 原件：正文 {len(op)} 处、表格单元 {len(oc)} 处 {oc[:2]} ⇒ "
          + ("闸有效 ✓" if ok else "**失效（要么没扫到表内，要么正文也漏了）**"))
    print(f"    候选正本（按指针）{cp[0].name if cp else '（无）'}：正文 {len(rp)} 处、表格单元 {len(rc)} 处"
          + ("（D1b 落字后应为 0）" if cp else ""))
    return 0 if ok else 1


def selfcheck_terms_scope():
    """必红夹具（统括官 16:5x 第③条，与"扫描域扩到表内"是同一个洞的另一个出口）：
    **把一个短形旧词只放在表格单元里，"旧词已清"的断言必须响**。天然正对照＝原件——
    表5-1（续表）里那格「启用阶段内PDE」是全称之外的截短形，全称匹配扫不到它。"""
    from docx import Document
    short = re.compile("|".join(re.escape(a) for _, a, _ in TERM_OPS))
    src = Document(str(SRC))
    cells = [c.text for t in src.tables for r in t.rows for c in r.cells if short.search(c.text)]
    paras = [p.text for p in src.paragraphs if short.search(p.text)]
    cp = [p] if (p := candidate()) else []          # 不猜"最新"：读候选正本指针（统括官 18:0x）
    now = ""
    if cp:
        d2 = Document(str(cp[0]))
        n = sum(1 for t in d2.tables for r in t.rows for c in r.cells if short.search(c.text)) \
            + sum(1 for p in d2.paragraphs if short.search(p.text))
        now = f"；候选正本（按指针）{cp[0].name} 两层合计残留 {n} 处（应为 0）"
    ok = len(cells) >= 1
    print(f"[必红夹具·旧词只在表格里必须响] 原件：正文 {len(paras)} 段、表格单元 {len(cells)} 格命中旧词 ⇒ "
          + ("尺覆盖表内 ✓" if ok else "**没覆盖：'旧词已清'这句没有凭据**") + now)
    return 0 if ok else 1


def selfcheck_rowids():
    """必红夹具（统括官 16:5x 第④条）：**往工单里塞一行带字母后缀的新行，`rows()` 的计数必须 +1**。
    喂谓词不喂扫描器——真的临时改 `WORK` 再调 `rows()`，不是把正则抄一遍来测。
    探针号取**没被占用**的后缀号：`D1c` 于 9/27 18:0x 已是真工单行（I1 拆出的 表5-7 半句），
    再拿它当探针就成"重复行"而不是"新行"，夹具会假过。"""
    global WORK
    base = [r["id"] for r in rows()]
    probe = next(c for c in ("D1c", "D1d", "J9z", "B9z") if c not in base)
    real, tmp = WORK, WORK.with_name("_rowid_probe.md")
    try:
        tmp.write_text(real.read_text(encoding="utf-8")
                       + f'| {probe} | 探针行（不是真工单行，跑完即删） | 探针 | 探针 | 探针 | 否 |' + chr(10),
                       encoding="utf-8")
        WORK = tmp
        got = [r["id"] for r in rows()]
    finally:
        WORK = real
        tmp.unlink(missing_ok=True)
    ok = (probe in got) and len(got) == len(base) + 1
    suffix = [b for b in base if re.search(r"[a-z]$", b)][:6]
    print(f"[必红夹具·带后缀行号必须被数到] 探针号 {probe}（基线里已存在的后缀号 {suffix}）"
          f" ⇒ 基线 {len(base)} 行 → 塞入后 {len(got)} 行、命中={probe in got}")
    if not ok:
        print("[INVALID] 分类器看不见带字母后缀的行 ⇒ 新加的行会静默消失，拒出对照表")
    return 0 if ok else 1



def selfcheck_pointer():
    """必红子检查（统括官 16:2x 第②③条）：**取件自拒**。喂的是 `candidate()` 谓词本身，
    不是把它的判据重抄一遍——在一枚临时 `.scratch` 假目录里造六种局面，只许"指针＝声明的候选正本、
    且顶层没有未登记副本"那一发过，其余五发必须全退 None。跑完删临时目录（那是我本轮造的探针件）。"""
    global OUT
    real = OUT
    out = []
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ptr_probe_"))
    try:
        OUT = tmp
        good = tmp / "装配副本-探针.docx"
        good.write_bytes(b"probe-copy-bytes-not-a-docx")
        sha = hashlib.sha256(good.read_bytes()).hexdigest()[:12]

        def ptr(text):
            (tmp / "候选正本.txt").write_text(text, encoding="utf-8")

        def shot(label, should_pass):
            out.append((label, should_pass, candidate() is not None))

        ptr(good.name + chr(10) + sha + chr(10))                           # 1 必过
        shot("基线：顶层只有声明件", True)
        s_old = tmp / "装配副本-探针-旧.docx"
        s_old.write_bytes(b"x")
        os.utime(s_old, (good.stat().st_atime, good.stat().st_mtime - 7200))   # 2 他点名的那一发
        shot("顶层扔一枚**比候选正本更旧**的副本", False)
        s_old.unlink()
        s_new = tmp / "装配副本-探针-新.docx"
        s_new.write_bytes(b"x")
        os.utime(s_new, (good.stat().st_atime, good.stat().st_mtime + 7200))   # 3 更新了却没登记
        shot("顶层扔一枚更新的副本（新交付件没改指针）", False)
        s_new.unlink()
        ptr("装配副本-不存在.docx" + chr(10) + sha + chr(10))               # 4 件不在
        shot("指针指向不存在的件", False)
        ptr(good.name + chr(10) + "0" * 12 + chr(10))                      # 5 sha 不符
        shot("指针 sha 与件不符", False)
        ptr("../" + SRC.name + chr(10) + sha + chr(10))                    # 6 相对路径／够到原件
        shot("指针写歪成相对路径（会够到原件）", False)
    finally:
        OUT = real
        import shutil as _sh
        _sh.rmtree(tmp, ignore_errors=True)
    n_pass = sum(1 for _, sp, _ in out if sp)
    bad = [f"{lb}：期望 {'取' if sp else '拒'}，实际 {'取' if gp else '拒'}"
           for lb, sp, gp in out if sp != gp]
    print(f"[必红子检查·取件自拒] 六发逐字喂 `candidate()`（{n_pass} 发应过／{len(out) - n_pass} 发应拒）⇒ "
          + ("全中 ✓，『扔一枚旧副本必须报未验』这一发真的响了" if not bad else f"**{len(bad)} 发判错**"))
    for x in bad:
        print("   失效：", x)
    return 0 if not bad else 1


def selfcheck_counts() -> int:
    r"""第十六条·必红子检查（统括官 9/28 00:1x：换了尺就得带上新尺的必红）——**「三数同框」的作用域真在起作用吗**。
    四发，**两枚 needle 都由现跑挑出来**（挑不到就报"未验"并退 1，不硬印、也不写死字面）：
    ① **只在条文出现**的串 ⇒ 数据行内必须 = 0、条文自身 > 0；
    ② **只在数据行出现**的号 ⇒ 数据行内必须 ≥ 1、条文自身必须 = 0；
    ③ 恒等式：数据行内 ＋ 条文自身 = 全文件裸跑（两发各校）；
    ④ **变异钩**：把作用域退化成 §10 那条旧写法（BRE 下 `\|` 是"或" ⇒ 匹配所有表行）⇒ 退化后必须与真尺不同；
       两档相同就说明我们自己的作用域也没生效（他那句"只能靠人不自红"的实物版）。"""
    txt = WORK.read_text(encoding="utf-8")
    ROW = re.compile(r"^\|\s*[A-J]\d+[a-z]?\s*\|")
    lines = txt.splitlines()
    prose = [ln for ln in lines if not ROW.match(ln)]
    rows = [ln for ln in lines if ROW.match(ln)]
    # 旧 §10 那条 `^\| [A-J]` 在 GNU BRE 下的实际作用域：**所有以 "| " 开头的行**（表头、分隔行、非 A-J 的回复表行都算）
    rows_bre = [ln for ln in lines if ln.startswith("| ")]
    other_tbl = [ln for ln in rows_bre if not ROW.match(ln)]          # 表行之中、但不在 A-J 数据行作用域内的那部分
    # ④的 needle：**只在"其他表行"出现**的号——真尺该数到 0，退化尺（旧 BRE）必须数到 >0；两把尺答一样＝对照是空的
    toks4 = [m.group(0) for ln in other_tbl for m in re.finditer(r"\d\.\d{3,}", ln)]

    def counts(needle, pool):
        return sum(len(re.findall(re.escape(needle), ln)) for ln in pool)

    counts_at = counts
    n4 = next((c for c in dict.fromkeys(toks4) if counts_at(c, other_tbl) > 0 and counts_at(c, rows) == 0), None)

    # ①挑一枚"只在条文"的串：候选都是登记件里常见的散文词，**逐个测两档**，第一枚满足 (0, >0) 的才被采用
    cands = ["冻结收口", "已落待抄", "判决理由排序更正", "作用域", "统括官 9/27 裁决", "三数同框", "条同一件事"]
    n1 = next((c for c in cands if counts(c, rows) == 0 and counts(c, prose) > 0), None)
    # ②挑一枚"只在数据行"的号：从真实数据行里刮 4 位以上的小数，逐个测，取第一枚 (≥1, 0)
    toks = [m.group(0) for ln in rows for m in re.finditer(r"\d\.\d{3,}", ln)]
    n2 = next((c for c in dict.fromkeys(toks) if counts(c, rows) >= 1 and counts(c, prose) == 0), None)
    bad = []
    for needle, kind in ((n1, "条文独有"), (n2, "数据行独有")):
        if needle is None:
            print(f"[闸·三数同框] **未验**：挑不出「{kind}」那一枚 needle ⇒ 这一发没测到（不硬印，也不换判据凑）")
            bad.append(kind)
            continue
        a, q, raw = needle_counts(needle, txt)
        c1 = (a == 0 and q > 0) if kind == "条文独有" else (a >= 1 and q == 0)
        c3 = (a + q == raw)
        print(f"[闸·三数同框] needle={needle!r}（{kind}，现跑挑出）⇒ 数据行内 = {a} ｜条文自身 = {q} ｜裸跑 = {raw}")
        print(f"    ①{'数据行内应 0 且条文 > 0' if kind == '条文独有' else '数据行内应 ≥1 且条文 = 0'} ⇒ {c1}"
              f" ｜③ 两半相加＝裸跑 ⇒ {c3}")
        if not c1:
            bad.append(f"{kind}:作用域档判错")
        if not c3:
            bad.append(f"{kind}:三数不守恒")
    # ④ 变异钩单列：真尺数不到（不在 A-J 作用域内）、放宽成"所有表行"后一定数得到；两把尺同答＝这一发是空的
    if n4 is None:
        print("[闸·三数同框] **未验**：挑不出「只在非 A-J 表行」的号 ⇒ 第④发没测到（不硬印）")
        bad.append("变异钩无 needle")
    else:
        a4 = counts(n4, rows)
        b4 = counts(n4, rows_bre)
        print(f"[闸·三数同框·④变异钩] needle={n4!r}（只在非 A-J 的表行）⇒ 真尺（A-J 作用域）= {a4}（应 0）"
              f" ｜放宽尺（所有 `| ` 表行）= {b4}（应 > 0）｜两把尺不同答 = {a4 != b4}"
              f" ｜注：§10 原文那条在 GNU BRE 下 `\\|`＝“或”⇒匹配的是整本，比这里更宽，两码别混称")
        if not (a4 == 0 and b4 > 0):
            bad.append("④变异钩没咬（永绿或挑错）")
    print(f"[闸·三数同框] " + ("四发全对 ✓（needle 一律 re.escape、作用域只声明在 `needle_counts()` 一处）"
                            if not bad else "**失效：" + "；".join(bad) + "**"))
    return 1 if bad else 0


def selfcheck_all(copy_path=None):
    """十六条子检查一次跑完；只声明一处，`--selfcheck/--verify/--all` 三处入口共用。"""
    return (selfcheck_caption() or selfcheck_guards() or selfcheck_fold()
            or selfcheck_console() or selfcheck_rows() or selfcheck_ids(copy_path)
            or selfcheck_stale() or selfcheck_terms_scope() or selfcheck_rowids()
            or selfcheck_normid() or selfcheck_table_shapes(copy_path)
            or selfcheck_directives() or selfcheck_undeclared() or selfcheck_pointer()
            or selfcheck_caliber() or selfcheck_counts())


CALIBER_ROWS = {"D1b": ("mean-of-cases", "T5矩阵test口径读数", ("0.5684348", "0.0471148"))}


def caliber_violations(new: str, proof: str, spec) -> list:
    """**谓词只这一处**。分两栏判是因为这三样的读者不同：
    ①**口径名**与②**所引正本（件名＋页码）**必须在**载荷**里——那半句会被粘进正文，读者要看见口径；
    ③**pooled 对照值**在本行的**凭据列**在场即可——它不进正文，是给下一个核数的人防「拿 pooled 当矛盾报」。
    三条任一缺席即报一条违规。断言原文在 §10 的断言行与 §12 第 25 条，这道闸只是让它会抛错。"""
    name, src, pooled = spec
    v = []
    if name not in new:
        v.append(f"载荷缺口径名「{name}」（换口径不写进要粘进正文的那半句＝读者只能猜）")
    if src not in new:
        v.append(f"载荷缺所引正本（「{src}」＋页码不在要粘的那半句里）")
    if not all(p in new or p in proof for p in pooled):
        v.append("本行凭据列缺 pooled 对照值 ⇒ 下一人会拿 pooled 当矛盾报")
    return v


def selfcheck_caliber() -> int:
    """第十五条·必红子检查（统括官 9/27 17:5x 第②条）：**「表内读数换了口径 ⇒ 新值必须自带口径声明」这条不能只是散文。**
    对**真工单行**跑谓词（必过），再把三条判据各拆掉一次跑（**三发都必须红**）——少红一发就说明这道闸是空的。"""
    seen = {r["id"]: (payload(r["new"]), r.get("why", "")) for r in rows()}
    bad = []
    for rid, spec in sorted(CALIBER_ROWS.items()):
        p, proof = seen.get(rid, ("", ""))
        if not p:
            print(f"[闸·口径声明] {rid}：工单里找不到这一行 ⇒ **未验**")
            bad.append(rid)
            continue
        v = caliber_violations(p, proof, spec)
        print(f"[闸·口径声明] {rid} 真行：{len(v)} 条违规（应为 0）" + ("✓ 三条齐（口径名＋正本页码＋pooled 对照值）" if not v else " " + "；".join(v)))
        bad += [f"{rid}:{x}" for x in v]
        # 三发必红：各拆一条判据，谓词必须报出来
        cuts = {"拆掉口径名": lambda s, q: (s.replace(spec[0], "平均值"), q),
                "拆掉正本页码": lambda s, q: (s.replace(spec[1], "某个读数件"), q),
                "拆掉 pooled 值": lambda s, q: (s, q.replace(spec[2][0], "?").replace(spec[2][1], "?"))}
        for label, f in cuts.items():
            s2, q2 = f(p, proof)
            got = caliber_violations(s2, q2, spec)
            if not got:
                bad.append(f"夹具 {rid}/{label}")
                print(f"    **夹具失效**：{label} 之后谓词仍不报 ⇒ 这条判据是摆设")
            else:
                print(f"    必红 {label:12s} ⇒ 报 {len(got)} 条 ✓（{got[0][:34]}）")
    print(f"[闸·口径声明] 真行 {'全过' if not [x for x in bad if not x.startswith('夹具')] else '有缺'}、"
          f"必红 {len(CALIBER_ROWS) * 3} 发{'全中' if not bad else f'**{len(bad)} 处失效：{bad[:3]}**'}")
    return 1 if bad else 0


def selfcheck_normid():
    """必红夹具（统括官 9/27 17:5x 第③条）：**同一号的几种写法必须判为同一号**。
    喂的是 `norm_id` 本身，不把比较重抄一遍：造三种载体（半角紧贴／正文里带空格／全角空格＋全角短横），
    归一尺必须三发全中，**不归一的旧尺必须少中一发**——否则这条断言在测一个不存在的东西。"""
    a = "表3-4"
    texts = [f"{a}  采样设置与选点规则", "数据预处理各步的口径汇总于表 3-4。", "见 表　3–4 的第三行"]
    got = sum(1 for t in texts if a in norm_id(t))
    naive = sum(1 for t in texts if a in t)
    ok = got == len(texts) and naive < len(texts)
    print(f"[必红夹具·号判定先归一] 归一尺 `norm_id()` 数到 {got}/{len(texts)}（应满）、不归一旧尺只有 {naive}"
          f"（必须少，否则夹具空转）⇒ " + ("空格／全角空格／全角短横三种写法同号 ✓" if ok
                                          else "**尺没咬住：'某号有没有被引用'会漏，孤号判定不可信**"))
    return 0 if ok else 1


def selfcheck_console():
    """必红子检查（统括官 15:0x 抓到）：夹具在**默认中文终端代码页**下不许崩。
    他裸跑 `python3 -c "...selfcheck_caption()"` 报 `UnicodeEncodeError: 'gbk' codec can't encode '\u21d2'`
    ⇒ "必红控制能在真实终端条件跑"当时没成立。这里用 `PYTHONIOENCODING=gbk:strict` 固定复现那台终端，
    并断言三条夹具都退 0（模块顶层已 reconfigure，所以裸 import 也不会崩）。"""
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING="gbk:strict")
    code = ("import sys; sys.path.insert(0, {0!r}); import assemble_thesis_docx as A; "
            "sys.exit(A.selfcheck_caption() or A.selfcheck_fold() or A.selfcheck_guards())").format(str(HERE.parent))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    crash = "UnicodeEncodeError" in (r.stdout or "") + (r.stderr or "")
    print(f"[必红子检查·终端代码页] PYTHONIOENCODING=gbk:strict 下三条夹具 rc={r.returncode} 编码崩={crash}"
          + ("（应 rc=0、崩=False）" if not (r.returncode == 0 and not crash) else " ✓"))
    if r.returncode or crash:
        print("   末行输出：", ((r.stdout or "") + (r.stderr or "")).strip().splitlines()[-3:])
        return 1
    return 0


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
    g.add_argument("--cells", type=pathlib.Path, help="在给定副本上改表内标签格（A16 这类「换标签」，整格唯一命中才做）")
    g.add_argument("--status", action="store_true",
                    help="只读：把投递该带的指纹块打出来（号只有一个来源＝这条命令的输出，不许手打）")
    g.add_argument("--md-gate", metavar="路径", help="把表格闸跑在这个**绝对化后的根**上并打印根号（消灭『--paths ../ 随 cwd 变』那一类不可比）")
    g.add_argument("--prov-boundary", action="store_true",
                    help="只读：把 39 张表分成「已可主张」与「不可主张（短位数／靠登记件背书）」两张清单")
    g.add_argument("--value-prov", action="store_true",
                    help="只读：论文表格里的每个数对一遍仓内结果件，报命中率与未命中坐标（含两发夹具）")
    g.add_argument("--cell-coords", action="store_true",
                    help="只读：把『改一格文字』那类工单行（CELL_TEXT_ROWS）解析成全份（表序,行,列）坐标，命中≠1 即退 1")
    g.add_argument("--count-needle", metavar="串", help="三数同框：数据行内／条文自身／全文件裸跑")
    g.add_argument("--selfcheck", action="store_true", help="跑十六条子检查（题注／豁免／折叠／代码页／行数／号整段相等／旧值域／旧词域／后缀行号／号归一／新表形状／§12 指令扫整条／未声明行不得变／取件自拒／换口径必带声明）")
    g.add_argument("--all", action="store_true",
                   help="一把跑完整链：新建副本 → 整写/术语/插段 → 三张新表（**不含来源列**，15:0x 裁定：来源由对照表承载）→ 表5-7 → 5.7 成对块。顺序固定，防每轮手接不同次序")
    ap.add_argument("--into", type=pathlib.Path, default=None,
                    help="--apply 用：写进指定副本（--all 串链用），不给就新建一个时间戳文件")
    ap.add_argument("--expect-changed", type=int, default=None,
                    help="--apply 用：期望被改段落数，不接等即 INVALID（闸三）")
    args = ap.parse_args()
    # **只读数、不碰 docx 的模式先分流**（统括官 9/28 00:1x：他在没装 python-docx 的解释器里跑 `--count-needle`
    # 直接崩在库缺失上 ⇒ 一个只读文本的正则凭什么要写作库？这条挪动本身就是那发"要一条必红"的前半。）
    if getattr(args, "count_needle", None):
        return count_needle(args.count_needle)
    import docx  # noqa: F401  ② 先确认库在，不在就别硬写
    from docx import Document

    if getattr(args, "status", None):
        return status_line()
    if getattr(args, "md_gate", None):
        return md_gate(args.md_gate)
    if getattr(args, "prov_boundary", None):
        bb = value_prov_boundary()
        if bb is None:
            print("[未验] 候选正本指针缺失/失效 ⇒ 不出边界清单")
            return 1
        ok, no, empty, floor, provable = bb
        print("[核对边界] A 级可证的位数档：" + "、".join(f"{d} 位（地板 {floor[d][0]:.0%}）" for d in sorted(provable))
              + "｜不可证档：" + "、".join(f"{d} 位（地板 {floor[d][0]:.0%}）" for d in sorted(floor) if d not in provable))
        print(f"    **已可主张 {len(ok)} 张**：")
        for x in ok:
            print("       ✓", x)
        print(f"    **不可主张 {len(no)} 张（每枚 ≤4 位读数的字符串命中不足为证 ⇒ 要 per-cell 归属表）**：")
        for x in no:
            print("       ✗", x)
        print(f"    无小数读数、不适用本尺 {len(empty)} 张：" + ("；".join(empty) or "无"))
        print(f"    合计 {len(ok)} + {len(no)} + {len(empty)} = {len(ok) + len(no) + len(empty)} 张表"
              f"（≠「全表已校」——**已校的是 {len(ok)} 张那一档**）")
        return 0
    if getattr(args, "value_prov", None):
        return value_prov_report()
    if args.cell_coords:
        return cell_coords_report()
    if args.selfcheck:
        return selfcheck_all()
    if args.verify:
        if selfcheck_all():   # 同上：尺先自证，再谈终检结论
            return 3
        return verify(args.verify)
    if args.all:
        import subprocess
        if selfcheck_all():   # 夹具不过 ⇒ 整条链不开跑（统括官 13:0x：不能靠每次手看）
            return 3
        st = SRC.stat()
        print(f"[原件只读] {st.st_size:,} B mtime={datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')} "
              f"sha256={hashlib.sha256(SRC.read_bytes()).hexdigest()[:12]}")
        dst = OUT / f"装配副本-{datetime.datetime.now():%Y%m%dT%H%M%S}.docx"
        shutil.copy2(SRC, dst)
        here = str(pathlib.Path(__file__).resolve())
        for step in (["--apply-into", str(dst)], ["--cells", str(dst)], ["--tables", str(dst)],
                     ["--t57", str(dst)], ["--pair57", str(dst)], ["--figs", str(dst)]):
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

    if args.cells:
        if not args.cells.exists():
            print(f"[INVALID] 副本不存在：{args.cells}", file=sys.stderr)
            return 2
        n = cells_op(args.cells)
        if n != len(CELL_OPS):
            print(f"[INVALID] 表内标签算子只做了 {n}/{len(CELL_OPS)} ⇒ 命中不唯一或表没找到，别当已落")
            return 1
        return 0

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
          f"5.7 成对块={'E2+E3+E4 齐' if eblk else '不齐 ⇒ 整块不推'} 待批={n_await(await_)} 人工={len(manual)}")
    for num, line in s12_directives():            # 散文条目里的指令也要每轮见面（详见该函数docstring）
        print(f"   [§12 指令·本轮必读] 第 {num} 条：{line}")
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
