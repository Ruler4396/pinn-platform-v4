"""经典压力重建基线（Stage 5 的对照臂）。判据见
paper-route2/速度推压力-经典基线同框-预注册设计-20261003.md §一/§五。

做法（稳态不可压动量方程的直接读法，PIV 压力重建里的标准一族，不需要任何压力测量、没有可调配的超参）：
    ∇p = ∇²u − Re·(u·∇)u          （star 单位，ν=1，对流系数取 Re——与真值 .edp 同一约定，R2-7 已否掉除法版）
  1) 用观测点上的速度做**薄板样条**补全（φ=r²ln r + 一次多项式尾巴；尺度无关 ⇒ 没有核参数可调）；
  2) 同一核的解析导数直接给出右端 f = ∇²u − Re·(u·∇)u（不在有限元里算 ∇²u：P1 场逐元素二阶导恒为 0，
     弱式投影出来的是单元边界跳变——C9 用它喂精确稠密真值时去均值误差 0.91–1.0，那是装置错不是方法性质。
     也不用多二次核：1689 点条件数 1.1e20，被自己的条件数闸拒）；
  3) 右端按 dof 序交给 FreeFEM 的 P1 空间，解梯度匹配最小二乘 ∫∇p·∇q = +∫f·∇q（自然边界 ⇒ 定到常数为止）；
  4) 按真值 .edp 那个打印循环逐顶点输出 x,y,p,bc_tag ⇒ 与 field_dense.csv 逐点同序，用同一把尺评。

C 档：
  C10 已知答案正对照——解析取 u=(αx, −αy)（散度自由、∇²u=0）⇒ f = −Re·α²(x,y) ⇒ p = −Re·α²(x²+y²)/2，
      整条管线（插值→解析导数→按 dof 序传数组→梯度匹配 LS→打印）须恢复到去均值相对误差 ≤1e-2；
      并配一条必红：把对流项系数符号翻掉，误差必须跳大到 >0.2。
  C11 输入只允许 (x,y,u,v)——观测表里出现任何有限压力值即拒绝运行。
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path

C10_TOL = 1.0e-2
C10_RED_TOL = 0.2


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def read_truth(dense: Path) -> list[dict]:
    with open(dense, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"[FAIL] 真值场是空的：{dense}")
    return rows


def read_obs_velocity_only(obs: Path) -> list[tuple[float, float, float, float]]:
    """C11＋C6：压力列必须一个有限值都没有；坐标必须能在真值场里逐位找到。"""
    with open(obs, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    pcol = "p_obs" if rows and "p_obs" in rows[0] else "p_star"
    leak = sum(1 for r in rows if _finite(r.get(pcol, "")))
    if leak:
        raise SystemExit(f"[FAIL] C11：观测表 {obs.name} 的压力列有 {leak} 个有限值——"
                         "这一臂的定义就是不给压力，泄漏了整条判决就不成立")
    out = []
    for r in rows:
        x, y = float(r["x_star"]), float(r["y_star"])
        u, v = float(r["u_obs" if "u_obs" in r else "u_star"]), float(r["v_obs" if "v_obs" in r else "v_star"])
        out.append((x, y, u, v))
    if not out:
        raise SystemExit(f"[FAIL] 观测表为空：{obs}")
    return out


RBF_KERNEL = "tps"      # 薄板样条 φ(r)=r²ln r；多二次核 ε=1.0 在 1689 点上证伪（条件数 1.1e20）
RBF_FLOOR = 1.0e-12    # r→0 时 ln 的地板，避免 -inf


def _rbf_solve(pts, vals, kernel=RBF_KERNEL):
    """薄板样条 φ(r)=r²ln r + 一次多项式尾巴（线性场被精确复现，C10 依赖这一点）。"""
    import numpy as np

    P = np.asarray(pts, dtype=np.float64)
    y = np.asarray(vals, dtype=np.float64)
    n = len(P)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
    d2 = np.maximum(d2, RBF_FLOOR)
    k = 0.5 * d2 * np.log(d2)                     # = r² ln r
    poly = np.hstack([np.ones((n, 1)), P])
    A = np.zeros((n + 3, n + 3), dtype=np.float64)
    A[:n, :n] = k
    A[:n, n:] = poly
    A[n:, :n] = poly.T
    b = np.zeros(n + 3, dtype=np.float64)
    b[:n] = y
    cond = float(np.linalg.cond(A))
    if not cond < 1.0e10:
        raise SystemExit(f"[FAIL] 插值矩阵条件数 {cond:.3g} 过大（点数 {n}，核={kernel}）——"
                         "这是装置问题，别靠正则化糊过去")
    sol = np.linalg.solve(A, b)
    return sol[:n], sol[n:], cond


def rbf_eval(pts, w, polyv, targets, kernel=RBF_KERNEL, want_deriv=False):
    """在 targets 上求值；want_deriv=True 时给解析的一阶与二阶导数。
    φ=r²ln r（记 ρ=r²）：φ=½ρlnρ, ∂x=(lnρ+1)dx, ∂²x=lnρ+1+2dx²/ρ。"""
    import numpy as np

    P = np.asarray(pts, dtype=np.float64)
    T = np.asarray(targets, dtype=np.float64)
    dx = T[:, None, 0] - P[None, :, 0]
    dy = T[:, None, 1] - P[None, :, 1]
    rho = np.maximum(dx ** 2 + dy ** 2, RBF_FLOOR)
    lr = np.log(rho)
    phi = 0.5 * rho * lr
    val = phi @ w + np.hstack([np.ones((len(T), 1)), T]) @ polyv
    if not want_deriv:
        return val
    dvx = ((lr + 1.0) * dx) @ w + polyv[1]
    dvy = ((lr + 1.0) * dy) @ w + polyv[2]
    d2x = (lr + 1.0 + 2.0 * dx ** 2 / rho) @ w
    d2y = (lr + 1.0 + 2.0 * dy ** 2 / rho) @ w
    return val, dvx, dvy, d2x, d2y


def rbf_complete(pts, vals, targets, kernel=RBF_KERNEL):
    w, pv, cond = _rbf_solve(pts, vals, kernel)
    return rbf_eval(pts, w, pv, targets, kernel), cond


def local_deriv(pts, vals, targets, k=10):
    """邻域二次最小二乘求导（PIV 压力重建里实际在用的做法）。
    不用整体样条的解析导数：它把量级抹掉了——稠密真值速度喂进来时
    corr(p_base,p_true)=0.895 但幅度只有真值的 10.6%；63 点的已知答案检查更是一阶导相对误差 21.4。"""
    import numpy as np

    P = np.asarray(pts, dtype=np.float64)
    T = np.asarray(targets, dtype=np.float64)
    y = np.asarray(vals, dtype=np.float64)
    kk = int(max(6, min(k, len(P))))
    gx = np.zeros(len(T))
    gy = np.zeros(len(T))
    for i in range(len(T)):
        d2 = ((P - T[i]) ** 2).sum(axis=1)
        j = np.argpartition(d2, kk - 1)[:kk]
        dx = P[j, 0] - T[i, 0]
        dy = P[j, 1] - T[i, 1]
        A = np.column_stack([np.ones(kk), dx, dy, dx * dx, dy * dy, dx * dy])
        sol = np.linalg.lstsq(A, y[j], rcond=None)[0]
        gx[i] = sol[1]
        gy[i] = sol[2]
    return gx, gy


def pressure_rhs(pts, obs_u, obs_v, targets, reynolds, convection, kernel=RBF_KERNEL):
    """经典臂右端走**压力泊松形式**：对动量方程取散度，
        −Δ(∇·u) + Δp + Re·∇·((u·∇)u) = 0，∇·u = 0 ⇒  **Δp = −Re·∇·((u·∇)u)**，
    黏性项整体退出（它只通过 ∇·u 进来），于是右端**只需要一阶导数**：
    f = convection·Re·(u·∇)u，弱式 `∫∇p·∇q + ∫f·∇q = 0`（自然边界 ⇒ 定到常数为止）。
    为什么不走 ∇p = ∇²u − Re(u·∇)u 那条梯度匹配：散点二阶导在这个细长域上不可靠
    （--deriv-check 实测纯拉普拉斯项相对误差 4.4e2），且 C9 喂**精确稠密真值**时去均值压力仍 0.185/0.91
    ⇒ 那是求导器的错，不是方法的性质（登记为设计件 §八 更5）。
    导数怎么给：值用薄板样条补全到节点，**导数在补全后的节点场上做邻域二次最小二乘**（local_deriv）。
    convection=0 那一臂（Stokes／写错方程）在此形式下退化成 Δp=0 的调和方程：
    速度-only 又不给压力边值时，线性路线对压力不提供信息——这句要随读数一起写，
    不许写成"基线在这一臂上被我们打败了"。"""
    wu = _rbf_solve(pts, obs_u, kernel)
    wv = _rbf_solve(pts, obs_v, kernel)
    cond = max(wu[2], wv[2])
    u = rbf_eval(pts, wu[0], wu[1], targets, kernel)
    v = rbf_eval(pts, wv[0], wv[1], targets, kernel)
    ux, uy = local_deriv(targets, u, targets)
    vx, vy = local_deriv(targets, v, targets)
    conv_u = u * ux + v * uy
    conv_v = u * vx + v * vy
    f1 = convection * reynolds * conv_u
    f2 = convection * reynolds * conv_v
    return f1, f2, cond


def mesh_preamble(cfd_edp: Path) -> str:
    """网格前段**逐字符取自该工况自己的求解件**（到 `mesh Th = buildmesh(...);` 为止）。
    硬编过一次就出过事：收缩比 beta 是逐工况不同的，拿 C-base 的 beta 去解 C-val 会造出另一张网格。"""
    text = cfd_edp.read_text(encoding="utf-8", errors="replace")
    lines, keep = text.split("\n"), []
    for ln in lines:
        keep.append(ln)
        if ln.strip().startswith("mesh Th") and ln.strip().endswith(";"):
            return "\n".join(keep) + "\n"
    raise SystemExit(f"[FAIL] {cfd_edp} 里找不到 `mesh Th = buildmesh(...);` 这一行，无法复用同一张网格")


EDP_NODES_TEMPLATE = """// 节点顺序导出（由 assim_pressure_baseline.py 生成）——网格前段取自该工况自己的 .edp
{preamble}
ofstream fo("{nodes}");
fo << "dof,x,y" << endl;
for (int i = 0; i < Th.nv; ++i) {{
  fo << i << "," << Th(i).x << "," << Th(i).y << endl;
}}
cout << "NODES-OK nv=" << Th.nv << endl;
"""


def build_nodes_edp(workdir: Path, preamble: str, nodes_path: Path) -> Path:
    edp = workdir / "nodes_dump.edp"
    edp.write_text(EDP_NODES_TEMPLATE.format(preamble=preamble, nodes=str(nodes_path)),
                   encoding="utf-8", newline="\n")
    return edp


def dump_mesh_order(workdir: Path, preamble: str) -> list[tuple[float, float]]:
    """节点顺序必须由 FreeFEM 自己说；两臂之间唯一的接口是"按 dof 下标赋值"，所以顺序错了就是错的。
    缓存名取网格前段的 sha16 ⇒ 换工况（换收缩比）不可能静默复用旧清单。"""
    import hashlib

    tag = hashlib.sha256(preamble.encode("utf-8")).hexdigest()[:16]
    nodes = workdir / ("mesh_nodes_%s.csv" % tag)
    if not nodes.exists():
        edp = build_nodes_edp(workdir, preamble, nodes)
        run_freefem(edp, workdir / ("nodes_dump_%s.log" % tag), marker="NODES-OK")
        print("MESH-ORDER dumped -> %s (网格前段 sha16=%s)" % (nodes.name, tag), flush=True)
    else:
        print("MESH-ORDER reuse %s (网格前段 sha16=%s)" % (nodes.name, tag), flush=True)
    with open(nodes, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for i, r in enumerate(rows):
        if int(r["dof"]) != i:
            raise SystemExit(f"[FAIL] 节点清单的 dof 不连续（第 {i} 行写着 {r['dof']}）")
        out.append((float(r["x"]), float(r["y"])))
    print("MESH-ORDER nv=%d 首点=(%g,%g) 末点=(%g,%g)" % (len(out), out[0][0], out[0][1],
                                                          out[-1][0], out[-1][1]), flush=True)
    return out


def _literal(name: str, values) -> str:
    """逐下标赋值：FreeFEM 4.9 的 `real[int] a = [ ... ]` 字面量有 **1024 个参数上限**
    （本网格 2113 个顶点，一条字面量就编不过：Sorry number of parameters > 1024）。"""
    parts = ["%s[%d]=%.10g;" % (name, i, float(v)) for i, v in enumerate(values)]
    return "\n".join(" ".join(parts[i:i + 8]) for i in range(0, len(parts), 8))


EDP_TEMPLATE = """// 经典压力重建（Stage 5 基线）——由 assim_pressure_baseline.py 生成，不要手改
{preamble}
fespace Qh(Th, P1);
Qh p, q, f1, f2;
int NOD = Th.nv;
if (NOD != {n_nodes}) {{
  cout << "NODE-MISMATCH nv=" << NOD << " want={n_nodes}" << endl;
  exit(1);
}}
real[int] fv1(NOD);
real[int] fv2(NOD);
{f1_lit}
{f2_lit}
f1[] = fv1;
f2[] = fv2;
solve pP(p, q) = int2d(Th)(dx(p)*dx(q) + dy(p)*dy(q) + 1.0e-10*p*q) + int2d(Th)(f1*dx(q) + f2*dy(q));

int[int] vTag(Th.nv);
for (int i = 0; i < Th.nv; ++i) vTag[i] = 0;
for (int be = 0; be < Th.nbe; ++be) {{
  int iv0 = Th.be(be)[0];
  int iv1 = Th.be(be)[1];
  int lab = Th.be(be).label;
  if (vTag[iv0] == 0 || ((lab == 1 || lab == 2) && vTag[iv0] == 3)) vTag[iv0] = lab;
  if (vTag[iv1] == 0 || ((lab == 1 || lab == 2) && vTag[iv1] == 3)) vTag[iv1] = lab;
}}
ofstream fo("{pred}");
fo << "x_star,y_star,p_star,bc_tag" << endl;
for (int i = 0; i < Th.nv; ++i) {{
  real xx = Th(i).x;
  real yy = Th(i).y;
  fo << xx << "," << yy << "," << p(xx,yy) << "," << vTag[i] << endl;
}}
cout << "ASSIM-OK nv=" << Th.nv << " tag={stem_tag}" << endl;
"""


def build_edp(workdir: Path, case: str, preamble: str, f1_vals, f2_vals, pred: Path,
              stem_tag: str = "") -> Path:
    text = EDP_TEMPLATE.format(preamble=preamble, n_nodes=len(f1_vals),
                               f1_lit=_literal("fv1", f1_vals), f2_lit=_literal("fv2", f2_vals),
                               pred=str(pred), stem_tag=stem_tag)
    edp = workdir / ("assim_%s.edp" % case)
    edp.write_text(text, encoding="utf-8", newline="\n")
    return edp


def run_freefem(edp: Path, log: Path, marker: str = "ASSIM-OK") -> str:
    proc = subprocess.run(["FreeFem++", "-nw", str(edp)], capture_output=True, text=True, timeout=900)
    log.write_text((proc.stdout or "") + "\n=== STDERR ===\n" + (proc.stderr or ""),
                   encoding="utf-8", newline="\n")
    if proc.returncode != 0 or marker not in (proc.stdout or ""):
        raise SystemExit("[FAIL] FreeFEM 没打印 %s（rc=%s），日志见 %s" % (marker, proc.returncode, log))
    return proc.stdout


def read_pred(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _coord_key(x, y):
    return (round(float(x), 5), round(float(y), 5))


def score(pred_rows, truth_rows, mesh_xy, labels=(1, 2)) -> dict:
    """去均值压力相对误差 + 压降相对误差（后者逐字符照 train_supervised.compute_case_metrics:639-640）。
    基线按 dof 序出、真值按 CSV 行出 ⇒ 对齐靠坐标逐点配对，**配不上就报缺哪几点**，不许改成最近邻。"""
    import numpy as np

    if len(pred_rows) != len(mesh_xy):
        raise SystemExit(f"[FAIL] 基线行数 {len(pred_rows)} != 网格顶点数 {len(mesh_xy)}")
    xp = np.array([float(r["x_star"]) for r in pred_rows])
    yp = np.array([float(r["y_star"]) for r in pred_rows])
    xm = np.array([c[0] for c in mesh_xy])
    ym = np.array([c[1] for c in mesh_xy])
    rel_off = float(np.max((np.abs(xp - xm) + np.abs(yp - ym)) / (np.maximum(np.abs(xm), np.abs(ym)) + 1.0)))
    if not rel_off <= 1.0e-5:
        raise SystemExit(f"[FAIL] 基线打印的坐标与网格清单不符（最大相对差 {rel_off:.3g}）")
    tmap = {}
    for r in truth_rows:
        k = _coord_key(r["x_star"], r["y_star"])
        if k in tmap:
            raise SystemExit(f"[FAIL] 真值场里有重复坐标 {k}，对齐尺不唯一")
        tmap[k] = r
    order, missing = [], 0
    for c in mesh_xy:
        r = tmap.get(_coord_key(c[0], c[1]))
        if r is None:
            missing += 1
            if missing <= 3:
                print("MISSING 网格顶点 (%.6g,%.6g) 在真值场里没有" % c)
            continue
        order.append(r)
    if missing:
        raise SystemExit(f"[FAIL] 网格顶点有 {missing} 个不在真值场里（共 {len(mesh_xy)}）——两把尺不同网格")
    p = np.array([float(r["p_star"]) for r in pred_rows])
    pt = np.array([float(r["p_star"]) for r in order])
    bc = np.array([int(r["bc_tag"]) for r in pred_rows])
    # 真值场用的是文字标签（field_dense.csv 的 boundary_type，含 interior），基线沿用 .edp 的数字标签
    lab = {"inlet": 1, "outlet": 2, "wall": 3}
    census = {}
    for r in order:
        b = r["boundary_type"]
        census[b] = census.get(b, 0) + 1
    unmapped = sorted(b for b in census if b not in lab)
    print("BOUNDARY-CENSUS %s（未映射的按 interior=0 处理，只有 inlet/outlet 进压降）"
          % " ".join("%s=%d" % (k, v) for k, v in sorted(census.items())), flush=True)
    for b in unmapped:
        if census[b] > 0 and b not in ("interior",):
            raise SystemExit(f"[FAIL] 出现未知边界标签 {b!r}（{census[b]} 个点），"
                             "不能默认它不参与压降")
    bc_t = np.array([lab.get(r["boundary_type"], 0) for r in order])
    if not np.array_equal(bc, bc_t):
        raise SystemExit("[FAIL] 边界标记不同序（bc_tag 不一致）")
    pm = p - p.mean()
    ptm = pt - pt.mean()
    denom = float(np.linalg.norm(ptm)) + 1.0e-12
    rel_p_meanfree = float(np.linalg.norm(pm - ptm) / denom)
    in_t, out_t = bc_t == labels[0], bc_t == labels[1]
    in_p, out_p = bc == labels[0], bc == labels[1]
    dp_t = float(pt[in_t].mean() - pt[out_t].mean())
    dp_p = float(p[in_p].mean() - p[out_p].mean())
    return {
        "rel_l2_p_meanfree": rel_p_meanfree,
        "rel_l2_p_raw": float(np.linalg.norm(p - pt) / (float(np.linalg.norm(pt)) + 1.0e-12)),
        "pressure_drop_truth": dp_t,
        "pressure_drop_pred": dp_p,
        "pressure_drop_rel_error": float(abs(dp_p - dp_t) / (abs(dp_t) + 1.0e-12)),
        "n": len(pred_rows),
    }


def rbf_selftest() -> int:
    """纯 numpy 的已知答案正对照（不需要 FreeFEM）：一次多项式尾巴 ⇒ 线性场必须被精确复现。"""
    import random

    import numpy as np

    rng = random.Random(7)
    pts = [(rng.uniform(0, 16), rng.uniform(-0.5, 0.5)) for _ in range(63)]
    tg = [(rng.uniform(0, 16), rng.uniform(-0.5, 0.5)) for _ in range(400)]
    for name, f, tol in (("线性 2x-3y+1", lambda x, y: 2 * x - 3 * y + 1, 1e-8),
                         ("常数", lambda x, y: 7.0, 1e-10)):
        got, cond = rbf_complete(pts, [f(x, y) for x, y in pts], tg)
        err = float(np.max(np.abs(got - np.array([f(x, y) for x, y in tg]))))
        print("RBF-SELFTEST %s: max_err=%.3g cond=%.3g -> %s"
              % (name, err, cond, "OK" if err <= tol else "BAD"))
        if not err <= tol:
            return 1
    return 0


def solve_chain(workdir: Path, stem: str, preamble: str, f1, f2, pred: Path, tag: str = "") -> list[dict]:
    edp = build_edp(workdir, stem, preamble=preamble, f1_vals=f1, f2_vals=f2, pred=pred,
                    stem_tag=tag or stem)
    run_freefem(edp, workdir / (stem + ".log"))
    return read_pred(pred)


def c10(workdir: Path, mesh_xy, preamble: str, reynolds=10.0, alpha=1.0) -> int:
    """已知答案正对照 + 一条必红控制。解析取 u=(αx,−αy)：散度自由、∇²u=0 ⇒ f=−Re·α²(x,y)
    ⇒ p = −Re·α²(x²+y²)/2（差一个常数，正好是被去掉的那个）。"""
    import random

    import numpy as np

    rng = random.Random(0)
    idx = rng.sample(range(len(mesh_xy)), 63)                   # 与观测预算同量级的点数
    obs = [mesh_xy[i] for i in idx]
    p_true = np.array([-reynolds * alpha * alpha * (x * x + y * y) / 2.0 for x, y in mesh_xy])
    p_true = p_true - p_true.mean()
    fails = []
    # 生成的 .edp 里**只准出现 ASCII**：中文进字符串字面量会把 FreeFEM 的词法器搞断
    # （实测 "End of String could not be found"，位置就在最后那行 cout）。中文只留在 python 侧打印。
    for convection, tag, want_pass in ((1.0, "ok", True), (-1.0, "red", False)):
        label = "正对照" if want_pass else "必红"
        pts = obs
        f1, f2, cond = pressure_rhs(pts, [alpha * x for x, _ in pts], [-alpha * y for _, y in pts],
                                    mesh_xy, reynolds, convection)
        pred = workdir / ("c10_pred_%s.csv" % tag)
        rows = solve_chain(workdir, "c10_" + tag, preamble=preamble, f1=f1, f2=f2, pred=pred,
                            tag="conv=%+g cond=%.3g" % (convection, cond))
        pm = np.array([float(r["p_star"]) for r in rows])
        pm = pm - pm.mean()
        rel = float(np.linalg.norm(pm - p_true) / (np.linalg.norm(p_true) + 1e-12))
        ok = rel <= C10_TOL
        print("C10 %s: conv=%+g nv=%d 去均值压力相对误差=%.4g 期望=%s -> %s"
              % (label, convection, len(rows), rel,
                 ("≤%g" % C10_TOL) if want_pass else (">%g" % C10_RED_TOL),
                 "OK" if ok == want_pass else "BAD"), flush=True)
        if want_pass and not ok:
            fails.append("正对照不过（%.4g > %g）⇒ 装置不可信" % (rel, C10_TOL))
        if (not want_pass) and not (rel > C10_RED_TOL):
            fails.append("必红控制没红（把对流系数翻号后误差只有 %.4g）⇒ 对流项根本没进装置" % rel)
    return 1 if fails else 0


def deriv_selftest(reynolds=10.0) -> int:
    """二阶导数的已知答案正对照（C10 的线性场让 ∇²u 恒为 0，那条路没被走过，所以 C9 错了才发现）。
    解析取 u=(x²y, −xy²)：散度自由；∇²u=(2y,−2x)；(u·∇)u=(x³y², x²y³) ⇒ f 全解析。"""
    import numpy as np

    # 点位用**格子**取，不用拒绝采样：定义域 14×0.8 里要塞 63 个"彼此至少 0.25"的点，
    # 面积上限 11.2 < 所需 12.3 ⇒ 拒绝采样永不终止（上一版就在这儿挂死，pkill 掉的）。
    xs = [1.0 + i * (14.0 - 1.0) / 8 for i in range(9)]
    ys = [-0.4 + j * 0.8 / 6 for j in range(7)]
    pts = [(x, y) for x in xs for y in ys]
    txs = [2.0 + i * (14.0 - 2.0) / 19 for i in range(20)]
    tys = [-0.3 + j * 0.6 / 19 for j in range(20)]
    tg = [(x, y) for x in txs for y in tys]
    obs_u = [x * x * y for x, y in pts]
    obs_v = [-x * y * y for x, y in pts]
    f1, f2, cond = pressure_rhs(pts, obs_u, obs_v, tg, reynolds, 1.0)
    a1 = np.array([2 * y - reynolds * x ** 3 * y ** 2 for x, y in tg])
    a2 = np.array([-2 * x - reynolds * x ** 2 * y ** 3 for x, y in tg])
    e1 = float(np.max(np.abs(f1 - a1)) / (np.max(np.abs(a1)) + 1e-12))
    e2 = float(np.max(np.abs(f2 - a2)) / (np.max(np.abs(a2)) + 1e-12))
    # 一阶导数单独核一遍：走**臂实际用的那条路**（样条补全值 → 邻域二次最小二乘求导）
    u_val = rbf_eval(pts, *_rbf_solve(pts, obs_u)[:2], tg)
    v_val = rbf_eval(pts, *_rbf_solve(pts, obs_v)[:2], tg)
    uxa, uya = local_deriv(tg, u_val, tg)
    vxa, vya = local_deriv(tg, v_val, tg)
    xs = np.array([x for x, _ in tg])
    ys = np.array([y for _, y in tg])
    errs = []
    for got, want in ((uxa, 2 * xs * ys), (uya, xs ** 2), (vxa, -(ys ** 2)), (vya, -2 * xs * ys)):
        errs.append(float(np.max(np.abs(got - want)) / (np.max(np.abs(want)) + 1e-12)))
    d1 = max(errs)
    print("DERIV-CHECK n_obs=%d n_tgt=%d cond=%.3g 邻域最小二乘一阶导最大相对误差=%.4g f1=%.4g f2=%.4g"
          % (len(pts), len(tg), cond, d1, e1, e2), flush=True)
    bad = [m for m, v in (("f1", e1), ("f2", e2), ("d1", d1)) if v > 1e-2]
    print("DERIV_SELFTEST " + ("ALL GREEN" if not bad else "FAILED | 超界：" + ",".join(bad)))
    return 1 if bad else 0


def run_case(args, truth_rows, obs, outdir: Path, workdir: Path, mesh_xy, preamble: str) -> dict:
    stem = "assim_%s_%s_%s" % (args.case, args.quota, args.equation)
    pred = outdir / (stem + "_pred.csv")
    convection = 1.0 if args.equation == "ns" else 0.0
    # 常规档与 C9(full) 走同一条路：观测集 = 该档那批点（full 就是全部顶点），
    # 插值核是薄板样条（多二次核在 1689 点条件数 1.1e20，被闸拒；换核后 C10 重新核过）
    pts = [(o[0], o[1]) for o in obs]
    f1, f2, cond = pressure_rhs(pts, [o[2] for o in obs], [o[3] for o in obs], mesh_xy,
                                args.reynolds, convection)
    rows = solve_chain(workdir, stem, preamble=preamble, f1=f1, f2=f2, pred=pred,
                       tag="%s/%s/%s cond=%.3g" % (args.case, args.quota, args.equation, cond))
    res = score(rows, truth_rows, mesh_xy)
    res.update(case=args.case, quota=args.quota, equation=args.equation, reynolds=args.reynolds,
               n_obs=len(obs), interp_cond=cond, pred_csv=str(pred))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/workspace/pinn-repro-2026")
    ap.add_argument("--case", help="工况目录 id，如 C-val_ns_re10")
    ap.add_argument("--quota", default="5pct")
    ap.add_argument("--equation", choices=("ns", "stokes"), default="ns")
    ap.add_argument("--reynolds", type=float, default=10.0)
    ap.add_argument("--selftest", action="store_true", help="C10 已知答案正对照 + 必红（需 FreeFEM）")
    ap.add_argument("--rbf-check", action="store_true", help="只核插值那一半（需 numpy，不需 FreeFEM）")
    ap.add_argument("--deriv-check", action="store_true", help="核二阶导数（∇²u）的已知答案，纯 numpy")
    ap.add_argument("--outdir", default="")
    a = ap.parse_args()
    root = Path(a.root)
    outdir = Path(a.outdir) if a.outdir else root / "model" / "results" / "baseline"
    workdir = outdir / "_work"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    if a.rbf_check:
        return rbf_selftest()
    if a.deriv_check:
        return deriv_selftest(reynolds=a.reynolds)
    if not a.case:
        raise SystemExit("[FAIL] 要 --case：网格前段必须取自该工况自己的 .edp（收缩比逐工况不同）")
    cdir = root / "model" / "cases" / "contraction_2d" / "data" / a.case
    base = re.sub(r"_ns_re[0-9][0-9eE.\-]*$", "", a.case)
    cfd_edp = root / "model" / "cases" / "contraction_2d" / "cfd" / base / ("%s_stokes.edp" % base)
    if not cfd_edp.exists():
        raise SystemExit(f"[FAIL] 找不到该工况的求解件 {cfd_edp}，没有它就无法复用同一张网格")
    preamble = mesh_preamble(cfd_edp)

    # 网格顺序先取（C10 与常规格都靠它；缓存名是网格前段的 sha16，换工况不会静默复用）
    mesh_xy = dump_mesh_order(workdir, preamble)

    if a.selftest:
        rc = rbf_selftest()
        if rc:
            return rc
        return c10(workdir, mesh_xy, preamble, reynolds=a.reynolds)

    truth = read_truth(cdir / "field_dense.csv")
    if a.quota == "full":
        # C9 上限对照：这一档**故意**读完整稠密真值速度（等价配额→100%），用来证明装置本身能做到。
        # 它不是对照臂的读数，只是装置的体检；quota 字段会原样落进产物名，读者一眼能分。
        obs = [(float(r["x_star"]), float(r["y_star"]),
                float(r["u_star"]), float(r["v_star"])) for r in truth]
        print("C9-FULL 用完整稠密真值速度（%d 点）喂装置；这不是基线臂，是装置体检" % len(obs), flush=True)
    else:
        obs = read_obs_velocity_only(cdir / ("obs_sparse_%s_velocity_only.csv" % a.quota))
    res = run_case(a, truth, obs, outdir, workdir, mesh_xy, preamble)
    print("BASELINE " + " ".join("%s=%s" % (k, v) for k, v in res.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
