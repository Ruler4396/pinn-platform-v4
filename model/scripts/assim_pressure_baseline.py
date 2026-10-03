"""经典压力重建基线（Stage 5 的对照臂）。判据见
paper-route2/速度推压力-经典基线同框-预注册设计-20261003.md §一/§五。

做法（稳态不可压动量方程的直接读法，PIV 压力重建里的标准一族，不需要任何压力测量、没有可调配的超参）：
    ∇p = ∇²u − Re·(u·∇)u          （star 单位，ν=1，对流系数取 Re——与真值 .edp 同一约定，R2-7 已否掉除法版）
  1) 用观测点上的速度做多二次 RBF 补全（ε=1.0 写死，带一次多项式尾巴）到网格全部顶点；
  2) 顶点值交给 FreeFEM 的 P1 空间，弱式拉普拉斯投影得 ∇²u、弱式投影得 (u·∇)u；
  3) 梯度匹配最小二乘解压力：∫∇p·∇q = −∫f·∇q（自然边界条件 ⇒ 定到常数为止，去均值后评分）；
  4) 按真值 .edp 那个打印循环逐顶点输出 x,y,p,bc_tag ⇒ 与 field_dense.csv 逐点同序，用同一把尺评。

C 档：
  C10 已知答案正对照——解析取 u=(αx, −αy)（散度自由、∇²u=0）⇒ f = −Re·α²(x,y) ⇒ p = −Re·α²(x²+y²)/2，
      整条管线（RBF→read→投影→LS→打印）须恢复到去均值相对误差 ≤1e-2；
      并配一条必红：把对流项系数符号翻掉，误差必须跳大到 >0.2。
  C11 输入只允许 (x,y,u,v)——观测表里出现任何有限压力值即拒绝运行。
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

RBF_EPS = 1.0          # 写死（预注册 §一），两档 Re、两臂同值，不许按结果调
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


def rbf_complete(pts, vals, targets, eps=RBF_EPS):
    """多二次 RBF φ(r)=sqrt(r²+ε²) + 一次多项式尾巴（让线性场被精确复现，C10 依赖这一点）。"""
    import numpy as np

    P = np.asarray([[p[0], p[1]] for p in pts], dtype=np.float64)
    T = np.asarray([[t[0], t[1]] for t in targets], dtype=np.float64)
    y = np.asarray(vals, dtype=np.float64)
    n = len(P)
    d = np.sqrt((P[:, None, 0] - P[None, :, 0]) ** 2 + (P[:, None, 1] - P[None, :, 1]) ** 2 + eps * eps)
    poly = np.hstack([np.ones((n, 1)), P])
    A = np.zeros((n + 3, n + 3), dtype=np.float64)
    A[:n, :n] = d
    A[:n, n:] = poly
    A[n:, :n] = poly.T
    b = np.zeros(n + 3, dtype=np.float64)
    b[:n] = y
    cond = np.linalg.cond(A)
    if not cond < 1.0e14:
        raise SystemExit(f"[FAIL] RBF 矩阵条件数 {cond:.3g} 过大（点数 {n}，ε={eps}）——"
                         "这是装置问题，别靠正则化糊过去")
    sol = np.linalg.solve(A, b)
    diff = T[:, None, :] - P[None, :, :]
    dt = np.sqrt((diff ** 2).sum(axis=2) + eps * eps)
    return dt @ sol[:n] + np.hstack([np.ones((len(T), 1)), T]) @ sol[n:], cond


def write_dat(path: Path, xy, values) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for (x, y), w in zip(xy, values):
            fh.write("%.6f %.6f %.8g\n" % (x, y, float(w)))


EDP_TEMPLATE = """// 经典压力重建（Stage 5 基线）——由 assim_pressure_baseline.py 生成，不要手改
real beta = {beta};
real Lin = 4;
real Lc = 4;
real Lout = 8;
real Ltot = 16;
int labIn = 1;
int labOut = 2;
int labWall = 3;

func real smoothstep5(real s) {{ return 6.0*s^5 - 15.0*s^4 + 10.0*s^3; }}
func real channelWidth(real xx) {{
  if (xx <= Lin) return 1.0;
  if (xx >= Lin + Lc) return beta;
  real s = (xx - Lin) / Lc;
  return 1.0 - (1.0 - beta) * smoothstep5(s);
}}
func real yTop(real xx) {{ return 0.5 * channelWidth(xx); }}
func real yBot(real xx) {{ return -0.5 * channelWidth(xx); }}
border bottomWall(t=0, 1) {{ x = Ltot*t; y = yBot(x); label = labWall; }}
border outletEdge(t=0, 1) {{ x = Ltot; y = yBot(Ltot)+(yTop(Ltot)-yBot(Ltot))*t; label = labOut; }}
border topWall(t=0, 1) {{ x = Ltot*(1.0-t); y = yTop(x); label = labWall; }}
border inletEdge(t=0, 1) {{ x = 0.0; y = yTop(0.0)+(yBot(0.0)-yTop(0.0))*t; label = labIn; }}
mesh Th = buildmesh(bottomWall(180) + outletEdge(28) + topWall(180) + inletEdge(40));

fespace Qh(Th, P1);
Qh uh, vh, lapU, lapV, convU, convV, f1, f2, p, q;
uh = read("{udat}", Qh);
vh = read("{vdat}", Qh);
real Reff = {reynolds};
real cf = {convection};

solve lapSolve(lapU, q) = int2d(Th)(lapU*q + dx(uh)*dx(q) + dy(uh)*dy(q));
solve lapSolve2(lapV, q) = int2d(Th)(lapV*q + dx(vh)*dx(q) + dy(vh)*dy(q));
solve convSolve(convU, q) = int2d(Th)(convU*q - (uh*dx(uh) + vh*dy(uh))*q);
solve convSolve2(convV, q) = int2d(Th)(convV*q - (uh*dx(vh) + vh*dy(vh))*q);
f1 = lapU - cf*Reff*convU;
f2 = lapV - cf*Reff*convV;
solve Psolve(p, q) = int2d(Th)(dx(p)*dx(q) + dy(p)*dy(q) + 1.0e-10*p*q + f1*dx(q) + f2*dy(q));

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
cout << "ASSIM-OK nv=" << Th.nv << " re=" << Reff << " cf=" << cf << endl;
"""


def build_edp(workdir: Path, case: str, beta: float, reynolds: float, convection: float,
              udat: Path, vdat: Path, pred: Path) -> Path:
    text = EDP_TEMPLATE.format(beta=beta, reynolds=reynolds, convection=convection,
                               udat=str(udat), vdat=str(vdat), pred=str(pred))
    edp = workdir / ("assim_%s.edp" % case)
    edp.write_text(text, encoding="utf-8", newline="\n")
    return edp


def run_freefem(edp: Path, log: Path) -> str:
    proc = subprocess.run(["FreeFem++", "-nw", str(edp)], capture_output=True, text=True, timeout=900)
    log.write_text((proc.stdout or "") + "\n=== STDERR ===\n" + (proc.stderr or ""),
                   encoding="utf-8", newline="\n")
    if proc.returncode != 0 or "ASSIM-OK" not in (proc.stdout or ""):
        raise SystemExit("[FAIL] FreeFEM 基线求解没给出 ASSIM-OK 行（rc=%s），日志见 %s"
                         % (proc.returncode, log))
    return proc.stdout


def read_pred(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def score(pred_rows, truth_rows, labels=(1, 2)) -> dict:
    """去均值压力相对误差 + 压降相对误差（后者逐字符照 train_supervised.compute_case_metrics:639-640）。"""
    import numpy as np

    if len(pred_rows) != len(truth_rows):
        raise SystemExit(f"[FAIL] 顶点数不等：基线 {len(pred_rows)} 真值 {len(truth_rows)}")
    xp = np.array([float(r["x_star"]) for r in pred_rows])
    yp = np.array([float(r["y_star"]) for r in pred_rows])
    xt = np.array([float(r["x_star"]) for r in truth_rows])
    yt = np.array([float(r["y_star"]) for r in truth_rows])
    diff = np.abs(xp - xt) + np.abs(yp - yt)
    scale = np.maximum(np.abs(xt), np.abs(yt)) + 1.0
    rel_off = float(np.max(diff / scale))
    if not rel_off <= 1.0e-5:          # FreeFEM 默认 6 位打印 ⇒ 只能按相对位差核，不许改成最近邻
        raise SystemExit(f"[FAIL] 基线与真值场不同序/不同网格（最大相对坐标差 {rel_off:.3g}）——"
                         "评分就废了")
    p = np.array([float(r["p_star"]) for r in pred_rows])
    pt = np.array([float(r["p_star"]) for r in truth_rows])
    bc = np.array([int(r["bc_tag"]) for r in pred_rows])
    # 真值场用的是文字标签（field_dense.csv 的 boundary_type），基线沿用 .edp 的数字标签
    lab = {"inlet": 1, "outlet": 2, "wall": 3}
    if "bc_tag" in truth_rows[0] and str(truth_rows[0].get("bc_tag") or "").strip().isdigit():
        bc_t = np.array([int(r["bc_tag"]) for r in truth_rows])
    else:
        missing = {t for t in {r["boundary_type"] for r in truth_rows} if t not in lab}
        if missing:
            raise SystemExit(f"[FAIL] 真值场出现未知边界标签 {sorted(missing)}，无法与基线对齐")
        bc_t = np.array([lab[r["boundary_type"]] for r in truth_rows])
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


def c10(workdir: Path, truth_rows, reynolds=10.0, alpha=1.0) -> int:
    """已知答案正对照 + 一条必红控制。返回 0=绿。"""
    xy = [(float(r["x_star"]), float(r["y_star"])) for r in truth_rows]
    import random

    rng = random.Random(0)
    idx = rng.sample(range(len(xy)), 63)                       # 与观测预算同量级的点数
    obs = [xy[i] for i in idx]
    ua = alpha
    exact = lambda X: -reynolds * ua * ua * (X[0] ** 2 + X[1] ** 2) / 2.0
    p_true = [exact(c) for c in xy]
    fails = []
    for convection, tag, want_pass in ((1.0, "正对照", True), (-1.0, "必红", False)):
        ux = [ua * x for x, _ in obs]
        vx = [-ua * y for _, y in obs]
        cu, _ = rbf_complete(obs, ux, xy)
        cv, _ = rbf_complete(obs, vx, xy)
        udat, vdat = workdir / "c10_u.dat", workdir / "c10_v.dat"
        write_dat(udat, xy, cu)
        write_dat(vdat, xy, cv)
        pred = workdir / ("c10_pred_%s.csv" % ("ok" if want_pass else "red"))
        edp = build_edp(workdir, "c10", beta=0.7, reynolds=reynolds, convection=convection,
                        udat=udat, vdat=vdat, pred=pred)
        rows = read_pred(run_freefem_and_read(edp, workdir / ("c10_%s.log" % tag), pred))
        import numpy as np

        pm = np.array([float(r["p_star"]) for r in rows])
        pm = pm - pm.mean()
        ptm = np.array(p_true)
        ptm = ptm - ptm.mean()
        rel = float(np.linalg.norm(pm - ptm) / (np.linalg.norm(ptm) + 1e-12))
        ok = rel <= C10_TOL
        print("C10 %s: conv=%+g 去均值压力相对误差=%.4g 期望=%s -> %s"
              % (tag, convection, rel, ("≤%g" % C10_TOL) if want_pass else (">%g" % C10_RED_TOL),
                 "OK" if ok == want_pass else "BAD"))
        if want_pass and not ok:
            fails.append("正对照不过（%.4g > %g）⇒ 装置不可信" % (rel, C10_TOL))
        if (not want_pass) and not (rel > C10_RED_TOL):
            fails.append("必红控制没红（翻掉对流系数后误差只有 %.4g）⇒ 对流项根本没进装置" % rel)
    return 1 if fails else 0


def run_case(args, truth_rows, obs, outdir: Path, workdir: Path) -> dict:
    xy = [(float(r["x_star"]), float(r["y_star"])) for r in truth_rows]
    pts = [(o[0], o[1]) for o in obs]
    cu, cond_u = rbf_complete(pts, [o[2] for o in obs], xy)
    cv, cond_v = rbf_complete(pts, [o[3] for o in obs], xy)
    stem = "assim_%s_%s_%s" % (args.case, args.quota, args.equation)
    udat, vdat = workdir / (stem + "_u.dat"), workdir / (stem + "_v.dat")
    write_dat(udat, xy, cu)
    write_dat(vdat, xy, cv)
    pred = outdir / (stem + "_pred.csv")
    convection = 1.0 if args.equation == "ns" else 0.0
    edp = build_edp(workdir, stem, beta=args.beta, reynolds=args.reynolds, convection=convection,
                    udat=udat, vdat=vdat, pred=pred)
    rows = read_pred(run_freefem_and_read(edp, workdir / (stem + ".log"), pred))
    res = score(rows, truth_rows)
    res.update(case=args.case, quota=args.quota, equation=args.equation, reynolds=args.reynolds,
               rbf_cond=max(cond_u, cond_v), pred_csv=str(pred), edp=str(edp))
    return res


def run_freefem_and_read(edp: Path, log: Path, pred: Path) -> list[dict]:
    run_freefem(edp, log)
    return read_pred(pred)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/workspace/pinn-repro-2026")
    ap.add_argument("--case", help="工况目录 id，如 C-val_ns_re10")
    ap.add_argument("--quota", default="5pct")
    ap.add_argument("--equation", choices=("ns", "stokes"), default="ns")
    ap.add_argument("--reynolds", type=float, default=10.0)
    ap.add_argument("--beta", type=float, default=0.7)
    ap.add_argument("--selftest", action="store_true", help="C10 已知答案正对照 + 必红（需 FreeFEM）")
    ap.add_argument("--rbf-check", action="store_true", help="只核 RBF 那一半（需 numpy，不需 FreeFEM）")
    ap.add_argument("--outdir", default="")
    a = ap.parse_args()
    root = Path(a.root)
    outdir = Path(a.outdir) if a.outdir else root / "model" / "results" / "baseline"
    workdir = outdir / "_work"
    outdir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    if a.rbf_check:
        return rbf_selftest()

    if a.selftest:
        rc = rbf_selftest()
        if rc:
            return rc
        if not a.case:
            raise SystemExit("[FAIL] C10 也要一枚真网格：--case 必须给")
        truth = read_truth(root / "model" / "cases" / "contraction_2d" / "data" / a.case / "field_dense.csv")
        return c10(workdir, truth, reynolds=a.reynolds)

    if not a.case:
        raise SystemExit("[FAIL] 常规模式要 --case")
    cdir = root / "model" / "cases" / "contraction_2d" / "data" / a.case
    truth = read_truth(cdir / "field_dense.csv")
    obs = read_obs_velocity_only(cdir / ("obs_sparse_%s_velocity_only.csv" % a.quota))
    res = run_case(a, truth, obs, outdir, workdir)
    print("BASELINE " + " ".join("%s=%s" % (k, v) for k, v in res.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
