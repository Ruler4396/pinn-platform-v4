#!/usr/bin/env python3
"""臂 C（必做1 基线三件套之一）：**POD + 观测点最小二乘**，不训练任何网络。

为什么必须有这一臂：导师问的是"神经网络到底买了什么"。若一个 99% 能量截断的 POD 基
加上 63 个观测点的最小二乘就能达到同样的重建精度，第 5 章的稀疏重建卖点就不成立。

做法（全部确定性，不用随机数）：
1. 用**同一批稠密真值**（train_cases 的 `field_dense.csv`）构造快照矩阵；每个工况的不规则点场
   经 IDW（k 近邻、幂 2）重采样到公共参数网格 ξ∈[0,1] × η∈[−0.5,0.5]，
   其中 ξ=(x*−x_min)/(x_max−x_min)、η=(y*−y_min)/(y_max−y_min)−0.5，**x/y 极差取该工况自己的场**。
2. 对三个场 (u,v,p) 拼成一个矩阵做 SVD，按能量阈值（默认 ≥99%）定模态数 r，打印奇异值谱。
3. 对新工况：把它的观测点（**与格1/格4 同一张 CSV**，默认 `obs_sparse_5pct.csv`）映到同一参数坐标，
   双线性取样模态场得到设计矩阵 A，最小二乘解系数 c；再在双线性意义上重建整个网格场。
4. 评分行集 = 该工况 `field_dense.csv` 的全部行（与双模型评估同一群体），同时另报"只取 interior 行"的子集，
   因为壁面行的真值≈1e-30 会把 pooled 指标的性质改掉（见修订方案 R1）。
   指标照仓库定义（`train_supervised.py:361-362` 的 relative_l2），mean-of-cases 与 pooled 两口径都出。

诚实边界（写进产物 JSON，不许被正文丢掉）：
- 参数化用**各工况自己的 x/y 包围盒**，不是贴中心线的曲线坐标 ⇒ 对收缩族是合理的（几何是 x 向缓变），
  对**弯曲族的 U 形中心线是粗近似**；`--family bend_2d` 时脚本自己打警告。
- POD 是线性流形，不保证逐点无滑移；壁面残差会比带硬包络的双模型大，这是方法差异不是 bug。
- 本臂**不进 progress.jsonl**（它没有 train 阶段，塞一行 phase=train/wall_ms=0 等于骗账本）；
  交付形态是一个独立 JSON + 终端表。

用法：
  python3 model/scripts/baselines_pod.py --family contraction_2d \
      --eval-cases C-val,C-test-1,C-test-2 --out out/pod_baseline_contraction.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CASES_ROOT = PROJECT_ROOT / "cases"
FAMILY_SUBDIR = {"contraction_2d": "contraction_2d", "bend_2d": "bend_2d"}
DEFAULT_TRAIN = {
    "contraction_2d": ["C-base", "C-train-1", "C-train-2", "C-train-3", "C-train-4", "C-train-5"],
    "bend_2d": ["B-base__ip_blunted", "B-train-1__ip_blunted", "B-train-2__ip_blunted", "B-train-3__ip_blunted"],
}


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        enc = getattr(stream, "encoding", "") or ""
        if enc.lower() not in ("utf-8", "utf8"):
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding="utf-8", errors="replace")


def parse_list(text: str) -> list[str]:
    import re
    return [t for t in re.split(r"[,;\s]+", (text or "").strip()) if t]


def load_case(family: str, case_id: str, fname: str):
    import numpy as np
    import csv as _csv
    path = CASES_ROOT / FAMILY_SUBDIR[family] / "data" / case_id / fname
    if not path.exists():
        raise SystemExit(f"[FAIL] 缺文件 {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(_csv.DictReader(fh))
    arr = {k: np.array([float(r[k]) for r in rows], dtype=np.float64)
           for k in rows[0].keys() if k not in ("case_id", "family", "boundary_type",
                                                "is_boundary", "sampling_tag", "noise_tag", "sample_id")}
    arr["boundary_type"] = [r.get("boundary_type", "") for r in rows]
    arr["_path"] = path
    arr["_n"] = len(rows)
    return arr


def to_param(arr, keys=("x_star", "y_star"), bounds=None):
    import numpy as np
    x, y = arr[keys[0]], arr[keys[1]]
    if bounds is None:
        bounds = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    x0, x1, y0, y1 = bounds
    xi = (x - x0) / max(x1 - x0, 1e-12)
    eta = (y - y0) / max(y1 - y0, 1e-12) - 0.5
    return xi, eta, bounds


def idw_to_grid(xi, eta, values, gx, gy, k=8):
    """把不规则点场重采样到 gx×gy 网格（反距离加权，k 近邻，幂 2）。确定性。"""
    import numpy as np
    grid_x, grid_y = np.meshgrid(gx, gy, indexing="ij")        # (nx, ny)
    pts = np.stack([xi, eta], axis=1)
    query = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    out = np.empty(query.shape[0], dtype=np.float64)
    chunk = 4096
    for s in range(0, query.shape[0], chunk):
        q = query[s:s + chunk]
        d2 = ((q[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        dk = np.take_along_axis(d2, idx, axis=1)
        near_zero = dk < 1e-24
        w = 1.0 / np.where(near_zero, 1.0, dk)                   # 幂 2 已由 d2 给出
        vw = values[idx]
        num = (w * vw).sum(axis=1)
        den = w.sum(axis=1)
        hit = np.any(near_zero, axis=1)
        if hit.any():                                            # 命中真点：直接取最近点值
            first = np.argmax(near_zero, axis=1)
            num[hit] = vw[np.arange(hit.sum()), first[hit]]
            den[hit] = 1.0
        out[s:s + chunk] = num / den
    return out.reshape(len(gx), len(gy))


def bilin_sample(field, xi, eta, gx, gy):
    import numpy as np
    fx = (np.asarray(xi) - gx[0]) / max(gx[-1] - gx[0], 1e-12) * (len(gx) - 1)
    fy = (np.asarray(eta) - gy[0]) / max(gy[-1] - gy[0], 1e-12) * (len(gy) - 1)
    fx = np.clip(fx, 0, len(gx) - 1 - 1e-9)
    fy = np.clip(fy, 0, len(gy) - 1 - 1e-9)
    i0 = np.floor(fx).astype(int); j0 = np.floor(fy).astype(int)
    i1 = np.minimum(i0 + 1, len(gx) - 1); j1 = np.minimum(j0 + 1, len(gy) - 1)
    tx = fx - i0; ty = fy - j0
    return ((1 - tx) * (1 - ty) * field[i0, j0] + tx * (1 - ty) * field[i1, j0]
            + (1 - tx) * ty * field[i0, j1] + tx * ty * field[i1, j1])


def relative_l2(pred, truth):
    import numpy as np
    return float(np.linalg.norm(pred - truth) / (np.linalg.norm(truth) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser(description="臂 C：POD + 观测点最小二乘（不训练网络）")
    ap.add_argument("--family", default="contraction_2d", choices=sorted(FAMILY_SUBDIR))
    ap.add_argument("--train-cases", default="", help="留空=该族默认训练工况")
    ap.add_argument("--eval-cases", default="C-val,C-test-1,C-test-2")
    ap.add_argument("--obs-files", default="obs_sparse_5pct.csv",
                    help="逗号分隔；每个文件对应一个'观测档'，与格1/格4 用同一张表")
    ap.add_argument("--nx", type=int, default=120)
    ap.add_argument("--ny", type=int, default=24)
    ap.add_argument("--energy", type=float, default=0.99, help="POD 截断保留的能量比例")
    ap.add_argument("--max-modes", type=int, default=40)
    ap.add_argument("--out", default="", help="产物 JSON 路径（不写则只打印）")
    ap.add_argument("--self-check", action="store_true",
                    help="极小配置跑通全链路并断言不变量（实例上几秒完成，本机无 numpy 时只能到 py_compile 这一层）")
    args = ap.parse_args()
    if args.self_check:
        args.nx, args.ny, args.max_modes = 8, 4, 3
        args.family, args.eval_cases = "contraction_2d", ["C-val"]
        args.obs_files = ["obs_sparse_5pct.csv"]
        args.out = str(Path(tempfile.gettempdir()) / "armC_selfcheck.json")
        print("[self-check] 极小配置 nx=8 ny=4 max_modes=3 eval=C-val，产物写临时目录")

    import numpy as np
    if args.family == "bend_2d":
        print("[warn] 弯曲族的参数网格用 x/y 包围盒，不贴 U 形中心线 ⇒ 该臂在弯曲族是粗近似，"
              "正文引用时必须带这句限定")
    train_cases = parse_list(args.train_cases) or DEFAULT_TRAIN[args.family]
    eval_cases = parse_list(args.eval_cases)
    obs_files = parse_list(args.obs_files)
    gx = np.linspace(0.0, 1.0, args.nx)
    gy = np.linspace(-0.5, 0.5, args.ny)
    t0 = time.perf_counter()

    # 1) 快照矩阵（每工况用自己的 x/y 包围盒；公共网格）
    snaps = []
    for cid in train_cases:
        dense = load_case(args.family, cid, "field_dense.csv")
        xi, eta, _ = to_param(dense)
        blk = []
        for fld in ("u_star", "v_star", "p_star"):
            blk.append(idw_to_grid(xi, eta, dense[fld], gx, gy, k=8))
        snaps.append(np.concatenate([b.ravel() for b in blk]))
    X = np.stack(snaps)                                          # (n_cases, 3*nx*ny)
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    r = int(np.searchsorted(energy, args.energy) + 1)
    r = max(1, min(r, args.max_modes, len(S)))
    basis = Vt[:r]                                               # (r, dof)
    print("[pod] 快照 %d 个（%s） 自由度=%d 模态数 r=%d（能量 %.4f，奇异值前 6=%s）"
          % (X.shape[0], ",".join(train_cases), X.shape[1], r, float(energy[r - 1]),
             np.array2string(S[:6], precision=3)))
    # 把每个模态拆成 (u,v,p) 三张网格场，便于对任意点双线性取样
    n = args.nx * args.ny
    mode_fields = [{"u": basis[i, 0 * n:1 * n].reshape(len(gx), len(gy)),
                    "v": basis[i, 1 * n:2 * n].reshape(len(gx), len(gy)),
                    "p": basis[i, 2 * n:3 * n].reshape(len(gx), len(gy))} for i in range(r)]

    results: dict = {}
    for obs_name in obs_files:
        per_case = []
        pooled_sq = {"u": [0.0, 0.0], "v": [0.0, 0.0], "p": [0.0, 0.0], "speed": [0.0, 0.0]}
        for cid in eval_cases:
            dense = load_case(args.family, cid, "field_dense.csv")
            xi_d, eta_d, _ = to_param(dense)
            obs = load_case(args.family, cid, obs_name)
            xi_o, eta_o, _ = to_param(obs, ("x_star", "y_star"))
            A = np.stack([np.concatenate([bilin_sample(mf["u"], xi_o, eta_o, gx, gy),
                                          bilin_sample(mf["v"], xi_o, eta_o, gx, gy),
                                          bilin_sample(mf["p"], xi_o, eta_o, gx, gy)])
                          for mf in mode_fields], axis=1)          # (3*nobs, r)
            b = np.concatenate([obs["u_obs"], obs["v_obs"], obs["p_obs"]])
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            rec = {"case_id": cid, "n_obs": int(obs["_n"]), "rank": r,
                   "resid_obs": float(np.linalg.norm(A @ coef - b) / (np.linalg.norm(b) + 1e-12))}
            for fld, key, truth_key in (("u", "rel_l2_u", "u_star"), ("v", "rel_l2_v", "v_star"),
                                        ("p", "rel_l2_p", "p_star")):
                grid = sum(coef[i] * mode_fields[i][fld] for i in range(r))
                rec[fld + "_pred"] = bilin_sample(grid, xi_d, eta_d, gx, gy)
                rec[key] = relative_l2(rec[fld + "_pred"], dense[truth_key])
            sp = np.sqrt(rec["u_pred"] ** 2 + rec["v_pred"] ** 2)
            st = np.sqrt(dense["u_star"] ** 2 + dense["v_star"] ** 2)
            rec["rel_l2_speed"] = relative_l2(sp, st)
            interior = np.array([bt == "interior" for bt in dense["boundary_type"]], dtype=bool)
            rec["n_rows"] = int(dense["_n"]); rec["n_interior"] = int(interior.sum())
            for fld, truth_key in (("u", "u_star"), ("p", "p_star")):
                rec["rel_l2_%s_interior" % fld] = relative_l2(rec[fld + "_pred"][interior],
                                                              dense[truth_key][interior])
            rec["rel_l2_speed_interior"] = relative_l2(sp[interior], st[interior])
            for fld, pred, truth in (("u", rec["u_pred"], dense["u_star"]),
                                     ("v", rec["v_pred"], dense["v_star"]),
                                     ("p", rec["p_pred"], dense["p_star"]),
                                     ("speed", sp, st)):
                pooled_sq[fld][0] += float(np.sum((pred - truth) ** 2))
                pooled_sq[fld][1] += float(np.sum(truth ** 2))
            per_case.append(rec)
        mac = {k: float(np.mean([c[k] for c in per_case])) for k in per_case[0]
               if isinstance(per_case[0][k], float)}
        pooled = {("rel_l2_" + fld): math.sqrt(sq[0]) / (math.sqrt(sq[1]) + 1e-12)
                  for fld, sq in pooled_sq.items()}
        results[obs_name] = {"cases": per_case, "mean_of_cases": mac,
                             "pooled_note": "pooled 口径 = 把所有工况的行拼起来算一次 relative_l2",
                             "pooled": pooled}
        print("[armC] %s → mean_of_cases speed=%.4f p=%.4f ; pooled speed=%.4f p=%.4f"
              % (obs_name, mac.get("rel_l2_speed", float("nan")), mac.get("rel_l2_p", float("nan")),
                 pooled["rel_l2_speed"], pooled["rel_l2_p"]))
        for c in per_case:
            print("        %-14s n_obs=%-4d speed=%.4f (interior %.4f) p=%.4f (interior %.4f) u=%.4f 观测残差=%.4f"
                  % (c["case_id"], c["n_obs"], c["rel_l2_speed"], c["rel_l2_speed_interior"],
                     c["rel_l2_p"], c["rel_l2_p_interior"], c["rel_l2_u"], c["resid_obs"]))

    payload = {"臂": "C_POD_最小二乘", "family": args.family, "train_cases": train_cases,
               "energy_threshold": args.energy, "rank": r,
               "singular_values": [float(s) for s in S[:min(len(S), 12)]],
               "cumulative_energy": [float(e) for e in energy[:min(len(energy), 12)]],
               "grid": {"nx": args.nx, "ny": args.ny,
                        "mapping": "xi=(x-xmin)/(xmax-xmin), eta=(y-ymin)/(ymax-ymin)-0.5，逐工况自包围盒"},
               "wall_ms": int((time.perf_counter() - t0) * 1000),
               "limits": ["线性流形，无逐点无滑移保证", "弯曲族的参数网格是包围盒近似，非中心线坐标",
                          "评分群体与双模型一致（全部 dense 行），另附 interior 子集",
                          "本臂不写 progress.jsonl（无训练阶段）"],
               "results": results}
    for key in list(results):
        for c in results[key]["cases"]:
            for fld in ("u", "v", "p"):
                c.pop(fld + "_pred", None)
    print("[armC] 总耗时 %d ms（无训练；单次运行，可重复 ⇒ 与 PINN 的墙钟不可直接比）" % payload["wall_ms"])
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        json.loads(out.read_text(encoding="utf-8"))
        print("[out] %s 已写并读回校验（rank=%d, 档=%s）" % (out, r, ",".join(obs_files)))
    if args.self_check:
        fails = []
        if r < 1 or r > len(S):
            fails.append("模态数 r=%d 不在 [1,%d]" % (r, len(S)))
        for obs_name, blk in results.items():
            mac = blk["mean_of_cases"]
            for key in ("rel_l2_u", "rel_l2_speed", "rel_l2_p", "rel_l2_speed_interior"):
                val = mac.get(key)
                if val is None or not math.isfinite(val) or not (0.0 <= val < 5.0):
                    fails.append("%s %s=%r 不在可解释范围" % (obs_name, key, val))
            if blk["pooled"]["rel_l2_speed"] > 5.0:
                fails.append("%s pooled 速度=%.4f 明显不合理" % (obs_name, blk["pooled"]["rel_l2_speed"]))
            for c in blk["cases"]:
                if not (0.0 <= c["resid_obs"] < 1.0):
                    fails.append("%s 观测最小二乘残差=%.4f ⇒ 拟合链路有问题" % (c["case_id"], c["resid_obs"]))
                if c["n_obs"] < r:
                    fails.append("%s 观测点 %d 少于模态数 %d ⇒ 系数不可定" % (c["case_id"], c["n_obs"], r))
        print("[self-check] %s" % ("PASS：rank=%d，两口径读数均在可解释范围，观测残差与点数满足前提" % r
                                    if not fails else "FAIL：" + "；".join(fails)))
        if fails:
            return 1
    return 0


if __name__ == "__main__":
    _utf8()
    raise SystemExit(main())
