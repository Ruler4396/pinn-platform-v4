#!/usr/bin/env python3
"""Round-trip (synthetic-truth) test of the distributed-inversion machinery -- NOT a result.

CAN prove: the parameterization, the observation design, the differentiable forward model and the
shape metric form a machine that turns -- the rank condition agrees with the DOF choice, a gradient
optimizer reduces the loss on a known distribution, and the shape error is computable and comparable
across arms.

CANNOT prove (and the artefact repeats this so no one can quote it out of context): the truth comes
from the SAME family of 1-D equations the inversion solves. That is exactly the self-comparison shape
this project bans as evidence, so nothing here may be read as "inversion works on the real
T-bifurcation", "PINN beats the baseline", or any accuracy/resolution claim. The real-physics version
takes a FreeFEM snapshot as truth -- that is the GPU job.

Gates fixed before the run (any FAIL => write "not feasible under this observation set", no GPU):
  G1 rank     : rank(d observations / d h_dof) at h_true == n_dof   (failure = observation design)
  G2 training : loss falls >= 3 orders of magnitude, gradients not all zero (failure = machine broken)
  G3 shape    : shape error of the trained arm <= half the SAME-DOF baseline arm, on the SAME table

Venue: stdlib only, except the trainable arm which needs torch (measured on the instance:
/usr/local/bin/python3 -> 2.3.1+cpu; the dolfinx prefix has NO torch, so it is not used here).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import impedance_baseline as ib                        # repo tooling: rank_by_elimination

MU, P_IN = 1.0, 12.0
LENGTHS = {"stem": 1.0, "up": 0.8, "down": 0.6}
W_REF = {"stem": 1.0, "up": 0.7, "down": 0.5}
KNOTS = {"stem": (0.34, 0.67, 1.0), "up": (0.5, 1.0), "down": (1.0,)}        # 3+2+1 = 6 DOF
H_TRUE = {"stem": (1.00, 0.95, 0.75), "up": (1.05, 0.95), "down": (1.10,)}  # asymmetric, one throat
PROBES = (0.25, 0.50, 0.75)                                                  # >=3 interior points/branch
NOISE_FRAC = 0.03                                                            # pre-registered floor
NQ = 400                       # midpoint quadrature for the stdlib arms
NQ_TORCH = 200                 # the trainable arm's grid, registered rather than silently cheaper
MAX_S_DEFAULT = 600.0

# one order, used by BOTH arms and by the Jacobian -- a permutation would silently compare
# pressure against flow, so the order is a constant, not something each arm invents.
OBS_ORDER = (["q_in", "q_up", "q_down", "p_junction"]
             + [f"p_{b}_{f:.2f}" for b in ("stem", "up", "down") for f in PROBES])
PARAMS = [(b, i) for b in ("stem", "up", "down") for i in range(len(KNOTS[b]))]   # 6 flattened DOF


class Stop(Exception):
    def __init__(self, phase, used, budget):
        Exception.__init__(self, phase)
        self.info = {"phase": phase, "used_s": round(used, 1), "budget_s": budget}


class Clock:
    def __init__(self, budget):
        self.t0 = time.perf_counter()
        self.budget = budget

    def enter(self, phase):
        used = time.perf_counter() - self.t0
        if used > self.budget:
            raise Stop(phase, used, self.budget)

    def used(self):
        return round(time.perf_counter() - self.t0, 2)


def h_at(branch, frac, dof):
    """Piecewise-linear h over the fixed knots; frac=0 is pinned to 1.0 (the reference width)."""
    xs = (0.0,) + tuple(KNOTS[branch])
    hs = (1.0,) + tuple(dof[branch])
    for i in range(len(xs) - 1):
        if frac <= xs[i + 1] or i == len(xs) - 2:
            span = xs[i + 1] - xs[i]
            w = 0.0 if span <= 0 else (frac - xs[i]) / span
            return hs[i] + w * (hs[i + 1] - hs[i])
    return hs[-1]


def resistance(branch, dof, upto=1.0, nq=NQ):
    """R(0..upto) = integral 12 mu / w(x)^3 dx, w = W_REF*h, midpoint quadrature."""
    step = LENGTHS[branch] * upto / nq
    acc = 0.0
    for i in range(nq):
        frac = (i + 0.5) / nq * upto
        w = max(W_REF[branch] * h_at(branch, frac, dof), 1.0e-9)
        acc += 12.0 * MU * step / (w ** 3)
    return acc


def forward(dof):
    """Series stem with parallel branches; pressures are partial integrals, not fraction*R."""
    Rs, Ru, Rd = (resistance(b, dof) for b in ("stem", "up", "down"))
    q_in = P_IN / (Rs + (Ru * Rd) / (Ru + Rd))
    p_j = P_IN - q_in * Rs
    q_u, q_d = p_j / Ru, p_j / Rd
    obs = {"q_in": q_in, "q_up": q_u, "q_down": q_d, "p_junction": p_j}
    for branch, q, head in (("stem", q_in, P_IN), ("up", q_u, p_j), ("down", q_d, p_j)):
        for f in PROBES:
            obs[f"p_{branch}_{f:.2f}"] = head - q * resistance(branch, dof, upto=f)
    return obs


def vec(obs_dict):
    return [obs_dict[k] for k in OBS_ORDER]


def dof_from_vec(v):
    dof, i = {}, 0
    for b in ("stem", "up", "down"):
        n = len(KNOTS[b])
        dof[b] = tuple(v[i:i + n])
        i += n
    return dof


def dof_to_vec(dof):
    return [x for b in ("stem", "up", "down") for x in dof[b]]


def make_noise(obs, seed, frac):
    """Deterministic multiplicative noise: own LCG, so the same obs_seed gives the same table anywhere."""
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    out = {}
    for k in OBS_ORDER:
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u = ((state >> 11) / float(1 << 53)) * 2.0 - 1.0
        out[k] = obs[k] * (1.0 + frac * u)
    return out


def jacobian(dof, rel_step=1.0e-4):
    """Central differences over all 6 flattened DOF -> rows = observables, cols = DOF."""
    v0 = dof_to_vec(dof)
    cols = []
    for j in range(len(v0)):
        vp, vm = list(v0), list(v0)
        vp[j] = v0[j] * (1.0 + rel_step)
        vm[j] = v0[j] * (1.0 - rel_step)
        fp = vec(forward(dof_from_vec(vp)))
        fm = vec(forward(dof_from_vec(vm)))
        d = 2.0 * rel_step * abs(v0[j])
        cols.append([(a - b) / d for a, b in zip(fp, fm)])
    return [list(row) for row in zip(*cols)]


def solve(mat, rhs):
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(mat)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[piv][i]) < 1.0e-16:
            continue
        a[i], a[piv] = a[piv], a[i]
        for r in range(n):
            if r == i:
                continue
            f = a[r][i] / a[i][i]
            for c in range(i, n + 1):
                a[r][c] -= f * a[i][c]
    return [a[i][n] / a[i][i] if abs(a[i][i]) > 1.0e-16 else 0.0 for i in range(n)]


def resid(v, target):
    """Relative residuals: pressures and flows differ by two orders of magnitude, and an unweighted
    least squares would quietly fit only the pressures."""
    return [(pr - t) / max(abs(t), 1.0e-9) for pr, t in zip(vec(forward(dof_from_vec(v))), target)]


def sse(v, target):
    return sum(r * r for r in resid(v, target))


def gauss_newton(target, v0, iters=200, lam=1.0e-3):
    """Damped Gauss-Newton over the SAME 6 DOF -- the zero-training, same-budget baseline arm."""
    v, hist = list(v0), []
    for _ in range(iters):
        cur = sse(v, target)
        hist.append(cur)
        J = jacobian(dof_from_vec(v))
        n = len(v)
        res = resid(v, target)
        Js = [[J[k][i] / max(abs(target[k]), 1.0e-9) for i in range(n)] for k in range(len(J))]
        A = [[sum(Js[k][i] * Js[k][j] for k in range(len(Js))) for j in range(n)] for i in range(n)]
        g = [sum(Js[k][i] * res[k] for k in range(len(Js))) for i in range(n)]
        moved = False
        for _try in range(14):
            Areg = [[A[i][j] + (lam if i == j else 0.0) for j in range(n)] for i in range(n)]
            st = solve(Areg, [-x for x in g])
            cand = [v[i] + st[i] for i in range(n)]
            if sse(cand, target) < cur:
                v, lam, moved = cand, max(lam * 0.5, 1.0e-9), True
                break
            lam *= 10.0
        if not moved:
            break
        if len(hist) > 2 and abs(hist[-2] - hist[-1]) < 1.0e-12 * max(hist[-1], 1.0e-30):
            break
    return v, hist


def shape_error(dof_hat, dof_true):
    """Relative L2 of h over a common grid, plus the throat (min h) location and depth."""
    num = den = 0.0
    for b in ("stem", "up", "down"):
        grid = [(i + 0.5) / 100.0 for i in range(100)]
        for f in grid:
            a, t = h_at(b, f, dof_hat), h_at(b, f, dof_true)
            num += (a - t) ** 2
            den += t * t
    l2 = math.sqrt(num / max(den, 1.0e-30))
    cand_h = min(((b, f, h_at(b, f, dof_hat)) for b in ("stem", "up", "down")
                  for f in [i / 100.0 for i in range(1, 101)]), key=lambda t: t[2])
    cand_t = min(((b, f, h_at(b, f, dof_true)) for b in ("stem", "up", "down")
                  for f in [i / 100.0 for i in range(1, 101)]), key=lambda t: t[2])
    return {"rel_l2_h": l2, "throat_hat": {"branch": cand_h[0], "frac": cand_h[1], "h": round(cand_h[2], 6)},
            "throat_true": {"branch": cand_t[0], "frac": cand_t[1], "h": round(cand_t[2], 6)},
            "throat_branch_match": cand_h[0] == cand_t[0],
            "throat_offset_frac_along_branch": round(abs(cand_h[1] - cand_t[1]), 4)}


def train_mlp(target, log):
    """The trainable arm: one small MLP per branch mapping x -> h(x). Absent torch => reported, not faked."""
    try:
        import torch
    except Exception as exc:
        return {"ran": False, "why": f"torch unavailable ({type(exc).__name__})"}
    torch.manual_seed(0)
    tgt = torch.tensor(target, dtype=torch.float64)
    nets = {b: torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Tanh(),
                                   torch.nn.Linear(8, 8), torch.nn.Tanh(),
                                   torch.nn.Linear(8, 1)).double() for b in ("stem", "up", "down")}
    ps = [p for n in nets.values() for p in n.parameters()]

    def hnet(branch, fracs):
        x = torch.tensor([[f] for f in fracs], dtype=torch.float64)
        return (1.0 + 0.6 * torch.tanh(nets[branch](x))).squeeze(-1)

    def resist(branch, upto):
        step = LENGTHS[branch] * upto / NQ_TORCH
        fr = [(i + 0.5) / NQ_TORCH * upto for i in range(NQ_TORCH)]
        ws = torch.clamp(hnet(branch, fr) * W_REF[branch], min=1.0e-6)
        return (12.0 * MU * step / ws ** 3).sum()

    def observe():
        Rs, Ru, Rd = resist("stem", 1.0), resist("up", 1.0), resist("down", 1.0)
        q_in = P_IN / (Rs + Ru * Rd / (Ru + Rd))
        p_j = P_IN - q_in * Rs
        q_u, q_d = p_j / Ru, p_j / Rd
        vals = {"q_in": q_in, "q_up": q_u, "q_down": q_d, "p_junction": p_j}
        for branch, q, head in (("stem", q_in, P_IN), ("up", q_u, p_j), ("down", q_d, p_j)):
            for f in PROBES:
                vals[f"p_{branch}_{f:.2f}"] = head - q * resist(branch, f)
        return torch.stack([vals[k] for k in OBS_ORDER])

    opt = torch.optim.Adam(ps, lr=2.0e-2)
    hist, gnorm = [], []
    for it in range(400):
        opt.zero_grad()
        loss = ((observe() - tgt) ** 2).sum()
        loss.backward()
        gnorm.append(float(sum((p.grad.norm() ** 2).item() for p in ps) ** 0.5))
        opt.step()
        hist.append(float(loss.item()))
        if it in (99, 199, 299, 399):
            log.append(f"  mlp step {it + 1}: loss={hist[-1]:.6e} grad_norm={gnorm[-1]:.3e}")
    dof_hat = {}
    for b in ("stem", "up", "down"):
        with __import__("torch").no_grad():
            dof_hat[b] = tuple(float(hnet(b, [f])[i]) for i, f in enumerate(KNOTS[b]))
    return {"ran": True, "torch_version": torch.__version__, "steps": len(hist),
            "loss_first": hist[0], "loss_last": hist[-1], "grad_norm_first": gnorm[0],
            "grad_norm_last": gnorm[-1], "trainable_parameters": len(ps),
            "hidden": "1x8x8x1 per branch (3 nets)", "quad_grid": NQ_TORCH, "dof_hat": dof_hat}


def main() -> int:
    ap = argparse.ArgumentParser(description="round-trip machinery test for the distributed inversion")
    ap.add_argument("--obs-seed", type=int, default=20260928)
    ap.add_argument("--out-root", required=True, help="artefact root, OUTSIDE the repository")
    ap.add_argument("--max-seconds", type=float, default=MAX_S_DEFAULT)
    ap.add_argument("--stdlib-only", action="store_true", help="skip the trainable arm (local gates check)")
    args = ap.parse_args()
    ck = Clock(args.max_seconds)
    log, rc = [], 0
    v = {"kind": "roundtrip_machinery_test", "obs_seed": args.obs_seed,
         "reads": "SYNTHETIC truth from the same 1-D family the inversion solves -- the machine turns; "
                  "nothing here shows inversion works on the real geometry, that a trained arm beats a "
                  "baseline, or any accuracy claim",
         "n_dof": len(PARAMS), "dof_names": [f"{b}{i+1}" for b, i in PARAMS],
         "knots": {b: list(KNOTS[b]) for b in KNOTS}, "h_true": {b: list(H_TRUE[b]) for b in KNOTS},
         "observables": list(OBS_ORDER), "probes_per_branch": len(PROBES), "noise_frac": NOISE_FRAC,
         "quad_grid_stdlib": NQ, "budget_s": args.max_seconds}
    try:
        ck.enter("G0_forward")
        obs = make_noise(forward(H_TRUE), args.obs_seed, NOISE_FRAC)
        target = vec(obs)
        v["observations"] = {k: obs[k] for k in OBS_ORDER}
        log.append(f"[G0] forward solved: {len(OBS_ORDER)} observables (4 node + "
                   f"{len(PROBES)}x3 interior stations), noise {NOISE_FRAC:.0%} at seed {args.obs_seed}")

        ck.enter("G1_rank")
        J = jacobian(H_TRUE)
        r = ib.rank_by_elimination(J, tol=1.0e-9)
        v["G1_rank"] = {"rows": len(J), "cols": len(J[0]), "rank": r, "n_dof": len(PARAMS),
                        "pass": r == len(PARAMS)}
        log.append(f"[G1] rank={r} of {len(J)}x{len(J[0])}, n_dof={len(PARAMS)} -> "
                   + ("PASS" if v["G1_rank"]["pass"] else "FAIL: observation design, not the optimizer"))

        ck.enter("G2a_same_dof_baseline")
        flat = [1.0] * len(PARAMS)
        v_base, hist = gauss_newton(target, flat)
        e_base = shape_error(dof_from_vec(v_base), H_TRUE)
        v["arm_baseline_same_dof"] = {"dof_hat": {b: list(x) for b, x in dof_from_vec(v_base).items()},
                                      "sse_first": hist[0], "sse_last": hist[-1], "iters": len(hist),
                                      "shape": e_base}
        log.append(f"[G2a] same-DOF Gauss-Newton baseline: sse {hist[0]:.4e} -> {hist[-1]:.4e} "
                   f"({len(hist)} its), shape rel_l2(h)={e_base['rel_l2_h']:.4f}")

        if args.stdlib_only:
            v["G2_training"] = {"pass": None, "why": "--stdlib-only, trainable arm not run here"}
            v["G3_shape"] = {"pass": None, "why": "same"}
            log.append("[G2b] trainable arm skipped on purpose (--stdlib-only)")
        else:
            ck.enter("G2b_trainable")
            t = train_mlp(target, log)
            if not t["ran"]:
                v["arm_trainable"] = t
                v["G2_training"] = {"pass": False, "why": t["why"]}
                v["G3_shape"] = {"pass": False, "why": "trainable arm never ran"}
                log.append(f"[G2b] NOT RUN: {t['why']} -- no substitute reading is written")
                rc = 2
            else:
                e_t = shape_error(t["dof_hat"], H_TRUE)
                drop = t["loss_first"] / max(t["loss_last"], 1.0e-300)
                v["arm_trainable"] = {**t, "shape": e_t, "loss_drop_ratio": drop}
                v["G2_training"] = {"loss_first": t["loss_first"], "loss_last": t["loss_last"],
                                    "drop_orders": math.log10(max(drop, 1.0e-300)),
                                    "grad_norm_first": t["grad_norm_first"],
                                    "grad_norm_last": t["grad_norm_last"],
                                    "pass": drop >= 1.0e3 and t["grad_norm_last"] > 0.0}
                log.append(f"[G2] loss {t['loss_first']:.6e} -> {t['loss_last']:.6e} "
                           f"(10^{math.log10(max(drop, 1e-300)):.2f} orders), "
                           f"grad_last={t['grad_norm_last']:.3e} -> "
                           + ("PASS" if v["G2_training"]["pass"] else "FAIL"))

                ck.enter("G3_shape")
                lim = 0.5 * e_base["rel_l2_h"]
                v["G3_shape"] = {"e_trainable": e_t["rel_l2_h"], "e_baseline_same_dof": e_base["rel_l2_h"],
                                 "must_be_le": lim, "pass": e_t["rel_l2_h"] <= lim,
                                 "dof_budget_note": f"trained arm has {t['trainable_parameters']} "
                                                    "trainable parameters against the baseline's "
                                                    f"{len(PARAMS)} DOF -- the extra freedom is registered "
                                                    "here, so a win is not free",
                                 "throat": {"trainable": e_t["throat_hat"], "baseline": e_base["throat_hat"],
                                            "true": e_base["throat_true"]}}
                log.append(f"[G3] shape rel_l2 trainable={e_t['rel_l2_h']:.4f} vs "
                           f"baseline={e_base['rel_l2_h']:.4f} (gate <= {lim:.4f}) -> "
                           + ("PASS" if v["G3_shape"]["pass"] else "FAIL"))
                log.append(f"[G3] throat: trainable={e_t['throat_hat']['branch']}@{e_t['throat_hat']['frac']:.2f} "
                           f"h={e_t['throat_hat']['h']} baseline={e_base['throat_hat']['branch']}@"
                           f"{e_base['throat_hat']['frac']:.2f} true={e_base['throat_true']['branch']}@"
                           f"{e_base['throat_true']['frac']:.2f}")
    except Stop as s:
        v["self_stopped"] = s.info
        log.append(f"[STOP] budget {s.info['budget_s']:.0f}s exceeded in phase {s.info['phase']} "
                   f"at {s.info['used_s']}s -- named, not hidden")
        rc = 3
    v["wall_s"] = ck.used()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out = out_root / f"roundtrip_h_seed{args.obs_seed}.json"
    out.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(log))
    print(f"json={out}")
    print(f"GATES rank={v.get('G1_rank', {}).get('pass')} training={v.get('G2_training', {}).get('pass')} "
          f"shape={v.get('G3_shape', {}).get('pass')} rc={rc} wall={v['wall_s']}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
