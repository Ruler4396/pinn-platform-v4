"""速度-only 观测下的压力恢复支路（Stage 5，判据见 paper-route2/速度推压力-经典基线同框-预注册设计-20261003.md）。

挂在上一轮的 NS 支路（train_velocity_pressure_independent_ns.py）之上，只补一件事，
另一件经现读确认主线本来就做对了（见 §二.2 注释），因此这里只做断言、不重写。

1. **要补的**：严格稀疏版的 `输出标准化器.fit` 在某一列**一个有限值都没有**时抛
   "严格稀疏训练无法从空观测列拟合输出标准化器"。速度-only 观测表里压力列正是全空 ⇒ 训练第一步就死。
   回退规则写死在此：压力尺度 = ‖同一 run 刚拟合好的速度标准化器 std 向量‖（star 单位 ρ=1，
   与动压同量级，是**单位约定**不是真值），均值取 0。
   顺序假设（速度先拟合、压力后拟合）**不靠环境成立**：没先见到速度拟合就直接 SystemExit。

2. **不补、只断言的**：`压力监督损失` 在无有限目标时已经返回 0（..._strict_sparse.py:253-258），
   本件用一条自检把它钉住（全 NaN 目标 ⇒ 必须恰好 0、不许 NaN），因为一旦哪天有人改了那半句，
   这一臂会静默变成"压力被 0 监督"。

3. **哨兵（C7）**：① 压力列全空的桩 ⇒ 必须走回退并打印来源；② 压力列有值的桩 ⇒ 必须**原样按主线**拟合、
   且回退分支一次都没进（进了就是我把泄漏当成功能）。
"""
import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NS_SCRIPT = "train_velocity_pressure_independent_ns.py"

_FIT_LOG = {"vel_std": None, "fallback_count": 0}
P_SCALE_MODE = "auto"      # auto＝‖观测速度std‖（预注册 §六.1 的约定）；one＝不归一（敏感性对照）


def load_dep(name: str):
    path = SCRIPT_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(path.stem, module)
    spec.loader.exec_module(module)
    return module


def _norm(v):
    return float(sum(float(x) ** 2 for x in v) ** 0.5)


def install_scaler_patch(bm) -> None:
    """把 输出标准化器.fit 换成带回退的版本；补丁没落上就红，不静默。"""
    cls = bm.输出标准化器
    if getattr(cls.fit.__func__, "_vonly", False) is True:      # 已经落过，别再套一层
        print("[NS-vonly] 补丁已在位（未重复安装）", flush=True)
        return
    orig = cls.fit.__func__

    def fit_vonly(cls_, values):
        import numpy as np

        arr = np.asarray(values, dtype=np.float64)
        finite_cols = [int(np.isfinite(arr[:, i]).sum()) for i in range(arr.shape[1])]
        if arr.shape[1] == 2:                                  # 速度列（u,v）：记下来当压力尺度的来源
            s = orig(cls_, arr)
            _FIT_LOG["vel_std"] = list(s.std)
            return s
        if all(c == 0 for c in finite_cols):                   # 一个有限值都没有 ⇒ 走回退
            if _FIT_LOG["vel_std"] is None:
                raise SystemExit(
                    "[FAIL] 压力列全空但先前没拟合过速度标准化器——回退尺度无来源。"
                    "顺序假设（速度先、压力后）已不成立，别改这里，去把尺度显式传进来。")
            p_scale = _norm(_FIT_LOG["vel_std"])
            if P_SCALE_MODE == "one":
                p_scale = 1.0
                print("[C7-fallback] 敏感性对照档：压力尺度=1.0（不归一，网络直接输出 p 的量值）；"
                      "这一档**不是**登记臂，只用来说明回退尺度本身的影响", flush=True)
            if not p_scale > 0:
                raise SystemExit(f"[FAIL] 回退压力尺度算出来是 {p_scale}，不是正数")
            _FIT_LOG["fallback_count"] += 1
            print("[C7-fallback] 压力观测列有限值个数=0 ⇒ 压力尺度回退取 ‖观测速度std‖=%.6g"
                  "（star 单位 ρ=1，与动压同量级；单位约定，不是真值）；mean=0；pressure_obs_used=0"
                  % p_scale, flush=True)
            mean = np.zeros((arr.shape[1],), dtype=np.float32)
            std = np.full((arr.shape[1],), p_scale, dtype=np.float32)
            return cls_(mean=mean, std=std)
        return orig(cls_, arr)                                 # 有值 ⇒ 原样交给主线（不许动）

    fit_vonly._vonly = True
    cls.fit = classmethod(fit_vonly)
    if getattr(cls.fit.__func__, "_vonly", False) is not True:
        raise SystemExit("[FAIL] 压力尺度回退补丁没落到 输出标准化器 上，训练会在第一步抛错")
    print("[NS-vonly] 补丁已落：输出标准化器.fit（空列回退）；压力监督损失未改（只断言）", flush=True)


def vonly_selftest() -> int:
    """C7 两桩 + 第 2 条断言。需要 numpy/torch ⇒ 只在实例上跑（本机 py_compile 到此为止）。"""
    fails = []
    import numpy as np
    import torch

    nsmod = load_dep(NS_SCRIPT)
    nsmod.BASE_SCRIPT = "train_velocity_pressure_independent_strict_sparse.py"
    nsmod.SCALE_VARIANT = "scaler"
    bm = nsmod.the_base()
    install_scaler_patch(bm)
    标准化器 = bm.输出标准化器

    # 桩①：速度列有值、压力列全空 —— 必须走回退
    vel = np.array([[0.8, 0.1], [1.0, 0.0], [1.2, -0.1]])
    s_v = 标准化器.fit(vel)
    empty_p = np.full((3, 1), np.nan)
    s_p = 标准化器.fit(empty_p)
    want = _norm(s_v.std)
    if _FIT_LOG["fallback_count"] != 1:
        fails.append("桩①没进回退分支（fallback_count=%d）" % _FIT_LOG["fallback_count"])
    if abs(float(s_p.std[0]) - want) > 1e-6:
        fails.append("回退尺度不等于 ‖速度std‖：得 %s 期望 %s" % (s_p.std[0], want))
    if abs(float(s_p.mean[0])) > 0:
        fails.append("回退均值应为 0，得 %s" % (s_p.mean[0],))

    # 桩②：压力列有值 —— 必须**不**走回退，且尺度就是列自身 std（泄漏当功能的那条路要红）
    before = _FIT_LOG["fallback_count"]
    real_p = np.array([[2.0], [4.0], [6.0]])
    s_q = 标准化器.fit(real_p)
    if _FIT_LOG["fallback_count"] != before:
        fails.append("桩②（压力列有值）竟然进了回退分支")
    if abs(float(s_q.std[0]) - float(real_p[:, 0].std())) > 1e-6:
        fails.append("桩②没按主线拟合：得 %s 期望 %s" % (s_q.std[0], real_p[:, 0].std()))

    # 桩③：顺序假设消失就红 —— 全新日志、先拟合压力列，必须 SystemExit
    saved = dict(_FIT_LOG)
    _FIT_LOG.update({"vel_std": None, "fallback_count": 0})
    try:
        标准化器.fit(empty_p)
    except SystemExit:
        pass
    else:
        fails.append("没有速度尺度时回退竟静默成功，哨兵不咬")
    _FIT_LOG.update(saved)

    # 第 2 条：压力监督损失在全 NaN 目标上必须恰好 0（不是 NaN、不是抛错）
    pred = torch.tensor([[0.3], [-1.2], [0.9]], dtype=torch.float32)
    truth = torch.full((3, 1), float("nan"), dtype=torch.float32)
    loss = bm.压力监督损失(pred, truth, 标准化器.fit(np.array([[1.0], [3.0]])))
    if torch.isnan(loss).any().item():
        fails.append("压力监督损失给出 NaN")
    elif abs(float(loss)) > 0:
        fails.append("压力监督损失在无目标时不是 0，得 %s" % float(loss))

    print("VONLY_SELFTEST " + ("ALL GREEN" if not fails else "FAILED | " + " | ".join(fails)), flush=True)
    return 1 if fails else 0


def main() -> int:
    global P_SCALE_MODE
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--vonly-selftest", action="store_true",
                    help="C7 三桩 + 压力监督空目标断言（需 numpy/torch ⇒ 实例上跑）")
    ap.add_argument("--vonly-pressure-scale", choices=("auto", "one"), default="auto",
                    help="回退尺度的取法：auto＝‖观测速度std‖（登记臂）；"
                         "one＝不归一，只作敏感性对照，用完即弃、不进判决")
    ap.add_argument("--base-script", choices=("mainline", "strict-sparse"), default="strict-sparse")
    known, rest = ap.parse_known_args()
    if known.vonly_selftest:
        return vonly_selftest()
    P_SCALE_MODE = known.vonly_pressure_scale
    print("[NS-vonly] 压力回退尺度档=%s" % P_SCALE_MODE, flush=True)

    nsmod = load_dep(NS_SCRIPT)
    if known.base_script == "strict-sparse":
        nsmod.BASE_SCRIPT = "train_velocity_pressure_independent_strict_sparse.py"
        nsmod.SCALE_VARIANT = "scaler"
    else:
        nsmod.BASE_SCRIPT = "train_velocity_pressure_independent.py"
        nsmod.SCALE_VARIANT = "dense"
    bm = nsmod.the_base()
    install_scaler_patch(bm)
    print("[NS-vonly] base=%s 观测口径=速度-only（压力列全空）" % nsmod.BASE_SCRIPT, flush=True)
    # --base-script 要原样递下去：NS 支路的 main 自己会 parse_known_args，
    # 我这边只改它的全局变量的话会被它按 default=mainline 覆回去（= 挂错主线、尺度口径也错）。
    sys.argv = [sys.argv[0], "--base-script", known.base_script] + rest
    return nsmod.main()


if __name__ == "__main__":
    raise SystemExit(main())
