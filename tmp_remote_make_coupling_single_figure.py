from pathlib import Path
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("model").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting import 配置中文绘图


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    配置中文绘图(prefer_serif=False)
    outdir = PROJECT_ROOT / "results" / "field_map_checks" / "chapter5_materials" / "reworked_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    dual_sparse = load_json(
        PROJECT_ROOT / "results" / "pinn" / "contraction_independent_geometry_sparse5clean_stagepde_v4" / "metrics.json"
    )
    single_sparse = load_json(
        PROJECT_ROOT / "results" / "supervised" / "contraction_single_mlp_geometry_sparse5_20260502" / "metrics.json"
    )
    mainline_history = pd.read_csv(
        PROJECT_ROOT / "results" / "pinn" / "contraction_independent_geometry_notemplate_stagepde_mainline_v4" / "history.csv"
    )
    stage2_last = mainline_history[mainline_history["阶段"] == "第二阶段_压力模型"].iloc[-1]
    stage3_last = mainline_history[mainline_history["阶段"] == "第三阶段_交替耦合"].iloc[-1]

    dual_case = dual_sparse["验证工况指标"][0]
    single_case = single_sparse["val_case_metrics"][0]
    dual_final = dual_sparse["最终验证指标"]
    single_extreme = single_sparse["val_extreme_metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    labels = ["速度Rel-L2", "压力Rel-L2", "最大速度误差", "壁面u残余"]
    single_values = [
        single_case["rel_l2_speed"],
        single_case["rel_l2_p"],
        single_extreme["max_speed_err_over_speed_max"],
        single_case["wall_max_abs_u_pred"],
    ]
    dual_values = [
        dual_final["rel_l2_speed"],
        dual_final["rel_l2_p"],
        dual_final["max_speed_err_over_speed_max"],
        dual_case["wall_max_abs_u_pred"],
    ]
    x = np.arange(len(labels))
    width = 0.34
    b1 = ax.bar(x - width / 2, single_values, width=width, color="#f59e0b", label="单网络MLP（5%稀疏监督）")
    b2 = ax.bar(x + width / 2, dual_values, width=width, color="#0f766e", label="双模型PDE耦合（5%稀疏监督）")
    for bars in (b1, b2):
        for bar in bars:
            value = bar.get_height()
            text = f"{value * 100:.2f}%" if value >= 0.001 else f"{value:.1e}"
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(single_values + dual_values) * 0.025, text, ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylabel("C-val验证指标")
    ax.set_title("稀疏观测下，双模型优势主要体现在速度结构和壁面一致性", fontsize=13, pad=10)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_ylim(0.0, max(single_values + dual_values) * 1.28)

    ax = axes[1]
    labels2 = ["速度Rel-L2", "压力Rel-L2", "最大速度误差", "最大压力误差"]
    uncoupled = [
        float(stage2_last["验证_rel_l2_speed"]),
        float(stage2_last["验证_rel_l2_p"]),
        float(stage2_last["验证_max_speed_err_over_speed_max"]),
        float(stage2_last["验证_max_p_err_over_p_range"]),
    ]
    coupled = [
        float(stage3_last["验证_rel_l2_speed"]),
        float(stage3_last["验证_rel_l2_p"]),
        float(stage3_last["验证_max_speed_err_over_speed_max"]),
        float(stage3_last["验证_max_p_err_over_p_range"]),
    ]
    y = np.arange(len(labels2))
    h = 0.34
    u_bars = ax.barh(y - h / 2, uncoupled, height=h, color="#fb923c", label="耦合前")
    c_bars = ax.barh(y + h / 2, coupled, height=h, color="#2563eb", label="耦合后")
    xmax = max(uncoupled + coupled)
    for bars in (u_bars, c_bars):
        for bar in bars:
            width_value = bar.get_width()
            ax.text(width_value + xmax * 0.025, bar.get_y() + bar.get_height() / 2, f"{width_value * 100:.2f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels2)
    ax.invert_yaxis()
    ax.set_xlabel("验证误差百分比")
    ax.set_title("同一双模型主线内，耦合阶段主要压低压力误差", fontsize=13, pad=10)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_xlim(0.0, xmax * 1.22)

    fig.suptitle("单网络基线与双模型耦合效果对照", fontsize=17, y=1.02)
    fig.text(
        0.5,
        -0.02,
        "注：左图使用5%稀疏观测训练；右图使用稠密主线训练过程中的耦合前后末值。压力Rel-L2并非所有对比中都占优，因此正文只归纳数据能够支持的优势。",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.95))
    out = outdir / "fig5_19_single_baseline_coupling_reworked.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
