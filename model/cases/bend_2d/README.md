# bend_2d

平滑弯曲流道 case 目录。

## 子目录
- `geometry/`：几何定义、采样边界、网格描述
- `cfd/`：CFD 配置、求解脚本、原始输出
- `data/`：训练/验证数据集
- `figures/`：流场图、误差图、剖面对比图

## 当前建议
优先从 `90°`、`Rc = 6W` 的大曲率半径弯道开始。

## 当前已落地
- `scripts/generate_bend_case.py` 可生成 `B-base` 等 case 的 geometry / dense field / sparse obs / meta 骨架。
- 当前脚本已支持 `synthetic_streamfunction_smoke` 与 `FreeFEM++ CFD` 两种 dense field 来源。
- `scripts/generate_bend_trainval_batch.py` 已可串行批量生成 `B-base + B-train-* + B-val` 的 FreeFEM++ CFD 数据与 manifest。
- 2026-04-01 新增多入口剖面 CFD 能力：`scripts/generate_bend_case.py` / `scripts/generate_bend_trainval_batch.py` 已支持 `--inlet-profile parabolic|blunted|skewed_top|skewed_bottom`，并可生成如 `B-train-2__ip_blunted` 这样的变体 case。
- `scripts/train_supervised.py` 已支持 `bend_2d` 监督训练基线，可直接读取 bend CFD 真值做 smoke / 正式 run。
