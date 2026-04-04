# contraction_2d

平滑收缩流道 case 目录。

## 子目录
- `geometry/`：几何定义、采样边界、网格描述
- `cfd/`：CFD 配置、求解脚本、原始输出
- `data/`：训练/验证数据集
- `figures/`：流场图、误差图、剖面对比图

## 当前建议
优先从宽度比 `1:0.75` 的平滑收缩基线开始。

## 当前已落地
- `scripts/generate_contraction_case.py` 可生成 `C-base` 等 case 的 geometry / dense field / sparse obs / meta 骨架。
- 当前脚本已支持 synthetic smoke 与 `FreeFEM++ CFD` 两种 dense field 来源。
- `C-base` 已生成首份 FreeFEM++ CFD 数据，可作为后续训练的真实基线输入。
- 正式 CFD 接入占位模板：`cfd/freefem_contraction_template.edp`。
