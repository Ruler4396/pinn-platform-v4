# 几何增强编码消融实验

## 实验设置

- 数据族：收缩流道 `contraction_2d`
- 训练工况：`C-base, C-train-1..5`；验证工况：`C-val`；测试工况：`C-test-1, C-test-2`
- 对比项 A：仅坐标 + 全局参数（`basic`）
- 对比项 B：坐标 + 几何增强编码（`geometry`，与主线一致，去掉 `inlet_profile_star`）
- 公平性约束：四组消融均使用 `soft wall`，其余主要训练超参与主线保持一致
- 稀疏重建口径：使用 `obs_sparse_5pct` 训练，在 dense test 上评估全场恢复误差

## 指标汇总

| 指标 | 仅坐标+全局参数 | 坐标+几何增强编码 | 绝对改善 | 相对改善 |
| --- | ---: | ---: | ---: | ---: |
| 几何泛化误差 | 0.5753 | 0.0308 | 0.5445 | 94.64% |
| 压力误差 | 0.2681 | 0.1843 | 0.0837 | 31.24% |
| 稀疏重建误差 | 0.7883 | 0.1643 | 0.6240 | 79.16% |

## 解释

- 几何泛化误差越低，说明模型面对未见收缩比/长度比几何时，速度场外推更稳定。
- 压力误差越低，说明几何编码不仅改善速度场，也改善压力恢复。
- 稀疏重建误差越低，说明在只给少量观测点时，几何增强编码更有利于恢复全场。

## 对应运行

- `basic_dense`: `contraction_independent_basic_dense_softwall_ablation_v1_20260420`
- `geometry_dense`: `contraction_independent_geometry_dense_softwall_ablation_v1_20260420`
- `basic_sparse`: `contraction_independent_basic_sparse5clean_softwall_ablation_v1_20260420`
- `geometry_sparse`: `contraction_independent_geometry_sparse5clean_softwall_ablation_v1_20260420`
