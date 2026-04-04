# 从 pinn_v3 迁移到 pinn_v4（2026-04-01）

## 迁移目的

`pinn_v3` 中已经累积了大量历史路线：

- 旧 supervised 基线
- hybrid 校正头路线
- streamfunction 路线
- 各类 PDE 不可替代性探针
- 多轮 sparse / noisy / pressure-only / velocity-only / inverse 等历史实验
- 大量历史结果目录

这些内容保留在 `pinn_v3` 中不删除，但后续继续主线实验时容易误用。

因此新建 `pinn_v4`，只保留当前继续实验所需内容。

## 已迁移内容

### 代码

- `src/data/*`
- `scripts/train_supervised.py`
- `scripts/train_velocity_pressure_independent.py`
- `scripts/evaluate_velocity_pressure_independent.py`
- `scripts/export_field_maps.py`
- `scripts/run_contraction_independent_mainline_lowimpact.sh`
- `scripts/generate_bend_case.py`
- `scripts/generate_bend_trainval_batch.py`
- `scripts/generate_contraction_case.py`
- `scripts/generate_contraction_trainval_batch.py`
- `scripts/generate_partial_observations.py`
- `scripts/evaluate_supervised.py`

### 数据

- `cases/bend_2d/*`
- `cases/contraction_2d/*`

## 不迁移的内容

以下内容继续保留在 `pinn_v3`：

- 历史结果目录 `results/`
- 旧路线相关大量专题文档
- 各类一次性 probe / 对照脚本
- 历史日志与归档说明

## 本轮顺手修复

在 `pinn_v4` 中，导图脚本还额外修复了两个问题：

1. 使用几何掩码屏蔽域外三角形，避免流道外被错误上色；
2. 相对误差改为百分比显示。

