# pinn_v4

`pinn_v4` 是从 `pinn_v3` 中抽离出来的**精简主线工作区**，只保留当前继续实验所需的：

- 最新主线：**独立速度模型 + 独立压力模型 + 阶段内部分控制方程 + 最终控制方程耦合**
- 训练/评估/导图脚本
- bend / contraction 两类流道的训练与验证数据
- 继续生成 CFD 与观测子集所需的数据脚本与几何定义

## 当前主线

当前默认主线不是旧的 hybrid 校正头，也不是共享主干分头结构，而是：

1. 速度模型单独训练；
2. 压力模型单独训练；
3. 在速度阶段加入连续性项；
4. 在压力阶段加入动量项；
5. 最后再做速度模型与压力模型的控制方程交替耦合。

## 为什么新开 v4

`pinn_v3` 中已经累积了大量历史脚本、历史结果与试验路线。为了避免后续继续误用旧脚本、旧基线和历史产物，`pinn_v4` 只保留当前主线所需内容；`pinn_v3` 仍完整保留，作为历史归档与结果仓库。

## 目录

- `scripts/`：训练、评估、导图、数据生成脚本
- `src/data/`：几何、case registry、FreeFEM++ 相关数据模块
- `cases/`：bend / contraction 的训练、验证、测试数据
- `docs/`：精简后的运行说明与迁移说明
- `results/`：整合仓中保留的正式展示结果、checkpoint、日志与论文资产

## 展示资产

为了把毕业设计项目整理成一个可完整展示的单仓库，整合仓额外纳入了一批 `v4` 结果快照，包括：

- 收缩流道正式主线 `contraction_independent_geometry_notemplate_stagepde_mainline_v4`
- 稀疏采样 / 均匀采样 / 含噪采样对照 run
- 对应训练日志、评估日志、`best.ckpt`、`metrics.json`、`history.csv`
- `field_map_checks/` 下的误差导图检查素材
- `thesis_assets/chapter5/` 下的第五章图表与 `manifest`

这些内容主要用于论文展示、结果回顾和仓库对外演示，而不是继续扩展为完整的大型结果仓。

## 快速入口

### 收缩流道正式低冲击主线

```bash
cd ./model
bash scripts/run_contraction_independent_mainline_lowimpact.sh
```

### 直接训练

```bash
cd ./model
python3 scripts/train_velocity_pressure_independent.py \
  --family contraction_2d \
  --feature-mode geometry \
  --drop-features inlet_profile_star \
  --velocity-stage-continuity-weight 0.3 \
  --pressure-stage-momentum-weight 0.5 \
  --max-physics-points 512 \
  --max-retries 1
```

### 用新导图脚本读取任意 predictions 目录

```bash
cd ./model
python3 scripts/export_field_maps.py \
  --predictions-dir /path/to/predictions_dir \
  --cases B-test-1__ip_blunted \
  --output-dir /path/to/output_dir \
  --error-mode rel \
  --relative-floor-ratio 0.01 \
  --max-retries 1
```

该导图脚本默认：

- 域外留白，不着色；
- 相对误差按百分比显示；
- 显式指定中文字体。
