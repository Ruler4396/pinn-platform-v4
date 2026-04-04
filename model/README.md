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
- `results/`：从现在开始在 v4 中产生的新结果

## 快速入口

### 收缩流道正式低冲击主线

```bash
cd /root/dev/pinn-platform-v4/model
bash scripts/run_contraction_independent_mainline_lowimpact.sh
```

### 直接训练

```bash
cd /root/dev/pinn-platform-v4/model
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
cd /root/dev/pinn-platform-v4/model
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
