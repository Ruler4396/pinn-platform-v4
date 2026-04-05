# REPRODUCTION_GUIDE

本仓库支持四种不同层级的复现。它们的难度和目标不同，建议先从前两级开始。

## 1. 展示级复现

目标：

- 跑起网站和 API
- 读取仓库中已经保留的 checkpoint、结果、论文素材
- 验证项目整体演示链路可用

这是当前最容易复现的一层，也是答辩、展示和仓库浏览者最容易完成的一层。

## 2. 结果核验级复现

目标：

- 基于仓库中已经保留的 case 数据和 checkpoint
- 重新执行评估、导图和论文素材整理脚本
- 复核关键 run 的指标、预测结果和图表输出

这一层不要求重新训练，也不要求重新生成 CFD 真值。

## 3. 训练级复现

目标：

- 基于仓库中已保留的 case 数据重新训练主线模型
- 重新生成 `metrics.json`、`history.csv`、`predictions/`

这一层能复现研究主线的方法与训练流程，但不保证得到与原始 run 完全一致的数值结果。

## 4. 全链路重建级复现

目标：

- 从几何与脚本出发重新生成 CFD 真值
- 重新构造稀疏观测
- 重新训练与评估

这一层额外依赖 `FreeFEM++`，门槛最高。

## 环境基线

当前公开结果对应的本机软件版本如下：

- Python `3.10.12`
- Node `22.22.0`
- npm `10.9.4`
- `numpy==2.2.6`
- `pandas==2.2.3`
- `matplotlib==3.10.8`
- `torch==2.11.0+cpu`

建议运行环境：

- Linux
- Python 3.10
- CPU 即可完成展示、评估与小规模复训
- 若要完全重建 CFD 真值，需要额外安装 `FreeFEM++`

## 一、从零开始准备环境

### 1. 克隆仓库

```bash
git clone git@github.com:Ruler4396/pinn-platform-v4.git
cd pinn-platform-v4
```

### 2. 安装 Python 依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r model/requirements.txt
```

### 3. 安装前端依赖

```bash
cd web
npm install
cd ..
```

## 二、展示级复现

### 1. 启动 API

```bash
python3 api/pinn_platform_api.py --host 127.0.0.1 --port 8011
```

### 2. 启动前端

```bash
cd web
npm run dev
```

### 3. 预期结果

- 前端页面能打开
- API 能读取 `model/results/` 中保留的关键 run
- 收缩流道可直接使用仓库中保留的正式 run 做推理
- 弯曲流道若缺少对应 checkpoint，会自动回退到 fallback 演示逻辑

## 三、结果核验级复现

### 1. 重新评估正式主线 run

```bash
cd model
python3 scripts/evaluate_velocity_pressure_independent.py \
  --family contraction_2d \
  --run-name contraction_independent_geometry_notemplate_stagepde_mainline_v4 \
  --eval-cases C-test-1,C-test-2 \
  --split-name test \
  --eval-source dense \
  --max-retries 1
```

### 2. 重新生成第五章素材

```bash
cd model
python3 scripts/prepare_chapter5_assets.py --max-retries 1
```

### 3. 可核验内容

- `model/results/pinn/`
- `model/results/field_map_checks/`
- `model/results/thesis_assets/chapter5/`

## 四、训练级复现

### 1. 运行正式主线训练脚本

```bash
cd model
bash scripts/run_contraction_independent_mainline_lowimpact.sh
```

### 2. 说明

- 脚本默认使用仓库中已经保留的 `cases/contraction_2d/` 数据
- 若目标 run 已经存在，脚本会先跳过训练再执行评估
- 如需强制重训，先移走对应的 `results/pinn/<run_name>/`

### 3. 结果预期

你应当能重新得到同一条主线的方法行为与输出结构，但以下内容不保证逐项完全一致：

- 最优 epoch
- 最终 checkpoint 参数
- 部分指标的小数级差异

原因包括：

- 神经网络训练存在随机性
- 不同 CPU / BLAS / PyTorch 小版本会带来细微差异

## 五、全链路重建级复现

### 1. 额外依赖

- `FreeFEM++`

### 2. 代表性命令

```bash
cd model
python3 scripts/generate_contraction_case.py \
  --case-id C-base \
  --field-source freefem_stokes_cfd \
  --max-retries 1
```

```bash
cd model
python3 scripts/generate_bend_case.py \
  --case-id B-base \
  --field-source freefem_stokes_cfd \
  --max-retries 1
```

### 3. 注意

- 当前仓库已经附带了正式 case 数据，因此大多数复现并不需要重新跑 FreeFEM
- 真正需要全链路重建时，再安装并验证 `FreeFEM++`

## 六、当前复现边界

本仓库已经能较好支持展示、核验和主线训练复现，但还存在以下边界：

- 没有把所有历史中间产物都纳入 Git
- 弯曲流道并未把全部历史权重都并入整合仓
- 一些历史文档仍保留旧项目名和旧工作区语义
- 少量 `trainval_manifest` 中的路径字段仍是历史元数据，不作为当前运行时依赖
- 全链路重建依赖外部 `FreeFEM++`

因此，更准确的说法是：

- 仓库已经支持“展示级复现”和“关键结果核验级复现”
- 仓库基本支持“主线训练级复现”
- 仓库尚未做到“任意机器一键无差别重建全部历史研究过程”

## 七、建议的复现顺序

建议其他使用者按下面顺序进行：

1. 先完成展示级复现，确认网站和 API 正常工作
2. 再完成结果核验级复现，确认关键 run 和图表能重新导出
3. 再尝试训练级复现，验证正式主线训练脚本可运行
4. 只有确实需要重建真值数据时，再进入 FreeFEM 全链路重建
