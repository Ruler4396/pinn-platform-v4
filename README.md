# pinn-platform-v4

`pinn-platform-v4` 是“基于 PINN 的微流控芯片内二维稳态流场稀疏重建与可视化系统设计与实现”对应的单仓库正式版项目。

它把原先分开的两个项目整理为一个完整闭环：

- `web/`：面向演示与分析的网站前端
- `api/`：连接前端与模型推理的统一后端接口
- `model/`：PINN 训练、评估、案例生成与论文资产工作区

项目目标不是替代高精度 CFD，而是在微流控芯片设计早期，提供一套兼顾物理一致性、推理速度、稀疏观测适应能力和可视化展示能力的快速重建方案。

## 项目背景

微流控芯片设计常常需要在多个通道方案之间快速比较。传统 CFD 仿真虽然精度高，但在频繁改参数、重建流场、做界面展示时成本较高；同时，真实测量往往只能获得少量、带噪声的观测点。

本项目面向二维、定常、低雷诺数微流控内流问题，选取两类典型参数化结构作为研究对象：

- 弯曲流道：对应流向连续偏转场景
- 收缩流道：对应截面连续变化与局部压降增强场景

围绕这两类结构，项目打通了：

1. 参数化几何建模
2. FreeFEM CFD 真值生成
3. 稀疏与含噪观测构造
4. 基于 PDE 约束的 PINN 重建
5. 离线评估与图像导出
6. 在线可视化系统接入

## 研究目标

本项目聚焦三个核心问题：

- 在少量观测点条件下，能否恢复速度场和压力场的整体分布
- 在轻度噪声扰动下，模型是否仍能保持稳定
- 模型结果能否直接服务于网站端的参数比较、结果查看和局部查询

对应到工程语境，就是为微流控芯片早期设计提供一个“比 CFD 更轻量、比纯插值更有物理依据”的快速验证工具。

## 方法概览

### 1. 参数化物理建模

项目统一采用二维、定常、不可压缩、低雷诺数流动设定，在无量纲框架下组织几何与场变量。控制方程以定常 Stokes 方程和连续性方程为基础，边界条件包括：

- 入口速度约束
- 壁面无滑移约束
- 出口压力参考约束

当前正式设置中，特征宽度取 `200 um`，入口平均速度取 `0.1 mm/s`，雷诺数远小于 `1`，符合黏性主导稳态内流的适用条件。

### 2. 双模型 PDE 耦合 PINN

本项目没有采用“单网络同时预测速度和压力”的方式，而是把问题拆成：

- 速度模型
- 压力模型

训练流程采用“先分开学习，再低学习率耦合修正”的组织方式。这样做的原因是速度和压力在数值尺度、训练难度和优化稳定性上并不完全一致，拆分后更容易控制训练过程，也更符合论文里的正式主线设计。

### 3. 统一数据链路

项目使用 FreeFEM 生成两类流道的 CFD 真值，并在统一数据契约下组织：

- 稠密真值场
- 边界点
- 几何信息
- 元数据
- 多采样率稀疏观测
- 含噪观测

这样训练、评估、图像导出和网站展示都建立在同一套基础数据之上，便于复现与扩展。

### 4. 区域感知与稀疏采样

为了更贴近“测点有限”的实际场景，项目围绕主体区、结构核心区、近壁区三类区域组织样本，并支持：

- 均匀随机采样
- 区域感知分层采样

这种设计的目的，是在相同观测预算下尽量保留收缩段、弯道转角和近壁区域等关键位置的流动信息。

## 系统功能

项目不仅包含离线训练，也包含完整的在线展示系统。当前网站支持：

- 预设案例选择
- 自定义收缩流道与弯曲流道参数
- 流体参数设置
- 观测采样率与噪声参数设置
- 速度场热力图显示
- 压力场热力图显示
- 流线图层切换
- 稀疏重建结果切换
- 任意点局部查询
- 主要指标显示，如 `Re`、峰值速度、平均压降、壁面代理量、曲率代理量等

从系统架构上看，整体按三层组织：

- 表示层：网页界面、参数编辑、结果展示
- 服务编排层：参数整理、请求调度、缓存与状态组织
- 模型适配层：统一不同推理来源的输入输出结构

## 论文对应的主要结论

根据论文中的正式实验与分析，本项目对应的方法具有以下特点：

- 在给定参数范围内，可以较好恢复速度场和压力场的整体分布
- 在 `5%` 左右的稀疏观测条件下，已经能达到较有实用价值的重建效果
- 在 `3%` 级别轻度噪声扰动下，模型性能会退化，但整体流场模式和压降趋势仍能保持稳定
- 对未见几何工况具有一定泛化能力，但几何偏移过大时，尤其是压力场误差会显著上升
- 阶段内 PDE 约束对速度场提升有限，但对压力场质量和局部压力误差控制更有帮助

论文的整体结论是：基于 PINN 的微流控流场稀疏重建在二维、稳态、低雷诺数、参数化典型通道场景下是可行的，并且适合与可视化系统结合，服务于方案比较和快速验证。

## 仓库结构

```text
pinn-platform-v4/
├─ README.md
├─ .gitignore
├─ docs/
├─ web/
├─ api/
├─ model/
└─ legacy/
```

### `web/`

- 来自原 `pinn-flow-visual-demo-v4`
- 保留前端源码、Vite 配置、测试和前端文档
- 已移除原仓内重复 `server/` 副本，统一以后端 `api/` 为准

### `api/`

- 当前统一 API 入口为 [`api/pinn_platform_api.py`](/root/dev/pinn-platform-v4/api/pinn_platform_api.py)
- 负责接收网页参数、组织几何与物理输入、调用模型并返回标准化结果
- 当前接口覆盖 `simulate`、`query-point`、`reconstruct`、`streamlines`、`probes`、`sweep` 等能力

### `model/`

- 来自原 `pinn_v4`
- 保留 `src/`、`scripts/`、`cases/`、`docs/`
- `results/` 默认不纳入 Git，仅保留结构和说明
- 负责训练、评估、案例生成、论文素材导出等核心研究工作

### `legacy/`

- 当前主要用于兼容历史资源说明
- 不作为正式主线的一部分

## 当前正式口径

当前整合仓对应的收缩流道正式主线 run 为：

- `contraction_independent_geometry_notemplate_stagepde_mainline_v4`

该 run 的 checkpoint 与配置默认保留在本地：

- `model/results/pinn/contraction_independent_geometry_notemplate_stagepde_mainline_v4/`

弯曲流道历史上对应的 run 名称仍然保留，但如果本地缺少对应 checkpoint，当前 API 会退回合成场 fallback，以保证网站在删除历史大权重后仍可正常启动和演示。

更详细的关键 run、整合改动与部署说明见：

- [`docs/INTEGRATION_AND_DEPLOYMENT.md`](/root/dev/pinn-platform-v4/docs/INTEGRATION_AND_DEPLOYMENT.md)
- [`docs/REPO_LAYOUT.md`](/root/dev/pinn-platform-v4/docs/REPO_LAYOUT.md)

## 快速开始

### 1. 前端开发

```bash
cd /root/dev/pinn-platform-v4/web
npm install
npm run dev
```

### 2. 启动 API

```bash
cd /root/dev/pinn-platform-v4
python3 api/pinn_platform_api.py --host 127.0.0.1 --port 8011
```

### 3. 模型训练 / 评估

```bash
cd /root/dev/pinn-platform-v4/model
bash scripts/run_contraction_independent_mainline_lowimpact.sh
```

## 线上访问

当前正式版网站入口：

- 页面：`https://aqsk.top/pinn-flow-visual-demo-v4/`
- API：`https://aqsk.top/api/pinn-v4/`

如果后续正式版路径调整，仓库中的部署口径以 `docs/INTEGRATION_AND_DEPLOYMENT.md` 为准。

## 适用范围与当前限制

本项目当前更适合以下场景：

- 二维
- 定常
- 低雷诺数
- 参数化典型微通道
- 少量观测点下的快速重建与可视化

当前限制主要包括：

- 对更复杂几何和更大参数范围的泛化能力仍有限
- 对真实实验级噪声与更复杂边界条件的适应性仍需继续验证
- 一部分历史文档仍保留旧项目名和旧路径表述
- `model/results/` 中的正式权重和评估产物默认不进 Git，需要本地或外部存储单独管理

## 相关文档

- 仓库结构说明：[`docs/REPO_LAYOUT.md`](/root/dev/pinn-platform-v4/docs/REPO_LAYOUT.md)
- 整合与部署说明：[`docs/INTEGRATION_AND_DEPLOYMENT.md`](/root/dev/pinn-platform-v4/docs/INTEGRATION_AND_DEPLOYMENT.md)
- 版本演进与试错时间线：[`docs/PROJECT_EVOLUTION_V1_TO_V4.md`](/root/dev/pinn-platform-v4/docs/PROJECT_EVOLUTION_V1_TO_V4.md)
- 模型工作区说明：[`model/README.md`](/root/dev/pinn-platform-v4/model/README.md)

## 说明

- 本仓库对应毕业论文中的正式整合版本，强调“研究链路 + 工程展示”一体化。
- `web/docs/` 与 `model/docs/` 中保留了原子项目文档，因此部分历史描述仍会提到旧目录名。
- 若后续需要把 README 再进一步改成“开源展示版”或“答辩提交版”，可以在此基础上继续压缩或扩写。
