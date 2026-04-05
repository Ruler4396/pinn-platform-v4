# PROJECT_EVOLUTION_V1_TO_V4

## 目的

这份文档用于按时间顺序整理本项目从 `pinn_bifurcation_controlled_v1`、`pinn_v2`、`pinn_v3` 到 `pinn_v4` 的演进过程，重点记录：

- 每一代项目在解决什么问题
- 哪些尝试被证明有效，最终沉淀为后续主线
- 哪些尝试失败、冻结，或只起到过渡作用
- 为什么项目会从一代切到下一代

这不是“把所有试验都写成成功故事”的回顾，而是保留真实的试错路径。

## 信息来源与可信度说明

### 直接可核对来源

- 本地文档与代码：
  - `pinn_v2/README.md`
  - `pinn_v2/docs/development.md`
  - `pinn_v2/docs/experiment_plan.md`
  - `pinn_v3/README.md`
  - `pinn_v3/docs/PROJECT_CHANGELOG.md`
  - `pinn_v3/docs/EXPERIMENT_LOG.md`
  - `pinn_v3/docs/MAINLINE_SWITCH_20260401.md`
  - `pinn_v3/docs/KEY_RUNS_AND_PROJECT_SUMMARY_20260404.md`
  - `pinn_v4/README.md`
  - `pinn_v4/docs/PROJECT_CHANGELOG.md`
  - `pinn_v4/docs/MIGRATION_FROM_V3_20260401.md`

### 远端补充来源

- GitHub 归档仓库：
  - `https://github.com/Ruler4396/pinn_bifurcation_controlled_v1`
  - `https://github.com/Ruler4396/pinn_v2`

### 关于时间戳的说明

- `v3`、`v4` 的关键时间点可以从本地文档标题与 changelog 直接恢复。
- `v1` 的关键时间点主要来自 GitHub 归档 README 中的归档日期与阶段文档日期。
- `v2` 在本地只保留了精简后的可复现工作区，没有保存完整历史结果和详细 changelog，因此其“具体哪一天发生了什么”无法完全恢复。文中关于 `v2` 的定位，部分属于基于现存材料的谨慎推断。

## 如何定义“成功”和“失败”

为避免混淆，这里做如下区分：

- 成功：某条路线不仅能跑通，还能形成稳定 benchmark、through-gate、3-seed 复核，或最终成为下一代主线基础。
- 失败：某条路线被明确记录为无效、退化、未 through-gate、被冻结，或在关键目标上始终无法闭环。
- 过渡：某条路线虽然没有成为最终主线，但帮助澄清问题、沉淀数据链路、评估口径或工程结构。

## 时间线总览

| 阶段 | 时间锚点 | 主题 | 结论 |
| --- | --- | --- | --- |
| `v1` | 2026-03-27 ~ 2026-03-28 可直接锚定 | 受控新主线，继续攻分叉/Y 型 benchmark ladder | 简单 benchmark 成功，高角度 Y 型长期不过 gate，最终停在 `P5~P7` |
| `v2` | 早于 `v3`，具体日期未完全恢复 | T/Y 分叉早期 PINN 工作区，强调 100% 基线、泛化、稀疏采样 | 形成了可复现训练骨架，但没有留下足够强的“正式成功主线”证据 |
| `v3` | 2026-03-30 起 | 从分叉回退，重启为 `contraction_2d + bend_2d` | 这是第一次真正形成论文级主线的阶段 |
| `v4` | 2026-04-01 起 | 从 `v3` 中抽离精简主线，冻结历史包袱 | 不是重新开题，而是把最成熟主线收束为可持续工作区 |

## 项目日志版时间线

下面这一部分按“日志”来写，而不是只做概括。每条都尽量回答三件事：

- 当时做了什么
- 相比上一阶段有什么进步
- 还剩哪些没有解决的问题

## v1 日志

### 2026-03-27：P5 正式退出，转向 P6 bounded residual

做了什么：

- 完成 `P5` 两条最小单机制入口的首轮远端 formal。
- 明确定性：
  - `v8 authority transition closure`：无效但可解释
  - `v9 semantic amplitude closure`：中性结果 / 无效但可解释
- 正式触发 `P5 scalar closure family` 的阶段退出复核。
- 新建 `P6 bounded residual` 的最小入口设计与远端实现计划。

带来的进步：

- 项目没有继续在 `v8 / v9` 上做无意义细修，而是明确承认这一家族当前不再值得继续投资源。
- “失败”第一次被写成显式决策，而不是隐性搁置。
- 下一阶段入口被提前设计出来，项目没有因为 P5 失败而停摆。

当时还没解决什么：

- 高角度 Y 型上的主坏验证链仍然存在。
- `bounded residual` 只是进入最小入口设计，还没证明能真正改善关键指标。

### 2026-03-27：P6 v10 首轮 formal，确认仍然 bad_validation

做了什么：

- 完成 `P6 v10 bounded residual` 首轮远端 formal。

带来的进步：

- 至少验证了：问题不是“bounded residual 还没接入”，而是“接入后改善极小，且副作用明显”。

当时还没解决什么：

- 主坏点仅微小改善。
- `crossflow_rel` 明显带脏。
- 结果仍是 `bad_validation=true`。
- 默认策略因此不是做 seed check，而是先回到机制复盘。

### 2026-03-27：v1 的中前期成果被统一写清

做了什么：

- 在 README 里把 `P1 ~ P4` 到当时为止已经完成的 through-gate 链条统一整理出来。

带来的进步：

- 项目终于明确区分出“已经稳定 through-gate 的 benchmark”和“当前仍未 through-gate 的复杂 benchmark”。
- 已确认成功的部分包括：
  - `P1` 收口
  - `P2` constant-width / single-outlet 到高曲率单弯折的 `3-seed` 低冲击确认
  - `P3` 多层 variable-width / single-outlet benchmark 的 `3-seed` 低冲击确认
  - `P4 owner-clean` canonical benchmark through-gate
  - `P4 junction fixed blending` 阶段由 `B3-v2 owner-safe hard-wall + fixed-blending pressure basis` 接住

当时还没解决什么：

- 这些成功都集中在单出口、低复杂结构、确定性骨架链条。
- 一旦进入高角度 Y 型，through-gate 仍然卡住。

### 2026-03-27：高角度 Y 型热点修补链完成，但仍未过 gate

做了什么：

- 对 `high-angle junction fixed blending` 做了一整套逐层热点修补：
  - `trunk-priority side assignment`
  - `truth-side / blending-authority`
  - `same-side / no-opposite-branch authority`
  - `wider trunk-authority band`
  - 分叉中心感知门函数
  - 双重重叠核心带成对共享向量骨架

带来的进步：

- `speed_max_rel` 从约 `0.400` 一路压到约 `0.318`
- 若只看局部热点，很多点位确实被精确修掉了
- 项目对“问题到底在什么结构层级”有了越来越清楚的认识

当时还没解决什么：

- through-gate 目标仍未达到
- 最大误差点不断从一个区域转移到下一个区域
- 项目最终确认：问题已经不是单点补丁，而是共享向量语义、局部基底和横向速度口径之间的结构错位

### 2026-03-28：归档并退役 v1

做了什么：

- 将 `pinn_bifurcation_controlled_v1` 作为归档仓推送到 GitHub。
- 本地与远端原主工作区进入清理退役流程。
- 默认策略从“继续扩实验”切换为“结构整理与文档降复杂”。

带来的进步：

- `v1` 的价值被完整保留：代码、文档、实验结构都留档，不再担心随着清场丢失。
- 项目终于承认：这一代的核心成果是“benchmark ladder 与失败边界”，不是“把高角度 Y 型彻底做成”。

当时还没解决什么：

- 高角度 Y 型的根问题仍未闭环。
- 后续必须决定：是继续顶复杂分叉，还是换一个更可收束的研究对象。

## v2 日志

### 早期过渡阶段：把分叉问题收成可复现工作区

做了什么：

- 建立 `pinn_v2` 工作区，围绕 `T / Y` 分叉流道组织最小可复现闭环：
  - `freefem/` 真值生成
  - `src/` 训练与评估
  - `configs/` 基线配置
  - `scripts/` 启动与实验脚本
  - `docs/` 开发规范与实验计划

带来的进步：

- 相比 `v1`，`v2` 更像一个可以直接上手复跑的项目骨架，而不是以阶段门禁为主的复杂实验母仓。
- 问题被整理为更清楚的三步：
  1. 先过 `100%` 采样基线
  2. 再看泛化
  3. 再做稀疏采样重建

当时还没解决什么：

- 仍然停留在 `T / Y` 分叉拓扑。
- 没有找到足够强、足够稳定的正式成功主线。
- 历史材料不足，无法恢复出像 `v3` 那样详细的 run 级演化记录。

### v2 时期的技术方向信号

做了什么：

- 从文件名上可以看到，此时已经开始试探：
  - `adaptive`
  - `streamfunction`
  这类替代方向

带来的进步：

- 说明项目并没有只盯着单一训练脚本，而是在摸索更稳的表达方式。

当时还没解决什么：

- 没有证据表明这些尝试在 `v2` 阶段形成了正式主线成果。
- 也没有足够文档支持把 `v2` 写成一个完整成功阶段。

### v2 的最终位置

做了什么：

- 保住了分叉研究的可复现骨架。

带来的进步：

- 为后续真正改道提供了“不要从零重新搭项目”的基础。

当时还没解决什么：

- thesis mainline 仍未出现。
- 分叉流道依旧过难。

## v3 日志

### 2026-03-30：正式从分叉回退，重启为 contraction + bend

做了什么：

- 初始化 `pinn_v3`。
- 明确把研究对象切换为：
  - `contraction_2d`
  - `bend_2d`
- 同时建立：
  - `PLAN`
  - `EXPERIMENT_LOG`
  - `PROJECT_RUNBOOK`
  - `PROJECT_CHANGELOG`
  - 训练协议、数据契约、图表清单等项目级文档

带来的进步：

- 这是项目第一次真正从“复杂分叉求闭环”转到“先做可控参数化几何”的战略转向。
- 研究对象一下子变得更适合：
  - 稳定收敛
  - 参数化比较
  - 论文分析
  - 系统展示

当时还没解决什么：

- 新几何虽然更简单，但还没有任何正式 baseline 结果。
- 还需要先证明数据链路、训练链路和泛化口径都成立。

### 2026-03-30：两类几何的数据链路落地

做了什么：

- 落地 `contraction_2d` 与 `bend_2d` 的：
  - 几何描述
  - FreeFEM CFD 生成
  - 稠密场组织
  - 稀疏 / noisy 观测构造
  - `trainval_manifest`
  - `DATASET_SPEC`

带来的进步：

- 从这一刻起，训练、评估、导图和后续网站接入有了统一数据契约。
- 数据生成不再是手工散落脚本，而是项目级生产链路。

当时还没解决什么：

- 还没有证明模型在这些数据上能训练到可用精度。

### 2026-03-30：supervised baseline 首次闭环

做了什么：

- 完成 `contraction_supervised_baseline`
- 完成 `bend_supervised_baseline`
- 补齐 test split 的 CFD 数据
- 完成两类几何的离线 test 泛化评估

带来的进步：

- 第一次形成了论文可写的 baseline 上界。
- 当时已经得到比较清楚的边界：
  - contraction：
    - 包络内插值可以
    - 强外推明显困难
  - bend：
    - 曲率半径族内泛化尚可
    - 转角结构外推失败

当时还没解决什么：

- 壁面约束仍不够紧
- 仅靠 supervised 还不足以回答“PINN 的价值在哪里”

### 2026-03-30：weak-physics PINN 正式接入

做了什么：

- 把 `train_pinn_finetune.py` 从占位变成可运行入口
- 支持：
  - supervised checkpoint 初始化
  - `L_obs + L_bc + L_pde`
  - Stokes 残差 warmup
  - dense / sparse / noisy 切换

带来的进步：

- 项目第一次具备真正意义上的 weak-physics PINN 微调能力。

当时还没解决什么：

- PDE 接入不等于 PDE 有稳定收益。
- 接下来必须证明：PDE 不是“只是挂在训练图里”。

### 2026-03-30：PDE 价值排查开始，结论并不统一

做了什么：

- 在 contraction 与 bend 上分别做了 PDE 作用排查。
- 引入 `physics_score`、更细的 history 留痕、dense split 物理指标等机制。

带来的进步：

- contraction：
  - 在 `5% sparse` 场景下，PDE 开始在压力误差、压降和散度上体现正向作用
- bend：
  - 证明了 PDE 会改变训练结果，但收益不稳定

当时还没解决什么：

- PDE 没有形成“一刀切普遍更好”的证据。
- 后续必须转向更细的 backbone、边界、观测口径联合分析。

### 2026-03-30：文献导向 streamfunction 路线被证明不适合作为主线

做了什么：

- 实现 `train_pinn_streamfunction.py`
- 运行：
  - `contraction_stream_sparse5_lit_v1_20260330`
  - `bend_stream_sparse5_lit_v1_20260330`

带来的进步：

- 明确验证了这条文献风格路线在本任务上的真实表现，而不是停留在“理论上可能更对”。

当时还没解决什么：

- 这条路线虽然散度很好，但整体场误差、压力误差和压降都明显落后。
- 结论很清楚：它不能成为当前 thesis mainline。

### 2026-03-31：wall-only hard no-slip 成为关键转折点

做了什么：

- 把原“wall + inlet 耦合式乘法边界强制”重构为“仅 wall 的指数饱和 hard no-slip”
- 对照 run：
  - `contraction_hybrid_geo_adapt_wallonly_v1_20260331`

带来的进步：

- 这是 `v3` 最关键的结构修复之一。
- 在 `C-val` 上：
  - `Rel-L2(|V|)` 从 `0.43812` 降到 `0.06248`
  - 最大速度误差从 `51.65%` 降到 `8.92%`

当时还没解决什么：

- 速度问题被大幅拉回来了，但还没有达到最终最优口径。
- 后续还要继续做 loss 收束、backbone 优化和多 seed 验证。

### 2026-03-31：contraction 主线真正成熟

做了什么：

- 新增 tail-aware supervised backbone
- 新增 base-output-aware correction head
- 跑出：
  - `contraction_supervised_tailaware_v1_20260331`
  - `contraction_hybrid_tailbase_basefeat_v1_20260331`
  - `contraction_hybrid_tailbase_basefeat_ensemble3_v1_20260331`

带来的进步：

- contraction 主线第一次形成了非常像“正式答案”的 run 组合。
- 代表性结果：
  - `Rel-L2(|V|)=0.04352`
  - `Rel-L2(p)=0.05334`
  - `max speed err=6.07%`
  - `max p err=7.20%`

当时还没解决什么：

- pressure 口径、课程学习、geometry feature 等细节仍需要继续比较和收束。

### 2026-03-31：bend 的突破来自几何感知，而不是照搬 contraction 配方

做了什么：

- 先试图把 contraction 的 tail-aware 配方移植到 bend
- 结果前 200 epoch 明显落后，未保留
- 随后改做 bend 专用几何感知输入：
  - `bend_supervised_geometry_v1_20260331`

带来的进步：

- 找到了 bend 主线真正有效的方向：显式几何表达
- 在 `B-val` 上达到：
  - `Rel-L2(|V|)=0.00529`
  - `Rel-L2(p)=0.00984`
  - `max speed err=2.13%`
  - `max p err=2.16%`

当时还没解决什么：

- 还需要把这个 backbone 接到 hybrid / PINN 主线上，看看是否既保留几何优势，又带来物理一致性增益。

### 2026-03-31：bend hybrid 主线建立，但 PDE 不可替代性证据始终不够强

做了什么：

- 基于 geometry backbone 跑出：
  - `bend_hybrid_geombackbone_basefeat_v1_20260331`
- 同时围绕 PDE 不可替代性做了大量 probe：
  - `pressure-only`
  - `flux-only inlet BC`
  - `holdout_turnbox`
  - `noise3 / noise5`
  - `gap65to4`
  - `interior_focus`
  - 各类 `BC-only vs PDE` 配对 run

带来的进步：

- bend hybrid 主线本身是成功的，至少它比早期不少旧路线更稳定、也更适合离线评估和导图。
- 项目对“PDE 什么时候有效”有了更细的认识。

当时还没解决什么：

- 很多 PDE probe 只带来很小改进，甚至局部变好全局变差。
- 因此它们不适合直接写成论文主结论。

### 2026-04-01：正式切换到独立速度 / 独立压力双模型主线

做了什么：

- 通过 `MAINLINE_SWITCH_20260401.md` 明确宣布：
  - 正式主线切换为独立速度模型 + 独立压力模型 + 控制方程交替耦合
- 冻结旧 hybrid 目标工况稀疏重建路线

带来的进步：

- 项目终于从“还在同时维护多条主线”进入“主线正式收束”状态。
- 这条新主线更适合：
  - 阶段内 PDE
  - 目标工况 sparse reconstruction
  - 更清晰的速度 / 压力训练职责分离

当时还没解决什么：

- 虽然主线切换了，但 `v3` 内部已经积累太多历史路线和结果目录。
- 如果继续在 `v3` 上推进，误用风险很高。

## v4 日志

### 2026-04-01：从 v3 抽离出精简主线工作区

做了什么：

- 新建 `pinn_v4`
- 从 `v3` 中迁移：
  - 当前双模型主线训练脚本
  - 评估脚本
  - 导图脚本
  - 数据生成脚本
  - bend / contraction 的训练与验证数据

带来的进步：

- 历史路线与正式主线第一次被物理分开。
- 后续新实验默认不再从旧 hybrid / old probe 路线起跑。

当时还没解决什么：

- `v4` 虽然精简了，但还不是最终交付形态。
- 网站、API 和模型仍然分散在不同目录 / 仓库。

### 2026-04-01：正式主线固定为 independent + stage PDE

做了什么：

- 固定 `v4` 当前主线为：
  - 独立速度模型
  - 独立压力模型
  - 阶段内部分控制方程
  - 最终控制方程交替耦合
- 同时把关键入口脚本切到这条主线

带来的进步：

- 这条线成了后续论文和整合仓的默认口径。

当时还没解决什么：

- 导图、章节素材和系统交付层面还需要继续整理。

### 2026-04-02：v4 从“能训”推进到“能交论文素材”

做了什么：

- 继续重写 `export_field_maps.py`
- 显式指定中文字体
- 支持域外留白、相对误差百分比、纵向堆叠布局、连续几何裁剪
- 新增 `prepare_chapter5_assets.py`
- 重整 `results/thesis_assets/chapter5/`

带来的进步：

- `v4` 不再只是训练工作区，而变成“论文素材生成工作区”。
- 场图可读性和可直接引用程度大幅提高。

当时还没解决什么：

- 仍然是模型工作区，不是完整系统交付。
- 一些 bend 结果仍依赖 `v3` 时代形成的历史 run。

## pinn-platform-v4 日志

### 2026-04-04：把网站、API、模型整合成一个单仓库

做了什么：

- 新建 `pinn-platform-v4`
- 把：
  - `pinn-flow-visual-demo-v4` 的前端整理进 `web/`
  - 演示 API 整理进 `api/`
  - `pinn_v4` 模型主线整理进 `model/`
- 补齐整合仓的 README、布局文档、部署说明

带来的进步：

- 毕设第一次以“单项目”的方式呈现，而不是两个并列仓库。
- “模型 + 网站 + 接口 + 文档”被收在一个工作区里。

当时还没解决什么：

- 旧文档和部分 case/manifest 中仍残留旧路径
- bend 权重并不完整，需要 fallback 兜底

### 2026-04-04：本地跑通整合仓并切线上

做了什么：

- 验证了：
  - `npm run check`
  - `npm run test`
  - `npm run build`
  - Python 脚本 `py_compile`
  - 本地 `/simulate`
- 调整 API 为整合仓相对路径优先
- 为 bend checkpoint 缺失场景增加 `SyntheticBendRuntime`
- 切换线上静态站与 systemd service
- 外网验证首页、`healthz`、contraction `/simulate`、bend `/simulate`

带来的进步：

- 这一步真正把前几代的研究内容变成“外网可访问的正式版系统”。

当时还没解决什么：

- bend 当前线上不是完整真实 checkpoint 口径，而是 fallback 兜底
- 旧路径残留还没完全清光

### 2026-04-04 ~ 2026-04-05：文档开始从“工作记录”升级成“交付说明”

做了什么：

- 重写整合仓 `README`
- 新增：
  - `INTEGRATION_AND_DEPLOYMENT.md`
  - `PROJECT_EVOLUTION_V1_TO_V4.md`

带来的进步：

- 项目终于具备了对外可解释的层次：
  - README 解释项目本身
  - integration 文档解释当前系统怎么跑
  - evolution 文档解释为什么会走到今天这一步

当时还没解决什么：

- 演进文档仍可继续细化到更具体的 run 表格级别
- 如果后续要答辩或公开展示，可能还需要再补一版“精简答辩版摘要”

## 1. pinn_bifurcation_controlled_v1

### 阶段目标

`v1` 的核心目标不是“大而全地统一推进所有 case”，而是围绕“最大误差 `< 5%`”建立一条受控新主线。它的策略非常明确：

- 先冻结指标定义
- 先建立 benchmark ladder
- 先做阶段门禁
- 先做单机制验证

也就是说，`v1` 的最大贡献之一，不是某个单独模型，而是把“如何做受控试错”这件事工程化了。

### 已被证明成功的部分

根据 GitHub 归档 README，`v1` 在以下阶段取得了明确成功：

- `P1` 已收口。
- `P2` 的 `constant-width / single-outlet` 链条已推进到高曲率单弯折，并完成 `3-seed` 低冲击确认。
- `P3` 的多层 benchmark 已完成：
  - 首个“直中心线 + 线性变截面 + 单出口” benchmark
  - “单弯折 + 线性变截面 + 单出口” benchmark
  - “高曲率单弯折 + 线性变截面 + 单出口” stress benchmark
  - “stronger linear taper” benchmark
  这些都完成了首个 `30/30` 低冲击 run 和 `3-seed` 低冲击确认。
- `P4` 中：
  - `owner-clean` 被正式冻结为 `canonical first benchmark`
  - `R2 deterministic split scaffold` 在 canonical benchmark 上进入健康区并达到 `bad_validation = false`
  - `owner-clean` 的 `3-seed` 稳定确认完成
  - `B3-v2 owner-safe hard-wall + fixed-blending pressure basis` 成为 `junction fixed blending` 阶段的 canonical candidate

这些结果说明：`v1` 在“简单单出口 / 低复杂结构 / 确定性骨架”这条 ladder 上，确实走通了很多层。

### 明确失败或未闭环的部分

`v1` 的主要失败，不是“完全跑不动”，而是当 benchmark 升到高角度 Y 型分叉时，误差热点不断迁移，始终无法 through-gate。

README 中已经明确记录的失败链包括：

- 高角度 `high-angle junction fixed blending` benchmark：
  - 首轮 formal 中 `speed_max_rel ≈ 0.400 > 0.30`
  - 后续通过一系列 hotspot 修复，把最大误差逐步从约 `0.400` 降到约 `0.323`
  - 但仍未 through-gate
  - 且热点不断从 turning-entry、overlap、junction center、double-overlap core 带向外迁移
- `P5`：
  - `v8 authority transition closure` 被正式定性为“无效但可解释”
  - `v9 semantic amplitude closure` 被正式定性为“中性结果 / 无效但可解释”，formal 结果与 `v7` 基线完全一致
- `P6`：
  - `v10 bounded residual` 首轮 formal 仅有微小改善
  - 但 `crossflow_rel` 明显带脏，仍是 `bad_validation=true`
- `P7 limited learner`：
  - `v14.2`：稳定，但无效
  - `v14.3`：结构正确，但收权过头
  - `v14.3a`：最小松绑有效，但恢复很弱
  - `v14.3b`：对 trust shape 的松绑无效，结果不变

### 这一代的最终意义

`v1` 的最终结论可以概括为：

- 分叉问题不是完全没有进展
- 但高角度 Y 型分叉下，想同时满足“独立、稳定、最大误差过 gate”，一直没有闭环
- 项目因此证明了 benchmark ladder 的价值，也暴露了复杂分叉拓扑在当前路线下的结构性瓶颈

换句话说，`v1` 成功地“证明了哪里能做成”，也同样成功地“证明了哪里暂时做不成”。

## 2. pinn_v2

### 当前能恢复出的定位

`v2` 本地保留下来的信息比较少，但仍能看出它的角色：这是一个面向 `T / Y` 分叉流道的早期 PINN 工作区，重点不是庞杂历史，而是保留可复现所需的最小闭环：

- `src/` 训练与评估实现
- `configs/` 基线与测试配置
- `freefem/` 真值场生成脚本
- `scripts/` 环境初始化与实验启动脚本
- `docs/` 开发规范与实验计划

### 明确计划中的主线

`v2` 的实验计划很清楚，分成三步：

1. `100%` 采样率基线
   - 先对 `T` / `Y` 流道完成真值生成、训练与评估
   - 先过 `relative_l2(u,v,p)` 与流量守恒偏差这两类验收
2. 泛化能力
   - 扫不同入口峰值速度，对应不同 `Re`
3. 稀疏采样重建
   - 降采样率
   - 对分岔区和近壁区加大采样权重

### 可以认定为成功的部分

`v2` 至少有三点过渡性成功：

- 它把分叉问题整理成了一个更清晰的可复现工作区，而不是 `v1` 那样庞大的阶段门禁系统。
- 它保留了 `T / Y` 双流道、真值生成、训练、评估、采样这些最基本骨架。
- 从 `quick_test_adaptive.py`、`quick_test_streamfunction.py` 这些文件名看，`v2` 已经在尝试 adaptive 和 streamfunction 等替代路线。

### 明确不足

但与 `v3` 以后相比，`v2` 的不足也很明显：

- 本地没有保存足够完整的正式结果文档，无法证明 `T / Y` 分叉在稀疏重建上形成了稳定成功样例。
- 也没有留下类似 `v3` 那样系统的 `PROJECT_CHANGELOG`、`EXPERIMENT_LOG` 和“关键 run 锚点”。
- 它仍停留在分叉拓扑上，而这正是 `v1` 已经暴露出最难稳定收口的区域。

### 这一代的最终意义

`v2` 更像是 `v1` 之后的一次“简化重组”，而不是一次真正彻底改道后的成功代。

比较谨慎的说法是：

- `v2` 帮助保住了分叉流道研究的可复现骨架
- 但没有留下足够强的证据表明它已经找到可长期延续的 thesis mainline
- 这也是后续 `v3` 决定从分叉问题回退、转向更简单参数化几何的重要背景

## 3. pinn_v3

### 为什么会切到 v3

`v3` 的 README 开宗明义就写了：这是一个“用于重新启动 2D PINN 流场研究的最小工作区”，并且明确从早期 `Y/T` 分叉回退，改为两类更容易稳定收敛和分析的几何：

- `contraction_2d`
- `bend_2d`

这一步是整个项目真正的方向性转折：

- 不再硬顶复杂分叉拓扑
- 改为先在更简单、可参数化、可稳定分析的几何上建立成功样例

### 2026-03-30：重启成功，数据与监督基线闭环

`2026-03-30` 是 `v3` 的启动日，也是第一批真正意义上的成功：

- 完成 `contraction_2d` 与 `bend_2d` 的 FreeFEM CFD 数据生产链
- 建立统一 `DATASET_SPEC`
- 完成两类几何的 supervised baseline
- 完成 test split 数据与泛化评估

这一天形成的关键结论包括：

- 收缩流道：
  - 包络内插值可以
  - 强外推仍明显困难
- 弯曲流道：
  - 曲率半径族内泛化尚可
  - 转角结构外推失败

这批结果的意义很大，因为它们第一次把“能做成什么、做不成什么”从复杂分叉，转移到了更清晰、可解释的两类几何上。

### 2026-03-30：一些路线被明确证明失败

`v3` 不是一路顺风，相反，它留下了很多很有价值的失败证据。

最典型的是文献导向 `psi,p` streamfunction PINN：

- `contraction_stream_sparse5_lit_v1_20260330`
- `bend_stream_sparse5_lit_v1_20260330`

这些 run 的共同特点是：

- 散度控制很好
- 但整体速度场、压力场和压降重建远差于当前 `u,v,p` 主线

这类失败很关键，因为它说明：

- “物理结构看起来更正统”不等于“在当前任务上更有效”
- 项目后续选择 `u,v,p` 主线，不是偷懒，而是被实验倒逼出来的

### 2026-03-30：PDE 开始显露价值，但只在部分场景成立

这一阶段还做了大量“PDE 是否真的起作用”的排查。

结果并不统一：

- 在 `contraction_2d` 的 `5% sparse` 场景中，PDE 开始在压力误差、压降恢复和散度控制上出现正向作用
- 在 `bend_2d` 上，PDE 虽然影响训练，但收益不稳定，甚至在某些设置下会恶化

这个结论很重要，因为它让 `v3` 没有误把“PDE 一定普遍更好”写成主线叙事，而是开始转向：

- 监督 backbone
- weak-physics 微调
- 场景化比较

### 2026-03-31：v3 真正进入论文主线期

`2026-03-31` 是 `v3` 最关键的一天。大量后续论文叙事、关键 run 和核心改动都集中在这一天。

#### 成功尝试 1：wall-only hard no-slip 重构

`README` 中明确记录：

- 原“wall + inlet 耦合式乘法边界强制”被重构为“仅 wall 的指数饱和 hard no-slip”
- 正式对照 run `contraction_hybrid_geo_adapt_wallonly_v1_20260331`
- `C-val` 上 `Rel-L2(|V|)` 从 `0.43812` 降到 `0.06248`
- 最大速度误差从 `51.65%` 降到 `8.92%`

这是整个项目里最具决定性的结构修正之一。

#### 成功尝试 2：tail-aware backbone + base-output-aware correction

这条线在 contraction 上形成了真正成熟的主线结果：

- `contraction_supervised_tailaware_v1_20260331`
- `contraction_hybrid_tailbase_basefeat_v1_20260331`
- `contraction_hybrid_tailbase_basefeat_ensemble3_v1_20260331`

其中 `contraction_hybrid_tailbase_basefeat_v1_20260331` 在 `C-val` 上达到：

- `Rel-L2(|V|)=0.04352`
- `Rel-L2(p)=0.05334`
- `max speed err=6.07%`
- `max p err=7.20%`

这已经是很明显的“主线级成功”。

#### 成功尝试 3：bend geometry-aware backbone

`bend` 的突破并不是直接来自 contraction 经验照搬，而是来自显式几何输入。

关键 run：

- `bend_supervised_geometry_v1_20260331`

该 run 在 `B-val` 上达到：

- `Rel-L2(|V|)=0.00529`
- `Rel-L2(p)=0.00984`
- `max speed err=2.13%`
- `max p err=2.16%`

这说明 bend 主线的核心增益之一，不是单纯增加 loss，而是把几何表达方式做对。

#### 成功尝试 4：bend geometry backbone warm-start hybrid

后续又在此基础上发展出：

- `bend_hybrid_geombackbone_basefeat_v1_20260331`

这是 bend 主线中最重要的 hybrid run 之一，也标志着：

- supervised 不再只是对照组
- 它变成了 hybrid / PINN 的 backbone

### 2026-03-31：大量“失败但有价值”的探索

同一天也做了很多没有变成最终主线的尝试，主要集中在“怎么证明 PDE 不可替代”这件事上。

从 `PROJECT_CHANGELOG.md` 可以看出，这些尝试包括：

- `pressure-only`
- `flux-only inlet BC`
- `holdout_turnbox`
- `noise5`
- `gap65to4`
- `interior_focus`
- 各类 `BC-only` vs `hybrid/PDE` probe

它们有一个共同特点：

- 很多实验不是完全失败
- 但没有形成足够强、足够稳定、足够可写进论文主结论的分离效果

具体表现为：

- 某些指标改善很小
- 某些局部变好但全局不稳
- 某些压力误差下降但压降误差上升
- 某些甚至反向退化

因此，这一阶段真正成熟的经验不是“PDE 在所有场景下都明显更强”，而是：

- PDE 的价值是条件性的
- 需要依赖 backbone、观测口径、边界约束和评估目标一起讨论

### 2026-04-01：独立速度/压力双模型成为新正式主线

`2026-04-01` 的 `MAINLINE_SWITCH_20260401.md` 已经写得很明确：

- `v3` 正式主线切换为：
  - 独立速度模型
  - 独立压力模型
  - 控制方程交替耦合
- 旧 hybrid correction 目标工况稀疏重建路线冻结为历史基线

这意味着 `v3` 又发生了一次重要收束：

- 不是简单地继续在旧 hybrid 路线上叠补丁
- 而是把“速度”和“压力”拆开，配合阶段内 PDE 继续推进

这一代的最终价值，就是把项目真正带到了“有 thesis 主线、也有反例和边界”的状态。

## 4. pinn_v4

### 为什么新开 v4

`v4` 不是完全重新开题，而是主动“减负”。

`MIGRATION_FROM_V3_20260401.md` 已写明原因：

- `v3` 已积累太多历史路线：
  - 旧 supervised 基线
  - hybrid 校正头路线
  - streamfunction 路线
  - PDE 不可替代性探针
  - inverse / pressure-only / noisy / probe 等历史实验
  - 大量结果目录

这些内容虽然重要，但继续放在同一个工作区里，容易误用。

### v4 保留下来的成功主线

`v4` 的正式主线被固定为：

1. 速度模型单独训练
2. 压力模型单独训练
3. 速度阶段加入连续性项
4. 压力阶段加入动量项
5. 最后再做控制方程交替耦合

也就是：

- 独立速度模型
- 独立压力模型
- 阶段内部分控制方程
- 最终控制方程耦合

### 这一代的成功

`v4` 的成功不在于“推翻前代另起炉灶”，而在于：

- 把 `v3` 后期真正有价值的主线提纯出来
- 清走大量历史噪音
- 把训练、评估、导图、章节素材整理做成更适合论文收口的工作区

`PROJECT_CHANGELOG.md` 中明确记录的正向成果包括：

- 新建 `v4`，承接 `v3` 最新主线
- 把训练、评估、导图和数据生成链条迁移进来
- 重写 `export_field_maps.py`
- 增强中文字体、相对误差百分比、域外留白、规则渲染网格插值等论文导图能力
- 新增 `prepare_chapter5_assets.py`，把第 5 章图片批量整理出来

### 这一代的限制

`v4` 的局限也要说清楚：

- 它不是一条完全全新的算法路线
- 它本质上是“把 `v3` 中已经赢下来的主线固化”
- 一些 bend 结果和论文素材仍依赖 `v3` 时期形成的历史 run

所以，`v4` 更像“主线收束版”，而不是“方法论大爆炸版”。

## 5. pinn-platform-v4（当前整合交付形态）

### 为什么还需要单独的整合仓

到 `v4` 为止，模型主线已经基本收束，但整个毕设项目仍然分散在两个方向不同的仓库里：

- 一个仓库偏网站与演示
- 一个仓库偏模型、训练与论文资产

这会带来几个问题：

- GitHub 提交层面看起来像两个项目，而不是一个完整毕设
- 前端与模型接口之间存在路径和部署边界
- 论文交付时，不容易直接解释“系统”和“模型”是如何合在一起工作的

因此，后续又出现了当前这个整合仓：

- `pinn-platform-v4`

它不是新的算法版本，而是当前最接近“正式交付版”的单仓库形态。

### 这一阶段真正完成了什么

当前整合仓完成的，不再是某个单独 loss 或单个 run 的比较，而是把前几代已经跑通的内容真正接到了一个可用系统里：

- 把网站前端整理到 `web/`
- 把统一 API 整理到 `api/`
- 把 `v4` 的模型主线整理到 `model/`
- 把部署、运行、关键 run、演进说明放进同一个仓库
- 把整合后的站点挂到正式外网地址并验证可访问

这一步的意义是：

- 研究链路第一次以“完整系统”而不是“多个分散工作区”形式呈现
- 论文中的“模型 + 数据 + 系统实现”真正被收成一个交付物

### 这一阶段的限制

当前整合仓仍然不是“所有历史内容都被优雅统一”：

- 一些历史文档仍保留旧项目名和旧绝对路径
- bend 当前线上口径存在 checkpoint 缺失时的 fallback
- `model/results/` 里的正式大权重与评估产物仍不适合直接入 Git

但从交付角度看，它已经足够代表当前毕设的正式版形态。

## 版本继承关系矩阵

下面这张表，比“版本号”本身更重要。它说明每一代到底把什么东西真正留下来了。

| 来源版本 | 留下来的核心资产 | 下一代如何继承 |
| --- | --- | --- |
| `v1` | benchmark ladder、阶段门禁、失败边界意识 | `v2` 继承了“先过基线、再看泛化、再做稀疏”的问题拆法 |
| `v2` | T/Y 可复现训练骨架、真值生成与采样问题定义 | `v3` 继承了“最小可复现闭环”，但放弃了继续硬攻复杂分叉 |
| `v3` | `contraction_2d` / `bend_2d` 数据链路、supervised backbone、hybrid 与 independent 主线、评估脚本、关键 run | `v4` 继承并冻结为精简主线 |
| `v4` | 精简主线工作区、章节导图工具、迁移后的稳定入口 | `pinn-platform-v4` 将其接入网站、API 和单仓库交付 |

## 跨版本总评：哪些尝试失败了，哪些真正成功了

### 可以明确归为失败或未闭环的路线

- `v1` 高角度 Y 型分叉 through-gate 闭环
- `v1` 的 `v8 / v9 / v10 / v14` 等后段修正家族
- `v3` 的文献导向 `psi,p` streamfunction 主线
- `v3` 中大量试图制造“PDE 不可替代性强分离”的 probe
- `v3` 中直接把 contraction 的 tail-aware 配方移植到 bend 的尝试

这些尝试并非毫无价值，但都没有成为最终正式主线。

### 可以明确归为成功并沉淀为后续主线的路线

- `v1` 的 benchmark ladder 与阶段门禁方法
- `v2` 的 T/Y 可复现训练骨架与采样问题定义
- `v3` 的 `contraction_2d + bend_2d` 参数化几何重启
- `v3` 的 supervised backbone 体系
- `v3` 的 wall-only hard no-slip 重构
- `v3` 的 tail-aware contraction 主线
- `v3` 的 bend geometry-aware backbone
- `v3` 后期独立速度 / 独立压力双模型主线
- `v4` 对上述主线的精简固化与论文资产整理能力

## 最终结论

如果把这四代项目放在一起看，可以得到一个很清楚的演进逻辑：

1. `v1` 证明了：复杂分叉问题不能靠局部补丁一路堆到闭环，必须有 benchmark ladder 和阶段门禁。
2. `v2` 证明了：T/Y 分叉问题可以被整理成可复现骨架，但光有骨架还不足以形成 thesis mainline。
3. `v3` 证明了：从复杂分叉回退到更简单的参数化几何，是一次正确的战略转向；项目也第一次真正得到可写进论文的成功主线。
4. `v4` 证明了：当主线已经基本成形后，下一步不是继续在历史包袱里打转，而是把真正有效的路线精简、固定并服务于交付。
5. `pinn-platform-v4` 证明了：最终交付不只是“有一个好 run”，而是把模型、接口、网站和说明文档收成一个完整系统。

从结果上看，这个项目真正的“成功”不是某一个单独 run，而是逐步完成了以下转变：

- 从复杂分叉迷宫，转向可控 benchmark
- 从“什么都想试”，转向论文叙事可收束的主线
- 从单纯训练模型，转向“数据链路 + 模型 + 评估 + 可视化系统”的完整闭环

## 推荐与 README 联动的阅读顺序

如果以后要向别人快速解释这个项目，推荐阅读顺序是：

1. 本文档：看版本演进和试错逻辑
2. `README.md`：看项目定位、方法和系统功能
3. `docs/INTEGRATION_AND_DEPLOYMENT.md`：看当前整合仓和正式部署口径
4. `model/docs/PROJECT_CHANGELOG.md`、`model/docs/MIGRATION_FROM_V3_20260401.md`：看 `v3 -> v4` 收束原因
