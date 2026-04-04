# PROJECT_CHANGELOG

> 说明：`2026-03-15` 的历史记录来自上游项目 `/root/dev/pinn-flow-visual-demo`。本工作副本自 `2026-04-02` 起独立演进。

## 2026-04-02

### 2026-04-02（补充）修复自定义预设回跳与 bend 远端坐标错位

- 修复顶部下拉中选择 `自定义收缩流道` 会被自动识别回 `收缩流道 / C-base` 的问题：
  - 现改为显式维护当前选中的 preset id
  - 当用户进入自定义分支后，不再因参数恰好与基线一致而自动跳回标准预设标签
- 修复 bend 在 remote 模式下的流场显示错位问题：
  - 根因是远端 API 返回的 bend 原始坐标系与前端横向 bend 几何显示坐标系不一致
  - 当前已在前端 remote adapter 中补上坐标映射：
    - `simulate / reconstruct / streamlines` 返回结果会先转换到前端展示坐标系
    - `query-point` 会先从前端展示坐标反变换回模型坐标，再请求后端
- 同时提升本地 preview 缓存版本，避免继续命中旧的 bend 错误缓存
- 继续修复 bend 热力图“几乎空白，只剩零散颜色点”的问题：
  - 根因是 bend 远端结果在前端坐标变换后，不再是规则网格
  - 原 rasterize 仍按规则网格渲染，导致热力图退化成稀疏点状
  - 当前已为 `FieldCanvas` 增加非规则散点场渲染路径，bend 会改用 scatter heatmap 方式铺底
- 同时调整下方三张卡片的响应式断点：
  - 桌面宽度下优先保持 `流道参数 / 流体与采样 / 点位查询` 三列横排
  - 避免过早退化成上中下单列堆叠
- 补充修复：此前 `.support-grid` 漏设 `display: grid`，导致三列模板未真正生效；现已修正
- 继续收口展示细节：
  - bend 主画布显示比例缩到约当前的 `80%`，减小相对收缩流道的垂直占用差距
  - 下方三张卡片改为等高拉齐
  - `点位查询` 返回值改为左右并排的紧凑行，减少空白并降低卡片高度压力

### 测试与校验

- 已通过：
  - `npm run check`
  - `npm run test`
  - `npm run build`

### 2026-04-02（补充）继续收口顶部与下方控制区

- 删除顶部 `remote 模式...` 长说明条，避免干扰首屏
- 顶部预设下拉不再显示“案例”文字标签，直接与操作按钮对齐
- 预设中的自定义项已拆为：
  - `自定义收缩流道`
  - `自定义弯曲流道`
- 流道类型现在跟随顶部预设下拉自动切换：
  - 删除参数卡内 `收缩流道 / 弯曲流道` 手动切换按钮
- 下方三张卡片重新排布为单行横向并排：
  - `流道参数`
  - `流体与采样`
  - `点位查询`

### 2026-04-02（补充）继续收口主界面：删掉路径曲线卡片、独立指标卡片与辅助分析卡片

- 根据最新页面收口要求，继续精简主界面结构：
  - 删除 `轴向中心线 / 喉后代表线` 两张卡片
  - 删除独立 `核心指标` 卡片，改为直接并入热力图主卡顶部
  - 删除只包含一个下拉框的 `主线案例` 卡片，并把案例选择器移到顶部工具条
- 重新核对 `辅助分析` 的能力边界后，确认：
  - 网站里原先的 `黏度校准 / 参数扫掠` 不是 `pinn_v3 / pinn_v4` 主线中的原生在线能力
  - 更接近演示站额外派生的轻量接口，而非当前应对外主张的真实 PINN 工作台能力
- 因此本轮已将 `辅助分析` 卡片从主界面移除，避免误导

### 测试与校验

- 已通过：
  - `npm run check`

### 2026-04-02（补充）首屏轻量化：本地 preview 缓存 + 按需加载拆分

- 前端新增本地 preview 缓存：
  - 文件：`src/lib/localResultCache.ts`
  - 默认 TTL：`1800000ms`
  - 首屏刷新时优先尝试命中浏览器本地缓存，减少重复等待
- 调整 `App.tsx` 首屏数据流：
  - 首屏 `simulate` 仅请求主热力图 + 指标所需 preview 分辨率
  - 切换到流线图层时才请求流线覆盖层
  - 稀疏点 / 重建继续保持独立按钮触发
- 调整远端接口与适配器：
  - `POST /simulate`：支持 `resolution / includeStreamlines / includeProbes / includeSparsePoints / includeReconstruction`
  - 新增 `POST /streamlines`
  - 新增 `POST /probes`
- demo adapter 同步支持相同的拆分式请求契约，避免本地开发与 production 行为脱节

### 测试与校验

- 已通过：
  - `python3 -m py_compile server/pinn_v4_api.py`
  - `npm run check`
  - `npm run test`
  - `npm run build`

### 2026-04-02（补充）bend 从 synthetic fallback 切到真实 checkpoint

- 调整 `server/pinn_v4_api.py` 的 bend 推理路由，不再继续使用 synthetic field：
  - `parabolic / skewed_top / skewed_bottom`
    - 当前接入 `pinn_v3/results/supervised/bend_supervised_geometry_v1_20260331/best.ckpt`
  - `blunted`
    - 当前接入 `pinn_v3/results/pinn/bend_independent_blunted_geometry_notemplate_medium_v1_20260401/best.ckpt`
- 新增两类 bend runtime：
  - `SupervisedFieldRuntime`
  - `IndependentFieldRuntime`
- `/healthz` 现会回显：
  - `contraction_run`
  - `bend_parabolic_run`
  - `bend_blunted_run`
- 当前策略说明：
  - `pinn_v4` 已直接承接 contraction
  - `pinn_v4` 目录下暂未发现同级可直接上线的 bend checkpoint，因此先以 `pinn_v3` 中评估表现较好的 bend run 接站
  - `skewed_*` 当前已不再走 synthetic fallback，但暂复用 parabolic bend 真实模型路径

### 测试与校验

- 已通过：
  - `python3 -m py_compile server/pinn_v4_api.py`
  - `systemctl restart pinn-flow-visual-demo-v4-api.service`
  - `curl https://aqsk.top/api/pinn-v4/healthz`
  - HTTPS `simulate` smoke：
    - bend `parabolic`：返回 `field_len=3565`、`streamlines_len=10`
    - bend `blunted`：返回 `field_len=3565`、`streamlines_len=10`
    - bend `skewed_top`：返回 `field_len=3565`、`streamlines_len=10`
  - bend 同请求重复调用可观察到缓存命中（`X-Response-Cache: HIT`）

### 2026-04-02（补充）尝试接入 pinn_v4：首版 remote 推理链路落地

- 新增最小 Python API 服务：
  - 文件：`server/pinn_v4_api.py`
  - systemd：`pinn-flow-visual-demo-v4-api.service`
  - 本地监听：`127.0.0.1:8011`
  - Nginx 反代：`/api/pinn-v4/`
- 前端 production 构建已切换为：
  - `VITE_INFERENCE_MODE=remote`
  - `VITE_API_BASE_URL=/api/pinn-v4`
  - `VITE_MAX_RETRIES=1`
- 本轮接入策略：
  - `contraction_2d`：接入 `pinn_v4` 当前主线 checkpoint  
    `results/pinn/contraction_independent_geometry_notemplate_stagepde_mainline_v4/best.ckpt`
  - `bend_2d`：由于当前 `pinn_v4` 工作区下未发现同级可直接上线的 bend checkpoint，先使用 v4 几何 + synthetic field 兜底，保证 remote 模式下网站整体可用
- 远端接口已对齐网站现有契约：
  - `/simulate`
  - `/query-point`
  - `/reconstruct`
  - `/calibrate-viscosity`
  - `/sweep`
- `reconstruct / calibrate / sweep` 当前采用轻量实现，优先保证答辩站能稳定走通 remote 链路
- 随后继续为 remote API 增加响应缓存：
  - 缓存对象：`simulate / reconstruct / sweep`
  - 默认缓存 TTL：`1800s`
  - 默认最大条数：`8`
  - 目的：避免刷新页面时对同一默认工况重复完整计算
  - 响应头会返回 `X-Response-Cache: MISS|HIT` 便于排查

### 测试与校验

- 已通过：
  - `python3 -m py_compile server/pinn_v4_api.py`
  - 本地 `healthz / simulate / query-point` smoke
  - `curl https://aqsk.top/api/pinn-v4/healthz`
  - 同一 `simulate` 请求二次调用可观察到 `MISS -> HIT`
  - `npm run check`
  - `npm run test`
  - `npm run build`

### 风险说明

- 当前站点已不是纯 demo，但也**还不是全量真实模型**：
  - 收缩流道：真实 v4 模型
  - 弯曲流道：暂时 synthetic fallback
- 因此后续若要宣称“v4 已完全接站”，还需要继续补上 bend checkpoint 与对应推理路径

### 2026-04-02（补充）布局重排：热力图上移独占主区域

- 按最新答辩视图要求，彻底改写工作台布局为：
  - 顶部单列：主热力图 / 压力图 / 流线切换区
  - 下部双列：参数、探针等必要卡片
- 不再继续依赖旧的“左中右三栏 + grid-area 修补”方式，改为显式的 `stage-column + support-grid` 结构，避免卡片继续互相压盖
- 删除执行状态卡片遗留后的无效布局语义，减少宽度竞争
- 放宽页面整体最大宽度，让热力图卡片获得更大的居中展示区域

### 2026-04-02（补充）bend 画布横向化

- 保持 `bend_2d` 参数语义不变，但将前端几何显示整体旋转为更适合横向答辩展示的 landscape 朝向
- 弯曲流道现在不再以“竖着拐上去”的视觉姿态出现，而是以更居中的横向弯道呈现
- 同步补充几何测试断言：验证 bend 场景在当前显示边界下宽度大于高度

### 测试与校验

- 已通过：
  - `npm run check`
  - `npm run test`
  - `npm run build`

### 阶段 3 + 阶段 4：页面叙事与发布路径收口

- 重构页面内容叙事：
  - Hero 区补充从旧 T/Y 演示迁移到 `contraction_2d / bend_2d` 的说明
  - 新增三张叙事卡：当前案例、稀疏观测链路、v4 发布路径
  - 在左侧面板补充“参数术语已切到论文主线”的说明
  - 在中间画布区补充按当前图层变化的观察提示
  - 在右侧增加“答辩讲解提纲”卡片，便于口头展示
- 调整术语与内容表达：
  - 去掉旧 `T / Y` 的主要叙事地位
  - 统一改为 `contraction / bend`、`region_aware`、`inlet profile` 等当前主线术语
  - 图表副标题和说明性文案同步改为当前论文语境
- 调整站点元信息：
  - `vite.config.ts` 的 base 改为 `/pinn-flow-visual-demo-v4/`
  - `index.html` 标题与 description 改为 v4 版本表达
- 完成 v4 路径上线准备并部署：
  - 新建静态目录 `/var/www/pinn-flow-visual-demo-v4`
  - Nginx 增加 `/pinn-flow-visual-demo-v4/` 路由
  - 已执行 `nginx -t` 与 `systemctl reload nginx`
  - 当前 `https://aqsk.top/pinn-flow-visual-demo-v4/` 已可访问
  - 旧路径 `https://aqsk.top/pinn-flow-visual-demo/` 继续保留维护页

### 测试与校验

- 已通过：
  - `npm run check`
  - `npm run test`
  - `npm run build`
  - `curl -I https://aqsk.top/pinn-flow-visual-demo-v4/`

### 2026-04-02（补充）布局修复：处理网格项重叠

- 针对实际页面仍出现的卡片相互压盖问题，继续收紧布局约束：
  - 为 workspace / chart grid 子项补充 `min-width: 0`
  - 为 panel / stage / chart / canvas 卡片补充 `overflow: hidden`
  - 为 `panel-head`、`stage-topbar` 补充换行能力
  - 为 `panel-note` 增加截断，避免长标签顶破列宽
  - 提前在较宽断点下将双曲线图改为单列，减少中栏横向挤压
- 继续删除左侧 preset 描述块，并将 geometry 顶部标签收短为 `contraction_2d / bend_2d`，进一步降低左栏高度与宽度压力

### 2026-04-02（补充）UI 收口：删除重叠的说明卡片

- 根据实际页面观察，移除一批会造成拥挤/重叠的说明性卡片与叙事块：
  - 顶部三张 story cards
  - 左侧 preset / geometry / sparse 区的额外说明块
  - 中间画布上方的视图说明块
  - 右侧“答辩讲解提纲”卡片
- 当前页面仅保留核心工作台：
  - 预设案例
  - 几何参数
  - 流体与采样
  - 主画布
  - 路径曲线
  - 点位查询
  - 指标
  - 辅助分析
  - 状态列表
- 这样做的目标是优先保证：
  - 不重叠
  - 不拥挤
  - 答辩时一眼就能找到主要功能区

### 文档

- 更新 `README.md`
- 更新 `docs/PROJECT_RUNBOOK.md`
- 更新本变更文档

## 2026-04-02

### 阶段 1 + 阶段 2：切换到 contraction / bend 主线

- 重构 `src/types/pinn.ts`：
  - 几何类型从 `straight / t_junction / y_junction` 改为 `contraction / bend`
  - 稀疏策略从 `feature_aware` 改为 `region_aware`
  - 扫掠变量从旧入口速度命名切为 `meanVelocity`
- 重构 `src/lib/presets.ts`：
  - 默认场景改为 `C-base`
  - 新增 `C-test-2`、`B-base`、`B-test-1-blunted` 等主线预设
- 重写 `src/lib/geometry.ts`：
  - 新增平滑收缩流道几何
  - 新增圆弧参数化弯曲流道几何
  - 增加 guide segments / stationing，供投影、流线与采样复用
- 重写 `src/lib/demoPhysics.ts`：
  - 收缩流道按局部宽度变化生成速度场与压降近似
  - 弯曲流道按曲率半径、弯角与入口剖面生成速度场近似
  - 稀疏采样与重建逻辑同步适配新几何
- 重构 `src/App.tsx`：
  - 左侧参数面板改为 `contraction / bend` 双体系
  - 新增预设案例选择器
  - 文案与图表标题同步切换到新主线
- 更新 `src/components/MetricCards.tsx` 与 `src/components/FieldCanvas.tsx`：
  - 指标标签改为更适合当前主线的通用指标
  - 画布语义改为收缩 / 弯曲流道
- 重写测试：
  - 几何测试切换到 contraction / bend
  - demoPhysics 测试补充收缩增强与 bend_2d 场景断言

### 测试与校验

- 已通过：
  - `npm run check`
  - `npm run test`
  - `npm run build`

### 文档

- 更新 `README.md`
- 更新 `docs/PROJECT_RUNBOOK.md`
- 更新本变更文档

## 2026-04-02

### 复制建立 v4 工作副本

- 新建工作目录：`/root/dev/pinn-flow-visual-demo-v4`
- 复制来源：`/root/dev/pinn-flow-visual-demo`
- 复制策略：
  - 保留源码、文档、配置与锁文件
  - 排除 `node_modules/`、`dist/` 与 `*.tsbuildinfo`
  - 先建立最小可回滚副本，再在此基础上继续修改
- 包名更新为 `pinn-flow-visual-demo-v4`，避免与原项目混淆
- 文档已同步改为新路径，并明确：
  - 当前副本用于对接 `pinn_v3 / pinn_v4` 更新后的几何结构与参数
  - 当前默认不直接覆盖原演示站发布目录

### 测试与校验

- 已通过：
  - `npm install --no-audit --no-fund`
  - `npm run check`
  - `npm run test`
  - `npm run build`

### 文档

- 更新 `README.md`
- 更新 `docs/PROJECT_RUNBOOK.md`
- 更新本变更文档

## 2026-03-15

### 官方几何修正 + 视觉重做（二轮）

- 按只读参考目录 `pinn_allcases_h15_20260314` 修正前端几何定义：
  - **Y 型** 改为官方 `official_geometry_v3` 的连续壁面拓扑
  - **T 型** 改为竖直入口、顶部左右水平出口的正式朝向
  - 默认场景改为 **Y-G5 风格参数**（`Wm=200 μm`、`Wb=100 μm`、`Lm=2 mm`、`Lb=2 mm`、`θ=60°`）
- 重写 `src/lib/geometry.ts`：
  - 从旧的矩形段拼接，改为正式多边形几何
  - 增加中心线、投影、边界与点内判定能力
- 重写 `src/lib/demoPhysics.ts`：
  - 数值演示层改为适配新 T/Y 拓扑
  - 主干 / 双支路方向场、压降与流线重新生成
  - 稀疏采样与重建逻辑同步适配新几何
- 重写 `FieldCanvas`：
  - 用 **连续热力图栅格渲染** 替代原先一块块点状/方块渲染
  - 新增 **等值线** 叠加，增强科研图感
  - 保留流线、稀疏采样点和探针，但整体做得更克制
- 全站样式改为统一白灰系统：
  - 删除此前偏黄页面底色与深黑画布的冲突
  - 改为更接近 Apple / OpenAI 产品页面的浅灰、白色、石墨层级
  - 图表也切换为浅色底，不再与主页面割裂
- 重做 App 状态管理：
  - 增加“参数已修改，待重算”提示
  - 避免草稿参数和已计算画布直接错位
- 修正 T / Y 切换退化问题：
  - 从 T 切回 Y 时不再沿用 90° 角导致画面看起来像 T 型
  - Y 型角度统一钳制在官方族范围内，防止几何退化
- 采样点与探针现统一裁剪在流道内部，并降低贴壁采样，避免视觉上“点跑到流道外”
- 答辩界面对外入口只保留 T / Y，不再暴露直流道模拟入口
- 补充单测：
  - 新增官方 Y/T 几何拓扑断言
  - 保留 demo 数值层稳定性测试

### 测试与校验

- 已通过：
  - `npm run check`
  - `npm run test`
  - `npm run build`

### 文档

- 更新 `README.md`
- 更新 `docs/PROJECT_RUNBOOK.md`
- 更新本变更文档
- 计划同步更新 `/root/dev/DEVELOPMENT_LOGBOOK.md`

## 2026-03-15

### 进一步收口

- 删除首页中大量说明性文字与“基准工况”式展示
- 将主界面收口为更聚焦的单工作台结构：
  - 左侧参数面板
  - 中部流场画布
  - 右侧探针 / 指标 / 扩展工具
- 进一步突出参数调节与可视化这两个最核心能力
- 保留重建、校准、扫掠功能，但降为次级工具，不再抢主界面叙事
- 重做流道画布视觉：
  - 改为更克制的深色技术画布
  - 使用 clipPath 约束流道内部热力图与流线
  - 降低原先“塑料感 / 玩具感”的白边与厚涂效果
  - 将边界描边、流线、稀疏点、探针标记做得更精细
- 精简指标卡与状态区文案，减少答辩时不必要的阅读负担

### 测试与校验

- 已通过：
  - `npm run check`
  - `npm run test`
  - `npm run build`

### 文档

- 更新 `README.md`
- 更新 `docs/PROJECT_RUNBOOK.md`
- 更新本变更文档

## 2026-03-15

### 重做

- 参考毕业论文初稿 `v5.1`，对站点进行一次大范围 UI 重做
- 将原先偏“AI dashboard / 霓虹玻璃拟态”的视觉方向，改为更接近商业公司产品站与技术工作台的混合风格
- 重新组织首页叙事结构：
  - 项目概览
  - 参数化工况配置
  - 流场工作台
  - 稀疏重建
  - 特征分析
  - 校准与扫掠
  - 系统实现说明
- 将论文中的核心表达前置到界面层：
  - T / Y 双拓扑
  - 三类特征区域分层采样
  - 15 μm 网格真值基线
  - 5%–15% 稀疏率演示
- 调整默认案例与参数表达，使其更贴近论文当前主线，而不是以流体预设切换作为首页主叙事
- 将入口速度展示单位改为更贴近论文语境的 `mm/s`
- 重写主页面排版与样式系统：
  - 顶部 Hero
  - 商业化 section 导航
  - 产品化 workbench 画布区
  - 对照式稀疏重建展示
  - 更清楚的特征指标说明卡
- 重写 `SectionNav`、`MetricCards`、`FieldCanvas`、`LineChart` 的文案与展示层级
- 更新默认预置工况为更贴近论文当前表达的 T / Y 几何案例

### 测试与校验

- 调整 demo 单测阈值，使其与新的默认工况规模一致
- 已通过：
  - `npm run check`
  - `npm run test`
  - `npm run build`

### 文档

- 更新 `README.md`
- 更新 `docs/PROJECT_RUNBOOK.md`
- 本次改版已同步补写到项目级文档与总日志

## 2026-03-14

### 新增

- 新建独立项目 `pinn-flow-visual-demo`
- 采用 `Vite + React + TypeScript` 搭建答辩演示站
- 建立统一的 `ScenarioInput / FieldPoint / ScenarioResult / InferenceAdapter` 契约
- 实现 demo 推理层：
  - 低雷诺数层流近似
  - 分叉流量分配启发式
  - 压降解析近似
  - 几何掩膜裁剪
- 实现 7 个一级导航：
  - 概览
  - 几何与参数
  - 流场可视化
  - 稀疏重建
  - 特征分析
  - 校准与扫掠
  - 方法说明
- 覆盖毕设演示所需 8 项功能：
  - 流场可视化
  - 流体参数配置
  - 微流控通道几何建模
  - 任意点查询
  - 稀疏数据流场重建
  - 流场特征提取
  - 物性参数校准辅助
  - 单一条件影响模拟
- 增加 `remote` 模式适配器
- 增加远端调用超时与最大重试机制：
  - `VITE_REQUEST_TIMEOUT_MS`
  - `VITE_MAX_RETRIES`
- 增加基础测试：
  - 几何掩膜
  - demo 推理稳定性
  - 重建 / 校准 / sweep

### 文档

- 补充 `README.md`
- 补充 `docs/PROJECT_RUNBOOK.md`
- 计划同步更新 `/root/dev/DEVELOPMENT_LOGBOOK.md`

### 边界说明

- 当前版本默认使用演示数值近似，不直接接入真实 PINN
- 目标优先级为“演示稳定 + 后续可替换”，不是当前阶段的最终数值精度
