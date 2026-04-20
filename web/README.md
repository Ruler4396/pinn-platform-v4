# PINN 微流控流场可视化演示站（v4 工作副本）

这是从 `/root/dev/pinn-flow-visual-demo` 复制出来的独立工作副本，用于在**不影响原答辩演示站**的前提下，对接 `/root/dev/pinn_v3`、`/root/dev/pinn_v4` 当前主线中已经变化的几何结构与可调参数。

> 当前保留 **demo numerical layer** 作为本地兜底；生产环境默认走 `remote`，已接入真实 checkpoint。

## 当前状态

- 已于 `2026-04-02` 从上游项目复制建立
- 旧公开演示站已下线，当前改造工作在本目录持续推进
- **阶段 1 + 阶段 2 已完成**：前端已从旧 `T/Y` 叙事迁移到 `contraction_2d / bend_2d`
- **阶段 3 + 阶段 4 已完成**：页面叙事、术语、讲解结构与发布路径已收口到 v4
- 当前 v4 已独立部署到 `/pinn-flow-visual-demo-v4/`
- 当前界面已进一步收口为**顶部热力图主舞台 + 下方双列控制/分析区**，并删除执行状态卡片
- 当前已接入 **remote inference** 线上链路：
  - `contraction_2d`：接 `pinn_v4` 真实 checkpoint
  - `bend_2d`：已接真实 bend checkpoint；当前采用 `pinn_v3` 中验证较稳定的几何感知模型承接

## 本轮已完成重点

- 数据结构从旧 `straight / t_junction / y_junction` 切换为：
  - `contraction`
  - `bend`
- 参数面板已同步为当前主线参数：
  - 收缩流道：`W`、`β`、`Lin/W`、`Lc/W`、`Lout/W`
  - 弯曲流道：`W`、`Lin/W`、`Rc/W`、`θ`、`inlet profile`、`Lout/W`
- 预设工况已切换为更接近 `pinn_v4` 的案例：
  - `C-base`
  - `C-test-2`
  - `B-base`
  - `B-test-1 + blunted`
- 几何绘制层已重写：
  - 收缩流道采用平滑收缩段轮廓
  - 弯曲流道采用圆弧中心线 + 曲率半径参数化
  - 弯曲流道显示方向已旋转为更适合横向展示的 landscape 朝向
- demo 数值层已重写为适配新几何：
  - 收缩流道突出喉部加速与压降变化
  - 弯曲流道突出曲率影响与入口剖面差异
- 单测已同步更新为 `contraction / bend` 体系
- 已新增页面叙事区块：
  - 当前案例摘要
  - 稀疏观测与重建链路说明
  - v4 发布路径说明
- 已完成独立 v4 路径部署：
  - `https://aqsk.top/pinn-flow-visual-demo-v4/`
- 旧入口继续保留维护页：
  - `https://aqsk.top/pinn-flow-visual-demo/`
- 最近一轮 UI 收口：
  - 热力图卡片上移并放大居中
  - 左右功能卡片改为位于热力图下方
  - 删除执行状态卡片与多余说明卡片，优先解决拥挤和重叠
- 最近一轮推理链路接入：
  - 新增本地 API 服务：`server/pinn_v4_api.py`
  - 线上 systemd 服务：`pinn-flow-visual-demo-v4-api.service`
  - 线上反代路径：`https://aqsk.top/api/pinn-v4/healthz`
  - 生产构建默认切换为 `VITE_INFERENCE_MODE=remote`
  - 已为 remote `simulate / reconstruct / sweep` 增加服务端响应缓存（TTL + 最大条数限制），减少刷新时对同一工况的重复推理
  - 已新增前端本地 preview 缓存：刷新时优先命中首屏热力图结果
  - 首屏数据改为轻量请求：先返回主热力图所需分辨率，流线 / 稀疏重建分开按需加载
  - 最新一轮 UI 收口：
    - 删除轴向中心线 / 喉后代表线卡片
    - 核心指标并入热力图主卡顶部
    - 主线案例下拉菜单移入顶部工具条，并与操作按钮对齐
    - 自定义选项拆分为“自定义收缩流道 / 自定义弯曲流道”
    - 流道类型跟随顶部下拉菜单自动切换，不再保留手动切换按钮
    - 流道参数 / 流体与采样 / 点位查询三张卡片横向并排在热力图下方
    - 辅助分析卡片已从主界面移除

## 当前保留功能

- 参数化几何建模：收缩流道 / 弯曲流道
- 流体参数配置：密度、黏度、平均入口速度、出口压力
- 速度热力图 / 压力热力图 / 流线图
- 任意点查询
- 稀疏重建切换显示

## 推理模式

```bash
VITE_INFERENCE_MODE=demo|remote
VITE_API_BASE_URL=http://127.0.0.1:8000/api/pinn
VITE_REQUEST_TIMEOUT_MS=15000
VITE_MAX_RETRIES=2
VITE_LOCAL_PREVIEW_CACHE_TTL_MS=1800000
```

- `demo`：本地数值演示层
- `remote`：远程真实 PINN / Python API
- `VITE_LOCAL_PREVIEW_CACHE_TTL_MS`：前端首屏 preview 缓存 TTL，默认 `1800000ms`

当前实际接入状态：

- **生产环境**：默认 `remote`
- **本地开发**：默认仍建议 `demo`
- **首屏加载策略**：
  - 默认先读取浏览器本地 preview 缓存
  - 若未命中，再请求远端轻量 `simulate`
  - 首屏只返回主热力图与指标所需数据
  - 流线、稀疏点/重建改为按需加载
- **当前主界面已删除的内容**：
  - 轴向中心线 / 喉后代表线卡片
  - 独立“核心指标”卡片
  - 独立“主线案例”卡片
  - “辅助分析”卡片
- **辅助分析说明**：
  - 严格来说，网站里原先的“黏度校准 / 参数扫掠”并不是 `pinn_v3 / pinn_v4` 主线中的原生在线能力，而是演示站额外做的轻量派生接口
  - 因此当前已从主界面删除，避免和真实 PINN 主线能力混淆
- **真实模型覆盖范围**：
  - 收缩流道：已接 `pinn_v4/results/pinn/contraction_independent_geometry_notemplate_stagepde_mainline_v4/best.ckpt`
  - 弯曲流道：
    - `parabolic`：当前走 `pinn_v4/results/pinn/bend_independent_geometry_notemplate_parabolic_mainline_v1_20260419/best.ckpt`
    - `skewed_top`：当前走 `pinn_v4/results/pinn/bend_independent_geometry_skewed_top_mainline_v1_20260419/best.ckpt`
    - `skewed_bottom`：当前走 `pinn_v4/results/pinn/bend_independent_geometry_skewed_bottom_mainline_v1_20260419/best.ckpt`
    - `blunted`：当前走 `pinn_v3/results/pinn/bend_independent_blunted_geometry_notemplate_medium_v1_20260401/best.ckpt`
    - 说明：四类入口剖面当前都已接真实双模型；`blunted / skewed_top / skewed_bottom` 为各自专门训练的入口剖面 checkpoint

## 本地开发

```bash
cd /root/dev/pinn-flow-visual-demo-v4
cp .env.example .env
npm install
npm run dev
```

默认开发地址：

- `http://127.0.0.1:4176`
- 若需与原项目同时运行，建议改用：`npm run dev -- --port 4177`

## 校验与构建

```bash
cd /root/dev/pinn-flow-visual-demo-v4
npm run check
npm run test
npm run build
```

## 发布目录

- 构建产物：`dist/`
- 当前已发布到：`/var/www/pinn-flow-visual-demo-v4`
- 对外路径：`https://aqsk.top/pinn-flow-visual-demo-v4/`
- 旧路径 `pinn-flow-visual-demo` 暂不切换，继续保留维护页

## 线上 API 服务

- systemd 服务：`pinn-flow-visual-demo-v4-api.service`
- 本地监听：`127.0.0.1:8011`
- 反代入口：`/api/pinn-v4/`
- 健康检查：

```bash
curl https://aqsk.top/api/pinn-v4/healthz
```

- 当前主要接口：
  - `/simulate`：首屏预览场，支持轻量分辨率与按需 include 选项
  - `/query-point`：点位查询
  - `/streamlines`：按需加载流线覆盖层
  - `/reconstruct`：按需加载稀疏点与重建场
  - `/probes`：当前前端主界面已不再使用，接口保留备用
