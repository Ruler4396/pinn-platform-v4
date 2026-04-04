# PROJECT_RUNBOOK

最后更新：2026-04-02

## 1. 项目定位

- 项目名：`pinn-flow-visual-demo-v4`
- 路径：`/root/dev/pinn-flow-visual-demo-v4`
- 类型：独立静态前端站点
- 用途：作为 `pinn-flow-visual-demo` 的独立工作副本，承接 `pinn_v3 / pinn_v4` 新几何与新参数后的前端改造
- 当前阶段：已完成阶段 1 ~ 4，当前前端语义、页面叙事与发布路径均已收口到 v4；并已打通首版 remote 推理链路

## 2. 运行前提

- Node.js：建议使用当前环境自带版本（已验证 `v22.x`）
- 包管理器：`npm`
- 默认不依赖数据库或后端服务

## 3. 环境变量

复制 `.env.example` 为 `.env` 后使用：

```bash
VITE_INFERENCE_MODE=demo
VITE_API_BASE_URL=http://127.0.0.1:8000/api/pinn
VITE_REQUEST_TIMEOUT_MS=15000
VITE_MAX_RETRIES=2
VITE_LOCAL_PREVIEW_CACHE_TTL_MS=1800000
```

说明：

- `demo`：本地演示数值层
- `remote`：远程真实推理接口
- `VITE_MAX_RETRIES`：限制失败重试次数，避免无限消耗
- `VITE_LOCAL_PREVIEW_CACHE_TTL_MS`：前端本地首屏 preview 缓存 TTL

## 4. 本地开发

```bash
cd /root/dev/pinn-flow-visual-demo-v4
npm install
npm run dev
```

默认端口：

- `4176`
- 如需与原项目并行运行，建议使用 `npm run dev -- --port 4177`

## 5. 低冲击校验流程

推荐顺序：

```bash
cd /root/dev/pinn-flow-visual-demo-v4
npm run check
npm run test
npm run build
```

说明：

- 先做类型检查
- 再跑轻量单元测试
- 最后执行正式构建

## 6. 当前界面重点

当前阶段重点：

1. 主几何已从旧 `T/Y` 迁移为 **收缩流道 / 弯曲流道**
2. 左侧参数面板已按 `pinn_v4` 当前主线切换为：
   - 收缩：`W / β / Lin/W / Lc/W / Lout/W`
   - 弯曲：`W / Rc/W / θ / inlet profile / Lin/W / Lout/W`
3. 已新增 `C-base / C-test-2 / B-base / B-test-1-blunted` 预设入口
4. demo 数值层已对齐新几何：
   - 收缩突出喉部加速与压降变化
   - 弯曲突出曲率影响与入口剖面差异
5. 画布、流线、采样点与重建链路均已切到新几何体系
6. 页面已收口为更稳定的工作台布局：
   - 顶部：热力图/流线主舞台
   - 热力图主卡顶部：直接显示核心指标
   - 顶部工具条：直接放置主线案例下拉菜单，并与操作按钮对齐
   - 下部：流道参数 / 流体与采样 / 点位查询三卡横向并排
7. 已删除执行状态卡片与一批说明性卡片，优先保证不重叠、不挤压
8. 已继续删除：
   - 轴向中心线 / 喉后代表线卡片
   - 独立核心指标卡片
   - 独立主线案例卡片
   - 辅助分析卡片
9. 顶部预设下拉的自定义项已拆为：
   - `自定义收缩流道`
   - `自定义弯曲流道`
   - 并由下拉菜单直接决定当前流道类型，不再保留参数卡内手动切换按钮
10. `bend_2d` 画布方向已旋转为更适合横向展示的 landscape 朝向
11. 已新增 Python API 服务承接 remote 模式：
   - `contraction_2d`：接 `pinn_v4` 真实 checkpoint
   - `bend_2d`：已切到真实 bend checkpoint 路径
     - `parabolic / skewed_top / skewed_bottom` → `pinn_v3` supervised geometry run `bend_supervised_geometry_v1_20260331`
     - `blunted` → `pinn_v3` independent geometry run `bend_independent_blunted_geometry_notemplate_medium_v1_20260401`
12. remote API 已增加内存响应缓存：
   - 目标接口：`simulate / reconstruct / sweep`
   - 默认 TTL：`1800s`
   - 默认最大缓存条数：`8`
   - 用于减少刷新或重复切换到同一工况时的重复计算
13. 首屏链路已进一步拆分为“先轻后全”：
   - 首屏先读浏览器本地 preview 缓存
   - 未命中时，只请求主热力图 + 指标所需 preview 分辨率
   - `streamlines`：切到流线图层时再单独请求
   - `reconstruct`：稀疏点与重建继续独立请求，不再占用首屏
14. 当前“辅助分析”不再作为主界面功能保留：
   - 原页面中的黏度校准 / 参数扫掠并非 `pinn_v3 / pinn_v4` 主线原生在线能力
   - 当前已从主界面删除，避免与真实模型能力混淆

## 7. 发布与回滚

### 7.1 发布

当前已采用**独立路径发布**，避免覆盖原线上目录 `pinn-flow-visual-demo`。

本次 v4 发布目录与路径：

```bash
cd /root/dev/pinn-flow-visual-demo-v4
npm run build
rsync -a --delete dist/ /var/www/pinn-flow-visual-demo-v4/
```

当前访问路径：

- `https://aqsk.top/pinn-flow-visual-demo-v4/`
- `https://aqsk.top/api/pinn-v4/healthz`

Nginx 说明：

- 已在 `/etc/nginx/sites-available/myweb` 中新增 `/pinn-flow-visual-demo-v4/` 路由
- 已新增 `/api/pinn-v4/` 反代到 `127.0.0.1:8011`
- 已执行 `nginx -t` 与 `systemctl reload nginx`
- 旧入口 `https://aqsk.top/pinn-flow-visual-demo/` 继续保留维护页，不直接切换

### 7.1A API 服务

当前服务：

```bash
systemctl status pinn-flow-visual-demo-v4-api.service
curl https://aqsk.top/api/pinn-v4/healthz
```

实现文件：

- `/root/dev/pinn-flow-visual-demo-v4/server/pinn_v4_api.py`

监听端口：

- `127.0.0.1:8011`

当前按需接口：

- `POST /simulate`
- `POST /streamlines`
- `POST /probes`
- `POST /reconstruct`

### 7.2 回滚

```bash
rm -rf /var/www/pinn-flow-visual-demo-v4
cp -a /var/www/pinn-flow-visual-demo-v4.bak.<timestamp> /var/www/pinn-flow-visual-demo-v4
```

## 8. 真实模型接入建议

未来训练与前端对接收口后：

1. 保持 `ScenarioInput / ScenarioResult` 契约不变
2. 后端提供 `/simulate /query-point /reconstruct /calibrate-viscosity /sweep`
3. 切换 `.env` 为 `VITE_INFERENCE_MODE=remote`
4. 保持有限重试与超时，避免答辩现场无限等待或无限请求

## 9. 已知限制

- 当前站点仍以“演示站”定位为主，不应直接当作正式实验报告数值结论
- 当前未启用登录态与后端持久化
- 当前 remote 仍是**部分真实接入**：
  - contraction：真实 v4 checkpoint
  - bend：真实 bend checkpoint 已接入，但来源暂为 `pinn_v3` 已验证 run，而非 `pinn_v4` 同级 run
  - `skewed_top / skewed_bottom`：当前复用 parabolic bend 真实模型路径，尚不是专门训练的独立 checkpoint
- 当前重点转为继续优化热力图显示与后续替换为同级 v4 bend checkpoint（如后续产出）
- 首屏缓存属于浏览器本地缓存；当参数变化或缓存 TTL 过期时，会重新请求 preview 结果
