# INTEGRATION_AND_DEPLOYMENT

## 项目定位

`pinn-platform-v4` 是毕设正式版单仓库，整合了：

- `web/`：前端展示与交互
- `api/`：网页调用的统一推理接口
- `model/`：PINN 训练、评估、案例与论文素材工作区

外网正式入口：

- 页面：`https://aqsk.top/pinn-flow-visual-demo-v4/`
- API：`https://aqsk.top/api/pinn-v4/`

## 当前关键运行口径

### contraction 主线

当前正式使用的收缩流道 run：

- `contraction_independent_geometry_notemplate_stagepde_mainline_v4`

API 会从下列位置读取该 run：

- `model/results/pinn/contraction_independent_geometry_notemplate_stagepde_mainline_v4/`

最关键文件通常包括：

- `best.ckpt`
- `config.json`
- `history.csv`
- `metrics.json`

这些文件默认只保留在本地，不纳入 Git。

### bend 口径

历史 bend run 名称仍保留为：

- `bend_supervised_geometry_v1_20260331`
- `bend_independent_blunted_geometry_notemplate_medium_v1_20260401`

但当前整合仓默认不要求本地必须存在这两个 checkpoint。

如果缺失对应 `best.ckpt`，API 会自动退回到 `SyntheticBendRuntime`：

- `synthetic_bend_parabolic_fallback`
- `synthetic_bend_blunted_fallback`

这样做的目的不是替代正式训练结果，而是保证：

- 网站可以稳定上线
- bend 页面可以正常演示
- 删除历史大权重后服务仍可重启

如果后续重新补齐 bend checkpoint，API 会优先加载真实模型，不再使用 fallback。

## 这次整合的关键改动

- 把原 `pinn-flow-visual-demo-v4` 前端整理到 `web/`
- 把原 `pinn_v4` 模型工作区整理到 `model/`
- 把原演示 API 整理到 `api/pinn_platform_api.py`
- 移除了前端内重复的 `server/` 副本，统一以后端 `api/` 为准
- 把 API 对 `pinn_v4` / `pinn_v3` 的硬编码绝对路径改成整合仓相对路径优先
- 为 bend 模型增加 checkpoint 缺失时的合成场 fallback
- 把关键模型脚本改成整合仓相对路径运行

## 当前上线方式

### 前端

- 构建目录：`web/dist`
- Nginx 发布路径：`/var/www/pinn-flow-visual-demo-v4`
- 页面访问前缀：`/pinn-flow-visual-demo-v4/`

### API

当前 systemd 服务：

- `pinn-flow-visual-demo-v4-api.service`

当前启动入口：

- `/root/dev/pinn-platform-v4/api/pinn_platform_api.py`

显式模型根目录环境变量：

- `PINN_PLATFORM_MODEL_ROOT=/root/dev/pinn-platform-v4/model`

## 已完成验证

本地验证：

- `npm run check`
- `npm run test`
- `npm run build`
- `python3 -m py_compile api/pinn_platform_api.py`
- `python3 -m py_compile model/scripts/prepare_chapter5_assets.py`
- `bash -n model/scripts/run_contraction_independent_mainline_lowimpact.sh`
- 本地 `/simulate` 的 contraction 与 bend 请求均可返回结果

外网验证：

- `GET /api/pinn-v4/healthz` 返回 200
- `POST /api/pinn-v4/simulate` 的 contraction 请求返回 200
- `POST /api/pinn-v4/simulate` 的 bend 请求返回 200
- `GET /pinn-flow-visual-demo-v4/` 返回 200

## 后续建议

- 若要恢复正式 bend 模型效果，补回对应 checkpoint 到 `model/results/`
- 若要把结果长期归档，优先考虑 Releases、LFS 或独立结果存储，而不是直接进主仓库
- 后续可继续清理 `web/docs/` 与 `model/docs/` 中残留的旧项目名和旧绝对路径表述
