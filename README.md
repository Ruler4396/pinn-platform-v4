# pinn-platform-v4

`pinn-platform-v4` 是把原先分开的 `pinn-flow-visual-demo-v4` 与 `pinn_v4` 重新整理后的单仓库工作区。

目标不是再维护“两套平行项目”，而是把毕设所需的三个层面放进同一个项目里：

- `web/`：前端演示站
- `api/`：连接前端与模型推理的 Python API
- `model/`：PINN 训练、评估、导图与数据工作区

## 当前结构

```text
pinn-platform-v4/
├─ README.md
├─ .gitignore
├─ docs/
├─ web/
├─ api/
└─ model/
```

## 目录说明

- `web/`
  - 来自 `pinn-flow-visual-demo-v4`
  - 保留前端源码、Vite 配置、测试与前端文档
  - 已去掉本仓内重复的 `server/` 副本，统一以后端 `api/` 为准
- `api/`
  - 当前放置统一 API 入口：`api/pinn_platform_api.py`
  - 负责把网页参数转换成模型输入，并读取 `model/` 工作区中的代码与结果
- `model/`
  - 来自 `pinn_v4`
  - 保留 `src/`、`scripts/`、`cases/`、`docs/`
  - `results/` 默认不纳入 Git，用于本地 checkpoint 和评估产物

## 当前整理原则

- 先做“最小可回滚”的单仓库整理，不破坏原项目目录。
- 原 `pinn-flow-visual-demo-v4` 与 `pinn_v4` 继续保留，便于对照与回退。
- 新仓库优先解决结构统一、路径统一、Git 边界统一。

## 运行入口

### 前端

```bash
cd /root/dev/pinn-platform-v4/web
npm install
npm run dev
```

### API

```bash
cd /root/dev/pinn-platform-v4
python3 api/pinn_platform_api.py --host 127.0.0.1 --port 8011
```

### 模型训练 / 评估

```bash
cd /root/dev/pinn-platform-v4/model
bash scripts/run_contraction_independent_mainline_lowimpact.sh
```

## 说明

- `web/docs/` 与 `model/docs/` 里保留了原子项目文档，因此部分历史描述仍会提到旧目录名。
- 整合仓的总体说明见 `docs/REPO_LAYOUT.md`。
- 关键 run、整合改动与当前上线信息见 `docs/INTEGRATION_AND_DEPLOYMENT.md`。
