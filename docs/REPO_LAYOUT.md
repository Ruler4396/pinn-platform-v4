# REPO_LAYOUT

## 背景

原工作方式里：

- `pinn-flow-visual-demo-v4` 负责前端站点与演示 API
- `pinn_v4` 负责模型训练、评估、导图与数据

这种方式在本地开发时还能接受，但对 GitHub 单仓库提交不友好，因为：

- 网站和模型分散在两个仓库语义里
- API 代码通过绝对路径依赖外部目录
- 发布、运行、说明文档都被拆散了

## 整合后的推荐结构

- `web/`
  - 面向用户的网页界面
  - 负责参数编辑、结果展示、前端缓存与远程请求
- `api/`
  - 面向前端的统一推理接口层
  - 负责场景参数解析、模型加载、输出格式整理
- `model/`
  - 面向训练与论文资产的模型工作区
  - 负责 case、脚本、训练、评估、导图和数据约定

## 为什么不把所有文件直接扁平堆在根目录

- Node / Python / 训练数据 / 结果文件的边界会变得很乱
- `.gitignore` 难以维护
- 以后做 CI、部署或分层说明时会更困难

## 当前保留与默认忽略

- 保留到整合仓：
  - `web/src`
  - `web/docs`
  - `api/pinn_platform_api.py`
  - `model/src`
  - `model/scripts`
  - `model/cases`
  - `model/docs`
- 默认不入 Git：
  - `web/node_modules`
  - `web/dist`
  - `web/backups`
  - `model/results`

## API 路径策略

API 已改为优先读取整合仓内部的 `model/`。

当前优先级：

1. 主模型代码根目录：`PROJECT_ROOT/model`
2. 若存在特殊 bend 历史结果，可通过环境变量覆盖

推荐后续继续整理时，把真正需要长期保留的 checkpoint 清点后，再决定是否单独放置到：

- 本仓本地 `model/results/`
- 外部私有结果目录
- GitHub Releases / LFS / 其他对象存储
