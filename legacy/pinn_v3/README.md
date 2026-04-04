# legacy/pinn_v3

这个目录预留给整合仓的历史 `pinn_v3` 兼容资源。

当前用途主要是两类：

- 少量仍依赖 `pinn_v3` 历史结果的 bend checkpoint
- 论文整理或对照分析时仍需要引用的旧 run 目录

默认不要求把整个 `pinn_v3` 复制进整合仓。

如果后续确实需要兼容历史 bend run，可通过以下环境变量把 API 或脚本指向外部旧目录：

- `PINN_PLATFORM_BEND_WORKSPACE_ROOT`
- `PINN_PLATFORM_V3_ROOT`
