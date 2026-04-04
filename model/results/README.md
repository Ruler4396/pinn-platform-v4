# model/results

这个目录保留给整合仓的本地模型结果使用。

- 默认不纳入 Git
- 可放置 checkpoint、metrics、history、导图和论文资产
- 当前 API 会优先从这里读取 `pinn` / `supervised` 结果

如果后续需要把少量轻量结果纳入版本库，建议只保留：

- `config.json`
- `metrics.json`
- `history.csv`
- 汇总 `csv/json`

不建议直接把大量 `best.ckpt`、评估图片或大批运行产物推到主仓库
