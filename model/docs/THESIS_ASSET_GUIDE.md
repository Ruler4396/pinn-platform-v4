# THESIS_ASSET_GUIDE（pinn_v4）

## 1. 目的
为论文写作阶段提供统一、稳定、便于检索的插图目录，避免继续直接从 `results/field_map_checks/chapter5_materials/` 中手工挑图。

## 2. 目录约定
- 论文正式插图目录：`results/thesis_assets/chapter5/`
- 原始导图与中间材料：`results/field_map_checks/chapter5_materials/`

## 3. 命名约定
- 第 5 章图片先按小节分目录，例如：
  - `results/thesis_assets/chapter5/5_1_baseline_convergence/`
  - `results/thesis_assets/chapter5/5_4_region_vs_uniform/`
- 每张图片使用语义化文件名，优先表达：
  - 工况或数据族；
  - 对比类型；
  - 图像用途（`speed_map` / `pressure_map` / `combined_map` / `summary` / `convergence`）。
- 后续新增第 5 章图片继续沿用同一规则，避免回到零散、难检索的命名方式。

## 4. 中文字体约定
- 所有论文正式图片均应显式指定中文字体文件，避免中文乱码。
- 当前统一通过以下工具完成：
  - 字体配置：`src/utils/plotting.py`
  - 场图导出：`scripts/export_field_maps.py`
  - 第 5 章图片整理：`scripts/prepare_chapter5_assets.py`

## 5. 版式约定
- 对于细长流道的速度场、压力场和误差图，优先使用**纵向堆叠**版式，而不是横向并排版式。
- 色标优先使用**横向色标**，放在各子图下方，避免进一步挤压主图宽度。
- 坐标轴刻度数量应适度控制，避免 Y 轴刻度标签相互挤压。
- 渲染时优先使用规则网格插值并叠加几何边界线，减少流道收缩/弯曲过渡位置的锯齿感。
- 柱状对比图的图例应优先放在**不遮挡柱顶数值标注**的位置；若右上角存在数值标签，优先改放左上角或子图外侧。

## 6. 使用方式
1. 先在 `results/field_map_checks/chapter5_materials/` 中准备指标文件、预测结果或候选图片来源。
2. 运行：

```bash
python3 scripts/prepare_chapter5_assets.py --max-retries 1
```

3. 到 `results/thesis_assets/chapter5/` 中按小节取图。
4. 若正文中需要新增图片，优先补充到对应小节目录，而不是继续散落到临时目录中。

## 7. 当前已整理内容
- `results/thesis_assets/chapter5/README.md`
- `results/thesis_assets/chapter5/manifest.json`
- 各小节子目录下的正式 PNG 图片
