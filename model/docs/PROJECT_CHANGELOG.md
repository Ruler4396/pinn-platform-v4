# PROJECT_CHANGELOG（pinn_v4）

## 2026-04-01
- 新建 `pinn_v4`，用于承接 `pinn_v3` 的最新主线。
- 已迁移当前双模型主线训练、评估、导图、数据生成所需脚本与数据。
- 已将当前正式主线固定为“独立速度模型 + 独立压力模型 + 阶段内部分控制方程 + 最终控制方程耦合”。
- 已重写 `scripts/export_field_maps.py`：支持域外留白、相对误差百分比显示、中文字体显式指定、按任意 predictions 目录导图。
- 已将 `scripts/run_contraction_independent_mainline_lowimpact.sh` 切换到 `pinn_v4` 路径，并默认启用阶段内部分控制方程参数。

## 2026-04-02
- 新增 `src/utils/plotting.py`，统一通过显式中文字体文件配置 Matplotlib，避免论文导图中文乱码。
- 更新 `scripts/export_field_maps.py`，改为复用统一中文字体配置工具，保证速度/压力场图中的中文标题稳定显示。
- 继续优化论文场图版式：`scripts/export_field_maps.py` 新增纵向堆叠布局与横向色标选项，缓解细长流道在横向拼图时“主图过窄、Y 轴刻度拥挤”的问题。
- 继续优化论文场图平滑性：`scripts/export_field_maps.py` 改为基于规则渲染网格插值并叠加几何边界线，降低收缩/弯曲过渡段边界锯齿感。
- 继续优化收缩/弯曲壁面锯齿问题：规则网格插值后对几何内缺失值使用最近邻补点，并改为用连续几何路径做裁剪，减少边界内侧出现白色锯齿缺口。
- 新增 `scripts/prepare_chapter5_assets.py`，可按小节批量重导出并整理第 5 章论文图片。
- 重整 `results/thesis_assets/chapter5/`：按 `5_1`～`5_7` 小节分目录统一整理第 5 章图片，并生成 `README.md` 与 `manifest.json` 便于后续直接取图。
- 更新 `docs/THESIS_ASSET_GUIDE.md`，将论文图片命名规则改为“小节目录 + 语义化文件名”。
- 修复 `ContractionGeometry` 缺少 `contains()` 导致 `scripts/export_field_maps.py` 无法对收缩流道结果执行几何掩码导图的问题。
- 修复 `scripts/prepare_chapter5_assets.py` 中 `plot_noise_summary()` 仍将速度误差图例硬编码在右上角的问题；现已改为左上角，避免遮挡右侧柱顶数值与保持 `dense_vs_noise_summary.png` 版式一致。
