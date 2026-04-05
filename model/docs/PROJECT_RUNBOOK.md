# PROJECT_RUNBOOK（model / pinn_v4）

## 项目定位

- 路径：`./model`
- 用途：承接 `pinn_v3` 当前最新主线，作为整合仓中的训练、评估、导图与论文收口工作区。
- 保留原则：只保留“当前继续实验必须用到”的脚本、数据与文档。

## 当前正式主线

当前主线统一为：

- 独立速度模型
- 独立压力模型
- 速度阶段内部启用连续性项
- 压力阶段内部启用动量项
- 最终控制方程交替耦合

对应训练脚本：

- `scripts/train_velocity_pressure_independent.py`

对应评估脚本：

- `scripts/evaluate_velocity_pressure_independent.py`

对应导图脚本：

- `scripts/export_field_maps.py`
- `scripts/prepare_chapter5_assets.py`

## 数据

已从 `pinn_v3` 迁移：

- `cases/bend_2d/`
- `cases/contraction_2d/`

其中保留：

- `data/`
- `cfd/`
- `geometry/`
- `figures/`
- `README.md`

## 运行约束

- 默认单线程：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`
- 默认低优先级：`nice -n 10`
- 默认低冲击：先评估、再训练；先小规模、再正式版
- 所有训练和导图脚本都必须设置 `--max-retries 1` 或其他显式上限

## 绘图约束

当前导图脚本已修正为：

1. 域外区域留白，不着色；
2. 相对误差按百分比显示；
3. 中文标题强制使用中文字体；
4. 可直接读取任意 predictions 目录，不强绑当前工作区结果目录；
5. 支持细长流道论文图使用纵向堆叠布局与横向色标，改善主图可读性。

当前论文图片整理脚本：

- 会按第 5 章小节统一输出到 `results/thesis_assets/chapter5/`
- 会显式指定中文字体文件，避免导出的 PNG 出现中文乱码
- 推荐命令：

```bash
python3 scripts/prepare_chapter5_assets.py --max-retries 1
```

## 推荐入口

### 收缩流道低冲击正式版

```bash
bash scripts/run_contraction_independent_mainline_lowimpact.sh
```

### 代表性导图

```bash
python3 scripts/export_field_maps.py \
  --predictions-dir historical/pinn_v3/results/pinn/bend_targetrecon_independent_btest1_sparse5_nobc_hardwall_stagepde_v1_20260401/evaluations/predictions_target \
  --cases B-test-1__ip_blunted \
  --output-dir results/field_map_checks/bend_target_stagepde_relpct \
  --error-mode rel \
  --relative-floor-ratio 0.01 \
  --max-retries 1
```
