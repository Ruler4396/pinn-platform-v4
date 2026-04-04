# C-base CFD artifacts

本目录已生成 `C-base` 的首份 FreeFEM++ CFD 数据：

- `C-base_stokes.edp`：自动生成的收缩流道稳态 Stokes 求解脚本
- `C-base_raw.csv`：FreeFEM++ 原始导出结果（字段：`x_star,y_star,u_star,v_star,p_star,bc_tag`）

说明：

- 当前解是 **dimensionless steady Stokes baseline**
- 其后处理结果已同步写入 `../../data/C-base/field_dense.csv`
- 后续若要提升精度，可继续细化网格、补充网格无关性实验，并扩展到更多 case
