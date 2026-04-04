# DATASET_SPEC

## 1. 目标

统一弯曲流道与收缩流道的数据目录、字段和命名规范，确保 CFD、监督训练、PINN 微调、评估与可视化共用同一套数据契约。

## 2. 目录约定

```text
cases/
├─ contraction_2d/
│  ├─ geometry/
│  ├─ cfd/
│  │  └─ <case_id>/
│  ├─ data/
│  │  └─ <case_id>/
│  │     ├─ geometry.json
│  │     ├─ field_dense.csv
│  │     ├─ boundary_points.csv
│  │     ├─ obs_sparse_1pct.csv
│  │     ├─ obs_sparse_5pct.csv
│  │     ├─ obs_sparse_10pct.csv
│  │     ├─ obs_sparse_15pct.csv
│  │     ├─ obs_uniform_1pct.csv
│  │     ├─ obs_uniform_5pct.csv
│  │     ├─ obs_uniform_10pct.csv
│  │     ├─ obs_uniform_15pct.csv
│  │     ├─ obs_sparse_5pct_noise_1pct.csv
│  │     ├─ obs_sparse_5pct_noise_3pct.csv
│  │     ├─ obs_sparse_5pct_noise_5pct.csv
│  │     └─ meta.json
│  └─ figures/
└─ bend_2d/
   └─ ...
```

## 3. 稠密场字段

`field_dense.csv` 推荐字段：

```text
sample_id,case_id,family,x_star,y_star,u_star,v_star,p_star,speed_star,
region_id,wall_distance_star,is_boundary,boundary_type
```

字段说明：

- `sample_id`：点编号
- `case_id`：如 `C-base`, `B-val`
- `family`：`contraction` 或 `bend`
- `x_star,y_star`：无量纲坐标
- `u_star,v_star,p_star`：无量纲场变量
- `speed_star`：无量纲速度模值
- `region_id`：特征区域编号
- `wall_distance_star`：无量纲壁面距离
- `is_boundary`：是否边界点
- `boundary_type`：`interior/inlet/wall/outlet`

## 4. 边界点字段

`boundary_points.csv` 推荐字段：

```text
sample_id,case_id,boundary_type,x_star,y_star,u_bc,v_bc,p_bc,normal_x,normal_y
```

说明：

- `u_bc,v_bc,p_bc`：目标边界值，若不适用可置空
- `normal_x,normal_y`：外法向，可用于后续 Neumann / traction 扩展

## 5. 稀疏观测点字段

`obs_sparse_*.csv` 推荐字段：

```text
sample_id,case_id,family,sampling_tag,noise_tag,
x_star,y_star,u_obs,v_obs,p_obs,
region_id,wall_distance_star
```

说明：

- `sampling_tag`：如 `uniform_5pct`、`region_aware_5pct`
- `noise_tag`：如 `clean`、`noise_1pct`
- `u_obs,v_obs,p_obs`：观测值，可含噪声

## 6. meta.json 建议字段

```json
{
  "case_id": "C-base",
  "family": "contraction",
  "geometry": {
    "W_um": 200,
    "beta": 0.70,
    "L_in_over_W": 4,
    "L_c_over_W": 4,
    "L_out_over_W": 8
  },
  "fluid": {
    "rho": 997.05,
    "mu": 8.902e-4,
    "nu": 8.928e-7
  },
  "flow": {
    "u_mean_mm_s": 0.1,
    "Re": 2.24e-2
  },
  "mesh": {
    "mesh_level": "medium",
    "h_over_W": 0.0333
  }
}
```

## 7. 区域编号建议

### contraction
- `0`：主体区
- `1`：喉部高梯度区
- `2`：近壁区

### bend
- `0`：主体区
- `1`：弯道核心区
- `2`：近壁区

## 8. 数据生成顺序

1. 生成 geometry 与边界标签
2. 生成 CFD 原始解
3. 重采样得到 `field_dense.csv`
4. 抽取边界点得到 `boundary_points.csv`
5. 依据规则生成 `obs_sparse_*.csv`
6. 写出 `meta.json`

## 9. 约束

- 所有导出字段统一使用无量纲量
- 所有 case 必须有 `meta.json`
- 稀疏观测必须保留 `sampling_tag` 与 `noise_tag`
- 不允许只保存图片而不保存原始数值数据

## 10. 当前阶段说明

- `scripts/generate_contraction_case.py` 当前已支持两种 dense field 来源：`synthetic_streamfunction_smoke` 与 `freefem_stokes_cfd`。
- 其中 `synthetic_streamfunction_smoke` 仅用于验证几何、字段、分层采样和目录链路。
- `freefem_stokes_cfd` 已可为 `contraction_2d/C-base` 生成首份正式 FreeFEM++ CFD 数据骨架。
- 正式论文实验必须用 FreeFEM++ 或其他 CFD 真值替换 `field_dense.csv` 与对应观测集。
- `cases/contraction_2d/cfd/freefem_contraction_template.edp` 仅为正式 CFD 接入占位模板。
