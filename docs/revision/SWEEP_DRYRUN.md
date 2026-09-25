# SWEEP_DRYRUN — T5/T6 扫描脚本的 dry-run 证据与自测记录

- 日期：2026-09-24（北京时间）。作者：方案架构线。
- 本轮**没有跑任何训练、没有上实例**；实例上的一切执行由用户统一投递。
- 新增 4 个文件（均未改动仓库任何既有文件；`git status` 只有这 4 个 `??` 条目）：

| 文件 | 行 | 职责 |
|---|---|---|
| `model/scripts/generate_observations_seeded.py` | 302 | obs_seed 入口：只读 `field_dense.csv` 重抽点位，产出 `obs_*__s{k}.csv`；含 `--verify-committed` 复现校验与两臂预算断言 |
| `model/scripts/sweep_lib.sh` | 433 | 公共函数：`date +%s%N` 计时、progress.jsonl 记账、跳已有、段预算提前退出、段末对账、递归删除自检 |
| `model/scripts/sweep_t5.sh` | 240 | T5 的 19 格矩阵 + T6 的 8 配对单元；argv 逐项照抄原启动器 |
| `model/scripts/analyze_sweep.py` | 325 | mean±std（两口径）、配对 Wilcoxon + t、最小可达 p、自动判"不可判" |

## 0. 环境事实的落地方式（T0 交给我的四条，逐条对应到代码）

| T0 结论 | 代码里的处置 |
|---|---|
| 无 `/usr/bin/time`、无 `bc` | 计时只用 `date +%s%N` 的整数运算（`sweep_lib.sh:_now_ns/elapsed_ms`）；`%N` 不被支持时回落到秒并打印说明。全文没有任何 `bc` 调用 |
| 镜像 numpy 1.26.4 会在 `train_supervised.py:588` 的 `np.trapezoid` 上崩 | `preflight` 硬校验 `numpy.__version__` 以 `2.` 开头，否则 `exit 1` 并把原因写在消息里（含"必须 pip3 install numpy==2.2.6"） |
| 墙钟单价 train 55.145 s / eval 1.48–1.59 s | 已把 `UNIT_SPARSE_TRAIN_SEC=59.52`（仓库落盘值）与 `UNIT_EVAL_SEC=1.55` 设为可覆盖的默认值；**注意 55.145 s 是实例读数、59.52 s 是原主机读数**，段预算用保守的那个 |
| seed=43 使 speed 从 0.0321→0.0207（−35.4%），大于表5-5 的两臂差 24% | 这条不再需要写进脚本——它已经是 T5/T6 存在的理由；`analyze_sweep.py` 的"不可判"判词把这类情况自动判成不可判 |

## 1. 怎么投递

```bash
cd /mnt/workspace/pinn-repro-2026           # 你的实例工作根目录，其下有 model/
export SWEEP_OUT_DIR="$PWD/out"
export BEND_UNIT_SEC=<标定值>                # 未设时脚本按 78s 估并打印 ESTIMATED=1
python3 model/scripts/generate_observations_seeded.py --family contraction_2d \
        --obs-seeds 0 --verify-committed     # 先证明抽样语义没被改动（必须全 IDENTICAL）
nohup bash model/scripts/sweep_t5.sh --seg 01 --budget-min 90 > out/seg01.out 2>&1 &
# 续跑/下一段：同段名重投即跳过已有 metrics.json；换段名 --seg 02 继续
python3 model/scripts/analyze_sweep.py --split test --metric rel_l2_speed --paired t5c04,t5c08 \
        --out model/docs/revision/T5_T6_stats.json
```

计划规模（dry-run 实测打印）：**实际要训练 107 个单元，串行 ≈124 min，6 路并行 ≈31 min，约 2 个 90 min 段**。
107 = T5 的 19 格 × 5 个训练种子（95）+ T6 的 12 个新单元（`obs_seed>0`），另铺排 4 行与 T5 同名、运行时走 `[skip-train]`。

## 2. argv 等价性核对（要求：与仓库既有 strict 启动器逐项一致）

### 2.1 核对方法

不是"看脚本觉得合理"，而是**从启动器源码反解**：解析 `run_strict_sparse_experiments.sh` 里 `run_dual` 的形参顺序（`:17-28`，注意它用的是 `local x="$1"; shift` 连环写法，所以**位置由源码顺序决定、不是按数字排**）、`wall_mode` 派生规则（`:29-32`）、python3 参数模板（`:38-76`），再把 14 条 `run_dual` 调用行的实参代入（`${CONTRA_TRAIN}`/`${BEND_BLUNT_TRAIN}` 展开），得到"原启动器会为那一格发出的 argv"，与 dry-run 打印的 argv 逐项比。稠密格另用 `run_contraction_independent_mainline_lowimpact.sh` 的 `COMMON_ARGS` 比。
脚本：`.scratch/verify_argv2.py`。

### 2.2 结果

14/14 格 **一致**，白名单外的差异为 0：

```
格3..格10（收缩 分层/均匀 1/5/10/15%）    一致
格13(basic)  格15..格18（弯道 分层 1/5/10/15%）  一致
格1(稠密主线 vs lowimpact 启动器)          一致
项数：原=37 新=38 | 新增=[--seed] | 缺失=无
唯一值差异：--run-name（白名单内）
```

允许的差异恰好三处：① 新增 `--seed`；② `--run-name` 的值；③ `obs_seed>0` 时观测源名多一个 `__s{k}` 后缀。

### 2.3 三条代表性 argv（完整原文）

格4（收缩·分层5%·strict 口径，对标 `contraction_strict_geometry_sparse5_stagepde_20260503`）：
```
python3 scripts/train_velocity_pressure_independent_strict_sparse.py --family contraction_2d --run-name rev2609_t5c04__s42__o0 --seed 42 --train-cases C-base,C-train-1,C-train-2,C-train-3,C-train-4,C-train-5 --val-cases C-val --feature-mode geometry --drop-features inlet_profile_star --train-velocity-source obs_sparse_5pct --val-velocity-source obs_sparse_5pct --train-pressure-source obs_sparse_5pct --val-pressure-source obs_sparse_5pct --velocity-hidden-layers 128,128,128 --pressure-hidden-layers 128,128,128 --activation silu --velocity-epochs 200 --pressure-epochs 200 --coupling-epochs 80 --velocity-lr 6e-4 --pressure-lr 6e-4 --coupling-velocity-lr 1e-4 --coupling-pressure-lr 1e-4 --wall-weight 0.0 --inlet-flux-weight 0.0 --continuity-weight 0.1 --velocity-stage-continuity-weight 0.3 --velocity-stage-momentum-weight 0.0 --outlet-pressure-weight 0.0 --pressure-drop-weight 0.0 --pressure-stage-momentum-weight 0.5 --velocity-wall-mode hard --hard-wall-sharpness 12 --coupling-momentum-weight 10.0 --coupling-continuity-weight 0.1 --coupling-velocity-supervision-weight 1.0 --coupling-pressure-supervision-weight 1.0 --max-physics-points 512 --print-every 40 --strict-sparse-scalers --max-retries 1
```
格1（收缩·稠密，走 `train_velocity_pressure_independent.py` + 表4-4 权重档；与 strict 档的区别只有 `inlet-flux 0.5 / outlet-pressure 1e-4 / pressure-drop 1.0` 和 `--print-every 20`、无 `--strict-sparse-scalers`）：
```
python3 scripts/train_velocity_pressure_independent.py --family contraction_2d --run-name rev2609_t5c01__s42__o0 --seed 42 --train-cases C-base,C-train-1,C-train-2,C-train-3,C-train-4,C-train-5 --val-cases C-val --feature-mode geometry --drop-features inlet_profile_star --train-velocity-source dense --val-velocity-source dense --train-pressure-source dense --val-pressure-source dense --velocity-hidden-layers 128,128,128 --pressure-hidden-layers 128,128,128 --activation silu --velocity-epochs 200 --pressure-epochs 200 --coupling-epochs 80 --velocity-lr 6e-4 --pressure-lr 6e-4 --coupling-velocity-lr 1e-4 --coupling-pressure-lr 1e-4 --wall-weight 0.0 --inlet-flux-weight 0.5 --continuity-weight 0.1 --velocity-stage-continuity-weight 0.3 --velocity-stage-momentum-weight 0.0 --outlet-pressure-weight 1e-4 --pressure-drop-weight 1.0 --pressure-stage-momentum-weight 0.5 --velocity-wall-mode hard --hard-wall-sharpness 12 --coupling-momentum-weight 10.0 --coupling-continuity-weight 0.1 --coupling-velocity-supervision-weight 1.0 --coupling-pressure-supervision-weight 1.0 --max-physics-points 512 --print-every 20 --max-retries 1
```
T6 均匀臂（与分层臂**只差观测源名** ⇒ 单变量配对）：
```
python3 scripts/train_velocity_pressure_independent_strict_sparse.py --family contraction_2d --run-name rev2609_t5c08__s42__o0 --seed 42 ... --train-velocity-source obs_uniform_5pct --val-velocity-source obs_uniform_5pct --train-pressure-source obs_uniform_5pct --val-pressure-source obs_uniform_5pct ... --velocity-wall-mode hard ... --strict-sparse-scalers --max-retries 1
```

## 3. 观测种子（两层种子的第二层）

推导规则与 `scripts/generate_contraction_case.py:137-156` **同式**（base = `42 + obs_seed*1000`）：

| 档 | 种子 | obs_seed=0 | obs_seed=1 |
|---|---|---|---|
| 分层 `region_aware` | `base + pct` | 47（5%） | 1047 |
| 均匀 `uniform` | `base + 100 + pct` | 147 | 1147 |
| 含噪基底（5% 分层） | `base + 5` | 47 | 1047 |
| 含噪扰动 | `base + 300 + noise_pct` | 345（3% 档） | 1345 |

已验证（`.scratch/verify_obs_entry.py`，本机可跑，不依赖 pandas）：
- 种子推导断言：**全部一致**
- 命名断言：`obs_seed=0` 用已入库文件名、`>0` 一律带 `__s{k}`；**全部一致**
- 护栏：`--obs-seeds` 含 0 且未加 `--verify-committed/--dry-run` 时直接拒绝执行（不许覆盖已入库点位）

**必须在实例上跑一次的正向校验**：`--obs-seeds 0 --verify-committed` 逐 sample_id 比对已入库 CSV。不通过就说明抽样语义被动过，整套 obs_seed 作废。

## 4. 三组闸门的反向验证（每条闸都要能红）

### 4.1 递归删除自检（`lint_no_recursive_delete`）
| 情形 | 期望 | 实测 |
|---|---|---|
| 两个脚本原文 | 绿 | 绿 |
| 注入一行 `rm -rf "${OUT_DIR}"` 的副本 | 红 | 红（打印命中行号） |
| 只含"绝不 rm -rf"说明文字的注释行 | 绿（不误伤） | 绿 |
匹配前先剥掉整行注释 —— 否则这条规则会逼人删注释来变绿。

### 4.2 段末对账（`seal_segment`）：内存计数器 vs `progress.jsonl`
`.scratch/verify_ledger.sh`，8 条控制 **8/8 符合预期**：
账目吻合→绿；本段 0 行→红；训练成功数对不上→红；失败数对不上→红；账本含无法解析行→红；有失败单元→红；完成数≠`EXPECTED_RUNS`→红；强制损坏开关→红。
关键设计：`grep -c` 在"没匹配"时返回**合法的 0**，所以"本段 0 行"绝不能当通过 —— 这正是这条闸要拦的情形。
另外：`rc=0 但 metrics.json 没落盘` 也记成非零（`book_rc=90`），否则对账会假绿。

### 4.3 "不可判"判词（`analyze_sweep.py:judge`）
用仓库真实评估文件派生合成结果树（`.scratch/verify_analyze_sweep.py`，不跑训练）：
- 两臂差 0.00642 < 组内 std 0.00959 → **`不可判：…落在换一次种子的波动带内，正文不得写谁占优`** ✅
- 把一臂整体压到 1/10 → **`t5c08 在该指标上占优`** ✅
即这条闸两个方向都能变，不是恒绿。

### 4.4 附带修掉的一个隐患：输出编码
两个 py 脚本会打印中文与 `⇒ ± ≥`，在非 UTF-8 控制台上会 `UnicodeEncodeError` 中途崩掉 —— 那表现为"**闸门自己死了**"，比红更难发现。已给两者加 `_make_stdout_utf8()`（切 utf-8 + `errors="replace"`），加之前本机跑统计确实以 rc=1 死在打印上，加之后 rc=0。Linux 实例本来就是 UTF-8，这条不影响实例，但保证换台 Windows 机器复核时不会误判成"统计失败"。

## 5. 两处与需求不同的实现决定（必须先说，否则算我擅自改口径）

1. **"两臂区域配额完全相同"做不到，也不该做。** 分层采样的定义就是按 `sparse_sampling.py:32-37` 的配额（收缩族 1/2/0 = 45/35/20）撒点，均匀采样是按种子随机撒；实测已入库文件：C-val 5% 分层 = 13/28/22，均匀 = 21/18/24。**若两臂配额相同，两臂就是同一策略，比较失效。**
   落地做法：断言改成**总点数相同 + 同一 interior 池 + 同一 obs_seed**（`assert_obs_budget`，两臂都是 63 点，不同就 `exit 1`），并把两臂各自的区域配额作为**读数**打印出来（它正是被试变量）。
2. **格13 的 basic 臂不能"与主线对齐到 hard"。** 硬包络要求特征里有 `wall_distance_frac`（`train_velocity_pressure_independent_strict_sparse.py:171-173` 直接 `raise ValueError`），而 basic 特征集只有 4 列、不含它 —— **这就是原启动器为什么给 basic 开 soft**。
   落地做法：新增格14 = `geometry 特征 + soft`，与格13 只差特征集一项（这才是能归因给编码的单变量对照）；格4（geometry+hard）继续作为主线读数。消融行将报三个格：basic+soft / geometry+soft / geometry+hard，把"特征"与"壁面构造"两个效应分开。

## 6. 本机没验证到的（不猜，留给实例）

1. `analyze_sweep.py --self-test` 与任何 `--paired`：本机**没有 scipy**，只验证到"明确报错、不降级手算"这条行为（`rc=1` + 提示要装支持 numpy 2.x 的 scipy；**具体最低版本我未核**，以实例上 pip 的解析结果为准）。真实 Wilcoxon/t 值必须在实例上出。
2. ~~`generate_observations_seeded.py` 的实际写文件与 `--verify-committed`（本机无 pandas/numpy，只验证了 `--dry-run` 与种子/命名断言）。~~
   **已更新（9/24 第二轮）**：这两条分支现在本机也过（用桩化 pandas，见 §8）。仍然只剩一件事要在实例上做：**真 pandas + 真抽样函数**下跑 `--verify-committed`，即"obs_seed=0 逐位复现已入库点位"这条数值等价断言。
3. 弯曲族单价：`BEND_UNIT_SEC` 未标定，脚本按 78 s 估并打印 `ESTIMATED=1`。标前所有弯曲 ETA 不可信。
4. 新增 run 的产物体积与 push/tar 耗时：实测一个 strict run 目录 = **1.05 MB**（best.ckpt 288 KB + history.csv 173 KB + evaluations 477 KB），107 run ⇒ **≈113 MB** 要回传。`tar` 与 push 吃不吃得住，只能在实例上看。
5. 真实 `progress.jsonl` 与 `segment_NN.tar.gz` 的回传链路（对账逻辑已用合成账本验过，但"真跑一段→tar→push→实例回收→重新 clone 校验"这条闭环只能在实例上过一次）。

## 7. 统计口径的两条硬限制（写进论文也写进这里）

- **主矩阵 n=5 只做 mean±std，不做显著性声称**：配对 Wilcoxon 在 n=5 时最小可达双侧 p = 2/2⁵ = **0.0625 > 0.05**，数学上不可能显著。`analyze_sweep.py` 每次配对都打印 `本组最小可达双侧 p`，n<6 时额外警告。显著性只由 T6 的 n=8（最小可达 p = 0.0078）承担。
- **两套口径都要出、且标明哪个进表**：`mean_of_cases`（逐工况先算再对工况取均值 = 论文表现用口径）与 `pooled`（`evaluations/metrics_*.json#global_metrics`）。二者在 test 上实测差 1.2 倍（0.0390 vs 0.0471），必须分列，否则自动核对闸门会产生假红。

## 8. obs 生成器在实例上崩溃的修复 + 本地不依赖实例的验证法（9/24 第二轮）

### 8.1 崩溃根因（不是环境问题，是我写死的一个不存在的键）

实例上 `--verify-committed` 报 `KeyError: '_n_points'`。查源码：`budget_check()` 读 `job["_n_points"]`，
而**全仓库没有任何一处写这个键**（抽样分支 `job.update(meta)` 写的是 `n_points`）。所以这不是 verify 独有的坑：
真跑一次普通写入（T6 那条路）会在**同一行**崩。我本地当时只测了 `--dry-run`，而 dry-run 在 `budget_check`
之前就 `return`，所以这道闸从未被本地跑到过 —— 漏测原因是"测试用的分支恰好绕过了坏分支"。

### 8.2 修法（断言一条没删，只把读数换成可信来源）

| 位置 | 改动 |
| --- | --- |
| `budget_check()` | `job["_n_points"]` → `job.get("n_points")`；**没有读数的作业直接算一条 problem**（"没有 n_points ⇒ 预算断言没测到东西"），只有一臂读数的组合也算 problem（"预算对齐没法判"）。原来这两条路会静默少一行表格，等于闸门空跑。 |
| 新增 `committed_stats()` | 纯 stdlib（`csv.DictReader`）数出已入库 CSV 的**行数、`region_id` 直方图、`sample_id` 序列**；缺 `sample_id` 列直接 `SystemExit`。 |
| 新增 `committed_path()` | 按 obs_seed=0 的命名规则定位已入库文件，verify 与 budget-only 共用一条路径推导，避免两处命名各写一遍。 |
| `--verify-committed` 分支 | 点数与区域直方图**覆盖为 `committed_stats()` 的读数**，重算值另存 `regen_n_points`。理由：拿我自己重算出来的数去断言"两臂预算相同"是循环论证 —— 抽样语义若真被动过，两臂会一起错、断言照样绿。现在断言量的是"训练实际吃到的那份 CSV"。 |
| 新增 `--budget-only` | 只读已入库 CSV 跑同一条预算断言：**不 import numpy/pandas、不写文件**，因此本地和实例都能跑。这条是"不依赖实例验证"的落点。 |
| `--allow-existing` | 从"报错前先看一眼"改成真跳过（`[skip-obs]`），并且**点数从磁盘上的现存文件读**；否则续跑时另一臂没读数，会触发 §8.2 第 1 行的"缺读数"红灯。 |
| `disp()` | `[obs]`/`[skip-obs]` 打印改用容错的相对路径。桩化自测里暴露出来的：`--write-root` 指到仓库外时 `relative_to()` 抛 `ValueError` 会打断正在写的作业 —— 同类"打印把主流程搞挂"的隐患，顺手补掉。 |
| 空写占位 | 删掉 `out_path.write_text("")` 那行占位写；它在写到一半崩溃时会把已有 CSV 清成 0 字节。 |

verify 汇总现在还额外执行 `problems`（预算不对齐也 `exit 1`），不再只报 verify 标签。

### 8.3 本地怎么验（一条命令，不连实例）

**`python3 model/scripts/selftest_observations.py`** —— 这条自测已随仓库提交，13 例全绿 rc=0。
它做三件事：A) 对 `budget_check` 打 6 条正/负对照（点数相等=绿、不等/缺臂/缺读数/跨 obs_seed 比=红）；
B) 用**桩化 pandas + 受控假抽样**把 §8.1 崩过的两条分支（写入、`--verify-committed`）端到端跑完，
含 3 条故意注入污染的负对照；C) 对两族全部已入库 CSV 跑纯 stdlib 的 `--budget-only`。
写-root 指到系统临时目录，`cases/` 只读不写。换一族工况同样通过：
`--family bend_2d --case B-val__ip_blunted --rate 0.05` ⇒ 同一行结论 rc=0。

配套的三条独立命令（本机实测尾行）：

```bash
cd pinn-platform-v4
python3 model/scripts/selftest_observations.py
# 结论：全部相符 ⇒ 记账与闸门可信（抽样数值等价仍须实例 --verify-committed）      SELFTEST_RC=0
python3 model/scripts/generate_observations_seeded.py --family contraction_2d,bend_2d --budget-only
# [OK] budget-only：60 个 (工况×采样率) 组合的两臂点数全部相同 … 'region': 96, 'uniform': 96   RC=0
python3 model/scripts/generate_observations_seeded.py --family contraction_2d,bend_2d --obs-seeds 0,1,2,3 --dry-run
# [dry-run] 480 个观测文件将被生成；抽样函数与种子推导规则不改                        RC=0
python3 model/scripts/generate_observations_seeded.py --family contraction_2d,bend_2d --verify-committed
# ModuleNotFoundError: No module named 'pandas'                                  RC=1  ← 诚实红灯
```

最后一条是本机的**边界声明**：没有 pandas 时 verify 只能这样红给你看，不许装绿。
所以自测里 verify 那条绿是"桩化重放"（真数据来自已入库 CSV，抽样函数被替换）——
它证明记账与判定链正确，**不证明** obs_seed=0 能逐位复现真抽样；那一条仍要在实例上跑一次
`--verify-committed`（期望 `'"identical": N'` 且 N=n_jobs）。负对照实测：点位换序判
`SAME-SET-DIFF-ORDER`、少抽一点判 `DIFFERENT`、两臂点数不同判"观测预算不对齐 {'region': 63, 'uniform': 62}"。

### 8.4 T5 主矩阵与这个生成器的依赖关系（已核，含证据）

`sweep_t5.sh:111-120`：`generate_obs()` 第一行 `[[ "${RUN_T6}" == 1 ]] || return 0`，循环里再
`[[ "${s}" == "0" ]] && continue`。实测 dry-run：

| 命令 | `[dry-run][obs]` 行数 | 结论 |
| --- | --- | --- |
| `bash model/scripts/sweep_t5.sh --t5 --dry-run`（默认 `--obs-seeds 0 1 2 3`） | 0 | **`--t5` 无论传什么 obs_seed 都不碰生成器** |
| `bash model/scripts/sweep_t5.sh --t5 --obs-seeds 0 --dry-run` | 0（95 条训练 argv=15 稠密+80 strict，另 95 条评估 argv，rc=0） | 同上 |
| `bash model/scripts/sweep_t5.sh --dry-run`（T5+T6 同开） | 3（seed 1/2/3，各一次） | 只有这条依赖生成器；生成器坏时 T6 会在开跑前 `[FAIL] 观测生成失败，停止` |

⇒ **T5 主矩阵 19 格 × 5 `train_seed`、`obs_seed=0` 全程只吃 `cases/**/obs_*.csv` 已入库文件**，
与本次修复无关，在跑的这段不用停。生成器的坑只影响 T6（`obs_seed=1..3`），且已修 + 桩化验过。

## 9. 产物落点：仓根不许再出现 `out/`（9/25 卫生修复）

`--dry-run` 曾在仓根生成 `out/dryrun_argv.txt` + `out/logs/`，把 git 工作区弄脏（`?? out/`）。
脏工作区的代价不只是难看：段末要 `git add` 产物 push 回仓，未跟踪的 dry-run 垃圾会被一起卷进去，
"提交里有什么"从此不可信。所以默认落点改成**与脚本仓分离**，`sweep_lib.sh:19-41` 的 `_pick_out_dir()`
按顺序取第一个可写者：

| 顺序 | 候选 | 命中条件 | 实测 |
| --- | --- | --- | --- |
| 1 | `$SWEEP_OUT_DIR` | 显式指定 | `OUT_DIR=/d/PINN-restart/.scratch/ovr_out` ✓ |
| 2 | `${SWEEP_WS_ROOT:-/mnt/workspace/pinn-repro-2026}/out` | 该工作区目录存在（实例上就是这条） | 用假 ws 模拟：`…/.scratch/fake_ws/out` ✓ |
| 3 | `<仓根的上一级>/.scratch/sweep_out` | 本机默认（**在 git 仓之外**） | `OUT_DIR=/d/PINN-restart/.scratch/sweep_out` ✓ |
| 4 | `${TMPDIR:-/tmp}/pinn-sweep-out` | 3 也不可写时 | 未触发 |

四条都不可写 ⇒ 打印 `INVALID: 找不到可写的产物目录` 并 `exit 1` —— **不退回仓根**，
因为"退回仓根"正是这次要修的毛病。实例上如果工作区不在 `/mnt/workspace/pinn-repro-2026`，
投递时带 `SWEEP_WS_ROOT=<你的工作区>` 或直接 `SWEEP_OUT_DIR=<产物目录>`。

段首 `[plan]` 现在会把选中的目录打印出来（`OUT_DIR=…`），所以"产物到底落在哪"是每条命令自己声明的，
不靠人记。本机自证：`bash model/scripts/sweep_t5.sh --t5 --dry-run` ⇒ 仓根 `out/` 不存在、
`git status --porcelain` 只有被改的两个脚本（无 `??`）。

注：本文件 §2 与 §8.3 的本地命令因此会把 dry-run 产物写到 `D:/PINN-restart/.scratch/sweep_out/`，
不再写进仓库。
