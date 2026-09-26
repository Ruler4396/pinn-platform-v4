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

## 10. 2026-09-25 · 列表参数误用的修复（`--seeds 42,43,…` 被当成一个种子）

### 10.1 现场与根因

实例上投主矩阵时传 `--seeds 42,43,44,45,46`：`SEEDS` 是"空格分词"的串，逗号写法整串变成**一个词** ⇒
plan 行显示"实际要训练=19"（19 格 × 1 个"种子"），run 名 `rev2609_t5c01__s42,43,44,45,46__o0`（目录名带逗号），
训练全部失败（无 metrics.json），`_progress_append` 里 `int(tseed)` 也吃不下。**最后是段末记账闸门抱住的**
（`INVALID: 段 01 在账本里 0 行` + `[STOP]`），没有假装跑过 OK——这道闸今天第一次在现场起作用。

### 10.2 修法（三档，全部不改判据、只把误用变成硬失败）

| 位置 | 改动 |
| --- | --- |
| `sweep_lib.sh` 新增 `norm_list()` | 逗号/分号/Tab/换行 → 统一空格分隔、折叠重复空白；纯 bash 内建（不引外部命令、不触发 glob）。`sweep_t5.sh` 在读完参数后对 `--seeds/--obs-seeds/T6_SEEDS/--only-cells` 全部规范化，并保留原值用于报错。 |
| `sweep_lib.sh` 新增 `validate_int_list()` | 逐项 `^[0-9]+$` 校验；**0 项也算错**（空矩阵不许当"没问题"跑过去）；不合格项连同**收到的原值**一起打印。由 `preflight()` 顶部的 `validate_sweep_lists` 钩子调用 ⇒ dry-run 同样受校验，且在 `mkdir` 之前，坏参数不会先造出目录。 |
| `sweep_lib.sh` 新增 `assert_run_name()` | run 名字符集限定 `^[A-Za-z0-9_.-]+$`，出现在 `train_dual`/`eval_run`/`train_mlp` 三个执行入口的最前面 ⇒ 逗号/空格撑开的名字直接 `exit 1`，不会写出畸形目录。 |
| `sweep_t5.sh` `[args]` 行 | 段首打印解析结果（`seeds=[42 43] obs_seeds=[0 1 2 3] …`），"你以为传了什么 vs 脚本实际读到什么"变成每条命令自己声明的。 |
| `generate_observations_seeded.py` | `parse_list()` 同样支持逗号/分号/空白；`--obs-seeds` 逐项整数校验、`--rates` 校验落在 (0,1]（`--rates 5` 会被拒，而不是生成 500% 采样）、`--strategies` 只许 region/uniform（`regio` 以前会被 else 分支**静默当成 region**）。全部把原值打出来。 |
| `--only-cells` | 逗号与空格都收，且只许 1..19 或 `t6`（`--only-cells 99` 以前会静默匹配不到任何格 ⇒ 一个单元都不跑还打印"完成"）。 |

### 10.3 三行自测命令（本机实测，都不跑训练）

```bash
cd pinn-platform-v4
bash model/scripts/sweep_t5.sh --t5 --seeds "42,43" --dry-run | grep -E "^\[args\]|实际要训练"
bash model/scripts/sweep_t5.sh --t5 --seeds "42 43" --dry-run | grep -E "^\[args\]|实际要训练"
# 两行输出必须逐字相同：[args] seeds=[42 43] … 与 [plan] … 实际要训练=38（19 格 × 2 种子；train 行也各 38）
bash model/scripts/sweep_t5.sh --t5 --seeds "42,x,44" --dry-run
# [FAIL] --seeds 含非整数项：x（收到的原值='42,x,44'）⇒ rc=1
```

同批负例（各自 rc=1，实测）：`--seeds ""` → `解析出 0 项`；`--only-cells 99` → `只许 1..19 或 t6`；
`--prefix "rev26,09"` → `run 名不合法（只许 [A-Za-z0-9_.-]）：'rev26,09_t5c01__s42__o0'`；
`--obs-seeds "1,x"` / `--rates 5` / `--strategies regio`（python 侧三条）。
正例：`--obs-seeds "1 2"` 与 `"1,2"` 都排 4 个作业；回归 `--budget-only` 两族 60 组合仍 rc=0。

### 10.4 「部分格完成」时 analyze_sweep.py 的行为（只核不改，探针在 `.scratch/plan_analyze_partial_probe.py`）

用合成结果树（`--results-root` 指到仓外）跑 5 条断言，全部符合预期：

| 输入 | 行为 |
| --- | --- |
| 前缀下 0 个 run | `INVALID：前缀 rev2609 在 …/pinn 下没有可解析的 run（不是'没有差异'，是根本没数据）` + rc=1 |
| 格4 满 5 种子、格8 只有 2、格13 只有 1、另有一个 run 缺 `evaluations/` 产物 | 正常出表：`[load] 可解析 run=8，缺评估文件=1`，每格带 n 与 std，`n=1` 那格标 **`n=1 无任何统计强度`**；rc=0 |
| 同上 + `--paired t5c04,t5c08`（本机无 scipy） | 聚合表先出，配对段明确报错要装 scipy，rc=1（不降级手算） |

⇒ **可以降级出表，缺的格报 MISSING/警告而不是崩。**一处缺口要记账（本轮按"不碰其他脚本"没修）：
`analyze_sweep.py:312` 的 `or cell_b not in grouped: continue` 会把**整臂一个 run 都没有**的那对对照
静默跳过（实测判词只出 1 条而不是 2 条），`:162` 里 `judge()` 的"一格没有读数 ⇒ 不可判"分支因此不可达。
中途看数的人可能把"这条没印"读成"没什么可报"。一行改法（等你点头再动）：
把跳过条件收紧成"两臂都不在" —— `if (cell_a not in grouped and cell_b not in grouped): continue`，
这样缺臂那对会走 `judge()` 印出「不可判：对照两臂里有一格没有读数」。

## 11. 2026-09-26 · 必做1 基线三件套（臂 A/B/C）+ 段末对账重做

### 11.1 接口（统括官定死，代码照此实现）

| 臂 | 入口 | 与双模型逐项对齐的部分 | 唯一被试变量 | 产物 |
| --- | --- | --- | --- | --- |
| A 单网络联合 PINN | `model/scripts/train_joint_upnp_pin.py`（`sweep_lib.sh:train_joint`，格 20/21） | 数据切分、观测表来源、输入/输出标准化（含 `--strict-sparse-scalers`）、**物理项直接复用双模型的 `方程耦合损失`**（把单网络包成"速度头/压力头"两个切片代理传进去）⇒ 求导链、尺度因子、512 点等距取点与双模型同源而非近似。**边界项的适用范围（9/26 更正）**：壁面/入口流量/出口压力/压降这四项要求"稠密网格上的对应点，与观测稀疏度无关"；臂 A 原先只在**稠密档**因"观测集=稠密集"侥幸成立，稀疏档第一粒即 `IndexError: mask[12498] vs tensor[496]`（实例 seg09）。现改为对 dense 与 obs 各一次前向、四项前都过 `核对点数()` 形状闸（不符时报具名 ValueError 而非裸 IndexError）。**验证状态（9/26 定档）**：① 形状/结构层由 `--shape-selftest` 证到（7 例全 OK，含变异 A/B/C 三条正对照）；② 两档各 **5/5 跑完 480 epoch**（段 13 实跑 `5 runs / 0 failures / 0 skipped / 0 gate-fail / 0 eval-fail`，逐粒 129.5–142.2 s，例 `rev2609b_t5c21__s42__o0` 打出 `[done] … val speed=0.0126 p=0.0419 (ep=480)`）；③ `44559b2` 之后 test 件齐 ⇒ 两档 test/mean-of-cases 都有数。**凭据换绑**：我原先要求看 `[smoke] step1 … l_wall=有限值`，而全量跑法根本不打那行——错在我把凭据设在不存在的日志上，不是『稀疏档未验过』。现第一 step 恒打 `[step1] … l_wall / l_inlet / l_outlet / l_pdrop … 边界项有限性=True` 并进 history.csv 四列⇒ 凭据 = 两档各 5/5 全量 + 三变异正对照 + 有限 val/test 指标 + `[step1]` 边界项行 | ① 一个网络而不是两个；② 一次训练而不是三阶段 | `results/pinn/<run>/{config.json,best.ckpt,history.csv,metrics.json,evaluations/metrics_{val,test}_dense.json}`，schema 与双模型评估产物同名 ⇒ `analyze_sweep.py` 不改即可读 |
| B 纯数据 MLP | 既有 `train_supervised.py`（`sweep_lib.sh:train_mlp_with_test`，格 22/23） | 同特征集、同网络规模、同观测表 | 物理项与壁面包络**全为 0**（该脚本本来就没有这两项） | `results/supervised/<run>/...` |
| C POD + 观测点最小二乘 | `model/scripts/baselines_pod.py`（`sweep_t5.sh:run_pod_baseline`） | 同一批稠密真值构造基；观测点用**与格1/格4 同一张 CSV** | 无网络、无训练 | **单独 JSON**（默认 `out/pod_baseline_contraction.json`）；**不写 progress.jsonl** —— 它没有 train 阶段，写一行 `phase=train, wall_ms=0` 等于骗账本 |

参数量对齐（臂 A）：单网络 `14→128×5→3` = **68,355**，双模型 `14→128×3→2` + `14→128×3→1` = **70,275** ⇒ 比值 **0.9727**。这条不是注释里的话，是闸门：
`--require-param-ratio 1 --param-ratio-tol 0.05`，超窗直接 `exit 1`（实测负例 `--hidden-layers 128,128,128` ⇒ 比值 0.5028 ⇒ `[FAIL] 参数量比值 … 不在 [0.95,1.05]`，rc=1）。

命名与主矩阵完全分开：前缀 `rev2609b_`、格号续到 `t5c20..t5c23`（臂 A/B 各两档），`--only-cells` 已放开 20..23 并对越界报红。

### 11.2 argv 清单（本机 dry-run 实跑，20 条训练命令）

```bash
bash model/scripts/sweep_t5.sh --baseline --dry-run | grep -cE "dry-run..train-(joint|mlp)"      # 20
bash model/scripts/sweep_t5.sh --baseline --dry-run >/dev/null; wc -l < <仓外产物目录>/dryrun_argv.txt   # 20（10 臂A + 10 臂B）
bash model/scripts/sweep_t5.sh --baseline --dry-run | grep armC                                  # 臂 C 的命令行与"不进账本"声明
```
机时对账：臂 A/B 各 2 档 × 5 种子 = 20 次训练。单价用本实例实测（收缩稠密 81.4 s、稀疏 53.7 s）⇒
串行 ≈ 10×81.4 + 10×53.7 = **1351 s ≈ 23 min**，与统括官给的 ≈25 min 一致；臂 C 秒级。¥0。

### 11.3 明天上机的顺序（先冒烟再全量）

1. `python3 model/scripts/train_joint_upnp_pin.py --run-name smoke_joint --epochs 8 --print-every 4 --seed 42 --require-param-ratio 1 --dry-run` 先核参数量（不占机时）；
2. 真跑一次冒烟：`--epochs 8`（几分钟）确认 ckpt/evaluations/metrics 三件套落盘且 `rel_l2_*` 是有限值；
3. `bash model/scripts/sweep_t5.sh --baseline --seg 09 --budget-min 40`；重投同段即续跑（`metrics.json` 在则 `[skip-train-joint]`）；
4. 臂 C：`python3 model/scripts/baselines_pod.py --self-check`（几秒，断言模态数、观测残差、两口径读数在可解释范围），再跑正式档。
5. 段末 `seal_segment` 现在会打印 `N runs / M failures / K skipped / G gate-fail / E eval-fail` 五个数，任一不符即 `INVALID` 并 rc=1。

### 11.4 段末对账为什么重做（缺陷与修法）

段 01 现场：`INVALID: 内存=87 账本=89 / 失败 内存=0 账本=1`，而数据其实是 95/95 满种子。三条根因与处置：

| # | 缺陷 | 修法 |
| --- | --- | --- |
| 1 | 账本是**追加式**的：同段重投续跑、失败后重试都会留新行，旧口径拿"段内 train 行数"比内存计数 ⇒ 必然假红 | 分两层核：① 本进程核（账本新增 `pid` 字段，`SWEEP_PID` 可注入以便自测）；② 段终态核（按 `(run, phase)` 取**最后一行**）；重试行单独打印 `重试行 N` 不判红 |
| 2 | `rc!=0` 的行不一定进失败计数（尤其 eval 阶段），"内存=0 账本=1"就是这么来的 | 每个 `rc!=0` 的行都进 `RUNS_FAIL`，并另计 `RUNS_EVAL_FAIL`；终态失败的 run、终态失败的评估都被列为 problem |
| 3 | `[skip-train]` 归属不明（跳过 = 账上该有别人的成功行） | 新增等式 `段内终态成功 run 数 == 本次成功 + 本次跳过`，对不上即红；`RUNS_GATE_FAIL` 单列（观测预算/点位生成失败不再被当成"跑完"或"预算截断"） |

自测（驱动**真实**的 `seal_segment`，不是复刻逻辑）：`bash model/scripts/selftest_ledger.sh` ⇒ **十二例全符合**（9/26 本机在已提交状态复跑，尾行 `总体：十二例全符合…`，rc=0），
其中 **③「账本缺一行」、⑥「段内 0 行」、⑪「给失败的 run 补评估」、⑫「补评估数不符」是必定 INVALID 的正对照**，证明闸门没被改成永远绿；②是段 01 那个假红的忠实回放（前进程 1 败 1 成 + 本进程 2 成 1 跳），现在判绿并打印"重试行 1"。退出码判级也逐例断言：①②⑩=0，③⑦⑧⑫=4（仅记账不符），④⑤⑥⑨⑪=1（数据不可信）。

### 11.5 臂 B 的指标进不了账本：`rel_l2_*` 整列 NA 是 schema 问题，不是漏跑（9/26 现查）

统括官给出臂 B 的两个数只能来自 `analyze_sweep.py` 的 supervised 根。本工单把**原因**查到了代码层，避免下一个人再从账本重算一遍：

1. 账本行的指标由 `sweep_lib.sh:_run_metrics_json()` 读取，它只认两套键：`metrics.json` 的 `最终验证指标.rel_l2_{u,v,p,speed}`（双模型/臂 A 的 payload），以及 `evaluations/metrics_val_dense.json` 的 `global_metrics.*`。
2. 臂 B 的 trainer 是 `train_supervised.py`，它写的 `metrics.json` 顶层键是 `best_epoch / best_val_total / train_case_metrics / val_case_metrics / …`（`train_supervised.py:1040-1049`），**没有** `最终验证指标` 这一层；它的评估件名是 `evaluations/metrics_{split_name}.json` 且 `--split-name test_dense`（`evaluate_supervised.py:175`、`sweep_lib.sh:train_mlp_with_test`），也不叫 `metrics_val_dense.json`。⇒ 两个 `first()` 全部落空，`_run_metrics_json` 返回 `{}`，PSV/账本对应列就写成 `NA`。
3. 实测核对：`T5矩阵run坐标索引-20260926.psv` 里 `metrics_root=supervised` 的 10 行（t5c22/t5c23 各 5）**全部** `rel_l2_u/speed/p = NA`，而 `wall_ms`、`rc`、`metrics_present=YES` 完好 ⇒ **NA 只发生在指标列，机时与存在性可信**。
4. 所以：**臂 B 的精度数只能从 `analyze_sweep.py --supervised-root`（默认跟随 `--results-root`）出**；账本/PSV 只能用来核机时与终态。这不是把闸门修绿就能顺手解决的事——要修得改 `_run_metrics_json` 认两套 schema 并回填历史行，属于读数口径变更，**本轮不做**（统括官正在实例上现场复验封段判级，不动 `sweep_lib.sh`）。
5. 顺带一条对表 5-9 有用的读数（同一枚索引现算，rc=0 各 5 粒，单位 s）：臂 A 稠密中位 139.6（136.7–146.7）、臂 A 分层5% 131.2（129.5–142.2）、**臂 B 稠密中位 199.3（120.9–212.9）**、**臂 B 分层5% 101.5（72.4–123.4）**。臂 B 两档区间都超过中位数的 1/3，散布来源未查（trainer 与批大小与 PINN 不同），所以正文引用只给区间、不给单点，也别把"无物理项"推成"更省机时"——稠密档实测它最慢。

## 12. 2026-09-26 第二批 · 统括官四条待办的处置与凭据

| # | 缺陷（他给的原值） | 处置 | 本机自测（命令 → 尾行） |
| --- | --- | --- | --- |
| 1 | `baselines_pod.py:153` 把 `args.eval_cases/obs_files` 赋成 list，`:163/:164` 又对 list 调 `parse_list()` ⇒ `AttributeError: 'list' object has no attribute 'strip'`。**只坏 `--self-check` 分支；真跑那条从未在任何机器上执行过** ⇒ §11.3 第 4 步的"几秒"当时无凭据。**（9/26 已闭，此条不再是欠项）**统括官第四次回传：`armC_selfcheck rc=0`、`armC_real rc=0`、总 5,709 ms、`rank=1`，产物 `out/pod_baseline_contraction.json` 读回校验 | `parse_list` 改为同时接受 str/list/None（清洗后返回），self-check 分支改回赋字符串；判定逻辑抽成单一函数 `检查不变量()`，`--self-check` 与新增的 `--assert-selftest` 共用一份（不留两份判定） | `python3 model/scripts/baselines_pod.py --assert-selftest` → `总体：全符 ⇒ 断言集能红能绿`（干净夹具绿 + **5 条注入各自判红并点名 A1/A2/A3/A4/A6**：压力读数超范围、观测点数少于模态数、观测残差=nan、pooled 不合理、少一个指标键）。崩溃点本身：`parse_list(["C-val"])==["C-val"]`、`parse_list(None)==[]` 等 6 例全过 rc=0。**真数据那条仍只能在实例上跑**（要 numpy + `cases/`）：`python3 model/scripts/baselines_pod.py --self-check`，几秒，产物写临时目录 |
| 2 | `analyze_sweep.py` 只 glob `results/pinn/` ⇒ 臂 B（写 `results/supervised/`）对统计管线完全不可见，`--paired t5c23,t5c04` 被静默跳过 | **决定：臂 B 的表走 analyze，给它加第二根**（`--supervised-root`，默认跟随 `--results-root`/`model/results/supervised`；`--only-pinn` 才回到旧行为并显式声明）。理由：臂 B 是与格1/格4 同观测表的**配对**对照，需要同一套 mean±sd、两口径与配对检验；索引只能出点估计。每行带 `metrics_root` 标签，出表首行打印"结果根=pinn,supervised"与各根 run 数，跨根不会互相冒充 | 合成树四例全 OK：`解析 run=20 格数=4`（双根）、`--only-pinn → 解析 run=15`、`缺一臂必须印不可判`（原判词 1 条→现在 2 条）、`--paired … → scipy 缺失 rc=1` |
| 3 | `sweep_t5.sh --baseline --dry-run` 打 `实际要训练=0`，而它自己铺了 20 个单元（预算闸逐项 cost 判定没坏，是人看的那行数错） | 基线段计划数/ETA 照 `BL_CELLS` 逐项累加进 `n_planned/eta_sec`，并明确"臂 C 不训练 ⇒ 不计入" | `[plan] 基线段 本段要训练=20（4 格 × 5 种子）` + `[plan] segment=seg00 实际要训练=20；串行总墙钟≈27 min`；回归：`--t5 → 95`、默认 → `107+4` 不变 |
| 4 | 臂 A dry-run 打 `权重档=strict-sparse` 而同一行实际是 `{0.5,1e-4,1.0}`=mainline-dense（sweep 只传逐项权重不传 preset ⇒ 数值对、标签错，会被当串档） | 标签由**实际权重反推**（`档位标签()`），与命令行 preset 不一致时当场 `[warn]` 并两个名字都打；sweep 侧改为显式传 `--weights-preset` | `--inlet-flux-weight 0.5 --outlet-pressure-weight 1e-4 --pressure-drop-weight 1.0` → `[warn] --weights-preset=strict-sparse 与实际权重不符…标注为 mainline-dense`；`--baseline --dry-run` 的 argv 里已含 `--weights-preset mainline-dense` |

**他给的环境事实，对本文件的影响**：`--obs-seeds 0 --verify-committed` 在实例上 `"identical": 72, "not_identical": []` ⇒ §8.3 我留的那条"数值等价只能在实例上做"的开口**已闭**，T6（obs_seed 1..3）解锁。实例有 pandas 2.2.3 / scipy 1.17.1 ⇒ `--paired` 的真实 Wilcoxon/t 能在实例上出，不降级手算。参数量比已在实例实测坐实：单网 68,355 / 双模 70,275 / 比值 0.9727。

**臂 C 的诚实边界不变**：`--self-check` 与真跑那条读的是 `cases/`，本轮我仍**没有**在任何机器上执行过完整的 POD 数值路径（本机无 numpy）；我能自证的是断言集本身能红能绿 + 崩溃点已修。第一次真跑的 rc 与 `[pod] rank=…` 那行请回传，我据此把 §11.3 第 4 步的"几秒"改成有凭据的数。

### 12.1 sd 口径统一（9/26，统括官指出我 §12.2 用了 ddof=0）

我在工单 §12.2 里挂的 格19 `±0.001750`、格16 `±0.002029` 与配对表 Δsd `0.077` 是**总体标准差（ddof=0）**，
而《T5矩阵配对读数》第一节与第四节的汇总表是**样本标准差（ddof=1）**；两者比值恰为 √(5/4)=1.118 ⇒ 不是数据分歧，是口径分歧。
处置：**全文只留一把 = 样本标准差 ddof=1**，理由是它同时是统计管线用的那个（`analyze_sweep.py:139` 用 `statistics.stdev`），
改动面最小。工单里相应改为：格19 `0.024347±0.001956`、格16 `0.025082±0.002269`、弯曲稠密 vs 分层 5% 的 Δsd `0.086`（t=1.92）、
消融 Δsd `1.888`（t=45.41）、1% 档 Δsd `1.052`（t=2.62）、表5-8 Δsd `0.003`（t=1.57）。
弯曲稠密单价与 K\* 同步改为**格19 五粒中位 171.6 s**（我原先用的 174.6 s 是 s42 单粒，且它出现在效率表的一处过期备注里，已一并改掉）：
`K*(用C)=49`、`K*(用C')=53`。表注固定句：「本表 sd 为 5 个训练种子的样本标准差（ddof=1）」。

### 12.2 臂 A 的真实单价、稀疏档崩点与对账第一次变红（9/26 seg09 实测，pin b76c6dc）

| 事实 | 数值 | 影响 |
| --- | --- | --- |
| 臂 A 稠密 5/5 成功 | 墙钟 136.7 / 138.4 / 139.6 / 140.9 s 量级（逐粒见 `wall_ms`） | **臂 A 的单价 ≈139 s，不是双模型的 81.4 s，也不是 runner 里拍的 85 s**。引用臂 A 的 K\* 必须用它：收缩族 `K\*=139/(0.501−0.083)=333`（完整推理口径）、`139/(0.501−0.228)=510`（在线重建口径）⇒ 工单表 5-9 按臂分行，避免"拿别臂的价钱当自己的" |
| 臂 A 分层 5% 第一粒崩 | `IndexError: mask[12498] vs tensor[496]`（`train_joint_upnp_pin.py:259` 把 dense 的 wall mask 拿去索引观测子集的前向） | 修法见 §11.1 的"边界项适用范围"；runner 单价随后调保守（稠密 145 s / 稀疏 100 s） |
| 段末对账第一次在真实缺陷上变红 | `5 runs / 1 failures / 0 skipped / 0 gate-fail / 0 eval-fail (账本段 09：6 行；本进程 pid=34914 成功 5 失败 1；段内 distinct train run 6 = 终态成功 5 + 终态失败 1；重试行 0)` → `INVALID: 有 1 个 run 终态失败：rev2609b_t5c21__s42__o0` | 旧口径会记成"失败 0"并放行。这是 pid 核 + `(run,phase)` 末行终态核的现场证据，与 `selftest_ledger.sh` 的合成正对照互补 ⇒ 闸门既没被改绿也没被改死 |

**臂 A 冒烟（补跑 seg12 之前先跑，秒级）**：
```bash
python3 model/scripts/train_joint_upnp_pin.py --run-name smoke_a21 --family contraction_2d \
  --train-cases C-base,C-train-1,C-train-2,C-train-3,C-train-4,C-train-5 --val-cases C-val \
  --train-velocity-source obs_sparse_5pct.csv --train-pressure-source obs_sparse_5pct.csv \
  --weights-preset strict-sparse --strict-sparse-scalers --max-steps 1
# 期望：[step1] … l_wall=… l_inlet=… l_outlet=… l_pdrop=… 边界项有限性=True，且 rc=0（dense 档把两个 source 换成 dense 再跑一次）
```
`--max-steps 1` 写出的 `metrics.json` 带 `"smoke": true`，`train_joint` 的续跑判断**不把它当已完成**（遇 smoke 标记就重训），所以探针不会占住正式格子的名字。

### 12.3 9/26 第三批：臂 A 缺 test 评估件（⑥）与 `--paired` 丢对（⑦），以及臂 C 的首次实测

| # | 缺陷 | 修法 | 本机/实例证据 |
| --- | --- | --- | --- |
| ⑥ | `results/pinn/rev2609b_t5c20__s42__o0/evaluations/` 里只有 `metrics_val_dense.json`；`grep -c "eval-test-cases" sweep_lib.sh` = **0** ⇒ sweep 从没把 test 工况递给臂 A，臂 A 两个口径都进不了表。而臂 B（`train_supervised`+`evaluate_supervised`）四件齐 | `train_joint` 加第 12 参 `eval_test`，dump 与真跑 argv 都追加 `--eval-test-cases`；臂 A 脚本加 `--eval-only`（装 `best.ckpt`、`轮数上限=0`、只重做评估并补写 `metrics_test_dense.json`）；**跳过训练不等于跳过评估**：`[skip-train-joint]` 分支里若 test 评估件缺失就跑 `--eval-only` 并落一行 `phase=eval_test` | 本机：`grep -c eval-test-cases sweep_lib.sh=4`；`--baseline --dry-run` 里 **10/10** 个臂 A 单元带 `--eval-test-cases C-test-1,C-test-2`；`--shape-selftest` 七例全 OK。补评估那趟由统括官投 `--only-cells 20`（~10 s，不重训） |
| ⑦ | `analyze_sweep.py --paired` 是普通参数 ⇒ 传三对只剩最后一对跑，且 `[accounting]` 行看不出来（静默丢对） | `--paired` 改 `action="append"` 并支持 `'a,b;c,d'`；新增 `解析配对()` 与 `核对配对数()`：请求 N 对 × M 口径 ⇒ 必须正好 N×M 条配对记录，少一条 `INVALID` 退出 | 本机：`解析配对` 四例全对；正对照「3 对×2 口径只出 2 条」当场 `INVALID`（rc=1），1 对×2 口径出 2 条则放行 ⇒ 不是永远绿 |
| 臂 C | 首次真跑（本机只能证断言集；数值路径此前无凭据，**现已由实例回传补齐**） | 读数按统括官实测入工单 D2/表 5-10；`--self-check` 打印 `rank=1` ⇒ 三快照的 POD 只留 1 个模态，这本身要在正文里说明（不是"模态数足够"的证据） | 实例：armC_selfcheck rc=0（1 s）、armC_real rc=0（7 s），总 5,709 ms；产物 `out/pod_baseline_contraction.json` |

**口径三分列（统括官 9/26 纠正，我此前引的 .psv 数是验证集）**：`.psv` 的 `rel_l2_*` 列 = **val**；第 5 章的 test 表只能用 `T5矩阵test口径读数-20260926.md`。已据此在工单里把消融换成 test 的 0.539923→0.148723（3.6 倍，val 为 0.5196→0.1362=3.8 倍）、弯曲「5%≈稠密」换成 test 的 +7.8%（val 是 +3.0%），并在 §0 立了「凡 .psv 得来的数标 (val)、凡进正文表的数标 (test)」的规则。

### 12.4 9/26 第四批：`--eval-only` 型段结构上永远封不了段（实例段 14）

**假红的归因（先把责任放对位置）**：那一段的**产物是完整的** —— 10 个 `[ok-eval-only]`、`0 failures`、
`analyze_sweep` 打 `[load] 缺评估文件=0`。红的是我的计数域错了：上一版对账的等式是
`段内终态成功的 train run 数 == 本次成功 + 本次跳过`，它隐含假设"本段每个 run 的 train 行也落在本段"。
补评估段恰恰不满足这个假设：它的 run 是**别的段**训完的（`--only-cells 20,21` 复用 `metrics.json`），
本段只写 `phase=eval_test` 行 ⇒ 等式左边=0、右边=10，永远 INVALID，与产物好坏无关。

**修法（不是把闸门改松，是换一个正确的核法）**：
- 新增 `RUNS_EVALONLY`：跳过训练且确实补了评估的单元记在这里，不再混进 `RUNS_SKIP`；
- 等式改成 `段内终态成功 train run 数 + 本段补评估 run 数 == 本次成功 + 跳过 + 补评估`；
- 两条新硬条件：① 本段补评估的 run 必须在**任意段**有一条终态成功的 train 行，否则报
  「给失败的 run 补评估没有意义」并红；② 内存的补评估数必须等于账本推出的数量。

**自测（`bash model/scripts/selftest_ledger.sh`，现在十二例，rc=0）**：
⑩ 是实例段 14 的忠实回放（别的段的 train 行 + 本段 3 条 eval_test）⇒ 判**绿**；
⑪⑫ 是两条必定 INVALID 的正对照（给一个从没训练成功的 run 补评估 / 内存数与账本不符）⇒ 判红并点名。
连同原有的「账本缺一行」「段内 0 行」，共四条必定红的正对照 ⇒ 这次改动不是把闸门改成永远绿。

### 12.5 段末判级分档：`fatal` 与"仅记账不符"给 runner 分开处理（9/26，回应统括官自认的那条）

你那条"runner 把 seg14 当 fatal 会连带跳过 analyze/index"的根因在我这侧：**seal 只有一个退出码**，
"数据不可信"和"只是我的计数没对上"在 runner 眼里长得一样。现在 `seal_segment` 先打一行机器可读的判级：

```
[seal] 判级=fatal|bookkeeping|clean  fatal=<n>  记账=<n>
```
- **rc=1（fatal，数据不可信）**：账本缺失/本段 0 行/坏行、有 run 终态训练失败、有 run 评估终态失败、
  给"任何段都没有成功 train 行"的 run 补评估、前置闸门没过、有失败单元、完成数 ≠ `EXPECTED_RUNS`。
  ⇒ runner 该停：这一段不能进分析。
- **rc=4（bookkeeping，仅记账不符）**：只有内存计数与账本对不上（本进程成功数/失败数/补评估数、
  跳过归属、同 run 多条成功行），**没有任何终态失败** ⇒ 产物完好，`analyze_sweep`/索引可以照跑，
  但本段不算封板，当天要修。消息里明写"产物完好，analyze/索引可以跑"。
- **rc=0（clean）**。

`bash model/scripts/selftest_ledger.sh` 十二例带**精确退出码断言**（`RCWANT`）：①②⑩=0、③⑦⑧⑫=4、④⑤⑥⑨⑪=1，全 OK；
其中 ③⑫ 是"缺一行/数不符"→ 必须落 4 而不是 1（否则你又会把完好的段当不可信跳过分析），⑤⑥⑨⑪ 必须落 1。

### 12.6 9/26 第五批：中文变量名让补评估段**走不到封段就炸**（实例 seg14），修完本机重放到 `[seal] 判级=clean`

实例原报错（统括官回传）：

```
[skip-train-joint] rev2609b_t5c20__s42__o0
sweep_lib.sh: 第 482 行： local: "是补评估=0": 不是有效的标识符
sweep_lib.sh: 行 502: ${是补评估}: 错误的替换
```

- **根因与封段逻辑无关**：bash 的变量名只接受 ASCII（字母/数字/下划线）。`local 是补评估=0` 报 *not a valid identifier*，
  下一行 `是补评估=1` 被当成"找不到命令"，`${是补评估}` 再报 *bad substitution* ——**三处连炸**，`set -e` 下 `train_joint` 直接死，`[seal]` 根本没机会跑。
  本机同一形态复现（bash 5.2.37）：`local: '是补评估=0': not a valid identifier`。
  Python 里中文标识符合法（`train_joint_upnp_pin.py` 的 `速度头/核对点数` 就是），所以这个习惯被顺手带进了 shell ——**shell 与 Python 的标识符规则不同，这条要记进纪律**。
- **为什么两道既有防线都漏了**：① `--dry-run` 在 `[skip-train-joint]` 之前就 return；② `selftest_ledger.sh` 驱动的是 `seal_segment` 本身，不经过 `train_joint`；
  ③ `bash -n` 只查语法不做名称解析。⇒ **覆盖缺口不是运气，是"这条分支只在续跑时才活"**。
- **修法**：改名 `is_eval_only`（ASCII），并在原地留一句为什么不许再写中文变量名。范围只这一处函数，`sweep_t5.sh` 未动。
- **新增防线（两条，都自带必定红的正对照）**：`bash model/scripts/selftest_joint_evalonly.sh`（本机可跑，不碰实例、不写仓库）
  - **静态闸**：扫 `model/scripts/**/*.sh` 的声明位/使用位/赋值位非 ASCII 标识符，本机现跑 **HITS=0**；
    把 `is_eval_only` 用 sed 改回中文名的**变异副本**上同一道闸报 **HITS=3** ⇒ 证明闸不是摆设。
  - **动态重放**：用桩 python 把真实 `train_joint` 的"已训完 + 缺 test 件"分支跑满 **10 粒（格 20/21 × seed 42–46）**，再接真实 `seal_segment` ⇒
    `10 × [ok-eval-only]`、计数 `done=0 fail=0 skip=0 evalonly=10 evalfail=0`、`[seal] 判级=clean fatal=0 记账=0`、整段 **rc=0**。
    变异副本同法重放则出现 bash 错误、rc≠0、且**打不出** `判级=clean`（= seg14 的真实形态）。
  - **第三条正对照**：让桩 python 对某一粒 `exit 7` ⇒ `fail=1 / evalfail=1`、`[seal] 判级=fatal`、**rc=1**，
    证明"补评估失败"不会被读成"这一段干净"（这一档如果也放绿，上面那条 clean 就等于没测）。
- **实例验收口径（统括官侧，~20 s）**：命令**必须带 `--baseline`**——
  `bash model/scripts/sweep_t5.sh --baseline --only-cells 20,21`（`--baseline` 会把 RUN_T5/RUN_T6 关掉、只铺三件套）；
  本机 dry-run 实测该组合出 **10 条** `[dry-run][train-joint]` 且**每条都含 `--eval-test-cases`**。
  **陷阱（本机实测，不是我推测）**：只写 `--only-cells 20,21` 而不带 `--baseline` ⇒ 三件套一条都不跑，
  而 **T6 配对单元不受 `--only-cells` 约束**（`cell_wanted` 只挂在 T5 主矩阵与基线两个循环上，`t6_matrix` 没挂），
  于是 plan 行显示"实际要训练=12"、dry-run 里出来的是 `t5c04` 与 `t5c08` **各 8 条 train 铺排**（12 个新单元 + 4 条与 T5 格4/格8 同名重复），臂 A 铺排数 **0**——**这一段的机时全花在 T6 上，臂 A 拿到 0 粒**。
  修法我没做（那是 T6 的准入面，且你正在实例上跑），先用"必须带 `--baseline`"这条绕开；要不要把 `--only-cells` 也挂到 `t6_matrix`，等你这一轮复验完再定。
  期望读数：10 个 `[ok-eval-only]` + 一行 `[seal] 判级=…` + rc（0=封上；4=只有计数没对上、产物仍可用；1=有终态失败，本段读数不可信）。
  若这些 run 的 `evaluations/metrics_test_dense.json` 其实都已存在，就会走成 `RUNS_SKIP`（10 个 skip、0 个 eval-only）——那是 seg14 之外的另一种终态，判级同样应为 clean。
