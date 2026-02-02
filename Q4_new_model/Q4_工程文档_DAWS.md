# Q4 工程文档：DAWS（Dynamic Adaptive Weighting System）落地方案

本文档用于指导代码实现 Q4 提出的新合成规则 DAWS，并复用现有 Q2 反事实仿真引擎（fixed-week sandbox）进行参数搜索、指标评估与可视化输出。

> 设计目标：把“规则（DAWS）”做成一个可插拔 method，能够像 Q2 的 rank/percent 一样在同一套数据与评估口径下运行；输出可直接进入论文（表格 + 图）。

---

## 0. 范围与约束

### 0.1 目标

- 实现 DAWS 的周淘汰规则（direct + save）以及决赛名次分布（distribution）。
- 在历史赛季数据上，基于 Q1 后验粉丝投票样本进行蒙特卡洛评估。
- 通过网格/约束搜索得到推荐参数（$w_{early}, w_{late}, N_{cut}, save$），并产出帕累托前沿或热力图。

### 0.2 重要约束（必须写进工程假设）

- **Fixed-week sandbox（与 Q2 一致）**：每一周只在“该周实际参赛名单”内做规则对比；不进行跨周的真实淘汰链式反事实，因为数据在选手被实际淘汰后通常没有其后续周表现（评分为 0 或缺失），无法构造其“若未淘汰的反事实评分”。
- Q1 提供的是粉丝投票的**后验样本**，不是 ground truth。代码层面将其视为“可能世界”的抽样来源：我们在不同的可能世界中测试 DAWS 的表现稳定性，而不是用后验样本“证明历史一定正确”。
- 工程上建议保留接口支持“子样本/扰动”：例如允许对 `fan_samples` 做子采样、或在调用 DAWS 之前对份额叠加小扰动，用以做稳健性检查（多次重复评估，比较指标波动）。

---

## 1. 复用的现有组件（Q2）

现有 Q2 引擎结构：

- 数据加载：`Q2_tuiyan/q2_loader.py` 产出 `WeekData`
  - `fan_samples`: shape `(S, N)`
  - `judge_scores`: shape `(N,)`
  - `judge_share`: shape `(N,)`
  - `contestants`: list[str]
- 核心规则核：`Q2_tuiyan/q2_kernel.py`
  - 提供 rank/percent、direct/save、finale distribution 等函数
- 评估与输出：`Q2_tuiyan/q2_analyze.py` + `Q2_tuiyan/q2_main.py`
  - 输出周级指标 `reversal_rate / tech_vulnerability / popularity_vulnerability`
  - 输出决赛分布与争议对比统计

DAWS 的落地策略：

- 尽量沿用 Q2 的数据结构（WeekData）与输出口径；
- 作为一个新的 `method family` 插入：`daws_direct / daws_save`（以及决赛 `daws`）。

---

## 2. DAWS 规则定义（实现规格）

### 2.1 输入/输出

输入（对某一周）：

- `fan_samples`: `(S, N)`，来自 Q1 的粉丝投票样本
- `judge_share`: `(N,)`，评委得分占比（Q2 已在 clean 数据中准备）
- `judge_scores`: `(N,)`，评委绝对分（用于 Save 的“评委在 bottom2 中选更差者”）
- `names`: `list[str]`，用于稳定的字典序打破平局
- `w_fan`: `float`，粉丝权重

输出：

- 周淘汰模拟：`eliminated_index: np.ndarray`，shape `(S,)`，每个样本世界被淘汰的选手索引
- 决赛名次：`placements: np.ndarray`，shape `(S, N)`，每个样本世界每位选手名次（1=冠军）

### 2.2 归一化约定（必须在代码中显式校验）

- `judge_share` 应满足 $\sum_i judge\_share_i \approx 1$。
- `fan_samples` 推荐为“份额”样本，满足每行 $\sum_i fan\_samples_{s,i} \approx 1$。

若发现 `fan_samples` 不是份额（例如是原始票数或未归一化），实现中必须在进入 percent/DAWS 前进行按行归一化：

$$fan\_share_{s,i}=\frac{fan\_samples_{s,i}}{\sum_k fan\_samples_{s,k}}$$

（注意：rank 系列不要求份额，但 DAWS/percent 要求份额。）

### 2.3 周淘汰（direct，方案 A：DAWS-Direct）

DAWS 合成得分（份额空间）：

$$C_{s,i}=w_{fan}\,F_{s,i} + (1-w_{fan})\,J_i$$

其中 $F_{s,i}$ 为粉丝份额，$J_i$ 为评委份额。

淘汰规则（单淘汰）：

- 当周淘汰 $C_{s,i}$ 最小者（$K=1$）。

若赛制要求双淘汰/无淘汰，可以在调用层传入当周应淘汰人数 $K$，按排序后的前 $K$ 个索引作为淘汰集合：
- 单淘汰：`k_elim = 1`；
- 双淘汰：`k_elim = 2`；
- 无淘汰周：`k_elim = 0`（只计算排名和 bottom 区，用于兴奋度分析）。

平局规则（与 Q2 保持一致，便于可比）：

- 先比 `judge_scores`（低者更差）
- 再比 `fan_samples`（低者更差）
- 再比 `names`（字典序稳定）

### 2.4 周淘汰（save，方案 B：DAWS-Save）

步骤（以单淘汰为例）：

1) 先按 direct 规则计算 $C_{s,i}$，得到排序；
2) 取综合得分最差的 2 人作为 Bottom-2；
3) 评委在 Bottom-2 中投票淘汰“评委分更低者”；若评委分相等，按粉丝更低者；再相等按名字；
4) 若当周为双淘汰周，可扩展为 Bottom-3：先在 Bottom-3 中保留评委分最高的一人，其余两人淘汰。

实现上，DAWS-Save 可以与 Q2 中的 `rank_save / percent_save` 保持同样的接口与行为风格，只是把底层排序逻辑换成 DAWS 的 $C_{s,i}$。

### 2.5 决赛名次分布（finale distribution）

对每个样本世界 $s$：

- 用 $C_{s,i}$ 从大到小排序得到名次（1=最大者）。
- tie-break 建议与 Q2 percent 的 lexsort 方向保持一致：
  - 合成分更高者更好
  - 粉丝份额更高者更好
  - 评委绝对分更高者更好
  - 名字字典序稳定

---

## 3. “两阶段”权重与切换逻辑

### 3.1 参数

- `w_early`: 常规赛粉丝权重
- `w_late`: 季后赛粉丝权重
- `N_cut`: 切换阈值（剩余人数）
- `save_enabled`: 是否启用 save

建议约束（来自 Q3 的解释口径，便于论文叙述）：

- $0 \le w_{late} \le w_{early} \le 1$（前期更尊重观众，后期更尊重技术）

### 3.2 在 fixed-week sandbox 下的实现方式

不做全季反事实淘汰链。

因此每一周的权重由“该周参赛人数 $N$”直接决定：

- 若 $N \le N_{cut}$，使用 $w_{late}$；
- 否则使用 $w_{early}$。

调用 Q2 时，可以在 `WeekData` 级别根据当周参赛人数自动计算出 `w_fan`，再传给 DAWS 内核，从而在 Q2 现有的 `method` 维度上自然新增 `DAWS-Direct` 与 `DAWS-Save` 两种方法，在相同的样本与周集合上跑一遍，实现“历史规则 vs 新规则”的并行评估。

---

## 4. 指标与输出（与 Q2 对齐）

### 4.1 周级核心指标（沿用 Q2 列名/含义）

对每个 `WeekData`、每个 method（包括新增的 `DAWS-Direct / DAWS-Save`）统计：

- `reversal_rate`：与真实淘汰不一致的概率（若该周有真实淘汰记录）；
- `tech_vulnerability`：评委第一名（或前几名，视 Q2 实现而定）被淘汰的概率，对应论文中的“技术冤死风险”；
- `popularity_vulnerability`：粉丝第一名被淘汰的概率，对应论文中的“观众有效性/人气被冤死风险”。

说明：这些指标在 Q2 里已定义并被主程序聚合；DAWS 只需补齐 method，即可在相同口径下比较 Rank / Percent / DAWS 的公平性与兴奋度表现。

### 4.2 DAWS 参数搜索输出

新增输出建议（写到 `outputs/q4_new_system/`）：

- `daws_search_grid.csv`
  - 每行一个参数组合：`w_early, w_late, N_cut, save_enabled`
  - 聚合指标：`mean_reversal_rate, mean_tech_vulnerability, mean_popularity_vulnerability`（以及分季统计）
- `daws_pareto_frontier.csv`
  - 从网格中抽取非支配解（基于公平/兴奋两个或三个指标）
- 图：
  - `daws_heatmap_*.png`（例如固定 N_cut、save 下的 $w_early \times w_late$ 热力图）
  - `daws_pareto.png`

### 4.3 与现行规则对比

在同一份输出里保留 Q2 的 baseline（rank/percent, direct/save），并新增 DAWS 的结果，使报告可直接引用：

- 总体：按 `method` 聚合均值/分位数，比较 DAWS 与 Rank / Percent 的整体表现；
- 分赛季：按 `season, method` 聚合，观察不同阶段赛制下 DAWS 是否仍然稳定；
- 争议赛季：例如 S2/S11/S27 的周级热力图，重点对比历史规则与 DAWS 在这些“粉丝与评委冲突赛季”中的表现差异；
- 决赛模块：在 Q2 的 finale 分布输出中增加 DAWS 决赛分布，比较冠军改变概率与关键选手（Jerry Rice, Bristol Palin, Bobby Bones 等）的名次/夺冠概率变化。

---

## 5. 代码结构建议（落地目录与职责）

我们采用“轻侵入、长期最干净”的方案：**对 Q2 的分析层做小重构，让 method 列表可注入**，但 Q2 默认行为保持完全不变。

### 5.1 为什么要做 method 注入

当前 Q2 的 `q2_analyze.py` 内部把 method 写死在 `_method_results(week)` 里，导致：

- 如果不改 Q2，就无法“原样复用”其 `analyze_* / plot_*` 来评估 DAWS；
- 如果复制 Q2 的 analyze 代码到 Q4，会产生重复与口径漂移风险。

因此我们把“返回哪些 method”的逻辑参数化，让 Q4 只提供 DAWS 的 method 列表即可复用整套分析与画图。

### 5.2 对 Q2 的最小改动（默认不变）

改动目标：仅新增可选参数，不改变现有调用路径。

1) 在 `Q2_tuiyan/q2_analyze.py`：

- 将 `_method_results(week)` 重命名/提升为公开的 `default_method_results(week)`（或保留旧名但对外暴露）。
- 给以下分析函数新增可选参数 `method_results_fn`：
  - `analyze_core_weeks(core_weeks, method_results_fn=None)`
  - `analyze_trackers(core_weeks, names, method_results_fn=None)`
  - `analyze_finales(finale_weeks, method_results_fn=None)`（如该函数内部固定 rank/percent，可先不改；DAWS 决赛分布建议新增一个 `analyze_finales_generic` 或在原函数加开关。）
- 当 `method_results_fn is None` 时，使用 `default_method_results`，从而保证 Q2 的 `q2_main.py` 输出完全一致。

2) `Q2_tuiyan/q2_main.py`：

- 不需要改（继续使用默认 method），从而保证原 Q2 结果完全不变。Q4 单独提供一个 runner（例如 `q4_main.py`）调用 `analyze_*` 时传入 DAWS 专用的 `method_results_fn`，生成 Q4 所需的补充输出。

### 5.3 Q4 侧代码结构（零复制复用 Q2 分析）

在 `Q4_new_model/` 新增（或后续补齐）文件：

- `q4_daws_kernel.py`：实现 `daws_direct / daws_save / daws_finale_distribution`
- `q4_methods.py`：提供 `daws_method_results_fn_factory(params)`，返回形如 `method_results_fn(week)->list[MethodResult]` 的可调用对象
- `q4_search.py`：参数网格/约束搜索，循环调用 Q2 的分析函数并聚合输出
- `q4_main.py`：CLI 入口，负责读取参数范围、输出目录、以及是否启用 save

这样 Q4 可以通过：

- `loaded = load_q2_data(...)`
- `core = analyze_core_weeks(loaded.core_weeks, method_results_fn=...)`

直接复用 Q2 的指标口径、表格输出和可视化工具。

---

## 6. 验收标准（Definition of Done）

功能验收：

- DAWS `direct/save` 在任意 `WeekData` 上运行不报错，输出 shape 正确。
- 当设置 `w_fan=0` 时，DAWS 退化为“纯评委份额”（淘汰应与 `judge_share` 最低者一致，save 模式只看 bottom2 再按 judge_scores）。
- 当设置 `w_fan=1` 时，DAWS 退化为“纯粉丝份额”（淘汰应与 fan 最低者一致，save 仍由评委在 bottom2 中选）。

一致性/可复现：

- 若对 fan_samples 进行子采样，子采样策略必须固定（例如 stride 或固定随机种子）。
- 输出 CSV 列名、含义与 Q2 保持一致或在 README 中说明。

性能：

- 单次全量跑（所有周 + 所有 method + 默认样本数）在可接受时间内完成；若不可接受，提供 `max_samples_per_week` 参数。

---

## 7. 风险点与处理策略

- **fan_samples 是否为份额**：必须在实现中检测；必要时归一化。
- **平局处理**：必须与 Q2 保持一致，否则结果可比性差、也难答辩。
- **全季反事实的误解**：文档与报告中明确“fixed-week sandbox”，避免评委质疑“你们怎么知道被淘汰后还会跳得一样好”。
- **稳健性**：在同一 DAWS 参数下，对 fan_samples 做子采样/噪声扰动后，多次跑出的核心指标波动应在可接受范围内（可在工程上通过简单重复运行+聚合均值/方差来检查）。

---

## 8. 下一步执行清单（写代码顺序）

1) 实现 DAWS kernel（direct/save/finale）并加最小单元测试（shape + 退化情况）。
2) 对 Q2 做“method 注入”小重构：给关键 `analyze_*` 增加 `method_results_fn`（默认 None 时保持原 4 种规则不变）。
3) 在 Q4 提供 `method_results_fn`（封装 DAWS 参数与 direct/save），复用 Q2 的 `analyze_core_weeks / analyze_trackers / (finale 分布相关分析)` 生成一版输出。
4) 添加 Q4 参数搜索 runner：跑网格、输出 `daws_search_grid.csv`、画帕累托图，并可选加入“留一赛季交叉验证”流程（在 S-1 个赛季上选参数，在被留出的赛季上检验），用工程结果支撑“不过拟合个别赛季”的论断。
5) 在报告中引用：Q3 解释分阶段、Q2 输出支撑参数选择、Q1 说明粉丝投票不确定性如何传播到结果，Q4 工程结果给出 DAWS 在公平性/兴奋度平面上的相对位置与推荐参数。
