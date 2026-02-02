# Q2 Counterfactual 输出说明（用于第二小问）

本目录用于回答题目第二小问（比较 Rank vs Percent、以及加入“评委从最后两名中选一人淘汰”的影响）。

## 一、论文主结果建议引用（主表/主图）

### 5.1 Rank vs Percent：同周差异有多大、差在哪边

- `rank_vs_percent_by_week.csv`
  - **周级差异**（在同一周、同一批 Q1 后验粉丝份额样本下对比）：
    - `disagreement_rate`：Rank-direct 与 Percent-direct 给出不同淘汰者的概率 $P(E_r\neq E_p)$。
    - `rank_more_fan_friendly_rate_if_disagree`：在发生分歧的样本中，Rank 淘汰者在粉丝端更“差”（更保护人气）的比例。
    - `rank_more_judge_friendly_rate_if_disagree`：在发生分歧的样本中，Rank 淘汰者在评委端更“差”（更保护技术）的比例。
- `rank_vs_percent_by_season.csv`
  - 将上面的周级指标按赛季求平均，得到赛季层面的对比结论。
- `rank_vs_percent_overall.csv`
  - 全部赛季/周汇总的总体结论。
- `rank_vs_percent_S27_heatmap.png`
  - S27 的周级热力图：展示 `disagreement_rate` 及方向性指标在各周的变化（用于论文中“关键赛季”展示）。

> 方向性指标的口径：以“被淘汰者在当周的粉丝/评委排序分位（badness quantile）”衡量，1 表示当周最差，0 表示当周最好。

### 5.4 “最终结果改变多少”：决赛冠军/名次的量化对比

- `finale_winner_change_by_week.csv`
  - 决赛周的 `winner_disagreement_rate`：Rank vs Percent 在该决赛周“冠军是否变”的概率。
- `finale_winner_change_by_season.csv`
  - 决赛周指标按赛季汇总（通常每赛季 1 次决赛周）。
- `finale_winner_change_overall.csv`
  - 决赛冠军改变概率的总体汇总。
- `finale_celebrity_outcome_deltas.csv`
  - **题目点名争议选手（默认 4 人）**在“决赛当周参赛集合固定”的前提下：
    - `win_prob_rank / win_prob_percent`：两种方法下的夺冠概率。
    - `delta_win_prob(rank_minus_percent)`：夺冠概率差。
    - `exp_place_rank / exp_place_percent` 与 `delta_exp_place(rank_minus_percent)`：期望名次差（负值表示 Rank 期望名次更好）。

### 5.5 非决赛周：制度影响的“赛季走多远”代理量化

- `tracker_elimination_probabilities.csv`
  - 争议选手逐周“当周被淘汰概率”曲线数据（对应 `trackers/` 里的折线图）。
- `tracker_survival_summary.csv`
  - 将逐周淘汰概率作为 hazard 近似，输出：
    - `survive_prob_through_modeled_weeks`：在被建模的这些周中都不淘汰的概率（“走到最后一周”的代理）。
    - `expected_exit_week_index / expected_exit_week_label`：期望出局周（将幸存者视为出局周 = n+1 的约定）。

> 注意：这是“逐周反事实沙盒（固定当周参赛集合）”下的近似汇总，不模拟反事实淘汰后粉丝票迁移与后续周参赛集合反馈。

## 二、辅助输出（建议放附录/自检）

### 5.2 Reversal Rate（稳健性/对齐度）

- `core_metrics_by_week.csv` / `core_metrics_by_season.csv` / `core_metrics_overall.csv`
  - `reversal_rate`：与历史真实淘汰不一致的概率（可作为机制复刻的稳健性自检）。

### 5.3 Tech/Popularity 指标（机制解释/附录）

- `core_metrics_by_week.csv` / `core_metrics_by_season.csv` / `core_metrics_overall.csv`
  - `tech_vulnerability`、`popularity_vulnerability`：更适合放附录或在个案解释时引用，不建议在正文主结果堆太多表。

> 说明：正文主结果建议优先引用 5.1、5.4、5.5，对齐 Q2 思路说明的编号体系。

## 三、分布可视化（决赛名次分布）

- `finale_distributions/`
  - 每个决赛周、每种方法的名次分布矩阵（csv + png 热力图）。文件名形如：`S{season}_W{week}_{method}.csv/.png`。

## 四、数据覆盖范围说明

- `skipped_weeks.csv`
  - 记录哪些 (season, week) 被跳过及原因（例如：双淘汰、无淘汰、Q1 未导出该周样本、对齐失败等）。
- `missing_in_npz_week_classification*.csv`
  - 对“缺失于 NPZ 的周”的类别汇总，便于在论文中说明覆盖范围。
