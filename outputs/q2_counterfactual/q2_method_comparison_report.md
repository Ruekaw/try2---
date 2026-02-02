# Q2 赛制反事实比较：结果二次加工报告

本报告用于把 Q2 的 Monte Carlo 推演指标与题目第二小问的表述更直接挂钩。

## 1. 我们用哪些指标回答题干？

- **规则更偏向粉丝投票？** 重点看 `p_elim_fan_bottom`（淘汰落在当周粉丝最低者的概率）与 `expected_fan_rank_elim`（被淘汰者的期望粉丝名次，越接近 N 越说明淘汰更由粉丝决定）。
- **规则更偏向评委评分？** 重点看 `p_elim_judge_bottom` 与 `expected_judge_rank_elim`（越接近 N 越由评委决定）。
- **加入‘评委救人’环节影响？** 重点看 `save_override_rate`：底二确定后，评委是否经常推翻‘综合分最低者’的淘汰决定。
- **是否能复刻历史淘汰？** 仍保留 `p_match_actual`/`reversal_rate`（注意：这不是题干唯一目标，只是校验口径一致性）。

## 2. 输出文件

- Overall 汇总：`derived_metrics_overall.csv`
- Season 汇总：`derived_metrics_by_season.csv`
- 机制差分表：`derived_pairwise_deltas.csv`

## 3. 机制差分（最贴题干的一张表）

下表给出 **A-B 的均值差**（正值表示 A 更高）：

| comparison                        | metric                   |   mean_delta |
|:----------------------------------|:-------------------------|-------------:|
| rank_minus_percent                | p_elim_judge_bottom      |     0.259859 |
| rank_minus_percent                | p_elim_fan_bottom        |    -0.301006 |
| rank_minus_percent                | expected_judge_rank_elim |     1.22399  |
| rank_minus_percent                | expected_fan_rank_elim   |    -0.69813  |
| rank_minus_percent                | save_override_rate       |   nan        |
| rank_save_minus_rank_direct       | p_elim_judge_bottom      |     0.200173 |
| rank_save_minus_rank_direct       | p_elim_fan_bottom        |    -0.140277 |
| rank_save_minus_rank_direct       | expected_judge_rank_elim |     0.352326 |
| rank_save_minus_rank_direct       | expected_fan_rank_elim   |    -0.750784 |
| rank_save_minus_rank_direct       | save_override_rate       |   nan        |
| percent_save_minus_percent_direct | p_elim_judge_bottom      |     0.180138 |
| percent_save_minus_percent_direct | p_elim_fan_bottom        |    -0.29195  |
| percent_save_minus_percent_direct | expected_judge_rank_elim |     0.817612 |
| percent_save_minus_percent_direct | expected_fan_rank_elim   |    -0.492805 |
| percent_save_minus_percent_direct | save_override_rate       |   nan        |

## 4. 写作提示（如何落到第二小问）

- 比较 **Rank vs Percent**：用 `rank_minus_percent` 这一行，看粉丝侧指标是否更大（更偏粉丝）以及评委侧指标是否更小（更弱评委）。
- 比较 **是否加入救人**：用 `rank_save_minus_rank_direct` / `percent_save_minus_percent_direct`，看 `save_override_rate` 是否显著>0，以及技术侧风险（如 `p_elim_judge_top`）是否下降。
- 个案（Jerry/Billy/Bristol/Bobby）：建议在正文里引用你们现有 tracker 曲线，同时补一句“该机制下他在某些周成为 bottom-2/被判淘汰的概率峰值”。