# DAWS 推荐参数

说明：搜索输出保留完整网格（透明可追溯），但同时给出自动推荐参数，避免人工挑选。

## fairness（公平优先：最小化 tech_vulnerability）
- method=daws_save, save=True, w_early=0.4, w_late=0.0, n_cut=5

## audience（观众优先：最小化 popularity_vulnerability）
- method=daws_direct, save=False, w_early=0.4, w_late=0.2, n_cut=4

## balanced（折衷：min-max 归一化后加权求和，tech 权重更高）
- method=daws_save, save=True, w_early=0.4, w_late=0.2, n_cut=4

使用方式：选定其中一个 objective 的参数后，用 `q4_main.py --eval` 固定参数出最终图表；若需要 DAWS-Save 曲线请加 `--save`。
