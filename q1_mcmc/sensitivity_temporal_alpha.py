# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C - Q1: 时间平滑参数 α 的灵敏度分析脚本

选择若干代表性赛季，在不同 temporal_alpha 设置下比较总体指标：
- 平均 PPC 一致性
- 平均接受率
- 平均 Hit Rate
- 收敛率

运行方式（在项目根目录下）：

    python -m q1_mcmc.sensitivity_temporal_alpha

会在 outputs/q1_mcmc/ 下生成 CSV 汇总，并在终端打印简易表格。
"""

from pathlib import Path
from typing import List, Optional
import sys

import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 与 main.py 一致：将 q1_mcmc 目录加入 sys.path，便于使用 config/engine 等模块
sys.path.insert(0, str(Path(__file__).parent))

from config import MCMCConfig, PathConfig, FilterConfig
from engine import create_engine


def run_temporal_alpha_sensitivity(
    seasons: Optional[List[int]] = None,
    alpha_grid: Optional[List[Optional[float]]] = None,
    n_samples: int = 1000,
    burn_in: int = 500,
    thin: int = 2,
    seed: int = 42,
) -> pd.DataFrame:
    """在给定赛季和 α 网格上跑一轮灵敏度分析。

    Parameters
    ----------
    seasons : list[int] 或 None
        要分析的赛季列表；None 时使用默认经典赛季。
    alpha_grid : list[Optional[float]] 或 None
        temporal_alpha 取值列表；None 表示关闭时间平滑（使用标量先验）。
        默认: [None, 10.0, 40.0, 100.0]
    n_samples, burn_in, thin : int
        MCMC 采样配置（为节省时间，默认用较小样本）。
    seed : int
        随机种子。

    Returns
    -------
    pd.DataFrame
        每行对应 (season, alpha_setting) 的总体统计。
    """

    # 1. 选取代表性赛季：覆盖不同规则区间
    #   - 2: 早期排名制（无评委救人）
    #   - 9: 典型百分比制赛季
    #   - 15: 全明星赛季（节奏特殊）
    #   - 23: 后期百分比制代表
    #   - 29: 新版排名制 + 评委救人
    if seasons is None:
        seasons = [2, 9, 15, 23, 29]

    # 2. α 网格：None=关闭时间平滑，其余为不同粘性强度
    if alpha_grid is None:
        alpha_grid = [None, 10.0, 20.0, 40.0, 100.0]

    # 路径配置：工作目录为项目根目录
    workspace = Path(__file__).parent.parent
    path_cfg = PathConfig(workspace=workspace)

    records = []
    # 保存每个 (season, alpha) 的长表结果，用于画折线图
    results_long = {}
    # 为每个赛季选出若干“主要选手”（在基准 α 下 fan_vote_mean 均值最高的前几名）
    main_contestants = {}

    for season in seasons:
        for alpha in alpha_grid:
            temporal_enabled = alpha is not None

            mcmc_cfg = MCMCConfig(
                n_samples=n_samples,
                burn_in=burn_in,
                thin=thin,
                proposal_scale=100.0,
                prior_alpha=1.0,
                temporal_smoothing_enabled=temporal_enabled,
                temporal_alpha=float(alpha) if alpha is not None else 40.0,
                temporal_beta=1.0,
                soft_elimination=True,
                violation_lambda_percent=50.0,
                violation_lambda_rank=3.0,
                judge_save_enabled=True,
                random_seed=seed,
            )

            filt_cfg = FilterConfig(
                season_range=(season, season),
                exclude_seasons=[],
            )

            engine = create_engine(mcmc_cfg, path_cfg, filt_cfg)
            engine.load_data()
            # 只推断单个赛季
            season_result = engine.infer_season(season)
            engine.results = {season: season_result}

            # 保存长表结果（单赛季）供后续画折线图
            long_df = engine.to_long_dataframe()
            alpha_key = "none" if alpha is None else float(alpha)
            results_long[(season, alpha_key)] = long_df

            # 在基准 α（通常为 None，对应不启用时间平滑）下选出该赛季的主要选手
            if season not in main_contestants and alpha is None and not long_df.empty:
                avg_votes = (
                    long_df
                    .groupby("celebrity_name")["fan_vote_mean"]
                    .mean()
                    .sort_values(ascending=False)
                )
                main_contestants[season] = list(avg_votes.head(3).index)

            summary = engine._compute_summary()

            records.append(
                {
                    "season": season,
                    "alpha_mode": alpha_key,
                    "temporal_enabled": temporal_enabled,
                    "total_weeks_inferred": summary["total_weeks_inferred"],
                    "convergence_rate": summary["convergence_rate"],
                    "mean_ppc": summary["mean_ppc_consistency"],
                    "mean_acceptance_rate": summary["mean_acceptance_rate"],
                    "mean_hit_rate": summary["mean_hit_rate"],
                }
            )

    df = pd.DataFrame.from_records(records)

    out_dir = path_cfg.get_output_dir()
    out_path = out_dir / "temporal_alpha_sensitivity_summary.csv"
    df.to_csv(out_path, index=False)

    # 补充绘制主要选手的 fan_vote_mean 随周次变化折线图（不同 α 叠加）
    if HAS_MATPLOTLIB:
        _plot_fan_vote_trends(seasons, alpha_grid, results_long, main_contestants, out_dir)
    else:
        print("matplotlib 未安装，跳过 fan_vote_mean 折线图绘制。")

    print("=== Temporal alpha sensitivity summary ===")
    print(df.to_string(index=False))
    print(f"\n已保存到: {out_path}")

    return df


def _plot_fan_vote_trends(
    seasons: List[int],
    alpha_grid: List[Optional[float]],
    results_long,
    main_contestants,
    out_dir: Path,
) -> None:
    """为每个赛季的主要选手绘制 fan_vote_mean 随时间变化的折线图。

    每个图对应 (season, celebrity)，横轴为 week，纵轴为 fan_vote_mean，
    不同颜色表示不同 temporal_alpha 设置。
    """

    for season in seasons:
        names = main_contestants.get(season)
        if not names:
            continue

        for name in names:
            has_any = False
            plt.figure(figsize=(6, 4))

            for alpha in alpha_grid:
                alpha_key = "none" if alpha is None else float(alpha)
                long_df = results_long.get((season, alpha_key))
                if long_df is None or long_df.empty:
                    continue

                sub = long_df[long_df["celebrity_name"] == name].copy()
                if sub.empty:
                    continue

                sub = sub.sort_values("week")
                label = "none" if alpha is None else f"{alpha:.0f}"
                plt.plot(
                    sub["week"],
                    sub["fan_vote_mean"],
                    marker="o",
                    label=f"α={label}",
                )
                has_any = True

            if not has_any:
                plt.close()
                continue

            plt.xlabel("Week")
            plt.ylabel("Fan vote mean")
            plt.title(f"Season {season} - {name}")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()

            safe_name = "".join(
                c for c in name.replace(" ", "_") if c.isalnum() or c in "_"
            )
            out_file = out_dir / f"fan_vote_trend_s{season}_{safe_name}.png"
            plt.savefig(out_file, dpi=150)
            plt.close()
            print(f"已保存选手折线图: {out_file}")


if __name__ == "__main__":
    run_temporal_alpha_sensitivity()
