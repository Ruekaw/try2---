# -*- coding: utf-8 -*-
"""
Q3 回归配置
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class Q3Config:
    project_root: Path = Path(__file__).resolve().parents[1]

    dwts_long_path: Path = project_root / "outputs" / "dwts_long_clean.csv"
    fan_long_path: Path = project_root / "outputs" / "q1_mcmc" / "fan_vote_estimates_long.csv"
    npz_dir: Path = project_root / "Q1_data_expo"
    output_dir: Path = project_root / "outputs" / "q3_regression"

    k_candidates: Sequence[int] = (3, 4, 5)
    min_k_cover: float = 0.80
    min_survival_ratio: float = 0.75

    # K 灵敏度：对这些 K 分别拟合并输出“跨 K 汇总表”。
    # 当前只关心 K=4 的结果，其他 K 不再额外拟合。
    k_sensitivity_values: Sequence[int] = (4,)
    k_sensitivity_skip_existing: bool = True

    eps: float = 1e-6

    # MCMC (PyMC/Bambi)
    draws: int = 1200 #原先1000
    tune: int = 1200 #原先1000
    chains: int = 4
    target_accept: float = 0.92
    random_seed: int = 42

    # 进度显示
    # True: 显示 PyMC 内部采样进度条；False: 只显示外层整体进度（更清爽）
    pymc_progressbar: bool = True

    # fan 数据使用策略
    use_fan_mean: bool = True
    use_fan_samples: bool = True
    fan_sample_sets: int = 30 #原先50
    sample_seed: int = 20240202
