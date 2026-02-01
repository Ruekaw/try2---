# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C - Q1: MCMC粉丝投票反推模型
配置模块 (config.py)

包含所有可调参数、路径配置和常量定义
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class MCMCConfig:
    """MCMC采样配置参数"""
    
    # === 采样参数 ===
    n_samples: int = 5000          # 保留的有效样本数
    burn_in: int = 2000            # 预热期样本数（将丢弃）
    thin: int = 2                  # 稀疏采样间隔
    
    # === 提议分布参数 ===
    proposal_scale: float = 100.0  # Dirichlet提议浓度缩放参数 κ
    
    # === 先验参数 ===
    prior_alpha: float = 1.0       # Dirichlet先验浓度 α
                                   # α=1: 均匀先验（最大熵）
                                   # α>1: 偏好均匀分布
                                   # α<1: 偏好稀疏分布

    # === 时间平滑（跨周先验）===
    temporal_smoothing_enabled: bool = True  # 是否启用“上周后验作为本周先验”
    temporal_alpha: float = 40.0              # 惯性系数 α（越大越粘）
    temporal_beta: float = 1.0                # 平滑项 β（避免零先验）
    
    # === 约束参数 ===
    soft_elimination: bool = True  # 是否使用软约束（推荐True）
    # 按赛制自适应的 λ（默认启用，避免 percent 与 rank 的违约量纲不一致导致的“隐式硬约束”）
    violation_lambda_percent: float = 50.0  # 百分比制（S3-S27）：违约通常是 0.0x 量级
    violation_lambda_rank: float = 3.0      # 排名制（S1-S2, S28+）：违约通常是整数 1,2,...
    
    # === 初始化参数 ===
    max_init_attempts: int = 10000 # 寻找可行初始解的最大尝试次数
    init_fallback_soft: bool = True # 初始化失败时是否回退到软约束
    
    # === 评委救人机制（S28+）===
    judge_save_enabled: bool = True  # 是否启用评委救人规则
    judge_save_bottom_k: int = 2     # 评委从 bottom-k 中选择淘汰
    
    # === 一致性阈值 ===
    ppc_threshold: float = 0.95    # PPC一致性阈值
    
    # === 并行参数 ===
    n_jobs: int = -1               # 并行作业数，-1表示使用所有CPU
    
    # === 随机种子 ===
    random_seed: Optional[int] = 42


@dataclass
class PathConfig:
    """路径配置"""
    
    # 工作目录（自动检测或手动指定）
    workspace: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # 输入数据
    input_csv: str = "outputs/dwts_long_clean.csv"
    
    # 输出目录
    output_dir: str = "outputs/q1_mcmc"
    
    # 输出文件名
    output_wide: str = "fan_vote_estimates_wide.csv"
    output_long: str = "fan_vote_estimates_long.csv"
    output_diagnostics: str = "mcmc_diagnostics.csv"
    output_summary: str = "inference_summary.json"
    
    def get_input_path(self) -> Path:
        return self.workspace / self.input_csv
    
    def get_output_dir(self) -> Path:
        output = self.workspace / self.output_dir
        output.mkdir(parents=True, exist_ok=True)
        return output
    
    def get_output_path(self, filename: str) -> Path:
        return self.get_output_dir() / filename


@dataclass
class FilterConfig:
    """数据过滤配置"""
    
    # 用于筛选参与推断的周
    use_is_competing_week: bool = True
    use_exclude_flag: bool = True
    
    # 额外排除条件
    exclude_withdrawn: bool = True        # 排除退赛选手
    exclude_no_elimination: bool = False  # 是否排除无淘汰周（由数据标记处理）
    
    # 赛季范围（None表示全部）
    season_range: Optional[tuple] = None  # e.g., (1, 34)
    
    # 特定赛季排除
    exclude_seasons: List[int] = field(default_factory=list)  # e.g., [15] 排除全明星赛季


# === 常量定义 ===

# 结合方法常量
COMBINE_PERCENT = "percent"  # 百分比制（S3-S27）
COMBINE_RANK = "rank"        # 排名制（S1-S2, S28+）

# 赛季规则映射
SEASON_COMBINE_METHOD = {
    **{s: COMBINE_RANK for s in [1, 2]},
    **{s: COMBINE_PERCENT for s in range(3, 28)},
    **{s: COMBINE_RANK for s in range(28, 35)},
}

# 评委救人机制启用赛季
JUDGE_SAVE_SEASONS = set(range(28, 35))

# 结果类型常量
RESULT_ELIMINATED = "eliminated"
RESULT_WINNER = "winner"
RESULT_FINALIST = "finalist"
RESULT_WITHDREW = "withdrew"


def get_default_config() -> tuple:
    """获取默认配置"""
    return MCMCConfig(), PathConfig(), FilterConfig()


def get_combine_method(season: int) -> str:
    """根据赛季获取结合方法"""
    return SEASON_COMBINE_METHOD.get(season, COMBINE_PERCENT)


def is_judge_save_season(season: int) -> bool:
    """判断是否为评委救人赛季"""
    return season in JUDGE_SAVE_SEASONS


def get_violation_lambda(mcmc_config: MCMCConfig, combine_method: str) -> float:
    """根据赛制选择软约束惩罚强度 λ。

    percent 与 rank 的违约值量纲差异很大：
    - percent 违约通常是 0.01(=1%) 量级
    - rank 违约通常是整数 1,2,...
    因此默认使用“按赛制自适应”的两套 λ。
    """
    if combine_method == COMBINE_RANK:
        return float(mcmc_config.violation_lambda_rank)
    return float(mcmc_config.violation_lambda_percent)


if __name__ == "__main__":
    # 测试配置
    mcmc_cfg, path_cfg, filter_cfg = get_default_config()
    print("=== MCMC配置 ===")
    print(f"样本数: {mcmc_cfg.n_samples}")
    print(f"预热期: {mcmc_cfg.burn_in}")
    print(f"稀疏间隔: {mcmc_cfg.thin}")
    print("惩罚强度: 按赛制自适应")
    print(f"  percent λ: {mcmc_cfg.violation_lambda_percent}")
    print(f"  rank    λ: {mcmc_cfg.violation_lambda_rank}")
    print()
    print("=== 路径配置 ===")
    print(f"工作目录: {path_cfg.workspace}")
    print(f"输入文件: {path_cfg.get_input_path()}")
    print(f"输出目录: {path_cfg.get_output_dir()}")
    print()
    print("=== 赛季规则 ===")
    for s in [1, 2, 3, 27, 28, 34]:
        method = get_combine_method(s)
        judge_save = is_judge_save_season(s)
        print(f"Season {s}: {method}, 评委救人={judge_save}")
