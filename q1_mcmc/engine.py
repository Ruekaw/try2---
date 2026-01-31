# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C - Q1: MCMC粉丝投票反推模型
推断引擎模块 (engine.py)

包含：
1. 数据加载与预处理
2. 周级推断调度
3. 赛季级并行处理
4. 结果汇总与导出
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
import warnings

from config import (
    MCMCConfig, PathConfig, FilterConfig,
    get_combine_method, is_judge_save_season,
    COMBINE_PERCENT, COMBINE_RANK
)
from sampler import UnifiedMCMCSampler, SamplingResult


@dataclass
class WeekResult:
    """单周推断结果"""
    season: int
    week: int
    combine_method: str
    is_finale: bool
    n_contestants: int
    contestant_names: List[str]
    eliminated_names: List[str]
    
    # 后验统计
    mean_votes: np.ndarray           # 后验均值
    std_votes: np.ndarray            # 后验标准差
    ci_lower: np.ndarray             # 95% CI 下界
    ci_upper: np.ndarray             # 95% CI 上界
    
    # 确定性指标
    certainty_index: np.ndarray      # λ = μ / (μ + σ)
    ci_width: np.ndarray             # Δ = Q97.5 - Q2.5
    
    # 一致性指标
    ppc_consistency: float           # PPC 一致性
    mean_violation: float            # 平均违约
    
    # 诊断
    acceptance_rate: float
    converged: bool
    init_attempts: int


@dataclass
class SeasonResult:
    """单赛季推断结果"""
    season: int
    combine_method: str
    week_results: Dict[int, WeekResult] = field(default_factory=dict)
    
    def add_week(self, week: int, result: WeekResult):
        self.week_results[week] = result
    
    @property
    def n_weeks(self) -> int:
        return len(self.week_results)
    
    @property
    def overall_ppc(self) -> float:
        if not self.week_results:
            return 0.0
        return np.mean([r.ppc_consistency for r in self.week_results.values()])


class UnifiedFanVoteInferenceEngine:
    """
    统一的粉丝投票推断引擎
    
    负责：
    1. 从清洗后的数据加载并筛选有效周
    2. 调度 MCMC 采样器进行逐周推断
    3. 汇总结果并导出
    """
    
    def __init__(
        self,
        mcmc_config: MCMCConfig,
        path_config: PathConfig,
        filter_config: FilterConfig
    ):
        self.mcmc_config = mcmc_config
        self.path_config = path_config
        self.filter_config = filter_config
        
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict[int, SeasonResult] = {}
    
    def load_data(self) -> pd.DataFrame:
        """加载并预处理数据"""
        input_path = self.path_config.get_input_path()
        
        if not input_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {input_path}")
        
        df = pd.read_csv(input_path)
        
        # 基本筛选
        if self.filter_config.use_exclude_flag and 'exclude_from_fan_vote_inference' in df.columns:
            df = df[df['exclude_from_fan_vote_inference'] == False].copy()
        
        if self.filter_config.exclude_withdrawn and 'is_withdrawn' in df.columns:
            df = df[df['is_withdrawn'] == False].copy()
        
        # 赛季范围筛选
        if self.filter_config.season_range:
            s_min, s_max = self.filter_config.season_range
            df = df[(df['season'] >= s_min) & (df['season'] <= s_max)].copy()
        
        # 排除特定赛季
        if self.filter_config.exclude_seasons:
            df = df[~df['season'].isin(self.filter_config.exclude_seasons)].copy()
        
        self.data = df
        return df
    
    def _get_week_data(
        self,
        season: int,
        week: int
    ) -> Tuple[pd.DataFrame, List[str], List[int], bool, Optional[List[int]]]:
        """
        获取某赛季某周的数据
        
        Returns:
            (周数据DataFrame, 选手名列表, 被淘汰索引, 是否决赛, 决赛名次列表)
        """
        df = self.data
        
        # 筛选本周数据
        week_df = df[(df['season'] == season) & (df['week'] == week)].copy()
        
        # 只保留本周参赛的选手
        if self.filter_config.use_is_competing_week and 'is_competing_week' in week_df.columns:
            week_df = week_df[week_df['is_competing_week'] == True].copy()
        else:
            # 备用：根据分数判断
            week_df = week_df[week_df['week_total_score'].notna() & (week_df['week_total_score'] > 0)].copy()
        
        if len(week_df) == 0:
            return week_df, [], [], False, None
        
        # 选手名列表（保持顺序）
        contestant_names = week_df['celebrity_name'].tolist()
        
        # 判断是否为决赛周
        is_finale = False
        placements = None
        
        if 'is_final_week' in week_df.columns:
            is_finale = week_df['is_final_week'].iloc[0] == True
        elif 'final_week' in week_df.columns and 'week' in week_df.columns:
            final_week = week_df['final_week'].iloc[0]
            is_finale = (week == final_week)
        
        if is_finale:
            # 获取决赛选手的名次
            placements = week_df['placement'].tolist()
        
        # 确定被淘汰选手
        eliminated_indices = []
        
        if is_finale:
            # 决赛周：不使用淘汰逻辑，而是用名次约束
            eliminated_indices = []
        else:
            # 普通周：根据 result_type 或其他标记确定被淘汰者
            # 被淘汰者是 last_week_scored == week 且 result_type == 'eliminated'
            for i, row in week_df.iterrows():
                last_week = row.get('last_week_scored', row.get('computed_last_week'))
                result_type = row.get('result_type', '')
                
                if last_week == week and result_type == 'eliminated':
                    idx = contestant_names.index(row['celebrity_name'])
                    eliminated_indices.append(idx)
        
        return week_df, contestant_names, eliminated_indices, is_finale, placements
    
    def _infer_week(
        self,
        season: int,
        week: int
    ) -> Optional[WeekResult]:
        """
        对单周执行 MCMC 推断
        """
        week_df, contestant_names, eliminated_indices, is_finale, placements = \
            self._get_week_data(season, week)
        
        if len(week_df) < 2:
            return None
        
        # 跳过无淘汰且非决赛的周
        if not is_finale and len(eliminated_indices) == 0:
            # 检查是否为无淘汰周
            if 'is_no_elimination_any' in week_df.columns:
                if week_df['is_no_elimination_any'].iloc[0]:
                    return None  # 无淘汰周，跳过
        
        # 获取评委总分
        judge_scores = week_df['week_total_score'].values.astype(float)
        
        # 获取结合方法
        combine_method = get_combine_method(season)
        
        # 检查是否启用评委救人
        judge_save = is_judge_save_season(season) and self.mcmc_config.judge_save_enabled
        
        # 创建采样器并执行
        sampler = UnifiedMCMCSampler(self.mcmc_config)
        result = sampler.sample(
            judge_scores=judge_scores,
            eliminated_indices=eliminated_indices,
            combine_method=combine_method,
            is_finale=is_finale,
            placements=placements,
            judge_save_enabled=judge_save
        )
        
        if not result.converged:
            warnings.warn(f"Season {season} Week {week}: MCMC未收敛")
        
        # 计算后验统计
        samples = result.samples
        mean_votes = np.mean(samples, axis=0)
        std_votes = np.std(samples, axis=0)
        ci_lower = np.percentile(samples, 2.5, axis=0)
        ci_upper = np.percentile(samples, 97.5, axis=0)
        
        # 确定性指标
        certainty_index = mean_votes / (mean_votes + std_votes + 1e-10)
        ci_width = ci_upper - ci_lower
        
        # 一致性指标
        if result.violation_history is not None:
            ppc = np.mean(result.violation_history == 0)
            mean_viol = np.mean(result.violation_history)
        else:
            ppc = 0.0
            mean_viol = np.nan
        
        eliminated_names = [contestant_names[i] for i in eliminated_indices]
        
        return WeekResult(
            season=season,
            week=week,
            combine_method=combine_method,
            is_finale=is_finale,
            n_contestants=len(contestant_names),
            contestant_names=contestant_names,
            eliminated_names=eliminated_names,
            mean_votes=mean_votes,
            std_votes=std_votes,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            certainty_index=certainty_index,
            ci_width=ci_width,
            ppc_consistency=ppc,
            mean_violation=mean_viol,
            acceptance_rate=result.acceptance_rate,
            converged=result.converged,
            init_attempts=result.init_attempts
        )
    
    def infer_season(self, season: int) -> SeasonResult:
        """
        对单赛季执行推断
        """
        combine_method = get_combine_method(season)
        season_result = SeasonResult(season=season, combine_method=combine_method)
        
        # 获取该赛季的所有周
        season_df = self.data[self.data['season'] == season]
        weeks = sorted(season_df['week'].unique())
        
        for week in weeks:
            week_result = self._infer_week(season, week)
            if week_result is not None:
                season_result.add_week(week, week_result)
        
        return season_result
    
    def infer_all(self, n_jobs: Optional[int] = None) -> Dict[int, SeasonResult]:
        """
        对所有赛季执行推断
        
        Args:
            n_jobs: 并行作业数（暂时使用串行以保证稳定性）
        """
        if self.data is None:
            self.load_data()
        
        seasons = sorted(self.data['season'].unique())
        
        print(f"开始推断 {len(seasons)} 个赛季...")
        
        # 目前使用串行处理（MCMC 已经较快）
        for season in tqdm(seasons, desc="赛季进度"):
            season_result = self.infer_season(season)
            self.results[season] = season_result
        
        return self.results
    
    def to_long_dataframe(self) -> pd.DataFrame:
        """
        将结果转换为长格式 DataFrame
        """
        records = []
        
        for season, season_result in self.results.items():
            for week, week_result in season_result.week_results.items():
                for i, name in enumerate(week_result.contestant_names):
                    records.append({
                        'season': season,
                        'week': week,
                        'celebrity_name': name,
                        'combine_method': week_result.combine_method,
                        'is_finale': week_result.is_finale,
                        'is_eliminated': name in week_result.eliminated_names,
                        'fan_vote_mean': week_result.mean_votes[i],
                        'fan_vote_std': week_result.std_votes[i],
                        'fan_vote_ci_lower': week_result.ci_lower[i],
                        'fan_vote_ci_upper': week_result.ci_upper[i],
                        'certainty_index': week_result.certainty_index[i],
                        'ci_width': week_result.ci_width[i],
                        'ppc_consistency': week_result.ppc_consistency,
                        'mean_violation': week_result.mean_violation,
                        'acceptance_rate': week_result.acceptance_rate,
                        'converged': week_result.converged
                    })
        
        return pd.DataFrame(records)
    
    def to_wide_dataframe(self) -> pd.DataFrame:
        """
        将结果转换为宽格式 DataFrame
        每行一个选手，包含所有周的粉丝投票估计
        """
        long_df = self.to_long_dataframe()
        
        # Pivot to wide format
        wide_df = long_df.pivot_table(
            index=['season', 'celebrity_name'],
            columns='week',
            values='fan_vote_mean',
            aggfunc='first'
        ).reset_index()
        
        # Rename columns
        wide_df.columns = [
            f'week{col}_fan_vote' if isinstance(col, (int, float)) else col
            for col in wide_df.columns
        ]
        
        return wide_df
    
    def export_results(self):
        """
        导出所有结果
        """
        output_dir = self.path_config.get_output_dir()
        
        # 长格式
        long_df = self.to_long_dataframe()
        long_path = output_dir / self.path_config.output_long
        long_df.to_csv(long_path, index=False)
        print(f"已导出长格式结果: {long_path}")
        
        # 宽格式
        wide_df = self.to_wide_dataframe()
        wide_path = output_dir / self.path_config.output_wide
        wide_df.to_csv(wide_path, index=False)
        print(f"已导出宽格式结果: {wide_path}")
        
        # 汇总统计
        summary = self._compute_summary()
        summary_path = output_dir / self.path_config.output_summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"已导出汇总统计: {summary_path}")
        
        return long_df, wide_df, summary
    
    def _compute_summary(self) -> Dict[str, Any]:
        """
        计算汇总统计
        """
        all_ppc = []
        all_acceptance = []
        total_weeks = 0
        converged_weeks = 0
        
        for season_result in self.results.values():
            for week_result in season_result.week_results.values():
                all_ppc.append(week_result.ppc_consistency)
                all_acceptance.append(week_result.acceptance_rate)
                total_weeks += 1
                if week_result.converged:
                    converged_weeks += 1
        
        return {
            'total_seasons': len(self.results),
            'total_weeks_inferred': total_weeks,
            'converged_weeks': converged_weeks,
            'convergence_rate': converged_weeks / max(total_weeks, 1),
            'mean_ppc_consistency': float(np.mean(all_ppc)) if all_ppc else 0.0,
            'std_ppc_consistency': float(np.std(all_ppc)) if all_ppc else 0.0,
            'mean_acceptance_rate': float(np.mean(all_acceptance)) if all_acceptance else 0.0,
            'config': {
                'n_samples': self.mcmc_config.n_samples,
                'burn_in': self.mcmc_config.burn_in,
                'thin': self.mcmc_config.thin,
                'violation_lambda': self.mcmc_config.violation_lambda,
                'soft_elimination': self.mcmc_config.soft_elimination
            }
        }


def create_engine(
    mcmc_config: Optional[MCMCConfig] = None,
    path_config: Optional[PathConfig] = None,
    filter_config: Optional[FilterConfig] = None
) -> UnifiedFanVoteInferenceEngine:
    """
    创建推断引擎实例
    """
    if mcmc_config is None:
        mcmc_config = MCMCConfig()
    if path_config is None:
        path_config = PathConfig()
    if filter_config is None:
        filter_config = FilterConfig()
    
    return UnifiedFanVoteInferenceEngine(mcmc_config, path_config, filter_config)


if __name__ == "__main__":
    # 测试推断引擎
    from config import MCMCConfig, PathConfig, FilterConfig
    
    # 使用较小的样本数进行测试
    mcmc_cfg = MCMCConfig(
        n_samples=500,
        burn_in=200,
        thin=1,
        random_seed=42
    )
    
    path_cfg = PathConfig()
    filter_cfg = FilterConfig(season_range=(1, 3))  # 只测试前3个赛季
    
    engine = create_engine(mcmc_cfg, path_cfg, filter_cfg)
    
    try:
        engine.load_data()
        print(f"已加载数据: {len(engine.data)} 行")
        
        # 测试单周推断
        print("\n测试 Season 1 Week 2...")
        result = engine._infer_week(1, 2)
        if result:
            print(f"选手: {result.contestant_names}")
            print(f"被淘汰: {result.eliminated_names}")
            print(f"后验均值: {result.mean_votes}")
            print(f"PPC一致性: {result.ppc_consistency:.3f}")
        
    except FileNotFoundError as e:
        print(f"测试跳过: {e}")
