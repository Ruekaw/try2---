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
from datetime import datetime

from config import (
    MCMCConfig, PathConfig, FilterConfig,
    get_combine_method, is_judge_save_season,
    COMBINE_PERCENT, COMBINE_RANK,
    get_violation_lambda
)
from sampler import UnifiedMCMCSampler, SamplingResult
from diagnostics import compute_hit_rate_for_week, compute_diagnostics, diagnose_convergence


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
    judge_scores: np.ndarray
    
    # 后验统计
    mean_votes: np.ndarray           # 后验均值
    std_votes: np.ndarray            # 后验标准差
    ci_lower: np.ndarray             # 95% CI 下界
    ci_upper: np.ndarray             # 95% CI 上界
    
    # 确定性指标（信噪比）
    snr: np.ndarray                  # SNR = μ / σ
    ci_width: np.ndarray             # Δ = Q97.5 - Q2.5
    
    # 一致性指标
    ppc_consistency: float           # PPC 一致性
    mean_violation: float            # 平均违约
    hit_rate: float                  # Hit Rate 命中率
    elimination_count: int           # 实际淘汰人数
    
    # 诊断
    acceptance_rate: float
    converged: bool
    init_attempts: int

    # 原始样本（可选）
    samples: Optional[np.ndarray] = None


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


# === 公共工具函数（去重，确保串行/并行一致）===

def _prepare_week_inputs(
    season: int,
    week: int,
    df: pd.DataFrame,
    filter_config: FilterConfig
) -> Optional[Tuple[pd.DataFrame, List[str], List[int], bool, Optional[List[int]]]]:
    """
    统一的“取本周数据 + 识别决赛/淘汰 + 跳过规则”逻辑。
    返回 None 表示该周无需推断/数据不足。
    """
    if 'season' in df.columns:
        week_df = df[(df['season'] == season) & (df['week'] == week)].copy()
    else:
        week_df = df[df['week'] == week].copy()

    # 只保留本周参赛的选手
    if filter_config.use_is_competing_week and 'is_competing_week' in week_df.columns:
        week_df = week_df[week_df['is_competing_week'] == True].copy()
    else:
        week_df = week_df[week_df['week_total_score'].notna() & (week_df['week_total_score'] > 0)].copy()

    if len(week_df) < 2:
        return None

    contestant_names = week_df['celebrity_name'].tolist()
    name_to_idx = {n: i for i, n in enumerate(contestant_names)}

    # 判断是否为决赛周
    is_finale = False
    placements = None

    if 'is_final_week' in week_df.columns:
        is_finale = week_df['is_final_week'].iloc[0] == True
    elif 'final_week' in week_df.columns:
        final_week = week_df['final_week'].iloc[0]
        is_finale = (week == final_week)

    if is_finale and 'placement' in week_df.columns:
        placements = week_df['placement'].tolist()

    # 确定被淘汰选手
    eliminated_indices: List[int] = []
    if not is_finale:
        for _, row in week_df.iterrows():
            last_week = row.get('last_week_scored', row.get('computed_last_week'))
            result_type = row.get('result_type', '')

            if last_week == week and result_type == 'eliminated':
                nm = row['celebrity_name']
                if nm in name_to_idx:
                    eliminated_indices.append(name_to_idx[nm])

    # 跳过无淘汰且非决赛的周
    if (not is_finale) and len(eliminated_indices) == 0:
        if 'is_no_elimination_any' in week_df.columns:
            if week_df['is_no_elimination_any'].iloc[0]:
                return None

    return week_df, contestant_names, eliminated_indices, is_finale, placements


def _build_temporal_prior(
    contestant_names: List[str],
    prev_names: Optional[List[str]],
    prev_mean_votes: Optional[np.ndarray],
    alpha: float,
    beta: float
) -> np.ndarray:
    """
    构造跨周先验：Dirichlet(alpha * v_{t-1} + beta)

    新加入/复活选手使用 1/n 作为上一周均值。
    """
    n = len(contestant_names)
    if n == 0:
        return np.array([])

    base = np.full(n, 1.0 / n, dtype=float)
    if prev_names and prev_mean_votes is not None and len(prev_names) == len(prev_mean_votes):
        prev_map = {name: float(prev_mean_votes[i]) for i, name in enumerate(prev_names)}
        for i, name in enumerate(contestant_names):
            if name in prev_map:
                base[i] = prev_map[name]

    s = float(np.sum(base))
    if s <= 0:
        base[:] = 1.0 / n
    else:
        base /= s

    return alpha * base + beta


def _infer_week_core(
    season: int,
    week: int,
    week_df: pd.DataFrame,
    contestant_names: List[str],
    eliminated_indices: List[int],
    is_finale: bool,
    placements: Optional[List[int]],
    mcmc_config: MCMCConfig,
    prior_alpha_override: Optional[np.ndarray] = None,
    survivor_indices_next: Optional[List[int]] = None,
    keep_samples: bool = False
) -> WeekResult:
    """统一的单周推断核心（串行/并行共用）。"""
    judge_scores = week_df['week_total_score'].values.astype(float)

    combine_method = get_combine_method(season)
    effective_lambda = get_violation_lambda(mcmc_config, combine_method)

    judge_save = is_judge_save_season(season) and mcmc_config.judge_save_enabled

    sampler = UnifiedMCMCSampler(mcmc_config)
    result = sampler.sample(
        judge_scores=judge_scores,
        eliminated_indices=eliminated_indices,
        combine_method=combine_method,
        is_finale=is_finale,
        placements=placements,
        judge_save_enabled=judge_save,
        survivor_indices=survivor_indices_next,
        prior_alpha_override=prior_alpha_override,
        violation_lambda_override=effective_lambda
    )

    # === 采样质量/收敛诊断 ===
    # sampler.sample() 当前只要跑完就会返回 converged=True（除非初始化失败）。
    # 这里用 ESS + 接受率 做真实的“likely_converged”判定，并把 WeekResult.converged 写成该判定。
    week_converged = bool(result.converged)
    if week_converged:
        try:
            # 为避免 ESS 计算在全量样本上过慢，限制诊断样本点数（最多约2000点）。
            diag_max_points = 2000
            stride = max(1, int(np.ceil(result.samples.shape[0] / diag_max_points)))
            diag_samples = result.samples[::stride]

            diag = compute_diagnostics(diag_samples, acceptance_rate=result.acceptance_rate)
            conv = diagnose_convergence(
                diag.ess,
                acceptance_rate=result.acceptance_rate,
                n_samples=int(diag_samples.shape[0]),
                ess_threshold=100,
                accept_range=(0.1, 0.8)
            )
            week_converged = bool(conv.get('likely_converged', False))
        except Exception as e:
            # 诊断失败不应打断推断；保留 sampler 的状态并告警。
            warnings.warn(f"Season {season} Week {week}: 收敛诊断失败: {e}")

    if not week_converged:
        warnings.warn(f"Season {season} Week {week}: MCMC采样质量不足（converged=False）")

    samples = result.samples
    mean_votes = np.mean(samples, axis=0)
    std_votes = np.std(samples, axis=0)
    ci_lower = np.percentile(samples, 2.5, axis=0)
    ci_upper = np.percentile(samples, 97.5, axis=0)

    snr = mean_votes / (std_votes + 1e-10)
    ci_width = ci_upper - ci_lower

    if result.violation_history is not None:
        ppc = float(np.mean(result.violation_history == 0))
        mean_viol = float(np.mean(result.violation_history))
    else:
        ppc = 0.0
        mean_viol = float('nan')

    eliminated_names = [contestant_names[i] for i in eliminated_indices]
    elimination_count = len(eliminated_indices)

    if is_finale or elimination_count == 0:
        hit_rate = float('nan')
    else:
        hit_rate = compute_hit_rate_for_week(
            mean_votes,
            judge_scores,
            eliminated_indices,
            combine_method,
            judge_save_enabled=judge_save,
            judge_save_bottom_k=mcmc_config.judge_save_bottom_k
        )

    return WeekResult(
        season=season,
        week=week,
        combine_method=combine_method,
        is_finale=is_finale,
        n_contestants=len(contestant_names),
        contestant_names=contestant_names,
        eliminated_names=eliminated_names,
        judge_scores=judge_scores,
        samples=samples if keep_samples else None,
        mean_votes=mean_votes,
        std_votes=std_votes,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        snr=snr,
        ci_width=ci_width,
        ppc_consistency=ppc,
        mean_violation=mean_viol,
        hit_rate=hit_rate,
        elimination_count=elimination_count,
        acceptance_rate=result.acceptance_rate,
        converged=week_converged,
        init_attempts=result.init_attempts
    )


def _infer_season_core(
    season: int,
    season_df: pd.DataFrame,
    mcmc_config: MCMCConfig,
    filter_config: FilterConfig,
    export_samples_dir: Optional[Path] = None,
    keep_samples: bool = False
) -> SeasonResult:
    """统一的赛季推断核心（串行/并行共用）。"""
    combine_method = get_combine_method(season)
    season_result = SeasonResult(season=season, combine_method=combine_method)

    export_payload: Dict[int, Dict[str, Any]] = {}

    weeks = sorted(season_df['week'].unique())
    prev_names: Optional[List[str]] = None
    prev_mean_votes: Optional[np.ndarray] = None
    for week in weeks:
        prepared = _prepare_week_inputs(season, week, season_df, filter_config)
        if prepared is None:
            continue
        week_df, contestant_names, eliminated_indices, is_finale, placements = prepared

        # 计算本周之后仍在场的选手索引，作为“可能被救的人”（评委救人约束用）
        survivor_indices_next: Optional[List[int]] = None
        if not is_finale and len(eliminated_indices) > 0:
            name_to_idx = {n: i for i, n in enumerate(contestant_names)}

            future_df = season_df[season_df['week'] > week].copy()
            if filter_config.use_is_competing_week and 'is_competing_week' in future_df.columns:
                future_df = future_df[future_df['is_competing_week'] == True].copy()
            else:
                future_df = future_df[future_df['week_total_score'].notna() & (future_df['week_total_score'] > 0)].copy()

            if len(future_df) > 0:
                survivor_names = set(future_df['celebrity_name'].tolist())
                survivor_indices_next = [
                    name_to_idx[nm]
                    for nm in contestant_names
                    if nm in survivor_names
                ] or None

        prior_override = None
        if mcmc_config.temporal_smoothing_enabled and prev_mean_votes is not None:
            prior_override = _build_temporal_prior(
                contestant_names,
                prev_names,
                prev_mean_votes,
                mcmc_config.temporal_alpha,
                mcmc_config.temporal_beta
            )

        week_result = _infer_week_core(
            season,
            week,
            week_df,
            contestant_names,
            eliminated_indices,
            is_finale,
            placements,
            mcmc_config,
            prior_alpha_override=prior_override,
            survivor_indices_next=survivor_indices_next,
            keep_samples=keep_samples or export_samples_dir is not None
        )
        if week_result is not None:
            season_result.add_week(week, week_result)
            prev_names = week_result.contestant_names
            prev_mean_votes = week_result.mean_votes

            if export_samples_dir is not None and week_result.samples is not None:
                export_payload[week] = {
                    'samples': week_result.samples,
                    'contestant_names': week_result.contestant_names,
                    'eliminated_names': week_result.eliminated_names,
                    'is_finale': week_result.is_finale,
                    'placements': placements if placements is not None else [],
                    'combine_method': week_result.combine_method,
                    'judge_scores': week_result.judge_scores
                }
                if not keep_samples:
                    week_result.samples = None

    if export_samples_dir is not None and export_payload:
        _export_season_samples_npz(
            season=season,
            combine_method=combine_method,
            export_dir=export_samples_dir,
            mcmc_config=mcmc_config,
            payload=export_payload
        )

    return season_result


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
    
    def _infer_week(
        self,
        season: int,
        week: int
    ) -> Optional[WeekResult]:
        """
        对单周执行 MCMC 推断
        """
        season_df = self.data[self.data['season'] == season]
        prepared = _prepare_week_inputs(season, week, season_df, self.filter_config)
        if prepared is None:
            return None
        week_df, contestant_names, eliminated_indices, is_finale, placements = prepared
        
        survivor_indices_next: Optional[List[int]] = None
        if not is_finale and len(eliminated_indices) > 0:
            name_to_idx = {n: i for i, n in enumerate(contestant_names)}

            future_df = season_df[season_df['week'] > week].copy()
            if self.filter_config.use_is_competing_week and 'is_competing_week' in future_df.columns:
                future_df = future_df[future_df['is_competing_week'] == True].copy()
            else:
                future_df = future_df[future_df['week_total_score'].notna() & (future_df['week_total_score'] > 0)].copy()

            if len(future_df) > 0:
                survivor_names = set(future_df['celebrity_name'].tolist())
                survivor_indices_next = [
                    name_to_idx[nm]
                    for nm in contestant_names
                    if nm in survivor_names
                ] or None
        return _infer_week_core(
            season,
            week,
            week_df,
            contestant_names,
            eliminated_indices,
            is_finale,
            placements,
            self.mcmc_config,
            survivor_indices_next=survivor_indices_next
        )
    
    def infer_season(self, season: int) -> SeasonResult:
        """
        对单赛季执行推断
        """
        season_df = self.data[self.data['season'] == season]
        return _infer_season_core(season, season_df, self.mcmc_config, self.filter_config)
    
    def infer_all(
        self,
        n_jobs: Optional[int] = None,
        use_parallel: bool = True,
        export_samples_dir: Optional[Path] = None,
        keep_samples: bool = False
    ) -> Dict[int, SeasonResult]:
        """
        对所有赛季执行推断（支持并行）
        
        Args:
            n_jobs: 并行作业数，None 表示使用 CPU核心数-1，1 表示串行
            use_parallel: 是否启用并行处理
        
        Returns:
            Dict[season, SeasonResult]
        """
        if self.data is None:
            self.load_data()
        
        seasons = sorted(self.data['season'].unique())
        n_seasons = len(seasons)
        
        # 确定并行进程数
        import os
        if n_jobs is None:
            cpu = os.cpu_count() or 1
            n_jobs = max(1, cpu - 1)
        n_jobs = min(n_jobs, n_seasons)
        
        # 决定是否使用并行
        if export_samples_dir is not None and not isinstance(export_samples_dir, Path):
            export_samples_dir = Path(export_samples_dir)

        if not use_parallel or n_jobs <= 1 or n_seasons <= 2:
            # 串行模式
            print(f"开始推断 {n_seasons} 个赛季（串行模式）...")
            for season in tqdm(seasons, desc="赛季进度"):
                season_df = self.data[self.data['season'] == season]
                season_result = _infer_season_core(
                    season,
                    season_df,
                    self.mcmc_config,
                    self.filter_config,
                    export_samples_dir=export_samples_dir,
                    keep_samples=keep_samples
                )
                self.results[season] = season_result
        else:
            # 并行模式
            print(f"开始推断 {n_seasons} 个赛季（并行模式，{n_jobs} 进程）...")
            self._infer_all_parallel(seasons, n_jobs, export_samples_dir, keep_samples)
        
        return self.results
    
    def _infer_all_parallel(
        self,
        seasons: List[int],
        n_jobs: int,
        export_samples_dir: Optional[Path] = None,
        keep_samples: bool = False
    ):
        """
        并行推断所有赛季
        
        使用 ProcessPoolExecutor 实现多进程并行
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # 准备各赛季需要的数据子集（避免传递整个 DataFrame）
        season_data_dict = {}
        for season in seasons:
            season_df = self.data[self.data['season'] == season].copy()
            season_data_dict[season] = season_df
        
        # 并行执行
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # 提交所有任务
            future_to_season = {
                executor.submit(
                    _infer_season_standalone,
                    season,
                    season_data_dict[season],
                    self.mcmc_config,
                    self.filter_config,
                    export_samples_dir,
                    keep_samples
                ): season
                for season in seasons
            }
            
            # 收集结果（带进度条）
            with tqdm(total=len(seasons), desc="赛季进度") as pbar:
                for future in as_completed(future_to_season):
                    season = future_to_season[future]
                    try:
                        season_result = future.result()
                        self.results[season] = season_result
                    except Exception as e:
                        warnings.warn(f"Season {season} 推断失败: {e}")
                    pbar.update(1)
    
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
                        'snr': week_result.snr[i],
                        'ci_width': week_result.ci_width[i],
                        'ppc_consistency': week_result.ppc_consistency,
                        'hit_rate': week_result.hit_rate,
                        'elimination_count': week_result.elimination_count,
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

        def _fallback_path(path: Path) -> Path:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return path.with_name(f"{path.stem}_{ts}{path.suffix}")
        
        # 长格式
        long_df = self.to_long_dataframe()
        long_path = output_dir / self.path_config.output_long
        try:
            long_df.to_csv(long_path, index=False)
            print(f"已导出长格式结果: {long_path}")
        except PermissionError:
            alt = _fallback_path(long_path)
            long_df.to_csv(alt, index=False)
            warnings.warn(f"无法写入文件（可能被占用）: {long_path}，已改写到: {alt}")
        
        # 宽格式
        wide_df = self.to_wide_dataframe()
        wide_path = output_dir / self.path_config.output_wide
        try:
            wide_df.to_csv(wide_path, index=False)
            print(f"已导出宽格式结果: {wide_path}")
        except PermissionError:
            alt = _fallback_path(wide_path)
            wide_df.to_csv(alt, index=False)
            warnings.warn(f"无法写入文件（可能被占用）: {wide_path}，已改写到: {alt}")
        
        # 汇总统计
        summary = self._compute_summary()
        summary_path = output_dir / self.path_config.output_summary
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"已导出汇总统计: {summary_path}")
        except PermissionError:
            alt = _fallback_path(summary_path)
            with open(alt, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            warnings.warn(f"无法写入文件（可能被占用）: {summary_path}，已改写到: {alt}")
        
        return long_df, wide_df, summary

    def export_samples_npz(self, export_dir: Optional[Path] = None):
        """
        将当前内存中的样本导出为按赛季的 .npz 文件
        仅当 WeekResult.samples 已保留时有效
        """
        if export_dir is None:
            export_dir = self.path_config.get_output_dir()
        if not isinstance(export_dir, Path):
            export_dir = Path(export_dir)

        for season, season_result in self.results.items():
            payload: Dict[int, Dict[str, Any]] = {}
            for week, week_result in season_result.week_results.items():
                if week_result.samples is None:
                    continue
                payload[week] = {
                    'samples': week_result.samples,
                    'contestant_names': week_result.contestant_names,
                    'eliminated_names': week_result.eliminated_names,
                    'is_finale': week_result.is_finale,
                    'placements': [],
                    'combine_method': week_result.combine_method,
                    'judge_scores': week_result.judge_scores
                }
            if payload:
                _export_season_samples_npz(
                    season=season,
                    combine_method=season_result.combine_method,
                    export_dir=export_dir,
                    mcmc_config=self.mcmc_config,
                    payload=payload
                )
    
    def _compute_summary(self) -> Dict[str, Any]:
        """
        计算汇总统计
        """
        all_ppc: List[float] = []
        all_acceptance: List[float] = []
        all_hit_rates: List[float] = []
        total_weeks = 0
        converged_weeks = 0

        for season_result in self.results.values():
            for week_result in season_result.week_results.values():
                all_ppc.append(week_result.ppc_consistency)
                all_acceptance.append(week_result.acceptance_rate)
                total_weeks += 1
                if week_result.converged:
                    converged_weeks += 1
                if not np.isnan(week_result.hit_rate):
                    all_hit_rates.append(week_result.hit_rate)

        return {
            'total_seasons': len(self.results),
            'total_weeks_inferred': total_weeks,
            'converged_weeks': converged_weeks,
            'convergence_rate': converged_weeks / max(total_weeks, 1),
            'mean_ppc_consistency': float(np.mean(all_ppc)) if all_ppc else 0.0,
            'std_ppc_consistency': float(np.std(all_ppc)) if all_ppc else 0.0,
            'mean_acceptance_rate': float(np.mean(all_acceptance)) if all_acceptance else 0.0,
            'mean_hit_rate': float(np.mean(all_hit_rates)) if all_hit_rates else 0.0,
            'std_hit_rate': float(np.std(all_hit_rates)) if all_hit_rates else 0.0,
            'config': {
                'n_samples': self.mcmc_config.n_samples,
                'burn_in': self.mcmc_config.burn_in,
                'thin': self.mcmc_config.thin,
                'violation_lambda_percent': self.mcmc_config.violation_lambda_percent,
                'violation_lambda_rank': self.mcmc_config.violation_lambda_rank,
                'soft_elimination': self.mcmc_config.soft_elimination
            }
        }


# === 独立函数（用于多进程并行）===

def _infer_season_standalone(
    season: int,
    season_df: pd.DataFrame,
    mcmc_config: MCMCConfig,
    filter_config: FilterConfig,
    export_samples_dir: Optional[Path] = None,
    keep_samples: bool = False
) -> SeasonResult:
    """
    独立的赛季推断函数（用于多进程）
    
    这个函数必须是模块级别的（不能是类方法），才能被 pickle 序列化
    """
    return _infer_season_core(
        season,
        season_df,
        mcmc_config,
        filter_config,
        export_samples_dir=export_samples_dir,
        keep_samples=keep_samples
    )


def _infer_week_standalone(
    season: int,
    week: int,
    season_df: pd.DataFrame,
    mcmc_config: MCMCConfig,
    filter_config: FilterConfig
) -> Optional[WeekResult]:
    """
    独立的单周推断函数（用于多进程）
    """
    prepared = _prepare_week_inputs(season, week, season_df, filter_config)
    if prepared is None:
        return None
    week_df, contestant_names, eliminated_indices, is_finale, placements = prepared

    survivor_indices_next: Optional[List[int]] = None
    if not is_finale and len(eliminated_indices) > 0:
        name_to_idx = {n: i for i, n in enumerate(contestant_names)}

        future_df = season_df[season_df['week'] > week].copy()
        if filter_config.use_is_competing_week and 'is_competing_week' in future_df.columns:
            future_df = future_df[future_df['is_competing_week'] == True].copy()
        else:
            future_df = future_df[future_df['week_total_score'].notna() & (future_df['week_total_score'] > 0)].copy()

        if len(future_df) > 0:
            survivor_names = set(future_df['celebrity_name'].tolist())
            survivor_indices_next = [
                name_to_idx[nm]
                for nm in contestant_names
                if nm in survivor_names
            ] or None
    return _infer_week_core(
        season,
        week,
        week_df,
        contestant_names,
        eliminated_indices,
        is_finale,
        placements,
        mcmc_config,
        survivor_indices_next=survivor_indices_next
    )


def _export_season_samples_npz(
    season: int,
    combine_method: str,
    export_dir: Path,
    mcmc_config: MCMCConfig,
    payload: Dict[int, Dict[str, Any]]
):
    """
    将单赛季样本导出为 .npz
    payload: week -> {samples, contestant_names, eliminated_names, is_finale, placements, combine_method, judge_scores}
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / f"season_{season}_samples.npz"

    weeks = sorted(payload.keys())
    data: Dict[str, Any] = {
        'weeks': np.array(weeks, dtype=int)
    }

    for wk in weeks:
        wk_payload = payload[wk]
        data[f"week_{wk}_samples"] = wk_payload['samples']
        data[f"week_{wk}_contestants"] = np.array(wk_payload['contestant_names'], dtype=object)
        data[f"week_{wk}_eliminated"] = np.array(wk_payload['eliminated_names'], dtype=object)
        data[f"week_{wk}_is_finale"] = np.array(wk_payload['is_finale'], dtype=bool)
        data[f"week_{wk}_placements"] = np.array(wk_payload['placements'], dtype=object)
        data[f"week_{wk}_combine_method"] = np.array(wk_payload['combine_method'], dtype=object)
        data[f"week_{wk}_judge_scores"] = np.array(wk_payload['judge_scores'], dtype=float)

    meta = {
        'season': int(season),
        'combine_method': str(combine_method),
        'export_time': datetime.now().isoformat(timespec='seconds'),
        'mcmc_config': {
            'n_samples': int(mcmc_config.n_samples),
            'burn_in': int(mcmc_config.burn_in),
            'thin': int(mcmc_config.thin),
            'proposal_scale': float(mcmc_config.proposal_scale),
            'prior_alpha': float(mcmc_config.prior_alpha),
            'soft_elimination': bool(mcmc_config.soft_elimination),
            'violation_lambda_percent': float(mcmc_config.violation_lambda_percent),
            'violation_lambda_rank': float(mcmc_config.violation_lambda_rank),
            'judge_save_enabled': bool(mcmc_config.judge_save_enabled),
            'judge_save_bottom_k': int(mcmc_config.judge_save_bottom_k),
            'temporal_smoothing_enabled': bool(mcmc_config.temporal_smoothing_enabled),
            'temporal_alpha': float(mcmc_config.temporal_alpha),
            'temporal_beta': float(mcmc_config.temporal_beta)
        }
    }
    data['meta'] = np.array(json.dumps(meta, ensure_ascii=False), dtype=object)

    np.savez_compressed(out_path, **data)


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

