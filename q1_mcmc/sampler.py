# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C - Q1: MCMC粉丝投票反推模型
MCMC采样器模块 (sampler.py)

包含：
1. Dirichlet 先验与提议分布
2. Metropolis-Hastings 采样器
3. 后验计算（软/硬约束）
"""

import numpy as np
from scipy.special import gammaln
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from config import MCMCConfig, COMBINE_PERCENT, COMBINE_RANK, get_violation_lambda
from scoring import (
    compute_total_scores,
    compute_total_violation,
    check_elimination_constraint,
    normalize_votes
)


@dataclass
class SamplingResult:
    """采样结果"""
    samples: np.ndarray              # shape (n_samples, n_contestants)
    acceptance_rate: float           # 接受率
    n_total_iterations: int          # 总迭代次数
    converged: bool                  # 是否收敛/成功
    init_attempts: int               # 初始化尝试次数
    violation_history: Optional[np.ndarray] = None  # 违约历史


class UnifiedMCMCSampler:
    """
    统一的MCMC粉丝投票采样器
    
    在单纯形空间上进行贝叶斯推断，使用 Metropolis-Hastings 算法
    """
    
    def __init__(self, config: MCMCConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
    
    def _log_dirichlet_pdf(self, v: np.ndarray, alpha: np.ndarray) -> float:
        """
        计算 Dirichlet 分布的对数概率密度
        
        log p(v | α) = log Γ(Σα) - Σ log Γ(αi) + Σ(αi - 1) log vi
        """
        v = np.clip(v, 1e-300, 1.0)  # 避免 log(0)
        log_beta = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
        return -log_beta + np.sum((alpha - 1) * np.log(v))
    
    def _log_prior(self, v: np.ndarray, alpha_override: Optional[np.ndarray] = None) -> float:
        """
        计算先验对数概率
        
        prior(v) = Dirichlet(α, α, ..., α)
        """
        n = len(v)
        if alpha_override is not None:
            alpha = alpha_override
        else:
            alpha = np.full(n, self.config.prior_alpha)
        return self._log_dirichlet_pdf(v, alpha)
    
    def _log_proposal(self, v_to: np.ndarray, v_from: np.ndarray) -> float:
        """
        计算提议分布的对数概率 q(v_to | v_from)
        
        q(v' | v) = Dirichlet(κ * v)
        """
        alpha = self.config.proposal_scale * v_from
        alpha = np.maximum(alpha, 1e-10)  # 确保 alpha > 0
        return self._log_dirichlet_pdf(v_to, alpha)
    
    def _propose(self, current: np.ndarray) -> np.ndarray:
        """
        从提议分布采样新状态
        
        v' ~ Dirichlet(κ * v_current)
        """
        alpha = self.config.proposal_scale * current
        alpha = np.maximum(alpha, 1e-10)
        proposed = self.rng.dirichlet(alpha)
        return normalize_votes(proposed)
    
    def _log_posterior(
        self,
        v: np.ndarray,
        judge_scores: np.ndarray,
        eliminated_indices: List[int],
        combine_method: str,
        is_finale: bool = False,
        placements: Optional[List[int]] = None,
        judge_save_enabled: bool = False,
        prior_alpha_override: Optional[np.ndarray] = None,
        violation_lambda: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        计算后验对数概率（未归一化）
        
        软约束模式：log π(v) = log prior(v) - λ * V(v)
        硬约束模式：log π(v) = log prior(v) 如果满足约束，否则 -inf
        
        Returns:
            (log_posterior, violation)
        """
        log_prior = self._log_prior(v, prior_alpha_override)
        
        violation = compute_total_violation(
            v, judge_scores, eliminated_indices, combine_method,
            is_finale, placements, judge_save_enabled,
            self.config.judge_save_bottom_k
        )
        
        if self.config.soft_elimination:
            if violation_lambda is None:
                violation_lambda = get_violation_lambda(self.config, combine_method)
            # 软约束
            log_posterior = log_prior - float(violation_lambda) * violation
        else:
            # 硬约束
            if violation > 0:
                log_posterior = -np.inf
            else:
                log_posterior = log_prior
        
        return log_posterior, violation
    
    def _find_valid_initial(
        self,
        n_contestants: int,
        judge_scores: np.ndarray,
        eliminated_indices: List[int],
        combine_method: str,
        is_finale: bool = False,
        placements: Optional[List[int]] = None,
        judge_save_enabled: bool = False
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        寻找满足约束的初始状态
        
        Returns:
            (初始状态, 尝试次数) 或 (None, 尝试次数) 如果失败
        """
        for attempt in range(self.config.max_init_attempts):
            # 随机生成 Dirichlet 样本
            v = self.rng.dirichlet(np.ones(n_contestants))
            
            # 检查是否满足约束
            if not self.config.soft_elimination:
                # 硬约束：必须完全满足
                satisfied = check_elimination_constraint(
                    v, judge_scores, eliminated_indices, combine_method,
                    judge_save_enabled, self.config.judge_save_bottom_k
                )
                if satisfied:
                    return v, attempt + 1
            else:
                # 软约束：接受任何初始状态，但优先选择低违约的
                violation = compute_total_violation(
                    v, judge_scores, eliminated_indices, combine_method,
                    is_finale, placements, judge_save_enabled,
                    self.config.judge_save_bottom_k
                )
                # 前 1000 次尝试寻找零违约，之后接受任意
                if violation == 0 or attempt >= 1000:
                    return v, attempt + 1
        
        # 回退：使用软约束模式的任意状态
        if self.config.init_fallback_soft:
            v = self.rng.dirichlet(np.ones(n_contestants))
            return v, self.config.max_init_attempts
        
        return None, self.config.max_init_attempts
    
    def sample(
        self,
        judge_scores: np.ndarray,
        eliminated_indices: List[int],
        combine_method: str,
        is_finale: bool = False,
        placements: Optional[List[int]] = None,
        judge_save_enabled: bool = False,
        prior_alpha_override: Optional[np.ndarray] = None,
        violation_lambda_override: Optional[float] = None
    ) -> SamplingResult:
        """
        执行 MCMC 采样
        
        Args:
            judge_scores: shape (n_contestants,) 评委总分
            eliminated_indices: 被淘汰选手索引列表
            combine_method: 'percent' 或 'rank'
            is_finale: 是否为决赛周
            placements: 决赛周时各选手最终名次
            judge_save_enabled: 是否启用评委救人
        
        Returns:
            SamplingResult 包含采样结果和诊断信息
        """
        n_contestants = len(judge_scores)
        
        # 计算总迭代次数
        total_iterations = self.config.burn_in + self.config.n_samples * self.config.thin
        
        # 寻找初始状态
        current, init_attempts = self._find_valid_initial(
            n_contestants, judge_scores, eliminated_indices, combine_method,
            is_finale, placements, judge_save_enabled
        )
        
        if current is None:
            # 初始化失败
            return SamplingResult(
                samples=np.full((self.config.n_samples, n_contestants), np.nan),
                acceptance_rate=0.0,
                n_total_iterations=0,
                converged=False,
                init_attempts=init_attempts
            )
        
        # 计算当前状态的后验
        current_log_post, current_violation = self._log_posterior(
            current, judge_scores, eliminated_indices, combine_method,
            is_finale, placements, judge_save_enabled, prior_alpha_override,
            violation_lambda_override
        )
        
        # 存储样本和诊断信息
        samples = []
        violations = []
        n_accepted = 0
        
        for t in range(total_iterations):
            # 提议新状态
            proposed = self._propose(current)
            
            # 计算提议状态的后验
            proposed_log_post, proposed_violation = self._log_posterior(
                proposed, judge_scores, eliminated_indices, combine_method,
                is_finale, placements, judge_save_enabled, prior_alpha_override,
                violation_lambda_override
            )
            
            # 计算 M-H 接受比
            # α = min(1, π(v') q(v|v') / π(v) q(v'|v))
            log_proposal_forward = self._log_proposal(proposed, current)
            log_proposal_backward = self._log_proposal(current, proposed)
            
            log_accept_ratio = (
                proposed_log_post - current_log_post +
                log_proposal_backward - log_proposal_forward
            )
            
            # 接受/拒绝
            if np.log(self.rng.random()) < log_accept_ratio:
                current = proposed
                current_log_post = proposed_log_post
                current_violation = proposed_violation
                n_accepted += 1
            
            # 预热后稀疏保存
            if t >= self.config.burn_in and (t - self.config.burn_in) % self.config.thin == 0:
                samples.append(current.copy())
                violations.append(current_violation)
        
        samples = np.array(samples)
        violations = np.array(violations)
        
        return SamplingResult(
            samples=samples,
            acceptance_rate=n_accepted / total_iterations,
            n_total_iterations=total_iterations,
            converged=True,
            init_attempts=init_attempts,
            violation_history=violations
        )


def create_sampler(config: Optional[MCMCConfig] = None) -> UnifiedMCMCSampler:
    """
    创建采样器实例
    """
    if config is None:
        config = MCMCConfig()
    return UnifiedMCMCSampler(config)


if __name__ == "__main__":
    # 测试采样器
    from config import MCMCConfig
    
    np.random.seed(42)
    
    # 配置
    config = MCMCConfig(
        n_samples=1000,
        burn_in=500,
        thin=2,
        soft_elimination=True
    )
    
    # 模拟数据：第1赛季第4周
    judge_scores = np.array([25.0, 20.0, 21.0, 26.0])
    eliminated_indices = [0]  # Rachel Hunter
    
    print("=== 排名制测试（S1-S2）===")
    sampler = UnifiedMCMCSampler(config)
    result = sampler.sample(
        judge_scores=judge_scores,
        eliminated_indices=eliminated_indices,
        combine_method=COMBINE_RANK
    )
    
    print(f"收敛: {result.converged}")
    print(f"接受率: {result.acceptance_rate:.3f}")
    print(f"初始化尝试: {result.init_attempts}")
    print(f"样本形状: {result.samples.shape}")
    
    # 后验统计
    mean_votes = np.mean(result.samples, axis=0)
    std_votes = np.std(result.samples, axis=0)
    print(f"\n后验均值: {mean_votes}")
    print(f"后验标准差: {std_votes}")
    
    # 违约统计
    ppc = np.mean(result.violation_history == 0)
    print(f"\nPPC一致性: {ppc:.3f}")
    print(f"平均违约: {np.mean(result.violation_history):.4f}")
    
    print("\n=== 百分比制测试（S3-S27）===")
    result2 = sampler.sample(
        judge_scores=judge_scores,
        eliminated_indices=eliminated_indices,
        combine_method=COMBINE_PERCENT
    )
    
    print(f"收敛: {result2.converged}")
    print(f"接受率: {result2.acceptance_rate:.3f}")
    mean_votes2 = np.mean(result2.samples, axis=0)
    print(f"后验均值: {mean_votes2}")
