# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C - Q1: MCMC粉丝投票反推模型
诊断与可视化模块 (diagnostics.py)

包含：
1. 有效样本量 (ESS) 计算
2. 蒙特卡洛标准误 (MCSE) 计算
3. 自相关分析
4. 收敛诊断
5. 可视化函数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

# 可选导入 matplotlib（用于可视化）
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib 未安装，可视化功能不可用")


@dataclass
class DiagnosticResult:
    """诊断结果"""
    ess: np.ndarray                  # 有效样本量
    mcse: np.ndarray                 # 蒙特卡洛标准误
    autocorr_time: np.ndarray        # 自相关时间
    rhat: Optional[np.ndarray]       # R-hat（多链时）
    acceptance_rate: float


def compute_autocorrelation(x: np.ndarray, max_lag: int = None) -> np.ndarray:
    """
    计算自相关函数
    
    Args:
        x: 1D 数组，单个参数的 MCMC 样本
        max_lag: 最大滞后期
    
    Returns:
        自相关系数数组
    """
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 2, 500)
    
    x = x - np.mean(x)
    variance = np.var(x)
    
    if variance < 1e-10:
        return np.zeros(max_lag)
    
    acf = np.zeros(max_lag)
    for k in range(max_lag):
        acf[k] = np.sum(x[:n-k] * x[k:]) / ((n - k) * variance)
    
    return acf


def compute_ess_geyer(samples: np.ndarray) -> Tuple[float, float]:
    """
    使用 Geyer 初始正序列估计器计算有效样本量
    
    基于 Geyer (1992) 的方法
    
    Args:
        samples: 1D 数组，单个参数的 MCMC 样本
    
    Returns:
        (ESS, 自相关时间 τ)
    """
    n = len(samples)
    acf = compute_autocorrelation(samples)
    
    # Geyer's initial positive sequence estimator
    # 找到第一个非正自相关对
    tau = 1.0
    max_lag = len(acf) // 2
    
    for k in range(max_lag):
        # 成对求和
        pair_sum = acf[2*k] + acf[2*k + 1] if 2*k + 1 < len(acf) else acf[2*k]
        
        if pair_sum < 0:
            break
        
        tau += 2 * pair_sum
    
    # 确保 tau >= 1
    tau = max(tau, 1.0)
    
    ess = n / tau
    
    return ess, tau


def compute_ess_batch(samples: np.ndarray) -> np.ndarray:
    """
    批量计算所有参数的 ESS
    
    Args:
        samples: shape (n_samples, n_params)
    
    Returns:
        shape (n_params,) ESS 数组
    """
    n_samples, n_params = samples.shape
    ess = np.zeros(n_params)
    
    for i in range(n_params):
        ess[i], _ = compute_ess_geyer(samples[:, i])
    
    return ess


def compute_mcse(samples: np.ndarray, ess: np.ndarray) -> np.ndarray:
    """
    计算蒙特卡洛标准误
    
    MCSE = σ / √ESS
    
    Args:
        samples: shape (n_samples, n_params)
        ess: shape (n_params,)
    
    Returns:
        shape (n_params,) MCSE 数组
    """
    std = np.std(samples, axis=0)
    mcse = std / np.sqrt(np.maximum(ess, 1))
    return mcse


def compute_diagnostics(
    samples: np.ndarray,
    acceptance_rate: float
) -> DiagnosticResult:
    """
    计算完整诊断指标
    
    Args:
        samples: shape (n_samples, n_params)
        acceptance_rate: 接受率
    
    Returns:
        DiagnosticResult
    """
    n_samples, n_params = samples.shape
    
    # ESS
    ess = compute_ess_batch(samples)
    
    # MCSE
    mcse = compute_mcse(samples, ess)
    
    # 自相关时间
    autocorr_time = n_samples / np.maximum(ess, 1)
    
    return DiagnosticResult(
        ess=ess,
        mcse=mcse,
        autocorr_time=autocorr_time,
        rhat=None,  # 单链无 R-hat
        acceptance_rate=acceptance_rate
    )


def diagnose_convergence(
    ess: np.ndarray,
    acceptance_rate: float,
    n_samples: int,
    ess_threshold: float = 100,
    accept_range: Tuple[float, float] = (0.1, 0.5)
) -> Dict[str, bool]:
    """
    诊断收敛状态
    
    Args:
        ess: 有效样本量
        acceptance_rate: 接受率
        n_samples: 总样本数
        ess_threshold: ESS 阈值
        accept_range: 接受率合理范围
    
    Returns:
        诊断结果字典
    """
    diagnostics = {
        'ess_sufficient': np.all(ess > ess_threshold),
        'min_ess': float(np.min(ess)),
        'mean_ess': float(np.mean(ess)),
        'acceptance_reasonable': accept_range[0] <= acceptance_rate <= accept_range[1],
        'likely_converged': np.all(ess > ess_threshold) and accept_range[0] <= acceptance_rate <= accept_range[1]
    }
    
    return diagnostics


# === 可视化函数 ===

def plot_trace(
    samples: np.ndarray,
    param_names: Optional[List[str]] = None,
    title: str = "Trace Plot",
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[plt.Figure]:
    """
    绘制迹图
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib 未安装")
        return None
    
    n_samples, n_params = samples.shape
    
    if param_names is None:
        param_names = [f"Contestant {i+1}" for i in range(n_params)]
    
    fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)
    if n_params == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(samples[:, i], alpha=0.7, lw=0.5)
        ax.set_ylabel(name, fontsize=9)
        ax.axhline(np.mean(samples[:, i]), color='r', ls='--', lw=1)
    
    axes[-1].set_xlabel("Iteration")
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_posterior_density(
    samples: np.ndarray,
    param_names: Optional[List[str]] = None,
    title: str = "Posterior Density",
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[plt.Figure]:
    """
    绘制后验密度图
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib 未安装")
        return None
    
    n_samples, n_params = samples.shape
    
    if param_names is None:
        param_names = [f"Contestant {i+1}" for i in range(n_params)]
    
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    for i, (ax, name) in enumerate(zip(axes[:n_params], param_names)):
        ax.hist(samples[:, i], bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(samples[:, i]), color='r', ls='--', label='Mean')
        ax.axvline(np.percentile(samples[:, i], 2.5), color='g', ls=':', label='95% CI')
        ax.axvline(np.percentile(samples[:, i], 97.5), color='g', ls=':')
        ax.set_title(name)
        ax.set_xlabel("Fan Vote Share")
    
    # 隐藏多余的子图
    for ax in axes[n_params:]:
        ax.set_visible(False)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig


def plot_certainty_heatmap(
    certainty_df: pd.DataFrame,
    season: int,
    title: str = None,
    figsize: Tuple[int, int] = (14, 8),
    cmap: str = 'RdYlGn'
) -> Optional[plt.Figure]:
    """
    绘制确定性热力图
    
    Args:
        certainty_df: 包含 week, celebrity_name, certainty_index 的 DataFrame
        season: 赛季号
        title: 标题
        figsize: 图形大小
        cmap: 颜色映射
    
    Returns:
        matplotlib Figure 或 None
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib 未安装")
        return None
    
    # Pivot 数据
    pivot_df = certainty_df.pivot(
        index='celebrity_name',
        columns='week',
        values='certainty_index'
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    im = ax.imshow(pivot_df.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # 设置标签
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([f"W{w}" for w in pivot_df.columns])
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, fontsize=8)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Certainty Index (λ)')
    
    if title is None:
        title = f"Season {season} - Fan Vote Certainty Heatmap"
    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel("Celebrity")
    
    plt.tight_layout()
    
    return fig


def plot_ppc_summary(
    results_df: pd.DataFrame,
    title: str = "PPC Consistency by Season",
    figsize: Tuple[int, int] = (12, 6)
) -> Optional[plt.Figure]:
    """
    绘制 PPC 一致性汇总图
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib 未安装")
        return None
    
    # 按赛季汇总
    season_ppc = results_df.groupby('season')['ppc_consistency'].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(season_ppc['season'], season_ppc['mean'], yerr=season_ppc['std'],
           capsize=3, alpha=0.7, edgecolor='black')
    
    ax.axhline(0.95, color='r', ls='--', label='95% threshold')
    ax.set_xlabel("Season")
    ax.set_ylabel("PPC Consistency")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    return fig


def generate_diagnostic_report(
    results_df: pd.DataFrame,
    output_dir: str
) -> str:
    """
    生成诊断报告
    
    Args:
        results_df: 长格式结果 DataFrame
        output_dir: 输出目录
    
    Returns:
        报告文本
    """
    report_lines = [
        "=" * 60,
        "MCMC Fan Vote Inference - Diagnostic Report",
        "=" * 60,
        "",
        f"Total records: {len(results_df)}",
        f"Seasons: {results_df['season'].nunique()}",
        f"Weeks: {results_df.groupby('season')['week'].nunique().sum()}",
        "",
        "--- Convergence Summary ---",
        f"Converged weeks: {results_df.groupby(['season', 'week'])['converged'].first().sum()}",
        f"Convergence rate: {results_df.groupby(['season', 'week'])['converged'].first().mean():.2%}",
        "",
        "--- PPC Consistency ---",
        f"Mean: {results_df['ppc_consistency'].mean():.3f}",
        f"Std: {results_df['ppc_consistency'].std():.3f}",
        f"Min: {results_df['ppc_consistency'].min():.3f}",
        f"Max: {results_df['ppc_consistency'].max():.3f}",
        "",
        "--- Acceptance Rate ---",
        f"Mean: {results_df['acceptance_rate'].mean():.3f}",
        f"Min: {results_df['acceptance_rate'].min():.3f}",
        f"Max: {results_df['acceptance_rate'].max():.3f}",
        "",
        "--- Certainty Index ---",
        f"Mean: {results_df['certainty_index'].mean():.3f}",
        f"Std: {results_df['certainty_index'].std():.3f}",
        "",
        "--- 95% CI Width ---",
        f"Mean: {results_df['ci_width'].mean():.4f}",
        f"Std: {results_df['ci_width'].std():.4f}",
        "",
        "=" * 60
    ]
    
    report = "\n".join(report_lines)
    
    # 保存报告
    report_path = f"{output_dir}/diagnostic_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"诊断报告已保存: {report_path}")
    
    return report


if __name__ == "__main__":
    # 测试诊断模块
    np.random.seed(42)
    
    # 生成模拟样本
    n_samples = 1000
    n_params = 4
    
    # 模拟有自相关的样本
    samples = np.zeros((n_samples, n_params))
    for i in range(n_params):
        samples[0, i] = np.random.random()
        for t in range(1, n_samples):
            samples[t, i] = 0.9 * samples[t-1, i] + 0.1 * np.random.random()
    
    # 归一化到单纯形
    samples = samples / samples.sum(axis=1, keepdims=True)
    
    print("=== ESS 测试 ===")
    ess = compute_ess_batch(samples)
    print(f"ESS: {ess}")
    print(f"ESS/n: {ess / n_samples}")
    
    print("\n=== MCSE 测试 ===")
    mcse = compute_mcse(samples, ess)
    print(f"MCSE: {mcse}")
    
    print("\n=== 完整诊断 ===")
    diag = compute_diagnostics(samples, acceptance_rate=0.3)
    print(f"ESS: {diag.ess}")
    print(f"MCSE: {diag.mcse}")
    print(f"自相关时间: {diag.autocorr_time}")
    
    print("\n=== 收敛诊断 ===")
    conv = diagnose_convergence(diag.ess, 0.3, n_samples)
    for k, v in conv.items():
        print(f"  {k}: {v}")
