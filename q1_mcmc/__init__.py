# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C - Q1: MCMC粉丝投票反推模型

模块列表：
- config.py: 参数配置
- scoring.py: 评分与约束计算
- sampler.py: MCMC采样器
- engine.py: 推断引擎
- diagnostics.py: 诊断与可视化
- main.py: 主启动器
"""

from .config import MCMCConfig, PathConfig, FilterConfig, get_combine_method
from .sampler import UnifiedMCMCSampler, SamplingResult
from .engine import UnifiedFanVoteInferenceEngine, create_engine
from .diagnostics import compute_diagnostics, DiagnosticResult

__all__ = [
    'MCMCConfig',
    'PathConfig', 
    'FilterConfig',
    'get_combine_method',
    'UnifiedMCMCSampler',
    'SamplingResult',
    'UnifiedFanVoteInferenceEngine',
    'create_engine',
    'compute_diagnostics',
    'DiagnosticResult'
]

__version__ = '1.0.0'
