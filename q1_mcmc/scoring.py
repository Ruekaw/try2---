# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C - Q1: MCMC粉丝投票反推模型
评分与约束模块 (scoring.py)

包含：
1. 综合得分计算（百分比制/排名制）
2. 淘汰约束检验
3. 违约程度计算（软约束）
4. 决赛周特殊约束
"""

import numpy as np
from scipy.stats import rankdata
from typing import List, Tuple, Optional
from config import COMBINE_PERCENT, COMBINE_RANK


def compute_judge_percentages(judge_scores: np.ndarray) -> np.ndarray:
    """
    计算评委评分百分比
    
    Args:
        judge_scores: shape (n_contestants,) 各选手的评委总分
    
    Returns:
        评委百分比 shape (n_contestants,)
    """
    total = np.sum(judge_scores)
    if total == 0:
        return np.ones_like(judge_scores) / len(judge_scores)
    return judge_scores / total


def compute_ranks(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """
    计算排名（分数高→排名低，即排名1为最佳）
    
    Args:
        scores: shape (n_contestants,) 各选手得分
        higher_is_better: True表示分数越高越好
    
    Returns:
        排名 shape (n_contestants,)，使用 min 方法处理并列
    """
    if higher_is_better:
        # 分数取负使得高分对应低排名
        return rankdata(-scores, method='min')
    else:
        return rankdata(scores, method='min')


def compute_total_scores_percent(
    fan_votes: np.ndarray,
    judge_scores: np.ndarray
) -> np.ndarray:
    """
    百分比制综合得分计算
    
    综合百分比 = 评委百分比 + 粉丝投票百分比
    
    Args:
        fan_votes: shape (n_contestants,) 粉丝投票份额（和为1）
        judge_scores: shape (n_contestants,) 评委总分
    
    Returns:
        综合百分比 shape (n_contestants,)
    """
    judge_pct = compute_judge_percentages(judge_scores)
    # fan_votes 已经是份额，直接相加
    return judge_pct + fan_votes


def compute_total_scores_rank(
    fan_votes: np.ndarray,
    judge_scores: np.ndarray
) -> np.ndarray:
    """
    排名制综合得分计算
    
    综合排名和 = 评委排名 + 粉丝排名
    注意：排名和越小越好
    
    Args:
        fan_votes: shape (n_contestants,) 粉丝投票份额（和为1）
        judge_scores: shape (n_contestants,) 评委总分
    
    Returns:
        综合排名和 shape (n_contestants,)
    """
    # 评委排名：分数高→排名低
    judge_ranks = compute_ranks(judge_scores, higher_is_better=True)
    # 粉丝排名：票数高→排名低
    fan_ranks = compute_ranks(fan_votes, higher_is_better=True)
    return judge_ranks + fan_ranks


def compute_total_scores(
    fan_votes: np.ndarray,
    judge_scores: np.ndarray,
    combine_method: str
) -> np.ndarray:
    """
    根据结合方法计算综合得分
    
    Args:
        fan_votes: shape (n_contestants,) 粉丝投票份额
        judge_scores: shape (n_contestants,) 评委总分
        combine_method: 'percent' 或 'rank'
    
    Returns:
        综合得分 shape (n_contestants,)
    """
    if combine_method == COMBINE_PERCENT:
        return compute_total_scores_percent(fan_votes, judge_scores)
    elif combine_method == COMBINE_RANK:
        return compute_total_scores_rank(fan_votes, judge_scores)
    else:
        raise ValueError(f"Unknown combine method: {combine_method}")


def get_bottom_k_indices(
    total_scores: np.ndarray,
    k: int,
    combine_method: str
) -> np.ndarray:
    """
    获取综合得分最差的 k 个选手索引
    
    Args:
        total_scores: 综合得分数组
        k: 返回的选手数量
        combine_method: 结合方法
    
    Returns:
        最差 k 个选手的索引数组
    """
    if combine_method == COMBINE_PERCENT:
        # 百分比制：得分越低越差
        sorted_indices = np.argsort(total_scores)
    else:
        # 排名制：排名和越大越差
        sorted_indices = np.argsort(-total_scores)
    
    return sorted_indices[:k]


def is_in_bottom_k(
    index: int,
    total_scores: np.ndarray,
    k: int,
    combine_method: str
) -> bool:
    """
    检查某选手是否在 bottom-k 中
    """
    bottom_k = get_bottom_k_indices(total_scores, k, combine_method)
    return index in bottom_k


def check_elimination_constraint(
    fan_votes: np.ndarray,
    judge_scores: np.ndarray,
    eliminated_indices: List[int],
    combine_method: str,
    judge_save_enabled: bool = False,
    judge_save_bottom_k: int = 2
) -> bool:
    """
    检查淘汰约束是否满足（硬约束）
    
    规则：
    - 普通周：被淘汰者必须在综合得分的 bottom-k 中
    - 评委救人周（S28+）：被淘汰者必须在 bottom-2 中
    
    Args:
        fan_votes: 粉丝投票份额
        judge_scores: 评委总分
        eliminated_indices: 被淘汰选手的索引列表
        combine_method: 结合方法
        judge_save_enabled: 是否启用评委救人
        judge_save_bottom_k: 评委救人的 bottom-k
    
    Returns:
        是否满足约束
    """
    total_scores = compute_total_scores(fan_votes, judge_scores, combine_method)
    k = len(eliminated_indices)
    
    if judge_save_enabled and k == 1:
        # 评委救人：被淘汰者只需在 bottom-2 中
        check_k = judge_save_bottom_k
    else:
        # 普通情况：被淘汰者必须恰好是 bottom-k
        check_k = k
    
    for idx in eliminated_indices:
        if not is_in_bottom_k(idx, total_scores, check_k, combine_method):
            return False
    
    return True


def compute_elimination_violation(
    fan_votes: np.ndarray,
    judge_scores: np.ndarray,
    eliminated_indices: List[int],
    combine_method: str,
    judge_save_enabled: bool = False,
    judge_save_bottom_k: int = 2
) -> float:
    """
    计算淘汰约束的违约程度（用于软约束）
    
    违约程度：被淘汰者偏离 bottom-k 的程度之和
    
    Args:
        同 check_elimination_constraint
    
    Returns:
        违约程度（非负浮点数，0表示满足约束）
    """
    total_scores = compute_total_scores(fan_votes, judge_scores, combine_method)
    n = len(fan_votes)
    k = len(eliminated_indices)
    
    if judge_save_enabled and k == 1:
        check_k = judge_save_bottom_k
    else:
        check_k = k
    
    # 获取 bottom-k 的阈值分数
    if combine_method == COMBINE_PERCENT:
        # 百分比制：找到第 check_k 小的分数作为阈值
        sorted_scores = np.sort(total_scores)
        threshold = sorted_scores[check_k - 1]  # bottom-k 的最大值
        
        violation = 0.0
        for idx in eliminated_indices:
            score = total_scores[idx]
            if score > threshold:
                # 分数比阈值高，说明不在 bottom-k，计算违约
                violation += (score - threshold)
    else:
        # 排名制：找到第 check_k 大的排名和作为阈值
        sorted_scores = np.sort(total_scores)[::-1]  # 降序
        threshold = sorted_scores[check_k - 1]  # bottom-k 的最小值
        
        violation = 0.0
        for idx in eliminated_indices:
            score = total_scores[idx]
            if score < threshold:
                # 排名和比阈值小，说明不在 bottom-k，计算违约
                violation += (threshold - score)
    
    return violation


def compute_finale_violation(
    fan_votes: np.ndarray,
    judge_scores: np.ndarray,
    placements: List[int],
    combine_method: str
) -> float:
    """
    计算决赛周约束的违约程度
    
    决赛约束（链式比较）：
    - 百分比制：第1名得分 > 第2名 > 第3名 ...
    - 排名制：第1名排名和 < 第2名 < 第3名 ...
    
    Args:
        fan_votes: 粉丝投票份额
        judge_scores: 评委总分
        placements: 选手对应的最终名次列表，例如 [1, 2, 3] 表示三个选手分别是冠亚季军
        combine_method: 结合方法
    
    Returns:
        违约程度
    """
    total_scores = compute_total_scores(fan_votes, judge_scores, combine_method)
    
    # 按名次排序选手索引
    # placements[i] 是第 i 个选手的名次
    n = len(placements)
    sorted_by_placement = sorted(range(n), key=lambda i: placements[i])
    
    violation = 0.0
    
    for i in range(n - 1):
        higher_rank_idx = sorted_by_placement[i]      # 名次更好的选手
        lower_rank_idx = sorted_by_placement[i + 1]   # 名次更差的选手
        
        score_higher = total_scores[higher_rank_idx]
        score_lower = total_scores[lower_rank_idx]
        
        if combine_method == COMBINE_PERCENT:
            # 百分比制：名次好的得分应该更高
            if score_lower > score_higher:
                violation += (score_lower - score_higher)
        else:
            # 排名制：名次好的排名和应该更小
            if score_higher > score_lower:
                violation += (score_higher - score_lower)
    
    return violation


def compute_total_violation(
    fan_votes: np.ndarray,
    judge_scores: np.ndarray,
    eliminated_indices: List[int],
    combine_method: str,
    is_finale: bool = False,
    placements: Optional[List[int]] = None,
    judge_save_enabled: bool = False,
    judge_save_bottom_k: int = 2
) -> float:
    """
    计算总违约程度
    
    Args:
        fan_votes: 粉丝投票份额
        judge_scores: 评委总分
        eliminated_indices: 被淘汰选手索引
        combine_method: 结合方法
        is_finale: 是否为决赛周
        placements: 决赛周时，各选手的最终名次
        judge_save_enabled: 是否启用评委救人
        judge_save_bottom_k: 评委救人 bottom-k
    
    Returns:
        总违约程度
    """
    if is_finale and placements is not None:
        return compute_finale_violation(fan_votes, judge_scores, placements, combine_method)
    else:
        return compute_elimination_violation(
            fan_votes, judge_scores, eliminated_indices,
            combine_method, judge_save_enabled, judge_save_bottom_k
        )


# === 辅助函数 ===

def normalize_votes(votes: np.ndarray) -> np.ndarray:
    """
    将投票归一化到单纯形上
    """
    votes = np.maximum(votes, 1e-10)  # 确保非负
    return votes / np.sum(votes)


def is_valid_simplex(v: np.ndarray, tol: float = 1e-8) -> bool:
    """
    检查向量是否在单纯形上
    """
    return np.all(v >= -tol) and np.abs(np.sum(v) - 1.0) < tol


if __name__ == "__main__":
    # 测试示例
    np.random.seed(42)
    
    # 模拟4个选手
    judge_scores = np.array([25.0, 20.0, 21.0, 26.0])  # 评委总分
    fan_votes = np.array([0.11, 0.37, 0.32, 0.20])     # 粉丝投票份额
    eliminated_indices = [0]  # Rachel Hunter 被淘汰
    
    print("=== 百分比制测试 ===")
    total_pct = compute_total_scores_percent(fan_votes, judge_scores)
    print(f"评委百分比: {compute_judge_percentages(judge_scores)}")
    print(f"综合百分比: {total_pct}")
    print(f"排序（低到高）: {np.argsort(total_pct)}")
    print(f"淘汰约束满足: {check_elimination_constraint(fan_votes, judge_scores, eliminated_indices, COMBINE_PERCENT)}")
    print(f"违约程度: {compute_elimination_violation(fan_votes, judge_scores, eliminated_indices, COMBINE_PERCENT)}")
    
    print("\n=== 排名制测试 ===")
    total_rank = compute_total_scores_rank(fan_votes, judge_scores)
    print(f"评委排名: {compute_ranks(judge_scores)}")
    print(f"粉丝排名: {compute_ranks(fan_votes)}")
    print(f"排名和: {total_rank}")
    print(f"排序（高到低）: {np.argsort(-total_rank)}")
    print(f"淘汰约束满足: {check_elimination_constraint(fan_votes, judge_scores, eliminated_indices, COMBINE_RANK)}")
    print(f"违约程度: {compute_elimination_violation(fan_votes, judge_scores, eliminated_indices, COMBINE_RANK)}")
    
    print("\n=== 决赛周测试 ===")
    placements = [2, 4, 3, 1]  # 假设第4个选手是冠军
    finale_violation = compute_finale_violation(fan_votes, judge_scores, placements, COMBINE_PERCENT)
    print(f"决赛违约程度（百分比制）: {finale_violation}")
