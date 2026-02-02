from __future__ import annotations

import numpy as np


def _name_key(names: list[str]) -> np.ndarray:
    arr = np.asarray(names, dtype=object)
    sorted_names = np.sort(arr)
    return np.searchsorted(sorted_names, arr).astype(float)


def _rank_with_ties_desc_1d(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    sorted_vals = values[order]
    start = np.ones_like(sorted_vals, dtype=bool)
    start[1:] = sorted_vals[1:] != sorted_vals[:-1]
    pos = np.arange(values.shape[0])
    start_pos = np.where(start, pos, 0)
    group_start = np.maximum.accumulate(start_pos)
    rank_sorted = group_start + 1
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = rank_sorted
    return ranks


def _rank_with_ties_desc_2d(values: np.ndarray) -> np.ndarray:
    """Competition ranking per row (descending), with ties sharing rank and gaps."""
    order = np.argsort(-values, axis=1, kind="mergesort")
    sorted_vals = np.take_along_axis(values, order, axis=1)
    start = np.ones_like(sorted_vals, dtype=bool)
    start[:, 1:] = sorted_vals[:, 1:] != sorted_vals[:, :-1]
    pos = np.arange(values.shape[1])[None, :]
    start_pos = np.where(start, pos, 0)
    group_start = np.maximum.accumulate(start_pos, axis=1)
    rank_sorted = group_start + 1
    ranks = np.empty_like(order, dtype=int)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = rank_sorted
    return ranks


def _choose_worst_vectorized(
    primary: np.ndarray,
    judge_scores: np.ndarray,
    fan_scores: np.ndarray,
    name_key: np.ndarray,
    primary_higher_is_worse: bool,
) -> np.ndarray:
    """Return worst index per row, breaking ties by judge(low), fan(low), name(lex)."""
    if primary_higher_is_worse:
        target = primary.max(axis=1, keepdims=True)
        mask = primary == target
    else:
        target = primary.min(axis=1, keepdims=True)
        mask = primary == target

    js = judge_scores[None, :]
    masked_js = np.where(mask, js, np.inf)
    min_j = masked_js.min(axis=1, keepdims=True)
    mask2 = mask & (js == min_j)

    masked_f = np.where(mask2, fan_scores, np.inf)
    min_f = masked_f.min(axis=1, keepdims=True)
    mask3 = mask2 & (fan_scores == min_f)

    nk = name_key[None, :]
    masked_nk = np.where(mask3, nk, np.inf)
    return masked_nk.argmin(axis=1).astype(int)


def simulate_elimination_rank_direct(
    fan_samples: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    name_key = _name_key(names)
    judge_rank = _rank_with_ties_desc_1d(judge_scores)
    fan_rank = _rank_with_ties_desc_2d(fan_samples)
    sum_rank = fan_rank + judge_rank[None, :]
    return _choose_worst_vectorized(sum_rank.astype(float), judge_scores, fan_samples, name_key, True)


def simulate_elimination_rank_save(
    fan_samples: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    name_key = _name_key(names)
    judge_rank = _rank_with_ties_desc_1d(judge_scores)
    fan_rank = _rank_with_ties_desc_2d(fan_samples)
    sum_rank = (fan_rank + judge_rank[None, :]).astype(float)

    worst1 = _choose_worst_vectorized(sum_rank, judge_scores, fan_samples, name_key, True)
    sum_rank2 = sum_rank.copy()
    sum_rank2[np.arange(sum_rank2.shape[0]), worst1] = -np.inf
    worst2 = _choose_worst_vectorized(sum_rank2, judge_scores, fan_samples, name_key, True)

    j1 = judge_scores[worst1]
    j2 = judge_scores[worst2]
    choose_second = j2 < j1

    tied = j1 == j2
    if np.any(tied):
        f1 = fan_samples[np.arange(fan_samples.shape[0]), worst1]
        f2 = fan_samples[np.arange(fan_samples.shape[0]), worst2]
        choose_second = np.where(tied & (f2 < f1), True, choose_second)
        tied2 = tied & (f1 == f2)
        if np.any(tied2):
            nk1 = name_key[worst1]
            nk2 = name_key[worst2]
            choose_second = np.where(tied2 & (nk2 < nk1), True, choose_second)

    return np.where(choose_second, worst2, worst1).astype(int)


def simulate_elimination_percent_direct(
    fan_samples: np.ndarray,
    judge_share: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    name_key = _name_key(names)
    total = fan_samples + judge_share[None, :]
    return _choose_worst_vectorized(total, judge_scores, fan_samples, name_key, False)


def simulate_elimination_percent_save(
    fan_samples: np.ndarray,
    judge_share: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    name_key = _name_key(names)
    total = fan_samples + judge_share[None, :]

    worst1 = _choose_worst_vectorized(total, judge_scores, fan_samples, name_key, False)
    total2 = total.copy()
    total2[np.arange(total2.shape[0]), worst1] = np.inf
    worst2 = _choose_worst_vectorized(total2, judge_scores, fan_samples, name_key, False)

    j1 = judge_scores[worst1]
    j2 = judge_scores[worst2]
    choose_second = j2 < j1

    tied = j1 == j2
    if np.any(tied):
        f1 = fan_samples[np.arange(fan_samples.shape[0]), worst1]
        f2 = fan_samples[np.arange(fan_samples.shape[0]), worst2]
        choose_second = np.where(tied & (f2 < f1), True, choose_second)
        tied2 = tied & (f1 == f2)
        if np.any(tied2):
            nk1 = name_key[worst1]
            nk2 = name_key[worst2]
            choose_second = np.where(tied2 & (nk2 < nk1), True, choose_second)

    return np.where(choose_second, worst2, worst1).astype(int)


def simulate_finale_rank_distribution(
    fan_samples: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    judge_rank = _rank_with_ties_desc_1d(judge_scores)
    s_count, n = fan_samples.shape
    placements = np.empty((s_count, n), dtype=int)
    for s in range(s_count):
        fan_rank = _rank_with_ties_desc_1d(fan_samples[s])
        sum_rank = judge_rank + fan_rank
        order = np.lexsort(
            (
                np.array(names),
                fan_samples[s],
                judge_scores,
                sum_rank,
            )
        )
        placements[s, order] = np.arange(1, n + 1)
    return placements


def simulate_finale_percent_distribution(
    fan_samples: np.ndarray,
    judge_share: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    s_count, n = fan_samples.shape
    placements = np.empty((s_count, n), dtype=int)
    for s in range(s_count):
        total = judge_share + fan_samples[s]
        order = np.lexsort(
            (
                np.array(names),
                -fan_samples[s],
                -judge_scores,
                -total,
            )
        )
        placements[s, order] = np.arange(1, n + 1)
    return placements