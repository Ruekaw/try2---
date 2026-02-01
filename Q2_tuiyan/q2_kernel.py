from __future__ import annotations

import numpy as np


def _rank_with_ties_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order, dtype=int)
    current_rank = 1
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = current_rank
        current_rank = j + 2
        i = j + 1
    return ranks


def _choose_worst(
    primary: np.ndarray,
    judge_scores: np.ndarray,
    fan_scores: np.ndarray,
    names: list[str],
    primary_higher_is_worse: bool,
) -> int:
    if primary_higher_is_worse:
        best_primary = primary.max()
        candidates = np.where(primary == best_primary)[0]
    else:
        best_primary = primary.min()
        candidates = np.where(primary == best_primary)[0]

    if len(candidates) == 1:
        return int(candidates[0])

    judge_vals = judge_scores[candidates]
    low_j = judge_vals.min()
    candidates = candidates[judge_vals == low_j]
    if len(candidates) == 1:
        return int(candidates[0])

    fan_vals = fan_scores[candidates]
    low_f = fan_vals.min()
    candidates = candidates[fan_vals == low_f]
    if len(candidates) == 1:
        return int(candidates[0])

    name_vals = np.array([names[i] for i in candidates])
    return int(candidates[np.argmin(name_vals)])


def _bottom2_indices_rank(sum_rank: np.ndarray, judge_scores: np.ndarray, fan_scores: np.ndarray, names: list[str]) -> np.ndarray:
    primary = sum_rank
    keys = np.lexsort(
        (
            np.array(names),
            fan_scores,
            judge_scores,
            -primary,
        )
    )
    return keys[:2]


def _bottom2_indices_percent(total: np.ndarray, judge_scores: np.ndarray, fan_scores: np.ndarray, names: list[str]) -> np.ndarray:
    primary = total
    keys = np.lexsort(
        (
            np.array(names),
            fan_scores,
            judge_scores,
            primary,
        )
    )
    return keys[:2]


def simulate_elimination_rank_direct(
    fan_samples: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    judge_rank = _rank_with_ties_desc(judge_scores)
    results = np.empty(fan_samples.shape[0], dtype=int)
    for s in range(fan_samples.shape[0]):
        fan_rank = _rank_with_ties_desc(fan_samples[s])
        sum_rank = judge_rank + fan_rank
        results[s] = _choose_worst(sum_rank, judge_scores, fan_samples[s], names, True)
    return results


def simulate_elimination_rank_save(
    fan_samples: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    judge_rank = _rank_with_ties_desc(judge_scores)
    results = np.empty(fan_samples.shape[0], dtype=int)
    for s in range(fan_samples.shape[0]):
        fan_rank = _rank_with_ties_desc(fan_samples[s])
        sum_rank = judge_rank + fan_rank
        bottom2 = _bottom2_indices_rank(sum_rank, judge_scores, fan_samples[s], names)
        j0, j1 = judge_scores[bottom2]
        if j0 != j1:
            results[s] = int(bottom2[0] if j0 < j1 else bottom2[1])
        else:
            f0, f1 = fan_samples[s][bottom2]
            if f0 != f1:
                results[s] = int(bottom2[0] if f0 < f1 else bottom2[1])
            else:
                name_vals = np.array([names[i] for i in bottom2])
                results[s] = int(bottom2[np.argmin(name_vals)])
    return results


def simulate_elimination_percent_direct(
    fan_samples: np.ndarray,
    judge_share: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    results = np.empty(fan_samples.shape[0], dtype=int)
    for s in range(fan_samples.shape[0]):
        total = judge_share + fan_samples[s]
        results[s] = _choose_worst(total, judge_scores, fan_samples[s], names, False)
    return results


def simulate_elimination_percent_save(
    fan_samples: np.ndarray,
    judge_share: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    results = np.empty(fan_samples.shape[0], dtype=int)
    for s in range(fan_samples.shape[0]):
        total = judge_share + fan_samples[s]
        bottom2 = _bottom2_indices_percent(total, judge_scores, fan_samples[s], names)
        j0, j1 = judge_scores[bottom2]
        if j0 != j1:
            results[s] = int(bottom2[0] if j0 < j1 else bottom2[1])
        else:
            f0, f1 = fan_samples[s][bottom2]
            if f0 != f1:
                results[s] = int(bottom2[0] if f0 < f1 else bottom2[1])
            else:
                name_vals = np.array([names[i] for i in bottom2])
                results[s] = int(bottom2[np.argmin(name_vals)])
    return results


def simulate_finale_rank_distribution(
    fan_samples: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
) -> np.ndarray:
    judge_rank = _rank_with_ties_desc(judge_scores)
    s_count, n = fan_samples.shape
    placements = np.empty((s_count, n), dtype=int)
    for s in range(s_count):
        fan_rank = _rank_with_ties_desc(fan_samples[s])
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