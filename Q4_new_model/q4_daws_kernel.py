from __future__ import annotations

import numpy as np


def _name_key(names: list[str]) -> np.ndarray:
    arr = np.asarray(names, dtype=object)
    sorted_names = np.sort(arr)
    return np.searchsorted(sorted_names, arr).astype(float)


def _ensure_row_normalized(samples: np.ndarray) -> np.ndarray:
    """Ensure fan samples are row-normalized shares.

    Q1/Q2 export is intended to be shares; this protects against unnormalized inputs.
    """

    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"fan_samples must be 2D (S,N); got shape={samples.shape}")

    row_sum = samples.sum(axis=1, keepdims=True)
    bad = ~np.isfinite(row_sum) | (row_sum <= 0)
    if np.any(bad):
        raise ValueError("fan_samples has non-finite or non-positive row sums")

    # If already close to 1, keep as-is to preserve exact Q1 posterior draws.
    if np.all(np.isclose(row_sum, 1.0, rtol=1e-3, atol=1e-6)):
        return samples

    return samples / row_sum


def _choose_worst_vectorized(
    primary: np.ndarray,
    judge_scores: np.ndarray,
    fan_scores: np.ndarray,
    name_key: np.ndarray,
    primary_higher_is_worse: bool,
) -> np.ndarray:
    """Return worst index per row, breaking ties by judge(low), fan(low), name(lex)."""

    if primary.ndim != 2:
        raise ValueError(f"primary must be 2D; got shape={primary.shape}")

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


def simulate_elimination_daws_direct(
    fan_samples: np.ndarray,
    judge_share: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
    w_fan: float,
) -> np.ndarray:
    """DAWS-Direct: eliminate argmin of C = w*F + (1-w)*J."""

    if not (0.0 <= float(w_fan) <= 1.0):
        raise ValueError(f"w_fan must be in [0,1]; got {w_fan}")

    fan_share = _ensure_row_normalized(fan_samples)
    judge_share = np.asarray(judge_share, dtype=float)
    judge_scores = np.asarray(judge_scores, dtype=float)

    if judge_share.ndim != 1 or judge_scores.ndim != 1:
        raise ValueError("judge_share and judge_scores must be 1D")

    s_count, n = fan_share.shape
    if judge_share.shape[0] != n or judge_scores.shape[0] != n:
        raise ValueError("dimension mismatch between fan_samples and judge vectors")

    name_key = _name_key(names)
    total = float(w_fan) * fan_share + (1.0 - float(w_fan)) * judge_share[None, :]
    return _choose_worst_vectorized(total, judge_scores, fan_share, name_key, False)


def simulate_elimination_daws_save(
    fan_samples: np.ndarray,
    judge_share: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
    w_fan: float,
) -> np.ndarray:
    """DAWS-Save: bottom-2 by DAWS, then judges eliminate lower judge_score."""

    if not (0.0 <= float(w_fan) <= 1.0):
        raise ValueError(f"w_fan must be in [0,1]; got {w_fan}")

    fan_share = _ensure_row_normalized(fan_samples)
    judge_share = np.asarray(judge_share, dtype=float)
    judge_scores = np.asarray(judge_scores, dtype=float)

    s_count, n = fan_share.shape
    if judge_share.shape[0] != n or judge_scores.shape[0] != n:
        raise ValueError("dimension mismatch between fan_samples and judge vectors")

    name_key = _name_key(names)
    total = float(w_fan) * fan_share + (1.0 - float(w_fan)) * judge_share[None, :]

    worst1 = _choose_worst_vectorized(total, judge_scores, fan_share, name_key, False)
    total2 = total.copy()
    total2[np.arange(s_count), worst1] = np.inf
    worst2 = _choose_worst_vectorized(total2, judge_scores, fan_share, name_key, False)

    j1 = judge_scores[worst1]
    j2 = judge_scores[worst2]
    choose_second = j2 < j1

    tied = j1 == j2
    if np.any(tied):
        f1 = fan_share[np.arange(s_count), worst1]
        f2 = fan_share[np.arange(s_count), worst2]
        choose_second = np.where(tied & (f2 < f1), True, choose_second)
        tied2 = tied & (f1 == f2)
        if np.any(tied2):
            nk1 = name_key[worst1]
            nk2 = name_key[worst2]
            choose_second = np.where(tied2 & (nk2 < nk1), True, choose_second)

    return np.where(choose_second, worst2, worst1).astype(int)


def simulate_finale_daws_distribution(
    fan_samples: np.ndarray,
    judge_share: np.ndarray,
    judge_scores: np.ndarray,
    names: list[str],
    w_fan: float,
) -> np.ndarray:
    """Finale placements under DAWS composite score (1=best)."""

    if not (0.0 <= float(w_fan) <= 1.0):
        raise ValueError(f"w_fan must be in [0,1]; got {w_fan}")

    fan_share = _ensure_row_normalized(fan_samples)
    judge_share = np.asarray(judge_share, dtype=float)
    judge_scores = np.asarray(judge_scores, dtype=float)

    s_count, n = fan_share.shape
    placements = np.empty((s_count, n), dtype=int)

    for s in range(s_count):
        total = float(w_fan) * fan_share[s] + (1.0 - float(w_fan)) * judge_share
        order = np.lexsort(
            (
                np.array(names),
                -fan_share[s],
                -judge_scores,
                -total,
            )
        )
        placements[s, order] = np.arange(1, n + 1)

    return placements
