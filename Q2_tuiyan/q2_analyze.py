from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from .q2_kernel import (
    simulate_elimination_rank_direct,
    simulate_elimination_rank_save,
    simulate_elimination_percent_direct,
    simulate_elimination_percent_save,
    simulate_finale_rank_distribution,
    simulate_finale_percent_distribution,
)
from .q2_loader import WeekData


@dataclass(frozen=True)
class MethodResult:
    name: str
    eliminations: np.ndarray


def _rank_with_ties_desc_2d(values: np.ndarray) -> np.ndarray:
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


def _rank_to_quantile_badness(rank: np.ndarray, n: int) -> np.ndarray:
    """Map rank (1=best) to quantile badness in [0,1] (1=worst)."""
    if n <= 1:
        return np.zeros_like(rank, dtype=float)
    return (rank.astype(float) - 1.0) / float(n - 1)


def default_method_results(week: WeekData) -> list[MethodResult]:
    return [
        MethodResult(
            name="rank_direct",
            eliminations=simulate_elimination_rank_direct(
                week.fan_samples, week.judge_scores, week.contestants
            ),
        ),
        MethodResult(
            name="rank_save",
            eliminations=simulate_elimination_rank_save(
                week.fan_samples, week.judge_scores, week.contestants
            ),
        ),
        MethodResult(
            name="percent_direct",
            eliminations=simulate_elimination_percent_direct(
                week.fan_samples, week.judge_share, week.judge_scores, week.contestants
            ),
        ),
        MethodResult(
            name="percent_save",
            eliminations=simulate_elimination_percent_save(
                week.fan_samples, week.judge_share, week.judge_scores, week.contestants
            ),
        ),
    ]


def _method_results(week: WeekData) -> list[MethodResult]:
    # Backwards-compatible alias for internal callers.
    return default_method_results(week)


def _progress(iterable, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc)


def analyze_core_weeks(
    core_weeks: list[WeekData],
    *,
    method_results_fn=None,
) -> pd.DataFrame:
    rows = []
    for week in _progress(core_weeks, desc="Core weeks"):
        if len(week.actual_eliminated) == 0:
            actual_idx = None
        else:
            actual_name = week.actual_eliminated[0]
            try:
                actual_idx = week.contestants.index(actual_name)
            except ValueError:
                actual_idx = None

        judge_top = int(np.argmax(week.judge_scores))

        results = method_results_fn(week) if method_results_fn is not None else _method_results(week)
        for result in results:
            elim = result.eliminations
            reversal = None
            if actual_idx is not None:
                reversal = 1.0 - np.mean(elim == actual_idx)

            tech_vul = float(np.mean(elim == judge_top))

            pop_top = np.argmax(week.fan_samples, axis=1)
            pop_vul = float(np.mean(elim == pop_top))

            rows.append(
                {
                    "season": week.season,
                    "week": week.week,
                    "method": result.name,
                    "reversal_rate": reversal,
                    "tech_vulnerability": tech_vul,
                    "popularity_vulnerability": pop_vul,
                }
            )

    return pd.DataFrame(rows)


def analyze_rank_vs_percent(core_weeks: list[WeekData]) -> pd.DataFrame:
    """Compare Rank vs Percent (direct) within the same week and posterior samples.

    Outputs week-level disagreement probability and directional deltas based on
    eliminated contestant's within-week fan/judge quantile-badness.
    """

    rows = []
    for week in _progress(core_weeks, desc="Rank vs Percent"):
        elim_rank = simulate_elimination_rank_direct(
            week.fan_samples, week.judge_scores, week.contestants
        )
        elim_percent = simulate_elimination_percent_direct(
            week.fan_samples, week.judge_share, week.judge_scores, week.contestants
        )

        disagree = elim_rank != elim_percent
        disagree_rate = float(np.mean(disagree))

        s_count, n = week.fan_samples.shape
        fan_rank = _rank_with_ties_desc_2d(week.fan_samples)
        fan_q = _rank_to_quantile_badness(fan_rank, n)
        judge_rank = _rank_with_ties_desc_1d(week.judge_scores)
        judge_q_1d = _rank_to_quantile_badness(judge_rank, n)

        idx = np.arange(s_count)
        qf_rank = fan_q[idx, elim_rank]
        qf_percent = fan_q[idx, elim_percent]
        qj_rank = judge_q_1d[elim_rank]
        qj_percent = judge_q_1d[elim_percent]

        delta_fan = qf_rank - qf_percent
        delta_judge = qj_rank - qj_percent

        if np.any(disagree):
            df = delta_fan[disagree]
            dj = delta_judge[disagree]
            fan_mean = float(np.mean(df))
            judge_mean = float(np.mean(dj))
            fan_pos_rate = float(np.mean(df > 0))
            judge_pos_rate = float(np.mean(dj > 0))
        else:
            fan_mean = 0.0
            judge_mean = 0.0
            fan_pos_rate = 0.0
            judge_pos_rate = 0.0

        rows.append(
            {
                "season": week.season,
                "week": week.week,
                "disagreement_rate": disagree_rate,
                "delta_fan_badness_mean_if_disagree": fan_mean,
                "delta_judge_badness_mean_if_disagree": judge_mean,
                "rank_more_fan_friendly_rate_if_disagree": fan_pos_rate,
                "rank_more_judge_friendly_rate_if_disagree": judge_pos_rate,
                "n_samples": int(s_count),
                "n_contestants": int(n),
            }
        )

    return pd.DataFrame(rows)


def plot_season_week_metric_heatmap(
    df: pd.DataFrame,
    season: int,
    metrics: list[str],
    title: str,
    output_path: Path,
    dpi: int,
) -> None:
    sub = df[df["season"] == season].copy()
    if sub.empty:
        return
    sub = sub.sort_values(["week"]).reset_index(drop=True)
    weeks = sub["week"].astype(int).tolist()
    data = np.vstack([sub[m].to_numpy(dtype=float) for m in metrics])

    plt.figure(figsize=(max(8, 0.55 * len(weeks)), 0.65 * len(metrics) + 2.8))
    norm = Normalize(vmin=0.0, vmax=1.0)
    plt.imshow(data, aspect="auto", cmap="viridis", norm=norm)
    plt.colorbar(label="Value")
    plt.yticks(ticks=np.arange(len(metrics)), labels=metrics)
    plt.xticks(ticks=np.arange(len(weeks)), labels=[str(w) for w in weeks], rotation=0)
    plt.xlabel("Week")
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def analyze_finales(finale_weeks: list[WeekData]) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for week in _progress(finale_weeks, desc="Finale weeks"):
        rank_place = simulate_finale_rank_distribution(
            week.fan_samples, week.judge_scores, week.contestants
        )
        percent_place = simulate_finale_percent_distribution(
            week.fan_samples, week.judge_share, week.judge_scores, week.contestants
        )

        for method, placements in (
            ("rank", rank_place),
            ("percent", percent_place),
        ):
            n = len(week.contestants)
            probs = np.zeros((n, n))
            for i in range(n):
                for p in range(1, n + 1):
                    probs[i, p - 1] = np.mean(placements[:, i] == p)
            df = pd.DataFrame(
                probs,
                index=week.contestants,
                columns=[f"place_{p}" for p in range(1, n + 1)],
            )
            key = f"S{week.season}_W{week.week}_{method}"
            outputs[key] = df

    return outputs


def analyze_finale_outcome_changes(
    finale_weeks: list[WeekData],
    celebrities_of_interest: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify final-outcome changes between Rank and Percent on finale weeks.

    Returns:
      - week_level: one row per finale week, including winner-disagreement rate.
      - celeb_level: one row per (finale week, celebrity) for celebrities_of_interest
        that are present in that finale.
    """

    if celebrities_of_interest is None:
        celebrities_of_interest = []

    week_rows: list[dict[str, object]] = []
    celeb_rows: list[dict[str, object]] = []

    for week in _progress(finale_weeks, desc="Finale changes"):
        rank_place = simulate_finale_rank_distribution(
            week.fan_samples, week.judge_scores, week.contestants
        )
        percent_place = simulate_finale_percent_distribution(
            week.fan_samples, week.judge_share, week.judge_scores, week.contestants
        )

        winner_rank = np.argmin(rank_place, axis=1)
        winner_percent = np.argmin(percent_place, axis=1)
        winner_disagree = float(np.mean(winner_rank != winner_percent))

        week_rows.append(
            {
                "season": int(week.season),
                "week": int(week.week),
                "n_finalists": int(len(week.contestants)),
                "winner_disagreement_rate": winner_disagree,
            }
        )

        if celebrities_of_interest:
            for celeb in celebrities_of_interest:
                if celeb not in week.contestants:
                    continue
                idx = week.contestants.index(celeb)
                win_rank = float(np.mean(rank_place[:, idx] == 1))
                win_percent = float(np.mean(percent_place[:, idx] == 1))
                exp_place_rank = float(np.mean(rank_place[:, idx]))
                exp_place_percent = float(np.mean(percent_place[:, idx]))
                celeb_rows.append(
                    {
                        "season": int(week.season),
                        "week": int(week.week),
                        "celebrity": celeb,
                        "win_prob_rank": win_rank,
                        "win_prob_percent": win_percent,
                        "delta_win_prob(rank_minus_percent)": win_rank - win_percent,
                        "exp_place_rank": exp_place_rank,
                        "exp_place_percent": exp_place_percent,
                        "delta_exp_place(rank_minus_percent)": exp_place_rank - exp_place_percent,
                    }
                )

    week_df = pd.DataFrame(week_rows)
    celeb_df = pd.DataFrame(celeb_rows)
    return week_df, celeb_df


def analyze_survival_from_tracker(
    tracker_df: pd.DataFrame,
    celebrities: list[str] | None = None,
) -> pd.DataFrame:
    """Convert per-week elimination probabilities into survival/exit summaries.

    Assumption: weekly elimination hazards are treated as independent across weeks
    (a standard approximation under the 'fixed-week sandbox' counterfactual).
    """

    if tracker_df.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "method",
                "celebrity",
                "n_weeks_modeled",
                "survive_prob_through_modeled_weeks",
                "expected_exit_week_index",
                "expected_exit_week_label",
            ]
        )

    df = tracker_df.copy()
    if celebrities is not None:
        df = df[df["celebrity"].isin(celebrities)].copy()

    rows: list[dict[str, object]] = []
    for (season, method, celeb), sub in df.groupby(["season", "method", "celebrity"], sort=True):
        sub = sub.sort_values(["week"]).reset_index(drop=True)
        hazards = sub["elimination_probability"].to_numpy(dtype=float)
        hazards = np.clip(hazards, 0.0, 1.0)
        n = hazards.shape[0]
        if n == 0:
            continue

        # Survival up to (but not including) each week.
        surv_before = np.ones(n, dtype=float)
        if n > 1:
            surv_before[1:] = np.cumprod(1.0 - hazards[:-1])

        exit_prob = surv_before * hazards
        survive_all = float(np.prod(1.0 - hazards))

        # Expected exit week in terms of modeled week index (1..n), with survivors assigned n+1.
        week_index = np.arange(1, n + 1, dtype=float)
        expected_idx = float(np.sum(week_index * exit_prob) + (n + 1) * survive_all)

        weeks = sub["week"].astype(int).tolist()
        # Map expected index to a rough week label by interpolation.
        if expected_idx <= 1:
            exp_label = f"W{weeks[0]}"
        elif expected_idx >= n + 1:
            exp_label = f">W{weeks[-1]}"
        else:
            lo = int(np.floor(expected_idx))
            lo = max(1, min(n, lo))
            exp_label = f"~W{weeks[lo - 1]}"

        rows.append(
            {
                "season": int(season),
                "method": str(method),
                "celebrity": str(celeb),
                "n_weeks_modeled": int(n),
                "survive_prob_through_modeled_weeks": survive_all,
                "expected_exit_week_index": expected_idx,
                "expected_exit_week_label": exp_label,
            }
        )

    return pd.DataFrame(rows)


def plot_finale_heatmap(
    df: pd.DataFrame, title: str, output_path: Path, dpi: int
) -> None:
    plt.figure(figsize=(0.6 * df.shape[1] + 4, 0.5 * df.shape[0] + 3))
    plt.imshow(df.values, aspect="auto", cmap="viridis")
    plt.colorbar(label="Probability")
    plt.xticks(ticks=np.arange(df.shape[1]), labels=df.columns, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(df.shape[0]), labels=df.index)
    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def analyze_trackers(
    core_weeks: list[WeekData],
    names: list[str],
    *,
    method_results_fn=None,
) -> pd.DataFrame:
    rows = []
    for week in _progress(core_weeks, desc="Tracker weeks"):
        results = method_results_fn(week) if method_results_fn is not None else _method_results(week)
        for result in results:
            for name in names:
                if name not in week.contestants:
                    continue
                idx = week.contestants.index(name)
                prob = float(np.mean(result.eliminations == idx))
                rows.append(
                    {
                        "season": week.season,
                        "week": week.week,
                        "method": result.name,
                        "celebrity": name,
                        "elimination_probability": prob,
                    }
                )
    return pd.DataFrame(rows)


def plot_tracker_lines(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    if df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for celeb in df["celebrity"].unique():
        sub = df[df["celebrity"] == celeb].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["season", "week", "method"]).reset_index(drop=True)
        points = (
            sub[["season", "week"]]
            .drop_duplicates()
            .sort_values(["season", "week"])
            .reset_index(drop=True)
        )
        x_labels = [f"S{int(r.season)}W{int(r.week)}" for r in points.itertuples(index=False)]
        x_map = {(int(r.season), int(r.week)): i for i, r in enumerate(points.itertuples(index=False))}

        plt.figure(figsize=(max(8, 0.6 * len(x_labels)), 4.5))
        for method in sub["method"].unique():
            s = sub[sub["method"] == method].sort_values(["season", "week"])
            x = np.array([x_map[(int(a), int(b))] for a, b in zip(s["season"], s["week"])])
            plt.plot(x, s["elimination_probability"], label=method, marker="o", linewidth=1.8)
        plt.title(f"Elimination Probability: {celeb}")
        plt.ylabel("Probability")
        plt.xlabel("Season-Week")
        plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        path = output_dir / f"tracker_{celeb.replace(' ', '_')}.png"
        plt.savefig(path, dpi=dpi)
        plt.close()