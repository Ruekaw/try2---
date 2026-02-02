from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, TypedDict

import numpy as np
import pandas as pd

import Q2_tuiyan.q2_analyze as q2_analyze
from Q2_tuiyan.q2_analyze import analyze_core_weeks
from Q2_tuiyan.q2_loader import LoadResult, WeekData

from .q4_methods import make_daws_method_results_fn


class DawsParams(TypedDict):
    w_early: float
    w_late: float
    n_cut: int
    save_enabled: bool


def _subsample_week(week: WeekData, max_samples: int | None, rng: np.random.Generator) -> WeekData:
    if max_samples is None:
        return week

    s_count = int(week.fan_samples.shape[0])
    max_samples = int(max_samples)
    if max_samples <= 0 or max_samples >= s_count:
        return week

    idx = rng.choice(s_count, size=max_samples, replace=False)
    fan = week.fan_samples[idx, :]

    return WeekData(
        season=week.season,
        week=week.week,
        contestants=week.contestants,
        fan_samples=fan,
        judge_scores=week.judge_scores,
        judge_share=week.judge_share,
        actual_eliminated=week.actual_eliminated,
        is_finale=week.is_finale,
    )


def subsample_loaded(
    loaded: LoadResult,
    *,
    max_samples_per_week: int | None,
    seed: int,
) -> LoadResult:
    rng = np.random.default_rng(int(seed))
    core = [_subsample_week(w, max_samples_per_week, rng) for w in loaded.core_weeks]
    finale = [_subsample_week(w, max_samples_per_week, rng) for w in loaded.finale_weeks]
    return LoadResult(core_weeks=core, finale_weeks=finale, skipped_weeks=loaded.skipped_weeks)


def iter_param_grid(
    *,
    w_early_list: Iterable[float],
    w_late_list: Iterable[float],
    n_cut_list: Iterable[int],
    save_list: Iterable[bool],
) -> Iterator[DawsParams]:
    for w_early in w_early_list:
        for w_late in w_late_list:
            if float(w_late) > float(w_early):
                continue
            for n_cut in n_cut_list:
                for save_enabled in save_list:
                    yield {
                        "w_early": float(w_early),
                        "w_late": float(w_late),
                        "n_cut": int(n_cut),
                        "save_enabled": bool(save_enabled),
                    }


def pareto_frontier(
    df: pd.DataFrame,
    *,
    metrics: list[str],
) -> pd.DataFrame:
    """Return non-dominated rows under minimization for all metrics."""

    if df.empty:
        return df

    vals = df[metrics].to_numpy(dtype=float)
    n = vals.shape[0]
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[i]:
                continue
            # j dominates i if <= on all and < on at least one
            if np.all(vals[j] <= vals[i]) and np.any(vals[j] < vals[i]):
                dominated[i] = True

    return df.loc[~dominated].copy().reset_index(drop=True)


def run_grid_search(
    loaded: LoadResult,
    *,
    out_dir: Path,
    w_early_list: list[float],
    w_late_list: list[float],
    n_cut_list: list[int],
    save_list: list[bool],
    max_samples_per_week: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate DAWS params on Q2 core weeks and write search outputs.

    Writes:
      - daws_search_grid.csv
      - daws_pareto_frontier.csv
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    loaded_sub = subsample_loaded(loaded, max_samples_per_week=max_samples_per_week, seed=seed)

    # Grid search can call analyze_core_weeks hundreds of times; disable tqdm noise.
    old_tqdm = getattr(q2_analyze, "tqdm", None)
    q2_analyze.tqdm = None

    rows: list[dict[str, object]] = []

    try:
        for params in iter_param_grid(
            w_early_list=w_early_list,
            w_late_list=w_late_list,
            n_cut_list=n_cut_list,
            save_list=save_list,
        ):
            w_early = float(params["w_early"])
            w_late = float(params["w_late"])
            n_cut = int(params["n_cut"])
            save_enabled = bool(params["save_enabled"])
            method_results_fn = make_daws_method_results_fn(
                w_early=w_early,
                w_late=w_late,
                n_cut=n_cut,
                save_enabled=save_enabled,
                include_baseline=False,
            )

            core_df = analyze_core_weeks(loaded_sub.core_weeks, method_results_fn=method_results_fn)
            if core_df.empty:
                continue

            agg = (
                core_df.groupby("method")
                .agg(
                    reversal_rate=("reversal_rate", "mean"),
                    tech_vulnerability=("tech_vulnerability", "mean"),
                    popularity_vulnerability=("popularity_vulnerability", "mean"),
                )
                .reset_index()
            )

            for _, r in agg.iterrows():
                if not str(r["method"]).startswith("daws_"):
                    continue
                rows.append(
                    {
                        "w_early": w_early,
                        "w_late": w_late,
                        "n_cut": n_cut,
                        "save_enabled": save_enabled,
                        "method": str(r["method"]),
                        "reversal_rate": float(r["reversal_rate"])
                        if pd.notna(r["reversal_rate"])
                        else np.nan,
                        "tech_vulnerability": float(r["tech_vulnerability"]),
                        "popularity_vulnerability": float(r["popularity_vulnerability"]),
                        "n_weeks": int(core_df[["season", "week"]].drop_duplicates().shape[0]),
                        "n_samples_per_week": int(
                            loaded_sub.core_weeks[0].fan_samples.shape[0]
                            if loaded_sub.core_weeks
                            else 0
                        ),
                    }
                )
    finally:
        q2_analyze.tqdm = old_tqdm

    grid_df = pd.DataFrame(rows)
    grid_path = out_dir / "daws_search_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    metrics = ["reversal_rate", "tech_vulnerability", "popularity_vulnerability"]
    pareto_df = pareto_frontier(grid_df.dropna(subset=metrics), metrics=metrics)
    pareto_path = out_dir / "daws_pareto_frontier.csv"
    pareto_df.to_csv(pareto_path, index=False)

    return grid_df, pareto_df
