from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
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


def _minmax_norm(series: pd.Series) -> pd.Series:
    x = series.astype(float)
    lo = float(x.min())
    hi = float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(x), dtype=float), index=series.index)
    return (x - lo) / (hi - lo)


def write_recommendations(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    prefer_method: str | None = None,
) -> pd.DataFrame:
    """Pick recommended parameter sets and write to disk.

    We keep the full grid for transparency, but also provide a single-row
    recommendation for common objectives to avoid manual cherry-picking.

    Objectives (all are minimization):
      - fairness: tech_vulnerability first
      - audience: popularity_vulnerability first
      - balanced: weighted sum of min-max normalized metrics
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        rec = pd.DataFrame([])
        rec.to_csv(out_dir / "daws_recommendations.csv", index=False)
        (out_dir / "daws_recommendations.md").write_text(
            "# DAWS 推荐参数\n\n(无可用搜索结果)\n",
            encoding="utf-8",
        )
        return rec

    metrics = ["reversal_rate", "tech_vulnerability", "popularity_vulnerability"]
    sub = df.dropna(subset=metrics).copy()
    if prefer_method is not None:
        sub = sub[sub["method"].astype(str) == str(prefer_method)].copy()
    if sub.empty:
        sub = df.dropna(subset=metrics).copy()

    # Ensure stable ordering
    sub = sub.sort_values(
        ["method", "save_enabled", "n_cut", "w_early", "w_late"],
        kind="mergesort",
    ).reset_index(drop=True)

    fairness = sub.sort_values(
        ["tech_vulnerability", "reversal_rate", "popularity_vulnerability"],
        kind="mergesort",
    ).head(1)

    audience = sub.sort_values(
        ["popularity_vulnerability", "reversal_rate", "tech_vulnerability"],
        kind="mergesort",
    ).head(1)

    # Balanced: normalize metrics then weighted sum (tech emphasized)
    w = {"reversal_rate": 1.0, "tech_vulnerability": 2.0, "popularity_vulnerability": 1.0}
    tmp = sub.copy()
    tmp["reversal_norm"] = _minmax_norm(tmp["reversal_rate"])
    tmp["tech_norm"] = _minmax_norm(tmp["tech_vulnerability"])
    tmp["pop_norm"] = _minmax_norm(tmp["popularity_vulnerability"])
    tmp["balanced_score"] = (
        w["reversal_rate"] * tmp["reversal_norm"]
        + w["tech_vulnerability"] * tmp["tech_norm"]
        + w["popularity_vulnerability"] * tmp["pop_norm"]
    )
    balanced = tmp.sort_values(["balanced_score"], kind="mergesort").head(1)

    def _format_row(row: pd.Series) -> str:
        return (
            f"method={row['method']}, save={bool(row['save_enabled'])}, "
            f"w_early={row['w_early']}, w_late={row['w_late']}, n_cut={int(row['n_cut'])}"
        )

    rec = pd.concat(
        [
            fairness.assign(objective="fairness"),
            audience.assign(objective="audience"),
            balanced.assign(objective="balanced"),
        ],
        ignore_index=True,
    )

    cols = [
        "objective",
        "method",
        "w_early",
        "w_late",
        "n_cut",
        "save_enabled",
        "reversal_rate",
        "tech_vulnerability",
        "popularity_vulnerability",
        "n_weeks",
        "n_samples_per_week",
    ]
    rec_out = rec[[c for c in cols if c in rec.columns]].copy()
    rec_out.to_csv(out_dir / "daws_recommendations.csv", index=False)

    md_lines = [
        "# DAWS 推荐参数",
        "",
        "说明：搜索输出保留完整网格（透明可追溯），但同时给出自动推荐参数，避免人工挑选。",
        "",
    ]

    md_lines += [
        "## fairness（公平优先：最小化 tech_vulnerability）",
        "- " + _format_row(fairness.iloc[0]),
        "",
        "## audience（观众优先：最小化 popularity_vulnerability）",
        "- " + _format_row(audience.iloc[0]),
        "",
        "## balanced（折衷：min-max 归一化后加权求和，tech 权重更高）",
        "- " + _format_row(balanced.iloc[0]),
        "",
        "使用方式：选定其中一个 objective 的参数后，用 `q4_main.py --eval` 固定参数出最终图表；若需要 DAWS-Save 曲线请加 `--save`。",
    ]

    (out_dir / "daws_recommendations.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return rec_out


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
    n_jobs: int = 1,
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

    def _eval_params(params: DawsParams) -> list[dict[str, object]]:
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
            return []

        agg = (
            core_df.groupby("method")
            .agg(
                reversal_rate=("reversal_rate", "mean"),
                tech_vulnerability=("tech_vulnerability", "mean"),
                popularity_vulnerability=("popularity_vulnerability", "mean"),
            )
            .reset_index()
        )

        out_rows: list[dict[str, object]] = []
        for _, r in agg.iterrows():
            if not str(r["method"]).startswith("daws_"):
                continue
            out_rows.append(
                {
                    "w_early": w_early,
                    "w_late": w_late,
                    "n_cut": n_cut,
                    "save_enabled": save_enabled,
                    "method": str(r["method"]),
                    "reversal_rate": float(r["reversal_rate"]) if pd.notna(r["reversal_rate"]) else np.nan,
                    "tech_vulnerability": float(r["tech_vulnerability"]),
                    "popularity_vulnerability": float(r["popularity_vulnerability"]),
                    "n_weeks": int(core_df[["season", "week"]].drop_duplicates().shape[0]),
                    "n_samples_per_week": int(
                        loaded_sub.core_weeks[0].fan_samples.shape[0] if loaded_sub.core_weeks else 0
                    ),
                }
            )
        return out_rows

    rows: list[dict[str, object]] = []
    all_params = list(
        iter_param_grid(
            w_early_list=w_early_list,
            w_late_list=w_late_list,
            n_cut_list=n_cut_list,
            save_list=save_list,
        )
    )
    total = len(all_params)
    if total == 0:
        grid_df = pd.DataFrame(rows)
        (out_dir / "daws_search_grid.csv").write_text("", encoding="utf-8")
        (out_dir / "daws_pareto_frontier.csv").write_text("", encoding="utf-8")
        return grid_df, grid_df

    try:
        n_jobs = int(n_jobs)
        if n_jobs <= 1:
            for i, params in enumerate(all_params, start=1):
                if total >= 25 and (i == 1 or i % 10 == 0 or i == total):
                    print(f"[Q4-search] {i}/{total} params...")
                rows.extend(_eval_params(params))
        else:
            # Threading works well here because heavy lifting is NumPy (releases GIL)
            # and avoids pickling huge arrays on Windows (spawn).
            max_workers = min(n_jobs, max(1, (os.cpu_count() or 1)))
            print(f"[Q4-search] running with jobs={max_workers}, params={total}")
            done = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_eval_params, p) for p in all_params]
                for fut in as_completed(futures):
                    rows.extend(fut.result())
                    done += 1
                    if total >= 25 and (done % 10 == 0 or done == total):
                        print(f"[Q4-search] {done}/{total} params done")
    finally:
        q2_analyze.tqdm = old_tqdm

    grid_df = pd.DataFrame(rows)
    if not grid_df.empty:
        grid_df = grid_df.sort_values(
            ["method", "save_enabled", "n_cut", "w_early", "w_late"],
            kind="mergesort",
        ).reset_index(drop=True)
    grid_path = out_dir / "daws_search_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    metrics = ["reversal_rate", "tech_vulnerability", "popularity_vulnerability"]
    pareto_df = pareto_frontier(grid_df.dropna(subset=metrics), metrics=metrics)
    pareto_path = out_dir / "daws_pareto_frontier.csv"
    pareto_df.to_csv(pareto_path, index=False)

    # Convenience: also write a single recommended parameter set per objective.
    # Prefer the Pareto set to avoid dominated choices.
    _ = write_recommendations(pareto_df if not pareto_df.empty else grid_df, out_dir=out_dir)

    return grid_df, pareto_df
