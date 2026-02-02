from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from Q2_tuiyan.q2_analyze import (
        analyze_core_weeks,
        analyze_rank_vs_percent,
        analyze_survival_from_tracker,
        analyze_trackers,
        plot_finale_heatmap,
        plot_season_week_metric_heatmap,
        plot_tracker_lines,
    )
    from Q2_tuiyan.q2_config import default_config, season_segment_combine_method
    from Q2_tuiyan.q2_kernel import (
        simulate_finale_rank_distribution,
        simulate_finale_percent_distribution,
    )
    from Q2_tuiyan.q2_loader import load_q2_data

    from Q4_new_model.q4_daws_kernel import simulate_finale_daws_distribution
    from Q4_new_model.q4_methods import compute_w_fan, make_daws_method_results_fn
    from Q4_new_model.q4_search import run_grid_search, subsample_loaded
else:
    from Q2_tuiyan.q2_analyze import (
        analyze_core_weeks,
        analyze_rank_vs_percent,
        analyze_survival_from_tracker,
        analyze_trackers,
        plot_finale_heatmap,
        plot_season_week_metric_heatmap,
        plot_tracker_lines,
    )
    from Q2_tuiyan.q2_config import default_config, season_segment_combine_method
    from Q2_tuiyan.q2_kernel import (
        simulate_finale_rank_distribution,
        simulate_finale_percent_distribution,
    )
    from Q2_tuiyan.q2_loader import load_q2_data

    from .q4_daws_kernel import simulate_finale_daws_distribution
    from .q4_methods import compute_w_fan, make_daws_method_results_fn
    from .q4_search import run_grid_search, subsample_loaded


def _finale_probability_table(placements: np.ndarray, contestants: list[str]) -> pd.DataFrame:
    s_count, n = placements.shape
    probs = np.zeros((n, n), dtype=float)
    for i in range(n):
        for p in range(1, n + 1):
            probs[i, p - 1] = float(np.mean(placements[:, i] == p))
    return pd.DataFrame(
        probs,
        index=contestants,
        columns=[f"place_{p}" for p in range(1, n + 1)],
    )


def _finale_winner_disagreement(place_a: np.ndarray, place_b: np.ndarray) -> float:
    winner_a = np.argmin(place_a, axis=1)
    winner_b = np.argmin(place_b, axis=1)
    return float(np.mean(winner_a != winner_b))


def run_eval(
    *,
    clean_csv: Path,
    npz_dir: Path,
    out_dir: Path,
    w_early: float,
    w_late: float,
    n_cut: int,
    save_enabled: bool,
    tracker_names: list[str],
    dpi: int,
    max_samples_per_week: int | None,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_q2_data(clean_csv, npz_dir)
    loaded = subsample_loaded(loaded, max_samples_per_week=max_samples_per_week, seed=seed)

    method_results_fn = make_daws_method_results_fn(
        w_early=w_early,
        w_late=w_late,
        n_cut=n_cut,
        save_enabled=save_enabled,
        include_baseline=True,
    )

    # Core week metrics (baseline + DAWS)
    core_df = analyze_core_weeks(loaded.core_weeks, method_results_fn=method_results_fn)
    if not core_df.empty:
        core_df["season_method"] = core_df["season"].map(season_segment_combine_method)
    core_df.to_csv(out_dir / "core_metrics_by_week.csv", index=False)

    if not core_df.empty:
        core_df.groupby("method").agg(
            reversal_rate=("reversal_rate", "mean"),
            tech_vulnerability=("tech_vulnerability", "mean"),
            popularity_vulnerability=("popularity_vulnerability", "mean"),
        ).reset_index().to_csv(out_dir / "core_metrics_overall.csv", index=False)

        core_df.groupby(["season", "method"]).agg(
            reversal_rate=("reversal_rate", "mean"),
            tech_vulnerability=("tech_vulnerability", "mean"),
            popularity_vulnerability=("popularity_vulnerability", "mean"),
        ).reset_index().to_csv(out_dir / "core_metrics_by_season.csv", index=False)

    # Rank vs Percent baseline diagnostic
    rp_week = analyze_rank_vs_percent(loaded.core_weeks)
    if not rp_week.empty:
        rp_week["season_method"] = rp_week["season"].map(season_segment_combine_method)
    rp_week.to_csv(out_dir / "rank_vs_percent_by_week.csv", index=False)

    if not rp_week.empty:
        rp_week.groupby("season").agg(
            disagreement_rate=("disagreement_rate", "mean"),
            rank_more_fan_friendly_rate_if_disagree=(
                "rank_more_fan_friendly_rate_if_disagree",
                "mean",
            ),
            rank_more_judge_friendly_rate_if_disagree=(
                "rank_more_judge_friendly_rate_if_disagree",
                "mean",
            ),
        ).reset_index().to_csv(out_dir / "rank_vs_percent_by_season.csv", index=False)

        rp_overall = (
            rp_week[
                [
                    "disagreement_rate",
                    "rank_more_fan_friendly_rate_if_disagree",
                    "rank_more_judge_friendly_rate_if_disagree",
                ]
            ]
            .mean(numeric_only=True)
            .to_frame()
            .T
        )
        rp_overall.to_csv(out_dir / "rank_vs_percent_overall.csv", index=False)

        plot_season_week_metric_heatmap(
            rp_week,
            season=27,
            metrics=[
                "disagreement_rate",
                "rank_more_fan_friendly_rate_if_disagree",
                "rank_more_judge_friendly_rate_if_disagree",
            ],
            title="S27: Rank vs Percent (Direct) - Disagreement & Direction",
            output_path=out_dir / "rank_vs_percent_S27_heatmap.png",
            dpi=dpi,
        )

    # Finale distributions: rank/percent (baseline) + daws
    finale_dir = out_dir / "finale_distributions"
    finale_dir.mkdir(parents=True, exist_ok=True)

    finale_change_rows: list[dict[str, object]] = []

    for week in loaded.finale_weeks:
        rank_place = simulate_finale_rank_distribution(
            week.fan_samples, week.judge_scores, week.contestants
        )
        percent_place = simulate_finale_percent_distribution(
            week.fan_samples, week.judge_share, week.judge_scores, week.contestants
        )

        w_fan = compute_w_fan(
            n_contestants=len(week.contestants),
            w_early=w_early,
            w_late=w_late,
            n_cut=n_cut,
        )
        daws_place = simulate_finale_daws_distribution(
            week.fan_samples,
            week.judge_share,
            week.judge_scores,
            week.contestants,
            w_fan,
        )

        for method, placements in (
            ("rank", rank_place),
            ("percent", percent_place),
            ("daws", daws_place),
        ):
            df = _finale_probability_table(placements, week.contestants)
            key = f"S{week.season}_W{week.week}_{method}"
            df.to_csv(finale_dir / f"{key}.csv")
            plot_finale_heatmap(
                df,
                title=f"Finale Placement Distribution ({key})",
                output_path=finale_dir / f"{key}.png",
                dpi=dpi,
            )

        finale_change_rows.append(
            {
                "season": int(week.season),
                "week": int(week.week),
                "n_finalists": int(len(week.contestants)),
                "winner_disagreement_rate_rank_vs_percent": _finale_winner_disagreement(
                    rank_place, percent_place
                ),
                "winner_disagreement_rate_daws_vs_percent": _finale_winner_disagreement(
                    daws_place, percent_place
                ),
                "winner_disagreement_rate_daws_vs_rank": _finale_winner_disagreement(
                    daws_place, rank_place
                ),
            }
        )

    pd.DataFrame(finale_change_rows).to_csv(
        out_dir / "finale_winner_change_with_daws_by_week.csv", index=False
    )

    # Trackers (baseline + DAWS)
    tracker_df = analyze_trackers(
        loaded.core_weeks, tracker_names, method_results_fn=method_results_fn
    )
    tracker_df.to_csv(out_dir / "tracker_elimination_probabilities.csv", index=False)
    plot_tracker_lines(tracker_df, out_dir / "trackers", dpi=dpi)

    survival_df = analyze_survival_from_tracker(tracker_df, celebrities=tracker_names)
    survival_df.to_csv(out_dir / "tracker_survival_summary.csv", index=False)

    pd.DataFrame(loaded.skipped_weeks).to_csv(out_dir / "skipped_weeks.csv", index=False)


def run_search_mode(
    *,
    clean_csv: Path,
    npz_dir: Path,
    out_dir: Path,
    w_early_list: list[float],
    w_late_list: list[float],
    n_cut_list: list[int],
    save_list: list[bool],
    max_samples_per_week: int | None,
    seed: int,
    jobs: int,
) -> None:
    loaded = load_q2_data(clean_csv, npz_dir)
    run_grid_search(
        loaded,
        out_dir=out_dir,
        w_early_list=w_early_list,
        w_late_list=w_late_list,
        n_cut_list=n_cut_list,
        save_list=save_list,
        max_samples_per_week=max_samples_per_week,
        seed=seed,
        n_jobs=int(jobs),
    )


def main() -> None:
    cfg = default_config()

    parser = argparse.ArgumentParser(description="Q4 DAWS runner (plugs into Q2 engine)")

    parser.add_argument(
        "--clean-csv",
        default=str(cfg.clean_csv),
        help="Absolute path to cleaned long CSV",
    )
    parser.add_argument(
        "--npz-dir",
        default=str(cfg.npz_dir),
        help="Absolute path to Q1 exported npz directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(cfg.root_dir / "outputs" / "q4_new_system"),
        help="Absolute path to output directory",
    )

    parser.add_argument(
        "--search-subdir",
        default="search",
        help="Subdirectory name under output-dir for grid search outputs",
    )

    # DAWS parameters for eval
    parser.add_argument("--w-early", type=float, default=0.65)
    parser.add_argument("--w-late", type=float, default=0.35)
    parser.add_argument("--n-cut", type=int, default=5)
    parser.add_argument("--save", action="store_true", help="Enable DAWS-Save")

    # Run modes
    parser.add_argument("--eval", action="store_true", help="Run evaluation outputs")
    parser.add_argument("--search", action="store_true", help="Run grid search outputs")

    # Search grids
    parser.add_argument(
        "--grid-w-early",
        default="0.4,0.5,0.6,0.7,0.8",
        help="Comma-separated coarse grid for w_early",
    )
    parser.add_argument(
        "--grid-w-late",
        default="0.0,0.1,0.2,0.3,0.4",
        help="Comma-separated coarse grid for w_late",
    )
    parser.add_argument(
        "--grid-n-cut",
        default="4,5,6",
        help="Comma-separated grid for n_cut",
    )
    parser.add_argument(
        "--grid-save",
        default="0,1",
        help="Comma-separated grid for save_enabled (0/1)",
    )

    # Performance/reproducibility
    parser.add_argument(
        "--max-samples-per-week",
        type=int,
        default=None,
        help="Subsample posterior draws per week for speed (e.g., 1000)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallelism for grid search (threads). Use 1 to disable.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dpi", type=int, default=cfg.png_dpi)

    args = parser.parse_args()

    clean_csv = Path(args.clean_csv).resolve()
    npz_dir = Path(args.npz_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    run_eval_flag = args.eval or (not args.eval and not args.search)
    run_search_flag = args.search

    if run_eval_flag:
        run_eval(
            clean_csv=clean_csv,
            npz_dir=npz_dir,
            out_dir=out_dir,
            w_early=float(args.w_early),
            w_late=float(args.w_late),
            n_cut=int(args.n_cut),
            save_enabled=bool(args.save),
            tracker_names=list(cfg.tracker_names),
            dpi=int(args.dpi),
            max_samples_per_week=args.max_samples_per_week,
            seed=int(args.seed),
        )

    if run_search_flag:
        def _parse_float_list(s: str) -> list[float]:
            return [float(x.strip()) for x in s.split(",") if x.strip()]

        def _parse_int_list(s: str) -> list[int]:
            return [int(x.strip()) for x in s.split(",") if x.strip()]

        w_early_list = _parse_float_list(args.grid_w_early)
        w_late_list = _parse_float_list(args.grid_w_late)
        n_cut_list = _parse_int_list(args.grid_n_cut)
        save_list = [bool(int(x)) for x in args.grid_save.split(",") if x.strip()]

        run_search_mode(
            clean_csv=clean_csv,
            npz_dir=npz_dir,
            out_dir=out_dir / str(args.search_subdir),
            w_early_list=w_early_list,
            w_late_list=w_late_list,
            n_cut_list=n_cut_list,
            save_list=save_list,
            max_samples_per_week=args.max_samples_per_week,
            seed=int(args.seed),
            jobs=int(args.jobs),
        )


if __name__ == "__main__":
    main()
