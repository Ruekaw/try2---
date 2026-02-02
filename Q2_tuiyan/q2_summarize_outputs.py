from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_csv(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    df = pd.read_csv(path)
    print(f"\n=== {name} ===")
    print(f"shape={df.shape}")
    print(f"columns={list(df.columns)}")
    print(df.head(5).to_string(index=False))
    return df


def main() -> None:
    base = Path(r"d:\WorkSpace\MCM\try2-规划\outputs\q2_counterfactual")

    rank_overall = _read_csv(base, "rank_vs_percent_overall.csv")
    core_overall = _read_csv(base, "core_metrics_overall.csv")
    finale_overall = _read_csv(base, "finale_winner_change_overall.csv")
    skipped = _read_csv(base, "skipped_weeks.csv")

    try:
        missing_sum = _read_csv(base, "missing_in_npz_week_classification_summary.csv")
    except FileNotFoundError:
        missing_sum = None
        print("\n(missing_in_npz_week_classification_summary.csv not found)")

    core_by_week = pd.read_csv(base / "core_metrics_by_week.csv")
    print("\n=== Coverage stats ===")
    print("core_metrics_by_week rows:", len(core_by_week))
    if "season" in core_by_week.columns:
        print("unique seasons in core:", core_by_week["season"].nunique())

    if len(skipped) > 0:
        print("skipped_weeks rows:", len(skipped))
        if "reason" in skipped.columns:
            print("skip reason counts (top 20):")
            print(skipped["reason"].value_counts().head(20).to_string())

    # Rank vs Percent: season extremes
    rank_by_season = pd.read_csv(base / "rank_vs_percent_by_season.csv")
    if "disagreement_rate" in rank_by_season.columns:
        top = rank_by_season.sort_values("disagreement_rate", ascending=False).head(8)
        bot = rank_by_season.sort_values("disagreement_rate", ascending=True).head(8)
        print("\n=== Rank vs Percent disagreement_rate (top seasons) ===")
        cols = [c for c in ["season", "n_weeks", "disagreement_rate"] if c in top.columns]
        print(top[cols].to_string(index=False))
        print("\n=== Rank vs Percent disagreement_rate (bottom seasons) ===")
        print(bot[cols].to_string(index=False))

    # Save vs Direct core metrics
    core_by_method = core_overall.copy()
    if "method" in core_by_method.columns:
        show_cols = [c for c in [
            "method",
            "n_weeks",
            "reversal_rate",
            "tech_vulnerability",
            "popularity_vulnerability",
        ] if c in core_by_method.columns]
        print("\n=== Core metrics overall (by method) ===")
        print(core_by_method[show_cols].to_string(index=False))

    # Finale winner disagreement
    if "winner_disagreement_rate" in finale_overall.columns:
        print("\n=== Finale winner disagreement (overall) ===")
        show_cols = [c for c in ["winner_disagreement_rate", "n_finales"] if c in finale_overall.columns]
        print(finale_overall[show_cols].to_string(index=False))

    # Controversy deltas
    deltas = pd.read_csv(base / "finale_celebrity_outcome_deltas.csv")
    print("\n=== Controversy celebrity outcome deltas ===")
    print(deltas.to_string(index=False))

    tracker_surv = pd.read_csv(base / "tracker_survival_summary.csv")
    print("\n=== Tracker survival summary (all rows) ===")
    print(tracker_surv.to_string(index=False))


if __name__ == "__main__":
    main()
