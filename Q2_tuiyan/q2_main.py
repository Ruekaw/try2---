from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from Q2_tuiyan.q2_config import default_config, season_segment_combine_method
    from Q2_tuiyan.q2_loader import load_q2_data
    from Q2_tuiyan.q2_analyze import (
        analyze_core_weeks,
        analyze_finales,
        analyze_trackers,
        plot_finale_heatmap,
        plot_tracker_lines,
    )
else:
    from .q2_config import default_config, season_segment_combine_method
    from .q2_loader import load_q2_data
    from .q2_analyze import (
        analyze_core_weeks,
        analyze_finales,
        analyze_trackers,
        plot_finale_heatmap,
        plot_tracker_lines,
    )


def run(out_dir: Path, tracker_names: list[str], dpi: int) -> None:
    cfg = default_config()
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_q2_data(cfg.clean_csv, cfg.npz_dir)

    core_df = analyze_core_weeks(loaded.core_weeks)
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

    finale_outputs = analyze_finales(loaded.finale_weeks)
    finale_dir = out_dir / "finale_distributions"
    finale_dir.mkdir(parents=True, exist_ok=True)

    for key, df in finale_outputs.items():
        df.to_csv(finale_dir / f"{key}.csv")
        plot_finale_heatmap(
            df,
            title=f"Finale Placement Distribution ({key})",
            output_path=finale_dir / f"{key}.png",
            dpi=dpi,
        )

    tracker_df = analyze_trackers(loaded.core_weeks, tracker_names)
    tracker_df.to_csv(out_dir / "tracker_elimination_probabilities.csv", index=False)
    plot_tracker_lines(tracker_df, out_dir / "trackers", dpi=dpi)

    pd.DataFrame(loaded.skipped_weeks).to_csv(out_dir / "skipped_weeks.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2 counterfactual engine")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default outputs/q2_counterfactual)",
    )
    parser.add_argument(
        "--tracker",
        nargs="*",
        default=None,
        help="Celebrity names for elimination probability tracking",
    )
    parser.add_argument("--dpi", type=int, default=None, help="PNG dpi")
    args = parser.parse_args()

    cfg = default_config()
    out_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir
    tracker = args.tracker if args.tracker else list(cfg.tracker_names)
    dpi = args.dpi if args.dpi else cfg.png_dpi

    run(out_dir=out_dir, tracker_names=tracker, dpi=dpi)


if __name__ == "__main__":
    main()