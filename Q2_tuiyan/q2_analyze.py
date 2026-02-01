from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def _method_results(week: WeekData) -> list[MethodResult]:
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


def _progress(iterable, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc)


def analyze_core_weeks(core_weeks: list[WeekData]) -> pd.DataFrame:
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

        for result in _method_results(week):
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


def analyze_trackers(core_weeks: list[WeekData], names: list[str]) -> pd.DataFrame:
    rows = []
    for week in _progress(core_weeks, desc="Tracker weeks"):
        for result in _method_results(week):
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
        plt.figure(figsize=(8, 4.5))
        for method in sub["method"].unique():
            s = sub[sub["method"] == method].sort_values(["season", "week"])
            x = np.arange(len(s))
            plt.plot(x, s["elimination_probability"], label=method)
        plt.title(f"Elimination Probability: {celeb}")
        plt.ylabel("Probability")
        plt.xlabel("Time (season-week order)")
        plt.legend()
        plt.tight_layout()
        path = output_dir / f"tracker_{celeb.replace(' ', '_')}.png"
        plt.savefig(path, dpi=dpi)
        plt.close()