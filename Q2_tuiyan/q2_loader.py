from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WeekData:
    season: int
    week: int
    contestants: list[str]
    fan_samples: np.ndarray  # shape: (S, N)
    judge_scores: np.ndarray  # shape: (N,)
    judge_share: np.ndarray  # shape: (N,)
    actual_eliminated: list[str]
    is_finale: bool


@dataclass(frozen=True)
class LoadResult:
    core_weeks: list[WeekData]
    finale_weeks: list[WeekData]
    skipped_weeks: list[dict[str, Any]]


def _normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    ).fillna(False)


def _normalize_name(name: str) -> str:
    return "".join(name.lower().strip().split())


def _get_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _extract_actual_eliminated(group: pd.DataFrame, week: int, is_finale: bool) -> list[str]:
    if is_finale:
        return []
    for col in ("is_eliminated_this_week", "eliminated_this_week"):
        if col in group.columns:
            flag = _normalize_bool(group[col])
            return group.loc[flag, "celebrity_name"].tolist()
    for col in ("last_week_scored", "computed_last_week"):
        if col in group.columns:
            return group.loc[group[col] == week, "celebrity_name"].tolist()
    return []


def _contestant_alignment(
    group: pd.DataFrame, contestants: list[str]
) -> tuple[pd.DataFrame | None, str | None]:
    name_col = "celebrity_name"
    if name_col not in group.columns:
        return None, "missing celebrity_name"

    group = group.copy()
    group["_norm_name"] = group[name_col].astype(str).map(_normalize_name)
    index_map: dict[str, list[int]] = {}
    for i, n in enumerate(group["_norm_name"].tolist()):
        index_map.setdefault(n, []).append(i)

    aligned_rows = []
    for raw in contestants:
        n = _normalize_name(raw)
        if n not in index_map:
            return None, f"name not found: {raw}"
        if len(index_map[n]) > 1:
            exact = group[group[name_col] == raw]
            if len(exact) == 1:
                aligned_rows.append(exact.iloc[0])
                continue
            return None, f"ambiguous name: {raw}"
        aligned_rows.append(group.iloc[index_map[n][0]])

    aligned = pd.DataFrame(aligned_rows).reset_index(drop=True)
    return aligned, None


def load_q2_data(clean_csv: Path, npz_dir: Path) -> LoadResult:
    df = pd.read_csv(clean_csv)

    season_col = _get_first_existing(df, ["season", "Season"]) or "season"
    week_col = _get_first_existing(df, ["week", "Week"]) or "week"

    if season_col not in df.columns or week_col not in df.columns:
        raise ValueError("CSV missing season/week columns.")

    if "is_present" in df.columns:
        df["_is_present"] = _normalize_bool(df["is_present"])
    elif "is_competing_week" in df.columns:
        df["_is_present"] = _normalize_bool(df["is_competing_week"])
    else:
        df["_is_present"] = True

    if "exclude_from_fan_vote_inference" in df.columns:
        df["_exclude"] = _normalize_bool(df["exclude_from_fan_vote_inference"])
    else:
        df["_exclude"] = False

    if "is_double_elimination" in df.columns:
        df["_double_elim"] = _normalize_bool(df["is_double_elimination"])
    else:
        df["_double_elim"] = False

    no_elim_cols = [c for c in ["is_no_elimination", "is_no_elimination_any"] if c in df.columns]
    if no_elim_cols:
        no_elim_vals = pd.DataFrame({c: _normalize_bool(df[c]) for c in no_elim_cols})
        df["_no_elim"] = no_elim_vals.any(axis=1)
    else:
        df["_no_elim"] = False

    if "is_final_week" in df.columns:
        df["_final_week"] = _normalize_bool(df["is_final_week"])
    else:
        df["_final_week"] = False

    core_weeks: list[WeekData] = []
    finale_weeks: list[WeekData] = []
    skipped_weeks: list[dict[str, Any]] = []

    npz_files = sorted(npz_dir.glob("season_*_samples.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No npz files in {npz_dir}")

    for npz_path in npz_files:
        season = int(npz_path.stem.split("_")[1])
        season_df = df[df[season_col] == season]
        with np.load(npz_path, allow_pickle=True) as data:
            weeks = data.get("weeks")
            if weeks is None:
                skipped_weeks.append(
                    {"season": season, "week": None, "reason": "npz missing weeks"}
                )
                continue
            weeks = [int(w) for w in weeks.tolist()]

            for wk in weeks:
                week_group = season_df[season_df[week_col] == wk]
                if week_group.empty:
                    skipped_weeks.append(
                        {"season": season, "week": wk, "reason": "week missing in csv"}
                    )
                    continue

                is_double = week_group["_double_elim"].any()
                is_no_elim = week_group["_no_elim"].any()
                is_finale = week_group["_final_week"].any()

                if is_double:
                    skipped_weeks.append(
                        {"season": season, "week": wk, "reason": "double elimination"}
                    )
                    continue
                if is_no_elim:
                    skipped_weeks.append(
                        {"season": season, "week": wk, "reason": "no elimination"}
                    )
                    continue

                names_key = f"week_{wk}_contestants"
                samples_key = f"week_{wk}_samples"
                if names_key not in data or samples_key not in data:
                    alt_names = f"week_{wk}_names"
                    if alt_names in data:
                        names_key = alt_names
                    else:
                        skipped_weeks.append(
                            {"season": season, "week": wk, "reason": "missing contestants key"}
                        )
                        continue
                contestants = [str(x) for x in data[names_key].tolist()]
                fan_samples = np.asarray(data[samples_key], dtype=float)

                week_group = week_group[week_group["_is_present"] & ~week_group["_exclude"]]
                if week_group.empty:
                    skipped_weeks.append(
                        {"season": season, "week": wk, "reason": "no valid contestants"}
                    )
                    continue

                aligned, err = _contestant_alignment(week_group, contestants)
                if aligned is None:
                    skipped_weeks.append(
                        {"season": season, "week": wk, "reason": err}
                    )
                    continue

                if "week_total_score" not in aligned.columns or "week_score_share" not in aligned.columns:
                    skipped_weeks.append(
                        {
                            "season": season,
                            "week": wk,
                            "reason": "missing judge score columns",
                        }
                    )
                    continue

                judge_scores = aligned["week_total_score"].to_numpy(dtype=float)
                judge_share = aligned["week_score_share"].to_numpy(dtype=float)
                actual_elim = _extract_actual_eliminated(week_group, wk, is_finale)

                week_data = WeekData(
                    season=season,
                    week=wk,
                    contestants=contestants,
                    fan_samples=fan_samples,
                    judge_scores=judge_scores,
                    judge_share=judge_share,
                    actual_eliminated=actual_elim,
                    is_finale=is_finale,
                )

                if is_finale:
                    finale_weeks.append(week_data)
                else:
                    core_weeks.append(week_data)

    return LoadResult(core_weeks=core_weeks, finale_weeks=finale_weeks, skipped_weeks=skipped_weeks)