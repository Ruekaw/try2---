"""DWTS (MCM 2026 C) data cleaning + feature engineering.

Input: wide CSV with columns like week{X}_judge{Y}_score.
Output: long (contestant-season-week) table + season summary + partner summary + QA report.

Usage (PowerShell):
  python clean_dwts.py --input "3-27赛季（百分比结合法）.csv" "剩下赛季（排名结合法）.csv" --outdir outputs

Notes:
- 0 scores are treated as "not present" (eliminated/withdrew) for within-week ranks and percentages.
    IMPORTANT: We assume there is NO "real 0" score in DWTS; 0 always means absence. This is validated in QA.
- N/A indicates structural missing (no judge4, or weeks not run). Weeks not run are dropped.
- Out-of-range judge scores (by default <1 or >20) are *flagged*. Use --allow-bonus to keep these in cleaned totals (still flagged).
- Withdrew contestants are explicitly tagged; downstream models should exclude them from fan vote inference.
- Partner historical stats have a "temporal leakage" variant (_prior) for strict time-ordered analysis.
- No-elimination weeks are detected and flagged via `is_no_elimination` / `is_no_elim_from_results`.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


WEEK_JUDGE_RE = re.compile(r"^week(?P<week>\d+)_judge(?P<judge>\d+)_score$")


US_STATE_TO_ABBR = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "district of columbia": "DC",
    "washington d.c.": "DC",
    "washington dc": "DC",
}


def _strip_or_nan(value: object) -> object:
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        s = value.strip()
        return s if s else np.nan
    return value


def normalize_country(value: object) -> object:
    value = _strip_or_nan(value)
    if pd.isna(value):
        return np.nan
    s = str(value)
    s_norm = re.sub(r"\s+", " ", s).strip()
    s_low = s_norm.lower()
    us_aliases = {"united states", "united states of america", "usa", "u.s.", "u.s.a.", "us", "u s"}
    if s_low in us_aliases:
        return "United States"
    return s_norm


def normalize_state(value: object) -> tuple[object, object]:
    value = _strip_or_nan(value)
    if pd.isna(value):
        return np.nan, np.nan
    s = str(value)
    s_norm = re.sub(r"\s+", " ", s).strip()
    s_low = s_norm.lower()

    # already an abbreviation
    if re.fullmatch(r"[A-Za-z]{2}", s_norm):
        return s_norm.upper(), s_norm.upper()

    abbr = US_STATE_TO_ABBR.get(s_low)
    return s_norm, abbr


def normalize_industry(raw: object) -> object:
    raw = _strip_or_nan(raw)
    if pd.isna(raw):
        return np.nan

    s = str(raw).strip()
    s_low = s.lower()

    def has_any(*keywords: str) -> bool:
        return any(k in s_low for k in keywords)

    if has_any("athlete", "nfl", "nba", "mlb", "nhl", "olympic", "swimmer", "skater", "fighter", "wrest", "boxing", "racing", "driver", "gymnast"):
        return "Athlete"
    if has_any("singer", "rapper", "musician", "band", "dj"):
        return "Musician"
    if has_any("actor", "actress"):
        return "Actor"
    if has_any("model", "beauty"):
        return "Model"
    if has_any("politician"):
        return "Politician"
    if has_any("tv personality", "reality", "social media"):
        return "Reality TV Star"
    if has_any("comedian", "host", "radio", "news", "anchor", "broadcaster"):
        return "TV/Host"

    return "Other"


# Possible result outcomes for more structured parsing
RESULT_ELIMINATED = "eliminated"
RESULT_WITHDREW = "withdrew"
RESULT_WINNER = "winner"
RESULT_FINALIST = "finalist"  # 2nd/3rd place
RESULT_ONGOING = "ongoing"  # still competing
RESULT_UNKNOWN = "unknown"


def parse_result_type(results_str: object) -> str:
    """Parse the results string into a canonical type."""
    if pd.isna(results_str):
        return RESULT_UNKNOWN
    s = str(results_str).lower().strip()
    if "withdrew" in s:
        return RESULT_WITHDREW
    if "eliminated" in s:
        return RESULT_ELIMINATED
    if "1st" in s or "winner" in s:
        return RESULT_WINNER
    if "2nd" in s or "3rd" in s or "place" in s:
        return RESULT_FINALIST
    return RESULT_UNKNOWN


@dataclass(frozen=True)
class ScorePolicy:
    strict_range: bool = True
    min_score: float = 1.0
    max_score: float = 20.0  # Allow bonus points as had 11, 12 scores)


def coerce_numeric(series: pd.Series) -> pd.Series:
    # Keep N/A as NaN; allow numeric strings.
    return pd.to_numeric(series.replace({"N/A": np.nan, "NA": np.nan, "": np.nan}), errors="coerce")


def load_inputs(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p, dtype=str)
        df["_source_file"] = p.name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    # Basic type coercions
    out["season"] = pd.to_numeric(out.get("season"), errors="coerce").astype("Int64")
    out["placement"] = pd.to_numeric(out.get("placement"), errors="coerce").astype("Int64")
    out["celebrity_age_during_season"] = pd.to_numeric(out.get("celebrity_age_during_season"), errors="coerce")

    # Normalize base text columns
    for col in [
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "results",
    ]:
        if col in out.columns:
            out[col] = out[col].map(_strip_or_nan)

    return out


def find_week_judge_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if WEEK_JUDGE_RE.match(c):
            cols.append(c)
    return cols


def to_long(df_wide: pd.DataFrame, score_policy: ScorePolicy) -> pd.DataFrame:
    id_cols = [
        "celebrity_name",
        "ballroom_partner",
        "celebrity_industry",
        "celebrity_homestate",
        "celebrity_homecountry/region",
        "celebrity_age_during_season",
        "season",
        "results",
        "placement",
        "_source_file",
    ]
    id_cols = [c for c in id_cols if c in df_wide.columns]

    score_cols = find_week_judge_columns(df_wide)
    if not score_cols:
        raise ValueError("No week/judge score columns found (expected like week1_judge1_score)")

    melted = df_wide.melt(id_vars=id_cols, value_vars=score_cols, var_name="week_judge", value_name="judge_score_raw")
    melted["judge_score_raw"] = melted["judge_score_raw"].replace({"N/A": np.nan, "NA": np.nan, "": np.nan})
    melted["judge_score_raw"] = coerce_numeric(melted["judge_score_raw"])

    week = melted["week_judge"].str.extract(WEEK_JUDGE_RE)["week"].astype(int)
    judge = melted["week_judge"].str.extract(WEEK_JUDGE_RE)["judge"].astype(int)
    melted = melted.assign(week=week, judge=judge)

    # Flag 0 as "not present" (eliminated/withdrew/not in show)
    melted["is_zero_score"] = melted["judge_score_raw"].fillna(np.nan).eq(0)

    # Out-of-range flag (excluding 0, which is handled separately)
    in_range = melted["judge_score_raw"].between(score_policy.min_score, score_policy.max_score, inclusive="both")
    melted["is_out_of_range"] = (~melted["judge_score_raw"].isna()) & (~melted["is_zero_score"]) & (~in_range)

    # Cleaned score used for *core logic* totals/ranks
    if score_policy.strict_range:
        melted["judge_score_clean"] = melted["judge_score_raw"].where(~melted["is_out_of_range"], np.nan)
    else:
        melted["judge_score_clean"] = melted["judge_score_raw"]

    # Keep 0 as 0 in clean column (so we can detect absence at week level)
    melted.loc[melted["is_zero_score"], "judge_score_clean"] = 0.0

    # Pivot back to one row per (contestant, season, week)
    base_cols = [c for c in id_cols]
    index_cols = base_cols + ["week"]

    wide_week = (
        melted.pivot_table(index=index_cols, columns="judge", values="judge_score_clean", aggfunc="first")
        .rename(columns=lambda j: f"judge{int(j)}_score")
        .reset_index()
    )

    # Also keep raw judge scores for reference
    wide_raw = (
        melted.pivot_table(index=index_cols, columns="judge", values="judge_score_raw", aggfunc="first")
        .rename(columns=lambda j: f"judge{int(j)}_score_raw")
        .reset_index()
    )

    # Out-of-range flags per judge
    wide_oob = (
        melted.pivot_table(index=index_cols, columns="judge", values="is_out_of_range", aggfunc="max")
        .rename(columns=lambda j: f"judge{int(j)}_out_of_range")
        .reset_index()
    )

    out = wide_week.merge(wide_raw, on=index_cols, how="left").merge(wide_oob, on=index_cols, how="left")

    judge_cols = sorted([c for c in out.columns if re.fullmatch(r"judge\d+_score", c)], key=lambda x: int(re.findall(r"\d+", x)[0]))
    judge_raw_cols = sorted([c for c in out.columns if re.fullmatch(r"judge\d+_score_raw", c)], key=lambda x: int(re.findall(r"\d+", x)[0]))

    # Count judges present for this week (exclude NaN, exclude 0 from presence count)
    out["n_judges_reported"] = out[judge_cols].notna().sum(axis=1).astype(int)
    out["n_judges_present"] = out[judge_cols].apply(lambda r: int(((~r.isna()) & (r != 0)).sum()), axis=1)

    # Raw completeness (used to identify accidental missingness)
    out["n_judges_reported_raw"] = out[judge_raw_cols].notna().sum(axis=1).astype(int)
    out["has_partial_na_raw"] = out[judge_raw_cols].isna().any(axis=1) & (~out[judge_raw_cols].isna().all(axis=1))

    # Absent/eliminated rows: all reported judge scores are 0 (or missing)
    # We treat all-zero as absence; if all NaN, structural missing.
    out["is_all_na"] = out[judge_raw_cols].isna().all(axis=1)
    out["is_all_zero"] = out[judge_raw_cols].fillna(0).eq(0).all(axis=1) & (~out["is_all_na"])
    out["is_present"] = (~out["is_all_na"]) & (~out["is_all_zero"])
    
    # Parse result type for more structured handling
    out["result_type"] = out["results"].map(parse_result_type)
    out["is_withdrew_result"] = out["result_type"] == RESULT_WITHDREW
    
    # Detect potential "real 0" score conflicts: 
    # If a row has 0 scores but results indicate ongoing competition (not eliminated/withdrew),
    # this could be a data quality issue or (unlikely) a real 0 score.
    # We flag these for manual review in QA.
    out["zero_score_conflict"] = (
        out["is_all_zero"] 
        & out["result_type"].isin([RESULT_WINNER, RESULT_FINALIST, RESULT_ONGOING, RESULT_UNKNOWN])
    )

    # Week totals
    out["week_total_score"] = out[judge_cols].where(out[judge_cols].ne(0), np.nan).sum(axis=1, min_count=1)
    out["week_total_score_raw"] = out[judge_raw_cols].where(out[judge_raw_cols].ne(0), np.nan).sum(axis=1, min_count=1)

    # Standardized metrics
    out["week_avg_score"] = out["week_total_score"] / out["n_judges_present"].replace(0, np.nan)
    out["week_max_possible"] = out["n_judges_present"].replace(0, np.nan) * 10.0
    out["week_score_percentage"] = out["week_total_score"] / out["week_max_possible"]

    # Drop structural weeks that never ran: for a season-week, if everybody is_all_na.
    season_week_all_na = out.groupby(["season", "week"], dropna=False)["is_all_na"].transform("all")
    out = out.loc[~season_week_all_na].copy()

    # Detect unexpected missing scores (non-structural, non-eliminated rows)
    # For each (season, week), infer the expected number of judge scores recorded from the max non-NA count.
    expected_raw = (
        out.groupby(["season", "week"], dropna=False)["n_judges_reported_raw"].transform("max").astype(int)
    )
    out["expected_judges_reported_raw"] = expected_raw
    out["unexpected_missing_scores"] = (
        out["is_present"]
        & out["has_partial_na_raw"]
        & (out["n_judges_reported_raw"] < out["expected_judges_reported_raw"])
    )

    # Combine method flag
    out["combine_method"] = np.where(out["season"].between(3, 27), "percent", "rank")

    return out


def add_weekly_relative_features(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()

    # For within-week comparison, exclude not-present rows (0 / eliminated / withdrew)
    df["week_score_rank"] = np.nan
    df["week_score_share"] = np.nan

    performed = df["is_present"] & df["week_total_score"].notna()

    # Infer "in competition" even if a contestant has a 0-scored (absent) week.
    # Some contestants miss a week (injury/illness) and return later; those weeks should not
    # be treated as elimination/exit for season-week active counts.
    last_week_scored = (
        df.loc[df["is_present"]]
        .groupby(["celebrity_name", "season"], dropna=False)["week"]
        .max()
        .rename("last_week_scored")
        .reset_index()
    )
    df = df.merge(last_week_scored, on=["celebrity_name", "season"], how="left")
    season_final_week = df.groupby("season", dropna=False)["week"].transform("max")
    df["season_final_week"] = season_final_week

    # How many contestants actually performed in the finale week?
    # (Some seasons have a 2-couple finale; 3rd place is determined earlier.)
    final_week_n_performers = (
        df.loc[(df["week"] == df["season_final_week"]) & (df["is_present"])]
        .groupby("season", dropna=False)["celebrity_name"]
        .nunique()
        .rename("final_week_n_performers")
        .reset_index()
    )
    df = df.merge(final_week_n_performers, on="season", how="left")
    df["final_week_n_performers"] = df["final_week_n_performers"].fillna(0).astype(int)

    expected_to_perform_in_finale = pd.Series(False, index=df.index)
    if "placement" in df.columns:
        expected_to_perform_in_finale = df["placement"].notna() & (
            df["placement"].astype(int) <= df["final_week_n_performers"]
        )
    df["expected_to_perform_in_finale"] = expected_to_perform_in_finale

    expected_last_week = df["last_week_scored"].copy()
    expected_last_week = expected_last_week.where(~df["expected_to_perform_in_finale"], df["season_final_week"])
    df["expected_last_week"] = expected_last_week

    df["is_competing_week"] = df["expected_last_week"].notna() & (df["week"] <= df["expected_last_week"])

    is_all_zero = df["is_all_zero"].fillna(False) if "is_all_zero" in df.columns else False

    # Missing finale scores: expected-to-perform contestants should appear in the final week,
    # but the data records all-zeros (likely missing/encoding issue).
    df["is_missing_finale_scores"] = (
        is_all_zero & (df["week"] == df["season_final_week"]) & df["expected_to_perform_in_finale"]
    )

    # Gap week: absent/zero-scored but still in competition (missed week then returned later)
    # Exclude the final week to avoid treating missing-finale data as a "gap".
    df["is_gap_week"] = (
        df["is_competing_week"]
        & (~df["is_present"])
        & is_all_zero
        & (df["week"] < df["expected_last_week"])
    )

    # Refine zero-score conflict: only flag 0-score rows that are NOT explained by a gap week.
    if "zero_score_conflict" in df.columns and "result_type" in df.columns:
        df["zero_score_conflict"] = (
            is_all_zero
            & (~df.get("is_all_na", pd.Series(False)).fillna(False))
            & (~df["is_gap_week"])
            & (~df["is_missing_finale_scores"])
            & df["is_competing_week"]
            & df["result_type"].isin([RESULT_WINNER, RESULT_FINALIST, RESULT_ONGOING, RESULT_UNKNOWN])
        )

    # Rank within (season, week)
    df.loc[performed, "week_score_rank"] = (
        df.loc[performed]
        .groupby(["season", "week"], dropna=False)["week_total_score"]
        .rank(ascending=False, method="min")
    )

    # Controversy feature at weekly granularity: judge rank vs final placement
    if "placement" in df.columns:
        df.loc[performed, "week_score_rank_discrepancy"] = df.loc[performed, "week_score_rank"] - df.loc[performed, "placement"].astype(float)

    # Percent share of judge totals within (season, week)
    denom = df.loc[performed].groupby(["season", "week"], dropna=False)["week_total_score"].transform("sum")
    df.loc[performed, "week_score_share"] = df.loc[performed, "week_total_score"] / denom.replace(0, np.nan)

    # Special weeks inferred by active-count changes (proxy for double/no elimination)
    active_counts = (
        df.loc[df["is_competing_week"]]
        .groupby(["season", "week"], dropna=False)["celebrity_name"]
        .nunique()
        .rename("n_active")
        .reset_index()
    )
    active_counts["n_active_next"] = active_counts.groupby("season")["n_active"].shift(-1)
    active_counts["delta_next"] = active_counts["n_active"] - active_counts["n_active_next"]

    active_counts["is_no_elimination"] = active_counts["delta_next"].eq(0)
    active_counts["is_double_elimination"] = active_counts["delta_next"].ge(2)

    df = df.merge(active_counts[["season", "week", "is_no_elimination", "is_double_elimination", "n_active"]], on=["season", "week"], how="left")
    
    # Mark final week (决赛周) for each season: the last week where at least one contestant is present
    # This is useful because the finale's rules often differ from regular weeks.
    final_week_per_season = (
        df.loc[df["is_competing_week"]]
        .groupby("season", dropna=False)["week"]
        .max()
        .rename("final_week")
        .reset_index()
    )
    df = df.merge(final_week_per_season, on="season", how="left")
    df["is_final_week"] = (df["week"] == df["final_week"]).fillna(False)
    
    # Also detect no-elimination from results text patterns (e.g., "No elimination")
    # Some results may explicitly mention it
    # This creates a secondary check for official no-elimination weeks
    no_elim_pattern = r"no\s*elimination|no\s*one\s*eliminated|everyone\s*safe"
    df["is_no_elim_from_results"] = df["results"].astype(str).str.contains(no_elim_pattern, case=False, na=False)
    
    # Combine both detection methods
    df["is_no_elimination_any"] = df["is_no_elimination"].fillna(False) | df["is_no_elim_from_results"]
    
    # For withdrew contestants: mark their last active week and subsequent weeks
    # This helps downstream models exclude post-withdraw data from fan vote inference
    df["weeks_since_exit"] = np.nan
    for (name, season), grp in df.groupby(["celebrity_name", "season"], dropna=False):
        if grp["is_withdrew_result"].any():
            # Find last week they were actually present
            present_weeks = grp.loc[grp["is_present"], "week"]
            if len(present_weeks) > 0:
                last_present_week = present_weeks.max()
                mask = (df["celebrity_name"] == name) & (df["season"] == season)
                df.loc[mask, "weeks_since_exit"] = df.loc[mask, "week"] - last_present_week
    
    # Mark if this row should be excluded from fan vote inference
    # (withdrew + not present, or weeks after withdrawal)
    df["exclude_from_fan_vote_inference"] = (
        (df["is_withdrew_result"] & ~df["is_present"]) |
        (df["weeks_since_exit"].fillna(0) > 0)
    )

    return df


def add_text_and_demo_features(df_long: pd.DataFrame, age_overrides: Optional[pd.DataFrame] = None, industry_overrides: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df_long.copy()

    # Industry normalization (with optional overrides)
    df["celebrity_industry_clean"] = df["celebrity_industry"].map(normalize_industry)
    if industry_overrides is not None and not industry_overrides.empty:
        # expected columns: celebrity_name, season(optional), celebrity_industry_clean
        ov = industry_overrides.copy()
        if "season" in ov.columns:
            ov["season"] = pd.to_numeric(ov["season"], errors="coerce").astype("Int64")
            df = df.merge(ov[["celebrity_name", "season", "celebrity_industry_clean"]].dropna(subset=["celebrity_industry_clean"]), on=["celebrity_name", "season"], how="left", suffixes=("", "_ov"))
            df["celebrity_industry_clean"] = df["celebrity_industry_clean_ov"].combine_first(df["celebrity_industry_clean"])
            df = df.drop(columns=["celebrity_industry_clean_ov"])
        else:
            df = df.merge(ov[["celebrity_name", "celebrity_industry_clean"]].dropna(subset=["celebrity_industry_clean"]), on=["celebrity_name"], how="left", suffixes=("", "_ov"))
            df["celebrity_industry_clean"] = df["celebrity_industry_clean_ov"].combine_first(df["celebrity_industry_clean"])
            df = df.drop(columns=["celebrity_industry_clean_ov"])

    # Country/state normalization
    df["celebrity_homecountry_clean"] = df["celebrity_homecountry/region"].map(normalize_country)
    state_pairs = df["celebrity_homestate"].map(normalize_state)
    df["celebrity_homestate_clean"] = state_pairs.map(lambda t: t[0])
    df["celebrity_homestate_abbr"] = state_pairs.map(lambda t: t[1])

    df["is_foreign"] = df["celebrity_homecountry_clean"].notna() & (df["celebrity_homecountry_clean"].str.lower() != "united states")

    # Age outlier flag + optional overrides
    df["celebrity_age_raw"] = df["celebrity_age_during_season"]
    # Age sanity check: treat obviously wrong ages (e.g., 0 or 200) as missing, then optionally override.
    df["age_outlier"] = df["celebrity_age_during_season"].notna() & (
        (df["celebrity_age_during_season"] <= 0) | (df["celebrity_age_during_season"] >= 120)
    )
    df.loc[df["age_outlier"], "celebrity_age_during_season"] = np.nan

    if age_overrides is not None and not age_overrides.empty:
        ov = age_overrides.copy()
        # expected columns: celebrity_name, season(optional), celebrity_age_during_season
        if "season" in ov.columns:
            ov["season"] = pd.to_numeric(ov["season"], errors="coerce").astype("Int64")
            df = df.merge(ov[["celebrity_name", "season", "celebrity_age_during_season"]].dropna(subset=["celebrity_age_during_season"]), on=["celebrity_name", "season"], how="left", suffixes=("", "_ov"))
            df["celebrity_age_during_season"] = df["celebrity_age_during_season_ov"].combine_first(df["celebrity_age_during_season"])
            df = df.drop(columns=["celebrity_age_during_season_ov"])
        else:
            df = df.merge(ov[["celebrity_name", "celebrity_age_during_season"]].dropna(subset=["celebrity_age_during_season"]), on=["celebrity_name"], how="left", suffixes=("", "_ov"))
            df["celebrity_age_during_season"] = df["celebrity_age_during_season_ov"].combine_first(df["celebrity_age_during_season"])
            df = df.drop(columns=["celebrity_age_during_season_ov"])

    # Withdrawal tag from results (keep for backward compat, but prefer result_type)
    df["is_withdrawn"] = df["results"].astype(str).str.contains("Withdrew", case=False, na=False)

    # Infer potential withdrawal / data-issue rows: unexpected missing scores while still present.
    # We only *label* it; downstream can decide how to treat.
    if "unexpected_missing_scores" in df.columns:
        df["is_withdrawn_inferred"] = df["unexpected_missing_scores"] & (~df["is_withdrawn"])
    else:
        df["is_withdrawn_inferred"] = False
    
    # Extract elimination week from results text (e.g., "Eliminated Week 5")
    elim_week_match = df["results"].astype(str).str.extract(r"(?:Eliminated|Withdrew).*?Week\s*(\d+)", flags=re.IGNORECASE)
    df["exit_week_from_results"] = pd.to_numeric(elim_week_match[0], errors="coerce").astype("Int64")
    
    # Validate: if exit_week_from_results exists, check if it matches last_week they were present
    # This helps catch data inconsistencies
    if "week" in df.columns:
        present_last_week = df.loc[df["is_present"]].groupby(["celebrity_name", "season"], dropna=False)["week"].max()
        df = df.merge(
            present_last_week.rename("computed_last_week").reset_index(),
            on=["celebrity_name", "season"],
            how="left"
        )
        df["exit_week_mismatch"] = (
            df["exit_week_from_results"].notna() & 
            df["computed_last_week"].notna() &
            (df["exit_week_from_results"] != df["computed_last_week"])
        )
    else:
        df["exit_week_mismatch"] = False

    return df


def build_season_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    df = df_long.copy()
    present = df["is_present"] & df["week_total_score"].notna()

    # last week active (proxy survival)
    last_week = df.loc[present].groupby(["celebrity_name", "season"], dropna=False)["week"].max().rename("last_week_active")

    # season average score percentage
    season_mean_pct = (
        df.loc[present]
        .groupby(["celebrity_name", "season"], dropna=False)["week_score_percentage"]
        .mean()
        .rename("season_mean_score_percentage")
    )

    base = (
        df.groupby(["celebrity_name", "season"], dropna=False)
        .agg(
            ballroom_partner=("ballroom_partner", "first"),
            celebrity_industry=("celebrity_industry", "first"),
            celebrity_industry_clean=("celebrity_industry_clean", "first"),
            celebrity_homecountry_clean=("celebrity_homecountry_clean", "first"),
            celebrity_homestate_abbr=("celebrity_homestate_abbr", "first"),
            is_foreign=("is_foreign", "max"),
            celebrity_age_during_season=("celebrity_age_during_season", "first"),
            results=("results", "first"),
            placement=("placement", "first"),
            combine_method=("combine_method", "first"),
            is_withdrawn=("is_withdrawn", "max"),
        )
        .reset_index()
    )

    base = base.merge(last_week.reset_index(), on=["celebrity_name", "season"], how="left")
    base = base.merge(season_mean_pct.reset_index(), on=["celebrity_name", "season"], how="left")

    # Judge rank by season_mean_score_percentage (higher is better)
    base["season_judge_rank"] = (
        base.groupby("season", dropna=False)["season_mean_score_percentage"].rank(ascending=False, method="min")
    )

    # Controversy feature: judge rank vs final placement
    base["score_rank_discrepancy"] = base["season_judge_rank"] - base["placement"].astype(float)

    return base


def build_partner_summary(season_summary: pd.DataFrame) -> pd.DataFrame:
    """Build partner-level summary statistics.
    
    NOTE: These are GLOBAL stats across all seasons. For time-ordered analysis
    (avoiding temporal leakage), use build_partner_summary_prior() instead.
    """
    s = season_summary.dropna(subset=["ballroom_partner"]).copy()

    out = (
        s.groupby("ballroom_partner", dropna=False)
        .agg(
            partner_seasons=("season", "nunique"),
            partner_contestants=("celebrity_name", "nunique"),
            partner_wins=("placement", lambda x: int((x == 1).sum())),
            partner_avg_placement=("placement", "mean"),
            partner_avg_last_week=("last_week_active", "mean"),
            partner_avg_season_score_pct=("season_mean_score_percentage", "mean"),
        )
        .reset_index()
    )
    return out


def build_partner_summary_prior(season_summary: pd.DataFrame) -> pd.DataFrame:
    """Build partner-level summary using only PRIOR seasons (no temporal leakage).
    
    For each (partner, season) pair, compute stats using only data from seasons < current.
    This is the recommended approach for predictive modeling.
    """
    s = season_summary.dropna(subset=["ballroom_partner"]).copy()
    s = s.sort_values(["ballroom_partner", "season"])
    
    results = []
    for partner, grp in s.groupby("ballroom_partner", dropna=False):
        grp = grp.sort_values("season")
        for i, (idx, row) in enumerate(grp.iterrows()):
            current_season = row["season"]
            prior_data = grp[grp["season"] < current_season]
            
            if len(prior_data) == 0:
                # First season for this partner - no prior data
                results.append({
                    "ballroom_partner": partner,
                    "season": current_season,
                    "partner_seasons_prior": 0,
                    "partner_contestants_prior": 0,
                    "partner_wins_prior": 0,
                    "partner_avg_placement_prior": np.nan,
                    "partner_avg_last_week_prior": np.nan,
                    "partner_avg_season_score_pct_prior": np.nan,
                })
            else:
                results.append({
                    "ballroom_partner": partner,
                    "season": current_season,
                    "partner_seasons_prior": prior_data["season"].nunique(),
                    "partner_contestants_prior": prior_data["celebrity_name"].nunique(),
                    "partner_wins_prior": int((prior_data["placement"] == 1).sum()),
                    "partner_avg_placement_prior": prior_data["placement"].mean(),
                    "partner_avg_last_week_prior": prior_data["last_week_active"].mean(),
                    "partner_avg_season_score_pct_prior": prior_data["season_mean_score_percentage"].mean(),
                })
    
    return pd.DataFrame(results)


def build_quality_report(df_long: pd.DataFrame) -> pd.DataFrame:
    """Build a comprehensive quality report for data validation.
    
    Key checks:
    - Zero score conflicts: rows with 0 scores but results indicate ongoing competition
    - Exit week mismatches: computed last week vs. results text don't match
    - Out-of-range scores: judge scores outside expected range
    - Withdrew handling: proper tagging and exclusion
    """
    df = df_long.copy()
    judge_oob_cols = [c for c in df.columns if c.endswith("_out_of_range")]

    report_rows = []
    
    # Basic counts
    report_rows.append(("rows_long", len(df)))
    report_rows.append(("rows_present", int(df["is_present"].sum())))
    report_rows.append(("rows_all_zero_absent", int(df["is_all_zero"].sum())))
    report_rows.append(("rows_all_na_structural", int(df["is_all_na"].sum())))
    
    # Result type breakdown
    if "result_type" in df.columns:
        for rt in df["result_type"].dropna().unique():
            # Count unique contestants per result type
            count = df[df["result_type"] == rt].groupby(["celebrity_name", "season"]).ngroups
            report_rows.append((f"contestants_result_{rt}", count))

    # Out-of-range judge scores
    if judge_oob_cols:
        any_oob = df[judge_oob_cols].fillna(False).any(axis=1)
        report_rows.append(("rows_with_any_out_of_range_judge", int(any_oob.sum())))
        # Detail: how many are >10 (bonus) vs <1 (error)
        judge_raw_cols = [c for c in df.columns if re.fullmatch(r"judge\d+_score_raw", c)]
        if judge_raw_cols:
            bonus_scores = df[judge_raw_cols].apply(lambda x: (x > 10).sum()).sum()
            below_min = df[judge_raw_cols].apply(lambda x: ((x < 1) & (x != 0) & x.notna()).sum()).sum()
            report_rows.append(("judge_scores_above_10_bonus", int(bonus_scores)))
            report_rows.append(("judge_scores_below_1_error", int(below_min)))
    
    # CRITICAL: Zero score conflict check (0分但结果显示仍在比赛)
    if "zero_score_conflict" in df.columns:
        conflict_count = int(df["zero_score_conflict"].sum())
        report_rows.append(("zero_score_conflict_rows", conflict_count))
        if conflict_count > 0:
            # List affected contestants for manual review
            conflict_examples = df.loc[df["zero_score_conflict"], ["celebrity_name", "season", "week", "results"]].head(10)
            report_rows.append(("zero_score_conflict_examples", conflict_examples.to_dict("records")))

    # Gap weeks: 0-score (absent) weeks where the contestant is still in competition and returns later
    if "is_gap_week" in df.columns:
        gap_count = int(df["is_gap_week"].fillna(False).sum())
        report_rows.append(("gap_week_rows", gap_count))
        if gap_count > 0:
            gap_examples = df.loc[df["is_gap_week"], ["celebrity_name", "season", "week", "results"]].head(10)
            report_rows.append(("gap_week_examples", gap_examples.to_dict("records")))

    # Missing finale scores: finalists/winner have all-zero scores on the season final week
    if "is_missing_finale_scores" in df.columns:
        miss_finale_count = int(df["is_missing_finale_scores"].fillna(False).sum())
        report_rows.append(("missing_finale_scores_rows", miss_finale_count))
        if miss_finale_count > 0:
            miss_finale_examples = df.loc[df["is_missing_finale_scores"], ["celebrity_name", "season", "week", "results"]].head(10)
            report_rows.append(("missing_finale_scores_examples", miss_finale_examples.to_dict("records")))

    report_rows.append(("age_outlier_rows", int(df.get("age_outlier", pd.Series(False)).fillna(False).sum())))
    report_rows.append(("withdrawn_rows", int(df.get("is_withdrawn", pd.Series(False)).fillna(False).sum())))
    report_rows.append(("withdrawn_inferred_rows", int(df.get("is_withdrawn_inferred", pd.Series(False)).fillna(False).sum())))
    report_rows.append(("unexpected_missing_scores_rows", int(df.get("unexpected_missing_scores", pd.Series(False)).fillna(False).sum())))
    
    # Exit week mismatch check
    if "exit_week_mismatch" in df.columns:
        mismatch_count = int(df["exit_week_mismatch"].sum())
        report_rows.append(("exit_week_mismatch_rows", mismatch_count))
    
    # No-elimination weeks
    if "is_no_elimination_any" in df.columns:
        no_elim_weeks = df.loc[df["is_no_elimination_any"], ["season", "week"]].drop_duplicates()
        report_rows.append(("no_elimination_week_count", len(no_elim_weeks)))
    
    # Rows excluded from fan vote inference
    if "exclude_from_fan_vote_inference" in df.columns:
        report_rows.append(("rows_excluded_from_fan_vote_inference", int(df["exclude_from_fan_vote_inference"].sum())))
    
    # Seasons and contestants summary
    report_rows.append(("unique_seasons", df["season"].nunique()))
    report_rows.append(("unique_contestants", df.groupby(["celebrity_name", "season"]).ngroups))
    report_rows.append(("unique_partners", df["ballroom_partner"].nunique()))

    return pd.DataFrame(report_rows, columns=["metric", "value"])


def read_optional_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Input CSV paths")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument(
        "--allow-bonus",
        action="store_true",
        help="If set, keep out-of-range judge scores (>20/<1) in cleaned totals (still flagged).",
    )
    parser.add_argument("--age-overrides", default="age_overrides.csv", help="Optional CSV to override ages")
    parser.add_argument("--industry-overrides", default="industry_overrides.csv", help="Optional CSV to override industries")

    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    score_policy = ScorePolicy(strict_range=(not args.allow_bonus))

    df_wide = load_inputs(input_paths)

    age_overrides = read_optional_csv(Path(args.age_overrides))
    industry_overrides = read_optional_csv(Path(args.industry_overrides))

    df_long = to_long(df_wide, score_policy=score_policy)
    df_long = add_weekly_relative_features(df_long)
    df_long = add_text_and_demo_features(df_long, age_overrides=age_overrides, industry_overrides=industry_overrides)

    season_summary = build_season_summary(df_long)
    partner_summary = build_partner_summary(season_summary)
    partner_summary_prior = build_partner_summary_prior(season_summary)

    # Mentor boost: merge partner stats back (global)
    df_long = df_long.merge(partner_summary, on="ballroom_partner", how="left")
    
    # Also merge time-ordered partner stats (no temporal leakage)
    df_long = df_long.merge(partner_summary_prior, on=["ballroom_partner", "season"], how="left")

    qa = build_quality_report(df_long)

    df_long.to_csv(outdir / "dwts_long_clean.csv", index=False, encoding="utf-8-sig")
    season_summary.to_csv(outdir / "dwts_season_summary.csv", index=False, encoding="utf-8-sig")
    partner_summary.to_csv(outdir / "dwts_partner_summary.csv", index=False, encoding="utf-8-sig")
    partner_summary_prior.to_csv(outdir / "dwts_partner_summary_prior.csv", index=False, encoding="utf-8-sig")
    qa.to_csv(outdir / "dwts_quality_report.csv", index=False, encoding="utf-8-sig")
    
    # Also output a detailed list of rows excluded from fan vote inference
    excluded = df_long[df_long.get("exclude_from_fan_vote_inference", pd.Series(False)).fillna(False)]
    if len(excluded) > 0:
        excluded[["celebrity_name", "season", "week", "results", "result_type", "is_present", "weeks_since_exit"]].to_csv(
            outdir / "dwts_excluded_from_inference.csv", index=False, encoding="utf-8-sig"
        )
    
    # Output zero-score conflicts for manual review
    if "zero_score_conflict" in df_long.columns:
        conflicts = df_long[df_long["zero_score_conflict"]]
        if len(conflicts) > 0:
            conflicts.to_csv(outdir / "dwts_zero_score_conflicts.csv", index=False, encoding="utf-8-sig")
            print(f"WARNING: {len(conflicts)} rows have zero scores but results indicate ongoing competition.")
            print("         Review dwts_zero_score_conflicts.csv for potential data issues.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
