# -*- coding: utf-8 -*-
"""
Q3 特征工程与 K 选择
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class KSelectionReport:
    chosen_k: int
    candidates: List[int]
    cover_rates: dict
    survival_rates: dict


def choose_k(
    df: pd.DataFrame,
    candidates: Iterable[int],
    min_cover: float,
    min_survival: float,
) -> KSelectionReport:
    df = df.copy()
    seasons = sorted(df["season"].dropna().unique().tolist())
    candidates = sorted(set(int(k) for k in candidates))

    k_complete = {}
    for s in seasons:
        s_df = df[df["season"] == s]
        max_week = int(s_df["week"].max())
        count = 0
        for wk in range(1, max_week + 1):
            wdf = s_df[s_df["week"] == wk]
            if wdf.empty:
                break
            if (wdf["is_competing_week"] == True).any():
                count += 1
            else:
                break
        k_complete[s] = count

    cover_rates = {}
    survival_rates = {}

    for k in candidates:
        cover = np.mean([k <= k_complete[s] for s in seasons]) if seasons else 0.0
        cover_rates[k] = float(cover)

        ratios = []
        for s in seasons:
            s_df = df[df["season"] == s]
            n1 = int((s_df[(s_df["week"] == 1) & (s_df["is_competing_week"] == True)]).shape[0])
            nk = int((s_df[(s_df["week"] == k) & (s_df["is_competing_week"] == True)]).shape[0])
            if n1 > 0:
                ratios.append(nk / n1)
        survival = float(np.mean(ratios)) if ratios else 0.0
        survival_rates[k] = survival

    eligible = [
        k for k in candidates
        if cover_rates.get(k, 0.0) >= min_cover and survival_rates.get(k, 0.0) >= min_survival
    ]
    chosen_k = min(eligible) if eligible else min(candidates)

    return KSelectionReport(
        chosen_k=chosen_k,
        candidates=list(candidates),
        cover_rates=cover_rates,
        survival_rates=survival_rates,
    )


def add_core_features(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    df = df.copy()

    # industry
    if "celebrity_industry_clean" in df.columns:
        df["industry"] = df["celebrity_industry_clean"]
    else:
        df["industry"] = df["celebrity_industry"]

    # age
    df["age"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce")
    df["age_c"] = df["age"] - df["age"].mean()

    # log(week)
    df["log_week"] = np.log(df["week"].astype(float))

    # centered-log for judge share
    df["y_judge"] = _centered_log_share(
        df,
        share_col="week_score_share",
        eps=eps,
    )

    return df


def add_fan_centered_log(df: pd.DataFrame, fan_col: str, eps: float) -> pd.Series:
    return _centered_log_share(df, share_col=fan_col, eps=eps)


def _centered_log_share(df: pd.DataFrame, share_col: str, eps: float) -> pd.Series:
    values = pd.to_numeric(df[share_col], errors="coerce").astype(float)
    safe = np.log(values + eps)
    grouped = df.groupby(["season", "week"], sort=False)
    centered = safe - grouped[share_col].transform(lambda x: np.log(pd.to_numeric(x, errors="coerce").astype(float) + eps).mean())
    return centered
