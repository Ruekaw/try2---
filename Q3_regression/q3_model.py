# -*- coding: utf-8 -*-
"""
Q3 模型拟合与汇总
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import bambi as bmb
import arviz as az


def fit_bambi_model(
    df: pd.DataFrame,
    outcome: str,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    random_seed: int,
    progressbar: bool = True,
):
    formula = f"{outcome} ~ age_c + log_week + industry + (1|ballroom_partner) + (1|season)"
    model = bmb.Model(formula, df)
    idata = model.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=random_seed,
        cores=chains,
        progressbar=progressbar,
    )
    return model, idata


def fixed_effect_names(idata) -> List[str]:
    names = []
    for v in idata.posterior.data_vars:
        if "|" in v:
            continue
        if v.startswith("sigma") or v.startswith("sd_") or v.startswith("chol"):
            continue
        names.append(v)
    return names


def summarize_fixed_effects(idata, hdi_prob: float = 0.95) -> pd.DataFrame:
    names = fixed_effect_names(idata)
    if not names:
        return pd.DataFrame()
    summary = az.summary(idata, var_names=names, hdi_prob=hdi_prob)
    return summary.reset_index().rename(columns={"index": "param"})


def _find_group_var(idata, group_name: str) -> Optional[str]:
    """Find the posterior variable that contains per-level random effects.

    Bambi may store group-level effects as:
    - `1|group` with dims (chain, draw, group__factor_dim)
    - `1|group_sigma` with dims (chain, draw)

    We must prefer the former; otherwise downstream summaries become empty.
    """
    if not hasattr(idata, "posterior"):
        return None

    candidates: List[str] = []
    for v in idata.posterior.data_vars:
        if "|" not in v or group_name not in v:
            continue
        arr = idata.posterior[v]
        other_dims = [d for d in arr.dims if d not in ("chain", "draw")]
        if other_dims:
            candidates.append(v)

    if not candidates:
        return None

    exact = f"1|{group_name}"
    if exact in candidates:
        return exact

    # Prefer shorter names (e.g., avoid offsets/aux vars if any)
    candidates.sort(key=lambda s: (len(s), s))
    return candidates[0]


def summarize_random_effects(idata, group_name: str, hdi_prob: float = 0.95) -> pd.DataFrame:
    var = _find_group_var(idata, group_name)
    if var is None:
        return pd.DataFrame()

    arr = idata.posterior[var]
    dims = [d for d in arr.dims if d not in ("chain", "draw")]
    if not dims:
        return pd.DataFrame()
    level_dim = dims[0]
    levels = arr.coords[level_dim].values

    mean = arr.mean(dim=("chain", "draw")).values
    hdi = az.hdi(arr, hdi_prob=hdi_prob).values

    out = pd.DataFrame({
        group_name: levels,
        "mean": mean,
        "hdi_lower": hdi[..., 0],
        "hdi_upper": hdi[..., 1],
    })
    return out


def summarize_delta(idata_fan, idata_judge, hdi_prob: float = 0.95) -> pd.DataFrame:
    fan_names = set(fixed_effect_names(idata_fan))
    judge_names = set(fixed_effect_names(idata_judge))
    common = sorted(fan_names.intersection(judge_names))
    if not common:
        return pd.DataFrame()

    rows = []
    for name in common:
        fan = idata_fan.posterior[name]
        judge = idata_judge.posterior[name]
        delta = fan - judge
        summary = az.summary(delta, hdi_prob=hdi_prob)
        summary = summary.reset_index().rename(columns={"index": "param"})
        # For scalar params (e.g., Intercept), normalize the name.
        # For vector params (e.g., industry levels), keep component labels like
        # `industry[Athlete]` so downstream comparisons remain interpretable.
        if len(summary) == 1:
            summary["param"] = name
        rows.append(summary)

    return pd.concat(rows, ignore_index=True)


def pro_effect_correlation(
    pro_fan: pd.DataFrame,
    pro_judge: pd.DataFrame,
    group_col: str = "ballroom_partner",
) -> Tuple[float, int]:
    if pro_fan.empty or pro_judge.empty:
        return float("nan"), 0

    merged = pro_fan[[group_col, "mean"]].merge(
        pro_judge[[group_col, "mean"]],
        on=group_col,
        how="inner",
        suffixes=("_fan", "_judge"),
    )
    if merged.empty:
        return float("nan"), 0

    corr = float(np.corrcoef(merged["mean_fan"], merged["mean_judge"])[0, 1])
    return corr, int(len(merged))
