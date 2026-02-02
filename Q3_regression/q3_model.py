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


def fixed_effect_mean_dict(idata, hdi_prob: float = 0.95) -> Dict[str, float]:
    """提取固定效应“逐参数(含分类水平展开)”的后验均值。

    例如 Bambi 会将 `industry` 展开为 `industry[Athlete]` 等参数名。
    这里返回的 key 直接对应 az.summary 的 index。
    """
    base_vars = fixed_effect_names(idata)
    if not base_vars:
        return {}

    summary = az.summary(idata, var_names=base_vars, hdi_prob=hdi_prob)
    return {str(idx): float(row["mean"]) for idx, row in summary.iterrows()}


def summarize_fixed_effects(idata, hdi_prob: float = 0.95) -> pd.DataFrame:
    names = fixed_effect_names(idata)
    if not names:
        return pd.DataFrame()
    summary = az.summary(idata, var_names=names, hdi_prob=hdi_prob)
    return summary.reset_index().rename(columns={"index": "param"})


def _find_group_var(idata, group_name: str) -> Tuple[Optional[str], Optional[str]]:
    """在 idata.posterior 中查找与分组变量对应的随机效应变量及其维度名。

    优先依据坐标名匹配（更稳健）：
    - 若某个变量在 arr.coords 中包含 group_name，则认为它是该分组的随机效应；
      返回 (var_name, group_name)。

    回退策略（兼容旧命名）：
    - 若未找到坐标名匹配，则在变量名中查找既包含 group_name 又包含 '|' 的变量，
      此时从其 dims 中挑选一个非链/抽样维度作为 level_dim。
    """

    # 1) 首选：看 coords 里是否有以 group_name 命名的维度
    for v in idata.posterior.data_vars:
        arr = idata.posterior[v]
        if group_name in arr.coords:
            return v, group_name

    # 2) 回退：根据变量名启发式匹配，然后从 dims 中挑一个非链/抽样维度
    for v in idata.posterior.data_vars:
        if group_name in v and "|" in v:
            arr = idata.posterior[v]
            dims = [d for d in arr.dims if d not in ("chain", "draw")]
            if dims:
                return v, dims[0]

    return None, None


def summarize_random_effects(idata, group_name: str, hdi_prob: float = 0.95) -> pd.DataFrame:
    var, level_dim = _find_group_var(idata, group_name)
    if var is None or level_dim is None:
        return pd.DataFrame()

    arr = idata.posterior[var]
    if level_dim not in arr.dims:
        return pd.DataFrame()
    levels = arr.coords[level_dim].values

    # 后验均值：对链和抽样两个维度求平均
    mean = np.asarray(arr.mean(dim=("chain", "draw")))

    # HDI：为避免不同 ArviZ 版本返回结构不一致，这里直接在
    # (chain, draw) 两个维度上用分位数近似 95% 区间
    stacked = np.asarray(arr.stack(_sample=("chain", "draw")))  # shape: (level_dim, n_samples) 或相近
    # 如果 stack 维顺序是 (_sample, level_dim)，则转置一下
    if stacked.shape[0] != len(levels) and stacked.shape[1] == len(levels):
        stacked = stacked.T

    alpha = 1.0 - hdi_prob
    lower_q = alpha / 2.0
    upper_q = 1.0 - alpha / 2.0
    hdi_lower = np.quantile(stacked, lower_q, axis=1)
    hdi_upper = np.quantile(stacked, upper_q, axis=1)

    out = pd.DataFrame({
        group_name: levels,
        "mean": mean,
        "hdi_lower": hdi_lower,
        "hdi_upper": hdi_upper,
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
        summary["base_var"] = name
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
