# -*- coding: utf-8 -*-
"""Q3: 系数对 K 的灵敏度分析（K=2/3/4 等）

功能：
- 对指定 K 列表分别拟合 Q3（评委、粉丝点估计、粉丝 NPZ 传播、Δβ、Pro 相关）。
- 若某个 K 的输出文件已存在，且 cfg.k_sensitivity_skip_existing=True，则直接复用，不重复拟合。
- 额外输出跨 K 的汇总表，便于写作时说明“结论对 K 稳健”。

用法（在项目根目录运行）：
- `python Q3_regression/q3_k_sensitivity.py`

输出：
- 保留各 K 的常规输出：`judge_fixed_summary_k{K}.csv`、`fan_fixed_summary_k{K}.csv`、`delta_fixed_summary_k{K}.csv`、...
- 新增跨 K 汇总：
  - `k_sensitivity_fixed_effects.csv`
  - `k_sensitivity_delta_fixed.csv`
  - `k_sensitivity_fan_sample_fixed_quantiles.csv`
  - `k_sensitivity_fan_sample_delta_quantiles.csv`
  - `k_sensitivity_pro_corr.csv`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from q3_config import Q3Config
from q3_features import add_core_features, add_fan_centered_log
from q3_loader import build_fan_sample_sets, build_npz_index, merge_dwts_fan, read_dwts_long, read_fan_long
from q3_model import (
    fit_bambi_model,
    fixed_effect_mean_dict,
    pro_effect_correlation,
    summarize_delta,
    summarize_fixed_effects,
    summarize_random_effects,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _read_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _required_paths_for_k(cfg: Q3Config, k: int) -> List[Path]:
    paths = [
        cfg.output_dir / f"judge_fixed_summary_k{k}.csv",
        cfg.output_dir / f"judge_pro_effects_k{k}.csv",
        cfg.output_dir / f"judge_season_effects_k{k}.csv",
        cfg.output_dir / f"panel_k{k}.csv",
    ]

    if cfg.use_fan_mean:
        paths += [
            cfg.output_dir / f"fan_fixed_summary_k{k}.csv",
            cfg.output_dir / f"fan_pro_effects_k{k}.csv",
            cfg.output_dir / f"fan_season_effects_k{k}.csv",
            cfg.output_dir / f"delta_fixed_summary_k{k}.csv",
            cfg.output_dir / f"pro_effect_correlation_k{k}.json",
        ]

    if cfg.use_fan_samples:
        paths += [
            cfg.output_dir / f"fan_sample_fixed_quantiles_k{k}.csv",
            cfg.output_dir / f"fan_sample_delta_quantiles_k{k}.csv",
        ]

    return paths


def _has_all_outputs(cfg: Q3Config, k: int) -> bool:
    return all(p.exists() for p in _required_paths_for_k(cfg, k))


def _run_single_k(
    *,
    cfg: Q3Config,
    base_df: pd.DataFrame,
    k: int,
    npz_index: Optional[dict],
) -> None:
    """对单个 K 运行 Q3，并落盘输出。"""

    df = base_df[(base_df["is_competing_week"] == True) & (base_df["week"] <= int(k))].copy()

    df = add_core_features(df, eps=cfg.eps)
    df["industry"] = df["industry"].astype("category")

    _save_csv(df, cfg.output_dir / f"panel_k{k}.csv")

    # 评委侧
    _, idata_j = fit_bambi_model(
        df=df.dropna(subset=["y_judge", "age_c", "log_week", "industry", "ballroom_partner", "season"]),
        outcome="y_judge",
        draws=cfg.draws,
        tune=cfg.tune,
        chains=cfg.chains,
        target_accept=cfg.target_accept,
        random_seed=cfg.random_seed + 10_000 + int(k),
        progressbar=cfg.pymc_progressbar,
    )

    judge_fixed = summarize_fixed_effects(idata_j)
    _save_csv(judge_fixed, cfg.output_dir / f"judge_fixed_summary_k{k}.csv")

    pro_judge = summarize_random_effects(idata_j, "ballroom_partner")
    season_judge = summarize_random_effects(idata_j, "season")
    _save_csv(pro_judge, cfg.output_dir / f"judge_pro_effects_k{k}.csv")
    _save_csv(season_judge, cfg.output_dir / f"judge_season_effects_k{k}.csv")

    # 粉丝点估计
    idata_f = None
    pro_fan = None
    if cfg.use_fan_mean:
        df_fan = df.copy()
        df_fan["y_fan"] = add_fan_centered_log(df_fan, "fan_vote_mean", eps=cfg.eps)
        df_fan = df_fan.dropna(subset=["y_fan"])

        _, idata_f = fit_bambi_model(
            df=df_fan,
            outcome="y_fan",
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            random_seed=cfg.random_seed + 20_000 + int(k),
            progressbar=cfg.pymc_progressbar,
        )

        fan_fixed = summarize_fixed_effects(idata_f)
        _save_csv(fan_fixed, cfg.output_dir / f"fan_fixed_summary_k{k}.csv")

        pro_fan = summarize_random_effects(idata_f, "ballroom_partner")
        season_fan = summarize_random_effects(idata_f, "season")
        _save_csv(pro_fan, cfg.output_dir / f"fan_pro_effects_k{k}.csv")
        _save_csv(season_fan, cfg.output_dir / f"fan_season_effects_k{k}.csv")

        delta = summarize_delta(idata_f, idata_j)
        _save_csv(delta, cfg.output_dir / f"delta_fixed_summary_k{k}.csv")

        corr, n_pairs = pro_effect_correlation(pro_fan, pro_judge)
        with open(cfg.output_dir / f"pro_effect_correlation_k{k}.json", "w", encoding="utf-8") as f:
            json.dump({"corr": corr, "n_pairs": n_pairs}, f, ensure_ascii=False, indent=2)

    # 粉丝 NPZ 传播
    if cfg.use_fan_samples:
        if npz_index is None:
            raise RuntimeError("use_fan_samples=True 但未提供 npz_index")

        sample_series = build_fan_sample_sets(
            df,
            npz_index=npz_index,
            n_sets=cfg.fan_sample_sets,
            seed=cfg.sample_seed + int(k),
        )

        judge_means = fixed_effect_mean_dict(idata_j)
        fan_means: List[Dict[str, float]] = []
        delta_means: List[Dict[str, float]] = []

        for i, s in enumerate(sample_series, start=1):
            df_s = df.copy()
            df_s["fan_vote_sample"] = s.values
            df_s["y_fan"] = add_fan_centered_log(df_s, "fan_vote_sample", eps=cfg.eps)
            df_s = df_s.dropna(subset=["y_fan"])

            _, idata_fs = fit_bambi_model(
                df=df_s,
                outcome="y_fan",
                draws=cfg.draws,
                tune=cfg.tune,
                chains=cfg.chains,
                target_accept=cfg.target_accept,
                random_seed=cfg.random_seed + 30_000 + int(k) * 100 + i,
                progressbar=cfg.pymc_progressbar,
            )

            fan_mean = fixed_effect_mean_dict(idata_fs)
            fan_means.append(fan_mean)
            delta_means.append({p: fan_mean[p] - judge_means.get(p, np.nan) for p in fan_mean.keys()})

        def _quantile_table(samples: List[Dict[str, float]], q=(0.025, 0.5, 0.975)) -> pd.DataFrame:
            if not samples:
                return pd.DataFrame()
            keys = sorted({k for s in samples for k in s.keys()})
            arr = np.full((len(samples), len(keys)), np.nan, dtype=float)
            for r_i, row in enumerate(samples):
                for c_i, key in enumerate(keys):
                    if key in row:
                        arr[r_i, c_i] = float(row[key])
            return pd.DataFrame({
                "param": keys,
                "q2.5": np.nanquantile(arr, q[0], axis=0),
                "q50": np.nanquantile(arr, q[1], axis=0),
                "q97.5": np.nanquantile(arr, q[2], axis=0),
                "p_gt_0": np.nanmean(arr > 0, axis=0),
            })

        fan_q = _quantile_table(fan_means)
        delta_q = _quantile_table(delta_means)

        _save_csv(fan_q, cfg.output_dir / f"fan_sample_fixed_quantiles_k{k}.csv")
        _save_csv(delta_q, cfg.output_dir / f"fan_sample_delta_quantiles_k{k}.csv")


def _stack_fixed_summary(df: Optional[pd.DataFrame], *, side: str, k: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    keep = [c for c in ["param", "mean", "sd", "hdi_2.5%", "hdi_97.5%", "r_hat", "ess_bulk", "ess_tail"] if c in df.columns]
    out = df[keep].copy()
    out.insert(0, "k", int(k))
    out.insert(1, "side", side)
    return out


def _stack_delta_summary(df: Optional[pd.DataFrame], *, k: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    keep = [c for c in ["param", "base_var", "mean", "sd", "hdi_2.5%", "hdi_97.5%", "r_hat", "ess_bulk", "ess_tail"] if c in df.columns]
    out = df[keep].copy()
    out.insert(0, "k", int(k))
    return out


def main() -> None:
    cfg = Q3Config()
    _ensure_dir(cfg.output_dir)

    print("[Q3-K] 读取与合并数据")
    dwts = read_dwts_long(cfg.dwts_long_path)
    fan = read_fan_long(cfg.fan_long_path)
    base_df = merge_dwts_fan(dwts, fan)

    npz_index = None
    if cfg.use_fan_samples:
        print("[Q3-K] 构建 NPZ 索引")
        npz_index = build_npz_index(cfg.npz_dir)

    k_values = [int(k) for k in cfg.k_sensitivity_values]

    print(f"[Q3-K] 计划运行/复用 K: {k_values}")

    for k in k_values:
        if cfg.k_sensitivity_skip_existing and _has_all_outputs(cfg, k):
            print(f"[Q3-K] K={k} 已存在输出 -> 跳过拟合（复用文件）")
            continue

        print(f"[Q3-K] 运行 K={k}（可能耗时较长）")
        _run_single_k(cfg=cfg, base_df=base_df, k=k, npz_index=npz_index)

    # ---- 汇总：跨 K 拼表 ----
    print("[Q3-K] 生成跨 K 汇总表")

    fixed_rows: List[pd.DataFrame] = []
    delta_rows: List[pd.DataFrame] = []
    fan_sample_fixed_rows: List[pd.DataFrame] = []
    fan_sample_delta_rows: List[pd.DataFrame] = []
    corr_rows: List[dict] = []

    for k in k_values:
        judge_fixed = _read_csv_if_exists(cfg.output_dir / f"judge_fixed_summary_k{k}.csv")
        fan_fixed = _read_csv_if_exists(cfg.output_dir / f"fan_fixed_summary_k{k}.csv")
        delta_fixed = _read_csv_if_exists(cfg.output_dir / f"delta_fixed_summary_k{k}.csv")

        fixed_rows.append(_stack_fixed_summary(judge_fixed, side="judge", k=k))
        if cfg.use_fan_mean:
            fixed_rows.append(_stack_fixed_summary(fan_fixed, side="fan", k=k))
            delta_rows.append(_stack_delta_summary(delta_fixed, k=k))

        if cfg.use_fan_samples:
            fan_q = _read_csv_if_exists(cfg.output_dir / f"fan_sample_fixed_quantiles_k{k}.csv")
            delta_q = _read_csv_if_exists(cfg.output_dir / f"fan_sample_delta_quantiles_k{k}.csv")
            if fan_q is not None and not fan_q.empty:
                fan_q = fan_q.copy()
                fan_q.insert(0, "k", int(k))
                fan_sample_fixed_rows.append(fan_q)
            if delta_q is not None and not delta_q.empty:
                delta_q = delta_q.copy()
                delta_q.insert(0, "k", int(k))
                fan_sample_delta_rows.append(delta_q)

        corr = _read_json_if_exists(cfg.output_dir / f"pro_effect_correlation_k{k}.json")
        if corr is not None:
            corr_rows.append({"k": int(k), **corr})

    fixed_all = pd.concat([r for r in fixed_rows if r is not None and not r.empty], ignore_index=True) if fixed_rows else pd.DataFrame()
    delta_all = pd.concat([r for r in delta_rows if r is not None and not r.empty], ignore_index=True) if delta_rows else pd.DataFrame()
    fan_q_all = pd.concat([r for r in fan_sample_fixed_rows if r is not None and not r.empty], ignore_index=True) if fan_sample_fixed_rows else pd.DataFrame()
    delta_q_all = pd.concat([r for r in fan_sample_delta_rows if r is not None and not r.empty], ignore_index=True) if fan_sample_delta_rows else pd.DataFrame()
    corr_all = pd.DataFrame(corr_rows)

    if not fixed_all.empty:
        _save_csv(fixed_all, cfg.output_dir / "k_sensitivity_fixed_effects.csv")
    if not delta_all.empty:
        _save_csv(delta_all, cfg.output_dir / "k_sensitivity_delta_fixed.csv")
    if not fan_q_all.empty:
        _save_csv(fan_q_all, cfg.output_dir / "k_sensitivity_fan_sample_fixed_quantiles.csv")
    if not delta_q_all.empty:
        _save_csv(delta_q_all, cfg.output_dir / "k_sensitivity_fan_sample_delta_quantiles.csv")
    if not corr_all.empty:
        _save_csv(corr_all, cfg.output_dir / "k_sensitivity_pro_corr.csv")

    print(f"[Q3-K] 完成：输出目录 -> {cfg.output_dir}")


if __name__ == "__main__":
    main()
