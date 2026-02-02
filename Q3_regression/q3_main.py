# -*- coding: utf-8 -*-
"""
Q3 回归主脚本：前K周切片 + 分层回归 + 评委/粉丝对比
"""
from __future__ import annotations

import warnings

# 静音：threadpoolctl 的 OpenMP 混用 RuntimeWarning（不改变行为，仅隐藏提示）。
# 该提示主要针对 Linux 的潜在死锁/崩溃风险；Windows 下通常仅是噪声。
warnings.filterwarnings(
    "ignore",
    message=r"Found Intel OpenMP[\s\S]*LLVM OpenMP[\s\S]*loaded at[\s\S]*the same time",
    category=RuntimeWarning,
    module=r"threadpoolctl",
)

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

sys.path.append(str(Path(__file__).resolve().parent))

from q3_config import Q3Config
from q3_loader import read_dwts_long, read_fan_long, merge_dwts_fan, build_npz_index, build_fan_sample_sets
from q3_features import choose_k, add_core_features, add_fan_centered_log
from q3_model import (
    fit_bambi_model,
    summarize_fixed_effects,
    summarize_random_effects,
    summarize_delta,
    pro_effect_correlation,
    fixed_effect_names,
    fixed_effect_mean_dict,
)


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _save_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def _quantile_table(samples: List[Dict[str, float]], q=(0.025, 0.5, 0.975)) -> pd.DataFrame:
    """对“多次拟合得到的后验均值”做跨样本集分位数汇总。

    允许不同 sample set 得到的参数集合不完全一致（例如某些行业水平缺失），此时用 NaN 补齐并用 nan* 统计。
    """
    if not samples:
        return pd.DataFrame()

    keys = sorted({k for s in samples for k in s.keys()})
    arr = np.full((len(samples), len(keys)), np.nan, dtype=float)
    for i, s in enumerate(samples):
        for j, k in enumerate(keys):
            if k in s:
                arr[i, j] = float(s[k])

    out = pd.DataFrame({
        "param": keys,
        "q2.5": np.nanquantile(arr, q[0], axis=0),
        "q50": np.nanquantile(arr, q[1], axis=0),
        "q97.5": np.nanquantile(arr, q[2], axis=0),
        "p_gt_0": np.nanmean(arr > 0, axis=0),
    })
    return out


def main():
    parser = argparse.ArgumentParser(description="Q3 回归：前K周切片 + 分层回归 + 评委/粉丝对比")
    parser.add_argument(
        "--no-fan-samples",
        action="store_true",
        help="关闭粉丝后验样本集（NPZ）不确定性传播（跳过 use_fan_samples 分支）",
    )
    parser.add_argument(
        "--no-fan-mean",
        action="store_true",
        help="关闭粉丝点估计拟合（跳过 use_fan_mean 分支）",
    )
    args = parser.parse_args()

    cfg = Q3Config()
    if args.no_fan_samples:
        cfg.use_fan_samples = False
    if args.no_fan_mean:
        cfg.use_fan_mean = False
    _ensure_dir(cfg.output_dir)

    def _wrap_progress(iterable, *, desc: str, total: int | None = None):
        if tqdm is None:
            return iterable
        return tqdm(iterable, desc=desc, total=total)

    fit_total = 1
    if cfg.use_fan_mean:
        fit_total += 1
    if cfg.use_fan_samples:
        fit_total += int(cfg.fan_sample_sets)
    fit_idx = 0

    # 1) 读取与合并
    print("[Q3] 阶段 1/5：读取与合并数据")
    dwts = read_dwts_long(cfg.dwts_long_path)
    fan = read_fan_long(cfg.fan_long_path)
    df = merge_dwts_fan(dwts, fan)

    # 2) 选择 K
    print("[Q3] 阶段 2/5：选择 K")
    k_report = choose_k(
        df,
        candidates=cfg.k_candidates,
        min_cover=cfg.min_k_cover,
        min_survival=cfg.min_survival_ratio,
    )
    k = k_report.chosen_k
    with open(cfg.output_dir / "k_selection.json", "w", encoding="utf-8") as f:
        json.dump(k_report.__dict__, f, ensure_ascii=False, indent=2)

    # 3) 前 K 周切片
    print(f"[Q3] 阶段 3/5：前 K 周切片 (K={k})")
    df = df[(df["is_competing_week"] == True) & (df["week"] <= k)].copy()

    # 4) 特征与 y_judge
    print("[Q3] 阶段 4/5：特征工程 & 评委侧拟合")
    df = add_core_features(df, eps=cfg.eps)
    df["industry"] = df["industry"].astype("category")

    _save_csv(df, cfg.output_dir / f"panel_k{k}.csv")

    # 5) 评委侧模型
    fit_idx += 1
    print(f"[Q3] 拟合进度 {fit_idx}/{fit_total}：评委模型")
    model_j, idata_j = fit_bambi_model(
        df=df.dropna(subset=["y_judge", "age_c", "log_week", "industry", "ballroom_partner", "season"]),
        outcome="y_judge",
        draws=cfg.draws,
        tune=cfg.tune,
        chains=cfg.chains,
        target_accept=cfg.target_accept,
        random_seed=cfg.random_seed,
        progressbar=cfg.pymc_progressbar,
    )

    judge_fixed = summarize_fixed_effects(idata_j)
    _save_csv(judge_fixed, cfg.output_dir / f"judge_fixed_summary_k{k}.csv")

    pro_judge = summarize_random_effects(idata_j, "ballroom_partner")
    season_judge = summarize_random_effects(idata_j, "season")
    _save_csv(pro_judge, cfg.output_dir / f"judge_pro_effects_k{k}.csv")
    _save_csv(season_judge, cfg.output_dir / f"judge_season_effects_k{k}.csv")

    # 6) 粉丝侧（点估计）
    if cfg.use_fan_mean:
        print("[Q3] 阶段 5/5：粉丝侧拟合（点估计/样本集）")
        df_fan = df.copy()
        df_fan["y_fan"] = add_fan_centered_log(df_fan, "fan_vote_mean", eps=cfg.eps)
        df_fan = df_fan.dropna(subset=["y_fan"])

        # 共同样本（common-sample）口径：
        # 为了保证“粉丝-评委差异”不被样本不一致影响，后续会用 df_fan
        # 在同一批行上再拟合一次评委模型，并输出 *_common_k{K} 文件。
        df_common = df_fan.dropna(
            subset=["y_judge", "age_c", "log_week", "industry", "ballroom_partner", "season"]
        ).copy()

        fit_idx += 1
        print(f"[Q3] 拟合进度 {fit_idx}/{fit_total}：粉丝模型（点估计）")
        model_f, idata_f = fit_bambi_model(
            df=df_fan,
            outcome="y_fan",
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            target_accept=cfg.target_accept,
            random_seed=cfg.random_seed,
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

        # 6b) 共同样本版（仅对齐评委侧，粉丝侧无需重跑）
        # 说明：粉丝点估计模型天然只使用 y_fan 可用行；这里把评委模型也对齐到同一批行。
        if not df_common.empty:
            fit_idx += 1
            print(f"[Q3] 拟合进度 {fit_idx}/{fit_total}：评委模型（共同样本对齐）")
            _, idata_jc = fit_bambi_model(
                df=df_common,
                outcome="y_judge",
                draws=cfg.draws,
                tune=cfg.tune,
                chains=cfg.chains,
                target_accept=cfg.target_accept,
                random_seed=cfg.random_seed + 10_000,
                progressbar=cfg.pymc_progressbar,
            )

            judge_fixed_c = summarize_fixed_effects(idata_jc)
            _save_csv(judge_fixed_c, cfg.output_dir / f"judge_fixed_summary_common_k{k}.csv")

            pro_judge_c = summarize_random_effects(idata_jc, "ballroom_partner")
            season_judge_c = summarize_random_effects(idata_jc, "season")
            _save_csv(pro_judge_c, cfg.output_dir / f"judge_pro_effects_common_k{k}.csv")
            _save_csv(season_judge_c, cfg.output_dir / f"judge_season_effects_common_k{k}.csv")

            delta_c = summarize_delta(idata_f, idata_jc)
            _save_csv(delta_c, cfg.output_dir / f"delta_fixed_summary_common_k{k}.csv")

            corr_c, n_pairs_c = pro_effect_correlation(pro_fan, pro_judge_c)
            with open(
                cfg.output_dir / f"pro_effect_correlation_common_k{k}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump({"corr": corr_c, "n_pairs": n_pairs_c}, f, ensure_ascii=False, indent=2)

    # 7) 粉丝侧（NPZ 后验样本）
    if cfg.use_fan_samples:
        npz_index = build_npz_index(cfg.npz_dir)
        sample_series = build_fan_sample_sets(
            df,
            npz_index=npz_index,
            n_sets=cfg.fan_sample_sets,
            seed=cfg.sample_seed,
        )

        fan_means: List[Dict[str, float]] = []
        delta_means: List[Dict[str, float]] = []
        judge_means = fixed_effect_mean_dict(idata_j)

        for i, s in _wrap_progress(
            list(enumerate(sample_series, start=1)),
            desc="[Q3] 粉丝后验样本集拟合",
            total=len(sample_series),
        ):
            df_s = df.copy()
            df_s["fan_vote_sample"] = s.values
            df_s["y_fan"] = add_fan_centered_log(df_s, "fan_vote_sample", eps=cfg.eps)
            df_s = df_s.dropna(subset=["y_fan"])

            fit_idx += 1
            if tqdm is None:
                print(f"[Q3] 拟合进度 {fit_idx}/{fit_total}：粉丝模型（样本集 {i}/{len(sample_series)}）")

            _, idata_fs = fit_bambi_model(
                df=df_s,
                outcome="y_fan",
                draws=cfg.draws,
                tune=cfg.tune,
                chains=cfg.chains,
                target_accept=cfg.target_accept,
                random_seed=cfg.random_seed + i,
                progressbar=cfg.pymc_progressbar,
            )

            fan_mean = fixed_effect_mean_dict(idata_fs)
            fan_means.append(fan_mean)

            delta_mean = {k: fan_mean[k] - judge_means.get(k, np.nan) for k in fan_mean.keys()}
            delta_means.append(delta_mean)

        fan_q = _quantile_table(fan_means)
        delta_q = _quantile_table(delta_means)

        _save_csv(fan_q, cfg.output_dir / f"fan_sample_fixed_quantiles_k{k}.csv")
        _save_csv(delta_q, cfg.output_dir / f"fan_sample_delta_quantiles_k{k}.csv")

    print(f"Q3 回归完成：输出目录 -> {cfg.output_dir}")


if __name__ == "__main__":
    main()
