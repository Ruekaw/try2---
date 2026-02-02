# -*- coding: utf-8 -*-
"""
Q3 数据加载与 NPZ 解析
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class NPZWeekSamples:
    season: int
    week: int
    contestants: List[str]
    name_to_idx: Dict[str, int]
    samples: np.ndarray  # shape (S, N)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower()


def read_dwts_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def read_fan_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def merge_dwts_fan(dwts_df: pd.DataFrame, fan_df: pd.DataFrame) -> pd.DataFrame:
    key = ["season", "week", "celebrity_name"]
    merged = dwts_df.merge(fan_df, on=key, how="left")
    return merged


def build_npz_index(npz_dir: Path) -> Dict[Tuple[int, int], NPZWeekSamples]:
    """
    读取 Q1 导出的 NPZ 文件，构建 (season, week) -> NPZWeekSamples 映射
    """
    index: Dict[Tuple[int, int], NPZWeekSamples] = {}
    for npz_path in sorted(npz_dir.glob("season_*_samples.npz")):
        season_str = npz_path.stem.replace("season_", "").replace("_samples", "")
        try:
            season = int(season_str)
        except ValueError:
            continue

        data = np.load(npz_path, allow_pickle=True)
        weeks = data.get("weeks", [])
        for wk in weeks:
            week = int(wk)
            contestants = list(data[f"week_{week}_contestants"])
            name_to_idx = {_normalize_name(nm): i for i, nm in enumerate(contestants)}
            samples = data[f"week_{week}_samples"]  # shape (S, N)
            index[(season, week)] = NPZWeekSamples(
                season=season,
                week=week,
                contestants=contestants,
                name_to_idx=name_to_idx,
                samples=samples,
            )
    return index


def build_fan_sample_sets(
    df: pd.DataFrame,
    npz_index: Dict[Tuple[int, int], NPZWeekSamples],
    n_sets: int,
    seed: int,
) -> List[pd.Series]:
    """
    为完整面板构造 n_sets 组 fan_share 样本序列。
    返回与 df 行对齐的 Series 列表（长度 n_sets）。
    """
    rng = np.random.default_rng(seed)
    series_list: List[pd.Series] = []

    # 预分组，减少重复索引
    df = df.copy()
    df["_row_id"] = np.arange(len(df))
    grouped = df.groupby(["season", "week"], sort=False)

    # 为每组预先确定 samples 句柄
    group_cache: List[Tuple[np.ndarray, Dict[str, int], np.ndarray]] = []
    for (season, week), g in grouped:
        key = (int(season), int(week))
        if key not in npz_index:
            samples = None
            name_to_idx = None
        else:
            samples = npz_index[key].samples
            name_to_idx = npz_index[key].name_to_idx
        row_ids = g["_row_id"].to_numpy()
        names = g["celebrity_name"].astype(str).to_list()
        group_cache.append((samples, name_to_idx, row_ids, names))

    for _ in range(n_sets):
        values = np.full(len(df), np.nan, dtype=float)
        for samples, name_to_idx, row_ids, names in group_cache:
            if samples is None or name_to_idx is None:
                continue
            s_idx = rng.integers(0, samples.shape[0])
            for rid, nm in zip(row_ids, names):
                col = name_to_idx.get(_normalize_name(nm))
                if col is not None:
                    values[rid] = float(samples[s_idx, col])
        series_list.append(pd.Series(values, index=df.index))

    return series_list
