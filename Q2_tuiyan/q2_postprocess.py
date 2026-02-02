from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from Q2_tuiyan.q2_config import default_config, season_segment_combine_method
    from Q2_tuiyan.q2_loader import load_q2_data, WeekData
    from Q2_tuiyan.q2_kernel import (
        simulate_elimination_rank_direct,
        simulate_elimination_rank_save,
        simulate_elimination_percent_direct,
        simulate_elimination_percent_save,
    )
else:
    from .q2_config import default_config, season_segment_combine_method
    from .q2_loader import load_q2_data, WeekData
    from .q2_kernel import (
        simulate_elimination_rank_direct,
        simulate_elimination_rank_save,
        simulate_elimination_percent_direct,
        simulate_elimination_percent_save,
    )


def _rank_with_ties_desc_1d(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    sorted_vals = values[order]
    start = np.ones_like(sorted_vals, dtype=bool)
    start[1:] = sorted_vals[1:] != sorted_vals[:-1]
    pos = np.arange(values.shape[0])
    start_pos = np.where(start, pos, 0)
    group_start = np.maximum.accumulate(start_pos)
    rank_sorted = group_start + 1
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = rank_sorted
    return ranks


def _rank_with_ties_desc_2d(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, axis=1, kind="mergesort")
    sorted_vals = np.take_along_axis(values, order, axis=1)
    start = np.ones_like(sorted_vals, dtype=bool)
    start[:, 1:] = sorted_vals[:, 1:] != sorted_vals[:, :-1]
    pos = np.arange(values.shape[1])[None, :]
    start_pos = np.where(start, pos, 0)
    group_start = np.maximum.accumulate(start_pos, axis=1)
    rank_sorted = group_start + 1
    ranks = np.empty_like(order, dtype=int)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = rank_sorted
    return ranks


def _actual_mechanism_name(season: int) -> str:
    """Our best-guess historical mechanism by season segment.

    - S1-S2: rank_direct
    - S3-S27: percent_direct
    - S28+: rank_save (rank + judge save)
    """

    if season <= 2:
        return "rank_direct"
    if 3 <= season <= 27:
        return "percent_direct"
    return "rank_save"


@dataclass(frozen=True)
class WeekDerived:
    season: int
    week: int
    method: str
    n_contestants: int
    reversal_rate: float | None
    p_match_actual: float | None
    p_elim_judge_top: float
    p_elim_judge_bottom: float
    p_elim_fan_top: float
    p_elim_fan_bottom: float
    expected_judge_rank_elim: float
    expected_fan_rank_elim: float
    save_override_rate: float | None


def _simulate_elims(week: WeekData) -> dict[str, np.ndarray]:
    elims: dict[str, np.ndarray] = {}
    elims["rank_direct"] = simulate_elimination_rank_direct(
        week.fan_samples, week.judge_scores, week.contestants
    )
    elims["rank_save"] = simulate_elimination_rank_save(
        week.fan_samples, week.judge_scores, week.contestants
    )
    elims["percent_direct"] = simulate_elimination_percent_direct(
        week.fan_samples, week.judge_share, week.judge_scores, week.contestants
    )
    elims["percent_save"] = simulate_elimination_percent_save(
        week.fan_samples, week.judge_share, week.judge_scores, week.contestants
    )
    return elims


def _compute_save_override_rate(week: WeekData, method: str, elim: np.ndarray) -> float | None:
    # “override” = final eliminated differs from the combined-score bottom-1
    if method == "rank_save":
        judge_rank = _rank_with_ties_desc_1d(week.judge_scores)
        fan_rank = _rank_with_ties_desc_2d(week.fan_samples)
        sum_rank = fan_rank + judge_rank[None, :]
        worst1 = np.argmax(sum_rank, axis=1)
        return float(np.mean(elim != worst1))

    if method == "percent_save":
        total = week.fan_samples + week.judge_share[None, :]
        worst1 = np.argmin(total, axis=1)
        return float(np.mean(elim != worst1))

    return None


def derive_metrics_by_week(core_weeks: list[WeekData]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for week in core_weeks:
        n = len(week.contestants)
        s_count = int(week.fan_samples.shape[0])

        judge_top_mask = week.judge_scores == np.max(week.judge_scores)
        judge_bottom_mask = week.judge_scores == np.min(week.judge_scores)

        fan_top_mask = week.fan_samples == np.max(week.fan_samples, axis=1, keepdims=True)
        fan_bottom_mask = week.fan_samples == np.min(week.fan_samples, axis=1, keepdims=True)

        judge_rank = _rank_with_ties_desc_1d(week.judge_scores)
        fan_rank = _rank_with_ties_desc_2d(week.fan_samples)

        actual_idx: int | None
        if len(week.actual_eliminated) == 0:
            actual_idx = None
        else:
            try:
                actual_idx = week.contestants.index(week.actual_eliminated[0])
            except ValueError:
                actual_idx = None

        elims = _simulate_elims(week)
        for method, elim in elims.items():
            elim = elim.astype(int)

            p_match_actual = None
            reversal = None
            if actual_idx is not None:
                p_match_actual = float(np.mean(elim == actual_idx))
                reversal = float(1.0 - p_match_actual)

            p_elim_judge_top = float(np.mean(judge_top_mask[elim]))
            p_elim_judge_bottom = float(np.mean(judge_bottom_mask[elim]))

            rows_idx = np.arange(s_count)
            p_elim_fan_top = float(np.mean(fan_top_mask[rows_idx, elim]))
            p_elim_fan_bottom = float(np.mean(fan_bottom_mask[rows_idx, elim]))

            expected_judge_rank_elim = float(np.mean(judge_rank[elim]))
            expected_fan_rank_elim = float(np.mean(fan_rank[rows_idx, elim]))

            save_override_rate = _compute_save_override_rate(week, method, elim)

            rows.append(
                {
                    "season": week.season,
                    "week": week.week,
                    "method": method,
                    "n_contestants": n,
                    "p_match_actual": p_match_actual,
                    "reversal_rate": reversal,
                    "p_elim_judge_top": p_elim_judge_top,
                    "p_elim_judge_bottom": p_elim_judge_bottom,
                    "p_elim_fan_top": p_elim_fan_top,
                    "p_elim_fan_bottom": p_elim_fan_bottom,
                    "expected_judge_rank_elim": expected_judge_rank_elim,
                    "expected_fan_rank_elim": expected_fan_rank_elim,
                    "save_override_rate": save_override_rate,
                    "season_method": season_segment_combine_method(int(week.season)),
                    "actual_mechanism": _actual_mechanism_name(int(week.season)),
                }
            )

    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame, group_cols: list[str], out_path: Path) -> None:
    if df.empty:
        df.to_csv(out_path, index=False)
        return

    metrics = [
        "reversal_rate",
        "p_match_actual",
        "p_elim_judge_top",
        "p_elim_judge_bottom",
        "p_elim_fan_top",
        "p_elim_fan_bottom",
        "expected_judge_rank_elim",
        "expected_fan_rank_elim",
        "save_override_rate",
    ]

    out = (
        df.groupby(group_cols)[metrics]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(group_cols)
        .reset_index(drop=True)
    )
    out.to_csv(out_path, index=False)


def _pairwise_deltas(df: pd.DataFrame, out_path: Path) -> None:
    """Produce difference tables that map directly to the prompt.

    - rank_direct vs percent_direct (mechanism comparison)
    - rank_save vs rank_direct (judge save effect under rank)
    - percent_save vs percent_direct (judge save effect under percent; hypothetical)

    Computed as mean(metric_A - metric_B) across comparable (season, week).
    """

    if df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return

    key_cols = ["season", "week"]
    metrics = [
        "reversal_rate",
        "p_match_actual",
        "p_elim_judge_bottom",
        "p_elim_fan_bottom",
        "p_elim_judge_top",
        "p_elim_fan_top",
        "expected_judge_rank_elim",
        "expected_fan_rank_elim",
        "save_override_rate",
    ]

    def pivot(method: str) -> pd.DataFrame:
        sub = df[df["method"] == method].copy()
        return sub[key_cols + metrics]

    pairs = [
        ("rank_direct", "percent_direct", "rank_minus_percent"),
        ("rank_save", "rank_direct", "rank_save_minus_rank_direct"),
        ("percent_save", "percent_direct", "percent_save_minus_percent_direct"),
    ]

    rows = []
    for a, b, label in pairs:
        da = pivot(a)
        db = pivot(b)
        merged = da.merge(db, on=key_cols, suffixes=("_a", "_b"))
        if merged.empty:
            continue
        for m in metrics:
            a_col = f"{m}_a"
            b_col = f"{m}_b"
            delta = (merged[a_col] - merged[b_col]).mean()
            rows.append({"comparison": label, "metric": m, "mean_delta": float(delta)})

    pd.DataFrame(rows).to_csv(out_path, index=False)


def _write_markdown_report(
    derived_week: pd.DataFrame,
    summary_overall: Path,
    summary_by_season: Path,
    deltas_path: Path,
    out_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Q2 赛制反事实比较：结果二次加工报告")
    lines.append("")
    lines.append("本报告用于把 Q2 的 Monte Carlo 推演指标与题目第二小问的表述更直接挂钩。")
    lines.append("")

    lines.append("## 1. 我们用哪些指标回答题干？")
    lines.append("")
    lines.append("- **规则更偏向粉丝投票？** 重点看 `p_elim_fan_bottom`（淘汰落在当周粉丝最低者的概率）与 `expected_fan_rank_elim`（被淘汰者的期望粉丝名次，越接近 N 越说明淘汰更由粉丝决定）。")
    lines.append("- **规则更偏向评委评分？** 重点看 `p_elim_judge_bottom` 与 `expected_judge_rank_elim`（越接近 N 越由评委决定）。")
    lines.append("- **加入‘评委救人’环节影响？** 重点看 `save_override_rate`：底二确定后，评委是否经常推翻‘综合分最低者’的淘汰决定。")
    lines.append("- **是否能复刻历史淘汰？** 仍保留 `p_match_actual`/`reversal_rate`（注意：这不是题干唯一目标，只是校验口径一致性）。")
    lines.append("")

    lines.append("## 2. 输出文件")
    lines.append("")
    lines.append(f"- Overall 汇总：`{summary_overall.name}`")
    lines.append(f"- Season 汇总：`{summary_by_season.name}`")
    lines.append(f"- 机制差分表：`{deltas_path.name}`")
    lines.append("")

    if not derived_week.empty:
        # quick headline numbers for deltas
        try:
            deltas = pd.read_csv(deltas_path)
        except Exception:
            deltas = pd.DataFrame()

        lines.append("## 3. 机制差分（最贴题干的一张表）")
        lines.append("")
        if deltas.empty:
            lines.append("（差分表为空：可能是某些方法没有可比周次。）")
        else:
            lines.append("下表给出 **A-B 的均值差**（正值表示 A 更高）：")
            lines.append("")
            # keep markdown short: only show key metrics
            focus = deltas[deltas["metric"].isin([
                "p_elim_fan_bottom",
                "p_elim_judge_bottom",
                "expected_fan_rank_elim",
                "expected_judge_rank_elim",
                "save_override_rate",
            ])].copy()
            if not focus.empty:
                lines.append(focus.to_markdown(index=False))
            else:
                lines.append(deltas.to_markdown(index=False))

        lines.append("")
        lines.append("## 4. 写作提示（如何落到第二小问）")
        lines.append("")
        lines.append("- 比较 **Rank vs Percent**：用 `rank_minus_percent` 这一行，看粉丝侧指标是否更大（更偏粉丝）以及评委侧指标是否更小（更弱评委）。")
        lines.append("- 比较 **是否加入救人**：用 `rank_save_minus_rank_direct` / `percent_save_minus_percent_direct`，看 `save_override_rate` 是否显著>0，以及技术侧风险（如 `p_elim_judge_top`）是否下降。")
        lines.append("- 个案（Jerry/Billy/Bristol/Bobby）：建议在正文里引用你们现有 tracker 曲线，同时补一句“该机制下他在某些周成为 bottom-2/被判淘汰的概率峰值”。")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(output_dir: Path) -> None:
    cfg = default_config()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_q2_data(cfg.clean_csv, cfg.npz_dir)

    derived_week = derive_metrics_by_week(loaded.core_weeks)
    derived_week.to_csv(output_dir / "derived_metrics_by_week.csv", index=False)

    _summarize(derived_week, ["method"], output_dir / "derived_metrics_overall.csv")
    _summarize(derived_week, ["season", "method"], output_dir / "derived_metrics_by_season.csv")
    _summarize(derived_week, ["season_method", "method"], output_dir / "derived_metrics_by_era.csv")

    deltas_path = output_dir / "derived_pairwise_deltas.csv"
    _pairwise_deltas(derived_week, deltas_path)

    _write_markdown_report(
        derived_week=derived_week,
        summary_overall=output_dir / "derived_metrics_overall.csv",
        summary_by_season=output_dir / "derived_metrics_by_season.csv",
        deltas_path=deltas_path,
        out_path=output_dir / "q2_method_comparison_report.md",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess Q2 outputs for prompt-aligned metrics")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default outputs/q2_counterfactual)",
    )
    args = parser.parse_args()

    cfg = default_config()
    out_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir
    run(out_dir)


if __name__ == "__main__":
    main()
