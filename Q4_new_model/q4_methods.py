from __future__ import annotations

from collections.abc import Callable

from Q2_tuiyan.q2_analyze import MethodResult, default_method_results
from Q2_tuiyan.q2_loader import WeekData

from .q4_daws_kernel import (
    simulate_elimination_daws_direct,
    simulate_elimination_daws_save,
)


def compute_w_fan(n_contestants: int, w_early: float, w_late: float, n_cut: int) -> float:
    if n_cut <= 0:
        raise ValueError(f"n_cut must be positive; got {n_cut}")
    if not (0.0 <= float(w_late) <= float(w_early) <= 1.0):
        raise ValueError("require 0 <= w_late <= w_early <= 1")

    if int(n_contestants) <= int(n_cut):
        return float(w_late)
    return float(w_early)


def make_daws_method_results_fn(
    *,
    w_early: float,
    w_late: float,
    n_cut: int,
    save_enabled: bool,
    include_baseline: bool = True,
) -> Callable[[WeekData], list[MethodResult]]:
    """Create a method_results_fn compatible with Q2 analyze_* functions."""

    def _fn(week: WeekData) -> list[MethodResult]:
        methods: list[MethodResult] = []
        if include_baseline:
            methods.extend(default_method_results(week))

        w_fan = compute_w_fan(
            n_contestants=len(week.contestants),
            w_early=w_early,
            w_late=w_late,
            n_cut=n_cut,
        )

        methods.append(
            MethodResult(
                name="daws_direct",
                eliminations=simulate_elimination_daws_direct(
                    week.fan_samples,
                    week.judge_share,
                    week.judge_scores,
                    week.contestants,
                    w_fan,
                ),
            )
        )

        if save_enabled:
            methods.append(
                MethodResult(
                    name="daws_save",
                    eliminations=simulate_elimination_daws_save(
                        week.fan_samples,
                        week.judge_share,
                        week.judge_scores,
                        week.contestants,
                        w_fan,
                    ),
                )
            )

        return methods

    return _fn
