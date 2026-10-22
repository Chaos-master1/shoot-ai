"""Uncertainty intervals for football analytics metrics.

Provides bootstrap-based confidence intervals for xG, xA, PSxG,
and composite metrics. All numpy-only, no scipy dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def bootstrap_metric(
    values: list[float],
    metric_fn: Callable[[np.ndarray], float] | None = None,
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int | None = None,
    return_all: bool = False,
) -> dict[str, Any]:
    """Bootstrap confidence interval for a metric computed on an array of values.

    Args:
        values: Array of observed values (e.g., xG values for shots).
        metric_fn: Function that takes a 1D array and returns a scalar.
                   Default is np.mean.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        return_all: If True, include all bootstrap estimates in output.

    Returns:
        Dict with point_estimate, ci_lower, ci_upper, std_error,
        and optionally bootstrap_samples.
    """
    data = np.array(values, dtype=np.float64)
    n = len(data)
    if n < 2:
        pe = float(data[0]) if n == 1 else 0.0
        return {
            "point_estimate": pe,
            "ci_lower": pe,
            "ci_upper": pe,
            "std_error": 0.0,
            "n": n,
        }

    fn = metric_fn or (lambda x: float(np.mean(x)))
    point_estimate = fn(data)

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    boot_estimates = np.zeros(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = data[rng.integers(0, n, size=n)]
        boot_estimates[i] = fn(sample)

    alpha = 1.0 - ci_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_estimates, lower_pct))
    ci_upper = float(np.percentile(boot_estimates, upper_pct))
    std_error = float(np.std(boot_estimates, ddof=1))

    result: dict[str, Any] = {
        "point_estimate": round(point_estimate, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "std_error": round(std_error, 6),
        "ci_level": ci_level,
        "n": n,
        "n_bootstrap": n_bootstrap,
    }
    if return_all:
        result["bootstrap_samples"] = boot_estimates.tolist()
    return result


def bootstrap_xg_confidence(
    xg_values: list[float],
    goal_flags: list[bool],
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Bootstrap confidence interval for total xG and xG ratio.

    Args:
        xg_values: xG values for each shot.
        goal_flags: Whether each shot was a goal.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level.
        seed: Random seed.

    Returns:
        Dict with total_xG, actual_goals, xG_CI, goals_CI, xG_per_shot_CI.
    """
    data = np.array(xg_values, dtype=np.float64)
    goals = np.array([1.0 if g else 0.0 for g in goal_flags], dtype=np.float64)
    n = len(data)
    if n < 2:
        return {
            "total_xG": float(np.sum(data)),
            "actual_goals": int(np.sum(goals)),
            "n_shots": n,
            "xG_CI": {"lower": float(np.sum(data)), "upper": float(np.sum(data))},
            "goals_CI": {"lower": int(np.sum(goals)), "upper": int(np.sum(goals))},
        }

    total_xg = float(np.sum(data))
    total_goals = int(np.sum(goals))

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    xg_sums = np.zeros(n_bootstrap, dtype=np.float64)
    goal_sums = np.zeros(n_bootstrap, dtype=np.float64)
    xg_means = np.zeros(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xg_sums[i] = np.sum(data[idx])
        goal_sums[i] = np.sum(goals[idx])
        xg_means[i] = np.mean(data[idx])

    alpha = 1.0 - ci_level
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)

    return {
        "total_xG": round(total_xg, 4),
        "actual_goals": total_goals,
        "n_shots": n,
        "xG_CI": {
            "lower": round(float(np.percentile(xg_sums, lo)), 4),
            "upper": round(float(np.percentile(xg_sums, hi)), 4),
        },
        "goals_CI": {
            "lower": round(float(np.percentile(goal_sums, lo)), 1),
            "upper": round(float(np.percentile(goal_sums, hi)), 1),
        },
        "xG_per_shot_CI": {
            "lower": round(float(np.percentile(xg_means, lo)), 4),
            "upper": round(float(np.percentile(xg_means, hi)), 4),
        },
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_psxg_confidence(
    psxg_values: list[float],
    goal_flags: list[bool],
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Bootstrap confidence interval for PSxG (post-shot xG).

    Args:
        psxg_values: PSxG values for each shot on target.
        goal_flags: Whether each shot was a goal.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level.
        seed: Random seed.

    Returns:
        Dict with total_PSxG, goals_against, PSxG_CI, goals_CI.
    """
    return bootstrap_xg_confidence(psxg_values, goal_flags, n_bootstrap, ci_level, seed)


def bootstrap_comparison(
    values_a: list[float],
    values_b: list[float],
    metric_fn: Callable[[np.ndarray], float] | None = None,
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
    seed: int | None = None,
) -> dict[str, Any]:
    """Bootstrap confidence interval for the difference between two groups.

    Args:
        values_a: Observed values for group A.
        values_b: Observed values for group B.
        metric_fn: Metric function (default: np.mean).
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level.
        seed: Random seed.

    Returns:
        Dict with diff, diff_CI, p_value (proportion of bootstrap diffs <= 0).
    """
    a = np.array(values_a, dtype=np.float64)
    b = np.array(values_b, dtype=np.float64)
    fn = metric_fn or (lambda x: float(np.mean(x)))

    diff_observed = fn(a) - fn(b)

    if len(a) < 2 or len(b) < 2:
        return {
            "diff_observed": diff_observed,
            "diff_CI": {"lower": diff_observed, "upper": diff_observed},
            "p_value": 0.5,
            "n_a": len(a),
            "n_b": len(b),
        }

    combined = np.concatenate([a, b])
    n_a, n_total = len(a), len(combined)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    diffs = np.zeros(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_total, size=n_total)
        boot_a = combined[idx[:n_a]]
        boot_b = combined[idx[n_a:]]
        diffs[i] = fn(boot_a) - fn(boot_b)

    alpha = 1.0 - ci_level
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)

    p_value = float(np.mean(diffs <= 0.0))

    return {
        "diff_observed": round(float(diff_observed), 6),
        "diff_CI": {
            "lower": round(float(np.percentile(diffs, lo)), 6),
            "upper": round(float(np.percentile(diffs, hi)), 6),
        },
        "p_value": round(p_value, 4),
        "n_a": n_a,
        "n_b": len(b),
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
    }
