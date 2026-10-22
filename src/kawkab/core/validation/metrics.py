"""Probabilistic-forecast validation metrics — numpy-only, no sklearn.

All functions accept plain numpy arrays (or lists) of predicted
probabilities and binary outcomes. They are pure functions: no I/O, no
logging of data values, safe for hypothesis-style property tests.

Conventions:
    - ``y_true``: 1.0 = event happened (goal), 0.0 = not
    - ``y_proba``: model probability of the event, in [0, 1]
    - NaN/inf inputs raise ``ValueError`` — a silent nan would propagate
      into the validation report, which must never contain a fabricated
      or garbage number.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "brier_score",
    "brier_skill_score",
    "reliability_curve",
    "reliability_residuals",
    "roc_auc",
    "log_loss",
    "calibration_error",
    "mean_absolute_error",
    "pearson_correlation",
    "bootstrap_ci",
]


def _check_pair(
    y_true: np.ndarray | list[Any], y_proba: np.ndarray | list[Any]
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if y_true.shape != y_proba.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_proba {y_proba.shape}")
    if y_true.size == 0:
        raise ValueError("empty input")
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_proba)):
        raise ValueError("non-finite input")
    if np.any((y_proba < 0.0) | (y_proba > 1.0)):
        raise ValueError("y_proba outside [0, 1]")
    if not np.all(np.isin(y_true, (0.0, 1.0))):
        raise ValueError("y_true must be binary 0/1")
    return y_true, y_proba


def brier_score(y_true: np.ndarray | list[Any], y_proba: np.ndarray | list[Any]) -> float:
    """Mean Brier score (lower is better; 0.25 = constant 0.5 guess)."""
    yt, yp = _check_pair(y_true, y_proba)
    return float(np.mean((yp - yt) ** 2))


def brier_skill_score(
    y_true: np.ndarray | list[Any],
    y_proba: np.ndarray | list[Any],
    *,
    reference: str = "climatology",
) -> float:
    """Brier skill score vs a reference forecast (higher is better).

    reference="climatology" uses the observed base rate as the reference
    forecast (the standard skill-score baseline).
    """
    yt, yp = _check_pair(y_true, y_proba)
    if reference == "climatology":
        ref = float(np.mean(yt))
    elif reference == "uniform":
        ref = 0.5
    else:
        raise ValueError(f"unknown reference {reference!r}")
    b_model = np.mean((yp - yt) ** 2)
    b_ref = np.mean((ref - yt) ** 2)
    if b_ref < 1e-12:
        return 0.0
    return float(1.0 - b_model / b_ref)


def reliability_curve(
    y_true: np.ndarray | list,
    y_proba: np.ndarray | list,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> list[dict[str, float]]:
    """Reliability (calibration) curve.

    Returns one dict per non-empty bin: ``bin_low``, ``bin_high``,
    ``mean_pred``, ``observed_rate``, ``n``. With strategy="uniform" the
    bins are equal-width over [0, 1]; "quantile" gives equal-count bins.
    """
    yt, yp = _check_pair(y_true, y_proba)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(yp, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.maximum.accumulate(edges)  # guard degenerate quantiles
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    out: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (yp >= lo) if i == n_bins - 1 else ((yp >= lo) & (yp < hi))
        n_in = int(np.sum(mask))
        if n_in == 0:
            continue
        out.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "mean_pred": float(np.mean(yp[mask])),
                "observed_rate": float(np.mean(yt[mask])),
                "n": n_in,
            }
        )
    return out


def reliability_residuals(
    y_true: np.ndarray | list,
    y_proba: np.ndarray | list,
    *,
    n_bins: int = 10,
) -> list[float]:
    """observed_rate - mean_pred per non-empty reliability bin."""
    curve = reliability_curve(y_true, y_proba, n_bins=n_bins)
    return [c["observed_rate"] - c["mean_pred"] for c in curve]


def roc_auc(y_true: np.ndarray | list, y_proba: np.ndarray | list) -> float:
    """Area under the ROC curve via the rank (Mann-Whitney U) statistic.

    Ties receive the average rank. Raises ValueError when y_true is
    single-class (AUC is undefined).
    """
    yt, yp = _check_pair(y_true, y_proba)
    n_pos = float(np.sum(yt == 1.0))
    n_neg = float(np.sum(yt == 0.0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC undefined for single-class y_true")

    # Average ranks for ties
    order = np.argsort(yp, kind="mergesort")
    yp_sorted = yp[order]
    ranks = np.empty_like(yp_sorted, dtype=np.float64)
    i = 0
    rank_val = 1.0
    while i < len(yp_sorted):
        j = i
        while j + 1 < len(yp_sorted) and yp_sorted[j + 1] == yp_sorted[i]:
            j += 1
        avg_rank = rank_val + (j - i) / 2.0
        ranks[i : j + 1] = avg_rank
        rank_val += j - i + 1
        i = j + 1

    # Unsort ranks back to original order
    orig_ranks = np.empty_like(ranks)
    orig_ranks[order] = ranks
    sum_pos = float(np.sum(orig_ranks[yt == 1.0]))
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def log_loss(y_true: np.ndarray | list[Any], y_proba: np.ndarray | list[Any]) -> float:
    """Mean binary log loss (lower is better). Clips p to avoid inf."""
    yt, yp = _check_pair(y_true, y_proba)
    p = np.clip(yp, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(yt * np.log(p) + (1.0 - yt) * np.log(1.0 - p)))


def calibration_error(
    y_true: np.ndarray | list[Any],
    y_proba: np.ndarray | list[Any],
    *,
    n_bins: int = 10,
) -> float:
    """Expected calibration error (ECE): |predicted - observed| weighted by bin size."""
    yt, yp = _check_pair(y_true, y_proba)
    total = float(len(yt))
    ece = 0.0
    for c in reliability_curve(yt, yp, n_bins=n_bins):
        ece += (c["n"] / total) * abs(c["mean_pred"] - c["observed_rate"])
    return float(ece)


def mean_absolute_error(y_true: np.ndarray | list, y_pred: np.ndarray | list) -> float:
    """Plain MAE for comparing two sets of per-shot estimates (e.g. vs SB xG)."""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("shape mismatch or empty input")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("non-finite input")
    return float(np.mean(np.abs(a - b)))


def pearson_correlation(a: list | np.ndarray, b: list | np.ndarray) -> float:
    """Pearson r; returns 0.0 for constant inputs (undefined correlation)."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape or a_arr.size < 2:
        raise ValueError("shape mismatch or fewer than 2 samples")
    sa, sb = float(np.std(a_arr)), float(np.std(b_arr))
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    cov = float(np.mean((a_arr - np.mean(a_arr)) * (b_arr - np.mean(b_arr))))
    return cov / (sa * sb)


def bootstrap_ci(
    values: list | np.ndarray,
    statistic: str = "mean",
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap CI for a statistic over per-shot values.

    statistic: "mean" | "sum" | "median". Returns
    ``{value, ci_lower, ci_upper, confidence, n}``.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("empty input")
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite input")

    fns: dict[str, Callable[[Any], Any]] = {
        "mean": np.mean,
        "sum": np.sum,
        "median": np.median,
    }
    fn = fns.get(statistic)
    if fn is None:
        raise ValueError(f"unknown statistic {statistic!r}")

    value = float(fn(arr))
    if arr.size < 5 or n_bootstrap < 2:
        return {
            "value": value,
            "ci_lower": value,
            "ci_upper": value,
            "confidence": confidence,
            "n": int(arr.size),
        }

    rng = np.random.default_rng(seed)
    stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = arr[rng.integers(0, arr.size, size=arr.size)]
        stats[i] = fn(sample)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return {
        "value": value,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "confidence": confidence,
        "n": int(arr.size),
    }
