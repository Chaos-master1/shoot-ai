"""Credible intervals for xG, xT, and VAEP metrics.

Provides uncertainty estimates using:
  - Beta conjugate prior for xG (per distance/angle bucket)
  - Bootstrap resampling for xT
  - Block bootstrap for VAEP (autocorrelated events)
"""

from __future__ import annotations

import math

import numpy as np

from kawkab.core.vaep import compute_vaep
from kawkab.core.xt_model import ExpectedThreatModel


def xg_credible_interval(
    shots: list[dict],
    alpha_prior: float = 1.1,
    beta_prior: float = 1.1,
    n_simulations: int = 10000,
) -> dict[str, float]:
    """Compute 90 % credible interval for total xG via Beta conjugate.

    Shots are bucketed by distance (5 m bins) and angle (10° bins).
    Each bucket's posterior is Beta(alpha_prior + goals, beta_prior +
    non_goals).  The interval is obtained by Monte Carlo simulation
    across all buckets.

    Parameters
    ----------
    shots : list[dict]
        Each dict must contain ``xg``, ``is_goal``, ``distance_m``,
        and ``angle_deg``.
    alpha_prior, beta_prior : float
        Beta prior shape parameters (default 1.1 — weakly informative).
    n_simulations : int
        Number of Monte Carlo draws (default 10000).

    Returns
    -------
    dict[str, float]
        ``total_xg``, ``lower_90``, ``upper_90``.
    """
    buckets: dict[tuple[int, int], dict[str, int]] = {}
    total_xg = 0.0

    for s in shots:
        d = int(s.get("distance_m", 18) / 5) * 5
        a = int(s.get("angle_deg", 30) / 10) * 10
        key = (d, a)
        if key not in buckets:
            buckets[key] = {"shots": 0, "goals": 0}
        buckets[key]["shots"] += 1
        if s.get("is_goal"):
            buckets[key]["goals"] += 1
        total_xg += s.get("xg", 0.0)

    if not buckets:
        return {"total_xg": 0.0, "lower_90": 0.0, "upper_90": 0.0}

    sim_goals = np.zeros(n_simulations, dtype=np.float64)
    for i in range(n_simulations):
        total = 0.0
        for data in buckets.values():
            alpha_i = alpha_prior + data["goals"]
            beta_i = beta_prior + data["shots"] - data["goals"]
            p = float(np.random.beta(alpha_i, beta_i))
            total += p * data["shots"]
        sim_goals[i] = total

    lower = float(np.percentile(sim_goals, 5))
    upper = float(np.percentile(sim_goals, 95))
    return {"total_xg": total_xg, "lower_90": lower, "upper_90": upper}


def _compute_total_xt(
    events: list[dict],
    rows: int = 20,
    cols: int = 32,
) -> float:
    """Build xT model from *events* and return home + away total."""
    model = ExpectedThreatModel(rows=rows, cols=cols)
    model.build_transition_matrix(events)
    return sum(model.compute_match_xt(events).values())


def xt_credible_interval(
    events: list[dict],
    zone_values: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    rows: int = 20,
    cols: int = 32,
) -> dict[str, float]:
    """Compute 95 % credible interval for total xT via bootstrap resampling.

    Parameters
    ----------
    events : list[dict]
        Event sequence (passes, carries, shots).
    zone_values : np.ndarray | None
        Ignored (kept for API compat); model is rebuilt per resample.
    n_bootstrap : int
        Number of bootstrap replicates (default 1000).
    rows, cols : int
        xT grid dimensions (default 20 × 32).

    Returns
    -------
    dict[str, float]
        ``total_xt``, ``lower_95``, ``upper_95``.
    """
    if not events:
        return {"total_xt": 0.0, "lower_95": 0.0, "upper_95": 0.0}

    n = len(events)
    base_xt = _compute_total_xt(events, rows, cols)
    replicates = np.zeros(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        sample = [events[int(j)] for j in indices]
        replicates[i] = _compute_total_xt(sample, rows, cols)

    lower = float(np.percentile(replicates, 2.5))
    upper = float(np.percentile(replicates, 97.5))
    return {"total_xt": base_xt, "lower_95": lower, "upper_95": upper}


def _compute_total_vaep(events: list[dict]) -> float:
    """Run VAEP on *events* and return sum of absolute VAEP values."""
    results = compute_vaep(events)
    return sum(abs(r.get("vaep_value", 0.0)) for r in results)


def vaep_credible_interval(
    events: list[dict],
    n_bootstrap: int = 1000,
    block_size: int = 20,
) -> dict[str, float]:
    """Compute 95 % credible interval for total |VAEP| via block bootstrap.

    Events are autocorrelated, so we resample *blocks* (contiguous
    segments) with replacement to preserve within-possession structure.

    Parameters
    ----------
    events : list[dict]
        Sorted event sequence.
    n_bootstrap : int
        Number of bootstrap replicates (default 1000).
    block_size : int
        Number of events per block (default 20).

    Returns
    -------
    dict[str, float]
        ``total_vaep``, ``lower_95``, ``upper_95``.
    """
    if not events:
        return {"total_vaep": 0.0, "lower_95": 0.0, "upper_95": 0.0}

    n = len(events)
    base_vaep = _compute_total_vaep(events)
    n_blocks = max(1, math.ceil(n / block_size))
    block_starts = [i * block_size for i in range(n_blocks)]

    replicates = np.zeros(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        sample: list[dict] = []
        for _ in range(n_blocks):
            start = int(np.random.choice(block_starts))
            end = min(start + block_size, n)
            sample.extend(events[start:end])
        replicates[i] = _compute_total_vaep(sample)

    lower = float(np.percentile(replicates, 2.5))
    upper = float(np.percentile(replicates, 97.5))
    return {"total_vaep": base_vaep, "lower_95": lower, "upper_95": upper}
