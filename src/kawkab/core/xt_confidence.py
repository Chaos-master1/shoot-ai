"""Bootstrap confidence intervals for xT values."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from kawkab.core.xt_model import ExpectedThreatModel


@dataclass
class XtInterval:
    mean: float
    ci_low: float
    ci_high: float
    std: float


def _normalize_event(e: dict) -> dict:
    return {
        "start_x": e.get("x", e.get("start_x", 50.0)),
        "start_y": e.get("y", e.get("start_y", 34.0)),
        "end_x": e.get("end_x", 55.0),
        "end_y": e.get("end_y", 34.0),
        "type": e.get("type", "pass"),
        "completed": e.get("completed", True),
        "is_goal": e.get("is_goal", False),
        "team": e.get("team", "home"),
    }


def _compute_xt_from_events(events: list[dict], grid_size: tuple[int, int]) -> dict[str, float]:
    model = ExpectedThreatModel(rows=grid_size[0], cols=grid_size[1])
    xt_events = [_normalize_event(e) for e in events if e.get("type") in ("pass", "carry", "shot")]
    if not xt_events:
        return {}
    model.build_transition_matrix(xt_events)

    zone_vals = model.get_zone_values()
    result: dict[str, float] = {}
    for r in range(zone_vals.shape[0]):
        for c in range(zone_vals.shape[1]):
            result[f"({r}, {c})"] = float(zone_vals[r, c])
    return result


def bootstrap_xt(
    events: list[dict], n_resamples: int = 1000, grid_size: tuple[int, int] = (20, 32)
) -> dict[str, XtInterval]:
    if not events:
        return {}

    full_values = _compute_xt_from_events(events, grid_size)
    zone_names = list(full_values.keys())

    bootstrap_samples: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(42)
    n = len(events)

    for _ in range(n_resamples):
        sample = [events[rng.randint(0, n - 1)] for _ in range(n)]
        try:
            vals = _compute_xt_from_events(sample, grid_size)
        except Exception:
            continue
        for zname in zone_names:
            v = vals.get(zname, full_values.get(zname, 0.0))
            bootstrap_samples[zname].append(v)

    intervals: dict[str, XtInterval] = {}
    for zname in zone_names:
        samples = bootstrap_samples.get(zname, [full_values.get(zname, 0.0)])
        arr = np.array(samples, dtype=np.float64)
        mean = float(np.mean(arr))
        low = float(np.percentile(arr, 2.5))
        high = float(np.percentile(arr, 97.5))
        std = float(np.std(arr))
        intervals[zname] = XtInterval(mean=mean, ci_low=low, ci_high=high, std=std)

    return intervals


def zone_xt_with_ci(
    events: list[dict],
    grid_size: tuple[int, int] = (20, 32),
    n_resamples: int = 100,
) -> dict[tuple[int, int], XtInterval]:
    intervals = bootstrap_xt(events, n_resamples=n_resamples, grid_size=grid_size)
    parsed: dict[tuple[int, int], XtInterval] = {}
    for key, interval in intervals.items():
        try:
            parts = key.strip("()").split(", ")
            row, col = int(parts[0]), int(parts[1])
            parsed[(row, col)] = interval
        except (ValueError, IndexError):
            pass
    return parsed
