"""Acute:Chronic Workload Ratio (ACWR) computation.

ACWR measures whether a player's current training load (acute, 7-day
rolling) relative to their chronic baseline (28-day rolling) is in a
safe zone. Based on Gabbett (2016): "The training-injury prevention
paradox: should athletes be training smarter and harder?"

Ranges (Gabbett 2016):
  - < 0.8:   Low / Under-training (may be deconditioned)
  - 0.8-1.3: Sweet spot (safe)
  - 1.3-1.5: High (increased risk)
  - > 1.5:   Very high (danger zone)

The acute load is the exponentially-weighted moving average (EWMA) of
daily loads (decay k=3, the standard "acute" time constant); chronic
load is the EWMA with k=7, which approximates a 28-day rolling baseline.
EWMA (Williams et al. 2017, "How to use acute:chronic workload ratios to
manage training load") reacts to load spikes faster than the
same-length rolling averages it replaced: a rolling 7-day average can
miss a mid-week spike entirely, while EWMA weights recent days heavier.

Caveat surfaced for consumers: for roughly the first 28 days both
averages are still warming up, and ratios computed from them are not
yet a trustworthy chronic baseline. Each row therefore carries a
`reliable` flag that turns on after 28 days of data.

Player load metric options:
  - total_distance_m (most common)
  - player_load (accelerometer-derived, Catapult PL units)
  - high_intensity_distance_m (sprint / HI running)
  - sRPE (session Rating of Perceived Exertion) — requires manual input
  - custom_weighted (combination)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

logger = logging.getLogger("acwr")


def compute_acwr(
    daily_loads: list[dict[str, Any]],
    load_field: str = "total_distance_m",
) -> list[dict[str, Any]]:
    """Compute ACWR for a player over a timeline of daily loads.

    Args:
        daily_loads: List of dicts with 'date' (str/date) and load_field (float).
                     Must be sorted by date ascending.
        load_field: Which field to use for load.

    Returns:
        List with same length as input, each dict augmented with:
          acute_load, chronic_load, acwr, load_category, reliable

    The first row initializes both averages to the day's own load, so
    day-one ACWR is 1.0 by construction; `reliable` is False until the
    chronic average has seen 28 days.
    """
    if not daily_loads:
        return []

    results = []
    acute_decay = 3.0  # Williams et al. (2017) acute time constant
    chronic_decay = 7.0  # ~28-day baseline equivalent
    acute_ewma: float | None = None
    chronic_ewma: float | None = None

    for i, entry in enumerate(daily_loads):
        load = float(entry.get(load_field, 0) or 0)

        if acute_ewma is None or chronic_ewma is None:
            acute_val = load
            chronic_val = load
        else:
            acute_val = acute_ewma + (load - acute_ewma) / acute_decay
            chronic_val = chronic_ewma + (load - chronic_ewma) / chronic_decay
        acute_ewma = acute_val
        chronic_ewma = chronic_val
        acute = acute_val
        chronic = chronic_val

        ratio = acute / max(chronic, 0.001)

        if ratio > 1.5:
            category = "very_high"
        elif ratio > 1.3:
            category = "high"
        elif ratio < 0.8:
            category = "low"
        else:
            category = "normal"

        results.append(
            {
                **entry,
                "acute_load": round(acute_ewma, 1),
                "chronic_load": round(chronic_ewma, 1),
                "acwr": round(ratio, 3),
                "load_category": category,
                "reliable": i >= 27,
            }
        )

    return results


def compute_acwr_from_sessions(
    sessions: list[dict[str, Any]],
    load_field: str = "total_distance_m",
    date_field: str = "start_time",
) -> list[dict[str, Any]]:
    """Compute ACWR from GPS session records, aggregating load by date.

    Rest days carry zero load and MUST be part of the series — a series
    built only from days with sessions inflates both averages and
    distorts the ratio. The date range from first to last session is
    therefore expanded and missing days filled with 0.

    Args:
        sessions: List of GPS session dicts with date_field and load_field.
        load_field: 'total_distance_m', 'player_load', 'hi_distance_m', etc.
        date_field: Field name for the date.

    Returns:
        List of daily ACWR results (one per day with data).
    """
    if not sessions:
        return []

    daily: dict[str, float] = defaultdict(float)
    parsed: dict[str, datetime] = {}
    for s in sessions:
        raw_date = s.get(date_field, "")
        if isinstance(raw_date, datetime):
            date_key = raw_date.strftime("%Y-%m-%d")
            parsed[date_key] = raw_date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif isinstance(raw_date, str):
            date_key = raw_date[:10]
        else:
            continue
        load = s.get(load_field, 0) or 0
        daily[date_key] += load

    if not daily:
        return []

    # Expand first→last session date, zero-filling rest days.
    from datetime import timedelta

    first = min(parsed.get(d, datetime.strptime(d, "%Y-%m-%d")) for d in daily)
    last = max(parsed.get(d, datetime.strptime(d, "%Y-%m-%d")) for d in daily)
    daily_loads: list[dict[str, Any]] = []
    day = first
    while day <= last:
        key = day.strftime("%Y-%m-%d")
        daily_loads.append({"date": key, load_field: daily.get(key, 0.0)})
        day += timedelta(days=1)

    return compute_acwr(daily_loads, load_field=load_field)


def assess_injury_risk(acwr_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess overall injury risk from ACWR trend data.

    Returns dict with risk level, dangerous days, and recommendations.
    """
    if not acwr_data:
        return {"risk_level": "unknown", "danger_days": 0, "recommendations": []}

    total = len(acwr_data)
    high_days = sum(1 for d in acwr_data if d.get("load_category") in ("high", "very_high"))
    low_days = sum(1 for d in acwr_data if d.get("load_category") == "low")
    sweet_days = total - high_days - low_days

    high_ratio = high_days / max(total, 1)
    latest = acwr_data[-1] if acwr_data else {}

    if latest.get("load_category") in ("high", "very_high"):
        risk = "critical" if high_ratio > 0.3 else "elevated"
    elif low_days > total * 0.5:
        risk = "deconditioned"
    else:
        risk = "normal"

    recommendations = []
    if risk == "critical":
        recommendations.extend(
            [
                "Reduce training load by 30-50% for 3-5 days",
                "Consider rest day or active recovery",
                "Monitor for early injury signs",
            ]
        )
    elif risk == "elevated":
        recommendations.extend(
            [
                "Maintain current load but avoid spikes",
                "Monitor player-reported fatigue",
            ]
        )
    elif risk == "deconditioned":
        recommendations.extend(
            [
                "Gradually increase load by 10% per week",
                "Focus on building base fitness",
            ]
        )
    else:
        recommendations.append("Continue current training load")

    return {
        "risk_level": risk,
        "total_days": total,
        "sweet_spot_days": sweet_days,
        "high_days": high_days,
        "low_days": low_days,
        "high_ratio": round(high_ratio, 2),
        "latest_acwr": latest.get("acwr"),
        "latest_category": latest.get("load_category"),
        "recommendations": recommendations,
    }
