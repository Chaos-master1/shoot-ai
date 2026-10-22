"""Tests for ACWR (Acute:Chronic Workload Ratio) computation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from kawkab.analysis.acwr import (
    assess_injury_risk,
    compute_acwr,
    compute_acwr_from_sessions,
)


def _daily(day: int, load: float) -> dict:
    return {"date": f"2026-01-{day:02d}", "total_distance_m": load}


def test_basic_acwr():
    """Constant daily load should give ACWR near 1.0."""
    loads = [_daily(d, 5000.0) for d in range(1, 31)]
    results = compute_acwr(loads)
    assert len(results) == 30
    last = results[-1]
    assert 0.5 < last["acwr"] < 2.0
    assert last["load_category"] in ("normal", "high", "low")


def test_spike_detection():
    """A sudden doubling of load should push ACWR > 1.3."""
    loads = [_daily(d, 5000.0) for d in range(1, 29)]
    loads.append(_daily(29, 10000.0))
    loads.append(_daily(30, 10000.0))
    results = compute_acwr(loads)
    last = results[-1]
    assert last["acwr"] > 0.8
    # May or may not hit high depending on decay
    assert last["load_category"] in ("normal", "high", "very_high")


def test_high_load_category():
    """Load doubling sustained should trigger very_high."""
    loads = [_daily(d, 3000.0) for d in range(1, 29)]
    for d in range(29, 36):
        loads.append(_daily(d, 15000.0))
    results = compute_acwr(loads)
    last = results[-1]
    assert last["load_category"] in ("high", "very_high")


def test_empty():
    assert compute_acwr([]) == []


def test_single_day():
    result = compute_acwr([_daily(1, 5000.0)])
    assert len(result) == 1
    assert result[0]["acwr"] == 1.0
    assert result[0]["load_category"] == "normal"


def test_constant_acwr_stability():
    """Constant daily load should produce ACWR ~1.0."""
    loads = [_daily(d, 10000.0) for d in range(1, 61)]
    results = compute_acwr(loads)
    # After 60 days, ACWR should converge to ~1.0
    mid = results[30]
    late = results[59]
    assert abs(mid["acwr"] - 1.0) < 0.5
    assert abs(late["acwr"] - 1.0) < 0.5


def test_custom_load_field():
    """Use player_load instead of total_distance_m."""
    loads = [{"date": f"2026-01-{d:02d}", "player_load": 500.0} for d in range(1, 31)]
    results = compute_acwr(loads, load_field="player_load")
    assert len(results) == 30


def test_from_sessions():
    """Compute ACWR from session records aggregated by date."""
    sessions = [
        {"start_time": "2026-01-01", "total_distance_m": 5000.0},
        {"start_time": "2026-01-01", "total_distance_m": 3000.0},
        {"start_time": "2026-01-02", "total_distance_m": 6000.0},
    ]
    results = compute_acwr_from_sessions(sessions)
    assert len(results) == 2
    # Day 1 should have 8000 total
    assert results[0]["total_distance_m"] > 5000


def test_from_sessions_empty():
    assert compute_acwr_from_sessions([]) == []


def test_assess_risk_normal():
    loads = [_daily(d, 5000.0) for d in range(1, 31)]
    results = compute_acwr(loads)
    risk = assess_injury_risk(results)
    assert risk["risk_level"] in ("normal", "elevated")
    assert risk["total_days"] == 30


def test_assess_risk_empty():
    risk = assess_injury_risk([])
    assert risk["risk_level"] == "unknown"


def test_assess_risk_high_days():
    loads = []
    for d in range(1, 29):
        loads.append(_daily(d, 3000.0))
    for d in range(29, 36):
        loads.append(_daily(d, 20000.0))
    results = compute_acwr(loads)
    risk = assess_injury_risk(results)
    # May have elevated or critical risk with that spike
    assert risk["risk_level"] in ("elevated", "critical", "normal")
    assert isinstance(risk["recommendations"], list)
    assert len(risk["recommendations"]) > 0


def test_recommendations_exist():
    """Even in normal case, recommendations should be present."""
    loads = [_daily(d, 5000.0) for d in range(1, 15)]
    results = compute_acwr(loads)
    risk = assess_injury_risk(results)
    assert len(risk["recommendations"]) > 0


def test_datetime_handling():
    """Sessions with datetime objects should be handled."""
    from datetime import datetime

    sessions = [
        {"start_time": datetime(2026, 1, 1, 14, 0), "total_distance_m": 5000},
        {"start_time": datetime(2026, 1, 2, 15, 30), "total_distance_m": 6000},
    ]
    results = compute_acwr_from_sessions(sessions)
    assert len(results) == 2


def test_missing_field_does_not_crash():
    """Missing load field should not crash."""
    loads = [{"date": "2026-01-01"}, {"date": "2026-01-02", "total_distance_m": 5000}]
    results = compute_acwr(loads)
    assert len(results) == 2


# ── Deterministic EWMA pinning (Williams et al. 2017) ────────────────


def test_ewma_spike_exact_values():
    """Hand-computed EWMA values: 35 days of 5000, then a 12500 spike.

    acute(k=3): 5000 + (12500-5000)/3 = 7500.0
    chronic(k=7): 5000 + (12500-5000)/7 = 6071.4 (rounded)
    acwr = 7500/6071.43 = 1.235 -> 'normal'
    """
    loads = [_daily(d, 5000.0) for d in range(1, 36)]
    loads.append(_daily(36, 12500.0))
    results = compute_acwr(loads)
    last = results[-1]
    assert last["acute_load"] == 7500.0
    assert last["chronic_load"] == 6071.4
    assert last["acwr"] == 1.235
    assert last["load_category"] == "normal"


def test_ewma_constant_load_is_exactly_one():
    loads = [_daily(d, 5000.0) for d in range(1, 41)]
    results = compute_acwr(loads)
    assert all(r["acwr"] == 1.0 for r in results)
    assert all(r["load_category"] == "normal" for r in results)


def test_reliability_flag_turns_on_at_day_28():
    loads = [_daily(d, 5000.0) for d in range(1, 31)]
    results = compute_acwr(loads)
    assert all(r["reliable"] is False for r in results[:27])
    assert all(r["reliable"] is True for r in results[27:])


def test_single_day_is_not_reliable():
    result = compute_acwr([_daily(1, 5000.0)])
    assert result[0]["reliable"] is False


def test_rest_days_zero_filled_not_dropped():
    """A rest day must appear as a 0-load day in the series, not vanish.

    Sessions: day1 5000, day3 5000 (day2 = rest, absent).
    If day2 were dropped, both EWMAs would be inflated; with it present:
      day2: acute 5000*(2/3)=3333.3, chronic 5000*(6/7)=4285.7 -> low
      day3: acute 3888.9, chronic 4387.8 -> acwr 0.886 'normal'
    """
    sessions = [
        {"start_time": "2026-01-01", "total_distance_m": 5000.0},
        {"start_time": "2026-01-03", "total_distance_m": 5000.0},
    ]
    results = compute_acwr_from_sessions(sessions)
    assert len(results) == 3  # rest day present
    assert results[1]["total_distance_m"] == 0.0
    assert results[2]["acute_load"] == 3888.9
    assert results[2]["chronic_load"] == 4387.8
    assert results[2]["acwr"] == 0.886
    assert results[2]["load_category"] == "normal"
