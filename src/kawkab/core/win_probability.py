"""Win probability model — xG Monte Carlo simulation.

Uses Poisson-sampled xG simulation to compute minute-by-minute
win/draw/loss probabilities. Each goal event triggers a re-simulation
with remaining time and remaining xG.

This replaces the legacy Elo-based model with a proper simulation approach.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import PCG64, Generator, SeedSequence

# Module-level local RNG to avoid global seed contamination
_rng = Generator(PCG64(SeedSequence(42)))


@dataclass
class WinProbabilityPoint:
    minute: float = 0.0
    home_win: float = 0.333
    draw: float = 0.333
    away_win: float = 0.333
    home_score: int = 0
    away_score: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "minute": round(self.minute, 1),
            "home_win": round(self.home_win, 3),
            "draw": round(self.draw, 3),
            "away_win": round(self.away_win, 3),
            "home_score": self.home_score,
            "away_score": self.away_score,
        }


@dataclass
class WinProbabilityReport:
    timeline: list[dict[str, Any]] = field(default_factory=list)
    starting_home_win: float = 0.333
    starting_draw: float = 0.333
    starting_away_win: float = 0.333
    final_home_win: float = 0.0
    final_draw: float = 0.0
    final_away_win: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeline": self.timeline,
            "starting_home_win": round(self.starting_home_win, 3),
            "starting_draw": round(self.starting_draw, 3),
            "starting_away_win": round(self.starting_away_win, 3),
            "final_home_win": round(self.final_home_win, 3),
            "final_draw": round(self.final_draw, 3),
            "final_away_win": round(self.final_away_win, 3),
        }


@functools.lru_cache(maxsize=128)
def _simulate_remaining(
    home_xg_remaining: float,
    away_xg_remaining: float,
    home_score_current: int,
    away_score_current: int,
    n_sims: int = 10000,
) -> tuple[float, float, float]:
    """Monte Carlo simulation of remaining match time.

    Samples goals from Poisson(home_xg) and Poisson(away_xg),
    returns (home_win_pct, draw_pct, away_win_pct).

    Args:
        home_xg_remaining: Expected goals remaining for home team.
        away_xg_remaining: Expected goals remaining for away team.
        home_score_current: Current home score.
        away_score_current: Current away score.
        n_sims: Number of Monte Carlo simulations.

    Returns:
        (home_win, draw, away_win) probabilities.
    """
    home_goals = _rng.poisson(lam=home_xg_remaining, size=n_sims)
    away_goals = _rng.poisson(lam=away_xg_remaining, size=n_sims)
    home_goals = np.nan_to_num(home_goals, nan=0)
    away_goals = np.nan_to_num(away_goals, nan=0)

    home_final = home_score_current + home_goals
    away_final = away_score_current + away_goals

    home_wins = np.sum(home_final > away_final)
    draws = np.sum(home_final == away_final)
    away_wins = n_sims - home_wins - draws

    return (
        float(home_wins) / n_sims,
        float(draws) / n_sims,
        float(away_wins) / n_sims,
    )


def compute_win_probability(
    events: list[dict[str, Any]],
    home_rating: float = 1500.0,
    away_rating: float = 1500.0,
    match_duration_minutes: float = 90.0,
) -> WinProbabilityReport:
    """Compute minute-by-minute win probability using xG Monte Carlo.

    Starts with pre-match probabilities, then re-simulates after
    each goal event using remaining xG.

    If no xG data is available in events, falls back to Elo-based model.

    Args:
        events: Sorted events with type, team, is_goal, timestamp, xg.
        home_rating: Home team Elo rating (fallback).
        away_rating: Away team Elo rating (fallback).
        match_duration_minutes: Total match duration.

    Returns:
        WinProbabilityReport with timeline and summary.
    """
    # Extract shot events with xG
    shot_events = [e for e in events if e.get("type") == "shot"]
    home_xg_total = sum(e.get("xg", 0.0) for e in shot_events if e.get("team") == "home")
    away_xg_total = sum(e.get("xg", 0.0) for e in shot_events if e.get("team") == "away")

    use_xg = home_xg_total > 0 or away_xg_total > 0

    # Pre-match probabilities
    if use_xg and (home_xg_total > 0 or away_xg_total > 0):
        hw, dr, aw = _simulate_remaining(home_xg_total, away_xg_total, 0, 0, 10000)
    else:
        # Fallback: Elo-based
        diff = (home_rating + 50.0) - away_rating
        expected_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        expected_away = 1.0 / (1.0 + 10.0 ** (diff / 400.0))
        draw_margin = 0.24 * (1.0 - abs(expected_home - expected_away))
        dr = max(0.08, min(0.38, draw_margin))
        remaining = 1.0 - dr
        s = expected_home + expected_away
        hw = expected_home / s * remaining
        aw = expected_away / s * remaining

    report = WinProbabilityReport(
        starting_home_win=hw,
        starting_draw=dr,
        starting_away_win=aw,
    )

    home_score = 0
    away_score = 0
    timeline: list[WinProbabilityPoint] = []

    timeline.append(
        WinProbabilityPoint(
            minute=0,
            home_win=hw,
            draw=dr,
            away_win=aw,
            home_score=0,
            away_score=0,
        )
    )

    sorted_events = sorted(events, key=lambda e: e.get("timestamp", 0))

    if use_xg:
        # Compute xG distribution over time for remaining-xG tracking
        _ = match_duration_minutes * 60.0
        home_xg_used = 0.0
        away_xg_used = 0.0

        for ev in sorted_events:
            minute = ev.get("timestamp", 0) / 60.0

            if ev.get("type") == "shot":
                if ev.get("team") == "home":
                    home_xg_used += ev.get("xg", 0.0)
                else:
                    away_xg_used += ev.get("xg", 0.0)

            if ev.get("is_goal"):
                if ev.get("team") == "home":
                    home_score += 1
                else:
                    away_score += 1

                minutes_elapsed = max(minute, 1.0)
                remaining_minutes = max(0.0, match_duration_minutes - minute)
                home_xg_rate = home_xg_used / minutes_elapsed
                away_xg_rate = away_xg_used / minutes_elapsed
                hw, dr, aw = _simulate_remaining(
                    home_xg_rate * remaining_minutes,
                    away_xg_rate * remaining_minutes,
                    home_score,
                    away_score,
                )

                timeline.append(
                    WinProbabilityPoint(
                        minute=minute,
                        home_win=hw,
                        draw=dr,
                        away_win=aw,
                        home_score=home_score,
                        away_score=away_score,
                    )
                )
                _ = minute
    else:
        # Legacy: Elo-based updates on goals
        for ev in sorted_events:
            if ev.get("type") != "shot" or not ev.get("is_goal"):
                continue
            minute = ev.get("timestamp", 0) / 60.0
            if ev.get("team") == "home":
                home_score += 1
            else:
                away_score += 1

            time_left = max(1.0, match_duration_minutes - minute)
            goal_diff = home_score - away_score
            x = goal_diff * (match_duration_minutes / time_left) * 0.18
            hw = 1.0 / (1.0 + math.exp(-x))
            aw = 1.0 / (1.0 + math.exp(x))
            dr = 1.0 - (hw + aw)
            if dr < 0:
                dr = 0.0
                s2 = hw + aw
                hw /= s2
                aw /= s2

            timeline.append(
                WinProbabilityPoint(
                    minute=minute,
                    home_win=hw,
                    draw=dr,
                    away_win=aw,
                    home_score=home_score,
                    away_score=away_score,
                )
            )

    report.timeline = [p.to_dict() for p in timeline]
    if timeline:
        last = timeline[-1]
        report.final_home_win = last.home_win
        report.final_draw = last.draw
        report.final_away_win = last.away_win

    return report
