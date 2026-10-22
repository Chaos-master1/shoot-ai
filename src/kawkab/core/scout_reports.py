"""Scout Report Generation — human-readable scouting reports comparing
target players against squad and league benchmarks.

Builds on the player similarity engine to produce structured,
metric-by-metric reports for recruitment analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from kawkab.core.player_similarity import STAT_MEANS, STAT_STDS

SCOUT_METRICS: list[str] = [
    "xG_per_90",
    "xA_per_90",
    "shots_per_90",
    "pass_completion_pct",
    "progressive_passes_per_90",
    "touches_in_box_per_90",
    "pressures_per_90",
    "tackles_per_90",
    "interceptions_per_90",
    "aerial_win_pct",
    "dribbles_completed_pct",
    "key_passes_per_90",
    "passes_into_final_third_per_90",
]

SCOUT_METRIC_DISPLAY_NAMES: dict[str, str] = {
    "xG_per_90": "xG per 90",
    "xA_per_90": "xA per 90",
    "shots_per_90": "Shots per 90",
    "pass_completion_pct": "Pass Completion %",
    "progressive_passes_per_90": "Progressive Passes per 90",
    "touches_in_box_per_90": "Touches in Box per 90",
    "pressures_per_90": "Pressures per 90",
    "tackles_per_90": "Tackles per 90",
    "interceptions_per_90": "Interceptions per 90",
    "aerial_win_pct": "Aerials Won %",
    "dribbles_completed_pct": "Dribbles Completed %",
    "key_passes_per_90": "Key Passes per 90",
    "passes_into_final_third_per_90": "Passes into Final Third per 90",
}

SCOUT_METRIC_MEANS: dict[str, float] = {
    "touches_in_box_per_90": 3.0,
    "dribbles_completed_pct": 55.0,
    "passes_into_final_third_per_90": 8.0,
}

SCOUT_METRIC_STDS: dict[str, float] = {
    "touches_in_box_per_90": 2.0,
    "dribbles_completed_pct": 15.0,
    "passes_into_final_third_per_90": 4.0,
}


def _z_to_percentile(z: float) -> float:
    return round(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) * 100.0, 1)


def _get_metric_mean(metric: str, league_stats: dict[str, float] | None = None) -> float:
    if league_stats and metric in league_stats:
        return float(league_stats[metric])
    if metric in STAT_MEANS:
        return STAT_MEANS[metric]
    return SCOUT_METRIC_MEANS.get(metric, 0.0)


def _get_metric_std(metric: str) -> float:
    if metric in STAT_STDS:
        return STAT_STDS[metric]
    return SCOUT_METRIC_STDS.get(metric, 1.0)


@dataclass
class ScoutReport:
    target_player_id: str
    target_player_name: str
    report_date: str
    strengths: list[str]
    weaknesses: list[str]
    similar_players: list[dict[str, Any]]
    comparison_table: dict[str, dict[str, Any]]
    recommendation: str
    comparable_rating: str


def generate_scout_report(
    player_id: str,
    player_name: str,
    player_stats: dict[str, Any],
    squad_stats: dict[str, Any],
    league_stats: dict[str, Any],
    similar_players: list[dict[str, Any]],
) -> ScoutReport:
    strengths: list[str] = []
    weaknesses: list[str] = []
    comparison_table: dict[str, dict[str, Any]] = {}
    percentiles: list[float] = []

    for metric in SCOUT_METRICS:
        raw = player_stats.get(metric)
        if raw is None:
            continue

        player_value = float(raw)
        league_mean = _get_metric_mean(metric, league_stats)
        league_std = _get_metric_std(metric)
        squad_avg = squad_stats.get(metric)
        if squad_avg is not None:
            squad_avg = float(squad_avg)

        z = (player_value - league_mean) / league_std if league_std > 0 else 0.0
        percentile = _z_to_percentile(z)
        percentiles.append(percentile)

        is_strength = percentile > 75.0
        is_weakness = percentile < 40.0

        if is_strength:
            strengths.append(SCOUT_METRIC_DISPLAY_NAMES.get(metric, metric))
        if is_weakness:
            weaknesses.append(SCOUT_METRIC_DISPLAY_NAMES.get(metric, metric))

        comparison_table[metric] = {
            "player_value": player_value,
            "squad_average": squad_avg,
            "league_average": league_mean,
            "percentile": percentile,
            "is_strength": is_strength,
            "is_weakness": is_weakness,
        }

    avg_percentile = sum(percentiles) / len(percentiles) if percentiles else 0.0

    if avg_percentile > 70.0:
        recommendation = (
            "Highly recommended target — statistically suited "
            "for top-level football across key metrics."
        )
    elif avg_percentile > 50.0:
        recommendation = (
            "Recommended target with development potential — core metrics are above average."
        )
    elif avg_percentile > 30.0:
        recommendation = (
            "Squad player with specific tactical utility — may excel in a defined system or role."
        )
    else:
        recommendation = (
            "Development project with raw attributes — "
            "requires coaching to reach competitive level."
        )

    if avg_percentile > 80.0:
        comparable_rating = "Elite level"
    elif avg_percentile > 60.0:
        comparable_rating = "Premier League level"
    elif avg_percentile > 40.0:
        comparable_rating = "Championship level"
    elif avg_percentile > 20.0:
        comparable_rating = "League One level"
    else:
        comparable_rating = "Development level"

    return ScoutReport(
        target_player_id=str(player_id),
        target_player_name=player_name,
        report_date=date.today().isoformat(),
        strengths=strengths,
        weaknesses=weaknesses,
        similar_players=similar_players,
        comparison_table=comparison_table,
        recommendation=recommendation,
        comparable_rating=comparable_rating,
    )
