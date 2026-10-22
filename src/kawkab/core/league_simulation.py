"""League Simulation — Monte Carlo league table simulation from xG rates."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class LeagueSimulationResult:
    n_simulations: int
    standings: list[dict]
    most_likely_table: list[dict]
    point_distributions: dict[str, list[int]]


def _poisson_goals(lambda_: float) -> int:
    if lambda_ <= 0:
        return 0
    return (
        random.poisson_variate(lambda_)
        if hasattr(random, "poisson_variate")
        else _poisson_knuth(lambda_)
    )


def _poisson_knuth(lambda_: float) -> int:
    if lambda_ <= 0:
        return 0
    L = math.exp(-lambda_)
    k = 0
    p = 1.0
    while p >= L and k < 30:
        k += 1
        p *= random.random()
    return k - 1 if k > 0 else 0


import math

from kawkab.core.perf_timing import timed


@timed()
def simulate_match(home_xg: float, away_xg: float) -> tuple[int, int]:
    home_goals = _poisson_knuth(home_xg)
    away_goals = _poisson_knuth(away_xg)
    return home_goals, away_goals


def _simulate_season(
    fixtures: list[dict],
    team_points: dict[str, int],
) -> dict[str, int]:
    points = dict(team_points)
    for f in fixtures:
        hg, ag = simulate_match(f.get("home_xg", 1.0), f.get("away_xg", 1.0))
        home = f["home_team"]
        away = f["away_team"]
        if hg > ag:
            points[home] = points.get(home, 0) + 3
        elif ag > hg:
            points[away] = points.get(away, 0) + 3
        else:
            points[home] = points.get(home, 0) + 1
            points[away] = points.get(away, 0) + 1
    return points


def _simulate_season_tracked(
    fixtures: list[dict],
    team_points: dict[str, int],
) -> dict[str, int]:
    """Simulate one season, returns dict of team -> points (same as _simulate_season)."""
    return _simulate_season(fixtures, team_points)


def simulate_league(
    fixtures: list[dict],
    current_table: list[dict],
    n_simulations: int = 10000,
    relegation_spots: int = 3,
    top4_spots: int = 4,
    use_xg: bool = True,
) -> LeagueSimulationResult:
    team_points_init: dict[str, int] = {}
    all_teams: list[str] = []
    for row in current_table:
        tid = row["team_id"]
        all_teams.append(tid)
        team_points_init[tid] = row.get("points", 0)

    for f in fixtures:
        if f["home_team"] not in all_teams:
            all_teams.append(f["home_team"])
        if f["away_team"] not in all_teams:
            all_teams.append(f["away_team"])
        if f["home_team"] not in team_points_init:
            team_points_init[f["home_team"]] = 0
        if f["away_team"] not in team_points_init:
            team_points_init[f["away_team"]] = 0

    all_point_histories: dict[str, list[int]] = defaultdict(list)
    title_count: dict[str, int] = defaultdict(int)
    top4_count: dict[str, int] = defaultdict(int)
    relegation_count: dict[str, int] = defaultdict(int)
    per_sim_positions: list[dict[str, int]] = []

    for _ in range(n_simulations):
        final_pts = _simulate_season(fixtures, team_points_init)
        sorted_teams = sorted(final_pts.items(), key=lambda x: (-x[1], all_teams.index(x[0])))
        sim_position: dict[str, int] = {}
        for pos, (tid, pts) in enumerate(sorted_teams):
            sim_position[tid] = pos + 1
            all_point_histories[tid].append(pts)
            if pos == 0:
                title_count[tid] += 1
            if pos < top4_spots:
                top4_count[tid] += 1
            if pos >= len(sorted_teams) - relegation_spots:
                relegation_count[tid] += 1
        per_sim_positions.append(sim_position)

    standings: list[dict] = []
    for tid in all_teams:
        pts_list = all_point_histories.get(tid, [0])
        avg_pts = round(sum(pts_list) / len(pts_list), 1)
        positions_for_team = [sim.get(tid, len(all_teams)) for sim in per_sim_positions]
        positions_for_team.sort()
        median_pos = (
            float(positions_for_team[len(positions_for_team) // 2]) if positions_for_team else 0.0
        )
        standings.append(
            {
                "team_id": tid,
                "avg_points": avg_pts,
                "title_pct": round(title_count.get(tid, 0) / n_simulations * 100, 1),
                "top4_pct": round(top4_count.get(tid, 0) / n_simulations * 100, 1),
                "relegation_pct": round(relegation_count.get(tid, 0) / n_simulations * 100, 1),
                "median_pos": median_pos,
            }
        )
    standings.sort(key=lambda x: (-x["avg_points"], all_teams.index(x["team_id"])))
    for pos, row in enumerate(standings):
        row["position"] = pos + 1

    most_likely: list[dict] = []
    point_distributions: dict[str, list[int]] = {}
    for tid in all_teams:
        pts_list = all_point_histories.get(tid, [0])
        bin_counts: dict[int, int] = defaultdict(int)
        for p in pts_list:
            bin_counts[p] += 1
        most_common_pts = max(bin_counts, key=bin_counts.get)  # type: ignore[arg-type,return-value]
        most_likely.append({"team_id": tid, "points": most_common_pts})
        point_distributions[tid] = pts_list

    most_likely.sort(key=lambda x: (-x["points"], all_teams.index(x["team_id"])))
    for pos, row in enumerate(most_likely):
        row["position"] = pos + 1

    return LeagueSimulationResult(
        n_simulations=n_simulations,
        standings=standings,
        most_likely_table=most_likely,
        point_distributions=point_distributions,
    )
