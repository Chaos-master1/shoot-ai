"""Form Analysis + Team of the Week — form streaks, home/away splits, TOTW.

All numpy-only, no pandas/scipy/sklearn.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]
POSITION_SLOTS: dict[str, int] = {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}


def _result_from_match(match: dict[str, Any], team: str) -> str:
    home_goals = match.get("home_goals", 0)
    away_goals = match.get("away_goals", 0)
    if team == match.get("home_team", ""):
        if home_goals > away_goals:
            return "W"
        if home_goals < away_goals:
            return "L"
        return "D"
    if team == match.get("away_team", ""):
        if away_goals > home_goals:
            return "W"
        if away_goals < home_goals:
            return "L"
        return "D"
    return "D"


def _points_from_result(result: str) -> int:
    return 3 if result == "W" else 1 if result == "D" else 0


class FormAnalyzer:
    """Analyze team form, streaks, home/away splits, and select Team of the Week."""

    def compute_form_streak(
        self,
        matches: list[dict[str, Any]],
        team: str,
    ) -> dict[str, Any]:
        results = [_result_from_match(m, team) for m in matches]

        streak_type = "none"
        streak_length = 0
        if results:
            last_result = results[-1]
            streak_length = 1
            for r in reversed(results[:-1]):
                if r == last_result:
                    streak_length += 1
                else:
                    break
            streak_type = last_result

        last_5 = results[-5:] if len(results) >= 5 else results
        points_last_5 = sum(_points_from_result(r) for r in last_5)
        ppg = points_last_5 / max(len(last_5), 1)

        gd_trend = []
        for m in matches:
            team_is_home = m.get("home_team", "") == team
            gf = m.get("home_goals", 0) if team_is_home else m.get("away_goals", 0)
            ga = m.get("away_goals", 0) if team_is_home else m.get("home_goals", 0)
            gd_trend.append(gf - ga)

        return {
            "streak_type": streak_type,
            "streak_length": streak_length,
            "last_5_results": "".join(last_5),
            "points_last_5": points_last_5,
            "ppg_last_5": round(ppg, 2),
            "goal_difference_trend": gd_trend,
            "total_points": sum(_points_from_result(r) for r in results),
        }

    def compute_rolling_xg_form(
        self,
        matches: list[dict[str, Any]],
        team: str,
    ) -> dict[str, Any]:
        window = 5
        xg_for = []
        xg_against = []

        for m in matches:
            team_is_home = m.get("home_team", "") == team
            xg_for.append(m.get("home_xg", 0.0) if team_is_home else m.get("away_xg", 0.0))
            xg_against.append(m.get("away_xg", 0.0) if team_is_home else m.get("home_xg", 0.0))

        if len(xg_for) < window:
            avg_for = float(np.mean(xg_for)) if xg_for else 0.0
            avg_against = float(np.mean(xg_against)) if xg_against else 0.0
            return {
                "rolling_xg_for": round(avg_for, 3),
                "rolling_xg_against": round(avg_against, 3),
                "rolling_xg_diff": round(avg_for - avg_against, 3),
                "matches_used": len(xg_for),
            }

        recent_for = xg_for[-window:]
        recent_against = xg_against[-window:]
        avg_for = float(np.mean(recent_for))
        avg_against = float(np.mean(recent_against))

        return {
            "rolling_xg_for": round(avg_for, 3),
            "rolling_xg_against": round(avg_against, 3),
            "rolling_xg_diff": round(avg_for - avg_against, 3),
            "matches_used": window,
        }

    def compute_home_away_split(
        self,
        matches: list[dict[str, Any]],
        team: str,
    ) -> dict[str, Any]:
        home_matches = [m for m in matches if m.get("home_team", "") == team]
        away_matches = [m for m in matches if m.get("away_team", "") == team]

        def _sum_stats(matches_list: list[dict[str, Any]], is_home: bool) -> dict[str, Any]:
            n = len(matches_list)
            points = 0
            total_xg = 0.0
            goals = 0
            clean_sheets = 0
            losses = 0
            for m in matches_list:
                gf = m.get("home_goals", 0) if is_home else m.get("away_goals", 0)
                ga = m.get("away_goals", 0) if is_home else m.get("home_goals", 0)
                xg = m.get("home_xg", 0.0) if is_home else m.get("away_xg", 0.0)
                goals += gf
                total_xg += xg
                if gf > ga:
                    points += 3
                elif gf == ga:
                    points += 1
                else:
                    losses += 1
                if ga == 0:
                    clean_sheets += 1
            ppg = points / max(n, 1)
            return {
                "matches": n,
                "points": points,
                "ppg": round(ppg, 2),
                "total_xg": round(total_xg, 2),
                "goals": goals,
                "clean_sheets": clean_sheets,
                "losses": losses,
            }

        return {
            "home": _sum_stats(home_matches, True),
            "away": _sum_stats(away_matches, False),
        }

    def select_team_of_the_week(
        self,
        player_stats: list[dict[str, Any]],
        match_weights: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        by_position: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ps in player_stats:
            pos = ps.get("position", "MID")
            base_pos = (
                "GK"
                if pos == "GK"
                else "DEF"
                if pos in ("DEF", "CB", "LB", "RB")
                else "MID"
                if pos in ("MID", "CM", "DM", "AM", "LM", "RM")
                else "FWD"
            )
            by_position[base_pos].append(ps)

        totw: dict[str, Any] = {}
        for pos_group in POSITION_ORDER:
            slots = POSITION_SLOTS[pos_group]
            candidates = by_position.get(pos_group, [])
            scored = []
            for ps in candidates:
                rating = ps.get("rating", 0.0)
                if match_weights:
                    match_id = ps.get("match_id", "default")
                    weight = match_weights.get(str(match_id), 1.0)
                    rating *= weight
                scored.append((rating, ps))

            scored.sort(key=lambda x: x[0], reverse=True)
            selected = [s[1] for s in scored[:slots]]

            while len(selected) < slots:
                selected.append({"name": "N/A", "position": pos_group, "rating": 0.0})

            totw[pos_group] = [
                {
                    "name": s.get("name", "Unknown"),
                    "position": s.get("position", pos_group),
                    "rating": s.get("rating", 0.0),
                }
                for s in selected
            ]

        return totw

    def analyze_league_standings(
        self,
        matches: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        teams: dict[str, dict[str, Any]] = {}

        for m in matches:
            home = m.get("home_team", "")
            away = m.get("away_team", "")
            hg = m.get("home_goals", 0)
            ag = m.get("away_goals", 0)

            for team in (home, away):
                if team not in teams:
                    teams[team] = {
                        "played": 0,
                        "wins": 0,
                        "draws": 0,
                        "losses": 0,
                        "gf": 0,
                        "ga": 0,
                        "results": [],
                    }

            teams[home]["played"] += 1
            teams[away]["played"] += 1
            teams[home]["gf"] += hg
            teams[home]["ga"] += ag
            teams[away]["gf"] += ag
            teams[away]["ga"] += hg

            if hg > ag:
                teams[home]["wins"] += 1
                teams[away]["losses"] += 1
                teams[home]["results"].append("W")
                teams[away]["results"].append("L")
            elif hg < ag:
                teams[away]["wins"] += 1
                teams[home]["losses"] += 1
                teams[home]["results"].append("L")
                teams[away]["results"].append("W")
            else:
                teams[home]["draws"] += 1
                teams[away]["draws"] += 1
                teams[home]["results"].append("D")
                teams[away]["results"].append("D")

        standings = []
        for team, data in teams.items():
            pts = data["wins"] * 3 + data["draws"] * 1
            gd = data["gf"] - data["ga"]
            last_5 = data["results"][-5:]
            form_arrow = "".join(last_5) if last_5 else ""
            standings.append(
                {
                    "team": team,
                    "played": data["played"],
                    "wins": data["wins"],
                    "draws": data["draws"],
                    "losses": data["losses"],
                    "goals_for": data["gf"],
                    "goals_against": data["ga"],
                    "goal_difference": gd,
                    "points": pts,
                    "form": form_arrow,
                }
            )

        standings.sort(key=lambda x: (-x["points"], -x["goal_difference"], -x["goals_for"]))
        for i, s in enumerate(standings):
            s["position"] = i + 1

        return standings

    def detect_form_crisis(
        self,
        matches: list[dict[str, Any]],
        team: str,
    ) -> dict[str, Any]:
        if not matches:
            return {
                "is_crisis": False,
                "streak_type": "none",
                "streak_length": 0,
                "recommended_action": "No matches to analyze",
            }

        results = [_result_from_match(m, team) for m in matches]

        consecutive_losses = 0
        for r in reversed(results):
            if r == "L":
                consecutive_losses += 1
            else:
                break

        winless_streak = 0
        for r in reversed(results):
            if r != "W":
                winless_streak += 1
            else:
                break

        if consecutive_losses >= 3:
            return {
                "is_crisis": True,
                "streak_type": "consecutive_losses",
                "streak_length": consecutive_losses,
                "recommended_action": "Urgent: Address defensive organization and team morale. Consider tactical changes.",
            }

        if winless_streak >= 5:
            return {
                "is_crisis": True,
                "streak_type": "winless",
                "streak_length": winless_streak,
                "recommended_action": "Schedule a team meeting. Review attacking patterns and set-piece routines.",
            }

        return {
            "is_crisis": False,
            "streak_type": "none",
            "streak_length": 0,
            "recommended_action": "Team form is stable. Continue current approach.",
        }


# ── Form by competition type ────────────────────────────────────────────────


def form_by_competition(
    team_id: str,
    matches: list[dict],
    competition_types: tuple[str, ...] = ("league", "cup", "friendly", "continental"),
) -> dict[str, dict]:
    """Break down form by competition type.

    Returns per-competition stats and a recent_form summary.
    """
    _competition_map: dict[str, list[dict]] = {ct: [] for ct in competition_types}

    for m in matches:
        comp_type = str(m.get("competition_type", m.get("competition", ""))).lower()
        matched = False
        for ct in competition_types:
            if ct in comp_type or comp_type in (ct, ct.lower()):
                _competition_map[ct].append(m)
                matched = True
                break
        if not matched:
            _competition_map.setdefault("other", []).append(m)

    result: dict[str, dict] = {}
    recent: dict[str, list[str]] = {}

    for ct in competition_types:
        comp_matches = _competition_map.get(ct, [])
        if not comp_matches:
            result[ct] = {
                "played": 0,
                "won": 0,
                "drawn": 0,
                "lost": 0,
                "goals_for": 0,
                "goals_against": 0,
                "points_per_game": 0.0,
                "win_pct": 0.0,
            }
            recent[ct] = []
            continue

        played = len(comp_matches)
        won = sum(1 for m in comp_matches if _result_from_match(m, team_id) == "W")
        drawn = sum(1 for m in comp_matches if _result_from_match(m, team_id) == "D")
        lost = sum(1 for m in comp_matches if _result_from_match(m, team_id) == "L")
        goals_for = 0
        goals_against = 0
        results_seq: list[str] = []

        for m in comp_matches:
            team_is_home = m.get("home_team", "") == team_id
            gf = m.get("home_goals", 0) if team_is_home else m.get("away_goals", 0)
            ga = m.get("away_goals", 0) if team_is_home else m.get("home_goals", 0)
            goals_for += gf
            goals_against += ga
            results_seq.append(_result_from_match(m, team_id))

        points = won * 3 + drawn
        result[ct] = {
            "played": played,
            "won": won,
            "drawn": drawn,
            "lost": lost,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "points_per_game": round(points / max(played, 1), 3),
            "win_pct": round(100.0 * won / max(played, 1), 1),
        }
        recent[ct] = results_seq

    result["recent_form"] = recent
    return result


def form_by_opponent_strength(
    team_id: str,
    matches: list[dict],
    strength_tiers: list[tuple[str, float, float]] | None = None,
) -> dict[str, dict]:
    """Break down form by opponent strength tier.

    Tiers are defined by opponent points_per_game relative to league max.
    Default: top (0.66–inf), mid (0.33–0.66), bottom (0–0.33).
    """
    if strength_tiers is None:
        strength_tiers = [("top", 0.66, float("inf")), ("mid", 0.33, 0.66), ("bottom", 0.0, 0.33)]

    opponent_strengths: dict[str, float] = {}
    for m in matches:
        opp = (
            m.get("away_team", "") if m.get("home_team", "") == team_id else m.get("home_team", "")
        )
        opp_ppg = float(m.get("opponent_strength", m.get("opponent_ppg", 0)))
        if opp not in opponent_strengths:
            opponent_strengths[opp] = opp_ppg

    tiered: dict[str, list[dict]] = {t[0]: [] for t in strength_tiers}

    for m in matches:
        team_is_home = m.get("home_team", "") == team_id
        opponent = m.get("away_team", "") if team_is_home else m.get("home_team", "")
        opponent_ppg = opponent_strengths.get(opponent, 0.0)
        max_ppg = max(opponent_strengths.values()) if opponent_strengths else 1.0
        relative_strength = opponent_ppg / max(max_ppg, 1.0)

        for tier_name, lower, upper in strength_tiers:
            if lower <= relative_strength < upper:
                tiered.setdefault(tier_name, []).append(m)
                break

    result: dict[str, dict] = {}
    for tier_name, tier_matches in tiered.items():
        if not tier_matches:
            result[tier_name] = {
                "played": 0,
                "won": 0,
                "drawn": 0,
                "lost": 0,
                "goals_for": 0,
                "goals_against": 0,
                "points_per_game": 0.0,
            }
            continue

        played = len(tier_matches)
        won = sum(1 for m in tier_matches if _result_from_match(m, team_id) == "W")
        drawn = sum(1 for m in tier_matches if _result_from_match(m, team_id) == "D")
        lost = sum(1 for m in tier_matches if _result_from_match(m, team_id) == "L")
        goals_for = 0
        goals_against = 0
        for m in tier_matches:
            team_is_home = m.get("home_team", "") == team_id
            gf = m.get("home_goals", 0) if team_is_home else m.get("away_goals", 0)
            ga = m.get("away_goals", 0) if team_is_home else m.get("home_goals", 0)
            goals_for += gf
            goals_against += ga
        points = won * 3 + drawn
        result[tier_name] = {
            "played": played,
            "won": won,
            "drawn": drawn,
            "lost": lost,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "points_per_game": round(points / max(played, 1), 3),
        }

    return result
