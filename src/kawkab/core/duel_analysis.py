"""Duel analysis — classify and analyze duel events.

Provides utilities for classifying duels as aerial vs ground based
on event metadata, and analyzing duel patterns across both teams.
"""

from __future__ import annotations

from collections import defaultdict


def classify_duel_type(event_metadata: dict) -> str:
    """Classify duel as 'aerial', 'ground', '50_50', or 'unknown'.

    Uses ball_height to distinguish aerial duels (height > 1.5m)
    from ground duels. Falls back to explicit duel_type metadata
    if available.

    Args:
        event_metadata: Event metadata dictionary.

    Returns:
        One of 'aerial', 'ground', '50_50', or 'unknown'.
    """
    height = event_metadata.get("ball_height", 0)
    if height > 1.5:
        return "aerial"
    duel_type = event_metadata.get("duel_type", "")
    if duel_type in ("aerial", "ground", "50_50"):
        return duel_type
    return "ground"


def analyze_duels(events: list[dict]) -> dict:
    """Full duel analysis across both teams.

    Args:
        events: List of event dictionaries containing duel events.

    Returns:
        Dict with team-level duel stats including totals by type,
        win rates, and per-player breakdowns.
    """
    team_stats: dict[str, dict] = {
        "home": {
            "total": 0,
            "aerial": 0,
            "ground": 0,
            "50_50": 0,
            "won": 0,
            "players": defaultdict(lambda: {"total": 0, "won": 0}),
        },
        "away": {
            "total": 0,
            "aerial": 0,
            "ground": 0,
            "50_50": 0,
            "won": 0,
            "players": defaultdict(lambda: {"total": 0, "won": 0}),
        },
    }

    for event in events:
        if event.get("type") != "duel":
            continue

        team = event.get("team", "home")
        if team not in team_stats:
            continue

        metadata = event.get("metadata", {}) or {}
        duel_type = classify_duel_type(metadata)
        won = event.get("won", False)

        team_stats[team]["total"] += 1
        team_stats[team][duel_type] += 1
        if won:
            team_stats[team]["won"] += 1

        player_id = event.get("track_id") or event.get("track_id_1")
        if player_id is not None:
            team_stats[team]["players"][str(player_id)]["total"] += 1
            if won:
                team_stats[team]["players"][str(player_id)]["won"] += 1

    result = {}
    for team, stats in team_stats.items():
        players_list = []
        for pid, pdata in stats["players"].items():
            players_list.append(
                {
                    "player_id": pid,
                    "total_duels": pdata["total"],
                    "duels_won": pdata["won"],
                    "win_rate": round(pdata["won"] / pdata["total"], 2)
                    if pdata["total"] > 0
                    else 0.0,
                }
            )
        players_list.sort(key=lambda x: x["total_duels"], reverse=True)
        result[team] = {
            "total_duels": stats["total"],
            "aerial_duels": stats["aerial"],
            "ground_duels": stats["ground"],
            "50_50": stats["50_50"],
            "duels_won": stats["won"],
            "win_rate": round(stats["won"] / stats["total"], 2) if stats["total"] > 0 else 0.0,
            "players": players_list,
        }

    return result
