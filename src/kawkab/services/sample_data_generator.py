"""Generate sample match data for demo purposes."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, cast


def generate_sample_match() -> dict:
    """Generate a realistic sample match with events, players, and teams."""
    random.seed(42)

    home_team = "FC Stars"
    away_team = "United Athletic"
    match_date = (datetime.now() - timedelta(days=2)).isoformat()

    home_players = [
        {"track_id": 1, "name": "Ahmed Hassan", "position": "GK", "jersey": 1},
        {"track_id": 2, "name": "Karim Mostafa", "position": "CB", "jersey": 4},
        {"track_id": 3, "name": "Youssef Ali", "position": "CB", "jersey": 5},
        {"track_id": 4, "name": "Omar Farouk", "position": "LB", "jersey": 3},
        {"track_id": 5, "name": "Amir Shady", "position": "RB", "jersey": 2},
        {"track_id": 6, "name": "Mohamed Nour", "position": "CM", "jersey": 8},
        {"track_id": 7, "name": "Hossam El-Din", "position": "CM", "jersey": 6},
        {"track_id": 8, "name": "Tarek Ziad", "position": "CAM", "jersey": 10},
        {"track_id": 9, "name": "Khaled Said", "position": "LW", "jersey": 7},
        {"track_id": 10, "name": "Mahmoud Fathy", "position": "RW", "jersey": 11},
        {"track_id": 11, "name": "Mostafa Ibrahim", "position": "ST", "jersey": 9},
        {"track_id": 12, "name": "Samir Lotfy", "position": "CM", "jersey": 14},
        {"track_id": 13, "name": "Abdallah Nasser", "position": "CB", "jersey": 15},
        {"track_id": 14, "name": "Hany Magdy", "position": "ST", "jersey": 18},
    ]

    away_players = [
        {"track_id": 101, "name": "John Smith", "position": "GK", "jersey": 1},
        {"track_id": 102, "name": "Mike Johnson", "position": "CB", "jersey": 4},
        {"track_id": 103, "name": "David Brown", "position": "CB", "jersey": 5},
        {"track_id": 104, "name": "Chris Lee", "position": "LB", "jersey": 3},
        {"track_id": 105, "name": "Tom Wilson", "position": "RB", "jersey": 2},
        {"track_id": 106, "name": "James Taylor", "position": "CM", "jersey": 8},
        {"track_id": 107, "name": "Ryan Clark", "position": "CDM", "jersey": 6},
        {"track_id": 108, "name": "Alex White", "position": "CAM", "jersey": 10},
        {"track_id": 109, "name": "Sam Green", "position": "LW", "jersey": 7},
        {"track_id": 110, "name": "Dan Black", "position": "RW", "jersey": 11},
        {"track_id": 111, "name": "Paul King", "position": "ST", "jersey": 9},
        {"track_id": 112, "name": "Luke Hall", "position": "CM", "jersey": 14},
        {"track_id": 113, "name": "Mark Stone", "position": "CB", "jersey": 15},
        {"track_id": 114, "name": "Ben Fox", "position": "ST", "jersey": 19},
    ]

    event_types = [
        "pass",
        "shot",
        "goal",
        "tackle",
        "foul",
        "corner",
        "save",
        "substitution",
        "offside",
        "yellow_card",
        "freekick",
        "cross",
        "dribble",
        "clearance",
        "interception",
    ]

    events = []
    t = 0.0
    event_id = 1
    home_score = 0
    away_score = 0

    # First half
    for _ in range(80):
        event_type = random.choice(event_types)
        team = random.choice(["home", "away"])
        player_pool = home_players if team == "home" else away_players
        player = random.choice(player_pool)
        x = random.uniform(0, 105)
        y = random.uniform(0, 68)

        metadata = cast(dict[str, Any], {"speed_kmh": round(random.uniform(5, 32), 1)})

        if event_type == "goal":
            if team == "home":
                home_score += 1
            else:
                away_score += 1
            metadata["goal_type"] = random.choice(["open_play", "set_piece", "penalty"])
            metadata["xG"] = round(random.uniform(0.05, 0.8), 3)

        if event_type == "shot":
            metadata["xG"] = round(random.uniform(0.02, 0.7), 3)
            metadata["on_target"] = random.choice([True, False])
            metadata["body_part"] = random.choice(["foot", "head"])

        events.append(
            {
                "id": event_id,
                "type": event_type,
                "team": team,
                "player_track_id": player["track_id"],
                "x": round(x, 1),
                "y": round(y, 1),
                "timestamp_s": round(t, 1),
                "period": 1,
                "metadata": metadata,
            }
        )
        event_id += 1
        t += random.uniform(8, 45)

    # Halftime
    t = max(t, 2700.0) + 900.0

    # Second half
    for _ in range(80):
        event_type = random.choice(event_types)
        team = random.choice(["home", "away"])
        player_pool = home_players if team == "home" else away_players
        player = random.choice(player_pool)
        x = random.uniform(0, 105)
        y = random.uniform(0, 68)

        metadata = cast(dict[str, Any], {"speed_kmh": round(random.uniform(5, 32), 1)})

        if event_type == "goal":
            if team == "home":
                home_score += 1
            else:
                away_score += 1
            metadata["goal_type"] = random.choice(["open_play", "set_piece", "counter"])
            metadata["xG"] = round(random.uniform(0.05, 0.8), 3)

        if event_type == "shot":
            metadata["xG"] = round(random.uniform(0.02, 0.7), 3)
            metadata["on_target"] = random.choice([True, False])

        events.append(
            {
                "id": event_id,
                "type": event_type,
                "team": team,
                "player_track_id": player["track_id"],
                "x": round(x, 1),
                "y": round(y, 1),
                "timestamp_s": round(t, 1),
                "period": 2,
                "metadata": metadata,
            }
        )
        event_id += 1
        t += random.uniform(8, 45)

    return {
        "match": {
            "name": f"{home_team} vs {away_team}",
            "home_team": home_team,
            "away_team": away_team,
            "date": match_date,
            "home_score": home_score,
            "away_score": away_score,
            "competition": "Premier League",
            "season": "2024/25",
        },
        "players": {"home": home_players, "away": away_players},
        "events": events,
        "total_events": len(events),
    }


def generate_and_save(path: str = "sample_match.json") -> str:
    import json as j

    data = generate_sample_match()
    with open(path, "w", encoding="utf-8") as f:
        j.dump(data, f, indent=2, ensure_ascii=False)
    return f"Sample match saved to {path} ({data['total_events']} events, {data['match']['home_score']}-{data['match']['away_score']})"
