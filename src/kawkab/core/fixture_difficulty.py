"""Fixture Difficulty — analyzes upcoming/remaining fixture difficulty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FixtureDifficultyReport:
    team_id: str
    fixtures: list[dict]
    avg_difficulty: float
    hardest_stretch: str
    easiest_stretch: str
    home_away_balance: float
    schedule_density: float


def _difficulty_color(score: float) -> str:
    if score < 40:
        return "green"
    if score <= 65:
        return "yellow"
    return "red"


def _describe_stretch(fixtures: list[dict], prefix: str) -> str:
    if not fixtures:
        return "N/A"
    parts = []
    for f in fixtures:
        opp = f.get("opponent", "?")
        venue = "H" if f.get("venue") == "home" else "A"
        d = f.get("difficulty_score", 0)
        parts.append(f"{opp}({venue},{d:.0f})")
    return f"{prefix}: " + ", ".join(parts)


def analyze_fixture_difficulty(
    team_id: str,
    fixtures: list[dict],
    opponent_strength: dict[str, float],
    home_advantage: float = 0.15,
) -> FixtureDifficultyReport:
    enriched: list[dict[str, Any]] = []
    for f in fixtures:
        opp_id = f.get("opponent_id", "")
        strength = opponent_strength.get(opp_id, 50.0)
        venue = f.get("venue", "home")
        if venue == "home":
            score = strength * (1.0 - home_advantage)
        else:
            score = strength * (1.0 + home_advantage * 0.5)
        score = max(1.0, min(100.0, score))
        color = _difficulty_color(score)
        enriched.append(
            {
                "opponent": opp_id,
                "venue": venue,
                "date": f.get("date", ""),
                "difficulty_score": round(score, 2),
                "color": color,
                "weight": round(score / 100.0, 3),
            }
        )

    avg_difficulty = (
        round(sum(e["difficulty_score"] for e in enriched) / len(enriched), 2) if enriched else 0.0
    )

    home_count = sum(1 for e in enriched if e["venue"] == "home")
    total = len(enriched)
    home_away_balance = round(home_count / total, 2) if total > 0 else 0.0

    schedule_density = 0.0
    dates = [f.get("date", "") for f in fixtures if f.get("date")]
    if len(dates) >= 2:
        try:
            from datetime import datetime

            parsed = [datetime.fromisoformat(d) for d in dates]
            gaps = [(parsed[i + 1] - parsed[i]).days for i in range(len(parsed) - 1)]
            schedule_density = round(sum(gaps) / len(gaps), 1) if gaps else 0.0
        except (ValueError, TypeError):
            schedule_density = 0.0

    hardest_stretch = ""
    easiest_stretch = ""
    if len(enriched) >= 3:
        best_sum = float("inf")
        worst_sum = -1.0
        best_idx = 0
        worst_idx = 0
        for i in range(len(enriched) - 2):
            total_score = sum(enriched[j]["difficulty_score"] for j in range(i, i + 3))
            if total_score > worst_sum:
                worst_sum = total_score
                worst_idx = i
            if total_score < best_sum:
                best_sum = total_score
                best_idx = i
        hardest_stretch = _describe_stretch(
            enriched[worst_idx : worst_idx + 3], "Hardest 3-match block"
        )
        easiest_stretch = _describe_stretch(
            enriched[best_idx : best_idx + 3], "Easiest 3-match block"
        )
    else:
        hardest_stretch = "Insufficient fixtures (need >=3) for stretch analysis"
        easiest_stretch = ""

    return FixtureDifficultyReport(
        team_id=team_id,
        fixtures=enriched,
        avg_difficulty=avg_difficulty,
        hardest_stretch=hardest_stretch,
        easiest_stretch=easiest_stretch,
        home_away_balance=home_away_balance,
        schedule_density=schedule_density,
    )
