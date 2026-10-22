"""Referee / Foul Pattern Analysis — decision patterns from foul and card events.

Produces a RefereeProfile with cards per game, home-team bias indicators,
inconsistency scores, and a match-level foul heatmap.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RefereeProfile:
    name: str
    matches_officiated: int
    cards_per_game: dict = field(default_factory=lambda: {"yellow": 0.0, "red": 0.0, "total": 0.0})
    fouls_per_game: float = 0.0
    home_team_advantage: float = 1.0
    most_common_foul_types: list[dict] = field(default_factory=list)
    penalty_rate: float = 0.0
    card_timing_distribution: dict = field(
        default_factory=lambda: {"first_half": 0, "second_half": 0}
    )
    inconsistency_score: float = 0.0
    trend: str = "stable"


@dataclass
class RefereeAnalysisReport:
    referee: RefereeProfile
    match_foul_heatmap: dict = field(default_factory=dict)
    foul_outcomes: dict[str, dict] = field(default_factory=dict)
    bias_indicators: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "referee": {
                "name": self.referee.name,
                "matches_officiated": self.referee.matches_officiated,
                "cards_per_game": self.referee.cards_per_game,
                "fouls_per_game": self.referee.fouls_per_game,
                "home_team_advantage": self.referee.home_team_advantage,
                "most_common_foul_types": self.referee.most_common_foul_types,
                "penalty_rate": self.referee.penalty_rate,
                "card_timing_distribution": self.referee.card_timing_distribution,
                "inconsistency_score": self.referee.inconsistency_score,
                "trend": self.referee.trend,
            },
            "match_foul_heatmap": self.match_foul_heatmap,
            "foul_outcomes": self.foul_outcomes,
            "bias_indicators": self.bias_indicators,
        }

    def summary_text(self) -> str:
        r = self.referee
        lines = [
            f"Referee: {r.name}",
            f"Matches officiated: {r.matches_officiated}",
            f"Cards per game: {r.cards_per_game['yellow']:.2f} yellow, {r.cards_per_game['red']:.2f} red",
            f"Fouls per game: {r.fouls_per_game:.2f}",
            f"Home team advantage ratio: {r.home_team_advantage:.2f}",
            f"Penalty rate: {r.penalty_rate:.2f} per 90",
        ]
        if self.bias_indicators:
            lines.append("Bias indicators:")
            for b in self.bias_indicators[:3]:
                lines.append(
                    f"  - {b['metric']}: home={b['home_value']}, away={b['away_value']} (diff={b['difference']})"
                )
        return "\n".join(lines)


def analyze_referee(
    referee_name: str,
    matches_data: list[dict],
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> RefereeAnalysisReport:
    if not matches_data:
        return RefereeAnalysisReport(
            referee=RefereeProfile(name=referee_name, matches_officiated=0),
            match_foul_heatmap={},
            foul_outcomes={},
            bias_indicators=[],
        )

    total_yellows = 0
    total_reds = 0
    total_fouls = 0
    total_minutes = 0
    total_penalties = 0
    foul_type_counts: dict[str, int] = defaultdict(int)
    home_fouls = 0
    away_fouls = 0
    home_cards = 0
    away_cards = 0
    card_half: dict[str, int] = {"first_half": 0, "second_half": 0}
    foul_heatmap: dict[str, int] = defaultdict(int)
    foul_outcome_data: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "card_count": 0, "card_types": defaultdict(int)}
    )
    per_match_cards: list[float] = []

    for match in matches_data:
        foul_events = match.get("foul_events", [])
        card_events = match.get("card_events", [])
        match_minutes = match.get("duration", 90)

        total_minutes += match_minutes
        n_fouls = len(foul_events)
        total_fouls += n_fouls
        n_yellows = sum(
            1
            for c in card_events
            if c.get("type", c.get("card_type", "")) in ("yellow", "yellow_card")
        )
        n_reds = sum(
            1
            for c in card_events
            if c.get("type", c.get("card_type", "")) in ("red", "red_card", "straight_red")
        )
        total_yellows += n_yellows
        total_reds += n_reds
        per_match_cards.append(n_yellows + n_reds)

        for foul in foul_events:
            foul_type = foul.get("type", foul.get("foul_type", "unknown"))
            foul_type_counts[foul_type] += 1

            zone_key = _zone_from_position(
                foul.get("x", pitch_width / 2),
                foul.get("y", pitch_length / 2),
                pitch_length,
                pitch_width,
            )
            foul_heatmap[zone_key] += 1

            team = foul.get("team", "")
            if team == match.get("home_team", ""):
                home_fouls += 1
            else:
                away_fouls += 1

            foul_outcome_data[foul_type]["count"] += 1

        for card in card_events:
            minute = int(card.get("minute", card.get("timestamp", 45)))
            if minute <= 45:
                card_half["first_half"] += 1
            else:
                card_half["second_half"] += 1

            team = card.get("team", "")
            if team == match.get("home_team", ""):
                home_cards += 1
            else:
                away_cards += 1

            card_type = card.get("type", card.get("card_type", "yellow"))
            associated_foul = card.get("foul_type", card.get("reason", "unknown"))
            if associated_foul in foul_outcome_data:
                foul_outcome_data[associated_foul]["card_count"] += 1
                foul_outcome_data[associated_foul]["card_types"][card_type] += 1

        penalties = sum(
            1 for ev in match.get("events", []) if ev.get("type") in ("penalty", "penalty_awarded")
        )
        total_penalties += penalties

    n_matches = len(matches_data)
    total_hours = total_minutes / 90.0

    home_team_advantage = home_fouls / max(away_fouls, 1)

    foul_type_list = sorted(
        [
            {"type": ft, "count": c, "pct": round(100 * c / max(total_fouls, 1), 1)}
            for ft, c in foul_type_counts.items()
        ],
        key=lambda x: int(str(x["count"])),
        reverse=True,
    )

    inconsistency_score = 0.0
    if len(per_match_cards) >= 3:
        mean_cards = sum(per_match_cards) / len(per_match_cards)
        variance = sum((c - mean_cards) ** 2 for c in per_match_cards) / len(per_match_cards)
        max_possible_var = (
            mean_cards * (max(per_match_cards) - mean_cards) if max(per_match_cards) > 0 else 1
        )
        inconsistency_score = min(1.0, variance / max(max_possible_var, 1))

    trend = "stable"
    if len(per_match_cards) >= 4:
        half = len(per_match_cards) // 2
        first_half_avg = sum(per_match_cards[:half]) / half
        second_half_avg = sum(per_match_cards[half:]) / (len(per_match_cards) - half)
        if second_half_avg > first_half_avg * 1.2:
            trend = "increasing"
        elif first_half_avg > second_half_avg * 1.2:
            trend = "decreasing"

    foul_outcomes_clean: dict[str, dict] = {}
    for ft, data in foul_outcome_data.items():
        foul_outcomes_clean[ft] = {
            "count": data["count"],
            "card_pct": round(100 * data["card_count"] / max(data["count"], 1), 1),
            "card_type_distribution": dict(data["card_types"]),
        }

    bias_indicators = [
        {
            "metric": "fouls",
            "home_value": home_fouls,
            "away_value": away_fouls,
            "difference": home_fouls - away_fouls,
        },
        {
            "metric": "cards",
            "home_value": home_cards,
            "away_value": away_cards,
            "difference": home_cards - away_cards,
        },
    ]

    profile = RefereeProfile(
        name=referee_name,
        matches_officiated=n_matches,
        cards_per_game={
            "yellow": round(total_yellows / max(total_hours, 1), 2),
            "red": round(total_reds / max(total_hours, 1), 2),
            "total": round((total_yellows + total_reds) / max(total_hours, 1), 2),
        },
        fouls_per_game=round(total_fouls / max(total_hours, 1), 2),
        home_team_advantage=round(home_team_advantage, 3),
        most_common_foul_types=foul_type_list[:10],
        penalty_rate=round(total_penalties / max(total_hours, 1), 2),
        card_timing_distribution=dict(card_half),
        inconsistency_score=round(inconsistency_score, 3),
        trend=trend,
    )

    return RefereeAnalysisReport(
        referee=profile,
        match_foul_heatmap=dict(foul_heatmap),
        foul_outcomes=foul_outcomes_clean,
        bias_indicators=bias_indicators,
    )


def _zone_from_position(x: float, y: float, pitch_length: float, pitch_width: float) -> str:
    x_norm = x / max(pitch_width, 1)
    y_norm = y / max(pitch_length, 1)
    if x_norm < 0.33:
        col = "left"
    elif x_norm < 0.67:
        col = "center"
    else:
        col = "right"
    if y_norm < 0.33:
        row = "defensive"
    elif y_norm < 0.67:
        row = "middle"
    else:
        row = "attacking"
    return f"{row}_{col}"
