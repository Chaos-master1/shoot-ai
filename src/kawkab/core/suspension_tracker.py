"""Suspension Tracker — tracks yellow card accumulation and suspension risk.

Configurable rules for yellow card thresholds, red card penalties,
and yellow card slate clearance after N matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlayerDiscipline:
    player_id: str
    player_name: str
    team: str
    competition: str
    yellow_cards: int = 0
    red_cards: int = 0
    suspension_threshold: int = 5
    current_yellow_count: int = 0
    matches_until_clear: int = 0
    is_suspended: bool = False
    suspension_details: str = ""
    fair_play_score: float = 100.0


@dataclass
class SuspensionReport:
    team: str
    competition: str
    players: list[PlayerDiscipline] = field(default_factory=list)
    total_yellows: int = 0
    total_reds: int = 0
    pending_suspensions: list[PlayerDiscipline] = field(default_factory=list)
    upcoming_risk: list[dict] = field(default_factory=list)
    fair_play_ranking: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "competition": self.competition,
            "players": [
                {
                    "player_id": p.player_id,
                    "player_name": p.player_name,
                    "team": p.team,
                    "yellow_cards": p.yellow_cards,
                    "red_cards": p.red_cards,
                    "current_yellow_count": p.current_yellow_count,
                    "matches_until_clear": p.matches_until_clear,
                    "is_suspended": p.is_suspended,
                    "suspension_details": p.suspension_details,
                    "fair_play_score": p.fair_play_score,
                }
                for p in self.players
            ],
            "total_yellows": self.total_yellows,
            "total_reds": self.total_reds,
            "pending_suspensions": [
                {
                    "player_id": p.player_id,
                    "player_name": p.player_name,
                    "suspension_details": p.suspension_details,
                }
                for p in self.pending_suspensions
            ],
            "upcoming_risk": self.upcoming_risk,
            "fair_play_ranking": self.fair_play_ranking,
        }

    def summary_text(self) -> str:
        lines = [
            f"Suspension Report: {self.team} ({self.competition})",
            f"Total yellows: {self.total_yellows}  |  Total reds: {self.total_reds}",
        ]
        if self.pending_suspensions:
            lines.append("Pending suspensions:")
            for p in self.pending_suspensions:
                lines.append(f"  - {p.player_name}: {p.suspension_details}")
        if self.upcoming_risk:
            lines.append("Upcoming risk:")
            for r in self.upcoming_risk:
                lines.append(f"  - {r['player']}: {r['cards_needed_for_suspension']} card(s) away")
        return "\n".join(lines)


def _default_suspension_rules() -> dict:
    return {
        "yellow_thresholds": [5, 10, 15],
        "yellow_suspension_matches": [1, 2, 3],
        "straight_red_matches": 3,
        "second_yellow_matches": 1,
        "clear_after_matches": 19,
        "fair_play_yellow_penalty": 5,
        "fair_play_red_penalty": 15,
    }


def analyze_suspensions(
    team_events: list[dict],
    competition: str,
    team_id: str,
    suspension_rules: dict | None = None,
) -> SuspensionReport:
    rules = suspension_rules or _default_suspension_rules()
    yellow_thresholds = rules.get("yellow_thresholds", [5, 10, 15])
    yellow_suspension_matches = rules.get("yellow_suspension_matches", [1, 2, 3])
    straight_red_matches = rules.get("straight_red_matches", 3)
    second_yellow_matches = rules.get("second_yellow_matches", 1)
    clear_after_matches = rules.get("clear_after_matches", 19)

    if not team_events:
        return SuspensionReport(team=team_id, competition=competition)

    player_data: dict[str, dict] = {}
    team_name = ""

    for ev in team_events:
        player_id = ev.get("player_id", ev.get("player", ""))
        player_name = ev.get("player_name", ev.get("player", player_id))
        team = ev.get("team", team_id)
        if not team_name and team:
            team_name = team

        if player_id not in player_data:
            player_data[player_id] = {
                "player_id": player_id,
                "player_name": player_name,
                "team": team,
                "yellows": 0,
                "reds": 0,
                "match_numbers": [],
                "straight_reds": 0,
                "second_yellows": 0,
            }

        ev_type = ev.get("type", "")
        if ev_type in ("yellow_card", "yellow"):
            player_data[player_id]["yellows"] += 1
            match_num = ev.get("match_number", ev.get("match_id", 0))
            player_data[player_id]["match_numbers"].append(match_num)
        elif ev_type in ("red_card", "red"):
            player_data[player_id]["reds"] += 1
            card_reason = ev.get("card_type", ev.get("reason", ""))
            if card_reason in ("second_yellow", "second_booking"):
                player_data[player_id]["second_yellows"] += 1
            else:
                player_data[player_id]["straight_reds"] += 1
            match_num = ev.get("match_number", ev.get("match_id", 0))
            player_data[player_id]["match_numbers"].append(match_num)

    if not team_name:
        team_name = team_id

    players: list[PlayerDiscipline] = []
    total_yellows = sum(pd["yellows"] for pd in player_data.values())
    total_reds = sum(pd["reds"] for pd in player_data.values())

    for pd in player_data.values():
        yellows = pd["yellows"]
        reds = pd["reds"]
        n_matches = len(set(pd["match_numbers"]))
        matches_until_clear = max(0, clear_after_matches - n_matches)

        current_yellows = yellows
        if matches_until_clear <= 0:
            current_yellows = 0

        is_suspended = False
        suspension_details = ""

        if reds > 0:
            is_suspended = True
            parts = []
            if pd["straight_reds"] > 0:
                parts.append(
                    f"{pd['straight_reds']} straight red(s): {straight_red_matches} match(es) each"
                )
            if pd["second_yellows"] > 0:
                parts.append(
                    f"{pd['second_yellows']} second yellow(s): {second_yellow_matches} match(es) each"
                )
            suspension_details = "; ".join(parts)

        for i, threshold in enumerate(yellow_thresholds):
            if current_yellows >= threshold and n_matches > 0:
                is_suspended = True
                ban = (
                    yellow_suspension_matches[i]
                    if i < len(yellow_suspension_matches)
                    else yellow_suspension_matches[-1]
                )
                suspension_details = f"{ban} match suspension for {threshold} yellow cards"
                if reds > 0:
                    suspension_details += " (also serving red card suspension)"
                break

        fair_play_score = 100.0
        yellow_penalty = rules.get("fair_play_yellow_penalty", 5)
        red_penalty = rules.get("fair_play_red_penalty", 15)
        fair_play_score -= yellows * yellow_penalty
        fair_play_score -= reds * red_penalty
        fair_play_score = max(0.0, min(100.0, fair_play_score))

        player = PlayerDiscipline(
            player_id=pd["player_id"],
            player_name=pd["player_name"],
            team=pd["team"],
            competition=competition,
            yellow_cards=yellows,
            red_cards=reds,
            suspension_threshold=yellow_thresholds[0] if yellow_thresholds else 5,
            current_yellow_count=current_yellows,
            matches_until_clear=matches_until_clear,
            is_suspended=is_suspended,
            suspension_details=suspension_details,
            fair_play_score=round(fair_play_score, 1),
        )
        players.append(player)

    pending_suspensions = [p for p in players if p.is_suspended]

    upcoming_risk: list[dict] = []
    for p in players:
        if p.is_suspended:
            continue
        for threshold in yellow_thresholds:
            cards_needed = threshold - p.current_yellow_count
            if 1 <= cards_needed <= 2:
                upcoming_risk.append(
                    {
                        "player": p.player_name,
                        "player_id": p.player_id,
                        "current_yellows": p.current_yellow_count,
                        "threshold": threshold,
                        "cards_needed_for_suspension": cards_needed,
                    }
                )
                break

    sorted_players = sorted(players, key=lambda p: p.fair_play_score)
    fair_play_ranking = None
    for i, p in enumerate(sorted_players):
        if p.player_id == team_id or p.team == team_name:
            fair_play_ranking = i + 1
            break

    return SuspensionReport(
        team=team_name,
        competition=competition,
        players=sorted(players, key=lambda p: (-p.red_cards, -p.yellow_cards)),
        total_yellows=total_yellows,
        total_reds=total_reds,
        pending_suspensions=pending_suspensions,
        upcoming_risk=upcoming_risk,
        fair_play_ranking=fair_play_ranking,
    )
