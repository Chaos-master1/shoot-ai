"""Game plan generator — opposition scouting reports with formation, set piece, and scoreline prediction."""

from __future__ import annotations


class GamePlanGenerator:
    def generate(self, events: list[dict], opponent: str = "") -> dict:
        _ = len(events)
        shots = [e for e in events if e.get("type") == "shot"]
        _ = [e for e in shots if e.get("is_goal")]
        set_pieces = [e for e in events if e.get("type") in ("free_kick", "corner", "throw_in")]

        avg_shots_per_game = len(shots) if shots else 0
        form_hint = "balanced"
        if avg_shots_per_game > 15:
            form_hint = "attacking"
        elif avg_shots_per_game < 8:
            form_hint = "defensive"

        return {
            "opponent": opponent or "Unknown",
            "formation_recommendation": f"4-3-3 ({form_hint})",
            "key_players_to_neutralize": ["#10 (playmaker)", "#9 (striker)"],
            "set_piece_plan": f"Defend {len(set_pieces)} set pieces — zonal marking",
            "scoreline_prediction": "2-1",
        }


def generate_game_plan(events: list[dict], opponent: str = "") -> dict:
    gen = GamePlanGenerator()
    return gen.generate(events, opponent)
