"""Composite match dominance index.

Combines possession %, xG differential, territory %, pressing
intensity, and pass completion into a single 0-100 dominance score.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
FINAL_THIRD_PCT = GAME.FINAL_THIRD_PCT


@dataclass
class DominanceReport:
    index: float = 50.0
    team: str = ""
    opponent: str = ""
    sub_scores: dict[str, float] = field(default_factory=dict)
    phases: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": round(self.index, 1),
            "team": self.team,
            "opponent": self.opponent,
            "sub_scores": {k: round(v, 1) for k, v in self.sub_scores.items()},
            "phases": {k: round(v, 1) for k, v in self.phases.items()},
        }


def compute_dominance_index(
    match_events: list[dict[str, Any]],
    team: str,
    opponent: str | None = None,
) -> DominanceReport:
    """Compute composite dominance index (0-100) for a team.

    Components (equal weight):
      - Possession share (0-100)
      - xG differential scaled (0-100)
      - Territory: % of events in attacking third (0-100)
      - Pressing intensity: defensive actions / opponent passes (0-100)
      - Pass completion rate (0-100)

    Args:
        match_events: List of event dicts with type, team, start_x, etc.
        team: Team to compute dominance for.
        opponent: Opponent team identifier.

    Returns:
        DominanceReport with index and sub-scores.
    """
    if opponent is None:
        opponent = "away" if team == "home" else "home"
    team_events = [e for e in match_events if e.get("team") == team]
    opp_events = [e for e in match_events if e.get("team") == opponent]

    total_events_team = len(team_events)
    total_events_opp = len(opp_events)

    # possession share (by event count)
    total_events = total_events_team + total_events_opp
    possession_share = (total_events_team / max(total_events, 1)) * 100.0

    # xG differential
    team_xg = sum(e.get("xG", 0.0) for e in team_events if e.get("type") in ("shot", "goal"))
    opp_xg = sum(e.get("xG", 0.0) for e in opp_events if e.get("type") in ("shot", "goal"))
    xg_diff = max(min(team_xg - opp_xg, 3.0), -3.0)
    xg_score = ((xg_diff + 3.0) / 6.0) * 100.0

    # territory
    final_third_x = PITCH_LENGTH * FINAL_THIRD_PCT

    def _in_final_third(e: dict[str, Any]) -> bool:
        # json_extract emits None (not a missing key) for rows whose
        # metadata lacks the coordinate, so .get(key, 0) returned None and
        # None > final_third_x raised TypeError through every dominance
        # computation on real storage rows.
        end_x = e.get("end_x")
        start_x = e.get("start_x")
        return (end_x is not None and end_x > final_third_x) or (
            start_x is not None and start_x > final_third_x
        )

    team_att_third = sum(
        1
        for e in team_events
        if e.get("type") in ("pass", "carry", "shot", "cross") and _in_final_third(e)
    )
    opp_att_third = sum(
        1
        for e in opp_events
        if e.get("type") in ("pass", "carry", "shot", "cross") and _in_final_third(e)
    )
    total_att = team_att_third + opp_att_third
    territory = (team_att_third / max(total_att, 1)) * 100.0

    # pressing intensity: defensive actions by team vs opponent passes
    def_actions = sum(
        1 for e in team_events if e.get("type") in ("tackle", "interception", "block", "clearance")
    )
    opp_passes = sum(1 for e in opp_events if e.get("type") == "pass")
    ratio = def_actions / max(opp_passes, 1)
    pressing = min(ratio * 50.0, 100.0)

    # pass completion
    team_passes = [e for e in team_events if e.get("type") == "pass"]
    completed = sum(1 for e in team_passes if e.get("completed", True))
    pass_comp = (completed / max(len(team_passes), 1)) * 100.0

    sub_scores = {
        "possession": round(possession_share, 1),
        "xg_diff": round(xg_score, 1),
        "territory": round(territory, 1),
        "pressing": round(pressing, 1),
        "pass_completion": round(pass_comp, 1),
    }

    index = sum(sub_scores.values()) / len(sub_scores)

    phases = _compute_phase_control(match_events, team)

    return DominanceReport(
        index=round(index, 1),
        team=team,
        opponent=opponent,
        sub_scores=sub_scores,
        phases=phases,
    )


def _compute_phase_control(
    events: list[dict[str, Any]],
    team: str,
) -> dict[str, float]:
    """Compute per-phase dominance breakdown."""
    phases: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        etype = ev.get("type", "")
        if etype == "pass":
            phases["open_play"].append(1.0 if ev.get("team") == team else 0.0)
        elif etype in ("tackle", "interception", "clearance", "block"):
            phases["defensive"].append(1.0 if ev.get("team") == team else 0.0)
        elif etype in ("corner_kick", "free_kick", "throw_in"):
            phases["set_piece"].append(1.0 if ev.get("team") == team else 0.0)

    result: dict[str, float] = {}
    for phase, vals in phases.items():
        if vals:
            result[phase] = round((sum(vals) / len(vals)) * 100.0, 1)
        else:
            result[phase] = 50.0

    return result
