"""Game state analysis — how team behavior changes with scoreline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GameStateMetrics:
    possession_pct: float = 0.0
    pass_completion_pct: float = 0.0
    shots_per_10min: float = 0.0
    defensive_line_height_m: float = 0.0
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "possession_pct": round(self.possession_pct, 1),
            "pass_completion_pct": round(self.pass_completion_pct, 1),
            "shots_per_10min": round(self.shots_per_10min, 2),
            "defensive_line_height_m": round(self.defensive_line_height_m, 1),
            "duration_s": round(self.duration_s, 1),
        }


@dataclass
class GameStateReport:
    home_winning: GameStateMetrics = field(default_factory=GameStateMetrics)
    drawing: GameStateMetrics = field(default_factory=GameStateMetrics)
    home_losing: GameStateMetrics = field(default_factory=GameStateMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_winning": self.home_winning.to_dict(),
            "drawing": self.drawing.to_dict(),
            "home_losing": self.home_losing.to_dict(),
        }


def analyze_game_state(
    events: list[dict[str, Any]],
    frame_data: list[dict[str, Any]],
    home_team_name: str = "home",
) -> GameStateReport:
    """Analyze how team behavior changes with game state.

    Args:
        events: List of event dicts with timestamp, type, team, completed.
        frame_data: Frame-level tracking data with timestamp, possession,
                   home_positions, away_positions.
        home_team_name: Name or identifier for the home team.

    Returns:
        GameStateReport with per-state metrics.
    """
    # Build goal timeline
    goal_times: list[tuple[float, str]] = []  # (time, scoring_team)
    for ev in events:
        if ev.get("type") == "shot" and ev.get("is_goal"):
            goal_times.append((ev.get("timestamp", 0.0), ev.get("team", "unknown")))
    goal_times.sort(key=lambda x: x[0])

    def get_state_at(t: float, current_goals: dict[str, int]) -> str:
        """Return 'home_winning', 'drawing', or 'home_losing'."""
        home_g = current_goals.get(home_team_name, 0)
        away_g = current_goals.get("away" if home_team_name != "away" else "home", 0)
        if home_g > away_g:
            return "home_winning"
        elif home_g < away_g:
            return "home_losing"
        return "drawing"

    # Get events by state before/after each goal
    pass_completed: dict[str, int] = {"home_winning": 0, "drawing": 0, "home_losing": 0}
    pass_attempted: dict[str, int] = {"home_winning": 0, "drawing": 0, "home_losing": 0}
    shots: dict[str, int] = {"home_winning": 0, "drawing": 0, "home_losing": 0}
    possession_frames: dict[str, int] = {"home_winning": 0, "drawing": 0, "home_losing": 0}
    total_frames: dict[str, int] = {"home_winning": 0, "drawing": 0, "home_losing": 0}
    def_line_sum: dict[str, float] = {"home_winning": 0.0, "drawing": 0.0, "home_losing": 0.0}
    def_line_count: dict[str, int] = {"home_winning": 0, "drawing": 0, "home_losing": 0}

    # Walk through frames, tracking current score
    current_goals: dict[str, int] = defaultdict(int)
    goal_idx = 0

    for fdata in frame_data:
        ts = fdata.get("timestamp", 0.0)

        # Update score if we passed a goal time
        while goal_idx < len(goal_times) and goal_times[goal_idx][0] <= ts:
            current_goals[goal_times[goal_idx][1]] += 1
            goal_idx += 1

        state = get_state_at(ts, current_goals)
        total_frames[state] += 1

        possession = fdata.get("possession", True)
        if possession:
            possession_frames[state] += 1

        # Defensive line
        home_pos = fdata.get("home_positions", [])
        _ = fdata.get("away_positions", [])
        if home_pos:
            def_line_sum[state] += sum(p[0] for p in home_pos) / len(home_pos)
            def_line_count[state] += 1

    # Walk through events for pass/shots by state
    current_goals = defaultdict(int)
    goal_idx = 0

    for ev in events:
        ts = ev.get("timestamp", 0.0)
        while goal_idx < len(goal_times) and goal_times[goal_idx][0] <= ts:
            current_goals[goal_times[goal_idx][1]] += 1
            goal_idx += 1

        state = get_state_at(ts, current_goals)

        if ev.get("type") == "pass":
            pass_attempted[state] += 1
            if ev.get("completed", True):
                pass_completed[state] += 1
        elif ev.get("type") == "shot":
            shots[state] += 1

    def _metrics(state: str) -> GameStateMetrics:
        tf = total_frames.get(state, 0)
        if tf == 0:
            return GameStateMetrics()
        dur = tf * 0.5  # approximate (frame_interval)
        pf = possession_frames.get(state, 0)
        pa = pass_attempted.get(state, 0) or 1
        return GameStateMetrics(
            possession_pct=(pf / tf) * 100.0,
            pass_completion_pct=(pass_completed.get(state, 0) / pa) * 100.0,
            shots_per_10min=(shots.get(state, 0) / dur) * 600.0 if dur > 0 else 0.0,
            defensive_line_height_m=def_line_sum.get(state, 0)
            / max(def_line_count.get(state, 1), 1)
            if def_line_count.get(state, 0) > 0
            else 0.0,
            duration_s=dur,
        )

    return GameStateReport(
        home_winning=_metrics("home_winning"),
        drawing=_metrics("drawing"),
        home_losing=_metrics("home_losing"),
    )
