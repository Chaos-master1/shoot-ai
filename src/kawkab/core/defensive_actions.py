"""Defensive actions map and final third entry analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kawkab.core.coords import FINAL_THIRD_X, STANDARD_PITCH, PitchConfig


@dataclass
class DefensiveAction:
    timestamp: float = 0.0
    team: str = "home"
    action_type: str = "tackle"  # tackle, interception, pressure, foul, block
    x: float = 0.0
    y: float = 34.0
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "team": self.team,
            "action_type": self.action_type,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "success": self.success,
        }


@dataclass
class DefensiveHeatmap:
    team: str = "home"
    grid: list[list[float]] = field(default_factory=list)
    grid_rows: int = 30
    grid_cols: int = 46
    total_actions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "grid": self.grid,
            "total_actions": self.total_actions,
        }


def extract_defensive_actions(
    events: list[dict[str, Any]],
    team: str = "home",
) -> list[DefensiveAction]:
    """Extract defensive actions from event data.

    Detects tackles, interceptions, pressures, blocks, and fouls
    from event types and metadata.

    Args:
        events: List of event dicts.
        team: Team to filter for.

    Returns:
        List of DefensiveAction.
    """
    actions: list[DefensiveAction] = []
    action_map = {
        "tackle": "tackle",
        "interception": "interception",
        "pressure": "pressure",
        "foul": "foul",
        "block": "block",
    }

    for ev in events:
        if ev.get("team") != team:
            continue
        etype = ev.get("type", "")
        mapped = action_map.get(etype)
        if mapped is None:
            continue

        actions.append(
            DefensiveAction(
                timestamp=ev.get("timestamp", 0),
                team=team,
                action_type=mapped,
                x=ev.get("start_x", 52.5),
                y=ev.get("start_y", 34.0),
                success=ev.get("completed", False),
            )
        )

    return actions


def build_defensive_heatmap(
    actions: list[DefensiveAction],
    grid_rows: int = 30,
    grid_cols: int = 46,
    pitch: PitchConfig = STANDARD_PITCH,
) -> DefensiveHeatmap:
    """Build a density heatmap from defensive actions.

    Each action contributes to its nearest grid cell using a Gaussian
    kernel for smooth distribution.

    Args:
        actions: List of defensive actions.
        grid_rows: Grid resolution (rows).
        grid_cols: Grid resolution (cols).
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.

    Returns:
        DefensiveHeatmap with normalized density grid.
    """
    if not actions:
        return DefensiveHeatmap(
            team="",
            grid=[[0.0] * grid_cols for _ in range(grid_rows)],
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )

    import numpy as np

    gx = (np.arange(grid_cols) + 0.5) * pitch.length_m / grid_cols
    gy = (np.arange(grid_rows) + 0.5) * pitch.width_m / grid_rows

    team = actions[0].team
    positions = np.array([(a.x, a.y) for a in actions], dtype=np.float64)

    dx = gx[np.newaxis, :, np.newaxis] - positions[np.newaxis, np.newaxis, :, 0]
    dy = gy[:, np.newaxis, np.newaxis] - positions[np.newaxis, np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    density = np.exp(-dist_sq / (2.0 * 4.0**2))  # sigma = 4m
    grid_vals = np.sum(density, axis=2)

    max_val = float(np.max(grid_vals))
    if max_val > 0:
        grid_vals = grid_vals / max_val

    return DefensiveHeatmap(
        team=team,
        grid=grid_vals.tolist(),
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        total_actions=len(actions),
    )


@dataclass
class FinalThirdEntry:
    timestamp: float = 0.0
    team: str = "home"
    entry_type: str = "pass"  # pass, carry, cross, through_ball, dribble
    x: float = 0.0
    y: float = 34.0
    succeeded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "team": self.team,
            "entry_type": self.entry_type,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "succeeded": self.succeeded,
        }


@dataclass
class FinalThirdReport:
    home_entries: int = 0
    away_entries: int = 0
    home_by_type: dict[str, int] = field(default_factory=dict)
    away_by_type: dict[str, int] = field(default_factory=dict)
    home_success_pct: float = 0.0
    away_success_pct: float = 0.0
    entries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_entries": self.home_entries,
            "away_entries": self.away_entries,
            "home_by_type": self.home_by_type,
            "away_by_type": self.away_by_type,
            "home_success_pct": round(self.home_success_pct, 1),
            "away_success_pct": round(self.away_success_pct, 1),
            "total_entries": self.home_entries + self.away_entries,
        }


def analyze_final_third_entries(
    events: list[dict[str, Any]],
    pitch: PitchConfig = STANDARD_PITCH,
    attacking_direction: str = "right",
) -> FinalThirdReport:
    """Analyze how teams enter the final third of the pitch.

    Detects events that cross the 2/3 pitch line (~70m) and classifies
    the method of entry.

    Args:
        events: List of event dicts.
        pitch_length: Pitch length in meters.

    Returns:
        FinalThirdReport with entry statistics.
    """
    final_third_x = FINAL_THIRD_X
    home_entries = 0
    away_entries = 0
    home_by_type: dict[str, int] = {}
    away_by_type: dict[str, int] = {}
    home_success = 0
    away_success = 0
    entries: list[dict[str, Any]] = []

    for ev in events:
        team = ev.get("team", "home")
        etype = ev.get("type", "")
        end_x = ev.get("end_x", 0)
        start_x = ev.get("start_x", 0)

        # Ball entered final third
        if attacking_direction == "left":
            entered = end_x < pitch.length_m - final_third_x <= start_x
        else:
            entered = start_x < final_third_x <= end_x
        if entered:
            # Classify entry method
            entry_type = "pass"
            if etype == "carry":
                entry_type = "carry"
            elif etype == "pass":
                pass_type = ev.get("pass_type", "standard")
                if pass_type == "through_ball":
                    entry_type = "through_ball"
                elif pass_type == "cross":
                    entry_type = "cross"
                else:
                    entry_type = "pass"

            succeeded = ev.get("completed", False) and etype != "foul"

            if team == "home":
                home_entries += 1
                home_by_type[entry_type] = home_by_type.get(entry_type, 0) + 1
                if succeeded:
                    home_success += 1
            else:
                away_entries += 1
                away_by_type[entry_type] = away_by_type.get(entry_type, 0) + 1
                if succeeded:
                    away_success += 1

            entries.append(
                FinalThirdEntry(
                    timestamp=ev.get("timestamp", 0),
                    team=team,
                    entry_type=entry_type,
                    x=end_x,
                    y=ev.get("end_y", 34.0),
                    succeeded=succeeded,
                ).to_dict()
            )

    return FinalThirdReport(
        home_entries=home_entries,
        away_entries=away_entries,
        home_by_type=home_by_type,
        away_by_type=away_by_type,
        home_success_pct=(home_success / max(home_entries, 1)) * 100.0,
        away_success_pct=(away_success / max(away_entries, 1)) * 100.0,
        entries=entries,
    )


# ── PPDA (Passes Per Defensive Action) ────────────────────────────────────

_ACTION_TYPES = {"tackle", "interception", "pressure", "foul", "block"}


def compute_ppda(events: list[dict], team: str, attacking_direction: str = "right") -> dict:
    """Compute PPDA (Passes Per Defensive Action) for a team.

    PPDA measures pressing intensity: how many opponent passes are allowed
    per defensive action in the attacking 60% of the pitch.

    Args:
        events: List of event dicts.
        team: Team name to compute PPDA for ("home" or "away").
        attacking_direction: Direction of attack ("right" or "left").

    Returns:
        dict with ppda, defensive_actions, opponent_passes.
    """
    pitch_length = 105.0
    press_x_threshold = pitch_length * 0.4  # attacking 60%
    if attacking_direction == "left":
        press_x_threshold = pitch_length - press_x_threshold
    opponent = "away" if team == "home" else "home"

    def_actions = 0
    opp_passes = 0

    for ev in events:
        etype = ev.get("type", "")
        ev_team = ev.get("team", "")
        ev_end_x = ev.get("end_x", 0)

        # Count defensive actions by the team in opponent's half
        if ev_team == team and etype in _ACTION_TYPES and ev_end_x >= press_x_threshold:
            def_actions += 1

        # Count opponent passes in their own half (i.e., where the pressing team
        # is trying to disrupt build-up)
        if ev_team == opponent and etype == "pass" and ev_end_x < pitch_length - press_x_threshold:
            opp_passes += 1

    ppda = opp_passes / max(def_actions, 1)
    return {
        "ppda": round(ppda, 2),
        "defensive_actions": def_actions,
        "opponent_passes": opp_passes,
        "team": team,
    }


def compute_ppda_both_teams(events: list[dict], attacking_direction: str = "right") -> dict:
    """Compute PPDA for both teams.

    Args:
        events: List of event dicts.
        attacking_direction: Direction of attack ("right" or "left").

    Returns:
        dict with home_ppda, away_ppda.
    """
    home = compute_ppda(events, "home", attacking_direction)
    away = compute_ppda(events, "away", attacking_direction)
    return {
        "success": True,
        "home_ppda": home["ppda"],
        "away_ppda": away["ppda"],
        "home_defensive_actions": home["defensive_actions"],
        "away_defensive_actions": away["defensive_actions"],
        "home_opponent_passes": home["opponent_passes"],
        "away_opponent_passes": away["opponent_passes"],
    }
