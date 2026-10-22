"""Momentum Index — quantifies which team is dominating play.

Combines recent xG differential, territorial advantage, pressing
intensity, and key passes into a single momentum score per window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kawkab.core.game_constants import GAME


@dataclass
class MomentumPoint:
    minute: float = 0.0
    momentum: float = 0.0  # positive = home dominant, negative = away dominant
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_territory_pct: float = 50.0
    home_passes_final_third: int = 0
    away_passes_final_third: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "minute": round(self.minute, 1),
            "momentum": round(self.momentum, 2),
            "home_xg": round(self.home_xg, 3),
            "away_xg": round(self.away_xg, 3),
            "home_territory_pct": round(self.home_territory_pct, 1),
            "home_passes_final_third": self.home_passes_final_third,
            "away_passes_final_third": self.away_passes_final_third,
        }


@dataclass
class MomentumReport:
    home_momentum_pct: float = 0.0
    away_momentum_pct: float = 0.0
    neutral_pct: float = 0.0
    timeline: list[dict[str, Any]] = field(default_factory=list)
    home_max_run: float = 0.0
    away_max_run: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "home_momentum_pct": round(self.home_momentum_pct, 1),
            "away_momentum_pct": round(self.away_momentum_pct, 1),
            "neutral_pct": round(self.neutral_pct, 1),
            "home_max_run": round(self.home_max_run, 2),
            "away_max_run": round(self.away_max_run, 2),
            "timeline": self.timeline,
        }


def compute_momentum_index(
    events: list[dict[str, Any]],
    frame_data: list[dict[str, Any]] | None = None,
    window_minutes: float = GAME.MOMENTUM_WINDOW_MINUTES,
    match_duration_minutes: float = 90.0,
) -> MomentumReport:
    """Compute minute-by-minute momentum index for a match.

    Momentum is a composite of:
    - xG differential in the window (weight 0.4)
    - Territorial advantage (weight 0.3)
    - Final-third passes (weight 0.2)
    - Shots on target differential (weight 0.1)

    Positive momentum = home team dominant.
    Negative momentum = away team dominant.

    Args:
        events: List of event dicts with type, xg, team, timestamp.
        frame_data: Optional frame-level tracking data with possession info.
        window_minutes: Rolling window size in minutes.
        match_duration_minutes: Total match duration.

    Returns:
        MomentumReport with timeline and aggregate stats.
    """
    if not events:
        return MomentumReport()

    window_seconds = window_minutes * 60.0
    points: list[MomentumPoint] = []
    num_windows = max(10, int(match_duration_minutes / 1.0))

    for i in range(num_windows + 1):
        window_end = (i / num_windows) * match_duration_minutes * 60.0
        window_start = max(0, window_end - window_seconds)
        minute = window_end / 60.0

        # xG in window
        home_xg = 0.0
        away_xg = 0.0
        home_shots_on_target = 0
        away_shots_on_target = 0
        home_final_third_passes = 0
        away_final_third_passes = 0

        for ev in events:
            ts = ev.get("timestamp", 0)
            if ts < window_start or ts > window_end:
                continue
            team = ev.get("team", "home")
            if ev.get("type") == "shot":
                xg = ev.get("xg", 0)
                if team == "home":
                    home_xg += xg
                    if ev.get("on_target"):
                        home_shots_on_target += 1
                else:
                    away_xg += xg
                    if ev.get("on_target"):
                        away_shots_on_target += 1
            elif ev.get("type") == "pass":
                end_x = ev.get("end_x", 0)
                if end_x > 70.0:  # final third
                    if team == "home":
                        home_final_third_passes += 1
                    else:
                        away_final_third_passes += 1

        xg_diff = home_xg - away_xg
        shot_diff = home_shots_on_target - away_shots_on_target
        pass_diff = home_final_third_passes - away_final_third_passes

        # Territory from frame data
        home_territory = 50.0
        if frame_data:
            frames_in_window = [
                f for f in frame_data if window_start <= f.get("timestamp", 0) <= window_end
            ]
            if frames_in_window:
                home_pos = 0
                for f in frames_in_window:
                    ball_pos = f.get("ball_pos")
                    if ball_pos is not None:
                        home_pos += 1 if ball_pos[0] > 52.5 else 0
                home_territory = (home_pos / len(frames_in_window)) * 100.0

        territory_diff = home_territory - 50.0

        # Composite momentum (scale to [-1, 1] roughly)
        xg_component = max(-1.0, min(1.0, xg_diff * 1.0)) * 0.4
        territory_component = (territory_diff / 50.0) * 0.3
        passes_component = max(-1.0, min(1.0, pass_diff / 5.0)) * 0.2
        shots_component = max(-1.0, min(1.0, shot_diff / 3.0)) * 0.1

        momentum = xg_component + territory_component + passes_component + shots_component
        momentum = max(-1.0, min(1.0, momentum))

        points.append(
            MomentumPoint(
                minute=minute,
                momentum=momentum,
                home_xg=home_xg,
                away_xg=away_xg,
                home_territory_pct=home_territory,
                home_passes_final_third=home_final_third_passes,
                away_passes_final_third=away_final_third_passes,
            )
        )

    # Aggregate
    _ = sum(p.momentum for p in points)
    _ = len(points) or 1
    home_mom_pct = 0.0
    away_mom_pct = 0.0
    neutral_count = 0
    for p in points:
        if p.momentum > 0.05:
            home_mom_pct += 1
        elif p.momentum < -0.05:
            away_mom_pct += 1
        else:
            neutral_count += 1
    n_pts = len(points) or 1
    home_mom_pct = (home_mom_pct / n_pts) * 100.0
    away_mom_pct = (away_mom_pct / n_pts) * 100.0
    neutral_pct = (neutral_count / n_pts) * 100.0

    # Max runs
    home_max_run = 0.0
    away_max_run = 0.0
    current_run = 0.0
    run_type = 0  # 1=home, -1=away, 0=neutral
    for p in points:
        if p.momentum > 0.05:
            if run_type == 1:
                current_run += 1.0
            else:
                current_run = 1.0
                run_type = 1
            home_max_run = max(home_max_run, current_run)
        elif p.momentum < -0.05:
            if run_type == -1:
                current_run += 1.0
            else:
                current_run = 1.0
                run_type = -1
            away_max_run = max(away_max_run, current_run)

    return MomentumReport(
        home_momentum_pct=home_mom_pct,
        away_momentum_pct=away_mom_pct,
        neutral_pct=neutral_pct,
        timeline=[p.to_dict() for p in points],
        home_max_run=home_max_run,
        away_max_run=away_max_run,
    )
