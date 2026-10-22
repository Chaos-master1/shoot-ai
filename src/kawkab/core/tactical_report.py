"""Comprehensive Tactical Report — aggregates all tactical analysis modules.

Combines shape analysis, pressing classification, passing triangles,
transitions, formations, and build-up into one match-level tactical report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kawkab.core.passing_triangles import PassingTriangleAnalyzer
from kawkab.core.pressing_classifier import PressingSystemReport, classify_pressing_system
from kawkab.core.tactical_periods import TacticalPeriodReport, detect_tactical_periods
from kawkab.core.tactical_shape_analyzer import ShapeReport, TacticalShapeAnalyzer
from kawkab.core.transitions import TransitionReport


@dataclass
class TeamTacticalProfile:
    team: str = "home"
    primary_shape: str = "unknown"
    primary_formation: str = "unknown"
    pressing_system: str = "unknown"
    pressing_style: str = "unknown"
    triangle_count: int = 0
    triangles_per_90: float = 0.0
    transition_count: int = 0
    build_up_success_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "primary_shape": self.primary_shape,
            "primary_formation": self.primary_formation,
            "pressing_system": self.pressing_system,
            "pressing_style": self.pressing_style,
            "triangle_count": self.triangle_count,
            "triangles_per_90": round(self.triangles_per_90, 1),
            "transition_count": self.transition_count,
            "build_up_success_rate": round(self.build_up_success_rate, 1),
        }


@dataclass
class TacticalReport:
    match_id: int = 0
    home: TeamTacticalProfile = field(default_factory=lambda: TeamTacticalProfile(team="home"))
    away: TeamTacticalProfile = field(default_factory=lambda: TeamTacticalProfile(team="away"))
    tactical_phases: TacticalPeriodReport = field(default_factory=TacticalPeriodReport)
    home_shape_report: ShapeReport = field(default_factory=lambda: ShapeReport(team="home"))
    away_shape_report: ShapeReport = field(default_factory=lambda: ShapeReport(team="away"))
    home_pressing: PressingSystemReport = field(
        default_factory=lambda: PressingSystemReport(team="home")
    )
    away_pressing: PressingSystemReport = field(
        default_factory=lambda: PressingSystemReport(team="away")
    )
    home_transitions: TransitionReport = field(default_factory=TransitionReport)
    away_transitions: TransitionReport = field(default_factory=TransitionReport)
    key_tactical_observations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "home": self.home.to_dict(),
            "away": self.away.to_dict(),
            "tactical_phases": self.tactical_phases.to_dict() if self.tactical_phases else {},
            "home_shape_report": self.home_shape_report.to_dict() if self.home_shape_report else {},
            "away_shape_report": self.away_shape_report.to_dict() if self.away_shape_report else {},
            "home_pressing": self.home_pressing.to_dict() if self.home_pressing else {},
            "away_pressing": self.away_pressing.to_dict() if self.away_pressing else {},
            "key_tactical_observations": self.key_tactical_observations,
        }


def _generate_observations(report: TacticalReport) -> list[str]:
    obs: list[str] = []

    # Shape comparison
    if report.home.primary_shape != report.away.primary_shape:
        obs.append(
            f"Different attacking shapes: {report.home.team} uses "
            f"{report.home.primary_shape}, {report.away.team} uses "
            f"{report.away.primary_shape}"
        )

    # Pressing comparison
    if report.home_pressing.primary_block_type != report.away_pressing.primary_block_type:
        obs.append(
            f"Contrasting pressing: {report.home.team} uses "
            f"{report.home_pressing.primary_block_type}, "
            f"{report.away.team} uses {report.away_pressing.primary_block_type}"
        )

    if (
        report.home_pressing.pressing_style == "man_oriented"
        and report.away_pressing.pressing_style == "zonal"
    ):
        obs.append(
            f"{report.home.team} uses man-oriented pressing while {report.away.team} stays zonal"
        )

    # Triangle dominance
    if report.home.triangle_count > report.away.triangle_count * 1.5:
        obs.append(
            f"{report.home.team} dominates passing triangles ("
            f"{report.home.triangle_count} vs {report.away.triangle_count})"
        )

    # Phase distribution
    if report.tactical_phases:
        hp = getattr(report.tactical_phases, "press_pct", 0)
        lb = getattr(report.tactical_phases, "low_block_pct", 0)
        if hp > lb and hp > 20:
            obs.append("Match dominated by high pressing phases")
        elif lb > hp and lb > 20:
            obs.append("Match featured extended low-block defensive phases")

    # Diamond midfield
    if report.home_shape_report and report.home_shape_report.diamond_midfield_pct > 30:
        obs.append(
            f"{report.home.team} used diamond midfield "
            f"({report.home_shape_report.diamond_midfield_pct:.0f}% of match)"
        )

    # Transitions
    if report.home_transitions:
        hc = getattr(report.home_transitions, "counter_attacks", 0)
        _ = getattr(report.home_transitions, "counter_attacks", 0)
        if hc > 5:
            obs.append(f"{report.home.team} relied on counter-attacks ({hc} total)")

    # Shape changes
    if report.home_shape_report and report.home_shape_report.shape_changes > 3:
        obs.append(
            f"{report.home.team} changed shape {report.home_shape_report.shape_changes} times"
        )

    if not obs:
        obs.append("Balanced tactical profile — no extreme disparities detected")

    return obs


def generate_tactical_report(
    events: list[dict[str, Any]],
    match_id: int = 0,
    home_team: str = "Home",
    away_team: str = "Away",
) -> TacticalReport:
    """Generate a comprehensive tactical report from match events.

    Args:
        events: List of match event dicts.
        match_id: Match identifier.
        home_team: Home team name.
        away_team: Away team name.

    Returns:
        TacticalReport aggregating all tactical analyses.
    """
    if not events:
        return TacticalReport(match_id=match_id)

    # Tactical phases
    # Build frame-like data for phase detection from events
    ts_min = min(e.get("timestamp", 0) for e in events)
    ts_max = max(e.get("timestamp", 0) for e in events)
    duration = max(ts_max - ts_min, 1.0)

    frame_data = []
    t = ts_min
    step = max(5.0, duration / 100)
    while t < ts_max:
        window = [e for e in events if t <= e.get("timestamp", 0) < t + step]
        home_events = [e for e in window if e.get("team") == "home"]
        team_possession = len(home_events) > len(window) / 2 if window else True
        frame_data.append(
            {
                "timestamp": t,
                "possession": team_possession,
                "home_positions": [],
                "away_positions": [],
                "ball_pos": None,
            }
        )
        t += step

    phases = detect_tactical_periods(frame_data) if frame_data else TacticalPeriodReport()

    # Shape analysis
    shape_analyzer = TacticalShapeAnalyzer()
    home_shape = shape_analyzer.analyze_shapes(events, team="home")
    away_shape = shape_analyzer.analyze_shapes(events, team="away")

    # Pressing
    home_pressing = classify_pressing_system(events, team="home")
    away_pressing = classify_pressing_system(events, team="away")

    # Triangles
    pta = PassingTriangleAnalyzer()
    home_tri_events = [e for e in events if e.get("team") == "home"]
    away_tri_events = [e for e in events if e.get("team") == "away"]
    home_triangles = pta.detect_passing_triangles(home_tri_events)
    away_triangles = pta.detect_passing_triangles(away_tri_events)

    home_minutes = duration / 60.0 if duration > 0 else 1.0
    home_tri_per_90 = (len(home_triangles) / home_minutes * 90) if home_minutes > 0 else 0
    away_tri_per_90 = (len(away_triangles) / home_minutes * 90) if home_minutes > 0 else 0

    # Build team profiles
    home_profile = TeamTacticalProfile(
        team=home_team,
        primary_shape=home_shape.primary_attacking_shape if home_shape else "unknown",
        primary_formation=home_shape.primary_attacking_shape if home_shape else "unknown",
        pressing_system=home_pressing.primary_block_type if home_pressing else "unknown",
        pressing_style=home_pressing.pressing_style if home_pressing else "unknown",
        triangle_count=len(home_triangles),
        triangles_per_90=home_tri_per_90,
        transition_count=len(
            [e for e in events if e.get("type") == "counter" and e.get("team") == "home"]
        ),
    )

    away_profile = TeamTacticalProfile(
        team=away_team,
        primary_shape=away_shape.primary_attacking_shape if away_shape else "unknown",
        primary_formation=away_shape.primary_attacking_shape if away_shape else "unknown",
        pressing_system=away_pressing.primary_block_type if away_pressing else "unknown",
        pressing_style=away_pressing.pressing_style if away_pressing else "unknown",
        triangle_count=len(away_triangles),
        triangles_per_90=away_tri_per_90,
        transition_count=len(
            [e for e in events if e.get("type") == "counter" and e.get("team") == "away"]
        ),
    )

    report = TacticalReport(
        match_id=match_id,
        home=home_profile,
        away=away_profile,
        tactical_phases=phases,
        home_shape_report=home_shape,
        away_shape_report=away_shape,
        home_pressing=home_pressing,
        away_pressing=away_pressing,
    )

    report.key_tactical_observations = _generate_observations(report)
    return report
