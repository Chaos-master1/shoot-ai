"""Set-piece analytics service.

Analyzes corner kicks, free kicks, and throw-ins with their outcomes:
- Corner kick delivery (inswinging/outswinging/short)
- First contact area (near post / far post / central / edge of box)
- Outcome (goal, shot, clearance, retention)
- Set-piece routines
- Throw-in direction and distance

Used by professional analysts to:
- Identify set-piece patterns (favorite targets, routines)
- Score teams' set-piece effectiveness
- Detect weaknesses in defensive set-pieces
- Build training plans around specific set-piece scenarios
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from kawkab.core.game_constants import GAME
from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SetPieceEvent:
    """Single set-piece occurrence with outcome."""

    set_piece_type: str  # 'corner', 'free_kick', 'throw_in'
    minute: int
    second: int
    team: str
    delivery_x: float  # pitch x of delivery (0=home goal, 105=away goal)
    delivery_y: float  # pitch y
    delivery_style: str  # 'inswinging', 'outswinging', 'short', 'driven', 'lofted'
    delivery_height: str  # 'low', 'medium', 'high'
    first_contact_x: float | None = None
    first_contact_y: float | None = None
    outcome: str = (
        "unknown"  # 'goal', 'shot', 'clearance', 'retention', 'interception', 'loss', 'offside'
    )
    target_player_track_id: int | None = None
    set_piece_routine: str | None = (
        None  # 'near_post_flick', 'far_post', 'edge_of_box', 'short_corner', 'long_corner'
    )
    confidence: float = 0.7


@dataclass
class SetPieceStats:
    """Aggregated set-piece stats for one team."""

    team: str
    total_corners: int = 0
    total_free_kicks: int = 0
    total_throw_ins: int = 0
    corners_to_shots: int = 0
    corners_to_goals: int = 0
    short_corners: int = 0
    inswinging_corners: int = 0
    outswinging_corners: int = 0
    near_post_targets: int = 0
    far_post_targets: int = 0
    central_targets: int = 0
    edge_of_box_targets: int = 0
    delivery_height_low: int = 0
    delivery_height_medium: int = 0
    delivery_height_high: int = 0
    shots_per_corner: float = 0.0
    goals_per_corner: float = 0.0
    threat_per_set_piece: float = 0.0
    common_routines: list[tuple[str, int]] = field(default_factory=list)
    favorite_target_zone: str = ""


@dataclass
class SetPieceReport:
    """Full set-piece report for a match."""

    home_stats: SetPieceStats
    away_stats: SetPieceStats
    home_events: list[SetPieceEvent]
    away_events: list[SetPieceEvent]
    home_threat_total: float
    away_threat_total: float
    set_piece_differential: float
    notes: list[str]


class SetPieceService:
    """Analyze set-piece patterns and effectiveness.

    Threat score per set piece:
    - Goal: 1.0
    - Shot: 0.4
    - Header: 0.3
    - Clearance by defense: -0.1
    - Out of play: -0.1
    - Retention in midfield: 0.05
    - Short corner recycle: 0.1
    """

    PITCH_LENGTH = GAME.PITCH_LENGTH_M
    PITCH_WIDTH = GAME.PITCH_WIDTH_M
    NEAR_POST_X = 5.5
    FAR_POST_X = 16.5
    NEAR_POST_Y = 5.5
    FAR_POST_Y = 62.5

    THREAT_SCORES = {
        "goal": 1.0,
        "shot": 0.4,
        "header": 0.3,
        "save": 0.2,
        "clearance": -0.05,
        "out_of_play": -0.1,
        "interception": -0.15,
        "retention_midfield": 0.05,
        "short_recycle": 0.1,
        "offside": 0.0,
    }

    def __init__(self) -> None:
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def classify_delivery(
        self, delivery_x: float, delivery_y: float, target_x: float, target_y: float
    ) -> dict[str, str]:
        """Classify delivery style and target zone from positions.

        For corners (delivery near corner flag, target in box):
        - 'inswinging' if delivery curves toward goal
        - 'outswinging' if delivery curves away
        - 'short' if target is near delivery point
        """
        style = "lofted"
        height = "medium"
        dx = target_x - delivery_x
        dy = target_y - delivery_y
        distance = math.hypot(dx, dy)
        if distance < 12:
            style = "short"
        elif distance < 25:
            style = "lofted"
        else:
            style = "driven"
        if (
            distance >= 12
            and (delivery_y < 10 or delivery_y > 58)
            and delivery_x > 50
            and target_x > 90
        ):
            if delivery_y < 10 and target_y > 30 or delivery_y > 58 and target_y < 38:
                style = "inswinging"
            else:
                style = "outswinging"
        target_zone = self._classify_target_zone(target_x, target_y)
        return {"style": style, "height": height, "target_zone": target_zone}

    def _classify_target_zone(self, x: float, y: float) -> str:
        """Classify target zone in the attacking third.

        Zones (defending goal at x=0, attacking goal at x=105):
        - near_post: x > 95, y < 10
        - far_post: x > 95, y > 58
        - central: x in 88-100, y in 25-43
        - edge_of_box: x in 88-105, y in 10-25 or 43-58
        - short: x < 95
        """
        if x < 95:
            return "short"
        if y < 10:
            return "near_post"
        if y > 58:
            return "far_post"
        if 10 <= y <= 25 or 43 <= y < 58:
            return "edge_of_box"
        if 25 <= y <= 43 and x >= 95:
            return "central"
        return "edge_of_box"

    def compute_threat(self, outcome: str) -> float:
        """Compute threat score for a set-piece outcome."""
        return self.THREAT_SCORES.get(outcome, 0.0)

    def detect_routine(self, event: SetPieceEvent) -> str:
        """Detect a named set-piece routine from the delivery pattern."""
        if event.set_piece_type == "corner":
            if event.first_contact_x is None or event.first_contact_y is None:
                return "unknown"
            zone = self._classify_target_zone(event.first_contact_x, event.first_contact_y)
            if zone == "near_post":
                return "near_post_corner"
            if zone == "far_post":
                return "far_post_corner"
            if zone == "short":
                return "short_corner"
            if event.delivery_style == "short":
                return "short_corner_recycle"
        elif event.set_piece_type == "free_kick":
            if event.first_contact_x is None or event.first_contact_y is None:
                return "unknown"
            if event.first_contact_x > 95 and event.first_contact_y < 10:
                return "near_post_fk"
            if event.first_contact_x > 95 and event.first_contact_y > 58:
                return "far_post_fk"
        return "unknown"

    def analyze(
        self, events: list[SetPieceEvent], home_team: str = "home", away_team: str = "away"
    ) -> SetPieceReport:
        """Generate full set-piece report from a list of events."""
        home_events = [e for e in events if e.team == home_team]
        away_events = [e for e in events if e.team == away_team]
        home_stats = self._build_stats(home_team, home_events)
        away_stats = self._build_stats(away_team, away_events)
        home_threat = sum(self.compute_threat(e.outcome) for e in home_events)
        away_threat = sum(self.compute_threat(e.outcome) for e in away_events)
        notes = self._generate_notes(home_stats, away_stats, home_threat, away_threat)
        return SetPieceReport(
            home_stats=home_stats,
            away_stats=away_stats,
            home_events=home_events,
            away_events=away_events,
            home_threat_total=home_threat,
            away_threat_total=away_threat,
            set_piece_differential=home_threat - away_threat,
            notes=notes,
        )

    def _build_stats(self, team: str, events: list[SetPieceEvent]) -> SetPieceStats:
        stats = SetPieceStats(team=team)
        if not events:
            return stats
        total_threat = 0.0
        routine_counts: Counter[str] = Counter()
        target_zone_counts: Counter[str] = Counter()
        for e in events:
            threat = self.compute_threat(e.outcome)
            total_threat += threat
            routine = self.detect_routine(e)
            if routine != "unknown":
                routine_counts[routine] += 1
            if e.first_contact_x is not None and e.first_contact_y is not None:
                zone = self._classify_target_zone(e.first_contact_x, e.first_contact_y)
                target_zone_counts[zone] += 1
            if e.set_piece_type == "corner":
                stats.total_corners += 1
                if e.delivery_style == "inswinging":
                    stats.inswinging_corners += 1
                if e.delivery_style == "outswinging":
                    stats.outswinging_corners += 1
                if e.delivery_style == "short":
                    stats.short_corners += 1
                if e.outcome == "shot":
                    stats.corners_to_shots += 1
                if e.outcome == "goal":
                    stats.corners_to_goals += 1
                if e.delivery_height == "low":
                    stats.delivery_height_low += 1
                elif e.delivery_height == "high":
                    stats.delivery_height_high += 1
                else:
                    stats.delivery_height_medium += 1
            elif e.set_piece_type == "free_kick":
                stats.total_free_kicks += 1
            elif e.set_piece_type == "throw_in":
                stats.total_throw_ins += 1
        if stats.total_corners > 0:
            stats.shots_per_corner = round(stats.corners_to_shots / stats.total_corners, 3)
            stats.goals_per_corner = round(stats.corners_to_goals / stats.total_corners, 3)
            stats.threat_per_set_piece = round(total_threat / stats.total_corners, 3)
        stats.near_post_targets = target_zone_counts.get("near_post", 0)
        stats.far_post_targets = target_zone_counts.get("far_post", 0)
        stats.central_targets = target_zone_counts.get("central", 0)
        stats.edge_of_box_targets = target_zone_counts.get("edge_of_box", 0)
        stats.common_routines = routine_counts.most_common(3)
        if target_zone_counts:
            stats.favorite_target_zone = target_zone_counts.most_common(1)[0][0]
        return stats

    def _generate_notes(
        self, home: SetPieceStats, away: SetPieceStats, home_threat: float, away_threat: float
    ) -> list[str]:
        notes: list[str] = []
        if home.total_corners > 0:
            notes.append(
                f"Home took {home.total_corners} corners: {home.corners_to_shots} shots, "
                f"{home.corners_to_goals} goals ({home.shots_per_corner * 100:.1f}% shot rate)"
            )
        if away.total_corners > 0:
            notes.append(
                f"Away took {away.total_corners} corners: {away.corners_to_shots} shots, "
                f"{away.corners_to_goals} goals ({away.shots_per_corner * 100:.1f}% shot rate)"
            )
        if home_threat > away_threat + 1.0:
            notes.append("Home team significantly more dangerous from set pieces")
        elif away_threat > home_threat + 1.0:
            notes.append("Away team significantly more dangerous from set pieces")
        if home.short_corners > 0 and home.total_corners > 0:
            short_pct = home.short_corners / home.total_corners * 100
            if short_pct > 30:
                notes.append(
                    f"Home uses short corners {short_pct:.0f}% of the time (tactical setup play)"
                )
        if home.favorite_target_zone and home.favorite_target_zone != "short":
            notes.append(f"Home favorite target zone: {home.favorite_target_zone}")
        if away.favorite_target_zone and away.favorite_target_zone != "short":
            notes.append(f"Away favorite target zone: {away.favorite_target_zone}")
        if not notes:
            notes.append("No significant set-piece activity")
        return notes

    def suggest_routine(self, team: str, recent_corners: list[SetPieceEvent]) -> list[str]:
        """Suggest set-piece routines based on past patterns.

        Returns recommendations like 'more short corners' or 'try far post'.
        """
        if not recent_corners:
            return ["No data yet; recommend filming all set pieces for analysis"]
        recommendations: list[str] = []
        stats = self._build_stats(team, recent_corners)
        if stats.total_corners >= 5:
            if stats.shots_per_corner < 0.15:
                recommendations.append(
                    f"Low shot rate ({stats.shots_per_corner * 100:.0f}%) — vary delivery types"
                )
            if stats.short_corners == 0 and stats.total_corners >= 3:
                recommendations.append("Try short corners to break down man-marking defenses")
            if stats.far_post_targets < stats.near_post_targets and stats.total_corners >= 3:
                recommendations.append("Opponents expect near-post — try more far-post deliveries")
            if stats.goals_per_corner == 0 and stats.total_corners >= 5:
                recommendations.append("0 goals from corners — needs set-piece coaching review")
        return recommendations if recommendations else ["Set-piece patterns look healthy"]
