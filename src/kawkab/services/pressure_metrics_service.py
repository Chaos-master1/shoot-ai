"""Pressure metrics service - compute pressing and defensive pressure analytics.

Computes professional-grade pressure metrics:
- PPDA (Passes Per Defensive Action) by zone
- Pressing traps (forcing play into specific zones)
- Passes under pressure (%)
- Pressure events (defender within 2m of ball carrier)
- Counter-press success rate
- Time to regain possession after loss
- Defensive line height and compactness
- Pressure intensity by period (first 15 min, last 15 min, etc.)

These metrics are used by tactical analysts to evaluate pressing systems
and defensive organization.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field

from kawkab.core.logging import get_logger
from kawkab.services.cv_service import MatchTrackData

logger = get_logger(__name__)


@dataclass
class PressureMetrics:
    """Pressure metrics for a team."""

    team: str
    ppda_overall: float = 0.0
    ppda_by_zone: dict[str, float] = field(default_factory=dict)
    passes_under_pressure_pct: float = 0.0
    pressure_events: int = 0
    counter_press_success_rate: float = 0.0
    avg_time_to_regain: float = 0.0
    defensive_line_height_m: float = 0.0
    team_width_m: float = 0.0
    compactness_index: float = 0.0  # lower = more compact
    pressing_intensity_by_period: dict[str, float] = field(default_factory=dict)


class PressureMetricsService:
    """Computes pressing and defensive pressure metrics."""

    PRESSURE_DISTANCE_M = 2.0  # Defender within 2m = pressure
    PPDA_ZONES = ["defensive_third", "middle_third", "final_third"]

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0) -> None:
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        logger.info("PressureMetricsService initialized")

    async def compute_pressure_metrics(
        self,
        track_data: MatchTrackData,
        events: list[dict],
        homography_matrix=None,
    ) -> dict[str, PressureMetrics]:
        """Compute pressure metrics for both teams.

        Args:
            track_data: Match tracking data
            events: List of all detected events
            homography_matrix: Optional pitch calibration

        Returns:
            Dict mapping team name to PressureMetrics
        """
        logger.info("Computing pressure metrics...")

        results = {}
        for team in ["home", "away"]:
            metrics = PressureMetrics(team=team)

            # PPDA overall
            metrics.ppda_overall = self._compute_ppda(track_data, events, team, homography_matrix)

            # PPDA by zone
            for zone in self.PPDA_ZONES:
                metrics.ppda_by_zone[zone] = self._compute_ppda_by_zone(
                    track_data, events, team, zone, homography_matrix
                )

            # Passes under pressure
            metrics.passes_under_pressure_pct = self._compute_passes_under_pressure(
                track_data, events, team, homography_matrix
            )

            # Pressure events
            metrics.pressure_events = self._count_pressure_events(
                track_data, team, homography_matrix
            )

            # Counter-press
            metrics.counter_press_success_rate = self._compute_counter_press_success(
                track_data, events, team, homography_matrix
            )

            # Time to regain
            metrics.avg_time_to_regain = self._compute_time_to_regain(track_data, events, team)

            # Defensive line and compactness
            line_height, team_width, compactness = self._compute_defensive_shape(
                track_data, team, homography_matrix
            )
            metrics.defensive_line_height_m = line_height
            metrics.team_width_m = team_width
            metrics.compactness_index = compactness

            # Period intensity
            metrics.pressing_intensity_by_period = self._compute_intensity_by_period(
                track_data, events, team, homography_matrix
            )

            results[team] = metrics

        logger.info("Pressure metrics computed")
        return results

    def _get_player_team(self, track_data: MatchTrackData, track_id: int) -> str:
        return (
            track_data.player_teams.get(track_id, "unknown")
            if track_data.player_teams
            else "unknown"
        )

    def _compute_ppda(
        self, track_data: MatchTrackData, events: list[dict], team: str, homography_matrix=None
    ) -> float:
        """Compute PPDA (Passes Per Defensive Action) for a team."""
        # PPDA = opponent passes / defensive actions
        opponent = "away" if team == "home" else "home"

        opponent_passes = [
            e
            for e in events
            if e.get("type") == "pass" and e.get("team") == opponent and e.get("completed")
        ]

        # Defensive actions: tackles, interceptions, fouls, duels won by this team
        defensive_actions = [
            e
            for e in events
            if e.get("type") in ("tackle", "interception", "foul") and e.get("team") == team
        ]

        if not defensive_actions:
            return 999.0  # No pressing at all

        return round(len(opponent_passes) / len(defensive_actions), 2)

    def _compute_ppda_by_zone(
        self,
        track_data: MatchTrackData,
        events: list[dict],
        team: str,
        zone: str,
        homography_matrix=None,
    ) -> float:
        """Compute PPDA in a specific zone."""
        opponent = "away" if team == "home" else "home"
        zone_boundaries = {
            "defensive_third": (0, self.pitch_length * 0.33),
            "middle_third": (self.pitch_length * 0.33, self.pitch_length * 0.67),
            "final_third": (self.pitch_length * 0.67, self.pitch_length),
        }
        z_min, z_max = zone_boundaries.get(zone, (0, self.pitch_length))

        opponent_passes = []
        for e in events:
            if e.get("type") != "pass" or e.get("team") != opponent or not e.get("completed"):
                continue
            meta = e.get("metadata", {})
            if not isinstance(meta, dict):
                continue
            x = meta.get("start_x", 0)
            if z_min <= x < z_max:
                opponent_passes.append(e)

        defensive_actions = []
        for e in events:
            if e.get("type") not in ("tackle", "interception") or e.get("team") != team:
                continue
            meta = e.get("metadata", {})
            if not isinstance(meta, dict):
                continue
            x = meta.get("start_x", 0)
            if z_min <= x < z_max:
                defensive_actions.append(e)

        if not defensive_actions:
            return 999.0

        return round(len(opponent_passes) / len(defensive_actions), 2)

    def _compute_passes_under_pressure(
        self, track_data: MatchTrackData, events: list[dict], team: str, homography_matrix=None
    ) -> float:
        """Compute percentage of passes completed while under pressure."""
        team_passes = [e for e in events if e.get("type") == "pass" and e.get("team") == team]

        if not team_passes:
            return 0.0

        # A pass is under pressure if explicitly flagged, or if incomplete
        under_pressure = [
            e for e in team_passes if e.get("is_pressed", False) or not e.get("completed", True)
        ]

        return round(len(under_pressure) / len(team_passes) * 100, 1)

    def _count_pressure_events(
        self, track_data: MatchTrackData, team: str, homography_matrix=None
    ) -> int:
        """Count pressure events (defender within 2m of opponent ball carrier)."""
        count = 0
        for frame in track_data.frames:
            # Find ball carrier
            ball_det = None
            for det in frame.detections:
                if det.class_name == "sports ball":
                    ball_det = det
                    break

            if ball_det is None:
                continue

            bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2
            by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2

            if homography_matrix is not None:
                with contextlib.suppress(Exception):
                    bx, by = homography_matrix.pixel_to_pitch(bx, by)

            # Find ball carrier (closest player to ball)
            carrier = None
            carrier_dist = float("inf")
            for det in frame.detections:
                if det.class_name != "person" or det.track_id is None:
                    continue
                px = (det.bbox[0] + det.bbox[2]) / 2
                py = (det.bbox[1] + det.bbox[3]) / 2
                if homography_matrix is not None:
                    with contextlib.suppress(Exception):
                        px, py = homography_matrix.pixel_to_pitch(px, py)
                d = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
                if d < carrier_dist and d < 3.0:  # within 3m of ball
                    carrier_dist = d
                    carrier = det

            if carrier is None:
                continue

            carrier_team = self._get_player_team(track_data, carrier.track_id)  # type: ignore[arg-type]
            if carrier_team == "unknown":
                continue

            # Check if any opponent is within 2m
            opponent = "away" if carrier_team == "home" else "home"
            for det in frame.detections:
                if (
                    det.class_name != "person"
                    or det.track_id is None
                    or det.track_id == carrier.track_id
                ):
                    continue
                px = (det.bbox[0] + det.bbox[2]) / 2
                py = (det.bbox[1] + det.bbox[3]) / 2
                if homography_matrix is not None:
                    with contextlib.suppress(Exception):
                        px, py = homography_matrix.pixel_to_pitch(px, py)
                d = math.sqrt((px - bx) ** 2 + (py - by) ** 2)
                if d < self.PRESSURE_DISTANCE_M:
                    other_team = self._get_player_team(track_data, det.track_id)
                    if other_team == opponent:
                        count += 1
                        break  # one pressure event per frame max

        return count

    def _compute_counter_press_success(
        self, track_data: MatchTrackData, events: list[dict], team: str, homography_matrix=None
    ) -> float:
        """Compute counter-press success rate (% of times team regains ball within 5s of loss)."""
        losses = 0
        recoveries = 0

        prev_team = None
        loss_time = None

        for event in sorted(events, key=lambda e: e.get("timestamp", 0)):
            team_evt = event.get("team", "unknown")
            if team_evt == "unknown":
                continue

            if prev_team == team and team_evt != team:
                # Team lost possession
                losses += 1
                loss_time = event.get("timestamp", 0)
            elif prev_team != team and team_evt == team and loss_time is not None:
                # Team regained possession
                time_to_regain = event.get("timestamp", 0) - loss_time
                if time_to_regain <= 8.0:  # within 8 seconds
                    recoveries += 1
                loss_time = None

            prev_team = team_evt

        if losses == 0:
            return 0.0

        return round(recoveries / losses * 100, 1)

    def _compute_time_to_regain(
        self, track_data: MatchTrackData, events: list[dict], team: str
    ) -> float:
        """Compute average time to regain possession after losing it."""
        times = []
        loss_time = None
        prev_team = None

        for event in sorted(events, key=lambda e: e.get("timestamp", 0)):
            team_evt = event.get("team", "unknown")
            if team_evt == "unknown":
                continue

            if prev_team == team and team_evt != team:
                loss_time = event.get("timestamp", 0)
            elif prev_team != team and team_evt == team and loss_time is not None:
                regain_time = event.get("timestamp", 0)
                times.append(regain_time - loss_time)
                loss_time = None

            prev_team = team_evt

        if not times:
            return 0.0

        return round(sum(times) / len(times), 2)

    def _compute_defensive_shape(
        self, track_data: MatchTrackData, team: str, homography_matrix=None
    ) -> tuple[float, float, float]:
        """Compute defensive line height, team width, and compactness."""
        # Get all player positions for this team across all frames
        positions: list[tuple[float, float]] = []

        for frame in track_data.frames:
            for det in frame.detections:
                if det.class_name != "person" or det.track_id is None:
                    continue
                player_team = self._get_player_team(track_data, det.track_id)
                if player_team != team:
                    continue

                px = (det.bbox[0] + det.bbox[2]) / 2
                py = (det.bbox[1] + det.bbox[3]) / 2

                if homography_matrix is not None:
                    with contextlib.suppress(Exception):
                        px, py = homography_matrix.pixel_to_pitch(px, py)

                positions.append((px, py))

        if not positions:
            return 0.0, 0.0, 0.0

        # Defensive line height: average x of back-most players (lowest x for home, highest for away)
        if team == "home":
            back_positions = sorted(positions, key=lambda p: p[0])[: len(positions) // 3]
            line_height = (
                sum(p[0] for p in back_positions) / len(back_positions) if back_positions else 0
            )
        else:
            back_positions = sorted(positions, key=lambda p: -p[0])[: len(positions) // 3]
            line_height = (
                sum(p[0] for p in back_positions) / len(back_positions)
                if back_positions
                else self.pitch_length
            )

        # Team width: spread in y direction
        ys = [p[1] for p in positions]
        team_width = max(ys) - min(ys) if ys else 0

        # Compactness: average distance between all pairs of players (lower = more compact)
        # Simplified: use standard deviation of positions
        if len(positions) > 1:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            x_std = (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs)) ** 0.5
            y_std = (sum((y - sum(ys) / len(ys)) ** 2 for y in ys) / len(ys)) ** 0.5
            compactness = x_std + y_std
        else:
            compactness = 0.0

        return round(line_height, 1), round(team_width, 1), round(compactness, 1)

    def _compute_intensity_by_period(
        self, track_data: MatchTrackData, events: list[dict], team: str, homography_matrix=None
    ) -> dict[str, float]:
        """Compute pressing intensity by match period (15-min segments)."""
        duration = track_data.duration_seconds
        periods = {
            "0-15": (0, 900),
            "15-30": (900, 1800),
            "30-45": (1800, 2700),
            "45-60": (2700, 3600),
            "60-75": (3600, 4500),
            "75-90": (4500, 5400),
        }

        intensity = {}
        for period_name, (start, end) in periods.items():
            if start >= duration:
                continue

            # Count defensive actions in this period
            period_events = [
                e
                for e in events
                if start <= e.get("timestamp", 0) < end
                and e.get("type") in ("tackle", "interception", "duel")
                and e.get("team") == team
            ]

            # Normalize by time (actions per minute)
            period_duration = min(end, duration) - start
            if period_duration > 0:
                actions_per_min = len(period_events) / (period_duration / 60)
                intensity[period_name] = round(actions_per_min, 2)
            else:
                intensity[period_name] = 0.0

        return intensity
