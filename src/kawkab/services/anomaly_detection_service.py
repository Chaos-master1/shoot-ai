"""Anomaly detection service - identify suspicious data and quality issues.

Scans match data for:
1. Impossible physical stats (speed > 40 km/h, distance > 15 km in 90 min)
2. Tracking quality issues (too many/few tracks, excessive fragmentation)
3. Missing or incomplete data (no ball detections, no team assignment)
4. Statistical outliers (unusual possession splits, extreme formations)
5. Event detection anomalies (too many/few shots, impossible pass distances)

Flags issues with severity levels and actionable recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Anomaly:
    """A single detected anomaly."""

    category: str  # 'physical', 'tracking', 'events', 'team', 'data_quality'
    severity: str  # 'critical', 'high', 'medium', 'low'
    metric: str
    expected_range: str
    actual_value: str
    description: str
    recommendation: str


class AnomalyDetectionService:
    """Detects anomalies and data quality issues in match analysis."""

    # Physical limits (elite human benchmarks)
    MAX_SPEED_KMH = 40.0
    MAX_DISTANCE_90MIN_M = 15000.0
    MIN_DISTANCE_90MIN_M = 2000.0
    MAX_PASS_DISTANCE_M = 80.0

    # Tracking quality thresholds
    MIN_TRACKS_EXPECTED = 18
    MAX_TRACKS_EXPECTED = 35
    MAX_FRAGMENTATION_RATE = 5.0
    MIN_TRACK_LIFETIME_PCT = 5.0

    def __init__(self) -> None:
        logger.info("AnomalyDetectionService initialized")

    async def detect_anomalies(
        self,
        track_data: Any | None = None,
        analysis: Any | None = None,
        events: list[dict] | None = None,
    ) -> list[Anomaly]:
        """Run full anomaly detection on match data.

        Args:
            track_data: MatchTrackData from CVService
            analysis: MatchAnalysis from AnalysisService
            events: List of detected events

        Returns:
            List of detected anomalies with severity and recommendations
        """
        anomalies: list[Anomaly] = []

        if track_data is not None:
            anomalies.extend(self._check_tracking_quality(track_data))

        if analysis is not None:
            anomalies.extend(self._check_physical_stats(analysis))
            anomalies.extend(self._check_team_stats(analysis))
            anomalies.extend(self._check_formations(analysis))

        if events is not None:
            anomalies.extend(self._check_events(events))

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        anomalies.sort(key=lambda a: severity_order.get(a.severity, 4))

        return anomalies

    def _check_tracking_quality(self, track_data: Any) -> list[Anomaly]:
        """Check for tracking quality issues."""
        anomalies = []
        metrics = getattr(track_data, "tracking_metrics", {}) or {}

        validated = metrics.get("validated_player_tracks", 0)
        _ = metrics.get("raw_tracks_detected", 0)
        fragmentation = metrics.get("fragmentation_rate", 0)
        quality = metrics.get("tracking_quality", "")

        if validated < self.MIN_TRACKS_EXPECTED:
            anomalies.append(
                Anomaly(
                    category="tracking",
                    severity="high",
                    metric="validated_tracks",
                    expected_range=f">= {self.MIN_TRACKS_EXPECTED}",
                    actual_value=str(validated),
                    description=f"Only {validated} player tracks detected (expected {self.MIN_TRACKS_EXPECTED}+). "
                    f"Players may be missed due to occlusion, poor video quality, or calibration issues.",
                    recommendation="Check video quality. Ensure camera is elevated and stable. "
                    "Verify pitch is visible. Consider manual calibration.",
                )
            )

        if validated > self.MAX_TRACKS_EXPECTED:
            anomalies.append(
                Anomaly(
                    category="tracking",
                    severity="medium",
                    metric="validated_tracks",
                    expected_range=f"<= {self.MAX_TRACKS_EXPECTED}",
                    actual_value=str(validated),
                    description=f"{validated} tracks detected (expected <= {self.MAX_TRACKS_EXPECTED}). "
                    f"Too many tracks suggests false positives (spectators, refs, bench).",
                    recommendation="Verify pitch mask is working. Check for spectators on sidelines. "
                    "Consider reducing confidence threshold.",
                )
            )

        if fragmentation > self.MAX_FRAGMENTATION_RATE:
            anomalies.append(
                Anomaly(
                    category="tracking",
                    severity="high",
                    metric="fragmentation_rate",
                    expected_range=f"<= {self.MAX_FRAGMENTATION_RATE}",
                    actual_value=f"{fragmentation:.2f}x",
                    description=f"Fragmentation rate is {fragmentation:.2f}x. "
                    f"This means each real player is split into {fragmentation:.1f} track IDs. "
                    f"Camera cuts, occlusion, and poor video quality cause this.",
                    recommendation="Use continuous video (not highlights). Ensure camera doesn't move. "
                    "Consider ReID-enhanced tracking. Kalman smoother may help for full matches.",
                )
            )

        if quality in ("poor", "very_poor"):
            anomalies.append(
                Anomaly(
                    category="tracking",
                    severity="high",
                    metric="tracking_quality",
                    expected_range="good or excellent",
                    actual_value=quality,
                    description=f"Tracking quality is '{quality}'. This will affect all downstream statistics.",
                    recommendation="Review video quality. Ensure players are clearly visible. "
                    "Consider using a higher camera angle.",
                )
            )

        return anomalies

    def _check_physical_stats(self, analysis: Any) -> list[Anomaly]:
        """Check for physically impossible player statistics."""
        anomalies = []
        players = getattr(analysis, "players", {}) or {}

        for tid, player in players.items():
            max_speed = getattr(player, "max_speed_kmh", 0) or 0
            distance = getattr(player, "distance_covered_m", 0) or 0

            if max_speed > self.MAX_SPEED_KMH:
                anomalies.append(
                    Anomaly(
                        category="physical",
                        severity="critical",
                        metric="max_speed",
                        expected_range=f"<= {self.MAX_SPEED_KMH} km/h",
                        actual_value=f"{max_speed:.1f} km/h",
                        description=f"Player {tid} has max speed of {max_speed:.1f} km/h. "
                        f"This exceeds the world record (Usain Bolt: 44.7 km/h). "
                        f"Likely caused by tracking noise or broadcast cut artifacts.",
                        recommendation="Check Kalman smoother settings. Verify frame timestamps. "
                        "Re-analyze with frame_skip=1 for higher accuracy.",
                    )
                )

            if distance > self.MAX_DISTANCE_90MIN_M:
                anomalies.append(
                    Anomaly(
                        category="physical",
                        severity="high",
                        metric="distance_covered",
                        expected_range=f"<= {self.MAX_DISTANCE_90MIN_M / 1000:.1f} km",
                        actual_value=f"{distance / 1000:.2f} km",
                        description=f"Player {tid} covered {distance / 1000:.2f} km. "
                        f"Elite players run ~10-12 km per 90 min. "
                        f"Excessive distance suggests tracking artifacts or repeated track IDs.",
                        recommendation="Check for track fragmentation causing duplicate counting. "
                        "Verify Kalman smoother is active for full matches.",
                    )
                )

        return anomalies

    def _check_team_stats(self, analysis: Any) -> list[Anomaly]:
        """Check for suspicious team-level statistics."""
        anomalies = []
        home = getattr(analysis, "home_team", None)
        away = getattr(analysis, "away_team", None)

        if home and away:
            home_poss = getattr(home, "possession_pct", 50) or 50
            away_poss = getattr(away, "possession_pct", 50) or 50

            if abs(home_poss - away_poss) > 40:
                dominant = "home" if home_poss > away_poss else "away"
                anomalies.append(
                    Anomaly(
                        category="team",
                        severity="medium",
                        metric="possession_split",
                        expected_range="40-60% split",
                        actual_value=f"{home_poss:.1f}% / {away_poss:.1f}%",
                        description=f"Extreme possession imbalance: {dominant} team had {max(home_poss, away_poss):.1f}%. "
                        f"This may indicate team assignment error or one team dominating.",
                        recommendation="Verify team color assignment is correct. Check for swapped teams. "
                        "Use swap_teams() if needed.",
                    )
                )

        return anomalies

    def _check_formations(self, analysis: Any) -> list[Anomaly]:
        """Check for unusual formations."""
        anomalies = []
        formations = getattr(analysis, "formations", {}) or {}

        valid_formations = {
            "4-3-3",
            "4-4-2",
            "4-2-3-1",
            "3-5-2",
            "3-4-3",
            "5-3-2",
            "5-4-1",
            "4-1-4-1",
            "4-5-1",
            "3-4-1-2",
            "3-6-1",
            "2-4-4",
            "4-3-1-2",
            "4-3-2-1",
            "3-3-4",
        }

        for team in ["home", "away"]:
            form_data = formations.get(team, {}) if isinstance(formations, dict) else {}
            if isinstance(form_data, dict):
                formation = form_data.get("formation", "unknown")
                if formation not in valid_formations and formation != "unknown":
                    anomalies.append(
                        Anomaly(
                            category="team",
                            severity="low",
                            metric="formation",
                            expected_range="standard formation",
                            actual_value=formation,
                            description=f"Detected unusual formation '{formation}' for {team} team. "
                            f"This may be due to tracking fragmentation or missing players.",
                            recommendation="Verify player tracking quality. Check if all players are detected. "
                            "Consider manual formation correction if needed.",
                        )
                    )

        return anomalies

    def _check_events(self, events: list[dict]) -> list[Anomaly]:
        """Check for suspicious event patterns."""
        anomalies: list[Anomaly] = []
        if not events:
            return anomalies

        shot_count = sum(1 for e in events if e.get("type") == "shot")
        pass_count = sum(1 for e in events if e.get("type") == "pass")

        if shot_count > 50:
            anomalies.append(
                Anomaly(
                    category="events",
                    severity="medium",
                    metric="shot_count",
                    expected_range="< 50 per match",
                    actual_value=str(shot_count),
                    description=f"Detected {shot_count} shots. Typical match has 15-30 shots. "
                    f"May indicate false positive shot detection (ball bounces, clearances).",
                    recommendation="Review shot detection thresholds. Check ball speed threshold. "
                    "Verify goal proximity detection.",
                )
            )

        if pass_count < 5:
            anomalies.append(
                Anomaly(
                    category="events",
                    severity="high",
                    metric="pass_count",
                    expected_range=">= 50 per match",
                    actual_value=str(pass_count),
                    description=f"Only {pass_count} passes detected. "
                    f"Typical match has 300-500 passes. Ball tracking may be failing.",
                    recommendation="Check ball detection confidence. Verify ball is visible. "
                    "Consider lowering ball_confidence_threshold.",
                )
            )

        # Check for impossible pass distances
        for event in events:
            if event.get("type") != "pass":
                continue
            meta = event.get("metadata", {})
            if isinstance(meta, dict):
                dist = meta.get("distance_m", 0)
                if dist > self.MAX_PASS_DISTANCE_M:
                    anomalies.append(
                        Anomaly(
                            category="events",
                            severity="medium",
                            metric="pass_distance",
                            expected_range=f"<= {self.MAX_PASS_DISTANCE_M}m",
                            actual_value=f"{dist:.1f}m",
                            description=f"Detected pass of {dist:.1f}m. "
                            f"Longest recorded pass in football is ~80m. "
                            f"May indicate tracking error or teleport artifact.",
                            recommendation="Check frame_skip value. Verify player positions. "
                            "Consider using Kalman-smoothed positions.",
                        )
                    )
                    break  # Only report once

        return anomalies

    async def generate_quality_report(self, anomalies: list[Anomaly]) -> dict[str, Any]:
        """Generate a human-readable quality report from anomalies."""
        critical = [a for a in anomalies if a.severity == "critical"]
        high = [a for a in anomalies if a.severity == "high"]
        medium = [a for a in anomalies if a.severity == "medium"]
        low = [a for a in anomalies if a.severity == "low"]

        score = max(0.0, 1.0 - len(critical) * 0.3 - len(high) * 0.15 - len(medium) * 0.05)

        return {
            "overall_score": round(score, 2),
            "total_issues": len(anomalies),
            "critical": len(critical),
            "high": len(high),
            "medium": len(medium),
            "low": len(low),
            "issues": [
                {
                    "category": a.category,
                    "severity": a.severity,
                    "metric": a.metric,
                    "expected": a.expected_range,
                    "actual": a.actual_value,
                    "description": a.description,
                    "recommendation": a.recommendation,
                }
                for a in anomalies
            ],
            "passes": len(critical) == 0 and len(high) == 0,
        }
