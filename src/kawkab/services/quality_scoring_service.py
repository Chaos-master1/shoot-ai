"""Quality scoring service - compute per-match data quality scores.

Provides a comprehensive quality score for each match analysis:
1. Tracking quality (0-1) - how well did BoT-SORT track players
2. Event detection quality (0-1) - how reliable are the detected events
3. Homography quality (0-1) - how good is the pitch calibration
4. Team assignment quality (0-1) - how confident is the team color clustering
5. Overall quality (0-1) - weighted composite of all factors

Stores scores in the database for historical tracking and improvement.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.core.paths import get_paths

logger = get_logger(__name__)


@dataclass
class QualityScores:
    """Quality scores for a match analysis."""

    overall: float = 0.0
    tracking: float = 0.0
    events: float = 0.0
    homography: float = 0.0
    team_assignment: float = 0.0


class QualityScoringService:
    """Computes and stores quality scores for match analyses."""

    def __init__(self) -> None:
        self._db_path = get_paths().database
        self._conn: sqlite3.Connection | None = None
        logger.info("QualityScoringService initialized")

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    async def compute_scores(
        self,
        track_data: Any | None = None,
        analysis: Any | None = None,
        homography_matrix: Any | None = None,
    ) -> QualityScores:
        """Compute quality scores from match data.

        Args:
            track_data: MatchTrackData from CVService
            analysis: MatchAnalysis from AnalysisService
            homography_matrix: HomographyMatrix from calibration

        Returns:
            QualityScores with all sub-scores and overall composite
        """
        tracking_score = self._compute_tracking_score(track_data)
        event_score = self._compute_event_score(analysis)
        homography_score = self._compute_homography_score(homography_matrix)
        team_score = self._compute_team_assignment_score(track_data)

        # Weighted composite
        overall = (
            tracking_score * 0.35 + event_score * 0.25 + homography_score * 0.20 + team_score * 0.20
        )

        return QualityScores(
            overall=round(overall, 3),
            tracking=round(tracking_score, 3),
            events=round(event_score, 3),
            homography=round(homography_score, 3),
            team_assignment=round(team_score, 3),
        )

    async def save_scores(
        self, match_id: int, scores: QualityScores, issues: list[dict] | None = None
    ) -> None:
        """Save quality scores to the database."""
        conn = self._get_conn()
        cursor = conn.cursor()

        warnings = [
            i["description"] for i in (issues or []) if i.get("severity") in ("medium", "low")
        ]
        critical_issues = [i for i in (issues or []) if i.get("severity") in ("critical", "high")]

        cursor.execute(
            """
            INSERT OR REPLACE INTO analysis_quality (
                match_id, overall_score, tracking_score, event_detection_score,
                homography_score, team_assignment_score, issues, warnings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                match_id,
                scores.overall,
                scores.tracking,
                scores.events,
                scores.homography,
                scores.team_assignment,
                json.dumps(critical_issues) if critical_issues else None,
                json.dumps(warnings) if warnings else None,
            ),
        )
        conn.commit()
        logger.info(f"Saved quality scores for match {match_id}: overall={scores.overall}")

    async def get_scores(self, match_id: int) -> QualityScores | None:
        """Get quality scores for a match."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analysis_quality WHERE match_id = ?", (match_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return QualityScores(
            overall=row["overall_score"] or 0,
            tracking=row["tracking_score"] or 0,
            events=row["event_detection_score"] or 0,
            homography=row["homography_score"] or 0,
            team_assignment=row["team_assignment_score"] or 0,
        )

    def _compute_tracking_score(self, track_data: Any | None) -> float:
        """Compute tracking quality score (0-1).

        Combines BoT-SORT metrics with MOT self-consistency when available.
        """
        if track_data is None:
            return 0.0

        metrics = getattr(track_data, "tracking_metrics", {}) or {}
        validated = metrics.get("validated_player_tracks", 0)
        _ = metrics.get("raw_tracks_detected", 1)
        fragmentation = metrics.get("fragmentation_rate", 0)
        quality = metrics.get("tracking_quality", "")

        # Score based on count ratio (expected 22 players)
        count_ratio = min(validated / 22, 1.0) if validated > 0 else 0.0
        if validated > 30:
            count_ratio = max(0.0, 1.0 - (validated - 30) / 20)

        # Score based on fragmentation (lower is better)
        frag_score = max(0.0, 1.0 - fragmentation / 5.0)

        # Quality label bonus
        quality_bonus = {"excellent": 1.0, "good": 0.8, "fair": 0.5, "poor": 0.2, "very_poor": 0.0}
        label_score = quality_bonus.get(quality, 0.5)

        # MOT self-consistency (py-motmetrics intrinsic metrics)
        mot_consistency = metrics.get("mot_self_consistency", None)
        if mot_consistency is not None:
            # Blend traditional metrics with MOT self-consistency
            base_score = count_ratio * 0.4 + frag_score * 0.3 + label_score * 0.3
            return base_score * 0.6 + mot_consistency * 0.4

        return count_ratio * 0.4 + frag_score * 0.3 + label_score * 0.3

    def _compute_event_score(self, analysis: Any | None) -> float:
        """Compute event detection quality score (0-1)."""
        if analysis is None:
            return 0.0

        events = getattr(analysis, "events", []) or []
        _ = len(events)
        shots = sum(1 for e in events if e.get("type") == "shot")
        passes = sum(1 for e in events if e.get("type") == "pass")

        # Expected: 15-30 shots, 300-500 passes per 90-min match
        # For any duration, scale proportionally
        duration = getattr(analysis, "duration_seconds", 90 * 60) or 90 * 60
        duration_factor = duration / (90 * 60)

        expected_shots = 20 * duration_factor
        expected_passes = 400 * duration_factor

        shot_score = min(shots / expected_shots, 1.0) if expected_shots > 0 else 0.0
        if shots > expected_shots * 2:
            shot_score = max(0.0, 1.0 - (shots - expected_shots * 2) / expected_shots)

        pass_score = min(passes / expected_passes, 1.0) if expected_passes > 0 else 0.0
        if passes > expected_passes * 2:
            pass_score = max(0.0, 1.0 - (passes - expected_passes * 2) / expected_passes)

        return shot_score * 0.4 + pass_score * 0.6

    def _compute_homography_score(self, homography_matrix: Any | None) -> float:
        """Compute homography quality score (0-1)."""
        if homography_matrix is None:
            return 0.0

        confidence = getattr(homography_matrix, "confidence", 0) or 0
        error_px = getattr(homography_matrix, "error_px", 100) or 100

        # Confidence score
        conf_score = min(confidence, 1.0)

        # Error score (lower error = higher score)
        error_score = max(0.0, 1.0 - error_px / 50.0)

        return conf_score * 0.6 + error_score * 0.4

    def _compute_team_assignment_score(self, track_data: Any | None) -> float:
        """Compute team assignment quality score (0-1)."""
        if track_data is None:
            return 0.0

        metrics = getattr(track_data, "tracking_metrics", {}) or {}
        team_info = metrics.get("team_detection", {})

        if not team_info.get("enabled", False):
            return 0.0

        assigned = team_info.get("assigned", 0)
        n_clusters = team_info.get("n_clusters", 0)

        # Score based on how many players were assigned
        # Expected ~22 players, but highlight reels may have fewer
        assigned_score = min(assigned / 20, 1.0) if assigned > 0 else 0.0

        # Cluster score (2 clusters = good, 1 = ambiguous, 3+ = refs/spectators)
        if n_clusters == 2:
            cluster_score = 1.0
        elif n_clusters == 3:
            cluster_score = 0.7
        elif n_clusters == 1:
            cluster_score = 0.3
        else:
            cluster_score = 0.5

        return assigned_score * 0.6 + cluster_score * 0.4

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
