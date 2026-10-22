"""Tests for quality scoring service."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.quality_scoring_service import (  # noqa: E402
    QualityScores,
    QualityScoringService,
)


class FakeTrackData:
    def __init__(self, tracking_metrics=None):
        self.tracking_metrics = tracking_metrics or {}


class FakeAnalysis:
    def __init__(self, events=None, duration_seconds=5400):
        self.events = events or []
        self.duration_seconds = duration_seconds


class FakeHomography:
    def __init__(self, confidence=0.0, error_px=100.0):
        self.confidence = confidence
        self.error_px = error_px


class TestComputeScoresNone:
    @pytest.mark.asyncio
    async def test_all_none_returns_zeros(self):
        svc = QualityScoringService()
        scores = await svc.compute_scores()
        assert isinstance(scores, QualityScores)
        assert scores.overall == 0.0
        assert scores.tracking == 0.0
        assert scores.events == 0.0
        assert scores.homography == 0.0
        assert scores.team_assignment == 0.0

    @pytest.mark.asyncio
    async def test_scores_rounded_to_3dp(self):
        svc = QualityScoringService()
        scores = await svc.compute_scores()
        assert len(str(scores.overall).split(".")[1]) <= 3 if "." in str(scores.overall) else True


class TestTrackingScore:
    @pytest.mark.asyncio
    async def test_perfect_tracking(self):
        svc = QualityScoringService()
        track = FakeTrackData(
            {
                "validated_player_tracks": 22,
                "raw_tracks_detected": 22,
                "fragmentation_rate": 0.0,
                "tracking_quality": "excellent",
            }
        )
        scores = await svc.compute_scores(track_data=track)
        assert scores.tracking > 0.8

    @pytest.mark.asyncio
    async def test_too_many_tracks_penalized(self):
        svc = QualityScoringService()
        track = FakeTrackData(
            {
                "validated_player_tracks": 60,
                "raw_tracks_detected": 100,
                "fragmentation_rate": 0.0,
                "tracking_quality": "excellent",
            }
        )
        scores = await svc.compute_scores(track_data=track)
        assert scores.tracking < 1.0

    @pytest.mark.asyncio
    async def test_poor_tracking_quality(self):
        svc = QualityScoringService()
        track = FakeTrackData(
            {
                "validated_player_tracks": 22,
                "fragmentation_rate": 1.0,
                "tracking_quality": "very_poor",
            }
        )
        scores = await svc.compute_scores(track_data=track)
        # count_ratio=1.0*0.4 + frag_score=0.8*0.3 + label_score=0.0*0.3 = 0.64
        assert scores.tracking == 0.64

    @pytest.mark.asyncio
    async def test_mot_consistency_used_when_present(self):
        svc = QualityScoringService()
        track = FakeTrackData(
            {
                "validated_player_tracks": 22,
                "fragmentation_rate": 0.5,
                "tracking_quality": "good",
                "mot_self_consistency": 0.9,
            }
        )
        scores = await svc.compute_scores(track_data=track)
        assert scores.tracking > 0.6

    @pytest.mark.asyncio
    async def test_high_fragmentation_lowers_score(self):
        svc = QualityScoringService()
        track = FakeTrackData(
            {
                "validated_player_tracks": 22,
                "fragmentation_rate": 10.0,
                "tracking_quality": "good",
            }
        )
        scores = await svc.compute_scores(track_data=track)
        # count_ratio=1.0*0.4 + frag_score=max(0,1-10/5)=0*0.3 + label_score=0.8*0.3 = 0.64
        assert scores.tracking == 0.64


class TestEventScore:
    @pytest.mark.asyncio
    async def test_expected_event_counts(self):
        svc = QualityScoringService()
        events = [{"type": "pass"} for _ in range(400)] + [{"type": "shot"} for _ in range(20)]
        analysis = FakeAnalysis(events=events, duration_seconds=5400)
        scores = await svc.compute_scores(analysis=analysis)
        assert scores.events >= 0.9

    @pytest.mark.asyncio
    async def test_few_events_lowers_score(self):
        svc = QualityScoringService()
        events = [{"type": "pass"} for _ in range(10)] + [{"type": "shot"} for _ in range(1)]
        analysis = FakeAnalysis(events=events, duration_seconds=5400)
        scores = await svc.compute_scores(analysis=analysis)
        assert scores.events < 0.3

    @pytest.mark.asyncio
    async def test_too_many_shots_penalized(self):
        svc = QualityScoringService()
        events = [{"type": "pass"} for _ in range(400)] + [{"type": "shot"} for _ in range(100)]
        analysis = FakeAnalysis(events=events, duration_seconds=5400)
        scores = await svc.compute_scores(analysis=analysis)
        assert scores.events < 1.0


class TestHomographyScore:
    @pytest.mark.asyncio
    async def test_perfect_homography(self):
        svc = QualityScoringService()
        # error_px=0.0 is falsy => code falls back to 100
        h = FakeHomography(confidence=1.0, error_px=0.01)
        scores = await svc.compute_scores(homography_matrix=h)
        # conf_score=1.0, error_score=max(0,1-0.01/50)=0.9998 => homography=0.6*1+0.4*0.9998=0.9999
        assert scores.homography > 0.99

    @pytest.mark.asyncio
    async def test_high_error_lowers_score(self):
        svc = QualityScoringService()
        # error_px=100 => error_score=max(0,1-100/50)=max(0,-1)=0 => homography=0.6*1+0.4*0=0.6
        h = FakeHomography(confidence=1.0, error_px=100.0)
        scores = await svc.compute_scores(homography_matrix=h)
        assert scores.homography == 0.6


class TestTeamAssignmentScore:
    @pytest.mark.asyncio
    async def test_team_detection_disabled_returns_zero(self):
        svc = QualityScoringService()
        track = FakeTrackData({"team_detection": {"enabled": False}})
        scores = await svc.compute_scores(track_data=track)
        assert scores.team_assignment == 0.0

    @pytest.mark.asyncio
    async def test_perfect_team_assignment(self):
        svc = QualityScoringService()
        track = FakeTrackData(
            {
                "team_detection": {
                    "enabled": True,
                    "assigned": 22,
                    "n_clusters": 2,
                }
            }
        )
        scores = await svc.compute_scores(track_data=track)
        assert scores.team_assignment > 0.8

    @pytest.mark.asyncio
    async def test_single_cluster_penalized(self):
        svc = QualityScoringService()
        track = FakeTrackData(
            {
                "team_detection": {
                    "enabled": True,
                    "assigned": 22,
                    "n_clusters": 1,
                }
            }
        )
        scores = await svc.compute_scores(track_data=track)
        assert scores.team_assignment < 0.8

    @pytest.mark.asyncio
    async def test_few_assigned_penalized(self):
        svc = QualityScoringService()
        track = FakeTrackData(
            {
                "validated_player_tracks": 22,
                "fragmentation_rate": 0.0,
                "tracking_quality": "good",
                "team_detection": {
                    "enabled": True,
                    "assigned": 5,
                    "n_clusters": 2,
                },
            }
        )
        scores = await svc.compute_scores(track_data=track)
        # assigned_score=min(5/20,1)=0.25, cluster_score=1.0 => team=0.25*0.6+1.0*0.4=0.55
        assert scores.team_assignment == 0.55


class TestOverallComposite:
    @pytest.mark.asyncio
    async def test_weighted_composite(self):
        svc = QualityScoringService()
        track = FakeTrackData(
            {
                "validated_player_tracks": 22,
                "fragmentation_rate": 0.0,
                "tracking_quality": "excellent",
                "team_detection": {"enabled": True, "assigned": 22, "n_clusters": 2},
            }
        )
        analysis = FakeAnalysis(
            events=[{"type": "pass"} for _ in range(400)] + [{"type": "shot"} for _ in range(20)]
        )
        h = FakeHomography(confidence=1.0, error_px=0.0)
        scores = await svc.compute_scores(track_data=track, analysis=analysis, homography_matrix=h)
        assert scores.overall > 0.5
        assert scores.overall <= 1.0


class TestSaveAndGetScores:
    @pytest.mark.asyncio
    async def test_save_scores_success(self):
        svc = QualityScoringService()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        svc._get_conn = MagicMock(return_value=mock_conn)

        scores = QualityScores(
            overall=0.85, tracking=0.9, events=0.8, homography=0.7, team_assignment=0.75
        )
        await svc.save_scores(1, scores)
        assert mock_cursor.execute.called
        assert mock_conn.commit.called

    @pytest.mark.asyncio
    async def test_save_scores_with_issues_serializes_without_nameerror(self):
        # Regression test: save_scores only reaches json.dumps(...) when
        # issues/warnings is non-empty (falsy short-circuits to None), so
        # the module's missing `import json` went unnoticed by
        # test_save_scores_success above, which never passed any issues.
        svc = QualityScoringService()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        svc._get_conn = MagicMock(return_value=mock_conn)

        scores = QualityScores(
            overall=0.5, tracking=0.5, events=0.5, homography=0.5, team_assignment=0.5
        )
        issues = [
            {"severity": "critical", "description": "Homography failed"},
            {"severity": "medium", "description": "Low frame rate"},
        ]
        await svc.save_scores(1, scores, issues=issues)

        params = mock_cursor.execute.call_args[0][1]
        issues_json, warnings_json = params[-2], params[-1]
        assert json.loads(issues_json) == [issues[0]]
        assert json.loads(warnings_json) == ["Low frame rate"]

    @pytest.mark.asyncio
    async def test_get_scores_returns_none_when_no_data(self):
        svc = QualityScoringService()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor
        svc._get_conn = MagicMock(return_value=mock_conn)

        result = await svc.get_scores(999)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_scores_returns_quality_scores(self):
        svc = QualityScoringService()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        class FakeRow:
            def __getitem__(self, key):
                mapping = {
                    "overall_score": 0.85,
                    "tracking_score": 0.9,
                    "event_detection_score": 0.8,
                    "homography_score": 0.7,
                    "team_assignment_score": 0.75,
                }
                return mapping.get(key, 0)

        mock_cursor.fetchone.return_value = FakeRow()
        mock_conn.cursor.return_value = mock_cursor
        svc._get_conn = MagicMock(return_value=mock_conn)

        result = await svc.get_scores(1)
        assert result is not None
        assert isinstance(result, QualityScores)
        assert result.overall == 0.85
        assert result.tracking == 0.9
        assert result.events == 0.8
