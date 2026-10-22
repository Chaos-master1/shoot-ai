"""Tests for 6 new services: positioning, development, workload, scouting, video review, pitch detection."""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


def _install_cv_stub() -> None:
    if "kawkab.services" in sys.modules:
        return
    services_mod = types.ModuleType("kawkab.services")
    sys.modules["kawkab.services"] = services_mod
    cv_mod = types.ModuleType("kawkab.services.cv_service")

    class FrameDetections:
        pass

    class MatchTrackData:
        pass

    cv_mod.FrameDetections = FrameDetections
    cv_mod.MatchTrackData = MatchTrackData
    sys.modules["kawkab.services.cv_service"] = cv_mod
    services_mod.cv_service = cv_mod
    paths_mod = types.ModuleType("kawkab.core.paths")

    class _Paths:
        def __init__(self):
            from pathlib import Path

            self.calibration_dir = Path("/tmp/cal")
            self.data_dir = Path("/tmp/data")

    paths_mod.get_paths = lambda: _Paths()
    sys.modules["kawkab.core.paths"] = paths_mod


_install_cv_stub()
_pos = load_service_module("pos_test", "positioning_service.py")
_dev = load_service_module("dev_test", "player_development_service.py")
_wk = load_service_module("wk_test", "workload_service.py")
_sc = load_service_module("sc_test", "scouting_service.py")
_vr = load_service_module("vr_test", "video_review_service.py")
_pd = load_service_module("pd_test", "pitch_detector.py")

PositioningService = _pos.PositioningService
RunType = _pos.RunType
Run = _pos.Run

PlayerDevelopmentService = _dev.PlayerDevelopmentService
TrendDirection = _dev.TrendDirection
PlayerMatchStat = _dev.PlayerMatchStat

WorkloadService = _wk.WorkloadService
RiskLevel = _wk.RiskLevel
WorkloadRecord = _wk.WorkloadRecord
WorkloadSource = _wk.WorkloadSource

ScoutingService = _sc.ScoutingService

VideoReviewService = _vr.VideoReviewService
AnnotationKind = _vr.AnnotationKind
ClipTag = _vr.ClipTag

PitchDetector = _pd.PitchDetector

from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class FakeBBox:
    cx: float
    cy: float
    w: float = 1.0
    h: float = 1.0


@dataclass
class FakeDetection:
    bbox: Any
    track_id: int
    team: str
    is_ball: bool = False
    class_name: str = "player"


@dataclass
class FakeFrame:
    frame_number: int
    timestamp: float = 0.0
    detections: list = field(default_factory=list)
    ball_position: tuple | None = None


@dataclass
class FakeMatchTrack:
    frames: list = field(default_factory=list)


class TestPositioningService:
    def test_empty_data(self) -> None:
        svc = PositioningService()
        report = svc.analyze(FakeMatchTrack(), "home")
        assert report.total_runs == 0
        assert "No tracking data" in report.notes[0]

    def test_short_run_filtered(self) -> None:
        svc = PositioningService(min_run_distance_m=10.0)
        frames = []
        for i in range(30):
            frames.append(
                FakeFrame(
                    frame_number=i,
                    detections=[
                        FakeDetection(
                            bbox=FakeBBox(cx=50 + i * 0.1, cy=34),
                            track_id=1,
                            team="home",
                        )
                    ],
                    ball_position=(60, 34),
                )
            )
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs == 0

    def test_long_run_detected(self) -> None:
        svc = PositioningService(min_run_distance_m=5.0, fps=30.0)
        frames = []
        for i in range(60):
            frames.append(
                FakeFrame(
                    frame_number=i,
                    detections=[
                        FakeDetection(
                            bbox=FakeBBox(cx=50 + i * 0.5, cy=34),
                            track_id=1,
                            team="home",
                        )
                    ],
                    ball_position=(60, 34),
                )
            )
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert report.total_runs >= 1
        assert any(r.run_type in (RunType.BEHIND_DEFENSE, RunType.DIAGONAL) for r in report.runs)

    def test_is_in_behind_defense(self) -> None:
        svc = PositioningService()
        assert svc.is_in_behind_defense(95, 80) is True
        assert svc.is_in_behind_defense(50, 80) is False

    def test_run_classification_diagonal(self) -> None:
        frames = []
        for i in range(40):
            frames.append(
                FakeFrame(
                    frame_number=i,
                    detections=[
                        FakeDetection(
                            bbox=FakeBBox(cx=50 + i * 0.3, cy=34 + i * 0.3),
                            track_id=2,
                            team="home",
                        )
                    ],
                )
            )
        svc = PositioningService(min_run_distance_m=3.0)
        report = svc.analyze(FakeMatchTrack(frames=frames), "home")
        assert any(r.run_type == RunType.DIAGONAL for r in report.runs)

    def test_xT_creation(self) -> None:
        svc = PositioningService()
        xt_behind = svc._estimate_created_xt(RunType.BEHIND_DEFENSE, 30, 5)
        xt_drop = svc._estimate_created_xt(RunType.DROP, 30, 5)
        assert xt_behind > 0
        assert xt_drop == 0
        assert xt_behind > xt_drop


class TestPlayerDevelopmentService:
    def make_history(self) -> list[PlayerMatchStat]:
        return [
            PlayerMatchStat("m1", "2024-01-01", 90, 50, 45, 10000, 30, 0.3, 0.4, 10, 1, 0, 50),
            PlayerMatchStat("m2", "2024-01-08", 90, 55, 50, 10500, 35, 0.5, 0.5, 12, 1, 1, 60),
            PlayerMatchStat("m3", "2024-01-15", 80, 60, 56, 11000, 40, 0.4, 0.6, 15, 0, 1, 65),
            PlayerMatchStat("m4", "2024-01-22", 90, 65, 62, 11200, 42, 0.6, 0.7, 18, 2, 1, 70),
            PlayerMatchStat("m5", "2024-01-29", 90, 70, 68, 11500, 45, 0.7, 0.8, 20, 1, 2, 75),
        ]

    def test_insufficient_data(self) -> None:
        svc = PlayerDevelopmentService()
        report = svc.analyze(1, "Test", "MF", [PlayerMatchStat("m1", "2024-01-01", 90)])
        assert report.overall_trend == TrendDirection.INSUFFICIENT_DATA
        assert report.matches_played == 1

    def test_improving_trend(self) -> None:
        svc = PlayerDevelopmentService(improvement_threshold=0.01)
        report = svc.analyze(1, "Test", "MF", self.make_history())
        assert report.matches_played == 5
        assert report.overall_trend in (TrendDirection.IMPROVING, TrendDirection.STABLE)
        pass_completion = next(t for t in report.trends if t.metric == "pass_completion")
        assert pass_completion.direction == TrendDirection.IMPROVING

    def test_declining_trend(self) -> None:
        svc = PlayerDevelopmentService(improvement_threshold=0.01)
        history = [
            PlayerMatchStat(
                f"m{i}",
                f"2024-01-{i:02}",
                90,
                50 - i * 5,
                45 - i * 5,
                10000,
                30,
                0.3,
                0.4,
                10,
                1,
                0,
                50,
            )
            for i in range(1, 6)
        ]
        report = svc.analyze(1, "Test", "MF", history)
        pass_completion = next(t for t in report.trends if t.metric == "pass_completion")
        assert pass_completion.direction == TrendDirection.DECLINING

    def test_strengths_and_improvements(self) -> None:
        svc = PlayerDevelopmentService()
        report = svc.analyze(1, "Test", "MF", self.make_history())
        assert isinstance(report.strengths, list)
        assert isinstance(report.areas_to_improve, list)

    def test_per_90_calculation(self) -> None:
        svc = PlayerDevelopmentService()
        assert svc._per_90(10, 90) == 10.0
        assert svc._per_90(10, 45) == 20.0
        assert svc._per_90(10, 0) == 0.0


class TestWorkloadService:
    def make_records(self) -> list[WorkloadRecord]:
        return [
            WorkloadRecord(f"2024-01-{i:02}", WorkloadSource.MATCH, 90, rpe=7.0, distance_m=10000)
            for i in range(1, 15)
        ]

    def test_empty_data(self) -> None:
        svc = WorkloadService()
        report = svc.analyze(1, "Test", [])
        assert report.risk_level == RiskLevel.INSUFFICIENT_DATA
        assert report.acwr == 0.0

    def test_normal_load(self) -> None:
        svc = WorkloadService()
        report = svc.analyze(1, "Test", self.make_records(), reference_date="2024-01-14")
        assert report.acute_load > 0
        assert report.chronic_load > 0
        assert report.acwr > 0

    def test_high_acwr_flag(self) -> None:
        svc = WorkloadService()

        def high_rpe(d):
            return WorkloadRecord(f"2024-01-{d:02}", WorkloadSource.MATCH, 90, rpe=8.5)

        def low_rpe(d):
            return WorkloadRecord(f"2024-01-{d:02}", WorkloadSource.MATCH, 90, rpe=4.0)

        chronic_records = [low_rpe(d) for d in range(1, 22)]
        acute_records = [high_rpe(d) for d in range(22, 29)]
        records = chronic_records + acute_records
        report = svc.analyze(1, "Test", records, reference_date="2024-01-28")
        assert any("threshold" in f for f in report.flags)

    def test_session_load_calculation(self) -> None:
        svc = WorkloadService()
        r = WorkloadRecord("2024-01-01", WorkloadSource.MATCH, 90, rpe=6.0)
        assert svc._session_load(r) == 540.0
        r2 = WorkloadRecord("2024-01-01", WorkloadSource.TRAINING, 60, rpe=4.0)
        assert svc._session_load(r2) == 240.0

    def test_days_between(self) -> None:
        svc = WorkloadService()
        assert svc._days_between("2024-01-01", "2024-01-08") == 7
        assert svc._days_between("invalid", "2024-01-08") == 0

    def test_classify_risk(self) -> None:
        svc = WorkloadService()
        assert svc._classify_risk(2.5, 3) == RiskLevel.VERY_HIGH
        assert svc._classify_risk(1.6, 3) == RiskLevel.HIGH
        assert svc._classify_risk(1.0, 3) == RiskLevel.LOW
        assert svc._classify_risk(0.5, 3) == RiskLevel.MODERATE
        assert svc._classify_risk(1.0, 0) == RiskLevel.INSUFFICIENT_DATA


class TestScoutingService:
    def make_matches(self) -> list[dict]:
        return [
            {
                "formation": "4-3-3",
                "possession_pct": 60,
                "ppda": 8,
                "set_piece_threat": 0.3,
                "set_piece_conceded": 0.1,
                "width_usage": 0.5,
                "build_up_style": "short",
                "scorers": [{"player": "A", "goals": 1}],
                "assisters": [{"player": "B", "assists": 1}],
                "xg_contributors": [{"player": "A", "xg": 0.4}],
            },
            {
                "formation": "4-3-3",
                "possession_pct": 65,
                "ppda": 7,
                "set_piece_threat": 0.4,
                "set_piece_conceded": 0.05,
                "width_usage": 0.6,
                "build_up_style": "short",
                "scorers": [{"player": "A", "goals": 2}],
                "assisters": [{"player": "C", "assists": 1}],
                "xg_contributors": [{"player": "A", "xg": 0.6}],
            },
            {
                "formation": "4-2-3-1",
                "possession_pct": 55,
                "ppda": 9,
                "set_piece_threat": 0.2,
                "set_piece_conceded": 0.15,
                "width_usage": 0.4,
                "build_up_style": "mixed",
                "scorers": [{"player": "A", "goals": 1}],
                "assisters": [{"player": "B", "assists": 2}],
                "xg_contributors": [{"player": "B", "xg": 0.3}],
            },
        ]

    def test_insufficient_matches(self) -> None:
        svc = ScoutingService()
        profile = svc.analyze("X", self.make_matches()[:2])
        assert profile.preferred_formation == "unknown"
        assert any("at least" in r for r in profile.recommended_tactics)

    def test_full_report(self) -> None:
        svc = ScoutingService()
        profile = svc.analyze("X", self.make_matches())
        assert profile.matches_analyzed == 3
        assert profile.preferred_formation == "4-3-3"
        assert profile.formation_changes == 1
        assert len(profile.top_scorers) > 0
        assert profile.top_scorers[0][0] == "A"

    def test_pressing_classification(self) -> None:
        svc = ScoutingService()
        assert svc._classify_press(7) == "high"
        assert svc._classify_press(10) == "medium"
        assert svc._classify_press(15) == "low"

    def test_aggregates_player_stats(self) -> None:
        svc = ScoutingService()
        result = svc._aggregate_player_stats(self.make_matches(), "scorers", "goals")
        assert result[0][0] == "A"
        assert result[0][1] == 4


class TestVideoReviewService:
    def test_create_session(self) -> None:
        svc = VideoReviewService()
        session = svc.create_session(1, 9000, fps=30.0)
        assert session.session_id
        assert session.match_id == 1
        assert session.total_frames == 9000

    def test_add_clip(self) -> None:
        svc = VideoReviewService()
        session = svc.create_session(1, 9000, fps=30.0)
        clip = svc.add_clip(session.session_id, "Big chance", 100, 200, tags=["chance"])
        assert clip is not None
        assert clip.start_ts == pytest.approx(100 / 30)
        assert clip.end_ts == pytest.approx(200 / 30)
        assert "chance" in clip.tags

    def test_swap_inverted_clip(self) -> None:
        svc = VideoReviewService()
        session = svc.create_session(1, 9000)
        clip = svc.add_clip(session.session_id, "Reversed", 300, 100)
        assert clip.start_frame == 100
        assert clip.end_frame == 300

    def test_add_annotation(self) -> None:
        svc = VideoReviewService()
        session = svc.create_session(1, 9000)
        ann = svc.add_annotation(
            session.session_id,
            AnnotationKind.ARROW,
            frame_number=150,
            geometry={"x1": 100, "y1": 200, "x2": 300, "y2": 250},
            text="run forward",
        )
        assert ann is not None
        assert ann.kind == AnnotationKind.ARROW
        assert "run forward" in ann.text

    def test_remove_clip_and_annotation(self) -> None:
        svc = VideoReviewService()
        session = svc.create_session(1, 9000)
        clip = svc.add_clip(session.session_id, "X", 100, 200)
        assert svc.remove_clip(session.session_id, clip.clip_id)
        ann = svc.add_annotation(session.session_id, AnnotationKind.CIRCLE, 100, {"x": 50, "y": 50})
        assert svc.remove_annotation(session.session_id, ann.annotation_id)

    def test_find_by_tag(self) -> None:
        svc = VideoReviewService()
        session = svc.create_session(1, 9000)
        svc.add_clip(session.session_id, "A", 0, 100, tags=[ClipTag.BUILD_UP.value])
        svc.add_clip(session.session_id, "B", 100, 200, tags=[ClipTag.GOAL.value])
        results = svc.find_clips_by_tag(session.session_id, ClipTag.GOAL.value)
        assert len(results) == 1
        assert results[0].title == "B"

    def test_export_import_roundtrip(self) -> None:
        svc = VideoReviewService()
        session = svc.create_session(1, 9000)
        svc.add_clip(session.session_id, "X", 0, 100, tags=[ClipTag.GOAL.value])
        svc.add_annotation(
            session.session_id, AnnotationKind.ARROW, 50, {"x1": 0, "y1": 0, "x2": 1, "y2": 1}
        )
        payload = svc.export_session(session.session_id)
        assert payload is not None
        new_svc = VideoReviewService()
        imported = new_svc.import_session(payload)
        assert imported is not None
        assert len(imported.clips) == 1
        assert len(imported.annotations) == 1


class TestPitchDetector:
    def test_unavailable_without_opencv(self, monkeypatch) -> None:
        pd = PitchDetector()
        if pd.available:
            pytest.skip("opencv is available in this env")
        guess = pd.detect(b"")
        assert guess.confidence == 0.0
        assert "no input" in guess.notes or "opencv" in guess.notes[0]

    def test_default_guess_dimensions(self) -> None:
        pd = PitchDetector()
        if pd.available:
            pytest.skip("opencv is available in this env")
        guess = pd.detect(b"")
        assert guess.image_width == 0

    def test_classify_lines(self) -> None:
        pd = PitchDetector()
        h, v = pd._classify_lines(
            [
                [[0, 0, 100, 0]],
                [[0, 0, 0, 100]],
            ],
            200,
            200,
        )
        assert len(h) == 1
        assert len(v) == 1

    def test_score_confidence(self) -> None:
        pd = PitchDetector()
        assert pd._score_confidence(0, 0) == 0.0
        assert pd._score_confidence(5, 5) > 0.5

    def test_empty_frame_fallback(self) -> None:
        pd = PitchDetector()
        if pd.available:
            pytest.skip("opencv is available in this env")
        guess = pd.detect(b"not a real image")
        assert guess.confidence == 0.0
