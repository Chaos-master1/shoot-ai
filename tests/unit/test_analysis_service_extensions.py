"""Tests for extended analysis_service methods: formation tracking, line-breaking passes, robust attribution."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
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


_install_cv_stub()
_as = load_service_module("as_test2", "analysis_service.py")
AnalysisService = _as.AnalysisService


@dataclass
class FakeBBox:
    cx: float
    cy: float = 0.0


@dataclass
class FakeDetection:
    bbox: FakeBBox
    track_id: int = 0
    team: str = "home"
    is_ball: bool = False
    class_name: str = "player"


@dataclass
class FakeFrame:
    frame_number: int
    timestamp: float = 0.0
    detections: list = field(default_factory=list)


@dataclass
class FakeMatchTrack:
    frames: list = field(default_factory=list)
    fps: float = 30.0


class TestFormationTracking:
    def test_empty_data(self) -> None:
        svc = AnalysisService()
        result = svc.track_formations(FakeMatchTrack(), window_minutes=5)
        assert result["changes"] == 0
        assert result["home_timeline"] == []

    def test_single_formation(self) -> None:
        svc = AnalysisService()
        frames = []
        for i in range(30):
            detections = [
                FakeDetection(bbox=FakeBBox(cx=20), team="home"),
                FakeDetection(bbox=FakeBBox(cx=25), team="home"),
                FakeDetection(bbox=FakeBBox(cx=30), team="home"),
                FakeDetection(bbox=FakeBBox(cx=50), team="home"),
                FakeDetection(bbox=FakeBBox(cx=55), team="home"),
                FakeDetection(bbox=FakeBBox(cx=80), team="home"),
                FakeDetection(bbox=FakeBBox(cx=85), team="home"),
                FakeDetection(bbox=FakeBBox(cx=88), team="home"),
                FakeDetection(bbox=FakeBBox(cx=90), team="home"),
                FakeDetection(bbox=FakeBBox(cx=92), team="home"),
            ]
            frames.append(FakeFrame(frame_number=i, timestamp=i / 30.0, detections=detections))
        result = svc.track_formations(FakeMatchTrack(frames=frames, fps=30), window_minutes=1)
        assert "home_timeline" in result
        assert len(result["home_timeline"]) >= 1

    def test_classify_formation(self) -> None:
        svc = AnalysisService()
        detections = [
            FakeDetection(bbox=FakeBBox(cx=20), team="home"),
            FakeDetection(bbox=FakeBBox(cx=25), team="home"),
            FakeDetection(bbox=FakeBBox(cx=30), team="home"),
            FakeDetection(bbox=FakeBBox(cx=50), team="home"),
            FakeDetection(bbox=FakeBBox(cx=55), team="home"),
            FakeDetection(bbox=FakeBBox(cx=60), team="home"),
            FakeDetection(bbox=FakeBBox(cx=80), team="home"),
            FakeDetection(bbox=FakeBBox(cx=85), team="home"),
            FakeDetection(bbox=FakeBBox(cx=90), team="home"),
            FakeDetection(bbox=FakeBBox(cx=95), team="home"),
        ]
        formation = svc._classify_formation_in_window([FakeFrame(0, 0, detections)], "home")
        assert formation == "3-3-4"

    def test_formation_change_detected(self) -> None:
        svc = AnalysisService()
        win1 = [
            FakeFrame(
                0,
                0,
                [
                    FakeDetection(bbox=FakeBBox(cx=20), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=25), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=30), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=50), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=55), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=60), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=80), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=85), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=90), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=95), team="home"),
                ],
            )
        ]
        win2 = [
            FakeFrame(
                0,
                0,
                [
                    FakeDetection(bbox=FakeBBox(cx=20), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=25), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=28), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=30), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=50), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=55), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=60), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=80), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=85), team="home"),
                    FakeDetection(bbox=FakeBBox(cx=88), team="home"),
                ],
            )
        ]
        f1 = svc._classify_formation_in_window(win1, "home")
        f2 = svc._classify_formation_in_window(win2, "home")
        assert f1 == "3-3-4"
        assert f2 == "4-3-3"
        assert f1 != f2

    def test_empty_window_returns_unknown(self) -> None:
        svc = AnalysisService()
        assert svc._classify_formation_in_window([], "home") == "unknown"


class TestLineBreakingPasses:
    def test_no_passes(self) -> None:
        svc = AnalysisService()
        assert svc.detect_line_breaking_passes([]) == []

    def test_short_pass_not_line_breaking(self) -> None:
        svc = AnalysisService()
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.4, "end_x_pct": 0.5},
            },
        ]
        assert svc.detect_line_breaking_passes(events) == []

    def test_long_forward_pass_detected(self) -> None:
        svc = AnalysisService()
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.8},
                "player_track_id": 7,
            },
        ]
        result = svc.detect_line_breaking_passes(events)
        assert len(result) == 1
        assert result[0]["lines_crossed"] >= 2
        assert result[0]["team"] == "home"

    def test_backward_pass_excluded(self) -> None:
        svc = AnalysisService()
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.8, "end_x_pct": 0.2},
            },
        ]
        assert svc.detect_line_breaking_passes(events) == []

    def test_failed_pass_excluded(self) -> None:
        svc = AnalysisService()
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": False,
                "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.9},
            },
        ]
        assert svc.detect_line_breaking_passes(events) == []

    def test_non_pass_event_excluded(self) -> None:
        svc = AnalysisService()
        events = [
            {
                "type": "shot",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.9},
            },
        ]
        assert svc.detect_line_breaking_passes(events) == []


class TestRobustAttribution:
    def test_explicit_track_id(self) -> None:
        svc = AnalysisService()
        events = [{"type": "pass", "team": "home", "player_track_id": 7}]
        result = svc.attribute_possession_robust(events)
        assert result[0]["attribution_source"] == "explicit"
        assert result[0]["player_track_id"] == 7

    def test_inferred_from_last_known(self) -> None:
        svc = AnalysisService()
        events = [
            {"type": "pass", "team": "home", "player_track_id": 7},
            {"type": "pass", "team": "home"},
        ]
        result = svc.attribute_possession_robust(events)
        assert result[1]["player_track_id"] == 7
        assert result[1]["attribution_source"] == "last_known"

    def test_unknown_when_no_history(self) -> None:
        svc = AnalysisService()
        events = [{"type": "pass", "team": "home"}]
        result = svc.attribute_possession_robust(events)
        assert result[0]["player_track_id"] == -1
        assert result[0]["attribution_source"] == "unknown"

    def test_separate_team_tracking(self) -> None:
        svc = AnalysisService()
        events = [
            {"type": "pass", "team": "home", "player_track_id": 7},
            {"type": "pass", "team": "away", "player_track_id": 3},
            {"type": "pass", "team": "home"},
            {"type": "pass", "team": "away"},
        ]
        result = svc.attribute_possession_robust(events)
        assert result[2]["player_track_id"] == 7
        assert result[3]["player_track_id"] == 3

    def test_empty_events(self) -> None:
        svc = AnalysisService()
        assert svc.attribute_possession_robust([]) == []
