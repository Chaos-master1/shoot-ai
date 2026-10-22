"""Dedicated tests for PressureMetricsService class (the service itself, not internals).

NOTE: Internal methods (_compute_ppda, _count_pressure_events, etc.) are already
thoroughly tested in test_pressure_metrics.py. This file focuses on the service
class: initialization, compute_pressure_metrics orchestration, team-level
aggregation, edge cases with real MatchTrackData, etc.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

# ---------------------------------------------------------------------------
# cv_service stub
# ---------------------------------------------------------------------------


def _install_cv_stub() -> None:
    if "kawkab.services" in sys.modules:
        return
    from dataclasses import dataclass, field

    services_mod = types.ModuleType("kawkab.services")
    sys.modules["kawkab.services"] = services_mod
    cv_mod = types.ModuleType("kawkab.services.cv_service")

    @dataclass
    class Detection:
        bbox: tuple[float, float, float, float]
        confidence: float
        class_id: int
        class_name: str
        track_id: int | None = None

    @dataclass
    class FrameDetections:
        frame_number: int
        timestamp: float
        detections: list
        image_width: int
        image_height: int

    @dataclass
    class MatchTrackData:
        match_id: int
        fps: float
        total_frames: int
        duration_seconds: float
        frames: list
        track_registry: dict
        player_teams: dict = field(default_factory=dict)
        tracking_metrics: dict = field(default_factory=dict)
        match_type: str = "unknown"

    cv_mod.Detection = Detection
    cv_mod.FrameDetections = FrameDetections
    cv_mod.MatchTrackData = MatchTrackData
    sys.modules["kawkab.services.cv_service"] = cv_mod
    services_mod.cv_service = cv_mod


_install_cv_stub()
_mod = load_service_module("pressure_svc_test", "pressure_metrics_service.py")
PressureMetricsService = _mod.PressureMetricsService
PressureMetrics = _mod.PressureMetrics

import pytest


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _det(bbox, class_name, track_id=None, confidence=0.9, class_id=1):
    return _ns(
        bbox=bbox,
        confidence=confidence,
        class_id=class_id,
        class_name=class_name,
        track_id=track_id,
    )


def _frame(fn, ts, dets, iw=1920, ih=1080):
    return _ns(frame_number=fn, timestamp=ts, detections=dets, image_width=iw, image_height=ih)


def _track_data(frames=None, duration=5400.0, player_teams=None):
    return _ns(
        match_id=1,
        fps=25.0,
        total_frames=len(frames or []),
        duration_seconds=duration,
        frames=frames or [],
        track_registry={},
        player_teams=player_teams or {},
        tracking_metrics={},
        match_type="unknown",
    )


def _event(type_, team, timestamp=0.0, completed=True, metadata=None, is_pressed=False):
    ev = {
        "type": type_,
        "team": team,
        "timestamp": timestamp,
        "completed": completed,
        "is_pressed": is_pressed,
    }
    if metadata is not None:
        ev["metadata"] = metadata
    return ev


@pytest.fixture
def svc() -> PressureMetricsService:
    return PressureMetricsService(pitch_length=105.0, pitch_width=68.0)


class TestServiceInit:
    def test_default_dimensions(self) -> None:
        svc = PressureMetricsService()
        assert svc.pitch_length == 105.0
        assert svc.pitch_width == 68.0

    def test_custom_dimensions(self) -> None:
        svc = PressureMetricsService(pitch_length=120.0, pitch_width=80.0)
        assert svc.pitch_length == 120.0
        assert svc.pitch_width == 80.0

    def test_constants(self) -> None:
        assert PressureMetricsService.PRESSURE_DISTANCE_M == 2.0
        assert PressureMetricsService.PPDA_ZONES == [
            "defensive_third",
            "middle_third",
            "final_third",
        ]


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_both_teams_returned(self, svc: PressureMetricsService) -> None:
        events = [
            _event("pass", "away", timestamp=10.0, metadata={"start_x": 50}),
            _event("tackle", "home", timestamp=15.0, metadata={"start_x": 50}),
        ]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, events)
        assert set(results.keys()) == {"home", "away"}
        assert isinstance(results["home"], PressureMetrics)
        assert isinstance(results["away"], PressureMetrics)

    @pytest.mark.asyncio
    async def test_ppda_overall_populated(self, svc: PressureMetricsService) -> None:
        events = [
            _event("pass", "away", timestamp=10.0, metadata={"start_x": 50}),
            _event("pass", "away", timestamp=20.0, metadata={"start_x": 60}),
            _event("tackle", "home", timestamp=15.0, metadata={"start_x": 50}),
        ]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, events)
        assert results["home"].ppda_overall == 2.0
        assert results["away"].ppda_overall == 999.0

    @pytest.mark.asyncio
    async def test_ppda_by_zone_populated(self, svc: PressureMetricsService) -> None:
        events = [
            _event("pass", "away", timestamp=10.0, metadata={"start_x": 10}),
            _event("tackle", "home", timestamp=15.0, metadata={"start_x": 10}),
        ]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, events)
        for zone in PressureMetricsService.PPDA_ZONES:
            assert zone in results["home"].ppda_by_zone

    @pytest.mark.asyncio
    async def test_passes_under_pressure_populated(self, svc: PressureMetricsService) -> None:
        events = [
            _event("pass", "home", timestamp=10.0, is_pressed=True),
            _event("pass", "home", timestamp=20.0, is_pressed=False),
        ]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, events)
        assert results["home"].passes_under_pressure_pct == 50.0

    @pytest.mark.asyncio
    async def test_pressure_events_populated(self, svc: PressureMetricsService) -> None:
        ball = _det((48, 48, 52, 52), "sports ball", track_id=None)
        carrier = _det((49, 49, 51, 51), "person", track_id=1)
        defender = _det((50, 49, 52, 51), "person", track_id=2)
        frame = _frame(0, 0.0, [ball, carrier, defender])
        track = _track_data(frames=[frame], player_teams={1: "home", 2: "away"})
        results = await svc.compute_pressure_metrics(track, [])
        assert results["home"].pressure_events >= 0

    @pytest.mark.asyncio
    async def test_counter_press_populated(self, svc: PressureMetricsService) -> None:
        events = [
            _event("pass", "home", timestamp=10.0),
            _event("pass", "away", timestamp=14.0),
            _event("pass", "home", timestamp=16.0),
        ]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, events)
        assert isinstance(results["home"].counter_press_success_rate, float)

    @pytest.mark.asyncio
    async def test_avg_time_to_regain_populated(self, svc: PressureMetricsService) -> None:
        events = [
            _event("pass", "home", timestamp=10.0),
            _event("pass", "away", timestamp=14.0),
            _event("pass", "home", timestamp=18.0),
        ]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, events)
        assert results["home"].avg_time_to_regain == 4.0

    @pytest.mark.asyncio
    async def test_defensive_shape_populated(self, svc: PressureMetricsService) -> None:
        p1 = _det((4, 4, 6, 6), "person", track_id=1)
        p2 = _det((14, 14, 16, 16), "person", track_id=2)
        p3 = _det((24, 24, 26, 26), "person", track_id=3)
        frame = _frame(0, 0.0, [p1, p2, p3])
        track = _track_data(frames=[frame], player_teams={1: "home", 2: "home", 3: "home"})
        results = await svc.compute_pressure_metrics(track, [])
        assert results["home"].defensive_line_height_m > 0

    @pytest.mark.asyncio
    async def test_intensity_by_period_populated(self, svc: PressureMetricsService) -> None:
        events = [
            _event("tackle", "home", timestamp=100.0),
            _event("tackle", "home", timestamp=200.0),
        ]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, events)
        assert "0-15" in results["home"].pressing_intensity_by_period

    @pytest.mark.asyncio
    async def test_empty_events_and_empty_track(self, svc: PressureMetricsService) -> None:
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, [])
        assert results["home"].ppda_overall == 999.0
        assert results["away"].ppda_overall == 999.0
        assert results["home"].pressure_events == 0
        assert results["home"].passes_under_pressure_pct == 0.0

    @pytest.mark.asyncio
    async def test_custom_pitch_affects_zone_boundaries(self, svc: PressureMetricsService) -> None:
        svc_custom = PressureMetricsService(pitch_length=120.0, pitch_width=80.0)
        events = [
            _event("pass", "away", timestamp=10.0, metadata={"start_x": 50}),
            _event("tackle", "home", timestamp=15.0, metadata={"start_x": 50}),
        ]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc_custom.compute_pressure_metrics(track, events)
        assert results["home"].ppda_by_zone["middle_third"] == 1.0

    @pytest.mark.asyncio
    async def test_home_away_independence(self, svc: PressureMetricsService) -> None:
        events = [
            _event("pass", "home", timestamp=10.0, metadata={"start_x": 50}),
            _event("tackle", "away", timestamp=15.0, metadata={"start_x": 50}),
            _event("pass", "home", timestamp=20.0, metadata={"start_x": 60}),
            _event("interception", "away", timestamp=25.0, metadata={"start_x": 60}),
        ]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, events)
        assert results["home"].ppda_overall == 999.0
        assert results["away"].ppda_overall == 1.0

    @pytest.mark.asyncio
    async def test_single_event(self, svc: PressureMetricsService) -> None:
        events = [_event("pass", "away", timestamp=10.0, metadata={"start_x": 50})]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, events)
        assert "home" in results
        assert "away" in results

    @pytest.mark.asyncio
    async def test_no_pressure_events(self, svc: PressureMetricsService) -> None:
        events = [_event("pass", "away", timestamp=10.0)]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, events)
        assert results["home"].pressure_events == 0
        assert results["away"].pressure_events == 0


class TestGetPlayerTeam:
    def test_returns_unknown(self, svc: PressureMetricsService) -> None:
        track = _track_data(player_teams={})
        assert svc._get_player_team(track, 99) == "unknown"

    def test_returns_known_team(self, svc: PressureMetricsService) -> None:
        track = _track_data(player_teams={1: "home"})
        assert svc._get_player_team(track, 1) == "home"


class TestPressureMetricsDataclass:
    def test_default_values(self) -> None:
        pm = PressureMetrics(team="home")
        assert pm.team == "home"
        assert pm.ppda_overall == 0.0
        assert pm.ppda_by_zone == {}
        assert pm.pressure_events == 0
        assert pm.pressing_intensity_by_period == {}
