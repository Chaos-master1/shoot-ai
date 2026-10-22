"""Tests for PhysicalLoadService."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


# ---------------------------------------------------------------------------
# cv_service stubs (for MatchTrackData import)
# ---------------------------------------------------------------------------


def _install_cv_service_stub() -> None:
    if "kawkab.services.cv_service" in sys.modules:
        return
    if "kawkab.services" not in sys.modules:
        sys.modules["kawkab.services"] = types.ModuleType("kawkab.services")
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
        detections: list[Detection]
        image_width: int
        image_height: int

    @dataclass
    class MatchTrackData:
        match_id: int
        fps: float
        total_frames: int
        duration_seconds: float
        frames: list[FrameDetections]
        track_registry: dict[int, dict[str, Any]]
        player_teams: dict[int, str] = field(default_factory=dict)
        tracking_metrics: dict[str, Any] = field(default_factory=dict)

    cv_mod.Detection = Detection
    cv_mod.FrameDetections = FrameDetections
    cv_mod.MatchTrackData = MatchTrackData
    sys.modules["kawkab.services.cv_service"] = cv_mod


_install_cv_service_stub()


@pytest.fixture(scope="module")
def pl_mod():
    return load_service_module("kawkab.services.physical_load_service", "physical_load_service.py")


class _FakeHomography:
    def pixel_to_pitch(self, x: float, y: float):
        return (x * 0.05, y * 0.05)


def _make_detection(
    bbox: tuple,
    track_id: int,
    class_name: str = "person",
) -> Any:
    """Create a Detection-like object (avoids importing real class)."""
    from dataclasses import dataclass

    @dataclass
    class _Det:
        bbox: tuple
        confidence: float
        class_id: int
        class_name: str
        track_id: int | None

    return _Det(bbox=bbox, confidence=0.9, class_id=0, class_name=class_name, track_id=track_id)


def _make_frame(frame_num: int, timestamp: float, dets: list) -> Any:
    @dataclass
    class _Frm:
        frame_number: int
        timestamp: float
        detections: list
        image_width: int = 640
        image_height: int = 480

    return _Frm(frame_number=frame_num, timestamp=timestamp, detections=dets)


def _make_track_data(frames: list) -> Any:
    @dataclass
    class _TD:
        match_id: int = 1
        fps: float = 30.0
        total_frames: int = 0
        duration_seconds: float = 0.0
        frames: list = field(default_factory=list)
        track_registry: dict = field(default_factory=dict)
        player_teams: dict = field(default_factory=dict)
        tracking_metrics: dict = field(default_factory=dict)

    return _TD(frames=frames)


class TestPhysicalLoadService:
    @pytest.mark.asyncio
    async def test_no_frames(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        td = _make_track_data(frames=[])
        result = await svc.compute_physical_load(td)
        assert result == {}

    @pytest.mark.asyncio
    async def test_single_frame(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        det = _make_detection((0, 0, 10, 20), track_id=1)
        frm = _make_frame(1, 0.0, [det])
        td = _make_track_data(frames=[frm])
        result = await svc.compute_physical_load(td)
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_person_detections(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        det = _make_detection((0, 0, 10, 20), track_id=1, class_name="sports ball")
        frames = [
            _make_frame(1, 0.0, [det]),
            _make_frame(2, 1.0 / 30, [det]),
        ]
        td = _make_track_data(frames=frames)
        result = await svc.compute_physical_load(td)
        assert result == {}

    @pytest.mark.asyncio
    async def test_single_player_straight_line(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        dets_t1 = [_make_detection((0, 0, 10, 20), track_id=1)]
        dets_t2 = [_make_detection((10, 0, 20, 20), track_id=1)]
        frames = [
            _make_frame(1, 0.0, dets_t1),
            _make_frame(2, 1.0, dets_t2),
        ]
        td = _make_track_data(frames=frames)
        result = await svc.compute_physical_load(td)
        assert 1 in result
        m = result[1]
        assert m.total_distance_m > 0
        assert m.total_distance_m == pytest.approx(10.0, rel=0.1)
        assert m.track_id == 1

    @pytest.mark.asyncio
    async def test_sprint_detection(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        dets = [_make_detection((i * 10, 0, i * 10 + 10, 20), track_id=1) for i in range(31)]
        frames = [_make_frame(i, i * 0.1, [dets[i]]) for i in range(31)]
        td = _make_track_data(frames=frames)
        result = await svc.compute_physical_load(td)
        assert 1 in result
        m = result[1]
        assert m.sprint_count >= 1 or m.high_intensity_distance_m > 0

    @pytest.mark.asyncio
    async def test_high_intensity_running(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        dets = [_make_detection((i * 5, 0, i * 5 + 10, 20), track_id=1) for i in range(21)]
        frames = [_make_frame(i, i * 0.1, [dets[i]]) for i in range(21)]
        td = _make_track_data(frames=frames)
        result = await svc.compute_physical_load(td)
        assert 1 in result
        m = result[1]
        assert m.total_distance_m > 0

    @pytest.mark.asyncio
    async def test_acceleration_deceleration(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        dets = []
        for i in range(20):
            speed = min(i, 20 - i) * 2
            x = speed * 0.1
            dets.append(_make_detection((x, 0, x + 10, 20), track_id=1))
        frames = [_make_frame(i, i * 0.1, [dets[i]]) for i in range(len(dets))]
        td = _make_track_data(frames=frames)
        result = await svc.compute_physical_load(td)
        assert 1 in result
        m = result[1]
        assert isinstance(m.acceleration_count, int)
        assert isinstance(m.deceleration_count, int)

    @pytest.mark.asyncio
    async def test_multiple_players(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        p1 = [_make_detection((0, 0, 10, 20), track_id=1)]
        p2 = [_make_detection((50, 50, 60, 70), track_id=2)]
        frames = [
            _make_frame(1, 0.0, p1 + p2),
            _make_frame(
                2,
                1.0,
                [
                    _make_detection((10, 0, 20, 20), track_id=1),
                    _make_detection((60, 50, 70, 70), track_id=2),
                ],
            ),
        ]
        td = _make_track_data(frames=frames)
        result = await svc.compute_physical_load(td)
        assert len(result) == 2
        assert 1 in result
        assert 2 in result
        assert result[1].track_id == 1
        assert result[2].track_id == 2

    @pytest.mark.asyncio
    async def test_max_speed(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        dets = [_make_detection((i * 15, 0, i * 15 + 10, 20), track_id=1) for i in range(11)]
        frames = [_make_frame(i, i * 0.1, [dets[i]]) for i in range(11)]
        td = _make_track_data(frames=frames)
        result = await svc.compute_physical_load(td)
        m = result[1]
        assert m.max_speed_kmh > 0
        assert m.max_speed_kmh <= 40.0

    @pytest.mark.asyncio
    async def test_homography_matrix_applied(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        homography = _FakeHomography()
        dets_t1 = [_make_detection((0, 0, 10, 20), track_id=1)]
        dets_t2 = [_make_detection((100, 0, 110, 20), track_id=1)]
        frames = [
            _make_frame(1, 0.0, dets_t1),
            _make_frame(2, 1.0, dets_t2),
        ]
        td = _make_track_data(frames=frames)
        result = await svc.compute_physical_load(td, homography_matrix=homography)
        assert 1 in result
        m = result[1]
        assert m.total_distance_m > 0
        assert m.total_distance_m < 20.0

    @pytest.mark.asyncio
    async def test_walking_and_jogging_zones(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        dets = [_make_detection((i, 0, i + 10, 20), track_id=1) for i in range(11)]
        frames = [_make_frame(i, i * 1.0, [dets[i]]) for i in range(11)]
        td = _make_track_data(frames=frames)
        result = await svc.compute_physical_load(td)
        m = result[1]
        total = (
            m.walking_distance_m
            + m.jogging_distance_m
            + m.high_intensity_distance_m
            + m.sprint_distance_m
        )
        assert total == pytest.approx(m.total_distance_m, rel=0.01)

    @pytest.mark.asyncio
    async def test_work_rest_ratio(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        dets = [_make_detection((i * 5, 0, i * 5 + 10, 20), track_id=1) for i in range(21)]
        frames = [_make_frame(i, i * 0.5, [dets[i]]) for i in range(21)]
        td = _make_track_data(frames=frames)
        result = await svc.compute_physical_load(td)
        m = result[1]
        assert m.work_rest_ratio >= 0.0

    @pytest.mark.asyncio
    async def test_metabolic_power_estimate(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        dets = [_make_detection((i * 10, 0, i * 10 + 10, 20), track_id=1) for i in range(11)]
        frames = [_make_frame(i, i * 0.2, [dets[i]]) for i in range(11)]
        td = _make_track_data(frames=frames)
        result = await svc.compute_physical_load(td)
        m = result[1]
        assert m.metabolic_power_estimate > 0

    @pytest.mark.asyncio
    async def test_team_summary(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        metrics_cls = pl_mod.PhysicalLoadMetrics
        m1 = metrics_cls(
            track_id=1,
            total_distance_m=1000,
            sprint_distance_m=200,
            sprint_count=5,
            max_speed_kmh=30.0,
            acceleration_count=10,
            deceleration_count=8,
            work_rest_ratio=0.5,
            metabolic_power_estimate=500,
            jogging_distance_m=300,
            walking_distance_m=200,
            high_intensity_distance_m=300,
        )
        m2 = metrics_cls(
            track_id=2,
            total_distance_m=1200,
            sprint_distance_m=150,
            sprint_count=3,
            max_speed_kmh=28.0,
            acceleration_count=7,
            deceleration_count=6,
            work_rest_ratio=0.4,
            metabolic_power_estimate=450,
            jogging_distance_m=400,
            walking_distance_m=250,
            high_intensity_distance_m=400,
        )
        player_loads = {1: m1, 2: m2}
        player_teams = {1: "home", 2: "home"}
        summary = await svc.compute_team_physical_summary(player_loads, player_teams)
        assert "home" in summary
        assert summary["home"]["players_analyzed"] == 2
        assert summary["home"]["avg_distance_m"] == 1100.0
        assert summary["home"]["total_sprints"] == 8

    @pytest.mark.asyncio
    async def test_team_summary_home_away(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        metrics_cls = pl_mod.PhysicalLoadMetrics
        m1 = metrics_cls(
            track_id=1,
            total_distance_m=1000,
            sprint_distance_m=100,
            sprint_count=2,
            max_speed_kmh=30.0,
            acceleration_count=5,
            deceleration_count=4,
            work_rest_ratio=0.5,
            metabolic_power_estimate=300,
            jogging_distance_m=200,
            walking_distance_m=100,
            high_intensity_distance_m=200,
        )
        m2 = metrics_cls(
            track_id=2,
            total_distance_m=900,
            sprint_distance_m=50,
            sprint_count=1,
            max_speed_kmh=28.0,
            acceleration_count=3,
            deceleration_count=2,
            work_rest_ratio=0.3,
            metabolic_power_estimate=250,
            jogging_distance_m=150,
            walking_distance_m=50,
            high_intensity_distance_m=100,
        )
        player_loads = {1: m1, 2: m2}
        player_teams = {1: "home", 2: "away"}
        summary = await svc.compute_team_physical_summary(player_loads, player_teams)
        assert "home" in summary
        assert "away" in summary
        assert summary["home"]["players_analyzed"] == 1
        assert summary["away"]["players_analyzed"] == 1

    @pytest.mark.asyncio
    async def test_track_id_none_is_skipped(self, pl_mod):
        svc = pl_mod.PhysicalLoadService()
        det = _make_detection((0, 0, 10, 20), track_id=None)
        frames = [
            _make_frame(1, 0.0, [det]),
            _make_frame(2, 1.0, [_make_detection((10, 0, 20, 20), track_id=None)]),
        ]
        td = _make_track_data(frames=frames)
        result = await svc.compute_physical_load(td)
        assert result == {}
