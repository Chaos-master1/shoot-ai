"""Tests for AdvancedEventDetectionService."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


# ---------------------------------------------------------------------------
# cv_service stubs (for MatchTrackData / FrameDetections import)
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
        confidence: float = 0.9
        class_id: int = 0
        class_name: str = "person"
        track_id: int | None = None

    @dataclass
    class FrameDetections:
        frame_number: int
        timestamp: float
        detections: list[Detection]
        image_width: int = 640
        image_height: int = 480

    @dataclass
    class MatchTrackData:
        match_id: int = 1
        fps: float = 30.0
        total_frames: int = 0
        duration_seconds: float = 0.0
        frames: list[FrameDetections] = field(default_factory=list)
        track_registry: dict[int, dict[str, Any]] = field(default_factory=dict)
        player_teams: dict[int, str] = field(default_factory=dict)
        tracking_metrics: dict[str, Any] = field(default_factory=dict)

    cv_mod.Detection = Detection
    cv_mod.FrameDetections = FrameDetections
    cv_mod.MatchTrackData = MatchTrackData
    sys.modules["kawkab.services.cv_service"] = cv_mod


_install_cv_service_stub()


@pytest.fixture(scope="module")
def ae_mod():
    return load_service_module(
        "kawkab.services.advanced_event_detection_service",
        "advanced_event_detection_service.py",
    )


class _FakeHomography:
    def pixel_to_pitch(self, x: float, y: float):
        return (x * 0.05, y * 0.05)


class _FailingHomography:
    """Always raises -- simulates a homography matrix that exists but
    whose pixel_to_pitch call fails (e.g. out-of-bounds input, degenerate
    matrix)."""

    def pixel_to_pitch(self, x: float, y: float):
        raise ValueError("simulated homography failure")


def _det(bbox: tuple, track_id: int | None = None, class_name: str = "person"):
    return types.SimpleNamespace(
        bbox=bbox,
        confidence=0.9,
        class_id=0,
        class_name=class_name,
        track_id=track_id,
    )


def _frame(num: int, ts: float, dets: list, w: int = 640, h: int = 480):
    return types.SimpleNamespace(
        frame_number=num,
        timestamp=ts,
        detections=dets,
        image_width=w,
        image_height=h,
    )


def _td(frames: list, player_teams: dict | None = None) -> Any:
    return types.SimpleNamespace(
        match_id=1,
        fps=30.0,
        total_frames=len(frames),
        duration_seconds=len(frames) / 30.0 if frames else 0.0,
        frames=frames,
        track_registry={},
        player_teams=player_teams or {},
        tracking_metrics={},
    )


class TestAdvancedEventDetectionService:
    def _svc(self, ae_mod):
        return ae_mod.AdvancedEventDetectionService()

    # -- detect_all_advanced_events --

    @pytest.mark.asyncio
    async def test_detect_all_empty_data(self, ae_mod):
        svc = self._svc(ae_mod)
        td = _td([])
        result = await svc.detect_all_advanced_events(td, [])
        assert result == []

    @pytest.mark.asyncio
    async def test_detect_all_preserves_base_events(self, ae_mod):
        svc = self._svc(ae_mod)
        td = _td([])
        base = [{"type": "pass", "timestamp": 0.0}]
        result = await svc.detect_all_advanced_events(td, base)
        assert result == base

    @pytest.mark.asyncio
    async def test_detect_all_sorts_by_timestamp(self, ae_mod):
        svc = self._svc(ae_mod)
        td = _td([])
        base = [{"type": "pass", "timestamp": 5.0}, {"type": "pass", "timestamp": 1.0}]
        result = await svc.detect_all_advanced_events(td, base)
        assert result[0]["timestamp"] == 1.0
        assert result[1]["timestamp"] == 5.0

    # -- dribbles --

    @pytest.mark.asyncio
    async def test_detect_dribbles_single_player_chain(self, ae_mod):
        svc = self._svc(ae_mod)
        # Ball detection must come FIRST so person detection sees ball_det.
        # Uncalibrated pixel positions are approximated to meters with
        # CARRY_PIXEL_TO_METER_RATIO (0.015 m/px), so the ball must move
        # ~1m-equivalent (>= ~67 px over the chain) to clear the 1.0m
        # dribble threshold. The old version of this test moved the ball
        # 20 px total and still passed only because raw pixels were
        # compared against the meter threshold directly (a "dribble" of
        # 20 meters from 20 pixels of movement).
        frames = [
            _frame(
                1,
                0.0,
                [
                    _det((5, 5, 15, 15), track_id=2, class_name="sports ball"),
                    _det((0, 0, 10, 10), track_id=1),
                ],
            ),
            _frame(
                2,
                0.1,
                [
                    _det((45, 5, 55, 15), track_id=2, class_name="sports ball"),
                    _det((40, 0, 50, 10), track_id=1),
                ],
            ),
            _frame(
                3,
                0.2,
                [
                    _det((85, 5, 95, 15), track_id=2, class_name="sports ball"),
                    _det((80, 0, 90, 10), track_id=1),
                ],
            ),
        ]
        td = _td(frames, player_teams={1: "home", 2: "unknown"})
        result = await svc.detect_all_advanced_events(td, [])
        dribbles = [e for e in result if e["type"] == "dribble"]
        assert len(dribbles) == 1
        assert dribbles[0]["track_id"] == 1
        assert dribbles[0]["distance_m"] > 0
        # 80 px * 0.015 m/px = 1.2m -- reported as meters, not pixels.
        assert 0.5 < dribbles[0]["distance_m"] < 5.0

    @pytest.mark.asyncio
    async def test_detect_dribbles_too_few_frames(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [
            _frame(
                1,
                0.0,
                [
                    _det((0, 0, 10, 10), track_id=1),
                    _det((5, 5, 15, 15), track_id=2, class_name="sports ball"),
                ],
            ),
            _frame(
                2,
                0.1,
                [
                    _det((1, 0, 11, 10), track_id=1),
                    _det((6, 5, 16, 15), track_id=2, class_name="sports ball"),
                ],
            ),
        ]
        td = _td(frames, player_teams={1: "home"})
        result = await svc.detect_all_advanced_events(td, [])
        dribbles = [e for e in result if e["type"] == "dribble"]
        assert len(dribbles) == 0

    @pytest.mark.asyncio
    async def test_detect_dribbles_no_ball(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [
            _frame(1, 0.0, [_det((0, 0, 10, 10), track_id=1)]),
            _frame(2, 0.1, [_det((1, 0, 11, 10), track_id=1)]),
            _frame(3, 0.2, [_det((2, 0, 12, 10), track_id=1)]),
        ]
        td = _td(frames, player_teams={1: "home"})
        result = await svc.detect_all_advanced_events(td, [])
        dribbles = [e for e in result if e["type"] == "dribble"]
        assert len(dribbles) == 0

    @pytest.mark.asyncio
    async def test_detect_dribbles_no_false_positive_on_homography_failure(self, ae_mod):
        # Regression test: when pixel_to_pitch raises, the code used to
        # silently keep bx/by in raw pixel space and still append them to
        # the possession chain -- reporting a "distance_m" computed from
        # pixel deltas (tens of "meters" for what's really a few pixels of
        # ball movement), and comparing pixel-scale distance against the
        # 1.0m dribble_min_distance threshold, guaranteeing a false-positive
        # dribble on essentially any ball movement once homography failed.
        # Same frames as test_detect_dribbles_single_player_chain (which
        # legitimately detects a dribble with a working homography-free
        # path), but with a homography that always fails.
        svc = self._svc(ae_mod)
        frames = [
            _frame(
                1,
                0.0,
                [
                    _det((5, 5, 15, 15), track_id=2, class_name="sports ball"),
                    _det((0, 0, 10, 10), track_id=1),
                ],
            ),
            _frame(
                2,
                0.1,
                [
                    _det((15, 5, 25, 15), track_id=2, class_name="sports ball"),
                    _det((1, 0, 11, 10), track_id=1),
                ],
            ),
            _frame(
                3,
                0.2,
                [
                    _det((25, 5, 35, 15), track_id=2, class_name="sports ball"),
                    _det((2, 0, 12, 10), track_id=1),
                ],
            ),
        ]
        td = _td(frames, player_teams={1: "home", 2: "unknown"})
        result = await svc.detect_all_advanced_events(
            td, [], homography_matrix=_FailingHomography()
        )
        dribbles = [e for e in result if e["type"] == "dribble"]
        assert len(dribbles) == 0

    # -- tackles --

    @pytest.mark.asyncio
    async def test_detect_tackles_intercepted_pass(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames, player_teams={1: "home", 2: "away", 3: "away"})
        base = [
            {
                "type": "pass",
                "timestamp": 0.5,
                "from_track_id": 1,
                "to_track_id": 3,
                "completed": False,
                "confidence": 0.6,
                "team": "home",
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        tackles = [e for e in result if e["type"] == "tackle"]
        assert len(tackles) == 1
        assert tackles[0]["team"] == "away"

    @pytest.mark.asyncio
    async def test_detect_tackles_completed_pass_not_tackle(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames, player_teams={1: "home", 2: "home"})
        base = [
            {
                "type": "pass",
                "timestamp": 0.5,
                "from_track_id": 1,
                "to_track_id": 3,
                "completed": True,
                "confidence": 0.9,
                "team": "home",
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        tackles = [e for e in result if e["type"] == "tackle"]
        assert len(tackles) == 0

    # -- interceptions --

    @pytest.mark.asyncio
    async def test_detect_interceptions_team_change_no_pass(self, ae_mod):
        svc = self._svc(ae_mod)
        # Ball within 60px of player, ball detection first
        frames = [
            _frame(
                1,
                0.0,
                [
                    _det((45, 45, 55, 55), track_id=3, class_name="sports ball"),
                    _det((40, 40, 50, 50), track_id=1),
                ],
            ),
            _frame(
                2,
                0.1,
                [
                    _det((45, 45, 55, 55), track_id=3, class_name="sports ball"),
                    _det((40, 40, 50, 50), track_id=2),
                ],
            ),
        ]
        td = _td(frames, player_teams={1: "home", 2: "away"})
        result = await svc.detect_all_advanced_events(td, [])
        interceptions = [e for e in result if e["type"] == "interception"]
        assert len(interceptions) == 1
        assert interceptions[0]["team"] == "away"

    @pytest.mark.asyncio
    async def test_detect_interceptions_same_team_no_event(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [
            _frame(
                1,
                0.0,
                [
                    _det((0, 0, 10, 10), track_id=1),
                    _det((50, 50, 60, 60), track_id=3, class_name="sports ball"),
                ],
            ),
            _frame(
                2,
                0.1,
                [
                    _det((0, 0, 10, 10), track_id=2),
                    _det((50, 50, 60, 60), track_id=3, class_name="sports ball"),
                ],
            ),
        ]
        td = _td(frames, player_teams={1: "home", 2: "home"})
        result = await svc.detect_all_advanced_events(td, [])
        interceptions = [e for e in result if e["type"] == "interception"]
        assert len(interceptions) == 0

    # -- clearances --

    @pytest.mark.asyncio
    async def test_detect_clearances_ball_forward_fast(self, ae_mod):
        svc = self._svc(ae_mod)
        homography = _FakeHomography()
        frames = [
            _frame(
                i,
                i * 0.1,
                [
                    _det((0, 0, 10, 10), track_id=1, class_name="sports ball"),
                ],
            )
            for i in range(5)
        ]
        td = _td(frames)
        result = await svc.detect_all_advanced_events(td, [], homography_matrix=homography)
        # Ball barely moves so speed is low -> no clearance
        clearances = [e for e in result if e["type"] == "clearance"]
        assert len(clearances) == 0

    @pytest.mark.asyncio
    async def test_detect_clearances_no_false_positive_on_homography_failure(self, ae_mod):
        # Regression test: when pixel_to_pitch raises, the code used to
        # silently fall back to pitch_x/pitch_y = bx/by (raw pixels) and
        # still append them to ball_history -- reporting a "speed_mps"
        # computed from pixel deltas (easily hundreds of "m/s" for a few
        # pixels of frame-to-frame ball movement) and comparing it against
        # the 8 m/s clearance speed threshold, which pixel-scale deltas
        # clear trivially. Large, fast pixel movement here would have
        # falsely triggered a clearance under the old behavior.
        svc = self._svc(ae_mod)
        frames = [
            _frame(
                i,
                i * 0.1,
                [
                    _det((i * 100, 0, i * 100 + 10, 10), track_id=1, class_name="sports ball"),
                ],
            )
            for i in range(5)
        ]
        td = _td(frames)
        result = await svc.detect_all_advanced_events(
            td, [], homography_matrix=_FailingHomography()
        )
        clearances = [e for e in result if e["type"] == "clearance"]
        assert len(clearances) == 0

    # -- crosses --

    @pytest.mark.asyncio
    async def test_detect_crosses_wide_pass_into_box(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames)
        base = [
            {
                "type": "pass",
                "timestamp": 1.0,
                "from_track_id": 1,
                "to_track_id": 2,
                "team": "home",
                "completed": True,
                "confidence": 0.8,
                "metadata": {"start_x": 50.0, "start_y": 5.0, "end_x": 95.0, "end_y": 34.0},
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        crosses = [e for e in result if e["type"] == "cross"]
        assert len(crosses) == 1
        assert crosses[0]["team"] == "home"

    @pytest.mark.asyncio
    async def test_detect_crosses_central_pass_not_cross(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames)
        base = [
            {
                "type": "pass",
                "timestamp": 1.0,
                "from_track_id": 1,
                "to_track_id": 2,
                "team": "home",
                "completed": True,
                "confidence": 0.8,
                "metadata": {"start_x": 30.0, "start_y": 34.0, "end_x": 60.0, "end_y": 34.0},
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        crosses = [e for e in result if e["type"] == "cross"]
        assert len(crosses) == 0

    # -- ball recoveries --

    @pytest.mark.asyncio
    async def test_detect_ball_recoveries(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames)
        # Need enough events between recoveries to overcome the 3-event cooldown
        base = [
            {"type": "pass", "timestamp": 0.0, "team": "home"},
            {"type": "pass", "timestamp": 1.0, "team": "away"},  # recovery 1 (home->away)
            {"type": "pass", "timestamp": 2.0, "team": "away"},
            {"type": "pass", "timestamp": 3.0, "team": "away"},
            {"type": "pass", "timestamp": 4.0, "team": "away"},
            {"type": "pass", "timestamp": 5.0, "team": "home"},  # recovery 2 (away->home)
        ]
        result = await svc.detect_all_advanced_events(td, base)
        recoveries = [e for e in result if e["type"] == "ball_recovery"]
        assert len(recoveries) == 2

    # -- blocks --

    @pytest.mark.asyncio
    async def test_detect_blocks_from_failed_pass(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames, player_teams={1: "home"})
        base = [
            {
                "type": "pass",
                "timestamp": 1.0,
                "from_track_id": 1,
                "completed": False,
                "confidence": 0.7,
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        blocks = [e for e in result if e["type"] == "block"]
        assert len(blocks) == 1

    @pytest.mark.asyncio
    async def test_detect_blocks_from_failed_shot(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames, player_teams={1: "home"})
        base = [
            {
                "type": "shot",
                "timestamp": 1.0,
                "from_track_id": 1,
                "completed": False,
                "confidence": 0.5,
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        blocks = [e for e in result if e["type"] == "block"]
        assert len(blocks) == 1

    # -- duels --

    @pytest.mark.asyncio
    async def test_detect_duels_opponents_near_ball(self, ae_mod):
        svc = self._svc(ae_mod)
        ball = _det((50, 50, 60, 60), track_id=None, class_name="sports ball")
        p1 = _det((48, 48, 58, 58), track_id=1)
        p2 = _det((52, 52, 62, 62), track_id=2)
        frames = [_frame(1, 0.0, [ball, p1, p2])]
        td = _td(frames, player_teams={1: "home", 2: "away"})
        result = await svc.detect_all_advanced_events(td, [])
        duels = [e for e in result if e["type"] == "duel"]
        assert len(duels) == 1
        assert duels[0]["team_1"] == "home"
        assert duels[0]["team_2"] == "away"

    @pytest.mark.asyncio
    async def test_detect_duels_same_team_no_event(self, ae_mod):
        svc = self._svc(ae_mod)
        ball = _det((50, 50, 60, 60), track_id=None, class_name="sports ball")
        p1 = _det((48, 48, 58, 58), track_id=1)
        p2 = _det((52, 52, 62, 62), track_id=2)
        frames = [_frame(1, 0.0, [ball, p1, p2])]
        td = _td(frames, player_teams={1: "home", 2: "home"})
        result = await svc.detect_all_advanced_events(td, [])
        duels = [e for e in result if e["type"] == "duel"]
        assert len(duels) == 0

    @pytest.mark.asyncio
    async def test_detect_duels_no_ball(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [_det((0, 0, 10, 10), track_id=1)])]
        td = _td(frames, player_teams={1: "home"})
        result = await svc.detect_all_advanced_events(td, [])
        duels = [e for e in result if e["type"] == "duel"]
        assert len(duels) == 0

    # -- progressive actions --

    @pytest.mark.asyncio
    async def test_detect_progressive_actions(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames)
        base = [
            {
                "type": "pass",
                "timestamp": 0.5,
                "team": "home",
                "confidence": 0.8,
                "metadata": {"start_x": 20.0, "end_x": 50.0},
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        progressive = [e for e in result if e["type"] == "progressive_action"]
        assert len(progressive) == 1
        assert progressive[0]["progress_m"] == 30.0

    @pytest.mark.asyncio
    async def test_detect_progressive_actions_away_team(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames)
        base = [
            {
                "type": "pass",
                "timestamp": 0.5,
                "team": "away",
                "confidence": 0.8,
                "metadata": {"start_x": 80.0, "end_x": 30.0},
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        progressive = [e for e in result if e["type"] == "progressive_action"]
        assert len(progressive) == 1

    @pytest.mark.asyncio
    async def test_detect_progressive_actions_short(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames)
        base = [
            {
                "type": "pass",
                "timestamp": 0.5,
                "team": "home",
                "confidence": 0.8,
                "metadata": {"start_x": 40.0, "end_x": 45.0},
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        progressive = [e for e in result if e["type"] == "progressive_action"]
        assert len(progressive) == 0

    # -- final third entries --

    @pytest.mark.asyncio
    async def test_detect_final_third_entries(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames)
        base = [
            {
                "type": "pass",
                "timestamp": 0.5,
                "team": "home",
                "confidence": 0.8,
                "metadata": {"start_x": 50.0, "end_x": 80.0},
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        entries = [e for e in result if e["type"] == "final_third_entry"]
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_detect_final_third_entries_not_entry(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames)
        base = [
            {
                "type": "pass",
                "timestamp": 0.5,
                "team": "home",
                "confidence": 0.8,
                "metadata": {"start_x": 30.0, "end_x": 40.0},
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        entries = [e for e in result if e["type"] == "final_third_entry"]
        assert len(entries) == 0

    # -- high turnovers --

    @pytest.mark.asyncio
    async def test_detect_high_turnovers(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames)
        base = [
            {
                "type": "pass",
                "timestamp": 0.5,
                "team": "home",
                "completed": False,
                "confidence": 0.8,
                "metadata": {"start_x": 80.0},
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        turnovers = [e for e in result if e["type"] == "high_turnover"]
        assert len(turnovers) == 1

    @pytest.mark.asyncio
    async def test_detect_high_turnovers_away(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames)
        base = [
            {
                "type": "pass",
                "timestamp": 0.5,
                "team": "away",
                "completed": False,
                "confidence": 0.8,
                "metadata": {"start_x": 20.0},
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        turnovers = [e for e in result if e["type"] == "high_turnover"]
        assert len(turnovers) == 1

    @pytest.mark.asyncio
    async def test_detect_high_turnovers_not_in_final_third(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames)
        base = [
            {
                "type": "pass",
                "timestamp": 0.5,
                "team": "home",
                "completed": False,
                "confidence": 0.8,
                "metadata": {"start_x": 30.0},
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        turnovers = [e for e in result if e["type"] == "high_turnover"]
        assert len(turnovers) == 0

    # -- carries (placeholder) --

    @pytest.mark.asyncio
    async def test_detect_carries_empty(self, ae_mod):
        svc = self._svc(ae_mod)
        result = svc._detect_carries(_td([]), [])
        assert result == []

    # -- goals --

    @pytest.mark.asyncio
    async def test_detect_goals_on_target_shot(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [
            _frame(
                1,
                0.0,
                [
                    _det((100, 100, 110, 110), track_id=None, class_name="sports ball"),
                ],
            )
        ]
        td = _td(frames)
        base = [
            {
                "type": "shot",
                "timestamp": 0.0,
                "team": "home",
                "on_target": True,
                "confidence": 0.6,
                "metadata": {"distance_to_goal_m": 10.0, "angle_to_goal_deg": 30.0},
            },
        ]
        result = await svc.detect_all_advanced_events(td, base)
        goals = [e for e in result if e["type"] == "goal"]
        assert len(goals) == 0  # no goal line crossing

    # -- corners (basic smoke test) --

    @pytest.mark.asyncio
    async def test_detect_corners_no_ball(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [_frame(1, 0.0, [])]
        td = _td(frames)
        result = await svc.detect_all_advanced_events(td, [])
        corners = [e for e in result if e["type"] == "corner"]
        assert len(corners) == 0

    # -- free kicks (basic smoke test) --

    @pytest.mark.asyncio
    async def test_detect_free_kicks_ball_moving_not_stationary(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [
            _frame(
                i,
                i * 0.1,
                [
                    _det((i * 5, 0, i * 5 + 10, 10), track_id=None, class_name="sports ball"),
                ],
            )
            for i in range(10)
        ]
        td = _td(frames)
        result = await svc.detect_all_advanced_events(td, [])
        fks = [e for e in result if e["type"] == "free_kick"]
        assert len(fks) == 0

    # -- throw-ins (basic smoke test) --

    @pytest.mark.asyncio
    async def test_detect_throw_ins_no_ball_lost(self, ae_mod):
        svc = self._svc(ae_mod)
        frames = [
            _frame(
                1,
                0.0,
                [
                    _det((0, 0, 10, 10), track_id=None, class_name="sports ball"),
                ],
            )
        ]
        td = _td(frames)
        result = await svc.detect_all_advanced_events(td, [])
        throw_ins = [e for e in result if e["type"] == "throw_in"]
        assert len(throw_ins) == 0

    # -- _get_player_team --

    def test_get_player_team_known(self, ae_mod):
        svc = self._svc(ae_mod)
        td_obj = _td([], player_teams={1: "home"})
        assert svc._get_player_team(td_obj, 1) == "home"

    def test_get_player_team_unknown(self, ae_mod):
        svc = self._svc(ae_mod)
        td_obj = _td([], player_teams={})
        assert svc._get_player_team(td_obj, 999) == "unknown"
