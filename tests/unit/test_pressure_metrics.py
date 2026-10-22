"""Tests for pressure metrics service - PPDA, pressing, defensive shape."""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


def _install_cv_stub() -> None:
    """Install minimal cv_service stub with dataclasses for pressure_metrics_service."""
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
_mod = load_service_module("pressure_test", "pressure_metrics_service.py")
PressureMetricsService = _mod.PressureMetricsService
PressureMetrics = _mod.PressureMetrics
MatchTrackData = _mod.MatchTrackData

import pytest

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Service fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def svc():
    return PressureMetricsService(pitch_length=105.0, pitch_width=68.0)


# ===================================================================
# PPDA
# ===================================================================


class TestPPDA:
    """Passes Per Defensive Action."""

    def test_ppda_no_defensive_actions(self, svc):
        events = [
            _event("pass", "away", timestamp=10.0, metadata={"start_x": 50}),
            _event("pass", "away", timestamp=20.0, metadata={"start_x": 60}),
        ]
        track = _track_data()
        result = svc._compute_ppda(track, events, "home")
        assert result == 999.0

    def test_ppda_normal(self, svc):
        events = [
            _event("pass", "away", timestamp=10.0, metadata={"start_x": 50}),
            _event("pass", "away", timestamp=20.0, metadata={"start_x": 60}),
            _event("pass", "away", timestamp=30.0, metadata={"start_x": 70}),
            _event("pass", "away", timestamp=40.0, metadata={"start_x": 80}),
            _event("pass", "away", timestamp=50.0, metadata={"start_x": 90}),
            _event("tackle", "home", timestamp=15.0),
            _event("interception", "home", timestamp=25.0),
        ]
        track = _track_data()
        result = svc._compute_ppda(track, events, "home")
        # 5 opponent passes / 2 defensive actions = 2.5
        assert result == 2.5

    def test_ppda_foul_counts_as_defensive_action(self, svc):
        events = [
            _event("pass", "away", timestamp=10.0, metadata={"start_x": 50}),
            _event("foul", "home", timestamp=15.0),
        ]
        track = _track_data()
        result = svc._compute_ppda(track, events, "home")
        assert result == 1.0

    def test_ppda_incomplete_passes_excluded(self, svc):
        events = [
            _event("pass", "away", timestamp=10.0, completed=False, metadata={"start_x": 50}),
            _event("pass", "away", timestamp=20.0, completed=True, metadata={"start_x": 60}),
            _event("tackle", "home", timestamp=15.0),
        ]
        track = _track_data()
        result = svc._compute_ppda(track, events, "home")
        # 1 completed opponent pass / 1 defensive action = 1.0
        assert result == 1.0

    def test_ppda_by_zone_normal(self, svc):
        events = [
            _event("pass", "away", timestamp=10.0, metadata={"start_x": 10}),
            _event("pass", "away", timestamp=20.0, metadata={"start_x": 50}),
            _event("pass", "away", timestamp=30.0, metadata={"start_x": 80}),
            _event("tackle", "home", timestamp=15.0, metadata={"start_x": 10}),
            _event("interception", "home", timestamp=25.0, metadata={"start_x": 50}),
        ]
        track = _track_data()
        ppda_def = svc._compute_ppda_by_zone(track, events, "home", "defensive_third")
        ppda_mid = svc._compute_ppda_by_zone(track, events, "home", "middle_third")
        ppda_final = svc._compute_ppda_by_zone(track, events, "home", "final_third")
        # defensive_third: x<34.65 → 1 pass, 1 tackle → 1.0
        assert ppda_def == 1.0
        # middle_third: 34.65 <= x < 70.35 → 1 pass, 1 interception → 1.0
        assert ppda_mid == 1.0
        # final_third: x >= 70.35 → 1 pass, 0 defensive actions → 999.0
        assert ppda_final == 999.0

    def test_ppda_by_zone_empty(self, svc):
        track = _track_data()
        result = svc._compute_ppda_by_zone(track, [], "home", "middle_third")
        assert result == 999.0


# ===================================================================
# Passes under pressure
# ===================================================================


class TestPassesUnderPressure:
    """Percentage of passes made while under pressure."""

    def test_passes_under_pressure_with_flag(self, svc):
        events = [
            _event("pass", "home", timestamp=10.0, is_pressed=True),
            _event("pass", "home", timestamp=20.0, is_pressed=False),
            _event("pass", "home", timestamp=30.0, is_pressed=True),
            _event("pass", "home", timestamp=40.0, is_pressed=False),
        ]
        track = _track_data()
        result = svc._compute_passes_under_pressure(track, events, "home")
        # 2 under pressure / 4 total = 50%
        assert result == 50.0

    def test_passes_under_pressure_no_flag_fallback(self, svc):
        events = [
            _event("pass", "home", timestamp=10.0, is_pressed=False, completed=False),
            _event("pass", "home", timestamp=20.0, completed=True),
        ]
        track = _track_data()
        result = svc._compute_passes_under_pressure(track, events, "home")
        # 1 not completed → counts as under pressure / 2 total = 50%
        assert result == 50.0

    def test_passes_under_pressure_no_team_passes(self, svc):
        events = [
            _event("pass", "away", timestamp=10.0),
        ]
        track = _track_data()
        result = svc._compute_passes_under_pressure(track, events, "home")
        assert result == 0.0

    def test_passes_under_pressure_all_pressed(self, svc):
        events = [
            _event("pass", "home", timestamp=10.0, is_pressed=True),
            _event("pass", "home", timestamp=20.0, is_pressed=True),
        ]
        track = _track_data()
        result = svc._compute_passes_under_pressure(track, events, "home")
        assert result == 100.0


# ===================================================================
# Pressure events
# ===================================================================


class TestPressureEvents:
    """Pressure events — defender within 2m of ball carrier."""

    def test_pressure_event_detected(self, svc):
        # Ball at (50, 50), carrier (home) at same spot, defender (away) at (51, 50)
        ball = _det((48, 48, 52, 52), "sports ball", track_id=None)
        carrier = _det((49, 49, 51, 51), "person", track_id=1)
        defender = _det((50, 49, 52, 51), "person", track_id=2)
        frame = _frame(0, 0.0, [ball, carrier, defender])
        track = _track_data(frames=[frame], player_teams={1: "home", 2: "away"})
        result = svc._count_pressure_events(track, "home")
        assert result == 1

    def test_pressure_event_no_defender_nearby(self, svc):
        # Defender far away (> 2m from ball center)
        ball = _det((48, 48, 52, 52), "sports ball", track_id=None)
        carrier = _det((49, 49, 51, 51), "person", track_id=1)
        defender = _det((100, 100, 104, 104), "person", track_id=2)
        frame = _frame(0, 0.0, [ball, carrier, defender])
        track = _track_data(frames=[frame], player_teams={1: "home", 2: "away"})
        result = svc._count_pressure_events(track, "home")
        assert result == 0

    def test_pressure_event_no_ball_detection(self, svc):
        frame = _frame(0, 0.0, [])
        track = _track_data(frames=[frame])
        result = svc._count_pressure_events(track, "home")
        assert result == 0

    def test_pressure_event_carrier_unknown_team(self, svc):
        ball = _det((48, 48, 52, 52), "sports ball", track_id=None)
        carrier = _det((49, 49, 51, 51), "person", track_id=1)
        defender = _det((50, 49, 52, 51), "person", track_id=2)
        frame = _frame(0, 0.0, [ball, carrier, defender])
        track = _track_data(frames=[frame], player_teams={})  # no team mapping
        result = svc._count_pressure_events(track, "home")
        assert result == 0

    def test_pressure_event_same_team_ignored(self, svc):
        # Both carrier and defender are on "home" → no pressure counted
        ball = _det((48, 48, 52, 52), "sports ball", track_id=None)
        carrier = _det((49, 49, 51, 51), "person", track_id=1)
        defender = _det((50, 49, 52, 51), "person", track_id=2)
        frame = _frame(0, 0.0, [ball, carrier, defender])
        track = _track_data(frames=[frame], player_teams={1: "home", 2: "home"})
        result = svc._count_pressure_events(track, "home")
        assert result == 0

    def test_pressure_event_multiple_frames(self, svc):
        ball = _det((48, 48, 52, 52), "sports ball", track_id=None)
        carrier = _det((49, 49, 51, 51), "person", track_id=1)
        defender = _det((50, 49, 52, 51), "person", track_id=2)
        f1 = _frame(0, 0.0, [ball, carrier, defender])
        f2 = _frame(1, 0.04, [ball, carrier, defender])
        track = _track_data(frames=[f1, f2], player_teams={1: "home", 2: "away"})
        result = svc._count_pressure_events(track, "home")
        assert result == 2


# ===================================================================
# Counter-press success rate
# ===================================================================


class TestCounterPressSuccess:
    """Counter-press: regain possession within 8s of loss."""

    def test_counter_press_success(self, svc):
        events = [
            _event("pass", "home", timestamp=10.0),
            _event("pass", "away", timestamp=14.0),  # home lost, away gained
            _event("pass", "home", timestamp=18.0),  # home regained in 4s ✓
        ]
        track = _track_data()
        result = svc._compute_counter_press_success(track, events, "home")
        assert result == 100.0

    def test_counter_press_no_recovery(self, svc):
        events = [
            _event("pass", "home", timestamp=10.0),
            _event("pass", "away", timestamp=14.0),  # loss
            _event("pass", "away", timestamp=30.0),  # opponent keeps ball
        ]
        track = _track_data()
        result = svc._compute_counter_press_success(track, events, "home")
        assert result == 0.0

    def test_counter_press_recovery_too_late(self, svc):
        events = [
            _event("pass", "home", timestamp=10.0),
            _event("pass", "away", timestamp=14.0),  # loss
            _event("pass", "home", timestamp=30.0),  # regained after 16s (> 8s)
        ]
        track = _track_data()
        result = svc._compute_counter_press_success(track, events, "home")
        assert result == 0.0

    def test_counter_press_no_losses(self, svc):
        events = [
            _event("pass", "home", timestamp=10.0),
            _event("pass", "home", timestamp=20.0),
        ]
        track = _track_data()
        result = svc._compute_counter_press_success(track, events, "home")
        assert result == 0.0

    def test_counter_press_partial_success(self, svc):
        events = [
            _event("pass", "home", timestamp=10.0),
            _event("pass", "away", timestamp=14.0),  # loss 1
            _event("pass", "home", timestamp=16.0),  # recovery 1 in 2s ✓
            _event("pass", "home", timestamp=30.0),
            _event("pass", "away", timestamp=34.0),  # loss 2
            _event("pass", "away", timestamp=50.0),  # no recovery
        ]
        track = _track_data()
        result = svc._compute_counter_press_success(track, events, "home")
        # 1 recovery / 2 losses = 50%
        assert result == 50.0


# ===================================================================
# Time to regain possession
# ===================================================================


class TestTimeToRegain:
    """Average time to regain possession after loss."""

    def test_time_to_regain_average(self, svc):
        events = [
            _event("pass", "home", timestamp=10.0),
            _event("pass", "away", timestamp=14.0),  # loss at 14
            _event("pass", "home", timestamp=18.0),  # regain at 18 → 4s
            _event("pass", "home", timestamp=30.0),
            _event("pass", "away", timestamp=34.0),  # loss at 34
            _event("pass", "home", timestamp=40.0),  # regain at 40 → 6s
        ]
        track = _track_data()
        result = svc._compute_time_to_regain(track, events, "home")
        # (4 + 6) / 2 = 5.0
        assert result == 5.0

    def test_time_to_regain_no_events(self, svc):
        track = _track_data()
        result = svc._compute_time_to_regain(track, [], "home")
        assert result == 0.0

    def test_time_to_regain_no_losses(self, svc):
        events = [
            _event("pass", "home", timestamp=10.0),
            _event("pass", "home", timestamp=20.0),
        ]
        track = _track_data()
        result = svc._compute_time_to_regain(track, events, "home")
        assert result == 0.0

    def test_time_to_regain_unknown_teams_skipped(self, svc):
        events = [
            _event("pass", "unknown", timestamp=10.0),
            _event("pass", "home", timestamp=20.0),
        ]
        track = _track_data()
        result = svc._compute_time_to_regain(track, events, "home")
        assert result == 0.0


# ===================================================================
# Defensive shape
# ===================================================================


class TestDefensiveShape:
    """Line height, team width, compactness."""

    def test_defensive_shape_home(self, svc):
        # Home team players with varying x positions
        p1 = _det((4, 4, 6, 6), "person", track_id=1)  # center (5, 5)
        p2 = _det((14, 14, 16, 16), "person", track_id=2)  # center (15, 15)
        p3 = _det((24, 24, 26, 26), "person", track_id=3)  # center (25, 25)
        p4 = _det((34, 34, 36, 36), "person", track_id=4)  # center (35, 35)
        p5 = _det((44, 44, 46, 46), "person", track_id=5)  # center (45, 45)
        p6 = _det((54, 54, 56, 56), "person", track_id=6)  # center (55, 55)
        frame = _frame(0, 0.0, [p1, p2, p3, p4, p5, p6])
        track = _track_data(
            frames=[frame],
            player_teams={1: "home", 2: "home", 3: "home", 4: "home", 5: "home", 6: "home"},
        )
        line_h, width, compact = svc._compute_defensive_shape(track, "home")
        # back_positions: bottom third (6//3 = 2) → x=5, 15 → line_height = 10
        assert line_h == 10.0
        # ys = 5, 15, 25, 35, 45, 55 → width = 55 - 5 = 50
        assert width == 50.0
        # compactness > 0
        assert compact > 0

    def test_defensive_shape_away(self, svc):
        # Away team players – back positions are highest x (sorted descending)
        p1 = _det((4, 4, 6, 6), "person", track_id=1)
        p2 = _det((44, 44, 46, 46), "person", track_id=2)
        p3 = _det((84, 84, 86, 86), "person", track_id=3)
        frame = _frame(0, 0.0, [p1, p2, p3])
        track = _track_data(frames=[frame], player_teams={1: "away", 2: "away", 3: "away"})
        line_h, width, compact = svc._compute_defensive_shape(track, "away")
        # back_positions: top third (3//3 = 1) → highest x: (85, 85) → line_height = 85
        assert line_h == 85.0
        # ys = 5, 45, 85 → width = 80
        assert width == 80.0
        assert compact > 0

    def test_defensive_shape_empty(self, svc):
        track = _track_data()
        result = svc._compute_defensive_shape(track, "home")
        assert result == (0.0, 0.0, 0.0)

    def test_defensive_shape_single_player(self, svc):
        p1 = _det((10, 10, 20, 20), "person", track_id=1)
        frame = _frame(0, 0.0, [p1])
        track = _track_data(frames=[frame], player_teams={1: "home"})
        line_h, width, compact = svc._compute_defensive_shape(track, "home")
        assert width == 0.0
        assert compact == 0.0


# ===================================================================
# Intensity by period
# ===================================================================


class TestIntensityByPeriod:
    """Pressing intensity (actions/min) in 15-min match periods."""

    def test_intensity_normal(self, svc):
        events = [
            _event("tackle", "home", timestamp=100.0),  # 0-15 period
            _event("tackle", "home", timestamp=200.0),  # 0-15 period
            _event("interception", "home", timestamp=1000.0),  # 15-30 period
            _event("duel", "home", timestamp=3000.0),  # 45-60 period
        ]
        track = _track_data(frames=[], duration=5400.0)
        result = svc._compute_intensity_by_period(track, events, "home")
        # 0-15: 2 actions / 15 min = 0.13
        assert result["0-15"] == 0.13
        # 15-30: 1 action / 15 min = 0.07
        assert result["15-30"] == 0.07
        # 45-60: 1 action / 15 min = 0.07
        assert result["45-60"] == 0.07

    def test_intensity_short_match(self, svc):
        # Only 10 minutes (600s) – only 0-15 period should be computed
        events = [_event("tackle", "home", timestamp=100.0)]
        track = _track_data(frames=[], duration=600.0)
        result = svc._compute_intensity_by_period(track, events, "home")
        assert "0-15" in result
        assert "15-30" not in result

    def test_intensity_empty_events(self, svc):
        track = _track_data(frames=[], duration=5400.0)
        result = svc._compute_intensity_by_period(track, [], "home")
        for period in ("0-15", "15-30", "30-45", "45-60", "60-75", "75-90"):
            assert result[period] == 0.0

    def test_intensity_opponent_events_not_counted(self, svc):
        events = [
            _event("tackle", "away", timestamp=100.0),
            _event("tackle", "home", timestamp=200.0),
        ]
        track = _track_data(frames=[], duration=5400.0)
        result = svc._compute_intensity_by_period(track, events, "home")
        # Only 1 home tackle in 0-15
        assert result["0-15"] == 0.07


# ===================================================================
# Full pipeline & edge cases
# ===================================================================


class TestFullPipeline:
    """End-to-end compute_pressure_metrics and edge cases."""

    @pytest.mark.asyncio
    async def test_compute_pressure_metrics_both_teams(self, svc):
        events = [
            _event("pass", "away", timestamp=10.0, metadata={"start_x": 50}),
            _event("tackle", "home", timestamp=15.0, metadata={"start_x": 50}),
            _event("pass", "home", timestamp=20.0, metadata={"start_x": 40}),
            _event("tackle", "away", timestamp=25.0, metadata={"start_x": 40}),
        ]
        ball = _det((48, 48, 52, 52), "sports ball", track_id=None)
        carrier = _det((49, 49, 51, 51), "person", track_id=1)
        defender = _det((50, 49, 52, 51), "person", track_id=2)
        frame = _frame(0, 0.0, [ball, carrier, defender])
        track = _track_data(frames=[frame], duration=5400.0, player_teams={1: "home", 2: "away"})
        results = await svc.compute_pressure_metrics(track, events)
        assert "home" in results
        assert "away" in results
        assert isinstance(results["home"], PressureMetrics)
        assert isinstance(results["away"], PressureMetrics)

    @pytest.mark.asyncio
    async def test_empty_events_and_empty_track(self, svc):
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, [])
        assert "home" in results
        assert "away" in results
        assert results["home"].ppda_overall == 999.0
        assert results["away"].ppda_overall == 999.0

    @pytest.mark.asyncio
    async def test_custom_pitch_dimensions(self, svc):
        svc_custom = PressureMetricsService(pitch_length=120.0, pitch_width=80.0)
        events = [
            _event("pass", "away", timestamp=10.0, metadata={"start_x": 60}),
            _event("tackle", "home", timestamp=15.0, metadata={"start_x": 60}),
        ]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc_custom.compute_pressure_metrics(track, events)
        # middle_third boundaries: 39.6 to 80.4
        assert results["home"].ppda_by_zone["middle_third"] == 1.0

    @pytest.mark.asyncio
    async def test_away_team_computation_independent(self, svc):
        events = [
            _event("pass", "home", timestamp=10.0, metadata={"start_x": 50}),
            _event("tackle", "away", timestamp=15.0, metadata={"start_x": 50}),
            _event("pass", "home", timestamp=20.0, metadata={"start_x": 60}),
            _event("interception", "away", timestamp=25.0, metadata={"start_x": 60}),
        ]
        track = _track_data(frames=[], duration=5400.0)
        results = await svc.compute_pressure_metrics(track, events)
        # home PPDA: opponent passes (0 for away) / home defensive actions (0) = 999
        assert results["home"].ppda_overall == 999.0
        # away PPDA: opponent passes (2 for home) / away defensive actions (2) = 1.0
        assert results["away"].ppda_overall == 1.0

    def test_get_player_team_returns_unknown(self, svc):
        track = _track_data(player_teams={})
        result = svc._get_player_team(track, 99)
        assert result == "unknown"

    def test_get_player_team_found(self, svc):
        track = _track_data(player_teams={1: "home"})
        result = svc._get_player_team(track, 1)
        assert result == "home"
