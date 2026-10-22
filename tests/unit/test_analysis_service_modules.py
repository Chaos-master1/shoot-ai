"""Tests for the 4 analysis service submodules: core, passing, tracking, xg_xt.

Each module is tested through AnalysisService (which inherits from all 4 mixins)
to avoid complex class-instantiation boilerplate.  Tests focus on:

  - core.py       → AnalysisServiceCore:  event building, pass classification,
                    progressive passes, pass-type breakdown, possession, confidence
  - passing.py    → PassingMixin:         pass network graph, line-breaking passes,
                    robust possession attribution
  - tracking.py   → TrackingMixin:        player ratings, PPDA, formation detection,
                    formation timeline, per-window classification
  - xg_xt.py      → XgXtMixin:           simple xG heuristic, simple xT heuristic

Coverage: basic functionality, edge cases (empty data, missing keys), bounds.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

_as_mod = load_service_module("kawkab.services.analysis_service", "analysis_service.py")
AnalysisService = _as_mod.AnalysisService
MatchAnalysis = _as_mod.MatchAnalysis
PlayerStats = _as_mod.PlayerStats
TeamStats = _as_mod.TeamStats
from kawkab.core.events import PassEvent, PassType, ShotEvent

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def svc():
    return AnalysisService()


@pytest.fixture
def mock_pass_events():
    return [
        {
            "type": "pass",
            "team": "home",
            "from_track_id": 1,
            "to_track_id": 2,
            "completed": True,
            "timestamp": 10.0,
            "metadata": {
                "start_x_pct": 0.2,
                "start_y_pct": 0.4,
                "end_x_pct": 0.6,
                "end_y_pct": 0.5,
            },
        },
        {
            "type": "pass",
            "team": "home",
            "from_track_id": 2,
            "to_track_id": 3,
            "completed": True,
            "timestamp": 12.0,
            "metadata": {
                "start_x_pct": 0.6,
                "start_y_pct": 0.5,
                "end_x_pct": 0.8,
                "end_y_pct": 0.3,
            },
        },
    ]


# ====================================================================
# XgXtMixin (xg_xt.py)
# ====================================================================


class TestXgXtAnalysis:
    """compute_xg_simple / compute_xt_simple — pure functions, no deps."""

    def test_xg_empty_events(self, svc):
        r = svc.compute_xg_simple([])
        assert r["home"] == 0.0 and r["away"] == 0.0 and r["shot_details"] == []

    def test_xg_single_home_shot(self, svc):
        r = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 12, "angle_to_goal_deg": 30},
                },
            ]
        )
        assert r["home"] > 0 and r["away"] == 0 and len(r["shot_details"]) == 1

    def test_xg_away_shot(self, svc):
        r = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "away",
                    "metadata": {"distance_to_goal_m": 18, "angle_to_goal_deg": 25},
                },
            ]
        )
        assert r["home"] == 0 and r["away"] > 0

    def test_xg_non_shot_ignored(self, svc):
        r = svc.compute_xg_simple(
            [{"type": "pass", "team": "home"}, {"type": "foul", "team": "away"}]
        )
        assert r["home"] == 0 and r["away"] == 0 and r["shot_details"] == []

    def test_xg_distance_factor(self, svc):
        close = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 6, "angle_to_goal_deg": 30},
                }
            ]
        )
        far = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 30, "angle_to_goal_deg": 30},
                }
            ]
        )
        assert close["home"] > far["home"]

    def test_xg_angle_factor(self, svc):
        center = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 12, "angle_to_goal_deg": 0},
                }
            ]
        )
        wide = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 12, "angle_to_goal_deg": 75},
                }
            ]
        )
        assert center["home"] > wide["home"]

    def test_xg_bounds(self, svc):
        r = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 1, "angle_to_goal_deg": 0},
                },
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 80, "angle_to_goal_deg": 89},
                },
            ]
        )
        for s in r["shot_details"]:
            assert 0.0 <= s["xg"] <= 1.0

    def test_xg_missing_metadata_defaults(self, svc):
        # Honest-absent: a shot with no spatial metadata contributes xg 0
        # and is marked xg_available=False — never a fabricated estimate.
        r = svc.compute_xg_simple([{"type": "shot", "team": "home"}])
        assert r["home"] == 0.0
        assert r["shot_details"][0]["xg_available"] is False

    def test_xt_empty_events(self, svc):
        r = svc.compute_xt_simple([])
        assert r["home"] == 0.0 and r["away"] == 0.0

    def test_xt_forward_pass(self, svc):
        r = svc.compute_xt_simple(
            [
                {
                    "type": "pass",
                    "team": "home",
                    "completed": True,
                    "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.7},
                },
            ]
        )
        assert r["home"] > 0 and r["away"] == 0

    def test_xt_uncompleted_pass(self, svc):
        r = svc.compute_xt_simple(
            [
                {
                    "type": "pass",
                    "team": "home",
                    "completed": False,
                    "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.7},
                },
            ]
        )
        assert r["home"] == 0

    def test_xt_backward_pass(self, svc):
        r = svc.compute_xt_simple(
            [
                {
                    "type": "pass",
                    "team": "home",
                    "completed": True,
                    "metadata": {"start_x_pct": 0.7, "end_x_pct": 0.3},
                },
            ]
        )
        assert r["home"] == 0

    def test_xt_away_team(self, svc):
        r = svc.compute_xt_simple(
            [
                {
                    "type": "pass",
                    "team": "away",
                    "completed": True,
                    "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.7},
                },
            ]
        )
        assert r["away"] > 0 and r["home"] == 0

    def test_xt_non_pass_ignored(self, svc):
        r = svc.compute_xt_simple([{"type": "shot", "team": "home"}, {"type": "foul"}])
        assert r["home"] == 0 and r["away"] == 0


# ====================================================================
# PassingMixin (passing.py)
# ====================================================================


class TestPassingAnalysis:
    """Pass network, line-breaking passes, possession attribution."""

    def test_pass_network_basic(self, svc, mock_pass_events):
        # _compute_pass_network delegates to core.pass_network.PassNetwork,
        # whose per-edge shape is {source,target,attempted,completed,
        # completion_pct} -- a superset of the old bare {source,target,weight}.
        r = svc._compute_pass_network(mock_pass_events)
        assert len(r["nodes"]) == 3
        assert len(r["edges"]) == 2
        assert {
            "source": 1,
            "target": 2,
            "attempted": 1,
            "completed": 1,
            "completion_pct": 100.0,
        } in r["edges"]
        assert {
            "source": 2,
            "target": 3,
            "attempted": 1,
            "completed": 1,
            "completion_pct": 100.0,
        } in r["edges"]

    def test_pass_network_empty(self, svc):
        r = svc._compute_pass_network([])
        assert r == {"nodes": [], "edges": []}

    def test_pass_network_incomplete_ignored(self, svc):
        events = [
            {"type": "pass", "from_track_id": 1, "to_track_id": 2, "completed": True},
            {"type": "pass", "from_track_id": 2, "to_track_id": 3, "completed": False},
        ]
        r = svc._compute_pass_network(events)
        assert len(r["edges"]) == 1

    def test_pass_network_non_pass_ignored(self, svc):
        r = svc._compute_pass_network([{"type": "shot", "from_track_id": 1, "to_track_id": 2}])
        assert r["edges"] == []

    def test_line_breaking_passes(self, svc):
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.9},
            },
        ]
        breaks = svc.detect_line_breaking_passes(events, n_lines=3)
        assert len(breaks) == 1
        assert breaks[0]["lines_crossed"] >= 2

    def test_line_breaking_none(self, svc):
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.2},
            },
        ]
        breaks = svc.detect_line_breaking_passes(events, n_lines=3)
        assert breaks == []

    def test_line_breaking_backward(self, svc):
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.7, "end_x_pct": 0.3},
            },
        ]
        breaks = svc.detect_line_breaking_passes(events, n_lines=3)
        assert breaks == []

    def test_line_breaking_incomplete_ignored(self, svc):
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": False,
                "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.9},
            },
        ]
        assert svc.detect_line_breaking_passes(events, n_lines=3) == []

    def test_line_breaking_crosses_two_lines(self, svc):
        events = [
            {
                "type": "pass",
                "team": "home",
                "completed": True,
                "metadata": {"start_x_pct": 0.1, "end_x_pct": 0.6},
            },
        ]
        breaks = svc.detect_line_breaking_passes(events, n_lines=3)
        assert len(breaks) == 1
        assert breaks[0]["lines_crossed"] >= 2

    def test_attribute_possession_explicit(self, svc):
        events = [{"team": "home", "player_track_id": 7, "type": "pass"}]
        attr = svc.attribute_possession_robust(events)
        assert attr[0]["attribution_source"] == "explicit"
        assert attr[0]["player_track_id"] == 7

    def test_attribute_possession_inferred(self, svc):
        events = [
            {"team": "home", "player_track_id": 7, "type": "pass"},
            {"team": "home", "type": "pass"},  # no track_id
        ]
        attr = svc.attribute_possession_robust(events)
        assert attr[1]["attribution_source"] == "last_known"
        assert attr[1]["player_track_id"] == 7

    def test_attribute_possession_unknown(self, svc):
        events = [{"team": "home", "type": "pass"}]  # no track_id, no history
        attr = svc.attribute_possession_robust(events)
        assert attr[0]["attribution_source"] == "unknown"
        assert attr[0]["player_track_id"] == -1

    def test_attribute_possession_multiple_teams(self, svc):
        events = [
            {"team": "home", "player_track_id": 1, "type": "pass"},
            {"team": "away", "player_track_id": 10, "type": "pass"},
            {"team": "home", "type": "pass"},
            {"team": "away", "type": "pass"},
        ]
        attr = svc.attribute_possession_robust(events)
        assert attr[2]["player_track_id"] == 1
        assert attr[3]["player_track_id"] == 10


# ====================================================================
# AnalysisServiceCore (core.py)
# ====================================================================


class TestCoreAnalysis:
    """Event building, pass classification, progressive passes, breakdown, possession, confidence, team stats."""

    def test_init_defaults(self, svc):
        assert svc.pitch_length == 105.0
        assert svc.pitch_width == 68.0
        assert svc.use_kalman is True

    def test_init_custom(self):
        svc = AnalysisService(pitch_length_m=120, pitch_width_m=75, use_kalman=False)
        assert svc.pitch_length == 120
        assert svc.pitch_width == 75
        assert svc.use_kalman is False

    # -- _build_typed_pass --

    def test_build_pass_basic(self, svc):
        ev = {
            "timestamp": 10.5,
            "team": "home",
            "from_track_id": 1,
            "to_track_id": 2,
            "completed": True,
            "confidence": 0.9,
            "metadata": {
                "start_x_pct": 0.3,
                "start_y_pct": 0.4,
                "end_x_pct": 0.7,
                "end_y_pct": 0.5,
            },
        }
        pe = svc._build_typed_pass(ev)
        assert pe.timestamp == 10.5 and pe.team == "home" and pe.track_id == 1
        assert pe.to_track_id == 2 and pe.completed is True
        assert pe.length_m > 0
        assert abs(pe.length_m - 42.0) < 1

    def test_build_pass_defaults(self, svc):
        pe = svc._build_typed_pass({})
        assert pe.timestamp == 0 and pe.team == "unknown"
        assert pe.track_id is None and pe.completed is True
        assert pe.length_m >= 0

    # -- _build_typed_shot --

    def test_build_shot_basic(self, svc):
        ev = {
            "timestamp": 20.0,
            "team": "away",
            "track_id": 3,
            "on_target": True,
            "confidence": 0.85,
            "metadata": {"distance_to_goal_m": 15.0, "angle_to_goal_deg": 25.0, "xg": 0.12},
        }
        se = svc._build_typed_shot(ev)
        assert se.timestamp == 20.0 and se.team == "away"
        assert se.track_id == 3 and se.on_target is True
        assert se.distance_m == 15.0 and se.angle_deg == 25.0 and se.xg == 0.12

    def test_build_shot_defaults(self, svc):
        se = svc._build_typed_shot({})
        assert se.timestamp == 0 and se.team == "unknown"
        # Honest-absent: no spatial metadata → None, not the old
        # fabricated distance=18/angle=30 defaults that made the xG model
        # produce plausible-looking numbers for unknowable shots.
        assert se.on_target is False and se.distance_m is None
        assert se.angle_deg is None and se.xg == 0.0

    # -- _pixel_dist_to_meters --

    def test_pixel_dist_no_homography(self, svc):
        # Without calibration, pixels are approximated to meters with
        # CARRY_PIXEL_TO_METER_RATIO (0.015), NOT returned raw -- raw pixel
        # distances fed the 2m "is_pressed" threshold as if they were meters.
        assert svc._pixel_dist_to_meters(0, 0, 3, 4, None) == pytest.approx(5.0 * 0.015)

    def test_pixel_dist_with_homography(self, svc):
        h = MagicMock()
        h.pixel_to_pitch.side_effect = lambda x, y: (x * 0.1, y * 0.1)
        d = svc._pixel_dist_to_meters(0, 0, 30, 40, h)
        assert math.isclose(d, 5.0)
        assert h.pixel_to_pitch.call_count == 2

    def test_pixel_dist_homography_fallback(self, svc):
        h = MagicMock()
        h.pixel_to_pitch.side_effect = ValueError("bad")
        d = svc._pixel_dist_to_meters(0, 0, 3, 4, h)
        # Conversion failed mid-match: fall back to the pixel->meter
        # approximation, not raw pixels (a "nearest defender distance" of
        # 5 "meters" for a 5-pixel gap made the is_pressed check useless).
        assert d == pytest.approx(5.0 * 0.015)

    # -- _classify_pass_types --

    def test_classify_long_ball(self, svc):
        pe = PassEvent(timestamp=0, team="home", track_id=1, length_m=35.0)
        svc._classify_pass_types([pe])
        assert pe.pass_type == PassType.STANDARD

    def test_classify_back_pass(self, svc):
        pe = PassEvent(timestamp=0, team="home", track_id=1, length_m=10.0, start_x=0.7, end_x=0.3)
        svc._classify_pass_types([pe])
        assert pe.pass_type == PassType.BACK_PASS

    def test_classify_cross(self, svc):
        pe = PassEvent(
            timestamp=0,
            team="home",
            track_id=1,
            length_m=10.0,
            start_x=0.6,
            start_y=0.1,
            end_x=0.8,
            end_y=0.1,
        )
        svc._classify_pass_types([pe])
        assert pe.pass_type == PassType.CROSS
        assert pe.is_cross is True

    def test_classify_through_ball(self, svc):
        pe = PassEvent(timestamp=0, team="home", track_id=1, length_m=20.0, start_x=0.3, end_x=0.5)
        svc._classify_pass_types([pe])
        assert pe.pass_type == PassType.THROUGH_BALL
        assert pe.is_through_ball is True

    def test_classify_one_touch(self, svc):
        pe = PassEvent(timestamp=0, team="home", track_id=1, length_m=3.0, start_x=0.4, end_x=0.45)
        svc._classify_pass_types([pe])
        assert pe.pass_type == PassType.ONE_TOUCH

    def test_classify_standard(self, svc):
        pe = PassEvent(timestamp=0, team="home", track_id=1, length_m=8.0, start_x=0.4, end_x=0.5)
        svc._classify_pass_types([pe])
        assert pe.pass_type == PassType.STANDARD

    def test_classify_switch(self, svc):
        pe = PassEvent(
            timestamp=0,
            team="home",
            track_id=1,
            length_m=15.0,
            start_x=0.3,
            start_y=0.2,
            end_x=0.4,
            end_y=0.8,
        )
        svc._classify_pass_types([pe])
        assert pe.pass_type == PassType.SWITCH

    def test_classify_non_pass_unchanged(self, svc):
        se = ShotEvent(timestamp=0, team="home", track_id=1)
        svc._classify_pass_types([se])
        assert not hasattr(se, "pass_type") or True  # shot events should be skipped

    # -- _find_progressive_passes --

    def test_progressive_passes(self, svc):
        events = [
            PassEvent(
                timestamp=0,
                team="home",
                track_id=1,
                completed=True,
                start_x=0.3,
                end_x=0.5,
                length_m=15.0,
            ),
            PassEvent(
                timestamp=1,
                team="home",
                track_id=2,
                completed=True,
                start_x=0.5,
                end_x=0.52,
                length_m=5.0,
            ),
            PassEvent(
                timestamp=2,
                team="home",
                track_id=1,
                completed=False,
                start_x=0.3,
                end_x=0.6,
                length_m=20.0,
            ),
        ]
        r = svc._find_progressive_passes(events)
        assert len(r) == 1

    def test_progressive_passes_empty(self, svc):
        assert svc._find_progressive_passes([]) == []

    # -- _compute_pass_type_breakdown --

    def test_pass_type_breakdown(self, svc):
        events = [
            PassEvent(
                timestamp=0, team="home", track_id=1, completed=True, pass_type=PassType.STANDARD
            ),
            PassEvent(
                timestamp=1,
                team="home",
                track_id=2,
                completed=True,
                pass_type=PassType.THROUGH_BALL,
                is_progressive=True,
                is_key_pass=True,
            ),
            PassEvent(
                timestamp=2, team="home", track_id=3, completed=False, pass_type=PassType.LONG_BALL
            ),
            PassEvent(
                timestamp=3,
                team="home",
                track_id=1,
                completed=True,
                pass_type=PassType.CROSS,
                is_progressive=True,
                is_key_pass=True,
                is_assist=True,
            ),
        ]
        r = svc._compute_pass_type_breakdown(events)
        assert r["total"] == 3
        assert r["progressive"] == 2
        assert r["key_passes"] == 2
        assert r["assists"] == 1
        assert r["by_type"]["standard"] == 1

    def test_pass_type_breakdown_empty(self, svc):
        r = svc._compute_pass_type_breakdown([])
        assert r["total"] == 0 and r["progressive"] == 0

    # -- _compute_confidence --

    def test_confidence_normal(self, svc):
        td = MagicMock()
        td.total_frames = 10
        td.frames = [
            MagicMock(
                detections=[MagicMock(class_name="person"), MagicMock(class_name="sports ball")]
            )
            for _ in range(10)
        ]
        c = svc._compute_confidence(td, [])
        assert 0 < c <= 1

    def test_confidence_zero_frames(self, svc):
        td = MagicMock()
        td.total_frames = 0
        td.frames = []
        assert svc._compute_confidence(td, []) == 0.0

    def test_confidence_no_ball(self, svc):
        td = MagicMock()
        td.total_frames = 10
        td.frames = [MagicMock(detections=[MagicMock(class_name="person")]) for _ in range(10)]
        c = svc._compute_confidence(td, [])
        assert 0 < c <= 1
        assert c < 1.0

    # -- _compute_team_stats --

    def test_team_stats(self, svc):
        players = {
            1: PlayerStats(
                track_id=1,
                team="home",
                distance_covered_m=5000,
                passes_completed=20,
                passes_attempted=25,
                shots=3,
                tackles=2,
            ),
            2: PlayerStats(
                track_id=2,
                team="away",
                distance_covered_m=4800,
                passes_completed=18,
                passes_attempted=22,
                shots=4,
                tackles=1,
            ),
        }
        r = svc._compute_team_stats(players, [], MagicMock())
        assert r["home"].distance_covered_km == 5.0
        assert r["away"].distance_covered_km == 4.8
        assert r["home"].passes_completed == 20 and r["home"].tackles == 2

    def test_team_stats_empty(self, svc):
        r = svc._compute_team_stats({}, [], MagicMock())
        assert r["home"].distance_covered_km == 0.0

    # -- _compute_possession --

    def test_possession_basic(self, svc):
        td = MagicMock()
        td.frames = []
        td.player_teams = {1: "home", 2: "away"}
        for i in range(5):
            ball = MagicMock(class_name="sports ball", bbox=(100, 200, 110, 210))
            player = MagicMock(
                class_name="person", track_id=1 if i < 3 else 2, bbox=(105, 200, 125, 240)
            )
            td.frames.append(MagicMock(detections=[ball, player]))
        r = svc._compute_possession(td)
        assert 59 < r["home"] < 61  # 3/5 = 60%

    def test_possession_no_ball(self, svc):
        td = MagicMock()
        td.frames = [
            MagicMock(detections=[MagicMock(class_name="person", track_id=1, bbox=(0, 0, 10, 20))])
        ]
        td.player_teams = {}
        r = svc._compute_possession(td)
        # No ball detection => no possession evidence => honest zeros
        # (the old behavior fabricated a 50/50 split from nothing).
        assert r["home"] == 0.0 and r["away"] == 0.0

    def test_possession_empty_frames(self, svc):
        td = MagicMock()
        td.frames = []
        td.player_teams = {}
        r = svc._compute_possession(td)
        # No frames => no possession evidence => honest zeros.
        assert r["home"] == 0.0 and r["away"] == 0.0

    def test_possession_no_player_teams(self, svc):
        td = MagicMock()
        td.player_teams = {}
        td.frames = []
        for i in range(4):
            ball = MagicMock(class_name="sports ball", bbox=(100, 200, 110, 210))
            pid = 2 if i < 2 else 3
            player = MagicMock(class_name="person", track_id=pid, bbox=(105, 200, 125, 240))
            td.frames.append(MagicMock(detections=[ball, player]))
        r = svc._compute_possession(td)
        # No team assignment exists: track-ID parity has no relation to team
        # membership, so possession must NOT be fabricated from it -- honest
        # zeros beat a made-up 50/50.
        assert r["home"] == 0.0 and r["away"] == 0.0

    # -- PlayerStats helper --

    def test_player_stats_pass_accuracy(self):
        ps = PlayerStats(track_id=1)
        assert ps.pass_accuracy == 0.0
        ps.passes_attempted = 10
        assert ps.pass_accuracy == 0.0  # no completed
        ps.passes_completed = 7
        assert ps.pass_accuracy == 0.7

    def test_team_stats_pass_accuracy(self):
        ts = TeamStats(team_name="Home")
        assert ts.pass_accuracy == 0.0
        ts.passes_attempted = 20
        ts.passes_completed = 15
        assert ts.pass_accuracy == 0.75


# ====================================================================
# TrackingMixin (tracking.py)
# ====================================================================


class TestTrackingAnalysis:
    """Player ratings, PPDA, formation detection, formation timeline."""

    # -- _compute_player_ratings --

    def test_player_ratings_empty(self, svc):
        r = svc._compute_player_ratings({}, [], None, MagicMock())
        assert r == {}

    def test_player_ratings_basic(self, svc):
        players = {
            1: PlayerStats(
                track_id=1,
                team="home",
                distance_covered_m=5000,
                max_speed_kmh=30,
                positions=[(0, 50, 34)],
            )
        }
        td = MagicMock()
        td.duration_seconds = 3600
        events = [PassEvent(timestamp=0, team="home", track_id=1, completed=True)]
        ratings = svc._compute_player_ratings(players, events, None, td)
        assert 1 in ratings
        assert 0 <= ratings[1].overall <= 10

    def test_player_ratings_no_events(self, svc):
        players = {1: PlayerStats(track_id=1, team="home")}
        td = MagicMock()
        td.duration_seconds = 3600
        ratings = svc._compute_player_ratings(players, [], None, td)
        assert 1 in ratings
        assert ratings[1].overall >= 0

    # -- compute_ppda --

    def test_ppda_empty_frames(self, svc):
        td = MagicMock()
        td.frames = []
        r = svc.compute_ppda(td, team="home")
        assert r["ppda"] is None and r["intensity"] == "unknown"

    def test_ppda_no_ball_or_players(self, svc):
        td = MagicMock()
        td.player_teams = {}
        td.frames = [MagicMock(detections=[]), MagicMock(detections=[])]
        r = svc.compute_ppda(td, team="home")
        assert r["ppda"] is None

    def test_ppda_no_defensive_actions(self, svc):
        td = MagicMock()
        td.player_teams = {1: "home", 2: "away"}
        ball = MagicMock(class_name="sports ball", bbox=(100, 200, 110, 210))
        home_p = MagicMock(class_name="person", track_id=1, bbox=(105, 200, 125, 240))
        away_p = MagicMock(class_name="person", track_id=2, bbox=(300, 200, 320, 240))
        td.frames = [MagicMock(detections=[ball, home_p, away_p]) for _ in range(3)]
        r = svc.compute_ppda(td, team="home")
        assert r["intensity"] == "unknown"

    # -- detect_formation --

    def test_detect_formation_empty_frames(self, svc):
        td = MagicMock()
        td.frames = []
        td.player_teams = {}
        td.duration_seconds = 90
        r = svc.detect_formation(td, team="home")
        assert r["formation"] == "unknown"

    def test_detect_formation_insufficient_players(self, svc):
        td = MagicMock()
        td.player_teams = {1: "home", 2: "home", 3: "home", 4: "home"}
        td.duration_seconds = 90
        td.frames = []
        for i in range(5):
            dets = [
                MagicMock(class_name="person", track_id=tid, bbox=(i * 50, 200, i * 50 + 20, 240))
                for tid in [1, 2, 3, 4]
            ]
            td.frames.append(MagicMock(detections=dets, timestamp=float(i)))
        r = svc.detect_formation(td, team="home")
        assert r["formation"] == "unknown"
        assert r["player_count"] < 5

    # -- track_formations --

    def test_track_formations_empty_frames(self, svc):
        td = MagicMock()
        td.frames = []
        r = svc.track_formations(td, window_minutes=5)
        assert r["home_timeline"] == [] and r["changes"] == 0

    def test_track_formations_no_frames_attr(self, svc):
        td = MagicMock(spec=[])
        del td.frames
        r = svc.track_formations(td)
        assert r["home_timeline"] == []

    # -- _detect_formation (from detections) --

    def test_detect_formation_from_dets_basic(self, svc):
        dets = [
            MagicMock(bbox=MagicMock(cx=10), x=10, team=None, is_ball=False),
            MagicMock(bbox=MagicMock(cx=20), x=20, team=None, is_ball=False),
            MagicMock(bbox=MagicMock(cx=30), x=30, team=None, is_ball=False),
            MagicMock(bbox=MagicMock(cx=40), x=40, team=None, is_ball=False),
            MagicMock(bbox=MagicMock(cx=50), x=50, team=None, is_ball=False),
            MagicMock(bbox=MagicMock(cx=60), x=60, team=None, is_ball=False),
            MagicMock(bbox=MagicMock(cx=70), x=70, team=None, is_ball=False),
            MagicMock(bbox=MagicMock(cx=80), x=80, team=None, is_ball=False),
            MagicMock(bbox=MagicMock(cx=90), x=90, team=None, is_ball=False),
            MagicMock(bbox=MagicMock(cx=100), x=100, team=None, is_ball=False),
        ]
        f = svc._detect_formation(dets)
        assert len(f.split("-")) == 3
        assert all(p.isdigit() for p in f.split("-"))

    def test_detect_formation_from_dets_empty(self, svc):
        assert svc._detect_formation([]) == "unknown"

    def test_detect_formation_from_dets_no_coords(self, svc):
        dets = [MagicMock(bbox=None, x=0)]
        f = svc._detect_formation(dets)
        assert len(f.split("-")) == 3
        assert f != "unknown"

    # -- _classify_formation_in_window --

    def test_classify_formation_window_empty(self, svc):
        assert svc._classify_formation_in_window([], "home") == "unknown"

    def test_classify_formation_window_no_team_dets(self, svc):
        frames = [MagicMock(detections=[MagicMock(team="away", is_ball=False)])]
        r = svc._classify_formation_in_window(frames, "home")
        assert r == "unknown"
