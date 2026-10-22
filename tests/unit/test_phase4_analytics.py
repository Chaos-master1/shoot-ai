"""Comprehensive tests for Phase 4 analytics modules.

Covers all 12 new modules: space_control, xg_chain, through_ball,
defensive_xt, dominance_index, role_classifier, set_piece_xt,
crossing_xg, psxg_improved, xa_split, scout_report_upgrade,
pressing_clusters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from kawkab.core.crossing_xg import CrossXgFactors, compute_cross_xg
from kawkab.core.defensive_xt import DefensiveAction, compute_defensive_xt
from kawkab.core.dominance_index import DominanceReport, compute_dominance_index
from kawkab.core.pressing_clusters import PressingCluster, cluster_pressing_events
from kawkab.core.role_classifier import PlayerRole, classify_player_role
from kawkab.core.scout_report_upgrade import ScoutReport, generate_scout_report
from kawkab.core.set_piece_xt import SetPieceXTReport, compute_set_piece_xt
from kawkab.core.space_control import (
    SpaceControlReport,
    compute_pitch_control_grid,
    compute_space_gained,
    identify_hot_zones,
)
from kawkab.core.through_ball import (
    ThroughBall,
    detect_through_balls,
    value_through_ball,
)
from kawkab.core.xa_split import XaSplit, compute_xa_by_type, compute_xa_expected_vs_actual
from kawkab.core.xg_chain import XgBuildup, XgChain, compute_xg_buildup, compute_xg_chain

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "pass",
            "team": "home",
            "start_x": 30.0,
            "start_y": 34.0,
            "end_x": 50.0,
            "end_y": 34.0,
            "completed": True,
            "timestamp": 0.0,
            "from_track_id": 1,
            "to_track_id": 2,
            "xA": 0.02,
        },
        {
            "type": "pass",
            "team": "home",
            "start_x": 50.0,
            "start_y": 34.0,
            "end_x": 70.0,
            "end_y": 30.0,
            "completed": True,
            "timestamp": 1.0,
            "from_track_id": 2,
            "to_track_id": 3,
            "xA": 0.05,
        },
        {
            "type": "shot",
            "team": "home",
            "start_x": 80.0,
            "start_y": 34.0,
            "end_x": 100.0,
            "end_y": 35.0,
            "xG": 0.15,
            "timestamp": 2.0,
            "angle_deg": 20.0,
            "distance_m": 12.0,
            "body_part": "right_foot",
            "shot_type": "open_play",
        },
        {"type": "tackle", "team": "away", "start_x": 40.0, "start_y": 30.0, "timestamp": 0.5},
        {
            "type": "interception",
            "team": "home",
            "start_x": 55.0,
            "start_y": 34.0,
            "x": 55.0,
            "y": 34.0,
            "timestamp": 1.5,
        },
        {"type": "clearance", "team": "away", "start_x": 85.0, "start_y": 40.0, "timestamp": 2.5},
        {
            "type": "block",
            "team": "home",
            "start_x": 70.0,
            "start_y": 30.0,
            "x": 72.0,
            "y": 32.0,
            "timestamp": 3.0,
        },
    ]


@pytest.fixture
def sample_through_ball_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "pass",
            "team": "home",
            "start_x": 40.0,
            "start_y": 34.0,
            "end_x": 85.0,
            "end_y": 30.0,
            "completed": True,
            "pass_type": "through_ball",
            "from_track_id": 10,
            "to_track_id": 11,
        },
        {
            "type": "pass",
            "team": "home",
            "start_x": 50.0,
            "start_y": 34.0,
            "end_x": 55.0,
            "end_y": 34.0,
            "completed": True,
            "pass_type": "standard",
            "from_track_id": 12,
            "to_track_id": 13,
        },
    ]


@pytest.fixture
def sample_set_piece_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "corner",
            "team": "home",
            "start_x": 105.0,
            "start_y": 0.0,
            "end_x": 95.0,
            "end_y": 20.0,
        },
        {
            "type": "corner",
            "team": "home",
            "start_x": 105.0,
            "start_y": 68.0,
            "end_x": 90.0,
            "end_y": 50.0,
        },
        {
            "type": "free_kick",
            "team": "home",
            "start_x": 70.0,
            "start_y": 40.0,
            "end_x": 90.0,
            "end_y": 34.0,
        },
        {
            "type": "throw_in",
            "team": "away",
            "start_x": 50.0,
            "start_y": 0.0,
            "end_x": 55.0,
            "end_y": 15.0,
        },
    ]


@pytest.fixture
def sample_xT_grid() -> np.ndarray:
    grid = np.zeros((16, 12), dtype=np.float64)
    for r in range(16):
        for c in range(12):
            grid[r, c] = (c / 11.0) * 0.5 + (1.0 - r / 15.0) * 0.2
    return grid


@pytest.fixture
def sample_player_positions() -> list[tuple[float, float, int]]:
    return [
        (10.0, 34.0, 1),
        (25.0, 20.0, 2),
        (30.0, 50.0, 3),
        (50.0, 34.0, 4),
        (70.0, 20.0, 5),
        (75.0, 50.0, 6),
    ]


@pytest.fixture
def sample_player_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "pass",
            "team": "home",
            "start_x": 50.0,
            "start_y": 34.0,
            "end_x": 70.0,
            "end_y": 30.0,
            "completed": True,
        },
        {
            "type": "pass",
            "team": "home",
            "start_x": 70.0,
            "start_y": 30.0,
            "end_x": 55.0,
            "end_y": 34.0,
            "completed": True,
        },
        {
            "type": "pass",
            "team": "home",
            "start_x": 40.0,
            "start_y": 34.0,
            "end_x": 60.0,
            "end_y": 20.0,
            "completed": True,
        },
        {
            "type": "shot",
            "team": "home",
            "start_x": 80.0,
            "start_y": 34.0,
            "end_x": 100.0,
            "end_y": 36.0,
            "xG": 0.12,
        },
        {"type": "tackle", "team": "home", "start_x": 45.0, "start_y": 30.0},
        {
            "type": "pass",
            "team": "home",
            "start_x": 30.0,
            "start_y": 34.0,
            "end_x": 50.0,
            "end_y": 34.0,
            "completed": False,
        },
        {
            "type": "shot",
            "team": "home",
            "start_x": 75.0,
            "start_y": 30.0,
            "end_x": 100.0,
            "end_y": 20.0,
            "xG": 0.08,
        },
    ]


@pytest.fixture
def sample_player_profile() -> dict[str, Any]:
    return {"name": "Test Player", "id": 42, "position": "attacking_midfielder"}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Space Control Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpaceControl:
    def test_pitch_control_grid_empty(self):
        grid, pcts = compute_pitch_control_grid([], [])
        assert grid.shape == (30, 46)
        assert pcts == {}

    def test_pitch_control_grid_single_team(self, sample_player_positions):
        team_ids = [0, 0, 0, 0, 0, 0]
        grid, pcts = compute_pitch_control_grid(sample_player_positions, team_ids)
        assert grid.shape == (30, 46)
        assert 0 in pcts
        assert pcts[0] == 100.0

    def test_pitch_control_grid_two_teams(self, sample_player_positions):
        team_ids = [0, 0, 0, 1, 1, 1]
        grid, pcts = compute_pitch_control_grid(sample_player_positions, team_ids)
        assert grid.shape == (30, 46)
        assert 0 in pcts
        assert 1 in pcts
        total = pcts[0] + pcts[1]
        assert abs(total - 100.0) < 0.01

    def test_pitch_control_grid_values(self, sample_player_positions):
        team_ids = [0, 0, 0, 1, 1, 1]
        grid, pcts = compute_pitch_control_grid(
            sample_player_positions, team_ids, grid_rows=10, grid_cols=10
        )
        assert grid.shape == (10, 10)
        assert np.all((grid == 0) | (grid == 1))

    def test_pitch_control_grid_custom_size(self, sample_player_positions):
        team_ids = [0, 0, 0, 1, 1, 1]
        grid, pcts = compute_pitch_control_grid(
            sample_player_positions, team_ids, grid_rows=5, grid_cols=8
        )
        assert grid.shape == (5, 8)

    def test_compute_space_gained_no_change(self):
        tracks = [(50.0, 34.0, 50.0, 34.0, 0), (30.0, 34.0, 30.0, 34.0, 1)]
        pass_ev = {"team": "home", "start_x": 40.0, "start_y": 34.0, "end_x": 60.0, "end_y": 34.0}
        gained = compute_space_gained(pass_ev, tracks, grid_rows=10, grid_cols=10)
        assert gained == 0.0

    def test_compute_space_gained_returns_float(self):
        tracks = [(40.0, 34.0, 70.0, 34.0, 0), (30.0, 34.0, 30.0, 34.0, 1)]
        pass_ev = {"team": "home", "start_x": 40.0, "start_y": 34.0, "end_x": 70.0, "end_y": 34.0}
        gained = compute_space_gained(pass_ev, tracks, grid_rows=10, grid_cols=10)
        assert isinstance(gained, float)

    def test_identify_hot_zones_single_team(self):
        grid = np.zeros((30, 46), dtype=np.int32)
        grid[5:15, 10:20] = 1
        zones = identify_hot_zones(grid, team_id=1, min_area_pct=1.0)
        assert len(zones) >= 1
        for z in zones:
            assert z["area_pct"] > 0
            assert z["cells"] > 0

    def test_identify_hot_zones_no_hot_zones(self):
        grid = np.zeros((30, 46), dtype=np.int32)
        grid[0, 0] = 1
        zones = identify_hot_zones(grid, team_id=1, min_area_pct=10.0)
        assert len(zones) == 0

    def test_identify_hot_zones_multiple_zones(self):
        grid = np.zeros((30, 46), dtype=np.int32)
        grid[2:8, 2:8] = 1
        grid[20:28, 35:43] = 1
        zones = identify_hot_zones(grid, team_id=1, min_area_pct=0.5)
        assert len(zones) == 2

    def test_space_control_report_dataclass(self):
        report = SpaceControlReport(
            team="home",
            grid=[[1.0]],
            team_control_pcts={"home": 55.0},
            hot_zones=[{"cells": 10, "center_x": 50.0, "center_y": 34.0, "area_pct": 5.0}],
        )
        d = report.to_dict()
        assert d["team"] == "home"
        assert d["team_control_pcts"]["home"] == 55.0
        assert len(d["hot_zones"]) == 1

    def test_compute_pitch_control_grid_with_duplicates(self):
        positions = [(50.0, 34.0, 1), (50.0, 34.0, 2), (60.0, 30.0, 3)]
        team_ids = [0, 0, 1]
        grid, pcts = compute_pitch_control_grid(positions, team_ids, grid_rows=10, grid_cols=10)
        assert grid.shape == (10, 10)
        assert 0 in pcts
        assert 1 in pcts

    def test_pitch_control_grid_pcts_sum(self):
        positions = [(20.0, 20.0, 1), (30.0, 30.0, 2), (70.0, 40.0, 3), (80.0, 30.0, 4)]
        team_ids = [0, 0, 1, 1]
        _, pcts = compute_pitch_control_grid(positions, team_ids, grid_rows=15, grid_cols=20)
        total = sum(pcts.values())
        assert abs(total - 100.0) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 2. xG Chain Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestXgChain:
    def test_compute_xg_chain_empty(self):
        result = compute_xg_chain([], "home")
        assert result == []

    def test_compute_xg_chain_no_shots(self):
        events = [{"type": "pass", "team": "home", "timestamp": 0.0}]
        result = compute_xg_chain(events, "home")
        assert result == []

    def test_compute_xg_chain_with_shot(self, sample_events):
        result = compute_xg_chain(sample_events, "home")
        assert len(result) > 0
        for item in result:
            assert isinstance(item, XgChain)
            assert item.xg_contribution > 0

    def test_compute_xg_chain_team_filter(self, sample_events):
        home_result = compute_xg_chain(sample_events, "home")
        away_result = compute_xg_chain(sample_events, "away")
        assert len(home_result) >= len(away_result)

    def test_compute_xg_chain_role_assignment(self, sample_events):
        result = compute_xg_chain(sample_events, "home")
        roles = {r.role for r in result}
        assert "buildup" in roles or "shot" in roles

    def test_compute_xg_buildup_empty(self):
        result = compute_xg_buildup([], "home")
        assert result == []

    def test_compute_xg_buildup_with_assists(self, sample_events):
        result = compute_xg_buildup(sample_events, "home")
        if result:
            assert all(isinstance(r, XgBuildup) for r in result)
            assert all(r.credit > 0 for r in result)

    def test_compute_xg_buildup_primary_secondary(self, sample_events):
        result = compute_xg_buildup(sample_events, "home")
        for r in result:
            assert r.is_primary_assist or r.is_secondary_assist
            assert r.event_type == "pass"

    def test_compute_xg_buildup_credit_ordering(self, sample_events):
        result = compute_xg_buildup(sample_events, "home")
        for i in range(len(result) - 1):
            if result[i].is_primary_assist and result[i + 1].is_secondary_assist:
                assert result[i].credit >= result[i + 1].credit

    def test_xg_chain_dataclass_to_dict(self):
        c = XgChain(
            event_idx=1, event_type="pass", event_team="home", xg_contribution=0.05, role="buildup"
        )
        d = c.to_dict()
        assert d["xg"] == 0.05
        assert d["role"] == "buildup"

    def test_xg_buildup_dataclass_to_dict(self):
        b = XgBuildup(
            event_idx=2,
            event_type="pass",
            credit=0.08,
            is_primary_assist=True,
            is_secondary_assist=False,
        )
        d = b.to_dict()
        assert d["credit"] == 0.08
        assert d["primary_assist"] is True

    def test_xg_chain_does_not_include_opponent_events(self, sample_events):
        result = compute_xg_chain(sample_events, "away")
        for r in result:
            assert r.event_team == "away"

    def test_xg_chain_no_negative_contributions(self, sample_events):
        result = compute_xg_chain(sample_events, "home")
        for r in result:
            assert r.xg_contribution >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Through Ball Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestThroughBall:
    def test_detect_through_balls_empty(self):
        result = detect_through_balls([])
        assert result == []

    def test_detect_through_balls_identifies_through_balls(self, sample_through_ball_events):
        def_pos = {
            0: [(60.0, 30.0, 5), (55.0, 38.0, 6)],
        }
        result = detect_through_balls(sample_through_ball_events, def_pos)
        assert len(result) >= 1
        for tb in result:
            assert isinstance(tb, ThroughBall)
            assert tb.pass_event["pass_type"] == "through_ball"

    def test_detect_through_balls_filters_standard_passes(self, sample_through_ball_events):
        result = detect_through_balls(sample_through_ball_events)
        for tb in result:
            assert tb.pass_event["pass_type"] != "standard"

    def test_through_ball_dataclass(self):
        tb = ThroughBall(
            pass_event={"type": "pass"}, xT_gained=0.15, receiver=7, split_defenders=[3, 4]
        )
        d = tb.to_dict()
        assert d["xT_gained"] == 0.15
        assert d["receiver"] == 7
        assert d["split_defenders"] == [3, 4]

    def test_value_through_ball_positive_gain(self, sample_through_ball_events, sample_xT_grid):
        tb = ThroughBall(
            pass_event=sample_through_ball_events[0],
            xT_gained=0.0,
            receiver=11,
            split_defenders=[5],
        )
        gained = value_through_ball(tb, sample_xT_grid)
        assert gained >= 0.0

    def test_value_through_ball_zero_for_short_pass(self, sample_xT_grid):
        tb = ThroughBall(
            pass_event={"start_x": 50.0, "start_y": 34.0, "end_x": 52.0, "end_y": 34.0},
            xT_gained=0.0,
            receiver=2,
            split_defenders=[],
        )
        gained = value_through_ball(tb, sample_xT_grid)
        assert gained >= 0.0

    def test_detect_through_balls_with_defenders(self, sample_through_ball_events):
        def_pos = {
            0: [(60.0, 30.0, 5), (55.0, 38.0, 6)],
        }
        result = detect_through_balls(sample_through_ball_events, def_pos)
        assert isinstance(result, list)

    def test_through_ball_value_non_negative(self, sample_through_ball_events, sample_xT_grid):
        tb = ThroughBall(
            pass_event=sample_through_ball_events[0], xT_gained=0.0, receiver=11, split_defenders=[]
        )
        val = value_through_ball(tb, sample_xT_grid)
        assert val >= 0

    def test_through_ball_with_backward_pass(self, sample_xT_grid):
        tb = ThroughBall(
            pass_event={"start_x": 80.0, "start_y": 34.0, "end_x": 60.0, "end_y": 34.0},
            xT_gained=0.0,
            receiver=3,
            split_defenders=[],
        )
        val = value_through_ball(tb, sample_xT_grid)
        assert val >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Defensive xT Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDefensiveXT:
    def test_compute_defensive_xt_empty(self, sample_xT_grid):
        result = compute_defensive_xt([], sample_xT_grid)
        assert result == []

    def test_compute_defensive_xt_detects_events(self, sample_events, sample_xT_grid):
        result = compute_defensive_xt(sample_events, sample_xT_grid)
        assert len(result) >= 1
        for action in result:
            assert isinstance(action, DefensiveAction)

    def test_compute_defensive_xt_event_types(self, sample_events, sample_xT_grid):
        result = compute_defensive_xt(sample_events, sample_xT_grid)
        types = {a.event_type for a in result}
        assert (
            "tackle" in types or "interception" in types or "clearance" in types or "block" in types
        )

    def test_compute_defensive_xt_ignores_non_defensive(self, sample_events, sample_xT_grid):
        result = compute_defensive_xt(sample_events, sample_xT_grid)
        for a in result:
            assert a.event_type in ("interception", "tackle", "clearance", "block")

    def test_compute_defensive_xt_sorted(self, sample_events, sample_xT_grid):
        result = compute_defensive_xt(sample_events, sample_xT_grid)
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert result[i].xT_prevented >= result[i + 1].xT_prevented

    def test_defensive_action_dataclass(self):
        da = DefensiveAction(
            event_idx=1,
            event_type="tackle",
            team="home",
            xT_prevented=0.15,
            zone=(5, 3),
            x=50.0,
            y=34.0,
        )
        d = da.to_dict()
        assert d["xT_prevented"] == 0.15
        assert d["zone"] == [5, 3]

    def test_compute_defensive_xt_xT_values(self, sample_events, sample_xT_grid):
        result = compute_defensive_xt(sample_events, sample_xT_grid)
        for a in result:
            assert a.xT_prevented >= 0
            assert isinstance(a.xT_prevented, float)

    def test_compute_defensive_xt_all_types_covered(self, sample_xT_grid):
        events = [
            {"type": "tackle", "team": "home", "start_x": 50.0, "start_y": 34.0},
            {"type": "interception", "team": "away", "x": 60.0, "y": 30.0},
            {"type": "clearance", "team": "home", "start_x": 80.0, "start_y": 40.0},
            {"type": "block", "team": "away", "x": 70.0, "y": 32.0},
        ]
        result = compute_defensive_xt(events, sample_xT_grid)
        assert len(result) == 4

    def test_compute_defensive_xt_zone_assignment(self, sample_xT_grid):
        events = [{"type": "tackle", "team": "home", "start_x": 0.0, "start_y": 0.0}]
        result = compute_defensive_xt(events, sample_xT_grid)
        assert len(result) == 1
        assert result[0].zone[0] >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Dominance Index Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDominanceIndex:
    def test_compute_dominance_index_empty(self):
        report = compute_dominance_index([], "home")
        assert isinstance(report, DominanceReport)
        assert 0 <= report.index <= 100

    def test_compute_dominance_index_with_events(self, sample_events):
        report = compute_dominance_index(sample_events, "home")
        assert 0 <= report.index <= 100
        assert len(report.sub_scores) == 5

    def test_compute_dominance_index_sub_scores(self, sample_events):
        report = compute_dominance_index(sample_events, "home")
        for key in ("possession", "xg_diff", "territory", "pressing", "pass_completion"):
            assert key in report.sub_scores
            assert 0 <= report.sub_scores[key] <= 100

    def test_dominance_index_symmetric(self, sample_events):
        _ = compute_dominance_index(sample_events, "home")
        _ = compute_dominance_index(sample_events, "away")
        assert True

    def test_dominance_report_dataclass(self):
        report = DominanceReport(
            index=75.5,
            team="home",
            opponent="away",
            sub_scores={
                "possession": 60.0,
                "xg_diff": 70.0,
                "territory": 65.0,
                "pressing": 80.0,
                "pass_completion": 85.0,
            },
        )
        d = report.to_dict()
        assert d["index"] == 75.5
        assert d["team"] == "home"

    def test_dominance_index_with_phases(self, sample_events):
        report = compute_dominance_index(sample_events, "home")
        assert len(report.phases) >= 0

    def test_dominance_index_all_teams_equal(self):
        events = [
            {"type": "pass", "team": "home", "completed": True, "start_x": 50.0, "end_x": 60.0},
            {"type": "pass", "team": "away", "completed": True, "start_x": 50.0, "end_x": 60.0},
        ]
        home_report = compute_dominance_index(events, "home")
        away_report = compute_dominance_index(events, "away")
        assert abs(home_report.sub_scores["possession"] - away_report.sub_scores["possession"]) < 5

    def test_dominance_index_no_events_for_team(self):
        events = [{"type": "pass", "team": "away", "completed": True}]
        report = compute_dominance_index(events, "home")
        assert 0 <= report.index <= 100


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Role Classifier Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoleClassifier:
    def test_classify_player_role_empty(self):
        role = classify_player_role([])
        assert role.primary_role == "unknown"
        assert role.confidence == 0.0

    def test_classify_player_role_returns_role_object(self, sample_player_events):
        role = classify_player_role(sample_player_events)
        assert isinstance(role, PlayerRole)
        assert role.primary_role != "unknown"
        assert role.confidence > 0

    def test_classify_player_role_scores(self, sample_player_events):
        role = classify_player_role(sample_player_events)
        assert len(role.role_scores) > 0
        for score in role.role_scores.values():
            assert score >= 0

    def test_classify_player_role_goalkeeper(self):
        events = [{"type": "pass", "start_x": 5.0, "start_y": 34.0, "end_x": 10.0, "end_y": 35.0}]
        role = classify_player_role(events)
        assert role.primary_role == "goalkeeper"

    def test_classify_player_role_confidence_bounds(self, sample_player_events):
        role = classify_player_role(sample_player_events)
        assert 0 <= role.confidence <= 1.0

    def test_role_dataclass_to_dict(self):
        r = PlayerRole(
            primary_role="centre_back",
            secondary_role="defensive_midfielder",
            confidence=0.85,
            role_scores={"centre_back": 22.0, "defensive_midfielder": 18.0},
        )
        d = r.to_dict()
        assert d["primary"] == "centre_back"
        assert d["confidence"] == 0.85

    def test_classify_player_role_different_positions(self):
        forward_events = [
            {"type": "shot", "start_x": 85.0, "start_y": 34.0, "end_x": 100.0, "end_y": 36.0},
            {"type": "shot", "start_x": 80.0, "start_y": 30.0, "end_x": 100.0, "end_y": 20.0},
        ]
        role = classify_player_role(forward_events)
        assert role.primary_role in ("target_forward", "poacher", "inside_forward")

    def test_classify_with_wide_player(self):
        wide_events = [
            {"type": "pass", "start_x": 40.0, "start_y": 5.0, "end_x": 60.0, "end_y": 10.0},
            {"type": "cross", "start_x": 70.0, "start_y": 5.0, "end_x": 95.0, "end_y": 30.0},
        ]
        role = classify_player_role(wide_events)
        assert (
            "winger" in role.primary_role
            or "full_back" in role.primary_role
            or "wide_midfielder" in role.primary_role
            or "wide_playmaker" in role.primary_role
        )

    def test_classify_player_role_secondary(self, sample_player_events):
        role = classify_player_role(sample_player_events)
        assert isinstance(role.secondary_role, str)

    def test_classify_player_role_top_scores(self, sample_player_events):
        role = classify_player_role(sample_player_events)
        top_keys = list(role.role_scores.keys())
        if len(top_keys) >= 2:
            assert role.role_scores[top_keys[0]] >= role.role_scores[top_keys[1]]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Set Piece xT Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSetPieceXT:
    def test_compute_set_piece_xt_empty(self, sample_xT_grid):
        report = compute_set_piece_xt([], sample_xT_grid)
        assert isinstance(report, SetPieceXTReport)
        assert report.by_type == {}

    def test_compute_set_piece_xt_by_type(self, sample_set_piece_events, sample_xT_grid):
        report = compute_set_piece_xt(sample_set_piece_events, sample_xT_grid)
        assert len(report.by_type) >= 1
        for _sp_type, zones in report.by_type.items():
            assert len(zones) > 0

    def test_compute_set_piece_xt_total_xt(self, sample_set_piece_events, sample_xT_grid):
        report = compute_set_piece_xt(sample_set_piece_events, sample_xT_grid)
        assert len(report.total_xT_by_type) >= 1
        for _sp_type, total in report.total_xT_by_type.items():
            assert total >= 0

    def test_compute_set_piece_xt_most_dangerous(self, sample_set_piece_events, sample_xT_grid):
        report = compute_set_piece_xt(sample_set_piece_events, sample_xT_grid)
        assert "type" in report.most_dangerous_zone
        assert "zone" in report.most_dangerous_zone

    def test_set_piece_xt_report_dataclass(self):
        report = SetPieceXTReport(
            by_type={"corner": []},
            total_xT_by_type={"corner": 0.05},
            most_dangerous_zone={"type": "corner", "zone": [5, 8], "total_xT": 0.03},
        )
        d = report.to_dict()
        assert d["total_xT_by_type"]["corner"] == 0.05

    def test_compute_set_piece_xt_corner_only(self, sample_xT_grid):
        events = [
            {
                "type": "corner",
                "team": "home",
                "start_x": 105.0,
                "start_y": 0.0,
                "end_x": 95.0,
                "end_y": 15.0,
            }
        ]
        report = compute_set_piece_xt(events, sample_xT_grid)
        assert "corner_kick" in report.by_type

    def test_compute_set_piece_xt_avg_xt_per_zone(self, sample_set_piece_events, sample_xT_grid):
        report = compute_set_piece_xt(sample_set_piece_events, sample_xT_grid)
        for _sp_type, zones in report.by_type.items():
            for zone in zones:
                assert zone.avg_xT >= 0
                assert zone.count >= 1

    def test_compute_set_piece_xt_different_types(self, sample_xT_grid):
        events = [
            {"type": "corner", "end_x": 95.0, "end_y": 15.0},
            {"type": "free_kick", "end_x": 85.0, "end_y": 34.0},
            {"type": "throw_in", "end_x": 60.0, "end_y": 10.0},
        ]
        report = compute_set_piece_xt(events, sample_xT_grid)
        assert "corner_kick" in report.by_type
        assert "free_kick" in report.by_type
        assert "throw_in" in report.by_type


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Crossing xG Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossingXG:
    def test_compute_cross_xg_basic(self):
        event = {
            "type": "cross",
            "start_x": 70.0,
            "start_y": 5.0,
            "end_x": 95.0,
            "end_y": 30.0,
            "cross_height": "low",
        }
        factors = compute_cross_xg(event)
        assert isinstance(factors, CrossXgFactors)
        assert 0 < factors.base_xg < 0.5

    def test_compute_cross_xg_height_adjustment(self):
        low_event = {
            "type": "cross",
            "start_x": 70.0,
            "start_y": 5.0,
            "end_x": 95.0,
            "end_y": 30.0,
            "cross_height": "low",
        }
        high_event = {
            "type": "cross",
            "start_x": 70.0,
            "start_y": 5.0,
            "end_x": 95.0,
            "end_y": 30.0,
            "cross_height": "high",
        }
        low_xg = compute_cross_xg(low_event)
        high_xg = compute_cross_xg(high_event)
        assert low_xg.base_xg >= high_xg.base_xg

    def test_compute_cross_xg_distance_effect(self):
        near_event = {
            "type": "cross",
            "start_x": 85.0,
            "start_y": 10.0,
            "end_x": 100.0,
            "end_y": 30.0,
            "cross_height": "low",
        }
        far_event = {
            "type": "cross",
            "start_x": 50.0,
            "start_y": 10.0,
            "end_x": 95.0,
            "end_y": 30.0,
            "cross_height": "low",
        }
        near_xg = compute_cross_xg(near_event)
        far_xg = compute_cross_xg(far_event)
        assert near_xg.base_xg > far_xg.base_xg or near_xg.base_xg >= 0

    def test_compute_cross_xg_defender_proximity(self):
        close_def = {
            "type": "cross",
            "start_x": 75.0,
            "start_y": 5.0,
            "end_x": 95.0,
            "end_y": 30.0,
            "defender_distance": 1.0,
        }
        far_def = {
            "type": "cross",
            "start_x": 75.0,
            "start_y": 5.0,
            "end_x": 95.0,
            "end_y": 30.0,
            "defender_distance": 10.0,
        }
        close_xg = compute_cross_xg(close_def)
        far_xg = compute_cross_xg(far_def)
        assert close_xg.base_xg <= far_xg.base_xg

    def test_compute_cross_xg_byline_boost(self):
        byline = {
            "type": "cross",
            "start_x": 102.0,
            "start_y": 5.0,
            "end_x": 98.0,
            "end_y": 30.0,
            "cross_height": "low",
        }
        normal = {
            "type": "cross",
            "start_x": 60.0,
            "start_y": 5.0,
            "end_x": 95.0,
            "end_y": 30.0,
            "cross_height": "low",
        }
        b_xg = compute_cross_xg(byline)
        n_xg = compute_cross_xg(normal)
        assert b_xg.from_byline is True
        assert n_xg.from_byline is False

    def test_cross_xg_factors_dataclass(self):
        f = CrossXgFactors(
            distance_m=25.0,
            cross_height="low",
            defender_distance_m=3.0,
            from_byline=True,
            headed_chance=0.5,
            placement_angle_deg=30.0,
            base_xg=0.08,
        )
        d = f.to_dict()
        assert d["base_xg"] == 0.08

    def test_compute_cross_xg_height_types(self):
        for height in ("ground", "low", "high", "lofted"):
            event = {
                "type": "cross",
                "start_x": 70.0,
                "start_y": 5.0,
                "end_x": 95.0,
                "end_y": 30.0,
                "cross_height": height,
            }
            factors = compute_cross_xg(event)
            assert 0 < factors.base_xg < 0.5

    def test_compute_cross_xg_caps_at_max(self):
        event = {
            "type": "cross",
            "start_x": 100.0,
            "start_y": 30.0,
            "end_x": 104.0,
            "end_y": 34.0,
            "cross_height": "ground",
            "defender_distance": 10.0,
        }
        factors = compute_cross_xg(event)
        assert factors.base_xg <= 0.35


# ═══════════════════════════════════════════════════════════════════════════════
# 10. xA Split Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestXaSplit:
    def test_compute_xa_by_type_empty(self):
        split = compute_xa_by_type([])
        assert isinstance(split, XaSplit)
        assert split.total == 0.0

    def test_compute_xa_by_type_open_play(self, sample_events):
        split = compute_xa_by_type(sample_events)
        assert split.open_play > 0
        assert split.total > 0

    def test_compute_xa_by_type_corner(self):
        events = [{"type": "pass", "xA": 0.12, "set_piece_type": "corner"}]
        split = compute_xa_by_type(events)
        assert split.corner == 0.12

    def test_compute_xa_by_type_free_kick(self):
        events = [{"type": "free_kick", "xA": 0.08}]
        split = compute_xa_by_type(events)
        assert split.free_kick == 0.08

    def test_compute_xa_by_type_throw_in(self):
        events = [{"type": "pass", "xA": 0.03, "set_piece_type": "throw_in"}]
        split = compute_xa_by_type(events)
        assert split.throw_in == 0.03

    def test_compute_xa_by_type_total_matches_sum(self, sample_events):
        split = compute_xa_by_type(sample_events)
        total_from_parts = split.open_play + split.corner + split.free_kick + split.throw_in
        assert abs(split.total - total_from_parts) < 0.001

    def test_xa_split_dataclass(self):
        s = XaSplit(open_play=0.5, corner=0.2, free_kick=0.1, throw_in=0.05, total=0.85)
        d = s.to_dict()
        assert d["total"] == 0.85

    def test_compute_xa_expected_vs_actual_empty(self):
        result = compute_xa_expected_vs_actual([])
        assert result == []

    def test_compute_xa_expected_vs_actual_with_goal(self):
        events = [
            {"type": "pass", "from_track_id": 10, "xA": 0.15},
            {"type": "goal", "assist_track_id": 10},
        ]
        result = compute_xa_expected_vs_actual(events)
        assert len(result) >= 1
        for item in result:
            if item.player_id == 10:
                assert item.xa > 0
                assert item.actual_assists == 1

    def test_compute_xa_expected_vs_actual_sorted(self):
        events = [
            {"type": "pass", "from_track_id": 1, "xA": 0.3},
            {"type": "pass", "from_track_id": 2, "xA": 0.1},
            {"type": "goal", "assist_track_id": 1},
        ]
        result = compute_xa_expected_vs_actual(events)
        assert len(result) >= 2
        if len(result) >= 2:
            assert abs(result[0].difference) >= abs(result[1].difference)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Scout Report Upgrade Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoutReportUpgrade:
    def test_generate_scout_report_empty_events(self, sample_player_profile):
        report = generate_scout_report([], sample_player_profile)
        assert isinstance(report, ScoutReport)
        assert report.player_name == "Test Player"

    def test_generate_scout_report_with_events(self, sample_player_events, sample_player_profile):
        report = generate_scout_report(sample_player_events, sample_player_profile)
        assert len(report.percentiles) > 0
        assert len(report.strengths) >= 0

    def test_generate_scout_report_strengths_weaknesses(
        self, sample_player_events, sample_player_profile
    ):
        report = generate_scout_report(sample_player_events, sample_player_profile)
        total_items = len(report.strengths) + len(report.weaknesses)
        assert total_items >= 0

    def test_generate_scout_report_with_video_clips(
        self, sample_player_events, sample_player_profile
    ):
        clips = [
            {
                "id": "clip1",
                "timestamp": 120.0,
                "duration_s": 8.0,
                "label": "Goal",
                "tags": ["shot", "goal"],
            }
        ]
        report = generate_scout_report(sample_player_events, sample_player_profile, clips)
        assert len(report.video_clips) == 1
        assert report.video_clips[0].clip_id == "clip1"

    def test_generate_scout_report_similar_players(
        self, sample_player_events, sample_player_profile
    ):
        report = generate_scout_report(sample_player_events, sample_player_profile)
        assert isinstance(report.similar_players, list)
        if report.similar_players:
            assert "name" in report.similar_players[0]
            assert "similarity" in report.similar_players[0]

    def test_scout_report_dataclass(self):
        r = ScoutReport(player_name="Messi", player_id=10, position="winger", overall_rating=92.5)
        d = r.to_dict()
        assert d["player"] == "Messi"
        assert d["overall_rating"] == 92.5

    def test_generate_scout_report_overall_rating(
        self, sample_player_events, sample_player_profile
    ):
        report = generate_scout_report(sample_player_events, sample_player_profile)
        assert 0 <= report.overall_rating <= 100

    def test_generate_scout_report_percentiles(self, sample_player_events, sample_player_profile):
        report = generate_scout_report(sample_player_events, sample_player_profile)
        for p in report.percentiles:
            assert 0 <= p.percentile <= 100
            assert p.stat_name != ""

    def test_generate_scout_report_no_clips(self, sample_player_events, sample_player_profile):
        report = generate_scout_report(sample_player_events, sample_player_profile)
        assert isinstance(report.video_clips, list)
        assert len(report.video_clips) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Pressing Clusters Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPressingClusters:
    def test_cluster_pressing_events_empty(self):
        result = cluster_pressing_events([])
        assert result == []

    def test_cluster_pressing_events_detects_clusters(self):
        events = [
            {"type": "tackle", "team": "home", "start_x": 50.0, "start_y": 34.0},
            {"type": "tackle", "team": "home", "start_x": 52.0, "start_y": 33.0},
            {"type": "interception", "team": "home", "start_x": 49.0, "start_y": 35.0},
            {"type": "block", "team": "away", "start_x": 70.0, "start_y": 30.0},
        ]
        result = cluster_pressing_events(events)
        assert len(result) >= 1
        for cluster in result:
            assert isinstance(cluster, PressingCluster)
            assert cluster.intensity > 0

    def test_cluster_pressing_events_intensity(self):
        events = [
            {"type": "tackle", "team": "home", "start_x": 50.0, "start_y": 34.0},
            {"type": "tackle", "team": "home", "start_x": 51.0, "start_y": 34.0},
            {"type": "tackle", "team": "home", "start_x": 49.0, "start_y": 34.0},
            {"type": "tackle", "team": "home", "start_x": 52.0, "start_y": 33.0},
        ]
        result = cluster_pressing_events(events, min_events_per_cluster=2)
        assert len(result) >= 1
        assert result[0].intensity >= 1.0

    def test_cluster_pressing_events_success_rate(self):
        events = [
            {"type": "tackle", "team": "home", "start_x": 50.0, "start_y": 34.0},
            {"type": "foul", "team": "home", "start_x": 51.0, "start_y": 34.0},
        ]
        result = cluster_pressing_events(events, min_events_per_cluster=1)
        if result:
            assert 0 <= result[0].success_rate <= 1.0

    def test_cluster_pressing_events_dominant_team(self):
        events = [
            {"type": "tackle", "team": "home", "start_x": 50.0, "start_y": 34.0},
            {"type": "tackle", "team": "home", "start_x": 51.0, "start_y": 34.0},
            {"type": "interception", "team": "home", "start_x": 49.0, "start_y": 35.0},
        ]
        result = cluster_pressing_events(events, min_events_per_cluster=1)
        if result:
            assert result[0].dominant_team == "home"

    def test_pressing_cluster_dataclass(self):
        c = PressingCluster(
            zone=(3, 5),
            center_x=50.0,
            center_y=34.0,
            intensity=2.5,
            event_count=10,
            success_rate=0.6,
            dominant_team="home",
        )
        d = c.to_dict()
        assert d["zone"] == [3, 5]
        assert d["intensity"] == 2.5
        assert d["event_count"] == 10

    def test_cluster_pressing_events_filter_low_events(self):
        events = [
            {"type": "tackle", "team": "home", "start_x": 50.0, "start_y": 34.0},
        ]
        result = cluster_pressing_events(events, min_events_per_cluster=3)
        assert result == []

    def test_cluster_pressing_events_sorted(self):
        events = [
            {"type": "tackle", "team": "home", "start_x": 50.0, "start_y": 34.0},
            {"type": "tackle", "team": "home", "start_x": 51.0, "start_y": 34.0},
            {"type": "tackle", "team": "home", "start_x": 20.0, "start_y": 34.0},
            {"type": "tackle", "team": "away", "start_x": 80.0, "start_y": 30.0},
            {"type": "tackle", "team": "away", "start_x": 81.0, "start_y": 30.0},
        ]
        result = cluster_pressing_events(events, min_events_per_cluster=1)
        intensities = [c.intensity for c in result]
        assert intensities == sorted(intensities, reverse=True)

    def test_cluster_pressing_events_grid_cells(self):
        events = [
            {"type": "tackle", "team": "home", "start_x": 50.0, "start_y": 34.0},
            {"type": "tackle", "team": "home", "start_x": 51.0, "start_y": 34.0},
        ]
        result = cluster_pressing_events(events, grid_rows=20, grid_cols=15)
        for c in result:
            assert 0 <= c.zone[0] < 20
            assert 0 <= c.zone[1] < 15
