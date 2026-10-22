"""Tests for 3 previously untested core modules (Sprint 1)."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

try:
    from kawkab.core.database_sharding import SeasonShardManager, get_season_key

    HAS_SHARDING = True
except ImportError:
    SeasonShardManager = None  # type: ignore
    get_season_key = None  # type: ignore
    HAS_SHARDING = False
from kawkab.core.mot_metrics import compute_mot_metrics
from kawkab.core.pressing_traps import PressingTrap
from kawkab.core.transitions import PhaseTransition
from kawkab.core.trap_transition_linkage import (
    TrapTransitionAnalysis,
    TrapTransitionLink,
    analyze_trap_transitions,
    summarize_trap_transition,
)

# ═════════════════════════════════════════════════════════════════════════════
# database_sharding
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_SHARDING, reason="database_sharding module archived")
class TestGetSeasonKey:
    def test_current_season(self):
        key = get_season_key()
        assert "-" in key
        parts = key.split("-")
        assert len(parts) == 2
        assert int(parts[0]) < int(parts[1])

    def test_from_date(self):
        key = get_season_key("2024-03-15")
        # Season boundary uses current month (not input month)
        parts = key.split("-")
        assert int(parts[1]) == int(parts[0]) + 1

    def test_from_date_july(self):
        key = get_season_key("2024-07-01")
        parts = key.split("-")
        assert int(parts[0]) == 2024
        assert int(parts[1]) == 2025

    def test_from_date_invalid(self):
        key = get_season_key("invalid")
        assert "-" in key


@pytest.mark.skipif(not HAS_SHARDING, reason="database_sharding module archived")
class TestSeasonShardManager:
    def test_init_creates_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = Path(tmp) / "shards"
            mgr = SeasonShardManager(str(shard_dir))
            assert shard_dir.exists()
            mgr.close()

    def test_get_shard_returns_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SeasonShardManager(str(tmp))
            conn = mgr._get_shard("2024-2025")
            assert isinstance(conn, sqlite3.Connection)
            mgr.close()

    def test_get_shard_same_season_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SeasonShardManager(str(tmp))
            c1 = mgr._get_shard("2024-2025")
            c2 = mgr._get_shard("2024-2025")
            assert c1 is c2
            mgr.close()

    def test_different_seasons_different_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SeasonShardManager(str(tmp))
            c1 = mgr._get_shard("2023-2024")
            c2 = mgr._get_shard("2024-2025")
            assert c1 is not c2
            mgr.close()

    def test_store_and_get_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SeasonShardManager(str(tmp))
            mid = mgr.store_match({"name": "Test Match", "home_team": "Home"})
            assert mid > 0
            m = mgr.get_match(mid)
            assert m is not None
            assert m["name"] == "Test Match"
            mgr.close()

    def test_get_match_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SeasonShardManager(str(tmp))
            assert mgr.get_match(99999) is None
            mgr.close()

    def test_store_and_get_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SeasonShardManager(str(tmp))
            mid = mgr.store_match({"name": "M"})
            events = [
                {"event_type": "pass", "timestamp": 1.0, "x": 50, "y": 30},
                {"event_type": "shot", "timestamp": 2.0, "x": 90, "y": 34},
            ]
            count = mgr.store_events(events, mid)
            assert count == 2
            retrieved = mgr.get_events(mid)
            assert len(retrieved) == 2
            mgr.close()

    def test_store_events_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SeasonShardManager(str(tmp))
            assert mgr.store_events([], 1) == 0
            mgr.close()

    def test_get_all_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SeasonShardManager(str(tmp))
            mgr.store_match({"name": "M1"})
            mgr.store_match({"name": "M2"})
            all_m = mgr.get_all_matches()
            assert len(all_m) >= 2
            mgr.close()

    def test_get_season_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SeasonShardManager(str(tmp))
            season = get_season_key()
            stats = mgr.get_season_stats(season)
            assert stats["season"] == season
            assert "matches" in stats
            assert "events" in stats
            mgr.close()

    def test_close_clears_connections(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SeasonShardManager(str(tmp))
            mgr._get_shard("2024-2025")
            assert len(mgr._shards) == 1
            mgr.close()
            assert len(mgr._shards) == 0

    def test_migrate_to_shards_source_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr = SeasonShardManager(str(tmp))
            result = mgr.migrate_to_shards("/nonexistent/db.sqlite")
            assert "error" in result


# ═════════════════════════════════════════════════════════════════════════════
# mot_metrics
# ═════════════════════════════════════════════════════════════════════════════


class TestComputeMotMetrics:
    def test_perfect_tracking(self):
        gt = {1: [(0, 10, 10), (1, 11, 10), (2, 12, 10)]}
        pred = {1: [(0, 10, 10), (1, 11, 10), (2, 12, 10)]}
        m = compute_mot_metrics(pred, gt)
        assert m["mota"] == 1.0
        assert m["idf1"] == 1.0
        assert m["false_positives"] == 0
        assert m["false_negatives"] == 0
        assert m["id_switches"] == 0
        assert m["fragments"] == 0

    def test_one_false_positive(self):
        gt = {1: [(0, 10, 10)]}
        pred = {1: [(0, 10, 10)], 2: [(0, 50, 50)]}
        m = compute_mot_metrics(pred, gt)
        assert m["false_positives"] == 1
        assert m["mota"] < 1.0

    def test_one_false_negative(self):
        gt = {1: [(0, 10, 10)], 2: [(0, 50, 50)]}
        pred = {1: [(0, 10, 10)]}
        m = compute_mot_metrics(pred, gt)
        assert m["false_negatives"] == 1
        assert m["mota"] < 1.0

    def test_one_id_switch(self):
        gt = {1: [(0, 10, 10), (1, 11, 10)]}
        pred = {2: [(0, 10, 10)], 3: [(1, 11, 10)]}
        m = compute_mot_metrics(pred, gt)
        assert m["id_switches"] == 1
        assert m["mota"] < 1.0

    def test_fragmentation(self):
        gt = {1: [(0, 10, 10), (1, 10, 10), (2, 10, 10)]}
        pred = {2: [(0, 10, 10), (2, 10, 10)]}
        m = compute_mot_metrics(pred, gt, fp_threshold=20.0)
        assert m["fragments"] == 1

    def test_empty_inputs(self):
        m = compute_mot_metrics({}, {})
        assert m["mota"] == 1.0
        assert m["motp"] == 0.0
        assert m["false_positives"] == 0
        assert m["false_negatives"] == 0

    def test_metrics_in_zero_to_one_range(self):
        gt = {1: [(0, 10, 10), (1, 15, 15)], 2: [(0, 50, 50)]}
        pred = {3: [(0, 10, 10), (1, 16, 16)], 4: [(0, 55, 55)], 5: [(1, 90, 90)]}
        m = compute_mot_metrics(pred, gt, fp_threshold=10.0)
        assert 0.0 <= m["mota"] <= 1.0
        assert 0.0 <= m["motp"] <= 100.0
        assert 0.0 <= m["idf1"] <= 1.0

    def test_single_frame(self):
        gt = {1: [(0, 10, 10)], 2: [(0, 20, 20)]}
        pred = {1: [(0, 10, 10)], 2: [(0, 20, 20)]}
        m = compute_mot_metrics(pred, gt)
        assert m["total_matches"] == 2
        assert m["mota"] == 1.0

    def test_motp_computed_correctly(self):
        gt = {1: [(0, 0, 0)]}
        pred = {1: [(0, 3, 4)]}  # distance = 5.0
        m = compute_mot_metrics(pred, gt, fp_threshold=10.0)
        assert m["total_matches"] == 1
        assert m["motp"] == 5.0


# ═════════════════════════════════════════════════════════════════════════════
# trap_transition_linkage
# ═════════════════════════════════════════════════════════════════════════════


class TestTrapTransitionLink:
    def test_create_link(self):
        link = TrapTransitionLink(
            trap_index=0,
            transition_index=1,
            time_delta=1.5,
            spatial_distance=10.0,
            goal_scored=False,
            shot_created=True,
        )
        assert link.trap_index == 0
        assert link.transition_index == 1
        assert link.time_delta == 1.5
        assert link.spatial_distance == 10.0
        assert link.shot_created is True
        assert link.goal_scored is False


class TestTrapTransitionAnalysis:
    def test_create_analysis(self):
        a = TrapTransitionAnalysis(
            total_traps=5,
            successful_traps=3,
            conversion_rate=0.6,
            goal_conversion_rate=0.2,
            avg_transition_time=2.1,
        )
        assert a.total_traps == 5
        assert a.conversion_rate == 0.6
        assert a.avg_transition_time == 2.1

    def test_empty_analysis(self):
        a = TrapTransitionAnalysis(total_traps=0, successful_traps=0)
        assert a.transitions_from_traps == []


class TestAnalyzeTrapTransitions:
    def make_trap(self, zone_name, regains=1, x_range=(0, 35), y_range=(0, 34)):
        return PressingTrap(
            zone_name=zone_name,
            zone_x_range=x_range,
            zone_y_range=y_range,
            regain_possession_count=regains,
            defensive_actions_in_zone=5,
        )

    def make_transition(self, timestamp, team="home", start_x=50, start_y=34):
        return PhaseTransition(
            timestamp=timestamp,
            team=team,
            start_x=start_x,
            start_y=start_y,
        )

    def test_with_valid_traps_and_transitions(self):
        trap = self.make_trap("central_mid", regains=1)
        trans = self.make_transition(timestamp=15.0, team="home")
        events = [
            {
                "type": "tackle",
                "team": "home",
                "timestamp": 10.0,
                "x": 50,
                "y": 34,
                "start_x": 50,
                "start_y": 34,
            },
            {"type": "pass", "team": "home", "timestamp": 12.0, "x": 55, "y": 34},
        ]
        result = analyze_trap_transitions([trap], [trans], events)
        assert result.total_traps == 1
        assert isinstance(result, TrapTransitionAnalysis)

    def test_with_no_traps(self):
        result = analyze_trap_transitions([], [], [])
        assert result.total_traps == 0
        assert result.successful_traps == 0

    def test_with_no_transitions_still_processes(self):
        trap = self.make_trap("central_mid", regains=1)
        events = [
            {
                "type": "tackle",
                "team": "home",
                "timestamp": 10.0,
                "x": 50,
                "y": 34,
                "start_x": 50,
                "start_y": 34,
            },
            {"type": "pass", "team": "home", "timestamp": 12.0, "x": 55, "y": 34},
        ]
        result = analyze_trap_transitions([trap], [], events)
        assert result.total_traps == 1
        assert result.successful_traps == 1
        assert result.transitions_from_traps == []

    def test_temporal_gap_too_large(self):
        trap = self.make_trap("central_mid", regains=1)
        trans = self.make_transition(timestamp=100.0, team="home")
        events = [
            {
                "type": "tackle",
                "team": "home",
                "timestamp": 10.0,
                "x": 50,
                "y": 34,
                "start_x": 50,
                "start_y": 34,
            },
            {"type": "pass", "team": "home", "timestamp": 12.0, "x": 55, "y": 34},
        ]
        result = analyze_trap_transitions([trap], [trans], events)
        assert result.transitions_from_traps == []


class TestSummarizeTrapTransition:
    def test_with_valid_data(self):
        a = TrapTransitionAnalysis(
            total_traps=10,
            successful_traps=5,
            transitions_from_traps=[
                TrapTransitionLink(0, 0, 1.5, 5.0, False, True),
            ],
            conversion_rate=0.5,
            goal_conversion_rate=0.1,
            avg_transition_time=1.5,
        )
        s = summarize_trap_transition(a)
        assert "trap_frequency" in s
        assert "chance_conversion" in s
        assert "avg_transition_time" in s
        assert "9.0" in s["trap_frequency"]
        assert "50%" in s["chance_conversion"]

    def test_with_empty_analysis(self):
        a = TrapTransitionAnalysis(total_traps=0, successful_traps=0)
        s = summarize_trap_transition(a)
        assert "No pressing traps" in s["trap_frequency"]
        assert "0%" in s["chance_conversion"]
        assert "No trap" in s["avg_transition_time"]

    def test_no_links(self):
        a = TrapTransitionAnalysis(
            total_traps=5,
            successful_traps=2,
            conversion_rate=0.0,
            goal_conversion_rate=0.0,
            avg_transition_time=0.0,
        )
        s = summarize_trap_transition(a)
        assert "No trap" in s["avg_transition_time"]
