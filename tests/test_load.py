"""Load testing suite — benchmarks key operations for performance regression detection.

Tests: xG computation throughput, event storage bulk speed, formation analysis,
pass network construction, pitch control computation.
"""

from __future__ import annotations

import json
import math
import random

import pytest

# Mark all tests as load/benchmark
pytestmark = [
    pytest.mark.load,
    pytest.mark.benchmark,
    pytest.mark.skipif(
        "not config.getoption('--run-load')",
        reason="Pass --run-load to run load/benchmark tests",
    ),
]


# --run-load is registered in tests/conftest.py (pytest_addoption is only
# recognized there, not in a regular test module).


# ── fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def large_event_set():
    """Generate 10,000 synthetic events for bulk processing benchmarks."""
    events = []
    for i in range(10000):
        events.append(
            {
                "id": i,
                "event_type": random.choice(
                    ["pass", "shot", "tackle", "carry", "receipt", "dribble"]
                ),
                "x": random.uniform(0, 105),
                "y": random.uniform(0, 68),
                "end_x": random.uniform(0, 105),
                "end_y": random.uniform(0, 68),
                "timestamp": random.uniform(0, 5400),
                "period": 1 if random.random() < 0.5 else 2,
                "from_track_id": random.randint(1, 22),
                "to_track_id": random.randint(1, 22),
            }
        )
    return events


@pytest.fixture
def large_match_list():
    """Generate 500 synthetic matches for database benchmarks."""
    matches = []
    teams = [
        "FC Stars",
        "United Athletic",
        "City FC",
        "Rovers SC",
        "Athletic Club",
        "Dynamo FC",
        "Wanderers",
        "United FC",
    ]
    for _i in range(500):
        matches.append(
            {
                "name": f"{random.choice(teams)} vs {random.choice(teams)}",
                "home_team": random.choice(teams),
                "away_team": random.choice(teams),
                "home_score": random.randint(0, 5),
                "away_score": random.randint(0, 5),
                "date": f"2025-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            }
        )
    return matches


# ── xG Throughput ─────────────────────────────────────────────────


def test_xg_computation_throughput(benchmark):
    """Benchmark xG computation for 1000 shots."""
    from kawkab.services.xg_model import XGConfig, compute_xg

    config = XGConfig()

    shots = []
    for _i in range(1000):
        shots.append(
            {
                "x": random.uniform(0, 105),
                "y": random.uniform(0, 68),
                "angle": random.uniform(0, math.pi / 2),
                "distance": random.uniform(5, 40),
                "big_chance": random.random() < 0.15,
                "header": random.random() < 0.1,
                "through_ball": random.random() < 0.05,
                "fast_break": random.random() < 0.08,
                "shot_type": random.choice(["left_foot", "right_foot", "head"]),
            }
        )

    def compute_all():
        results = []
        for s in shots:
            xg = compute_xg(
                s["x"],
                s["y"],
                s["angle"],
                s["distance"],
                s.get("big_chance", False),
                s.get("header", False),
                s.get("through_ball", False),
                s.get("fast_break", False),
                config,
            )
            results.append(xg)
        return results

    result = benchmark(compute_all)
    assert len(result) == 1000
    assert all(0 <= x <= 1 for x in result)


# ── Event Storage Bulk Speed ──────────────────────────────────────


def test_event_storage_bulk_throughput(large_event_set, benchmark, tmp_path):
    """Benchmark bulk storage of 10,000 events."""
    from kawkab.core.database_sharding import SeasonShardManager

    _ = tmp_path / "load_test.db"
    shard = SeasonShardManager(str(tmp_path))

    match_id = shard.store_match(
        {
            "name": "Load Test Match",
            "home_team": "Test A",
            "away_team": "Test B",
        }
    )
    shard.store_events(large_event_set, match_id)

    def read_all():
        return shard.get_events(match_id)

    result = benchmark(read_all)
    assert len(result) == 10000


# ── Formation Analysis Throughput ─────────────────────────────────


def test_formation_analysis_throughput(large_event_set, benchmark):
    """Benchmark formation analysis for a large event set."""
    from kawkab.services.formation_analysis import FormationAnalyzer

    analyzer = FormationAnalyzer()

    def analyze():
        return analyzer.analyze(large_event_set)

    result = benchmark(analyze)
    assert result is not None


# ── Pass Network Throughput ───────────────────────────────────────


def test_pass_network_throughput(large_event_set, benchmark):
    """Benchmark pass network construction for large event set."""
    from kawkab.services.pass_network import PassNetworkBuilder

    builder = PassNetworkBuilder()

    def build():
        return builder.build(large_event_set)

    result = benchmark(build)
    assert result is not None


# ── Pitch Control Throughput ──────────────────────────────────────


def test_pitch_control_throughput(benchmark):
    """Benchmark pitch control computation over a grid."""
    from kawkab.services.ball_physics_pitch_control import BallPhysicsPitchControl

    control = BallPhysicsPitchControl()
    home_positions = [{"x": random.uniform(0, 105), "y": random.uniform(0, 68)} for _ in range(11)]
    away_positions = [{"x": random.uniform(0, 105), "y": random.uniform(0, 68)} for _ in range(11)]
    ball_pos = {"x": 50.0, "y": 34.0}
    ball_vel = {"x": 5.0, "y": 0.0}

    def compute():
        return control.compute_pitch_control(home_positions, away_positions, ball_pos, ball_vel)

    result = benchmark(compute)
    assert result is not None


# ── Database Shard Throughput ─────────────────────────────────────


def test_database_shard_throughput(large_match_list, benchmark):
    """Benchmark storing and retrieving 500 matches across shards."""
    import tempfile

    from kawkab.core.database_sharding import SeasonShardManager

    with tempfile.TemporaryDirectory() as tmpdir:
        shard = SeasonShardManager(tmpdir)

        def store_all():
            for m in large_match_list:
                shard.store_match(m)
            return shard.get_all_matches()

        result = benchmark(store_all)
        assert len(result) == 500


# ── GPU Acceleration Detection ────────────────────────────────────


def test_gpu_detection_speed(benchmark):
    """Benchmark GPU backend detection (should be cached/fast)."""
    from kawkab.core.gpu_acceleration import detect_gpu

    def detect():
        return detect_gpu()

    result = benchmark(detect)
    assert result in ("cuda", "mps", "opencl", "cpu")


# ── JSON Serialization Throughput ─────────────────────────────────


def test_json_serialization_throughput(large_event_set, benchmark):
    """Benchmark JSON serialization of 10,000 events (common export path)."""

    def serialize():
        return json.dumps(large_event_set)

    result = benchmark(serialize)
    assert len(result) > 0
    assert isinstance(result, str)


# ── Season Shard Migration ────────────────────────────────────────


def test_shard_migration_throughput(large_match_list, large_event_set, benchmark, tmp_path):
    """Benchmark migration from monolithic to sharded database."""
    import sqlite3

    from kawkab.core.database_sharding import SeasonShardManager

    source_db = tmp_path / "monolithic.db"
    conn = sqlite3.connect(str(source_db))
    conn.executescript("""
        CREATE TABLE matches (id INTEGER PRIMARY KEY, name TEXT, home_team TEXT, away_team TEXT,
            home_score INTEGER, away_score INTEGER, date TEXT, data TEXT);
        CREATE TABLE events (id INTEGER PRIMARY KEY, match_id INTEGER, event_type TEXT,
            timestamp REAL, x REAL, y REAL, data TEXT);
    """)
    for i, m in enumerate(large_match_list):
        conn.execute(
            "INSERT INTO matches (id, name, home_team, away_team, home_score, away_score, date, data) VALUES (?,?,?,?,?,?,?,'{}')",
            (
                i + 1,
                m["name"],
                m["home_team"],
                m["away_team"],
                m["home_score"],
                m["away_score"],
                m["date"],
            ),
        )
    conn.commit()
    conn.close()

    shard = SeasonShardManager(str(tmp_path / "shards"))

    def migrate():
        return shard.migrate_to_shards(str(source_db))

    result = benchmark(migrate)
    assert result["matches"] == 500

    # Cleanup
    shard.close()
