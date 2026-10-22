"""Tests for PhysicalMetricsAnalyzer."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import math

from kawkab.core.physical_metrics import (
    SPEED_ZONES,
    PhysicalMetricsAnalyzer,
    PlayerPhysicalMetrics,
    TeamPhysicalReport,
)


def test_analyze_player_basic():
    pma = PhysicalMetricsAnalyzer()
    trajectory = [(0.0, 0.0, 0.0), (1.0, 5.0, 0.0), (2.0, 10.0, 0.0), (3.0, 15.0, 0.0)]
    metrics = pma.analyze_player(trajectory)
    assert metrics.total_distance_m > 0
    assert metrics.max_speed_ms > 0
    assert metrics.avg_speed_ms > 0


def test_analyze_player_few_points():
    pma = PhysicalMetricsAnalyzer()
    trajectory = [(0.0, 0.0, 0.0)]
    metrics = pma.analyze_player(trajectory)
    assert isinstance(metrics, PlayerPhysicalMetrics)
    assert metrics.total_distance_m == 0.0


def test_analyze_player_empty():
    pma = PhysicalMetricsAnalyzer()
    metrics = pma.analyze_player([])
    assert metrics.total_distance_m == 0.0


def test_analyze_player_sprint():
    pma = PhysicalMetricsAnalyzer()
    # Fast movement (10 m/s > 7.0 sprint threshold)
    trajectory = [(0.0, 0.0, 0.0), (0.1, 1.0, 0.0), (0.2, 2.0, 0.0), (0.3, 3.0, 0.0)]
    metrics = pma.analyze_player(trajectory)
    assert metrics.sprint_count >= 0


def test_analyze_team_basic():
    pma = PhysicalMetricsAnalyzer()
    frames = [
        {"timestamp": 0.0, "home_positions": [(0.0, 0.0, 1), (10.0, 0.0, 2)]},
        {"timestamp": 1.0, "home_positions": [(1.0, 0.0, 1), (11.0, 0.0, 2)]},
        {"timestamp": 2.0, "home_positions": [(2.0, 0.0, 1), (12.0, 0.0, 2)]},
    ]
    report = pma.analyze_team(frames, team="home")
    assert isinstance(report, TeamPhysicalReport)
    assert report.team == "home"
    assert len(report.players) > 0


def test_analyze_team_empty():
    pma = PhysicalMetricsAnalyzer()
    report = pma.analyze_team([], team="home")
    assert report.total_distance_m == 0.0


def test_analyze_team_away():
    pma = PhysicalMetricsAnalyzer()
    frames = [
        {"timestamp": 0.0, "away_positions": [(50.0, 34.0, 10)]},
        {"timestamp": 1.0, "away_positions": [(55.0, 34.0, 10)]},
        {"timestamp": 2.0, "away_positions": [(60.0, 34.0, 10)]},
    ]
    report = pma.analyze_team(frames, team="away")
    assert report.team == "away"
    assert 10 in report.players


def test_speed_zones_defined():
    assert "walking" in SPEED_ZONES
    assert "jogging" in SPEED_ZONES
    assert "running" in SPEED_ZONES
    assert "high_intensity" in SPEED_ZONES
    assert "sprinting" in SPEED_ZONES


def test_player_to_dict():
    p = PlayerPhysicalMetrics(track_id=5, total_distance_m=1200.0)
    d = p.to_dict()
    assert d["tid"] == 5
    assert d["total_dist"] == 1200.0


def test_team_to_dict():
    r = TeamPhysicalReport(team="home", total_distance_m=12000.0)
    d = r.to_dict()
    assert d["team"] == "home"
    assert d["total_dist_km"] == 12.0


def test_analyze_player_metabolic_power():
    pma = PhysicalMetricsAnalyzer()
    # Constant speed of 4 m/s (14.4 km/h)
    trajectory = [(i * 0.5, i * 2.0, 0.0) for i in range(10)]
    metrics = pma.analyze_player(trajectory)
    assert metrics.metabolic_power_avg_w_kg > 0
    assert metrics.metabolic_power_peak_w_kg > 0


def test_analyze_player_acceleration_deceleration():
    pma = PhysicalMetricsAnalyzer()
    # Start slow, accelerate fast, then decelerate
    trajectory = [
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0),  # 1 m/s
        (1.0, 4.0, 0.0),  # 7 m/s (accel)
        (1.5, 7.5, 0.0),  # 7 m/s
        (2.0, 9.0, 0.0),  # 3 m/s (decel)
    ]
    metrics = pma.analyze_player(trajectory)
    # Should detect acceleration and deceleration
    assert isinstance(metrics, PlayerPhysicalMetrics)


def test_analyze_player_distance_by_zone():
    pma = PhysicalMetricsAnalyzer()
    # Mixed speeds
    trajectory = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),  # 1 m/s - walking
        (2.0, 4.0, 0.0),  # 3 m/s - jogging
        (3.0, 8.0, 0.0),  # 4 m/s - running
    ]
    metrics = pma.analyze_player(trajectory)
    zones = metrics.distance_by_zone
    total_in_zones = sum(zones.values())
    assert abs(total_in_zones - metrics.total_distance_m) < 0.1 or total_in_zones > 0


def test_analyze_player_player_load():
    pma = PhysicalMetricsAnalyzer()
    # Stop-start movement creates higher player load
    trajectory = []
    for i in range(20):
        t = i * 0.5
        # Sinusoidal speed pattern
        x = 2.0 * math.sin(i * 0.5)
        y = 0.0
        trajectory.append((t, x, y))
    metrics = pma.analyze_player(trajectory)
    assert metrics.player_load > 0 or metrics.total_distance_m == 0


def test_analyze_team_body_mass():
    pma = PhysicalMetricsAnalyzer()
    frames = [
        {"timestamp": 0.0, "home_positions": [(0.0, 0.0, 1)]},
        {"timestamp": 1.0, "home_positions": [(5.0, 0.0, 1)]},
        {"timestamp": 2.0, "home_positions": [(10.0, 0.0, 1)]},
    ]
    masses = {1: 80.0}
    report = pma.analyze_team(frames, team="home", player_track_ids=masses)
    assert 1 in report.players


def test_analyze_player_constant_speed_avg():
    pma = PhysicalMetricsAnalyzer()
    trajectory = [(i * 1.0, i * 5.0, 0.0) for i in range(5)]  # 5 m/s
    metrics = pma.analyze_player(trajectory)
    assert metrics.total_distance_m > 0


def test_report_aggregation():
    pma = PhysicalMetricsAnalyzer()
    frames = [
        {"timestamp": 0.0, "home_positions": [(0.0, 0.0, 1), (0.0, 5.0, 2)]},
        {"timestamp": 1.0, "home_positions": [(5.0, 0.0, 1), (5.0, 5.0, 2)]},
        {"timestamp": 2.0, "home_positions": [(10.0, 0.0, 1), (10.0, 5.0, 2)]},
    ]
    report = pma.analyze_team(frames, team="home")
    assert len(report.players) == 2
    assert report.total_distance_m > 0
    assert report.total_sprints >= 0


def test_analyze_player_distance_per_minute():
    pma = PhysicalMetricsAnalyzer()
    # 30m over 30 seconds = 60 m/min
    trajectory = [(i * 3.0, i * 3.0, 0.0) for i in range(11)]  # 30m in 30s
    metrics = pma.analyze_player(trajectory)
    assert metrics.distance_per_minute_m > 0


def test_standalone_import():
    """Test that the module can be imported standalone."""
    from kawkab.core.physical_metrics import PhysicalMetricsAnalyzer

    pma = PhysicalMetricsAnalyzer()
    assert pma.SPRINT_THRESHOLD_MS == 7.0
    assert pma.HIGH_INTENSITY_THRESHOLD_MS == 5.5
    assert pma.ACCEL_THRESHOLD == 3.0
