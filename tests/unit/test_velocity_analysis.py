"""Tests for Acceleration / Velocity Analysis module."""

from kawkab.core.velocity_analysis import VelocityAnalyzer


def _make_trajectory() -> list[tuple[float, float, float]]:
    pts = []
    for i in range(20):
        t = i * 0.5
        x = 50.0 + i * 1.0
        y = 34.0 + (i % 3) * 0.5
        pts.append((t, x, y))
    return pts


class TestComputePlayerVelocity:
    def test_returns_velocity_and_acceleration(self):
        va = VelocityAnalyzer()
        traj = _make_trajectory()
        result = va.compute_player_velocity(traj)
        assert "velocities" in result
        assert "accelerations" in result
        assert len(result["velocities"]) >= 1

    def test_single_point(self):
        va = VelocityAnalyzer()
        result = va.compute_player_velocity([(0, 50, 34)])
        assert result["avg_speed"] == 0.0

    def test_empty_trajectory(self):
        va = VelocityAnalyzer()
        result = va.compute_player_velocity([])
        assert result["avg_speed"] == 0.0

    def test_two_points(self):
        va = VelocityAnalyzer()
        result = va.compute_player_velocity([(0, 50, 34), (1, 60, 34)])
        assert result["max_speed"] > 0


class TestAnalyzeSprints:
    def test_detects_sprints(self):
        va = VelocityAnalyzer()
        profile = [2.0, 3.0, 8.0, 9.0, 8.5, 7.5, 3.0, 2.0]
        result = va.analyze_sprints(profile, threshold=7.0)
        assert result["sprint_count"] >= 1
        assert result["max_speed"] >= 8.0

    def test_no_sprints(self):
        va = VelocityAnalyzer()
        profile = [2.0, 3.0, 4.0, 3.0, 2.0]
        result = va.analyze_sprints(profile, threshold=7.0)
        assert result["sprint_count"] == 0

    def test_empty_profile(self):
        va = VelocityAnalyzer()
        result = va.analyze_sprints([])
        assert result["sprint_count"] == 0

    def test_custom_threshold(self):
        va = VelocityAnalyzer()
        profile = [3.0, 5.0, 6.0, 5.5, 3.0]
        result = va.analyze_sprints(profile, threshold=5.0)
        assert result["sprint_count"] >= 1


class TestComputeAccelerationZones:
    def test_returns_zones(self):
        va = VelocityAnalyzer()
        traj = _make_trajectory()
        zones = va.compute_acceleration_zones(traj)
        assert "high_intensity" in zones
        assert "moderate" in zones
        assert "low" in zones
        assert zones["total"] >= 0

    def test_empty_traj(self):
        va = VelocityAnalyzer()
        zones = va.compute_acceleration_zones([])
        assert zones["total"] == 0


class TestAnalyzeTeamVelocity:
    def test_aggregates_team_stats(self):
        va = VelocityAnalyzer()
        traj = _make_trajectory()
        players = {1: traj, 2: traj}
        result = va.analyze_team_velocity(players)
        assert result["total_sprints"] >= 0
        assert result["total_distance_m"] >= 0

    def test_empty_players(self):
        va = VelocityAnalyzer()
        result = va.analyze_team_velocity({})
        assert result["avg_speed"] == 0.0


class TestComputeFatigueIndex:
    def test_returns_fatigue(self):
        va = VelocityAnalyzer()
        profile = [5.0, 4.8, 4.5, 4.0, 3.5, 3.0]
        result = va.compute_fatigue_index(profile, window_minutes=2)
        assert "fatigue_index" in result
        assert "fatigue_level" in result

    def test_no_fatigue(self):
        va = VelocityAnalyzer()
        profile = [4.0, 4.0, 4.0, 4.0]
        result = va.compute_fatigue_index(profile, window_minutes=2)
        assert result["fatigue_index"] < 1.0

    def test_short_profile(self):
        va = VelocityAnalyzer()
        result = va.compute_fatigue_index([3.0])
        assert result["fatigue_level"] == "none"


class TestGenerateVelocityReport:
    def test_generates_report(self):
        va = VelocityAnalyzer()
        tracking = {
            "frames": [
                {
                    "timestamp": 0.0,
                    "detections": [
                        {"class_name": "person", "track_id": 1, "x": 50, "y": 34},
                    ],
                },
                {
                    "timestamp": 0.5,
                    "detections": [
                        {"class_name": "person", "track_id": 1, "x": 52, "y": 34},
                    ],
                },
            ]
        }
        report = va.generate_velocity_report([], tracking)
        assert "team" in report
        assert "players" in report

    def test_no_tracking_data(self):
        va = VelocityAnalyzer()
        report = va.generate_velocity_report([], {})
        assert "error" in report
