"""Tests for pressing efficiency module."""

import pytest

from kawkab.core.pressing_efficiency import PressingEfficiencyAnalyzer


class TestPressingEfficiency:
    """Test suite for PressingEfficiencyAnalyzer."""

    def make_events(self, base_time=0.0, team="home"):
        """Helper to build minimal events for testing."""
        return []

    def test_compute_trap_to_shot_rate_detection(self):
        """Basic detection: 1 trap, followed by a shot."""
        events = [
            {"type": "tackle", "team": "home", "timestamp": 10.0, "x": 70, "y": 34},
            {
                "type": "shot",
                "team": "home",
                "timestamp": 12.0,
                "x": 95,
                "y": 34,
                "xg": 0.2,
                "is_goal": False,
            },
        ]
        pea = PressingEfficiencyAnalyzer()
        result = pea.compute_trap_to_shot_rate(events)
        assert result["home"]["traps"] == 1
        assert result["home"]["shots_from_traps"] == 1
        assert result["home"]["conversion_rate"] == 1.0

    def test_compute_trap_to_goal_rate_detection(self):
        """1 trap followed by a goal."""
        events = [
            {"type": "interception", "team": "home", "timestamp": 10.0, "x": 70, "y": 34},
            {
                "type": "shot",
                "team": "home",
                "timestamp": 12.0,
                "x": 95,
                "y": 34,
                "xg": 0.4,
                "is_goal": True,
            },
        ]
        pea = PressingEfficiencyAnalyzer()
        result = pea.compute_trap_to_goal_rate(events)
        assert result["home"]["goals_from_traps"] == 1
        assert result["home"]["conversion_rate"] == 1.0

    def test_analyze_high_press_efficiency_ratio(self):
        """Traps in attacking third vs shots conceded after losing press."""
        events = [
            {"type": "tackle", "team": "home", "timestamp": 10.0, "x": 80, "y": 34},
            {"type": "shot", "team": "away", "timestamp": 12.0, "x": 95, "y": 34, "xg": 0.1},
        ]
        pea = PressingEfficiencyAnalyzer()
        result = pea.analyze_high_press_efficiency(events)
        # 1 trap in attacking third, 1 shot after — ratio = 1
        assert result["home"] == 1.0

    def test_analyze_high_press_efficiency_empty(self):
        """No events → both teams have 0.0 efficiency."""
        pea = PressingEfficiencyAnalyzer()
        result = pea.analyze_high_press_efficiency([])
        assert result["home"] == 0.0
        assert result["away"] == 0.0

    def test_compute_press_recovery_attack_counts(self):
        """Successful recovery followed by chance."""
        events = [
            {"type": "tackle", "team": "home", "timestamp": 10.0, "x": 70, "y": 34},
            {"type": "pass", "team": "home", "timestamp": 12.0, "x": 75, "y": 34},
            {"type": "shot", "team": "home", "timestamp": 14.0, "x": 95, "y": 34, "xg": 0.15},
        ]
        pea = PressingEfficiencyAnalyzer()
        result = pea.compute_press_recovery_attack(events)
        assert result["home"]["recoveries"] == 1
        assert result["home"]["chances_created"] == 1
        assert result["home"]["xg_created"] == pytest.approx(0.15, rel=0.01)

    def test_compute_press_recovery_attack_empty(self):
        """No events → zeros."""
        pea = PressingEfficiencyAnalyzer()
        result = pea.compute_press_recovery_attack([])
        assert result["home"]["recoveries"] == 0
        assert result["home"]["chances_created"] == 0
        assert result["home"]["xg_created"] == 0.0

    def test_edge_case_no_traps(self):
        """No trap events → zeros across all metrics."""
        events = [
            {"type": "pass", "team": "home", "timestamp": 10.0, "x": 50, "y": 34},
            {"type": "shot", "team": "home", "timestamp": 12.0, "x": 95, "y": 34, "xg": 0.1},
        ]
        pea = PressingEfficiencyAnalyzer()
        ts = pea.compute_trap_to_shot_rate(events)
        assert ts["home"]["traps"] == 0
        assert ts["home"]["conversion_rate"] == 0.0

    def test_edge_case_all_traps_lead_to_shots(self):
        """Multiple traps, all leading to shots."""
        events = []
        for i in range(3):
            events.append(
                {"type": "tackle", "team": "home", "timestamp": float(10 + i * 5), "x": 70, "y": 34}
            )
            events.append(
                {
                    "type": "shot",
                    "team": "home",
                    "timestamp": float(12 + i * 5),
                    "x": 95,
                    "y": 34,
                    "xg": 0.1,
                    "is_goal": False,
                }
            )
        pea = PressingEfficiencyAnalyzer()
        result = pea.compute_trap_to_shot_rate(events)
        assert result["home"]["traps"] == 3
        assert result["home"]["shots_from_traps"] == 3
        assert result["home"]["conversion_rate"] == 1.0

    def test_edge_case_no_traps_lead_to_anything(self):
        """Traps exist but never lead to shots."""
        events = [
            {"type": "tackle", "team": "home", "timestamp": 10.0, "x": 70, "y": 34},
            {"type": "pass", "team": "home", "timestamp": 12.0, "x": 60, "y": 34},
            {"type": "pass", "team": "away", "timestamp": 14.0, "x": 50, "y": 34},
        ]
        pea = PressingEfficiencyAnalyzer()
        result = pea.compute_trap_to_shot_rate(events)
        assert result["home"]["traps"] == 1
        assert result["home"]["shots_from_traps"] == 0
        assert result["home"]["conversion_rate"] == 0.0

    def test_mixed_some_traps_lead_to_shots(self):
        """Mixed: some traps produce shots, some don't."""
        events = [
            {"type": "tackle", "team": "home", "timestamp": 10.0, "x": 70, "y": 34},
            {
                "type": "shot",
                "team": "home",
                "timestamp": 12.0,
                "x": 95,
                "y": 34,
                "xg": 0.1,
                "is_goal": False,
            },
            {"type": "foul", "team": "home", "timestamp": 20.0, "x": 75, "y": 34},
            {"type": "pass", "team": "home", "timestamp": 22.0, "x": 60, "y": 34},
            {"type": "tackle", "team": "home", "timestamp": 30.0, "x": 65, "y": 34},
            {
                "type": "shot",
                "team": "home",
                "timestamp": 32.0,
                "x": 90,
                "y": 34,
                "xg": 0.2,
                "is_goal": True,
            },
        ]
        pea = PressingEfficiencyAnalyzer()
        result = pea.compute_trap_to_shot_rate(events)
        assert result["home"]["traps"] == 3
        assert result["home"]["shots_from_traps"] == 2
        assert result["home"]["conversion_rate"] == pytest.approx(2.0 / 3.0, rel=0.01)

    def test_with_real_pressing_traps_module_import(self):
        """Import the real pressing_traps module alongside this one."""
        from kawkab.core.pressing_traps import detect_pressing_traps

        events = [
            {"type": "tackle", "team": "home", "timestamp": 10.0, "x": 80, "y": 34},
            {
                "type": "shot",
                "team": "home",
                "timestamp": 12.0,
                "x": 95,
                "y": 34,
                "xg": 0.3,
                "is_goal": True,
            },
        ]
        report = detect_pressing_traps(events, "home")
        assert report is not None
        pea = PressingEfficiencyAnalyzer()
        ts = pea.compute_trap_to_shot_rate(events)
        assert "home" in ts

    def test_team_specific_filtering(self):
        """Only the specified team's traps are counted."""
        events = [
            {"type": "tackle", "team": "home", "timestamp": 10.0, "x": 70, "y": 34},
            {"type": "shot", "team": "home", "timestamp": 12.0, "x": 95, "y": 34, "xg": 0.1},
            {"type": "interception", "team": "away", "timestamp": 20.0, "x": 30, "y": 34},
            {"type": "shot", "team": "away", "timestamp": 22.0, "x": 5, "y": 34, "xg": 0.2},
        ]
        pea = PressingEfficiencyAnalyzer()
        result = pea.compute_trap_to_shot_rate(events)
        assert result["home"]["traps"] == 1
        assert result["away"]["traps"] == 1
        assert result["home"]["shots_from_traps"] == 1
        assert result["away"]["shots_from_traps"] == 1


class TestPressingEfficiencyAway:
    """Additional away-team specific coverage."""

    def test_away_team_efficiency(self):
        events = [
            {"type": "tackle", "team": "away", "timestamp": 10.0, "x": 70, "y": 34},
            {
                "type": "shot",
                "team": "away",
                "timestamp": 12.0,
                "x": 5,
                "y": 34,
                "xg": 0.2,
                "is_goal": True,
            },
        ]
        pea = PressingEfficiencyAnalyzer()
        ts = pea.compute_trap_to_shot_rate(events)
        assert ts["away"]["shots_from_traps"] == 1
        assert ts["away"]["conversion_rate"] == 1.0
        tg = pea.compute_trap_to_goal_rate(events)
        assert tg["away"]["goals_from_traps"] == 1
