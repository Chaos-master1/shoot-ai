"""Tests for phase transition analysis."""

from kawkab.core.transitions import PhaseTransition, TransitionReport, detect_transitions


class TestTransitions:
    def test_empty_events(self):
        report = detect_transitions([])
        assert report.home_transitions == 0
        assert report.away_transitions == 0

    def test_single_possession_no_transition(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp": 0,
                "start_x": 50,
                "start_y": 34,
                "end_x": 60,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp": 10,
                "start_x": 60,
                "start_y": 34,
                "end_x": 70,
                "completed": True,
            },
        ]
        report = detect_transitions(events)
        assert report.home_transitions == 0

    def test_possession_change_detects_transition(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp": 0,
                "start_x": 50,
                "start_y": 34,
                "end_x": 65,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "away",
                "timestamp": 5,
                "start_x": 65,
                "start_y": 34,
                "end_x": 40,
                "completed": True,
            },
        ]
        report = detect_transitions(events)
        assert report.away_transitions == 1

    def test_counter_attack_classification(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp": 0,
                "start_x": 30,
                "start_y": 34,
                "end_x": 55,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "away",
                "timestamp": 3,
                "start_x": 55,
                "start_y": 34,
                "end_x": 85,
                "completed": True,
            },
        ]
        report = detect_transitions(events)
        # Away ball moves 30m in 3s → counter_attack
        assert report.away_counter_attacks == 1

    def test_gegenpress_classification(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp": 0,
                "start_x": 80,
                "start_y": 34,
                "end_x": 85,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "away",
                "timestamp": 2,
                "start_x": 55,
                "start_y": 34,
                "end_x": 50,
                "completed": True,
            },
        ]
        report = detect_transitions(events)
        # Ball moves >15m from attacking half (prev_x=80, start_x=55) → counter
        assert report.away_counter_attacks == 1

    def test_organized_classification(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp": 0,
                "start_x": 40,
                "start_y": 34,
                "end_x": 42,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "away",
                "timestamp": 15,
                "start_x": 42,
                "start_y": 34,
                "end_x": 50,
                "completed": True,
            },
        ]
        report = detect_transitions(events)
        # Slow speed → organized
        assert report.away_counter_attacks == 0

    def test_shot_outcome(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp": 0,
                "start_x": 30,
                "start_y": 34,
                "end_x": 50,
                "completed": True,
            },
            {
                "type": "shot",
                "team": "away",
                "timestamp": 4,
                "start_x": 50,
                "start_y": 34,
                "end_x": 105,
                "is_goal": False,
                "completed": True,
            },
        ]
        report = detect_transitions(events)
        assert report.away_transitions == 1

    def test_phase_transition_to_dict(self):
        pt = PhaseTransition(
            timestamp=600,
            team="home",
            transition_type="counter_attack",
            start_x=40,
            duration_s=4.5,
            speed_mps=8.0,
            outcome="shot",
            ended_in_final_third=True,
        )
        d = pt.to_dict()
        assert d["team"] == "home"
        assert d["duration_s"] == 4.5

    def test_report_to_dict(self):
        report = TransitionReport(
            home_transitions=5,
            away_transitions=3,
            home_counter_attacks=2,
            away_counter_attacks=1,
            home_shot_conversion=20.0,
            away_shot_conversion=10.0,
        )
        d = report.to_dict()
        assert d["home_transitions"] == 5
        assert d["total_transitions"] == 8
