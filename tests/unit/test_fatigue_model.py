"""Tests for fatigue/substitution model."""

from kawkab.core.fatigue_model import compute_fatigue


class TestFatigueModel:
    def test_empty_events(self):
        report = compute_fatigue([], 90)
        assert report.home_fatigue == []
        assert report.away_fatigue == []
        assert len(report.substitutions) == 0
        assert report.home_avg_fatigue == 0.0

    def test_single_player_passes(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 0,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
                "timestamp": 0,
            },
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 80,
                "end_y": 40,
                "timestamp": 300,
            },
        ]
        report = compute_fatigue(events, 90)
        assert len(report.home_fatigue) == 1
        assert report.home_fatigue[0]["track_id"] == 1
        assert report.home_fatigue[0]["distance_covered_m"] > 0
        assert report.home_fatigue[0]["fatigue_index"] >= 0.0

    def test_multiple_players_both_teams(self):
        events = [
            {
                "type": "run",
                "team": "home",
                "track_id": 1,
                "start_x": 0,
                "start_y": 34,
                "end_x": 80,
                "end_y": 0,
                "timestamp": 0,
            },
            {
                "type": "run",
                "team": "away",
                "track_id": 5,
                "start_x": 105,
                "start_y": 34,
                "end_x": 20,
                "end_y": 68,
                "timestamp": 0,
            },
            {
                "type": "shot",
                "team": "home",
                "track_id": 1,
                "start_x": 90,
                "start_y": 34,
                "end_x": 100,
                "end_y": 34,
                "timestamp": 300,
                "is_goal": True,
            },
        ]
        report = compute_fatigue(events, 90)
        assert len(report.home_fatigue) == 1
        assert len(report.away_fatigue) == 1
        assert report.home_avg_fatigue > 0.0
        assert report.away_avg_fatigue > 0.0

    def test_substitution_detection(self):
        events = [
            {
                "type": "substitution",
                "team": "home",
                "player_in": 11,
                "player_out": 7,
                "timestamp": 2700,
            },
        ]
        report = compute_fatigue(events, 90)
        assert len(report.substitutions) == 1
        assert report.substitutions[0]["track_id_in"] == 11
        assert report.substitutions[0]["track_id_out"] == 7
        assert report.substitutions[0]["team"] == "home"

    def test_high_intensity_raises_fatigue(self):
        low_intensity = [
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 55,
                "end_y": 34,
                "timestamp": t,
            }
            for t in range(0, 5400, 60)
        ]
        high_intensity = (
            low_intensity
            + [
                {
                    "type": "shot",
                    "team": "home",
                    "track_id": 1,
                    "start_x": 90,
                    "start_y": 34,
                    "end_x": 100,
                    "end_y": 34,
                    "timestamp": t,
                    "is_goal": True,
                }
                for t in range(0, 5400, 120)
            ]
            + [
                {
                    "type": "tackle",
                    "team": "home",
                    "track_id": 1,
                    "start_x": 50,
                    "start_y": 34,
                    "end_x": 55,
                    "end_y": 34,
                    "timestamp": t,
                }
                for t in range(0, 5400, 180)
            ]
        )
        r_low = compute_fatigue(low_intensity, 90)
        r_high = compute_fatigue(high_intensity, 90)
        assert r_high.home_fatigue[0]["fatigue_index"] > r_low.home_fatigue[0]["fatigue_index"]
