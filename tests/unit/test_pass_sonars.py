"""Tests for pass sonars."""

from kawkab.core.pass_sonars import compute_pass_sonars


class TestPassSonars:
    def test_empty_events(self):
        result = compute_pass_sonars([])
        assert result == []

    def test_single_player_passes(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 70,
                "end_y": 40,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 30,
                "end_y": 20,
                "completed": False,
            },
        ]
        result = compute_pass_sonars(events)
        assert len(result) == 1
        assert result[0]["track_id"] == "1"
        assert result[0]["total_passes"] == 2

    def test_multiple_players(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 70,
                "end_y": 40,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "away",
                "track_id": 5,
                "start_x": 60,
                "start_y": 30,
                "end_x": 40,
                "end_y": 20,
                "completed": True,
            },
        ]
        result = compute_pass_sonars(events)
        assert len(result) == 2

    def test_filters_non_pass(self):
        events = [
            {
                "type": "shot",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 90,
                "end_y": 34,
            },
        ]
        result = compute_pass_sonars(events)
        assert result == []

    def test_sector_distribution(self):
        events = []
        for i in range(12):
            angle = i * 30
            import math

            rad = math.radians(angle)
            events.append(
                {
                    "type": "pass",
                    "team": "home",
                    "track_id": 1,
                    "start_x": 50,
                    "start_y": 34,
                    "end_x": 50 + 20 * math.cos(rad),
                    "end_y": 34 + 20 * math.sin(rad),
                    "completed": True,
                }
            )
        result = compute_pass_sonars(events, sectors=12)
        assert len(result) == 1
        assert result[0]["total_passes"] == 12
        assert len(result[0]["sectors"]) == 12
