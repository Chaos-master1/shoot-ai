"""Tests for export converters."""

import json

from kawkab.core.export_converters import (
    to_opta_csv,
    to_spadl_csv,
    to_statsbomb_json,
)


class TestStatsBombExport:
    def test_empty_events(self):
        result = to_statsbomb_json([])
        assert json.loads(result) == []

    def test_pass_event_export(self):
        events = [
            {
                "type": "pass",
                "timestamp": 10.0,
                "team": "home",
                "track_id": 1,
                "to_track_id": 2,
                "completed": True,
                "metadata": {
                    "start_x_pct": 0.3,
                    "start_y_pct": 0.5,
                    "end_x_pct": 0.6,
                    "end_y_pct": 0.5,
                },
            }
        ]
        result = json.loads(to_statsbomb_json(events))
        assert len(result) == 1
        assert result[0]["event_type"] == "pass"
        assert result[0]["team"]["name"] == "Home"
        assert "pass" in result[0]

    def test_shot_event_export(self):
        events = [
            {
                "type": "shot",
                "timestamp": 30.0,
                "team": "away",
                "track_id": 3,
                "on_target": True,
                "metadata": {"distance_to_goal_m": 12.0, "angle_to_goal_deg": 10.0},
            }
        ]
        result = json.loads(to_statsbomb_json(events))
        assert len(result) == 1
        assert result[0]["event_type"] == "shot"
        assert "shot" in result[0]

    def test_period_detection(self):
        events = [{"type": "pass", "timestamp": 3000.0, "team": "home"}]  # 50 min
        result = json.loads(to_statsbomb_json(events))
        assert result[0]["period"] == 2

    def test_team_and_player_info(self):
        events = [{"type": "pass", "timestamp": 10.0, "team": "home", "track_id": 7}]
        result = json.loads(to_statsbomb_json(events))
        assert result[0]["player"]["id"] == 7


class TestSpadlExport:
    def test_empty_events(self):
        result = to_spadl_csv([])
        assert "game_id" in result  # header row

    def test_csv_structure(self):
        events = [
            {
                "type": "pass",
                "timestamp": 10.0,
                "team": "home",
                "track_id": 1,
                "metadata": {
                    "start_x_pct": 0.3,
                    "start_y_pct": 0.5,
                    "end_x_pct": 0.6,
                    "end_y_pct": 0.5,
                },
            }
        ]
        result = to_spadl_csv(events)
        lines = result.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "pass" in lines[1]

    def test_shot_spadl(self):
        events = [{"type": "shot", "timestamp": 30.0, "team": "away", "track_id": 3}]
        result = to_spadl_csv(events)
        assert "shot" in result

    def test_carry_spadl(self):
        events = [{"type": "carry", "timestamp": 5.0, "team": "home", "track_id": 1}]
        result = to_spadl_csv(events)
        assert "carry" in result


class TestOptaExport:
    def test_empty_events(self):
        result = to_opta_csv([])
        assert "match_id" in result

    def test_opta_structure(self):
        events = [
            {
                "type": "pass",
                "timestamp": 10.0,
                "team": "home",
                "track_id": 1,
                "metadata": {
                    "start_x_pct": 0.3,
                    "start_y_pct": 0.5,
                    "end_x_pct": 0.6,
                    "end_y_pct": 0.5,
                },
            }
        ]
        result = to_opta_csv(events)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert "pass" in lines[1]
        assert "10" in lines[1]  # timestamp

    def test_shot_outcome(self):
        events = [
            {"type": "shot", "timestamp": 30.0, "team": "home", "track_id": 3, "on_target": True}
        ]
        result = to_opta_csv(events)
        assert "1" in result.split("\n")[1].split(",")[-2]  # outcome=1 for on target
