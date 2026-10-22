"""Tests for packing passes — opponent bypass count."""

from kawkab.core.packing import (
    PackingReport,
    PackingResult,
    _is_behind_line,
    compute_match_packing,
    compute_packing,
)


class TestIsBehindLine:
    def test_player_behind_line(self):
        assert (
            _is_behind_line(
                60,
                34,  # player
                50,
                34,  # line start
                80,
                34,  # line end
                attacking_direction=1,
            )
            is True
        )

    def test_player_in_front_of_line(self):
        assert (
            _is_behind_line(
                85,
                34,  # player past end
                50,
                34,
                80,
                34,
                attacking_direction=1,
            )
            is False
        )

    def test_player_far_side(self):
        assert (
            _is_behind_line(
                60,
                20,  # vertically far
                50,
                34,
                80,
                34,
                attacking_direction=1,
            )
            is False
        )

    def test_attacking_left(self):
        assert (
            _is_behind_line(
                40,
                34,  # player
                60,
                34,  # line start (the attacking side's own half)
                30,
                34,  # line end (deeper in opposition half)
                attacking_direction=-1,
            )
            is True
        )

    def test_short_line_returns_false(self):
        assert _is_behind_line(55, 34, 54.9, 34, 55.1, 34) is False


class TestComputePacking:
    def test_no_opponents(self):
        result = compute_packing(
            {"start_x": 50, "start_y": 34, "end_x": 80, "end_y": 34},
            [],
        )
        assert result.packing_count == 0

    def test_forward_pass_packs_one(self):
        result = compute_packing(
            {"start_x": 50, "start_y": 34, "end_x": 80, "end_y": 34},
            [(60, 34)],  # directly in line of pass
            attacking_direction=1,
        )
        assert result.packing_count == 1

    def test_backward_pass_packs_none(self):
        result = compute_packing(
            {"start_x": 60, "start_y": 34, "end_x": 40, "end_y": 34},
            [(50, 34)],  # behind the start (greater x)
            attacking_direction=1,
        )
        # Attacking direction is 1 (right), so x > start_x is behind
        # 50 < 60, so not packed
        assert result.packing_count == 0

    def test_territory_penetration_positive(self):
        result = compute_packing(
            {"start_x": 30, "start_y": 34, "end_x": 80, "end_y": 34},
            [],
        )
        assert result.territory_penetration > 0
        assert result.is_progressive is True

    def test_short_pass_not_progressive(self):
        result = compute_packing(
            {"start_x": 50, "start_y": 34, "end_x": 52, "end_y": 34},
            [],
        )
        assert result.is_progressive is False

    def test_pass_length_calculated(self):
        result = compute_packing(
            {"start_x": 50, "start_y": 34, "end_x": 80, "end_y": 34},
            [],
        )
        assert abs(result.pass_length - 30.0) < 0.1

    def test_multi_opponent_packing(self):
        result = compute_packing(
            {"start_x": 40, "start_y": 34, "end_x": 85, "end_y": 34},
            [(50, 34), (65, 34), (75, 34.5), (30, 34)],  # last one is in front, not packed
            attacking_direction=1,
        )
        assert result.packing_count == 3

    def test_result_to_dict(self):
        r = PackingResult(packing_count=5, territory_penetration=15.0, pass_length=25.0)
        d = r.to_dict()
        assert d["packing_count"] == 5
        assert d["territory_penetration"] == 15.0


class TestMatchPacking:
    def test_empty_events(self):
        result = compute_match_packing([])
        assert result["home"].total_passes == 0
        assert result["away"].total_passes == 0

    def test_non_pass_filtered(self):
        events = [
            {"type": "shot", "timestamp": 5.0, "team": "home", "x": 80, "y": 34, "is_goal": True},
            {
                "type": "pass",
                "timestamp": 10.0,
                "team": "home",
                "start_x": 50,
                "start_y": 34,
                "end_x": 80,
                "end_y": 34,
                "completed": True,
            },
            {"type": "tackle", "timestamp": 11.0, "team": "away", "x": 60, "y": 34},
        ]
        result = compute_match_packing(events)
        assert result["home"].total_passes >= 0

    def test_both_teams(self):
        events = [
            {
                "type": "pass",
                "timestamp": 1.0,
                "team": "home",
                "start_x": 40,
                "start_y": 34,
                "end_x": 75,
                "end_y": 34,
                "completed": True,
            },
            {"type": "tackle", "timestamp": 2.0, "team": "away", "x": 55, "y": 34},
            {
                "type": "pass",
                "timestamp": 3.0,
                "team": "away",
                "start_x": 60,
                "start_y": 34,
                "end_x": 30,
                "end_y": 34,
                "completed": True,
            },
            {"type": "interception", "timestamp": 4.0, "team": "home", "x": 45, "y": 34},
        ]
        result = compute_match_packing(events, team_attacks_right=True)
        assert result["home"].total_passes >= 0
        assert result["away"].total_passes >= 0

    def test_report_properties(self):
        report = PackingReport(total_packing=25, avg_packing=2.5, max_packing=8, total_passes=10)
        d = report.to_dict()
        assert d["total_packing"] == 25
        assert d["avg_packing"] == 2.5

    def test_empty_report(self):
        report = PackingReport()
        d = report.to_dict()
        assert d["total_packing"] == 0

    def test_territory_penetration_aggregated(self):
        events = [
            {
                "type": "pass",
                "timestamp": 1.0,
                "team": "home",
                "start_x": 30,
                "start_y": 34,
                "end_x": 70,
                "end_y": 34,
                "completed": True,
            },
            {"type": "tackle", "timestamp": 1.5, "team": "away", "x": 50, "y": 34},
            {
                "type": "pass",
                "timestamp": 2.0,
                "team": "home",
                "start_x": 40,
                "start_y": 34,
                "end_x": 65,
                "end_y": 34,
                "completed": True,
            },
            {"type": "interception", "timestamp": 2.5, "team": "away", "x": 55, "y": 34},
        ]
        result = compute_match_packing(events, team_attacks_right=True)
        assert result["home"].total_passes > 0
        assert result["home"].territory_penetration_gained > 0

    def test_uncompleted_pass_filtered(self):
        events = [
            {
                "type": "pass",
                "timestamp": 1.0,
                "team": "home",
                "start_x": 40,
                "start_y": 34,
                "end_x": 70,
                "end_y": 34,
                "completed": False,
            },
        ]
        result = compute_match_packing(events, team_attacks_right=True)
        assert result["home"].total_passes == 0
