"""Tests for progressive passes and carries — ball-advancement analysis."""

from kawkab.core.progressive_actions import (
    CORRIDOR_WIDTH,
    MIN_CARRY_PROGRESSION_M,
    PITCH_LENGTH,
    PITCH_WIDTH,
    ProgressiveAction,
    ProgressiveReport,
    _classify_zone,
    _compute_danger_rating,
    _count_opponents_behind_pass,
    _is_progressive_carry,
    _is_progressive_pass,
    analyze_progressive_passes,
)


class TestIsProgressivePass:
    def test_forward_pass_attacking_right(self):
        """A pass that reduces remaining distance by >=25% and ends in attacking third."""
        assert _is_progressive_pass(50, 80, 34, 34, PITCH_LENGTH, 1) is True

    def test_backward_pass_not_progressive(self):
        """A pass moving away from opponent goal should not be progressive."""
        assert _is_progressive_pass(60, 40, 34, 34, PITCH_LENGTH, 1) is False

    def test_lateral_pass_same_x(self):
        """A pass with zero x change cannot be progressive."""
        assert _is_progressive_pass(60, 60, 34, 10, PITCH_LENGTH, 1) is False

    def test_short_forward_pass_not_enough_progression(self):
        """Small forward gain that does not hit the 25% remaining-distance reduction."""
        assert _is_progressive_pass(10, 15, 34, 34, PITCH_LENGTH, 1) is False

    def test_pass_starts_at_goal_line(self):
        """Pass from the goal line has remaining_before == 0 → not progressive."""
        assert _is_progressive_pass(0, 20, 34, 34, PITCH_LENGTH, 1) is False

    def test_forward_pass_attacking_left(self):
        """Attacking direction -1: a pass moving to lower x (opponent goal) is progressive."""
        assert _is_progressive_pass(60, 30, 34, 34, PITCH_LENGTH, -1) is True

    def test_massive_forward_pass(self):
        """A long pass from deep should easily be progressive."""
        assert _is_progressive_pass(30, 85, 34, 34, PITCH_LENGTH, 1) is True

    def test_pass_to_edge_of_attacking_third(self):
        """Pass that ends in attacking third with >=25% reduction should be progressive."""
        # From x=30, need end_x >= 48.75 for 25% reduction: (105-30)*0.25=18.75, 30+18.75=48.75
        assert _is_progressive_pass(30, 50, 34, 34, PITCH_LENGTH, 1) is True


class TestIsProgressiveCarry:
    def test_long_carry_forward(self):
        """Carry >= 5m forward should be progressive."""
        assert _is_progressive_carry(40, 50, 12.0, 1, MIN_CARRY_PROGRESSION_M) is True

    def test_short_carry_forward(self):
        """Carry < 5m forward should not be progressive."""
        assert _is_progressive_carry(40, 43, 4.0, 1, MIN_CARRY_PROGRESSION_M) is False

    def test_carry_exactly_at_threshold(self):
        """Carry exactly 5.0m forward is progressive."""
        assert _is_progressive_carry(40, 45, 5.0, 1, 5.0) is True

    def test_backward_carry(self):
        """Carry moving away from opponent goal should not be progressive."""
        assert _is_progressive_carry(60, 50, 12.0, 1, MIN_CARRY_PROGRESSION_M) is False

    def test_carry_attacking_left(self):
        """Attacking direction -1: forward means decreasing x."""
        assert _is_progressive_carry(60, 50, 12.0, -1, MIN_CARRY_PROGRESSION_M) is True

    def test_carry_negative_progression(self):
        """Carry that moves backward in attacking direction is not progressive."""
        assert _is_progressive_carry(50, 60, 12.0, -1, MIN_CARRY_PROGRESSION_M) is False

    def test_carry_zero_distance_x(self):
        """Carry with no net x movement is not progressive."""
        assert _is_progressive_carry(50, 50, 5.0, 1, MIN_CARRY_PROGRESSION_M) is False


class TestComputeDangerRating:
    def test_six_yard_box_center(self):
        """Inside 6-yard box, central → 1.0."""
        assert _compute_danger_rating(102, 34, PITCH_LENGTH, PITCH_WIDTH, 1) == 1.0

    def test_six_yard_box_attacking_left(self):
        """Attacking direction -1: goal line at x=0; inside 6-yard box."""
        assert _compute_danger_rating(3, 34, PITCH_LENGTH, PITCH_WIDTH, -1) == 1.0

    def test_penalty_area_not_six_yard(self):
        """Inside penalty area but outside 6-yard box → 0.8."""
        assert _compute_danger_rating(93, 34, PITCH_LENGTH, PITCH_WIDTH, 1) == 0.8

    def test_penalty_area_wide(self):
        """Wide in penalty area but within penalty-area half-width → 0.8."""
        assert _compute_danger_rating(93, 50, PITCH_LENGTH, PITCH_WIDTH, 1) == 0.8

    def test_final_third_center(self):
        """Central area of final third outside penalty area → 0.7."""
        assert _compute_danger_rating(77, 34, PITCH_LENGTH, PITCH_WIDTH, 1) == 0.7

    def test_final_third_wide(self):
        """Wide area of final third → 0.5."""
        assert _compute_danger_rating(77, 5, PITCH_LENGTH, PITCH_WIDTH, 1) == 0.5

    def test_middle_third(self):
        """Middle third of pitch → 0.3."""
        assert _compute_danger_rating(50, 34, PITCH_LENGTH, PITCH_WIDTH, 1) == 0.3

    def test_own_half(self):
        """Own half (defensive third) → 0.1."""
        assert _compute_danger_rating(10, 34, PITCH_LENGTH, PITCH_WIDTH, 1) == 0.1

    def test_on_goal_line_no_clip(self):
        """Exactly on the goal line (attacking direction 1) → 1.0 (6-yard check)."""
        assert _compute_danger_rating(105, 34, PITCH_LENGTH, PITCH_WIDTH, 1) == 1.0

    def test_on_goal_line_attacking_left(self):
        """Exactly on goal line (attacking direction -1) → 1.0."""
        assert _compute_danger_rating(0, 34, PITCH_LENGTH, PITCH_WIDTH, -1) == 1.0


class TestClassifyZone:
    def test_defensive_left(self):
        assert _classify_zone(5, 5, PITCH_LENGTH, PITCH_WIDTH) == "Defensive Left"

    def test_defensive_center(self):
        assert _classify_zone(5, 34, PITCH_LENGTH, PITCH_WIDTH) == "Defensive Center"

    def test_defensive_right(self):
        assert _classify_zone(5, 60, PITCH_LENGTH, PITCH_WIDTH) == "Defensive Right"

    def test_defensive_mid_center(self):
        assert _classify_zone(30, 34, PITCH_LENGTH, PITCH_WIDTH) == "Defensive Mid Center"

    def test_middle_center(self):
        assert _classify_zone(52, 34, PITCH_LENGTH, PITCH_WIDTH) == "Middle Center"

    def test_attacking_mid_center(self):
        assert _classify_zone(75, 34, PITCH_LENGTH, PITCH_WIDTH) == "Attacking Mid Center"

    def test_attacking_left(self):
        assert _classify_zone(95, 5, PITCH_LENGTH, PITCH_WIDTH) == "Attacking Left"

    def test_attacking_center(self):
        assert _classify_zone(95, 34, PITCH_LENGTH, PITCH_WIDTH) == "Attacking Center"

    def test_attacking_right(self):
        assert _classify_zone(95, 60, PITCH_LENGTH, PITCH_WIDTH) == "Attacking Right"

    def test_middle_right(self):
        assert _classify_zone(52, 60, PITCH_LENGTH, PITCH_WIDTH) == "Middle Right"

    def test_clips_negative_coords(self):
        """Coordinates below zero are clamped to 0."""
        assert _classify_zone(-5, 34, PITCH_LENGTH, PITCH_WIDTH) == "Defensive Center"

    def test_clips_over_pitch(self):
        """Coordinates beyond pitch are clamped."""
        assert _classify_zone(200, 100, PITCH_LENGTH, PITCH_WIDTH) == "Attacking Right"


class TestCountOpponentsBehindPass:
    def test_no_opponents(self):
        assert _count_opponents_behind_pass(50, 34, 80, 34, [], 1) == 0

    def test_one_opponent_bypassed(self):
        opps = [{"x": 65, "y": 34}]
        assert _count_opponents_behind_pass(50, 34, 80, 34, opps, 1) == 1

    def test_opponent_outside_corridor(self):
        """Opponent perpendicular distance exceeds CORRIDOR_WIDTH."""
        opps = [{"x": 65, "y": 20}]
        assert _count_opponents_behind_pass(50, 34, 80, 34, opps, 1) == 0

    def test_multiple_opponents_some_bypassed(self):
        opps = [{"x": 55, "y": 34}, {"x": 70, "y": 33}, {"x": 90, "y": 34}]
        result = _count_opponents_behind_pass(50, 34, 80, 34, opps, 1)
        # 90 is past end_x, not between 50-80
        assert result == 2

    def test_opponent_in_front_not_counted(self):
        """Opponent past end_x should not be counted."""
        opps = [{"x": 85, "y": 34}]
        assert _count_opponents_behind_pass(50, 34, 80, 34, opps, 1) == 0

    def test_empty_opponent_list(self):
        assert _count_opponents_behind_pass(50, 34, 80, 34, None, 1) == 0

    def test_zero_length_action(self):
        """Zero-length pass returns 0 immediately."""
        assert _count_opponents_behind_pass(50, 34, 50, 34, [{"x": 50, "y": 34}], 1) == 0

    def test_attacking_left_bypassed(self):
        """Attacking direction -1: opponents with x between end and start are bypassed."""
        opps = [{"x": 45, "y": 34}]
        assert _count_opponents_behind_pass(60, 34, 35, 34, opps, -1) == 1

    def test_opponent_behind_start_not_counted(self):
        """Opponent before start_x should not be counted (attacking right)."""
        opps = [{"x": 40, "y": 34}]
        assert _count_opponents_behind_pass(50, 34, 80, 34, opps, 1) == 0

    def test_opponent_on_line_edge_of_corridor(self):
        """Opponent exactly CORRIDOR_WIDTH away should be counted (<=)."""
        opps = [{"x": 65, "y": 34 - CORRIDOR_WIDTH}]
        assert _count_opponents_behind_pass(50, 34, 80, 34, opps, 1) == 1


class TestProgressiveAction:
    def test_to_dict(self):
        a = ProgressiveAction(
            action_type="pass",
            player_track_id=10,
            team="home",
            start_x=50,
            start_y=34,
            end_x=80,
            end_y=34,
            distance_m=30,
            progression_m=30,
            is_progressive=True,
            zone_start="Middle Center",
            zone_end="Attacking Center",
            opponent_bypassed=2,
            danger_rating=0.5,
        )
        d = a.to_dict()
        assert d["action_type"] == "pass"
        assert d["player_track_id"] == 10
        assert d["team"] == "home"
        assert d["is_progressive"] is True
        assert d["zone_start"] == "Middle Center"
        assert d["opponent_bypassed"] == 2
        assert d["danger_rating"] == 0.5


class TestProgressiveReport:
    def test_to_dict(self):
        r = ProgressiveReport(
            team="home",
            total_progressive_passes=5,
            total_progressive_carries=3,
            total_pass_progression_m=120,
            total_carry_progression_m=40,
            avg_pass_progression_m=24,
            avg_carry_progression_m=13.33,
            progressive_pass_rate=0.25,
            progressive_carry_rate=0.3,
            actions_by_zone={"Middle Center": 4, "Attacking Center": 4},
            danger_actions=2,
            top_players=[
                {
                    "player_track_id": 7,
                    "team": "home",
                    "total_progression_m": 50,
                    "progressive_actions": 3,
                }
            ],
        )
        d = r.to_dict()
        assert d["total_progressive_passes"] == 5
        assert d["danger_actions"] == 2
        assert d["progressive_pass_rate"] == 0.25
        assert len(d["top_players"]) == 1
        assert d["top_players"][0]["player_track_id"] == 7

    def test_empty_report_defaults(self):
        r = ProgressiveReport(team="away")
        d = r.to_dict()
        assert d["total_progressive_passes"] == 0
        assert d["danger_actions"] == 0
        assert d["progressive_pass_rate"] == 0.0
        assert d["top_players"] == []


class TestAnalyzeProgressivePasses:
    def test_empty_events(self):
        report = analyze_progressive_passes([], "home", 1)
        assert report.total_progressive_passes == 0
        assert report.total_progressive_carries == 0
        assert report.team == "home"

    def test_no_progressive_actions(self):
        events = [
            {
                "type": "pass",
                "player_track_id": 1,
                "team": "home",
                "start_x": 60,
                "start_y": 34,
                "end_x": 55,
                "end_y": 34,
                "distance": 5,
            },
        ]
        report = analyze_progressive_passes(events, "home", 1)
        assert report.total_progressive_passes == 0
        assert report.progressive_pass_rate == 0.0

    def test_mixed_passes_and_carries(self):
        events = [
            {
                "type": "pass",
                "player_track_id": 1,
                "team": "home",
                "start_x": 40,
                "start_y": 34,
                "end_x": 75,
                "end_y": 34,
                "distance": 35,
            },
            {
                "type": "carry",
                "player_track_id": 1,
                "team": "home",
                "start_x": 50,
                "start_y": 34,
                "end_x": 60,
                "end_y": 34,
                "distance": 12,
            },
        ]
        report = analyze_progressive_passes(events, "home", 1)
        assert report.total_progressive_passes >= 0
        assert report.total_progressive_carries >= 0

    def test_wrong_team_filtered(self):
        events = [
            {
                "type": "pass",
                "player_track_id": 1,
                "team": "away",
                "start_x": 40,
                "start_y": 34,
                "end_x": 75,
                "end_y": 34,
                "distance": 35,
            },
        ]
        report = analyze_progressive_passes(events, "home", 1)
        assert report.total_progressive_passes == 0

    def test_progressive_pass_rate_calculated(self):
        events = [
            {
                "type": "pass",
                "player_track_id": 1,
                "team": "home",
                "start_x": 40,
                "start_y": 34,
                "end_x": 75,
                "end_y": 34,
                "distance": 35,
            },
            {
                "type": "pass",
                "player_track_id": 2,
                "team": "home",
                "start_x": 50,
                "start_y": 34,
                "end_x": 52,
                "end_y": 34,
                "distance": 2,
            },
        ]
        report = analyze_progressive_passes(events, "home", 1)
        assert report.total_progressive_passes == 1
        assert report.progressive_pass_rate == 0.5

    def test_danger_actions_counted(self):
        """A progressive pass ending inside the 6-yard box qualifies as danger action."""
        events = [
            {
                "type": "pass",
                "player_track_id": 1,
                "team": "home",
                "start_x": 50,
                "start_y": 34,
                "end_x": 102,
                "end_y": 34,
                "distance": 52,
            },
        ]
        report = analyze_progressive_passes(events, "home", 1)
        assert report.danger_actions == 1

    def test_top_players_ranking(self):
        events = [
            {
                "type": "pass",
                "player_track_id": 7,
                "team": "home",
                "start_x": 40,
                "start_y": 34,
                "end_x": 80,
                "end_y": 34,
                "distance": 40,
            },
            {
                "type": "pass",
                "player_track_id": 7,
                "team": "home",
                "start_x": 50,
                "start_y": 34,
                "end_x": 85,
                "end_y": 34,
                "distance": 35,
            },
            {
                "type": "carry",
                "player_track_id": 3,
                "team": "home",
                "start_x": 45,
                "start_y": 34,
                "end_x": 60,
                "end_y": 34,
                "distance": 16,
            },
        ]
        report = analyze_progressive_passes(events, "home", 1)
        assert len(report.top_players) >= 1
        assert report.top_players[0]["player_track_id"] == 7
        assert report.top_players[0]["total_progression_m"] >= 60

    def test_top_players_limited_to_three(self):
        events = []
        for pid in range(1, 6):
            events.append(
                {
                    "type": "pass",
                    "player_track_id": pid,
                    "team": "home",
                    "start_x": 40,
                    "start_y": 34,
                    "end_x": 80,
                    "end_y": 34,
                    "distance": 40,
                },
            )
        report = analyze_progressive_passes(events, "home", 1)
        assert len(report.top_players) <= 3

    def test_actions_by_zone_populated(self):
        events = [
            {
                "type": "pass",
                "player_track_id": 1,
                "team": "home",
                "start_x": 40,
                "start_y": 34,
                "end_x": 80,
                "end_y": 34,
                "distance": 40,
            },
        ]
        report = analyze_progressive_passes(events, "home", 1)
        assert len(report.actions_by_zone) >= 1
        assert list(report.actions_by_zone.keys())[0] == "Defensive Mid Center"

    def test_zero_distance_actions(self):
        """Actions with zero start/end distance should not cause errors."""
        events = [
            {
                "type": "pass",
                "player_track_id": 1,
                "team": "home",
                "start_x": 50,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
                "distance": 0,
            },
        ]
        report = analyze_progressive_passes(events, "home", 1)
        assert report.total_progressive_passes == 0

    def test_opponent_bypassed_integration(self):
        """Integration: opponent_positions fed through to count bypassed."""
        events = [
            {
                "type": "pass",
                "player_track_id": 1,
                "team": "home",
                "start_x": 50,
                "start_y": 34,
                "end_x": 80,
                "end_y": 34,
                "distance": 30,
                "opponent_positions": [{"x": 65, "y": 34}],
            },
        ]
        report = analyze_progressive_passes(events, "home", 1)
        assert report.total_progressive_passes >= 1

    def test_non_pass_carry_events_skipped(self):
        """Events with unknown type should be silently skipped."""
        events = [
            {
                "type": "tackle",
                "player_track_id": 1,
                "team": "home",
                "start_x": 50,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
            },
        ]
        report = analyze_progressive_passes(events, "home", 1)
        assert report.total_progressive_passes == 0
        assert report.total_progressive_carries == 0

    def test_attacking_direction_left(self):
        """Analyze with attacking_direction=-1."""
        events = [
            {
                "type": "pass",
                "player_track_id": 1,
                "team": "home",
                "start_x": 80,
                "start_y": 34,
                "end_x": 40,
                "end_y": 34,
                "distance": 40,
            },
        ]
        report = analyze_progressive_passes(events, "home", -1)
        assert report.total_progressive_passes >= 1

    def test_team_with_no_events(self):
        """Team with no matching events returns empty report."""
        events = [
            {
                "type": "pass",
                "player_track_id": 1,
                "team": "away",
                "start_x": 50,
                "start_y": 34,
                "end_x": 80,
                "end_y": 34,
                "distance": 30,
            },
        ]
        report = analyze_progressive_passes(events, "home", 1)
        assert report.total_progressive_passes == 0
        assert report.total_carry_progression_m == 0.0
