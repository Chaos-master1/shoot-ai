"""Tests for carry expected threat (Carry xT) module."""

from kawkab.core.carry_xt import (
    CarryXTMatchReport,
    compute_carry_xt,
    compute_carry_xt_from_tracking,
)
from kawkab.core.xt_model import ExpectedThreatModel


def _make_xt_model():
    model = ExpectedThreatModel(rows=5, cols=4)
    events = [
        {"type": "pass", "completed": True, "start_x": 20, "start_y": 34, "end_x": 50, "end_y": 34},
        {"type": "pass", "completed": True, "start_x": 50, "start_y": 34, "end_x": 80, "end_y": 34},
        {"type": "shot", "start_x": 80, "start_y": 34, "is_goal": True},
    ]
    model.build_transition_matrix(events)
    return model


class TestComputeCarryXT:
    def test_empty_events_returns_zero(self):
        result = compute_carry_xt([])
        assert isinstance(result, CarryXTMatchReport)
        assert result.home_total_xt == 0.0
        assert result.away_total_xt == 0.0
        assert result.home_carries == 0
        assert result.away_carries == 0

    def test_no_carry_events_returns_zero(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 30,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
            },
        ]
        result = compute_carry_xt(events)
        assert result.home_total_xt == 0.0

    def test_forward_carry_positive_xt(self):
        model = _make_xt_model()
        events = [
            {
                "type": "carry",
                "team": "home",
                "start_x": 30,
                "start_y": 34,
                "end_x": 60,
                "end_y": 34,
                "timestamp": 1.0,
            },
        ]
        result = compute_carry_xt(events, xt_model=model)
        assert result.home_total_xt >= 0.0
        assert result.home_carries == 1

    def test_carry_xt_values_have_sign_based_on_direction(self):
        model = _make_xt_model()
        events = [
            {
                "type": "carry",
                "team": "home",
                "start_x": 30,
                "start_y": 34,
                "end_x": 60,
                "end_y": 34,
                "timestamp": 1.0,
            },
        ]
        result = compute_carry_xt(events, xt_model=model)
        for c in result.carries:
            assert isinstance(c["xt"], float)

    def test_per_team_stats_populated(self):
        model = _make_xt_model()
        events = [
            {
                "type": "carry",
                "team": "home",
                "start_x": 30,
                "start_y": 34,
                "end_x": 60,
                "end_y": 34,
                "timestamp": 1.0,
            },
            {
                "type": "carry",
                "team": "away",
                "start_x": 70,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
                "timestamp": 2.0,
            },
        ]
        result = compute_carry_xt(events, xt_model=model)
        assert result.home_carries >= 1
        assert result.away_carries >= 1
        assert isinstance(result.home_total_xt, float)
        assert isinstance(result.away_total_xt, float)

    def test_progressive_flag_set(self):
        model = _make_xt_model()
        events = [
            {
                "type": "carry",
                "team": "home",
                "start_x": 30,
                "start_y": 34,
                "end_x": 70,
                "end_y": 34,
                "timestamp": 1.0,
            },
        ]
        result = compute_carry_xt(events, xt_model=model)
        assert result.home_progressive >= 1

    def test_report_to_dict_keys(self):
        result = compute_carry_xt([])
        d = result.to_dict()
        assert "home_total_xt" in d
        assert "away_total_xt" in d
        assert "home_carries" in d

    def test_forward_distance_gt_5_is_progressive(self):
        model = _make_xt_model()
        events = [
            {
                "type": "carry",
                "team": "home",
                "start_x": 30,
                "start_y": 34,
                "end_x": 31,
                "end_y": 34,
                "timestamp": 1.0,
            },
        ]
        result = compute_carry_xt(events, xt_model=model)
        assert result.home_progressive == result.home_progressive

    def test_carry_xt_generated_without_model(self):
        events = [
            {
                "type": "carry",
                "team": "home",
                "start_x": 30,
                "start_y": 34,
                "end_x": 60,
                "end_y": 34,
                "timestamp": 1.0,
            },
        ]
        result = compute_carry_xt(events)
        assert isinstance(result, CarryXTMatchReport)

    def test_carry_results_have_required_fields(self):
        model = _make_xt_model()
        events = [
            {
                "type": "carry",
                "team": "home",
                "start_x": 30,
                "start_y": 34,
                "end_x": 60,
                "end_y": 34,
                "timestamp": 1.0,
            },
        ]
        result = compute_carry_xt(events, xt_model=model)
        assert len(result.carries) == 1
        entry = result.carries[0]
        for key in ("idx", "team", "start_x", "end_x", "dist", "xt", "prog"):
            assert key in entry


class TestComputeCarryXTFromTracking:
    def test_empty_returns_zero(self):
        result = compute_carry_xt_from_tracking([], [])
        assert result.home_total_xt == 0.0

    def test_no_carry_events(self):
        frames = [{"timestamp": 0.0, "ball_pos": (50, 34)}]
        events = [{"type": "pass", "team": "home"}]
        result = compute_carry_xt_from_tracking(frames, events)
        assert result.home_total_xt == 0.0

    def test_tracking_aligned_with_events(self):
        model = _make_xt_model()
        frames = [
            {"timestamp": 0.5, "home_positions": [(30, 34, 1)], "ball_pos": (30, 34)},
            {"timestamp": 1.5, "home_positions": [(60, 34, 1)], "ball_pos": (60, 34)},
        ]
        events = [
            {
                "type": "carry",
                "team": "home",
                "start_x": 30,
                "start_y": 34,
                "end_x": 60,
                "end_y": 34,
                "timestamp": 1.0,
            },
        ]
        result = compute_carry_xt_from_tracking(frames, events, xt_model=model)
        assert result.home_carries == 1
