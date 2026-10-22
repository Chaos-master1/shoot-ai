"""Tests for Expected Threat (xT) model."""

import numpy as np

from kawkab.core.xt_model import ExpectedThreatModel


class TestExpectedThreatModel:
    def test_zone_from_position(self):
        model = ExpectedThreatModel(rows=5, cols=4)
        # Middle of pitch
        assert model._zone_from_position(52.5, 34.0) == (2, 2)
        # Near home goal
        assert model._zone_from_position(0.0, 0.0) == (0, 0)
        # Near away goal
        assert model._zone_from_position(105.0, 68.0) == (4, 3)

    def test_transition_matrix_builds(self):
        model = ExpectedThreatModel(rows=5, cols=4)
        events = [
            {
                "type": "pass",
                "completed": True,
                "start_x": 20,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
            },
            {
                "type": "pass",
                "completed": True,
                "start_x": 50,
                "start_y": 34,
                "end_x": 80,
                "end_y": 34,
            },
            {
                "type": "carry",
                "completed": True,
                "start_x": 10,
                "start_y": 20,
                "end_x": 30,
                "end_y": 20,
            },
        ]
        model.build_transition_matrix(events)
        assert model._transition is not None
        assert len(model._transition) > 0

    def test_threat_increases_towards_goal(self):
        model = ExpectedThreatModel(rows=5, cols=4)
        events = [
            {
                "type": "pass",
                "completed": True,
                "start_x": 20,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
            },
            {
                "type": "pass",
                "completed": True,
                "start_x": 50,
                "start_y": 34,
                "end_x": 85,
                "end_y": 34,
            },
            {"type": "shot", "start_x": 85, "start_y": 34, "is_goal": True},
        ]
        model.build_transition_matrix(events)
        zone_vals = model.get_zone_values()
        # Last column (near goal) should have higher threat than first column
        assert np.mean(zone_vals[:, 3]) >= np.mean(zone_vals[:, 0])

    def test_action_xt_positive_for_forward_pass(self):
        model = ExpectedThreatModel(rows=5, cols=4)
        events = [
            {
                "type": "pass",
                "completed": True,
                "start_x": 20,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
            },
            {
                "type": "pass",
                "completed": True,
                "start_x": 50,
                "start_y": 34,
                "end_x": 80,
                "end_y": 34,
            },
        ]
        model.build_transition_matrix(events)
        xt = model.compute_action_xt(20, 34, 80, 34)
        assert xt >= 0.0

    def test_action_xt_zero_for_no_change(self):
        model = ExpectedThreatModel(rows=5, cols=4)
        events = [
            {
                "type": "pass",
                "completed": True,
                "start_x": 30,
                "start_y": 34,
                "end_x": 30,
                "end_y": 34,
            }
        ]
        model.build_transition_matrix(events)
        xt = model.compute_action_xt(30, 34, 30, 34)
        assert xt == 0.0

    def test_match_xt_returns_dict(self):
        model = ExpectedThreatModel(rows=5, cols=4)
        events = [
            {
                "type": "pass",
                "completed": True,
                "start_x": 20,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
                "team": "home",
            },
            {
                "type": "pass",
                "completed": True,
                "start_x": 50,
                "start_y": 34,
                "end_x": 80,
                "end_y": 34,
                "team": "away",
            },
        ]
        result = model.compute_match_xt(events)
        assert "home" in result
        assert "away" in result

    def test_empty_events(self):
        model = ExpectedThreatModel(rows=5, cols=4)
        result = model.compute_match_xt([])
        assert result["home"] == 0.0
        assert result["away"] == 0.0

    def test_zone_grid_output_shape(self):
        model = ExpectedThreatModel(rows=5, cols=4)
        events = [
            {
                "type": "pass",
                "completed": True,
                "start_x": 20,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
            }
        ]
        model.build_transition_matrix(events)
        grid = model.get_zone_grid()
        assert len(grid) == 5
        assert len(grid[0]) == 4
