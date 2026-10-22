"""Tests for Expected Assists model — Poisson pass-to-shot sequence model."""

from kawkab.core.xa_model import ExpectedAssistModel, XAMatchReport, XAResult
from kawkab.core.xg_model import EnhancedXgModel


class TestExpectedAssistModel:
    def test_zone_from_position_outside(self):
        model = ExpectedAssistModel()
        row, col = model._zone_from_position(100, 34)
        assert 0 <= row < model.rows
        assert 0 <= col < model.cols

    def test_zone_from_position_center(self):
        model = ExpectedAssistModel()
        row, col = model._zone_from_position(52.5, 34)
        assert 0 <= row < model.rows
        assert 0 <= col < model.cols

    def test_compute_xa_basic(self):
        model = ExpectedAssistModel()
        result = model.compute_xa(100, 34, "standard", 15)
        assert isinstance(result, XAResult)
        assert result.xa >= 0

    def test_xa_through_ball_higher_than_standard(self):
        model = ExpectedAssistModel()
        std = model.compute_xa(90, 34, "standard", 20)
        tb = model.compute_xa(90, 34, "through_ball", 20)
        assert tb.xa > std.xa

    def test_xa_progressive_higher(self):
        model = ExpectedAssistModel()
        normal = model.compute_xa(90, 34, "standard", 15, is_progressive=False)
        prog = model.compute_xa(90, 34, "standard", 15, is_progressive=True)
        assert prog.xa > normal.xa

    def test_xa_pressure_lower(self):
        model = ExpectedAssistModel()
        normal = model.compute_xa(90, 34, "standard", 15, under_pressure=False)
        pressed = model.compute_xa(90, 34, "standard", 15, under_pressure=True)
        assert pressed.xa < normal.xa

    def test_xa_central_higher_than_wide(self):
        model = ExpectedAssistModel()
        central = model.compute_xa(95, 34, "standard", 10)
        wide = model.compute_xa(95, 5, "standard", 10)
        assert central.xa > wide.xa

    def test_compute_match_xa(self):
        model = ExpectedAssistModel()
        events = [
            {
                "type": "pass",
                "team": "home",
                "end_x": 95,
                "end_y": 34,
                "start_x": 60,
                "start_y": 34,
                "pass_type": "through_ball",
                "is_progressive": True,
                "under_pressure": False,
                "timestamp": 10,
            },
            {
                "type": "pass",
                "team": "away",
                "end_x": 90,
                "end_y": 30,
                "start_x": 70,
                "start_y": 30,
                "pass_type": "standard",
                "is_progressive": False,
                "under_pressure": False,
                "timestamp": 20,
            },
            {"type": "shot", "team": "home"},
        ]
        report = model.compute_match_xa(events)
        assert isinstance(report, XAMatchReport)
        assert report.home_xa > 0
        assert report.away_xa > 0
        assert len(report.pass_details) == 2

    def test_empty_events(self):
        model = ExpectedAssistModel()
        report = model.compute_match_xa([])
        assert report.home_xa == 0
        assert report.away_xa == 0

    def test_to_dict(self):
        report = XAMatchReport(home_xa=1.2, away_xa=0.8)
        d = report.to_dict()
        assert d["home"] == 1.2
        assert d["away"] == 0.8
        assert d["total"] == 2.0


class TestSequenceModel:
    def test_shot_arrival_prob_in_result(self):
        model = ExpectedAssistModel()
        result = model.compute_xa(95, 34, "standard", 15)
        assert result.shot_arrival_prob > 0
        assert result.expected_shot_xg > 0
        assert result.sequence_xa > 0

    def test_sequence_xa_same_as_standard_by_default(self):
        model = ExpectedAssistModel()
        seq = model.compute_xa(95, 34, "standard", 15, use_sequence_model=True)
        legacy = model.compute_xa(95, 34, "standard", 15, use_sequence_model=False)
        assert seq.xa > 0
        assert legacy.xa > 0

    def test_legacy_mode_still_works(self):
        model = ExpectedAssistModel()
        result = model.compute_xa(90, 34, "cross", 20, use_sequence_model=False)
        assert result.xa > 0

    def test_custom_xg_model(self):
        xgm = EnhancedXgModel()
        model = ExpectedAssistModel(xg_model=xgm)
        result = model.compute_xa(100, 34, "standard", 10)
        assert 0 < result.xa < 1

    def test_match_report_includes_sequence_xa(self):
        model = ExpectedAssistModel()
        events = [
            {
                "type": "pass",
                "team": "home",
                "end_x": 95,
                "end_y": 34,
                "start_x": 60,
                "start_y": 34,
                "pass_type": "standard",
                "is_progressive": True,
                "under_pressure": False,
                "timestamp": 10,
            },
        ]
        report = model.compute_match_xa(events)
        assert report.home_sequence_xa >= 0
        d = report.to_dict()
        assert "home_sequence_xa" in d

    def test_pass_type_mult_mapping(self):
        model = ExpectedAssistModel()
        for ptype in ["standard", "through_ball", "cross", "long_ball"]:
            result = model.compute_xa(90, 34, ptype, 20)
            assert result.xa >= 0


class TestMonteCarlo:
    def test_mc_returns_basic_stats(self):
        model = ExpectedAssistModel()
        events = [
            {
                "type": "pass",
                "team": "home",
                "end_x": 95,
                "end_y": 34,
                "start_x": 60,
                "start_y": 34,
                "pass_type": "standard",
                "is_progressive": False,
                "under_pressure": False,
                "timestamp": 10,
            },
        ]
        result = model.monte_carlo_sequence_xa(events, n_simulations=500, seed=42)
        assert result["home_xa"] >= 0
        assert result["away_xa"] == 0
        assert result["total_xa"] >= 0
        assert result["n_passes"] == 1

    def test_mc_seeded_reproducible(self):
        model = ExpectedAssistModel()
        events = [
            {
                "type": "pass",
                "team": "home",
                "end_x": 95,
                "end_y": 34,
                "start_x": 60,
                "start_y": 34,
                "pass_type": "through_ball",
                "is_progressive": True,
                "under_pressure": False,
                "timestamp": 10,
            },
        ]
        a = model.monte_carlo_sequence_xa(events, n_simulations=1000, seed=42)
        b = model.monte_carlo_sequence_xa(events, n_simulations=1000, seed=42)
        assert a["total_xa"] == b["total_xa"]

    def test_mc_empty_events(self):
        model = ExpectedAssistModel()
        result = model.monte_carlo_sequence_xa([], n_simulations=100, seed=0)
        assert result["total_xa"] == 0
        assert result["n_passes"] == 0

    def test_mc_non_pass_filtered(self):
        model = ExpectedAssistModel()
        events = [
            {"type": "shot", "team": "home"},
            {
                "type": "pass",
                "team": "home",
                "end_x": 100,
                "end_y": 34,
                "start_x": 50,
                "start_y": 34,
                "pass_type": "cross",
                "is_progressive": False,
                "under_pressure": False,
                "timestamp": 5,
            },
        ]
        result = model.monte_carlo_sequence_xa(events, n_simulations=100, seed=0)
        assert result["n_passes"] == 1

    def test_mc_two_teams(self):
        model = ExpectedAssistModel()
        events = [
            {
                "type": "pass",
                "team": "home",
                "end_x": 95,
                "end_y": 34,
                "start_x": 60,
                "start_y": 34,
                "pass_type": "standard",
                "is_progressive": False,
                "under_pressure": False,
                "timestamp": 5,
            },
            {
                "type": "pass",
                "team": "away",
                "end_x": 90,
                "end_y": 30,
                "start_x": 70,
                "start_y": 30,
                "pass_type": "through_ball",
                "is_progressive": True,
                "under_pressure": False,
                "timestamp": 15,
            },
        ]
        result = model.monte_carlo_sequence_xa(events, n_simulations=500, seed=42)
        assert result["home_xa"] > 0
        assert result["away_xa"] > 0


class TestXAResult:
    def test_to_dict(self):
        r = XAResult(xa=0.12, base_prob=0.05, pass_type_mult=1.5, distance_factor=1.2)
        d = r.to_dict()
        assert d["xa"] == 0.12
        assert "shot_arrival_prob" in d

    def test_new_fields_present(self):
        r = XAResult()
        d = r.to_dict()
        assert "sequence_xa" in d
        assert "shot_arrival_prob" in d


class TestCrossSubtypes:
    def test_early_cross_multiplier(self):
        model = ExpectedAssistModel()
        std = model.compute_xa(95, 10, "cross", 15)
        early = model.compute_xa(95, 10, "cross", 15, cross_subtype="early")
        assert early.pass_type_mult == 1.7
        assert early.xa > std.xa

    def test_cutback_cross_multiplier(self):
        model = ExpectedAssistModel()
        std = model.compute_xa(95, 10, "cross", 15)
        cut = model.compute_xa(95, 10, "cross", 15, cross_subtype="cutback")
        assert cut.pass_type_mult == 1.8
        assert cut.xa > std.xa

    def test_driven_cross_multiplier(self):
        model = ExpectedAssistModel()
        std = model.compute_xa(95, 10, "cross", 15)
        driven = model.compute_xa(95, 10, "cross", 15, cross_subtype="driven")
        assert driven.pass_type_mult == 1.4
        assert driven.xa < std.xa

    def test_lofted_cross_multiplier(self):
        model = ExpectedAssistModel()
        std = model.compute_xa(95, 10, "cross", 15)
        lofted = model.compute_xa(95, 10, "cross", 15, cross_subtype="lofted")
        assert lofted.pass_type_mult == 1.3
        assert lofted.xa < std.xa

    def test_unknown_cross_subtype_falls_back_to_default(self):
        model = ExpectedAssistModel()
        std = model.compute_xa(95, 10, "cross", 15)
        unknown = model.compute_xa(95, 10, "cross", 15, cross_subtype="fizz")
        assert unknown.pass_type_mult == 1.5
        assert unknown.xa == std.xa

    def test_cross_subtype_ignored_for_non_cross_passes(self):
        model = ExpectedAssistModel()
        std = model.compute_xa(95, 34, "standard", 15)
        with_sub = model.compute_xa(95, 34, "standard", 15, cross_subtype="early")
        assert with_sub.pass_type_mult == 1.0
        assert with_sub.xa == std.xa

    def test_cross_subtype_from_event_dict(self):
        model = ExpectedAssistModel()
        event = {
            "end_x": 95,
            "end_y": 10,
            "start_x": 70,
            "start_y": 10,
            "pass_type": "cross",
            "cross_subtype": "cutback",
            "is_progressive": False,
            "under_pressure": False,
        }
        result = model.compute_pass_xa(event)
        assert result.pass_type_mult == 1.8

    def test_cross_assist_with_subtype(self):
        model = ExpectedAssistModel()
        std = model.compute_xa(95, 10, "cross_assist", 15)
        cut = model.compute_xa(95, 10, "cross_assist", 15, cross_subtype="cutback")
        assert cut.pass_type_mult == 1.8
        assert cut.xa > std.xa
