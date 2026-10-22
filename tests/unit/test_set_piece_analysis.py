"""Tests for set piece analysis module — delivery quality scoring."""

import pytest

from kawkab.core.set_piece_analysis import (
    DeliveryZone,
    SetPieceReport,
    SetPieceSummary,
    _classify_set_piece_type,
    _compute_delivery_quality,
    _delivery_zone_label,
    _estimate_set_piece_xg,
    analyze_set_pieces,
)
from kawkab.core.xg_model import EnhancedXgModel


class TestSetPieceAnalysis:
    def test_empty_events(self):
        report = analyze_set_pieces([])
        assert report.total_set_pieces == 0
        assert isinstance(report, SetPieceReport)

    def test_classify_corner_kick(self):
        sp = _classify_set_piece_type({"event_type": "corner_kick"})
        assert sp == "corner_kick"

    def test_classify_free_kick(self):
        sp = _classify_set_piece_type({"event_type": "free_kick"})
        assert sp == "free_kick"

    def test_classify_penalty(self):
        sp = _classify_set_piece_type({"event_type": "penalty"})
        assert sp == "penalty"

    def test_classify_normal_shot(self):
        sp = _classify_set_piece_type({"event_type": "shot", "type": "shot", "metadata": {}})
        assert sp is None

    def test_classify_set_piece_shot(self):
        sp = _classify_set_piece_type(
            {"event_type": "shot", "type": "shot", "metadata": {"set_piece": "corner_kick"}}
        )
        assert sp == "corner_kick"

    def test_delivery_zone_near_post(self):
        z = _delivery_zone_label(95, 10)
        assert z == "Near Post"

    def test_delivery_zone_far_post(self):
        z = _delivery_zone_label(95, 55)
        assert z == "Far Post"

    def test_delivery_zone_deep(self):
        z = _delivery_zone_label(10, 34)
        assert z == "Deep"

    def test_delivery_zone_edge_box(self):
        z = _delivery_zone_label(70, 34)
        assert z == "Edge of Box"

    def test_delivery_zone_midfield(self):
        z = _delivery_zone_label(50, 34)
        assert z == "Midfield"

    def test_single_corner_no_shot(self):
        events = [
            {
                "event_type": "corner_kick",
                "type": "corner_kick",
                "team": "home",
                "x": 1,
                "y": 1,
                "xg": 0.0,
                "is_goal": False,
                "timestamp": 10,
            }
        ]
        report = analyze_set_pieces(events)
        assert report.total_set_pieces == 1
        assert report.home_set_pieces == 1
        assert len(report.summaries) == 1
        assert report.summaries[0]["type"] == "corner_kick"
        assert report.summaries[0]["count"] == 1

    def test_corner_with_goal(self):
        events = [
            {
                "event_type": "corner_kick",
                "type": "corner_kick",
                "team": "home",
                "x": 1,
                "y": 1,
                "xg": 0.0,
                "is_goal": False,
                "timestamp": 10,
            },
            {
                "event_type": "shot",
                "type": "shot",
                "team": "home",
                "x": 92,
                "y": 34,
                "xg": 0.35,
                "is_goal": True,
                "timestamp": 12,
                "metadata": {"set_piece": "corner_kick"},
            },
        ]
        report = analyze_set_pieces(events)
        assert report.total_set_pieces == 2
        assert report.home_goals == 1
        assert report.home_total_xg == pytest.approx(0.35, rel=0.01)
        corner = [s for s in report.summaries if s["type"] == "corner_kick"]
        assert len(corner) == 1
        assert corner[0]["shots"] >= 1

    def test_report_to_dict(self):
        report = SetPieceReport(total_set_pieces=5, home_set_pieces=3, away_set_pieces=2)
        d = report.to_dict()
        assert d["total_set_pieces"] == 5
        assert d["home_set_pieces"] == 3

    def test_both_teams_set_pieces(self):
        events = [
            {
                "event_type": "free_kick",
                "type": "free_kick",
                "team": "home",
                "x": 80,
                "y": 34,
                "xg": 0.0,
                "is_goal": False,
                "timestamp": 5,
            },
            {
                "event_type": "free_kick",
                "type": "free_kick",
                "team": "away",
                "x": 20,
                "y": 34,
                "xg": 0.0,
                "is_goal": False,
                "timestamp": 15,
            },
            {
                "event_type": "corner_kick",
                "type": "corner_kick",
                "team": "home",
                "x": 1,
                "y": 1,
                "xg": 0.0,
                "is_goal": False,
                "timestamp": 25,
            },
        ]
        report = analyze_set_pieces(events)
        assert report.total_set_pieces == 3
        assert report.home_set_pieces == 2
        assert report.away_set_pieces == 1

    def test_set_piece_summary_to_dict(self):
        s = SetPieceSummary(
            type="corner_kick",
            count=5,
            shots=3,
            goals=1,
            total_xg=0.75,
            avg_xg_per_set_piece=0.15,
            conversion_rate=0.2,
            threat_rating=0.45,
        )
        d = s.to_dict()
        assert d["type"] == "corner_kick"
        assert d["count"] == 5
        assert d["shots"] == 3
        assert d["goals"] == 1
        assert d["total_xg"] == 0.75
        assert d["conversion_rate"] == 0.2

    def test_delivery_zone_to_dict(self):
        z = DeliveryZone(label="Near Post", count=4, shots=2, goals=1, total_xg=0.35)
        d = z.to_dict()
        assert d["label"] == "Near Post"
        assert d["count"] == 4
        assert d["shots"] == 2
        assert d["goals"] == 1
        assert d["total_xg"] == 0.35

    def test_multiple_delivery_zones(self):
        events = [
            {
                "event_type": "corner_kick",
                "type": "corner_kick",
                "team": "home",
                "x": 1,
                "y": 1,
                "pass_end_x": 95,
                "pass_end_y": 10,
                "xg": 0.0,
                "is_goal": False,
                "timestamp": 10,
            },
            {
                "event_type": "corner_kick",
                "type": "corner_kick",
                "team": "home",
                "x": 1,
                "y": 1,
                "pass_end_x": 95,
                "pass_end_y": 55,
                "xg": 0.0,
                "is_goal": False,
                "timestamp": 20,
            },
            {
                "event_type": "free_kick",
                "type": "free_kick",
                "team": "away",
                "x": 80,
                "y": 34,
                "pass_end_x": 70,
                "pass_end_y": 34,
                "xg": 0.0,
                "is_goal": False,
                "timestamp": 30,
            },
        ]
        report = analyze_set_pieces(events)
        zone_labels = [z["label"] for z in report.delivery_zones]
        assert "Near Post" in zone_labels
        assert "Far Post" in zone_labels
        assert "Edge of Box" in zone_labels
        assert len(report.delivery_zones) == 3


class TestDeliveryQuality:
    def test_delivery_quality_range(self):
        q = _compute_delivery_quality(95, 34, "corner_kick")
        assert 0.0 <= q <= 1.0

    def test_near_goal_higher_quality(self):
        near = _compute_delivery_quality(100, 34, "corner_kick")
        far = _compute_delivery_quality(60, 34, "corner_kick")
        assert near > far

    def test_corner_near_post_vs_deep(self):
        near = _compute_delivery_quality(95, 10, "corner_kick")
        deep = _compute_delivery_quality(20, 34, "corner_kick")
        assert near > deep

    def test_goal_bonus_applied(self):
        no_shot = _compute_delivery_quality(95, 34, "corner_kick", has_shot=False, xg_value=0)
        shot = _compute_delivery_quality(95, 34, "corner_kick", has_shot=True, xg_value=0.1)
        assert shot > no_shot

    def test_high_xg_extra_bonus(self):
        low = _compute_delivery_quality(95, 34, "corner_kick", has_shot=True, xg_value=0.05)
        high = _compute_delivery_quality(95, 34, "corner_kick", has_shot=True, xg_value=0.5)
        assert high >= low


class TestEstimateSetPieceXg:
    def test_returns_positive(self):
        xg = _estimate_set_piece_xg(95, 34, "corner_kick")
        assert xg > 0

    def test_penalty_high(self):
        xg = _estimate_set_piece_xg(105, 34, "penalty")
        assert xg > 0.5

    def test_corner_higher_than_throw_in(self):
        corner = _estimate_set_piece_xg(95, 34, "corner_kick")
        throw = _estimate_set_piece_xg(95, 34, "throw_in")
        assert corner > throw

    def test_near_goal_higher_than_far(self):
        near = _estimate_set_piece_xg(100, 34, "free_kick")
        far = _estimate_set_piece_xg(50, 34, "free_kick")
        assert near > far

    def test_with_xg_model(self):
        xgm = EnhancedXgModel()
        xg = _estimate_set_piece_xg(95, 34, "free_kick", xg_model=xgm)
        assert xg > 0


class TestReportDeliveryQuality:
    def test_overall_quality_in_report(self):
        events = [
            {
                "event_type": "corner_kick",
                "type": "corner_kick",
                "team": "home",
                "x": 1,
                "y": 1,
                "pass_end_x": 95,
                "pass_end_y": 10,
                "xg": 0.0,
                "is_goal": False,
                "timestamp": 10,
            },
            {
                "event_type": "corner_kick",
                "type": "corner_kick",
                "team": "home",
                "x": 1,
                "y": 1,
                "pass_end_x": 90,
                "pass_end_y": 34,
                "xg": 0.0,
                "is_goal": False,
                "timestamp": 20,
            },
        ]
        report = analyze_set_pieces(events)
        assert report.overall_delivery_quality > 0

    def test_summary_has_delivery_quality(self):
        events = [
            {
                "event_type": "corner_kick",
                "type": "corner_kick",
                "team": "home",
                "x": 1,
                "y": 1,
                "pass_end_x": 95,
                "pass_end_y": 34,
                "xg": 0.0,
                "is_goal": False,
                "timestamp": 10,
            },
        ]
        report = analyze_set_pieces(events)
        for s in report.summaries:
            assert "delivery_quality" in s
            assert "expected_xg_from_delivery" in s

    def test_zone_weights_in_delivery_zones(self):
        events = [
            {
                "event_type": "corner_kick",
                "type": "corner_kick",
                "team": "home",
                "x": 1,
                "y": 1,
                "pass_end_x": 95,
                "pass_end_y": 34,
                "xg": 0.0,
                "is_goal": False,
                "timestamp": 10,
            },
        ]
        report = analyze_set_pieces(events)
        for z in report.delivery_zones:
            assert "delivery_quality" in z
            assert "zone_weight" in z
