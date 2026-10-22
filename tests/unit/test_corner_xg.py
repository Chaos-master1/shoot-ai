"""Tests for corner kick xG model."""

import pytest

from kawkab.core.corner_xg import CornerKickXgModel, _classify_delivery_zone


class TestCornerXgModel:
    """Test suite for CornerKickXgModel."""

    def test_compute_corner_danger_rating_near_post(self):
        """Near-post delivery has danger around 0.08."""
        corner = {
            "type": "corner_kick",
            "team": "home",
            "end_x": 103,
            "end_y": 34,
        }
        rating = CornerKickXgModel.compute_corner_danger_rating(corner)
        # near-post base is 0.08, no modifier
        assert rating == pytest.approx(0.08, abs=0.01)

    def test_compute_corner_danger_rating_far_post(self):
        """Far-post delivery has danger around 0.10."""
        corner = {
            "type": "corner_kick",
            "team": "home",
            "end_x": 103,
            "end_y": 42,
        }
        rating = CornerKickXgModel.compute_corner_danger_rating(corner)
        # far-post base is 0.10
        assert rating == pytest.approx(0.10, abs=0.01)

    def test_compute_corner_danger_rating_short(self):
        """Short corner has danger around 0.04."""
        corner = {
            "type": "corner_kick",
            "team": "home",
            "end_x": 90,
            "end_y": 15,
        }
        rating = CornerKickXgModel.compute_corner_danger_rating(corner)
        assert rating == pytest.approx(0.04, abs=0.01)

    def test_compute_corner_danger_rating_deep(self):
        """Deep delivery has danger around 0.02."""
        corner = {
            "type": "corner_kick",
            "team": "home",
            "end_x": 80,
            "end_y": 55,
        }
        rating = CornerKickXgModel.compute_corner_danger_rating(corner)
        assert rating == pytest.approx(0.02, abs=0.01)

    def test_compute_corner_danger_rating_inswinging_bonus(self):
        """Inswinging delivery gets +0.02 bonus."""
        near = {
            "type": "corner_kick",
            "team": "home",
            "end_x": 103,
            "end_y": 34,
            "delivery_type": "inswinging",
        }
        rating = CornerKickXgModel.compute_corner_danger_rating(near)
        assert rating == pytest.approx(0.08 + 0.02, abs=0.01)

    def test_compute_corner_xg_total(self):
        """xG from shots preceded by a corner within 3 events."""
        events = [
            {"type": "corner_kick", "team": "home", "timestamp": 10.0, "x": 104, "y": 0},
            {
                "type": "shot",
                "team": "home",
                "timestamp": 12.0,
                "x": 95,
                "y": 34,
                "xg": 0.35,
                "is_goal": False,
            },
            {"type": "corner_kick", "team": "away", "timestamp": 20.0, "x": 0, "y": 68},
            {
                "type": "shot",
                "team": "away",
                "timestamp": 22.0,
                "x": 5,
                "y": 34,
                "xg": 0.12,
                "is_goal": True,
            },
        ]
        result = CornerKickXgModel.compute_corner_xg(events)
        assert result["total_xg"] == pytest.approx(0.47, rel=0.01)
        assert result["home_xg"] == pytest.approx(0.35, rel=0.01)
        assert result["away_xg"] == pytest.approx(0.12, rel=0.01)

    def test_compute_corner_efficiency_rates(self):
        """Shot and goal conversion rates."""
        events = [
            {"type": "corner_kick", "team": "home", "timestamp": 10.0, "x": 104, "y": 0},
            {
                "type": "shot",
                "team": "home",
                "timestamp": 12.0,
                "x": 95,
                "y": 34,
                "xg": 0.35,
                "is_goal": True,
            },
            {"type": "corner_kick", "team": "home", "timestamp": 20.0, "x": 104, "y": 0},
            {"type": "pass", "team": "home", "timestamp": 22.0, "x": 80, "y": 34},
        ]
        result = CornerKickXgModel.compute_corner_efficiency(events)
        assert result["home"]["corners"] == 2
        assert result["home"]["shots"] == 1
        assert result["home"]["goals"] == 1
        assert result["home"]["shot_conversion"] == 0.5
        assert result["home"]["goal_conversion"] == 0.5

    def test_analyze_delivery_zones_classification(self):
        """Corners classified into correct zones."""
        events = [
            {"type": "corner_kick", "team": "home", "timestamp": 10.0, "end_x": 103, "end_y": 34},
            {"type": "corner_kick", "team": "home", "timestamp": 20.0, "end_x": 103, "end_y": 42},
            {"type": "corner_kick", "team": "home", "timestamp": 30.0, "end_x": 90, "end_y": 15},
        ]
        result = CornerKickXgModel.analyze_delivery_zones(events)
        assert "near-post" in result["home"]
        assert "far-post" in result["home"]
        assert "short" in result["home"]

    def test_analyze_delivery_zones_per_team(self):
        """Zones tracked separately per team."""
        events = [
            {"type": "corner_kick", "team": "home", "timestamp": 10.0, "end_x": 103, "end_y": 34},
            {"type": "corner_kick", "team": "away", "timestamp": 20.0, "end_x": 103, "end_y": 42},
        ]
        result = CornerKickXgModel.analyze_delivery_zones(events)
        assert "home" in result
        assert "away" in result

    def test_edge_case_no_corners_zeros(self):
        """No corners → zero totals."""
        events = [
            {"type": "pass", "team": "home", "timestamp": 10.0},
        ]
        xg = CornerKickXgModel.compute_corner_xg(events)
        assert xg["total_xg"] == 0.0
        eff = CornerKickXgModel.compute_corner_efficiency(events)
        assert eff["home"]["corners"] == 0
        zones = CornerKickXgModel.analyze_delivery_zones(events)
        assert zones == {}

    def test_edge_case_corner_no_follow_up_no_xg(self):
        """Corner with no follow-up shot gives no xG."""
        events = [
            {"type": "corner_kick", "team": "home", "timestamp": 10.0, "x": 104, "y": 0},
        ]
        xg = CornerKickXgModel.compute_corner_xg(events)
        assert xg["total_xg"] == 0.0

    def test_complete_3_corners_2_shots_1_goal(self):
        """3 corners, 2 leading to shots, 1 goal."""
        events = [
            {"type": "corner_kick", "team": "home", "timestamp": 10.0, "x": 104, "y": 0},
            {
                "type": "shot",
                "team": "home",
                "timestamp": 12.0,
                "x": 95,
                "y": 34,
                "xg": 0.35,
                "is_goal": True,
            },
            {"type": "corner_kick", "team": "home", "timestamp": 20.0, "x": 104, "y": 0},
            {
                "type": "shot",
                "team": "home",
                "timestamp": 22.0,
                "x": 90,
                "y": 34,
                "xg": 0.08,
                "is_goal": False,
            },
            {"type": "corner_kick", "team": "home", "timestamp": 30.0, "x": 104, "y": 0},
            {"type": "pass", "team": "home", "timestamp": 32.0, "x": 80, "y": 34},
        ]
        xg = CornerKickXgModel.compute_corner_xg(events)
        assert xg["total_xg"] == pytest.approx(0.43, rel=0.01)
        assert xg["home_xg"] == pytest.approx(0.43, rel=0.01)
        eff = CornerKickXgModel.compute_corner_efficiency(events)
        assert eff["home"]["corners"] == 3
        assert eff["home"]["shots"] == 2
        assert eff["home"]["goals"] == 1
        assert eff["home"]["shot_conversion"] == pytest.approx(2.0 / 3.0, rel=0.01)

    def test_classify_delivery_zone_short(self):
        assert _classify_delivery_zone(90, 15) == "short"

    def test_classify_delivery_zone_deep(self):
        assert _classify_delivery_zone(80, 55) == "deep"

    def test_classify_delivery_zone_near_post(self):
        assert _classify_delivery_zone(103, 34) == "near-post"

    def test_classify_delivery_zone_far_post(self):
        assert _classify_delivery_zone(103, 42) == "far-post"

    def test_classify_delivery_zone_edge_box(self):
        assert _classify_delivery_zone(95, 30) == "edge-of-box"
