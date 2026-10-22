"""Tests for crossing analysis module."""

from kawkab.core.crossing_analysis import (
    CrossingAnalysis,
    CrossingReport,
    _zone_label,
)


class TestZoneLabel:
    def test_near_post_6yd(self):
        z = _zone_label(100, 10)
        assert z == "near_post_6yd"

    def test_far_post_6yd(self):
        z = _zone_label(100, 58)
        assert z == "far_post_6yd"

    def test_near_post_18yd(self):
        z = _zone_label(92, 15)
        assert z == "near_post_18yd"

    def test_far_post_18yd(self):
        z = _zone_label(92, 55)
        assert z == "far_post_18yd"

    def test_edge_of_box(self):
        z = _zone_label(78, 34)
        assert z == "edge_of_box"

    def test_deep(self):
        z = _zone_label(30, 34)
        assert z == "deep"

    def test_zone_boundary_near_post(self):
        z = _zone_label(100, 33)
        assert "near_post" in z


class TestClassifyCross:
    def setup_method(self):
        self.ca = CrossingAnalysis()

    def test_early_cross(self):
        ct = self.ca.classify_cross({"start_x": 30, "end_x": 80})
        assert ct == "early"

    def test_driven_cross(self):
        ct = self.ca.classify_cross({"start_x": 60, "end_x": 90, "height": "low", "metadata": {}})
        assert ct == "driven"

    def test_floated_cross(self):
        ct = self.ca.classify_cross({"start_x": 60, "end_x": 90, "height": "high", "metadata": {}})
        assert ct == "floated"

    def test_pulled_back(self):
        ct = self.ca.classify_cross({"start_x": 80, "end_x": 60})
        assert ct == "pulled_back"

    def test_byline_cross(self):
        ct = self.ca.classify_cross({"start_x": 98, "end_x": 100})
        assert ct == "byline"

    def test_corner_cross(self):
        ct = self.ca.classify_cross({"start_x": 104, "end_x": 100, "corner": True, "metadata": {}})
        assert ct == "byline"

    def test_default_driven(self):
        ct = self.ca.classify_cross({"start_x": 60, "end_x": 85})
        assert ct == "driven"


class TestDangerRating:
    def setup_method(self):
        self.ca = CrossingAnalysis()

    def test_byline_more_dangerous_than_deep(self):
        byline = self.ca.compute_cross_danger_rating(
            {"start_x": 98, "end_x": 100, "end_y": 34, "metadata": {}}
        )
        deep = self.ca.compute_cross_danger_rating(
            {"start_x": 0, "end_x": 5, "end_y": 34, "metadata": {}}
        )
        assert byline > deep

    def test_pulled_back_high_danger(self):
        pb = self.ca.compute_cross_danger_rating(
            {"start_x": 80, "end_x": 60, "end_y": 34, "metadata": {}}
        )
        assert pb > 0.5

    def test_danger_clipped_to_one(self):
        dr = self.ca.compute_cross_danger_rating(
            {"start_x": 100, "end_x": 105, "end_y": 34, "metadata": {}}
        )
        assert 0.0 <= dr <= 1.0

    def test_danger_min_zero(self):
        dr = self.ca.compute_cross_danger_rating(
            {"start_x": 0, "end_x": 5, "end_y": 68, "metadata": {}}
        )
        assert dr >= 0.0


class TestAnalyzeCrosses:
    def setup_method(self):
        self.ca = CrossingAnalysis()

    def test_basic_counts(self):
        events = [
            {
                "type": "cross",
                "start_x": 60,
                "end_x": 90,
                "end_y": 34,
                "height": "low",
                "metadata": {},
            },
            {
                "type": "cross",
                "start_x": 30,
                "end_x": 80,
                "end_y": 34,
                "height": "high",
                "metadata": {},
            },
            {"type": "cross", "start_x": 80, "end_x": 60, "end_y": 34, "metadata": {}},
            {"type": "cross", "start_x": 96, "end_x": 100, "end_y": 10, "metadata": {}},
            {
                "type": "cross",
                "start_x": 50,
                "end_x": 90,
                "end_y": 50,
                "height": "low",
                "metadata": {},
            },
        ]
        report = self.ca.analyze_crosses(events)
        assert report.total_crosses == 5
        assert isinstance(report, CrossingReport)
        assert sum(report.crosses_by_type.values()) == 5

    def test_analyze_empty(self):
        report = self.ca.analyze_crosses([])
        assert report.total_crosses == 0
        assert report.avg_danger_rating == 0.0

    def test_analyze_corner_crosses(self):
        events = [
            {
                "type": "cross",
                "start_x": 104,
                "end_x": 100,
                "end_y": 10,
                "corner": True,
                "metadata": {},
            },
            {
                "type": "cross",
                "start_x": 60,
                "end_x": 85,
                "end_y": 34,
                "height": "low",
                "metadata": {},
            },
        ]
        report = self.ca.analyze_crosses(events)
        assert report.corner_crosses == 1
        assert report.total_crosses == 2

    def test_headed_shot_tracking(self):
        events = [
            {
                "type": "cross",
                "start_x": 60,
                "end_x": 90,
                "end_y": 34,
                "height": "high",
                "metadata": {"headed_shot_created": True},
            },
        ]
        report = self.ca.analyze_crosses(events)
        assert report.headed_shots_created == 1

    def test_goal_tracking(self):
        events = [
            {
                "type": "cross",
                "start_x": 60,
                "end_x": 90,
                "end_y": 34,
                "height": "low",
                "metadata": {"goal_created": True},
            },
        ]
        report = self.ca.analyze_crosses(events)
        assert report.goals_created == 1

    def test_report_cross_objects(self):
        events = [
            {
                "type": "cross",
                "start_x": 70,
                "end_x": 95,
                "end_y": 20,
                "height": "low",
                "metadata": {},
            },
        ]
        report = self.ca.analyze_crosses(events)
        assert len(report.crosses) == 1
        assert report.crosses[0].cross_type == "driven"

    def test_report_to_dict(self):
        events = [
            {
                "type": "cross",
                "start_x": 60,
                "end_x": 90,
                "end_y": 34,
                "height": "low",
                "metadata": {},
            },
        ]
        report = self.ca.analyze_crosses(events)
        d = report.to_dict()
        assert d["total_crosses"] == 1
        assert "crosses_by_type" in d
        assert "zone_heatmap" in d


class TestZoneHeatmap:
    def setup_method(self):
        self.ca = CrossingAnalysis()

    def test_heatmap_basic(self):
        events = [
            {"type": "cross", "start_x": 60, "end_x": 100, "end_y": 10},
            {"type": "cross", "start_x": 50, "end_x": 100, "end_y": 55},
            {"type": "cross", "start_x": 70, "end_x": 90, "end_y": 15},
        ]
        hm = self.ca.compute_cross_zone_heatmap(events)
        assert sum(hm.values()) == 3
        assert len(hm) >= 2

    def test_heatmap_empty(self):
        hm = self.ca.compute_cross_zone_heatmap([])
        assert hm == {}

    def test_heatmap_only_crosses(self):
        events = [
            {"type": "pass", "start_x": 50, "end_x": 70, "end_y": 34},
            {"type": "shot", "start_x": 90, "end_y": 34},
        ]
        hm = self.ca.compute_cross_zone_heatmap(events)
        assert hm == {}

    def test_heatmap_deep_cross(self):
        events = [
            {"type": "cross", "start_x": 30, "end_x": 35, "end_y": 34},
        ]
        hm = self.ca.compute_cross_zone_heatmap(events)
        assert hm.get("deep", 0) == 1

    def test_heatmap_edge_of_box(self):
        events = [
            {"type": "cross", "start_x": 60, "end_x": 78, "end_y": 34},
        ]
        hm = self.ca.compute_cross_zone_heatmap(events)
        assert hm.get("edge_of_box", 0) == 1
