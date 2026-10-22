"""Tests for Passing Triangles / Third-Man Combinations module."""

from kawkab.core.passing_triangles import PassingTriangleAnalyzer


def _make_pass_event(
    team: str,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    from_track_id: int = 1,
    to_track_id: int = 2,
    timestamp: float = 0,
    completed: bool = True,
) -> dict:
    return {
        "type": "pass",
        "team": team,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "from_track_id": from_track_id,
        "to_track_id": to_track_id,
        "timestamp": timestamp,
        "completed": completed,
    }


def _make_shot_event(team: str, timestamp: float, is_goal: bool = False) -> dict:
    return {"type": "shot", "team": team, "timestamp": timestamp, "is_goal": is_goal, "xg": 0.1}


class TestDetectPassingTriangles:
    def test_detects_triangle(self):
        events = [
            _make_pass_event("home", 50, 34, 60, 34, 1, 2, 0),
            _make_pass_event("home", 60, 34, 55, 40, 2, 3, 1),
            _make_pass_event("home", 55, 40, 52, 30, 3, 1, 2),
        ]
        pta = PassingTriangleAnalyzer()
        triangles = pta.detect_passing_triangles(events)
        assert len(triangles) == 1
        assert triangles[0]["pass_count"] == 3
        assert triangles[0]["area_sqm"] > 0

    def test_empty_events(self):
        pta = PassingTriangleAnalyzer()
        assert pta.detect_passing_triangles([]) == []

    def test_no_triangle_few_events(self):
        events = [_make_pass_event("home", 50, 34, 60, 34, 1, 2, 0)]
        pta = PassingTriangleAnalyzer()
        assert len(pta.detect_passing_triangles(events)) == 0

    def test_too_few_players_no_triangle(self):
        events = [
            _make_pass_event("home", 50, 34, 60, 34, 1, 2, 0),
            _make_pass_event("home", 60, 34, 65, 34, 2, 3, 1),
            _make_pass_event("home", 65, 34, 70, 34, 3, 4, 2),
        ]
        pta = PassingTriangleAnalyzer()
        triangles = pta.detect_passing_triangles(events)
        assert len(triangles) == 0

    def test_time_window_respected(self):
        events = [
            _make_pass_event("home", 50, 34, 60, 34, 1, 2, 0),
            _make_pass_event("home", 60, 34, 55, 40, 2, 3, 31),
            _make_pass_event("home", 55, 40, 52, 30, 3, 1, 62),
        ]
        pta = PassingTriangleAnalyzer()
        triangles = pta.detect_passing_triangles(events)
        assert len(triangles) == 0

    def test_zone_classification(self):
        events = [
            _make_pass_event("home", 80, 34, 90, 34, 1, 2, 0),
            _make_pass_event("home", 90, 34, 95, 40, 2, 3, 1),
            _make_pass_event("home", 95, 40, 100, 30, 3, 1, 2),
        ]
        pta = PassingTriangleAnalyzer()
        triangles = pta.detect_passing_triangles(events)
        assert len(triangles) >= 1
        assert triangles[0]["zone"] == "attacking"


class TestDetectThirdManCombinations:
    def test_detects_third_man(self):
        events = [
            _make_pass_event("home", 50, 34, 60, 34, 1, 2, 0),
            _make_pass_event("home", 60, 34, 50, 34, 2, 1, 1),
            _make_pass_event("home", 50, 34, 75, 40, 1, 3, 2),
        ]
        pta = PassingTriangleAnalyzer()
        combos = pta.detect_third_man_combinations(events)
        assert len(combos) >= 1
        assert combos[0]["is_progressive"] is True

    def test_empty_events(self):
        pta = PassingTriangleAnalyzer()
        assert pta.detect_third_man_combinations([]) == []

    def test_no_combo_no_return_pass(self):
        events = [
            _make_pass_event("home", 50, 34, 60, 34, 1, 2, 0),
            _make_pass_event("home", 60, 34, 70, 34, 2, 3, 1),
            _make_pass_event("home", 70, 34, 80, 34, 3, 4, 2),
        ]
        pta = PassingTriangleAnalyzer()
        assert len(pta.detect_third_man_combinations(events)) == 0

    def test_leads_to_shot(self):
        events = [
            _make_pass_event("home", 50, 34, 60, 34, 1, 2, 0),
            _make_pass_event("home", 60, 34, 50, 34, 2, 1, 1),
            _make_pass_event("home", 50, 34, 80, 40, 1, 3, 2),
            _make_shot_event("home", 8, False),
        ]
        pta = PassingTriangleAnalyzer()
        combos = pta.detect_third_man_combinations(events)
        assert any(c["leads_to_shot"] for c in combos)

    def test_time_gap_breaks_combo(self):
        events = [
            _make_pass_event("home", 50, 34, 60, 34, 1, 2, 0),
            _make_pass_event("home", 60, 34, 50, 34, 2, 1, 10),
            _make_pass_event("home", 50, 34, 70, 40, 1, 3, 11),
        ]
        pta = PassingTriangleAnalyzer()
        assert len(pta.detect_third_man_combinations(events)) == 0


class TestAnalyzeTriangleNetwork:
    def test_returns_network(self):
        events = [
            _make_pass_event("home", 50, 34, 60, 34, 1, 2, 0),
            _make_pass_event("home", 60, 34, 55, 40, 2, 3, 1),
            _make_pass_event("home", 55, 40, 52, 30, 3, 1, 2),
        ]
        pta = PassingTriangleAnalyzer()
        net = pta.analyze_triangle_network(events, "home")
        assert net["triangle_count"] == 1
        assert net["avg_area_sqm"] > 0

    def test_no_triangles(self):
        pta = PassingTriangleAnalyzer()
        net = pta.analyze_triangle_network([], "home")
        assert net["triangle_count"] == 0


class TestComputeTriangleEfficiency:
    def test_returns_stats(self):
        events = [
            _make_pass_event("home", 50, 34, 60, 34, 1, 2, 0),
            _make_pass_event("home", 60, 34, 55, 40, 2, 3, 1),
            _make_pass_event("home", 55, 40, 52, 30, 3, 1, 2),
            _make_pass_event("home", 30, 34, 40, 34, 4, 5, 3),
        ]
        pta = PassingTriangleAnalyzer()
        eff = pta.compute_triangle_efficiency(events, "home")
        assert "triangles_per_90" in eff
        assert "triangle_completion_rate_pct" in eff


class TestGetTriangleHeatmap:
    def test_returns_heatmap(self):
        events = [
            _make_pass_event("home", 80, 34, 90, 34, 1, 2, 0),
            _make_pass_event("home", 90, 34, 95, 40, 2, 3, 1),
            _make_pass_event("home", 95, 40, 100, 30, 3, 1, 2),
        ]
        pta = PassingTriangleAnalyzer()
        hm = pta.get_triangle_heatmap(events, "home")
        assert "zone_counts" in hm
        assert hm["total_triangles"] >= 0
