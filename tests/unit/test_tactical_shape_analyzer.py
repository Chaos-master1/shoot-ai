"""Tests for Tactical Shape Analyzer — shape classification, diamond detection, support angles."""

from kawkab.core.tactical_shape_analyzer import (
    TacticalShapeAnalyzer,
    _classify_attacking_shape,
    _classify_line_count,
    _compute_support_angles,
    _detect_diamond_midfield,
    _find_triangles_in_shape,
)


def test_classify_3_2_5():
    positions = [
        (5, 34),  # GK
        (20, 18),
        (20, 34),
        (20, 50),  # 3 defenders
        (40, 28),
        (40, 42),  # 2 midfield anchors
        (60, 8),
        (60, 22),
        (60, 34),
        (60, 48),
        (60, 60),
    ]  # 5 attackers
    result = _classify_attacking_shape(positions)
    assert result == "3-2-5"


def test_classify_4_3_3():
    positions = [  # 11 players: GK + 4 def + 3 mid + 3 att
        (5, 34),  # GK
        (20, 12),
        (20, 26),
        (20, 42),
        (20, 56),  # 4 def (tight x cluster)
        (45, 18),
        (45, 34),
        (45, 50),  # 3 mid
        (70, 15),
        (70, 34),
        (70, 53),  # 3 att
    ]
    result = _classify_attacking_shape(positions)
    assert result == "4-3-3"


def test_classify_4_4_2():
    positions = [
        (5, 34),  # GK
        (20, 12),
        (20, 26),
        (20, 42),
        (20, 56),  # 4 def
        (45, 12),
        (45, 26),
        (45, 42),
        (45, 56),  # 4 mid
        (70, 25),
        (70, 43),  # 2 att
    ]
    result = _classify_attacking_shape(positions)
    assert result == "4-4-2"


def test_classify_3_4_3():
    positions = [
        (5, 34),  # GK
        (20, 18),
        (20, 34),
        (20, 50),  # 3 def
        (45, 12),
        (45, 26),
        (45, 42),
        (45, 56),  # 4 mid
        (70, 15),
        (70, 34),
        (70, 53),  # 3 att
    ]
    result = _classify_attacking_shape(positions)
    assert result == "3-4-3"


def test_classify_insufficient_players():
    assert _classify_attacking_shape([(0, 0), (1, 1)]) == "insufficient_players"


def test_classify_line_count_3():
    assert _classify_line_count([4, 3, 3]) == "4-3-3"


def test_classify_line_count_4_4_2():
    assert _classify_line_count([4, 4, 2]) == "4-4-2"


def test_classify_line_count_5_3_2():
    assert _classify_line_count([5, 3, 2]) == "5-3-2"


def test_classify_line_count_4_line():
    assert _classify_line_count([4, 2, 3, 1]) == "4-2-3-1"


def test_classify_line_count_3_2_5():
    assert _classify_line_count([3, 2, 5, 0]) == "3-2-5"


def test_detect_diamond_midfield():
    positions = [
        (30, 34),  # deep CDM
        (40, 15),
        (40, 53),  # wide LM, RM
        (50, 34),
    ]  # advanced CAM
    assert _detect_diamond_midfield(positions, x_threshold=8.0)


def test_detect_no_diamond():
    positions = [(30, 34), (30, 38), (32, 34), (50, 34)]
    assert not _detect_diamond_midfield(positions, x_threshold=15.0)


def test_detect_diamond_few_players():
    assert not _detect_diamond_midfield([(0, 0), (1, 1), (2, 2)])


def test_support_angles():
    carrier = (50, 34)
    teammates = [(60, 34), (50, 20), (40, 34)]
    supports = _compute_support_angles(carrier, teammates)
    assert len(supports) == 3
    assert all(s["distance_m"] > 0 for s in supports)
    assert any(s["is_forward"] for s in supports)


def test_support_angles_empty():
    assert _compute_support_angles((50, 34), []) == []


def test_support_angles_same_position():
    assert _compute_support_angles((50, 34), [(50, 34)]) == []


def test_find_triangles_in_shape():
    positions = [(0, 0), (10, 0), (5, 10), (100, 100)]
    triangles = _find_triangles_in_shape(positions, max_dist=15.0)
    assert len(triangles) == 1  # First 3 form a triangle


def test_find_triangles_no_nearby():
    positions = [(0, 0), (100, 0), (50, 100)]
    triangles = _find_triangles_in_shape(positions, max_dist=20.0)
    assert len(triangles) == 0


class TestTacticalShapeAnalyzer:
    def test_analyze_shapes_empty(self):
        analyzer = TacticalShapeAnalyzer()
        report = analyzer.analyze_shapes([], team="home")
        assert report.team == "home"
        assert report.primary_attacking_shape == "unknown"

    def test_analyze_shapes_basic(self):
        events = []
        for i in range(30):
            events.append(
                {"team": "home", "timestamp": float(i), "from_track_id": i % 10, "type": "pass"}
            )
        analyzer = TacticalShapeAnalyzer()
        report = analyzer.analyze_shapes(events, team="home")
        assert report.team == "home"
        assert report.primary_attacking_shape != ""

    def test_analyze_shapes_away_team(self):
        events = [
            {"team": "away", "timestamp": float(i), "from_track_id": i % 10, "type": "pass"}
            for i in range(20)
        ]
        analyzer = TacticalShapeAnalyzer()
        report = analyzer.analyze_shapes(events, team="away")
        assert report.team == "away"

    def test_shape_report_to_dict(self):
        from kawkab.core.tactical_shape_analyzer import ShapeReport, ShapeSnapshot

        report = ShapeReport(
            team="home",
            primary_attacking_shape="4-3-3",
            primary_defensive_shape="4-5-1",
            shape_changes=2,
            diamond_midfield_pct=15.0,
            avg_triangles_per_frame=3.5,
            snapshots=[ShapeSnapshot(timestamp=0, attacking_shape="4-3-3")],
        )
        d = report.to_dict()
        assert d["team"] == "home"
        assert d["primary_attacking_shape"] == "4-3-3"
        assert d["shape_changes"] == 2
        assert d["diamond_midfield_pct"] == 15.0

    def test_shape_snapshot_to_dict(self):
        from kawkab.core.tactical_shape_analyzer import ShapeSnapshot

        ss = ShapeSnapshot(timestamp=10.0, attacking_shape="3-2-5", triangle_count=4)
        d = ss.to_dict()
        assert d["t"] == 10.0
        assert d["att_shape"] == "3-2-5"
        assert d["triangles"] == 4


class TestAnalyzeWindow:
    def test_window_few_players(self):
        analyzer = TacticalShapeAnalyzer()
        events = [{"team": "home", "from_track_id": 1, "start_x": 0.5, "start_y": 0.5}]
        ss = analyzer._analyze_window(events, timestamp=0, team="home")
        assert ss.attacking_shape == "unknown"
        assert ss.triangle_count == 0

    def test_window_sufficient_players(self):
        analyzer = TacticalShapeAnalyzer()
        events = []
        # GK at deep pos, then 10 outfield in 4-4-2 lines
        events.append(
            {"team": "home", "from_track_id": 0, "start_x": 5.0, "start_y": 34.0, "type": "pass"}
        )
        for i in range(4):
            events.append(
                {
                    "team": "home",
                    "from_track_id": i + 1,
                    "start_x": 15.0,
                    "start_y": float(10 + i * 16),
                    "type": "pass",
                }
            )
        for i in range(4):
            events.append(
                {
                    "team": "home",
                    "from_track_id": i + 5,
                    "start_x": 35.0,
                    "start_y": float(10 + i * 16),
                    "type": "pass",
                }
            )
        for i in range(2):
            events.append(
                {
                    "team": "home",
                    "from_track_id": i + 9,
                    "start_x": 55.0,
                    "start_y": float(20 + i * 28),
                    "type": "pass",
                }
            )
        ss = analyzer._analyze_window(events, timestamp=0, team="home")
        assert ss.attacking_shape != "unknown"
