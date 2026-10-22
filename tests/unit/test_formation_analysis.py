"""Tests for formation shape analysis."""

import pytest

from kawkab.core.formation_analysis import (
    FormationAnalyzer,
    FormationMatchReport,
    FormationSnapshot,
)


class TestFormationAnalyzer:
    def test_classify_formation(self):
        fa = FormationAnalyzer()
        positions = [(10 + i * 8, 34) for i in range(10)]
        formation = fa._classify_formation(positions)
        assert formation != "unknown"
        parts = formation.split("-")
        assert len(parts) >= 2

    def test_compactness_all_same_position(self):
        fa = FormationAnalyzer()
        positions = [(50, 34), (50, 34), (50, 34)]
        c = fa._compute_compactness(positions)
        assert c == 0.0

    def test_compactness_spread(self):
        fa = FormationAnalyzer()
        positions = [(10, 10), (100, 60)]
        c = fa._compute_compactness(positions)
        assert c > 0.0

    def test_width(self):
        fa = FormationAnalyzer()
        positions = [(50, 5), (50, 63)]
        w = fa._compute_width(positions)
        assert w == pytest.approx(58.0, rel=0.1)

    def test_depth(self):
        fa = FormationAnalyzer()
        positions = [(10, 34), (100, 34)]
        d = fa._compute_depth(positions)
        assert d == pytest.approx(90.0, rel=0.1)

    def test_analyze_empty_frames(self):
        fa = FormationAnalyzer()
        report = fa.analyze_team_shape([], team="home")
        assert isinstance(report, FormationMatchReport)
        assert report.avg_width_in == 0.0

    def test_analyze_with_few_players(self):
        fa = FormationAnalyzer()
        frames = [
            {
                "timestamp": 0.0,
                "possession": True,
                "home_positions": [(50, 34)],
                "away_positions": [(50, 40)],
            }
        ]
        report = fa.analyze_team_shape(frames, team="home")
        assert isinstance(report, FormationMatchReport)

    def test_analyze_with_full_frame(self):
        fa = FormationAnalyzer()
        home = [(10 + i * 8, 20 + i * 4) for i in range(10)]
        frames = [
            {
                "timestamp": 0.0,
                "possession": True,
                "home_positions": home,
                "away_positions": [(80, 34)],
            }
        ]
        report = fa.analyze_team_shape(frames, team="home")
        assert report.avg_width_in > 0
        assert report.avg_depth_in > 0

    def test_in_vs_out_possession(self):
        fa = FormationAnalyzer()
        home_compact = [(45 + i * 2, 34) for i in range(10)]
        home_spread = [(10 + i * 10, 34) for i in range(10)]
        frames = [
            {
                "timestamp": 0.0,
                "possession": True,
                "home_positions": home_compact,
                "away_positions": [(80, 34)],
            },
            {
                "timestamp": 10.0,
                "possession": False,
                "home_positions": home_spread,
                "away_positions": [(80, 34)],
            },
        ]
        report = fa.analyze_team_shape(frames, team="home")
        assert report.in_possession_formation != "unknown"

    def test_snapshot_to_dict(self):
        snap = FormationSnapshot(
            timestamp=10.0, width=40.0, depth=30.0, compactness=8.5, possession=True
        )
        d = snap.to_dict()
        assert d["t"] == 10.0
        assert d["w"] == 40.0
        assert d["pos"] is True

    def test_report_to_dict(self):
        report = FormationMatchReport(avg_width_in=45.0, avg_width_out=35.0)
        d = report.to_dict()
        assert d["avg_width_in"] == 45.0
        assert d["avg_width_out"] == 35.0
