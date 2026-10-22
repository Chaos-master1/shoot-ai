"""Tests for Flank Preference Analysis module."""

from kawkab.core.flank_analysis import FlankAnalyzer, classify_zone


def _make_pass(
    team: str,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    completed: bool = True,
    timestamp: float = 0,
) -> dict:
    return {
        "type": "pass",
        "team": team,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "completed": completed,
        "timestamp": timestamp,
    }


class TestClassifyZone:
    def test_left_zone(self):
        assert classify_zone(20) == "left"

    def test_center_zone(self):
        assert classify_zone(52) == "center"

    def test_right_zone(self):
        assert classify_zone(80) == "right"


class TestAnalyzeBuildUpSide:
    def test_detects_dominant_side(self):
        fa = FlankAnalyzer()
        events = [
            _make_pass("home", 20, 10, 40, 10),
            _make_pass("home", 20, 10, 50, 10),
            _make_pass("home", 25, 50, 45, 55),
        ]
        result = fa.analyze_build_up_side(events, "home")
        assert result["total_actions"] > 0
        assert result["dominant_side"] in ("left", "center", "right")

    def test_no_build_up_events(self):
        fa = FlankAnalyzer()
        events = [_make_pass("home", 60, 34, 80, 34)]
        result = fa.analyze_build_up_side(events, "home")
        assert result["dominant_side"] == "center"

    def test_empty_events(self):
        fa = FlankAnalyzer()
        result = fa.analyze_build_up_side([], "home")
        assert result["total_actions"] == 0


class TestAnalyzeAttackSide:
    def test_detects_attack_side(self):
        fa = FlankAnalyzer()
        events = [
            _make_pass("home", 80, 10, 90, 10),
            _make_pass("home", 80, 50, 90, 55),
        ]
        result = fa.analyze_attack_side(events, "home")
        assert result["total_actions"] > 0

    def test_no_attack_events(self):
        fa = FlankAnalyzer()
        events = [_make_pass("home", 30, 34, 40, 34)]
        result = fa.analyze_attack_side(events, "home")
        assert result["dominant_side"] == "center"


class TestComputeFlankEffectiveness:
    def test_returns_flank_stats(self):
        fa = FlankAnalyzer()
        events = [
            _make_pass("home", 20, 10, 30, 10),
            _make_pass("home", 80, 50, 90, 55),
            {
                "type": "shot",
                "team": "home",
                "start_x": 80,
                "start_y": 34,
                "xg": 0.2,
                "is_goal": False,
            },
        ]
        result = fa.compute_flank_effectiveness(events, "home")
        assert result["team"] == "home"
        assert "left" in result
        assert "right" in result


class TestDetectFlankSwitches:
    def test_detects_switches(self):
        fa = FlankAnalyzer()
        events = [
            _make_pass("home", 50, 5, 70, 60, True),  # left to right flank switch
        ]
        result = fa.detect_flank_switches(events, "home")
        assert result["switch_count"] >= 0

    def test_no_switches(self):
        fa = FlankAnalyzer()
        events = [_make_pass("home", 50, 34, 60, 34)]
        result = fa.detect_flank_switches(events, "home")
        assert result["switch_count"] == 0


class TestGenerateFlankReport:
    def test_returns_both_teams(self):
        fa = FlankAnalyzer()
        events = [
            _make_pass("home", 20, 10, 30, 10),
            _make_pass("away", 20, 50, 30, 55),
        ]
        report = fa.generate_flank_report(events)
        assert "home" in report
        assert "away" in report
        assert "build_up" in report["home"]
        assert "flank_switches" in report["home"]

    def test_empty_events(self):
        fa = FlankAnalyzer()
        report = fa.generate_flank_report([])
        assert report["home"]["build_up"]["total_actions"] == 0
        assert report["away"]["flank_switches"]["switch_count"] == 0
