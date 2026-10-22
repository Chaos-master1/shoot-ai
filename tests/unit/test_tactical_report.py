"""Tests for Tactical Report — comprehensive tactical report aggregator."""

from kawkab.core.tactical_report import (
    TacticalReport,
    TeamTacticalProfile,
    _generate_observations,
    generate_tactical_report,
)


def _make_event(team, etype, ts=0, x=50, y=34, completed=True, is_goal=False):
    return {
        "team": team,
        "type": etype,
        "start_x": x,
        "start_y": y,
        "timestamp": ts,
        "completed": completed,
        "is_goal": is_goal,
        "from_track_id": 1 if team == "home" else 11,
    }


class TestGenerateTacticalReport:
    def test_empty_events(self):
        report = generate_tactical_report([], match_id=1)
        assert report.match_id == 1
        assert report.home.primary_shape == "unknown"

    def test_basic_report(self):
        events = []
        for i in range(40):
            team = "home" if i % 2 == 0 else "away"
            events.append(_make_event(team, "pass", ts=float(i), x=float(30 + i)))
        events.append(_make_event("home", "shot", ts=41, is_goal=True))
        report = generate_tactical_report(events, match_id=1)
        assert report.match_id == 1
        assert report.home.team == "Home"
        assert report.away.team == "Away"

    def test_report_with_both_teams(self):
        events = [_make_event("home", "pass", ts=float(i)) for i in range(10)]
        events += [_make_event("away", "pass", ts=float(i + 10)) for i in range(10)]
        report = generate_tactical_report(events, match_id=5)
        assert report.match_id == 5
        assert isinstance(report.to_dict(), dict)
        d = report.to_dict()
        assert "home" in d
        assert "away" in d
        assert "key_tactical_observations" in d


class TestTeamTacticalProfile:
    def test_profile_to_dict(self):
        profile = TeamTacticalProfile(
            team="Home",
            primary_shape="4-3-3",
            primary_formation="4-3-3",
            pressing_system="high_block",
            pressing_style="man_oriented",
            triangle_count=15,
            triangles_per_90=12.5,
            transition_count=4,
            build_up_success_rate=78.5,
        )
        d = profile.to_dict()
        assert d["team"] == "Home"
        assert d["primary_shape"] == "4-3-3"
        assert d["triangles_per_90"] == 12.5
        assert d["triangle_count"] == 15

    def test_default_profile(self):
        profile = TeamTacticalProfile()
        assert profile.team == "home"


class TestGenerateObservations:
    def test_observations_different_shapes(self):
        report = TacticalReport(match_id=1)
        report.home.primary_shape = "4-3-3"
        report.away.primary_shape = "5-3-2"
        report.home_pressing.primary_block_type = "high_block"
        report.away_pressing.primary_block_type = "low_block"
        report.home.triangle_count = 20
        report.away.triangle_count = 5
        obs = _generate_observations(report)
        assert len(obs) > 0
        assert any("Different attacking shapes" in o for o in obs)
        assert any("Contrasting pressing" in o for o in obs)
        assert any("dominates passing triangles" in o for o in obs)

    def test_observations_empty(self):
        report = TacticalReport(match_id=1)
        obs = _generate_observations(report)
        assert len(obs) > 0  # At least balanced observation

    def test_diamond_observation(self):
        report = TacticalReport(match_id=1)
        report.home_shape_report.diamond_midfield_pct = 50.0
        obs = _generate_observations(report)
        assert any("diamond midfield" in o for o in obs)


class TestTacticalReportToDict:
    def test_full_to_dict(self):
        report = generate_tactical_report([], match_id=42)
        d = report.to_dict()
        assert d["match_id"] == 42
        assert "tactical_phases" in d
        assert "home_shape_report" in d
        assert "away_shape_report" in d
        assert "home_pressing" in d
        assert "away_pressing" in d
        assert "key_tactical_observations" in d
