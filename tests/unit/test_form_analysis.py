"""Tests for form analysis + Team of the Week module."""

from kawkab.core.form_analysis import (
    POSITION_ORDER,
    POSITION_SLOTS,
    FormAnalyzer,
    _points_from_result,
    _result_from_match,
)


class TestFormAnalysis:
    def setup_method(self):
        self.analyzer = FormAnalyzer()

    def _make_match(self, home, away, hg, ag, hx=1.5, ax=1.0):
        return {
            "home_team": home,
            "away_team": away,
            "home_goals": hg,
            "away_goals": ag,
            "home_xg": hx,
            "away_xg": ax,
        }

    def test_compute_form_streak_winning(self):
        matches = [
            self._make_match("Team A", "Team B", 2, 0),
            self._make_match("Team A", "Team C", 3, 1),
            self._make_match("Team D", "Team A", 0, 1),
        ]
        streak = self.analyzer.compute_form_streak(matches, "Team A")
        assert streak["streak_type"] == "W"
        assert streak["streak_length"] >= 1
        assert streak["total_points"] == 9

    def test_compute_form_streak_losing(self):
        matches = [
            self._make_match("Team A", "Team B", 0, 2),
            self._make_match("Team A", "Team C", 1, 3),
        ]
        streak = self.analyzer.compute_form_streak(matches, "Team A")
        assert streak["streak_type"] == "L"
        assert streak["streak_length"] == 2

    def test_compute_rolling_xg_form(self):
        matches = [
            self._make_match("Team A", "Team B", 2, 0, hx=1.8, ax=0.5),
            self._make_match("Team A", "Team C", 1, 1, hx=1.2, ax=0.9),
            self._make_match("Team D", "Team A", 0, 2, hx=0.3, ax=1.5),
            self._make_match("Team A", "Team E", 3, 0, hx=2.1, ax=0.4),
            self._make_match("Team F", "Team A", 1, 4, hx=0.8, ax=2.0),
        ]
        xg = self.analyzer.compute_rolling_xg_form(matches, "Team A")
        assert "rolling_xg_for" in xg
        assert "rolling_xg_against" in xg
        assert xg["matches_used"] == 5

    def test_compute_home_away_split(self):
        matches = [
            self._make_match("Team A", "Team B", 2, 0),
            self._make_match("Team C", "Team A", 1, 1),
            self._make_match("Team A", "Team D", 3, 1),
        ]
        split = self.analyzer.compute_home_away_split(matches, "Team A")
        assert "home" in split
        assert "away" in split
        assert split["home"]["matches"] == 2
        assert split["away"]["matches"] == 1

    def test_select_team_of_the_week(self):
        stats = [
            {"name": "Player1", "position": "GK", "rating": 8.5, "match_id": 1},
            {"name": "Player2", "position": "DEF", "rating": 7.8, "match_id": 1},
            {"name": "Player3", "position": "DEF", "rating": 7.2, "match_id": 1},
            {"name": "Player4", "position": "DEF", "rating": 7.0, "match_id": 1},
            {"name": "Player5", "position": "DEF", "rating": 6.8, "match_id": 1},
            {"name": "Player6", "position": "MID", "rating": 8.0, "match_id": 1},
            {"name": "Player7", "position": "MID", "rating": 7.5, "match_id": 1},
            {"name": "Player8", "position": "MID", "rating": 7.1, "match_id": 1},
            {"name": "Player9", "position": "FWD", "rating": 8.2, "match_id": 1},
            {"name": "Player10", "position": "FWD", "rating": 7.9, "match_id": 1},
            {"name": "Player11", "position": "FWD", "rating": 7.6, "match_id": 1},
        ]
        totw = self.analyzer.select_team_of_the_week(stats)
        assert "GK" in totw
        assert "DEF" in totw
        assert "MID" in totw
        assert "FWD" in totw

    def test_select_team_of_the_week_positions(self):
        stats = [
            {"name": "P1", "position": "GK", "rating": 9.0, "match_id": 1},
            {"name": "P2", "position": "CB", "rating": 8.0, "match_id": 1},
            {"name": "P3", "position": "LB", "rating": 7.5, "match_id": 1},
            {"name": "P4", "position": "RB", "rating": 7.3, "match_id": 1},
            {"name": "P5", "position": "DEF", "rating": 7.0, "match_id": 1},
            {"name": "P6", "position": "CM", "rating": 8.5, "match_id": 1},
            {"name": "P7", "position": "DM", "rating": 7.8, "match_id": 1},
            {"name": "P8", "position": "AM", "rating": 7.2, "match_id": 1},
        ]
        totw = self.analyzer.select_team_of_the_week(stats)
        assert len(totw["GK"]) == 1
        assert len(totw["DEF"]) == 4
        assert len(totw["MID"]) == 3
        assert len(totw["FWD"]) == 3

    def test_analyze_league_standings(self):
        matches = [
            self._make_match("Team A", "Team B", 2, 0),
            self._make_match("Team C", "Team A", 1, 1),
            self._make_match("Team B", "Team C", 0, 3),
        ]
        standings = self.analyzer.analyze_league_standings(matches)
        assert len(standings) == 3
        for s in standings:
            assert "points" in s
            assert "goal_difference" in s
            assert "form" in s

    def test_analyze_league_standings_tiebreaker(self):
        matches = [
            self._make_match("Team A", "Team B", 2, 1),
            self._make_match("Team C", "Team D", 1, 0),
            self._make_match("Team A", "Team C", 1, 0),
            self._make_match("Team B", "Team D", 2, 0),
        ]
        standings = self.analyzer.analyze_league_standings(matches)
        points = [s["points"] for s in standings]
        assert points == sorted(points, reverse=True)

    def test_detect_form_crisis_losses(self):
        matches = [
            self._make_match("Team A", "Team B", 0, 1),
            self._make_match("Team A", "Team C", 0, 2),
            self._make_match("Team D", "Team A", 3, 0),
        ]
        crisis = self.analyzer.detect_form_crisis(matches, "Team A")
        assert crisis["is_crisis"] is True
        assert crisis["streak_type"] == "consecutive_losses"
        assert crisis["streak_length"] >= 3

    def test_detect_form_crisis_winless(self):
        matches = [
            self._make_match("Team A", "Team B", 0, 0),
            self._make_match("Team C", "Team A", 1, 1),
            self._make_match("Team A", "Team D", 1, 2),
            self._make_match("Team E", "Team A", 0, 0),
            self._make_match("Team A", "Team F", 0, 1),
        ]
        crisis = self.analyzer.detect_form_crisis(matches, "Team A")
        assert crisis["is_crisis"] is True
        assert crisis["streak_type"] == "winless"

    def test_detect_form_crisis_no_crisis(self):
        matches = [
            self._make_match("Team A", "Team B", 2, 0),
            self._make_match("Team C", "Team A", 0, 1),
            self._make_match("Team A", "Team D", 1, 1),
        ]
        crisis = self.analyzer.detect_form_crisis(matches, "Team A")
        assert crisis["is_crisis"] is False

    def test_empty_matches_list(self):
        streak = self.analyzer.compute_form_streak([], "Team A")
        assert streak["streak_type"] == "none"
        assert streak["streak_length"] == 0
        xg = self.analyzer.compute_rolling_xg_form([], "Team A")
        assert xg["matches_used"] == 0
        split = self.analyzer.compute_home_away_split([], "Team A")
        assert split["home"]["matches"] == 0
        assert split["away"]["matches"] == 0
        standings = self.analyzer.analyze_league_standings([])
        assert standings == []
        crisis = self.analyzer.detect_form_crisis([], "Team A")
        assert crisis["is_crisis"] is False

    def test_result_from_match(self):
        match = self._make_match("Home", "Away", 2, 1)
        assert _result_from_match(match, "Home") == "W"
        assert _result_from_match(match, "Away") == "L"

    def test_result_from_match_draw(self):
        match = self._make_match("Home", "Away", 1, 1)
        assert _result_from_match(match, "Home") == "D"
        assert _result_from_match(match, "Away") == "D"

    def test_points_from_result(self):
        assert _points_from_result("W") == 3
        assert _points_from_result("D") == 1
        assert _points_from_result("L") == 0

    def test_position_constants(self):
        assert POSITION_ORDER == ["GK", "DEF", "MID", "FWD"]
        assert POSITION_SLOTS["GK"] == 1
        assert POSITION_SLOTS["DEF"] == 4
        assert POSITION_SLOTS["MID"] == 3
        assert POSITION_SLOTS["FWD"] == 3
