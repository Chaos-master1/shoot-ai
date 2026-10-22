"""Tests for Referee / Foul Pattern Analysis module."""

from __future__ import annotations

from kawkab.core.referee_analysis import analyze_referee


def make_match(
    foul_events: list | None = None,
    card_events: list | None = None,
    home: str = "Home",
    away: str = "Away",
    events: list | None = None,
) -> dict:
    return {
        "foul_events": foul_events or [],
        "card_events": card_events or [],
        "home_team": home,
        "away_team": away,
        "duration": 90,
        "events": events or [],
    }


class TestRefereeAnalysis:
    def test_empty_matches_returns_default_profile(self):
        report = analyze_referee("Ref A", [])
        assert report.referee.name == "Ref A"
        assert report.referee.matches_officiated == 0
        assert report.referee.cards_per_game["total"] == 0.0

    def test_cards_per_game_computed(self):
        matches = [
            make_match(card_events=[{"type": "yellow", "minute": 20, "team": "Home"}]),
            make_match(
                card_events=[
                    {"type": "yellow", "minute": 30, "team": "Away"},
                    {"type": "red", "minute": 50, "team": "Home"},
                ]
            ),
        ]
        report = analyze_referee("Ref B", matches)
        assert report.referee.matches_officiated == 2
        assert report.referee.cards_per_game["yellow"] > 0
        assert report.referee.cards_per_game["red"] > 0
        assert report.referee.cards_per_game["total"] > 0

    def test_home_team_advantage_computed(self):
        matches = [
            make_match(
                foul_events=[
                    {"type": "tackle", "team": "Home", "x": 30, "y": 50},
                    {"type": "tackle", "team": "Home", "x": 40, "y": 50},
                    {"type": "tackle", "team": "Away", "x": 50, "y": 50},
                ]
            ),
        ]
        report = analyze_referee("Ref C", matches)
        assert report.referee.home_team_advantage == 2.0

    def test_card_timing_distribution(self):
        matches = [
            make_match(
                card_events=[
                    {"type": "yellow", "minute": 20, "team": "Home"},
                    {"type": "yellow", "minute": 55, "team": "Away"},
                ]
            ),
        ]
        report = analyze_referee("Ref D", matches)
        assert report.referee.card_timing_distribution["first_half"] == 1
        assert report.referee.card_timing_distribution["second_half"] == 1

    def test_foul_outcomes_mapped(self):
        matches = [
            make_match(
                foul_events=[
                    {"type": "tackle", "team": "Home", "x": 30, "y": 50},
                    {"type": "push", "team": "Away", "x": 50, "y": 50},
                ],
                card_events=[
                    {"type": "yellow", "minute": 20, "team": "Home", "foul_type": "tackle"},
                ],
            ),
        ]
        report = analyze_referee("Ref E", matches)
        assert "tackle" in report.foul_outcomes
        assert "push" in report.foul_outcomes
        assert report.foul_outcomes["tackle"]["card_pct"] > 0

    def test_summary_text_produces_output(self):
        matches = [
            make_match(
                foul_events=[
                    {"type": "tackle", "team": "Home", "x": 30, "y": 50},
                ]
            ),
        ]
        report = analyze_referee("Ref F", matches)
        summary = report.summary_text()
        assert "Referee: Ref F" in summary
        assert "Matches officiated:" in summary

    def test_penalty_rate_computed(self):
        matches = [
            make_match(events=[{"type": "penalty", "minute": 60}]),
            make_match(events=[{"type": "penalty_awarded", "minute": 30}]),
        ]
        report = analyze_referee("Ref G", matches)
        assert report.referee.penalty_rate > 0

    def test_bias_indicators_populated(self):
        matches = [
            make_match(
                foul_events=[{"type": "foul", "team": "Home", "x": 50, "y": 50}],
                card_events=[{"type": "yellow", "minute": 30, "team": "Home"}],
            ),
        ]
        report = analyze_referee("Ref H", matches)
        assert len(report.bias_indicators) == 2
        assert report.bias_indicators[0]["metric"] == "fouls"

    def test_foul_heatmap_generated(self):
        matches = [
            make_match(
                foul_events=[
                    {"type": "tackle", "team": "Home", "x": 30, "y": 50},
                    {"type": "push", "team": "Away", "x": 60, "y": 80},
                ]
            ),
        ]
        report = analyze_referee("Ref I", matches)
        assert len(report.match_foul_heatmap) > 0

    def test_most_common_foul_types(self):
        matches = [
            make_match(
                foul_events=[
                    {"type": "tackle", "team": "Home", "x": 30, "y": 50},
                    {"type": "tackle", "team": "Away", "x": 50, "y": 50},
                    {"type": "push", "team": "Home", "x": 40, "y": 50},
                ]
            ),
        ]
        report = analyze_referee("Ref J", matches)
        assert len(report.referee.most_common_foul_types) >= 2
        assert report.referee.most_common_foul_types[0]["type"] == "tackle"

    def test_cards_per_game_zero_when_no_cards(self):
        matches = [make_match()]
        report = analyze_referee("Ref K", matches)
        assert report.referee.cards_per_game["yellow"] == 0.0
        assert report.referee.cards_per_game["red"] == 0.0

    def test_trend_stable_for_few_matches(self):
        matches = [make_match(), make_match()]
        report = analyze_referee("Ref L", matches)
        assert report.referee.trend == "stable"

    def test_trend_detected_for_many_matches(self):
        matches = [
            make_match(
                card_events=[
                    {"type": "yellow", "minute": 20, "team": "H", "foul_type": "tackle"},
                ]
            )
            for _ in range(6)
        ]
        report = analyze_referee("Ref M", matches)
        assert report.referee.trend in ("stable", "increasing", "decreasing")

    def test_inconsistency_score_with_variance(self):
        matches = [
            make_match(
                card_events=[
                    {"type": "yellow", "minute": 20, "team": "H", "foul_type": "t"},
                ]
            ),
            make_match(
                card_events=[
                    {"type": "yellow", "minute": 30, "team": "A", "foul_type": "t"},
                    {"type": "red", "minute": 50, "team": "H", "foul_type": "t"},
                    {"type": "yellow", "minute": 70, "team": "H", "foul_type": "t"},
                ]
            ),
            make_match(),
        ]
        report = analyze_referee("Ref N", matches)
        assert 0.0 <= report.referee.inconsistency_score <= 1.0
