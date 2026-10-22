"""Tests for Suspension Tracker module."""

from __future__ import annotations

from kawkab.core.suspension_tracker import (
    analyze_suspensions,
)


def make_card_event(
    player_id: str, player_name: str, ev_type: str, team: str = "Team A", match_num: int = 1
) -> dict:
    return {
        "player_id": player_id,
        "player_name": player_name,
        "type": ev_type,
        "team": team,
        "match_number": match_num,
    }


class TestSuspensionTracker:
    def test_empty_events_returns_empty_report(self):
        report = analyze_suspensions([], "Premier League", "team_1")
        assert report.team == "team_1"
        assert report.competition == "Premier League"
        assert report.total_yellows == 0
        assert report.total_reds == 0
        assert len(report.players) == 0

    def test_yellow_card_accumulation_tracked(self):
        events = [
            make_card_event("p1", "Player 1", "yellow_card", match_num=1),
            make_card_event("p1", "Player 1", "yellow_card", match_num=2),
        ]
        report = analyze_suspensions(events, "Liga", "team_1")
        assert report.total_yellows == 2
        p1 = next(p for p in report.players if p.player_id == "p1")
        assert p1.yellow_cards == 2

    def test_suspension_threshold_triggers(self):
        events = [
            make_card_event("p1", "Player 1", "yellow_card", match_num=i) for i in range(1, 6)
        ]
        report = analyze_suspensions(events, "Liga", "team_1")
        p1 = next(p for p in report.players if p.player_id == "p1")
        assert p1.is_suspended is True
        assert "suspension" in p1.suspension_details.lower()
        assert len(report.pending_suspensions) == 1

    def test_upcoming_risk_detected(self):
        events = [
            make_card_event("p1", "Player 1", "yellow_card", match_num=i) for i in range(1, 5)
        ]
        report = analyze_suspensions(events, "Liga", "team_1")
        assert len(report.upcoming_risk) >= 1
        risk = report.upcoming_risk[0]
        assert risk["player"] == "Player 1"
        assert risk["cards_needed_for_suspension"] == 1

    def test_red_card_suspension(self):
        events = [
            make_card_event("p1", "Player 1", "red_card", match_num=1),
        ]
        report = analyze_suspensions(events, "Liga", "team_1")
        p1 = next(p for p in report.players if p.player_id == "p1")
        assert p1.is_suspended is True
        assert p1.red_cards == 1

    def test_summary_text_produces_output(self):
        events = [
            make_card_event("p1", "Player 1", "yellow_card", match_num=1),
            make_card_event("p2", "Player 2", "red_card", match_num=1),
        ]
        report = analyze_suspensions(events, "Liga", "team_1")
        summary = report.summary_text()
        assert "Suspension Report" in summary
        assert "Total yellows:" in summary

    def test_custom_suspension_rules(self):
        rules = {
            "yellow_thresholds": [3, 6, 9],
            "yellow_suspension_matches": [1, 2, 3],
            "straight_red_matches": 5,
            "second_yellow_matches": 2,
            "clear_after_matches": 10,
            "fair_play_yellow_penalty": 5,
            "fair_play_red_penalty": 15,
        }
        events = [
            make_card_event("p1", "Player 1", "yellow_card", match_num=i) for i in range(1, 4)
        ]
        report = analyze_suspensions(events, "Liga", "team_1", suspension_rules=rules)
        p1 = next(p for p in report.players if p.player_id == "p1")
        assert p1.is_suspended is True
        assert p1.suspension_threshold == 3

    def test_fair_play_score_computed(self):
        events = [
            make_card_event("p1", "Player 1", "yellow_card", match_num=1),
            make_card_event("p1", "Player 1", "red_card", match_num=2),
        ]
        report = analyze_suspensions(events, "Liga", "team_1")
        p1 = next(p for p in report.players if p.player_id == "p1")
        assert p1.fair_play_score < 100.0

    def test_to_dict_output(self):
        events = [make_card_event("p1", "Player 1", "yellow_card", match_num=1)]
        report = analyze_suspensions(events, "Liga", "team_1")
        d = report.to_dict()
        assert "team" in d
        assert "players" in d
        assert "pending_suspensions" in d

    def test_no_suspension_for_low_yellows(self):
        events = [make_card_event("p1", "Player 1", "yellow_card", match_num=1)]
        report = analyze_suspensions(events, "Liga", "team_1")
        p1 = next(p for p in report.players if p.player_id == "p1")
        assert p1.is_suspended is False
        assert len(report.pending_suspensions) == 0

    def test_multiple_players_tracked(self):
        events = [
            make_card_event("p1", "Player 1", "yellow_card", match_num=1),
            make_card_event("p2", "Player 2", "yellow_card", match_num=1),
            make_card_event("p3", "Player 3", "red_card", match_num=1),
        ]
        report = analyze_suspensions(events, "Liga", "team_1")
        assert len(report.players) == 3
