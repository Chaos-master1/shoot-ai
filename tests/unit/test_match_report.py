"""Tests for Day After Match Report module."""

from __future__ import annotations

from kawkab.core.match_report import (
    DayAfterMatchReport,
    ReportTemplate,
    generate_match_report,
)


class TestMatchReport:
    def test_empty_events_returns_valid_structure(self):
        report = generate_match_report(
            match_id="m1",
            match_meta={
                "title": "Test",
                "date": "2025-01-01",
                "competition": "League",
                "result": "0-0",
            },
            events=[],
        )
        assert report.match_id == "m1"
        assert len(report.sections) == 10
        assert isinstance(report, DayAfterMatchReport)

    def test_goal_events_appear_in_key_moments(self):
        events = [
            {"type": "goal", "team": "home", "player": "P1", "minute": 23, "xg": 0.45},
        ]
        report = generate_match_report(
            "m1", {"title": "T", "date": "2025-01-01", "competition": "L", "result": "1-0"}, events
        )
        goals = [m for m in report.key_moments if m.type == "goal"]
        assert len(goals) == 1
        assert goals[0].minute == 23
        assert goals[0].player == "P1"

    def test_xg_above_03_appears_as_big_chance(self):
        events = [
            {"type": "shot", "team": "away", "player": "P2", "minute": 55, "xg": 0.55},
            {"type": "shot", "team": "home", "player": "P3", "minute": 60, "xg": 0.12},
        ]
        report = generate_match_report(
            "m1", {"title": "T", "date": "2025-01-01", "competition": "L", "result": "0-0"}, events
        )
        big_chances = [m for m in report.key_moments if m.type == "big_chance"]
        assert len(big_chances) == 1
        assert big_chances[0].xg == 0.55

    def test_executive_summary_non_empty(self):
        events = [
            {"type": "goal", "team": "home", "player": "P1", "minute": 10, "xg": 0.5},
            {"type": "goal", "team": "away", "player": "P2", "minute": 80, "xg": 0.3},
        ]
        report = generate_match_report(
            "m1",
            {"title": "Big Match", "date": "2025-01-01", "competition": "Cup", "result": "1-1"},
            events,
        )
        assert len(report.executive_summary) > 0
        assert "Big Match" in report.executive_summary

    def test_areas_for_improvement_generated_from_phase_data(self):
        events = [{"type": "shot", "team": "home", "player": "P1", "minute": 30, "xg": 0.1}]
        phase_xg = {
            "home_buildup_xg": 0.8,
            "away_buildup_xg": 1.5,
            "home_set_piece_xg": 0.1,
            "away_set_piece_xg": 0.6,
        }
        report = generate_match_report(
            "m1",
            {"title": "T", "date": "2025-01-01", "competition": "L", "result": "0-0"},
            events,
            phase_xg_report=phase_xg,
        )
        assert len(report.areas_for_improvement) > 0
        assert any("set piece" in a.lower() for a in report.areas_for_improvement)

    def test_to_markdown_produces_valid_markdown(self):
        report = generate_match_report(
            "m1",
            {"title": "Test", "date": "2025-01-01", "competition": "League", "result": "0-0"},
            [],
        )
        md = report.to_markdown()
        assert md.startswith("# Day After Match Report")
        assert "## Executive Summary" in md
        assert "## Key Moments" in md

    def test_report_template_excludes_sections(self):
        tmpl = ReportTemplate(name="brief", sections=["executive_summary", "key_moments"])
        report = generate_match_report(
            "m1",
            {"title": "T", "date": "2025-01-01", "competition": "L", "result": "0-0"},
            [],
            template=tmpl,
        )
        included = [s for s in report.sections if s.included]
        excluded = [s for s in report.sections if not s.included]
        assert len(included) == 2
        assert len(excluded) == 8
        assert included[0].title == "Executive Summary"
        assert included[1].title == "Key Moments"

    def test_brief_detail_level_truncates(self):
        tmpl = ReportTemplate(name="brief", detail_level="brief")
        events = [
            {"type": "goal", "team": "home", "player": "P", "minute": i, "xg": 0.5}
            for i in range(10)
        ]
        report = generate_match_report(
            "m1",
            {"title": "T", "date": "2025-01-01", "competition": "L", "result": "1-0"},
            events,
            template=tmpl,
        )
        km_section = [s for s in report.sections if s.title == "Key Moments"][0]
        assert "(truncated for brief mode)" in km_section.content

    def test_narrative_overrides_executive_summary(self):
        events = [{"type": "goal", "team": "home", "player": "P", "minute": 10, "xg": 0.5}]
        report = generate_match_report(
            "m1",
            {"title": "T", "date": "2025-01-01", "competition": "L", "result": "1-0"},
            events,
            narrative="Custom narrative text.",
        )
        assert report.executive_summary == "Custom narrative text."

    def test_player_ratings_highlights_populated(self):
        ratings = [
            {"name": "P1", "rating": 8.5, "highlight": "Scored a brace"},
            {"name": "P2", "rating": 7.0, "highlight": "Solid defending"},
        ]
        report = generate_match_report(
            "m1",
            {"title": "T", "date": "2025-01-01", "competition": "L", "result": "1-0"},
            [],
            player_ratings=ratings,
        )
        assert len(report.player_ratings_highlights) == 2
        assert report.player_ratings_highlights[0]["player"] == "P1"

    def test_set_piece_analysis_generated(self):
        events = [
            {"type": "corner", "team": "home", "minute": 15},
            {"type": "free_kick", "team": "home", "minute": 30},
            {"type": "shot", "team": "home", "minute": 31, "xg": 0.4, "set_piece": True},
        ]
        report = generate_match_report(
            "m1", {"title": "T", "date": "2025-01-01", "competition": "L", "result": "0-0"}, events
        )
        assert len(report.set_piece_analysis) > 0
        assert "corner" in report.set_piece_analysis

    def test_tactical_observations_from_events(self):
        events = [
            {"type": "pass", "team": "home", "minute": 1},
            {"type": "pass", "team": "home", "minute": 2},
        ] * 30
        report = generate_match_report(
            "m1", {"title": "T", "date": "2025-01-01", "competition": "L", "result": "0-0"}, events
        )
        assert len(report.tactical_observations) > 0

    def test_what_worked_well_generated(self):
        events = [{"type": "goal", "team": "home", "player": "P", "minute": 10, "xg": 0.5}]
        report = generate_match_report(
            "m1", {"title": "T", "date": "2025-01-01", "competition": "L", "result": "1-0"}, events
        )
        assert len(report.what_worked_well) > 0

    def test_key_moments_sorted_by_minute(self):
        events = [
            {"type": "goal", "team": "home", "player": "P1", "minute": 80, "xg": 0.3},
            {"type": "goal", "team": "home", "player": "P2", "minute": 10, "xg": 0.5},
        ]
        report = generate_match_report(
            "m1", {"title": "T", "date": "2025-01-01", "competition": "L", "result": "2-0"}, events
        )
        assert report.key_moments[0].minute == 10
        assert report.key_moments[1].minute == 80

    def test_card_events_in_key_moments(self):
        events = [
            {"type": "card", "team": "away", "player": "P3", "minute": 45, "card_type": "yellow"},
        ]
        report = generate_match_report(
            "m1", {"title": "T", "date": "2025-01-01", "competition": "L", "result": "0-0"}, events
        )
        cards = [m for m in report.key_moments if m.type == "card"]
        assert len(cards) == 1
        assert "yellow" in cards[0].description

    def test_to_json_valid(self):
        report = generate_match_report(
            "m1", {"title": "Test", "date": "2025-01-01", "competition": "L", "result": "0-0"}, []
        )
        js = report.to_json()
        assert '"match_id": "m1"' in js
        assert js.startswith("{")

    def test_phase_breakdown_included(self):
        phase = {
            "home_buildup_xg": 1.2,
            "away_buildup_xg": 0.8,
            "home_set_piece_xg": 0.3,
            "away_set_piece_xg": 0.1,
        }
        report = generate_match_report(
            "m1",
            {"title": "T", "date": "2025-01-01", "competition": "L", "result": "1-0"},
            [],
            phase_xg_report=phase,
        )
        assert report.phase_breakdown is not None
        assert "1.20" in report.phase_breakdown
