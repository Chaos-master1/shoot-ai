"""Tests for the pre-match briefing service."""

from __future__ import annotations

import pytest

from kawkab.analysis.prematch_briefing import (
    FormEntry,
    InjuryEntry,
    KeyBattle,
    MatchPrediction,
    PredictedLineupPlayer,
    PreMatchBriefing,
    PreMatchBriefingService,
    TacticalRecommendations,
)

# ── helpers ────────────────────────────────────────────────────────


@pytest.fixture
def service():
    return PreMatchBriefingService()


_FORM_SAMPLE = [
    {
        "result": "W",
        "opponent": "Team A",
        "home_score": 2,
        "away_score": 0,
        "competition": "League",
        "date": "2026-06-01",
    },
    {
        "result": "D",
        "opponent": "Team B",
        "home_score": 1,
        "away_score": 1,
        "competition": "League",
        "date": "2026-06-08",
    },
    {
        "result": "W",
        "opponent": "Team C",
        "home_score": 3,
        "away_score": 1,
        "competition": "Cup",
        "date": "2026-06-15",
    },
    {
        "result": "W",
        "opponent": "Team D",
        "home_score": 2,
        "away_score": 0,
        "competition": "League",
        "date": "2026-06-22",
    },
    {
        "result": "W",
        "opponent": "Team E",
        "home_score": 1,
        "away_score": 0,
        "competition": "League",
        "date": "2026-06-29",
    },
]


# ── data class tests ───────────────────────────────────────────────


class TestFormEntry:
    def test_defaults(self):
        f = FormEntry()
        assert f.result == ""

    def test_to_dict(self):
        f = FormEntry("W", "Opp", 2, 1, "League", "2026-07-01")
        d = f.to_dict()
        assert d["result"] == "W"
        assert d["opponent"] == "Opp"


class TestInjuryEntry:
    def test_to_dict(self):
        i = InjuryEntry("Player A", "Hamstring", "2 weeks", "medium")
        d = i.to_dict()
        assert d["player"] == "Player A"
        assert d["severity"] == "medium"


class TestPredictedLineupPlayer:
    def test_to_dict_rounds_rating(self):
        p = PredictedLineupPlayer("CF", "Striker A", 9, 87.56)
        d = p.to_dict()
        assert d["rating"] == 87.6
        assert d["position"] == "CF"


class TestKeyBattle:
    def test_default_importance(self):
        kb = KeyBattle()
        assert kb.importance == "medium"

    def test_to_dict_our_advantage_none(self):
        kb = KeyBattle("A", "B", "high", None)
        d = kb.to_dict()
        assert d["our_advantage"] is None


class TestTacticalRecommendations:
    def test_to_dict_includes_nested(self):
        tr = TacticalRecommendations(
            suggested_formation="4-3-3",
            key_battles=[KeyBattle("A", "B", "high", True)],
        )
        d = tr.to_dict()
        assert d["suggested_formation"] == "4-3-3"
        assert d["key_battles"][0]["our_player"] == "A"
        assert d["key_battles"][0]["our_advantage"] is True


class TestMatchPrediction:
    def test_to_dict_rounds(self):
        mp = MatchPrediction(1.234, 0.567, 0.4567, 0.2891, 0.2542, 0.6789, "high")
        d = mp.to_dict()
        assert d["home_score"] == 1.2
        assert d["away_score"] == 0.6
        assert d["win_probability"] == 0.457
        assert d["btts_probability"] == 0.679
        assert d["confidence"] == "high"


# ── PreMatchBriefing tests ────────────────────────────────────────


class TestPreMatchBriefing:
    def test_default_auto_generates_metadata(self):
        b = PreMatchBriefing()
        assert b.generated_at
        assert len(b.briefing_id) == 12

    def test_custom_id_and_timestamp(self):
        ts = "2026-07-13T10:00:00+00:00"
        b = PreMatchBriefing(generated_at=ts, briefing_id="abc123")
        assert b.generated_at == ts
        assert b.briefing_id == "abc123"

    def test_to_dict_structure(self):
        b = PreMatchBriefing(
            home_team="Home FC",
            away_team="Away FC",
            competition="Premier League",
            venue="Stadium",
            our_team="Home FC",
        )
        d = b.to_dict()
        assert d["match_info"]["home_team"] == "Home FC"
        assert d["our_team"]["name"] == "Home FC"
        assert d["tactical"]["suggested_formation"] == ""

    def test_to_dict_opponent_is_away_when_our_team_is_home(self):
        b = PreMatchBriefing(home_team="Home", away_team="Away", our_team="Home")
        d = b.to_dict()
        assert d["opponent"]["name"] == "Away"

    def test_to_dict_opponent_is_home_when_our_team_is_away(self):
        b = PreMatchBriefing(home_team="Home", away_team="Away", our_team="Away")
        d = b.to_dict()
        assert d["opponent"]["name"] == "Home"

    def test_to_markdown_basic(self):
        b = PreMatchBriefing(home_team="Home", away_team="Away", competition="League")
        md = b.to_markdown()
        assert "Home vs Away" in md
        assert "League" in md

    def test_to_markdown_with_injuries(self):
        b = PreMatchBriefing(
            home_team="Home",
            away_team="Away",
            our_team="Home",
            our_injuries=[InjuryEntry("P1", "Knee", "2 weeks", "high")],
        )
        md = b.to_markdown()
        assert "P1" in md
        assert "Knee" in md
        assert "2 weeks" in md

    def test_to_markdown_with_predicted_lineup(self):
        b = PreMatchBriefing(
            home_team="H",
            away_team="A",
            our_team="H",
            our_predicted_lineup=[PredictedLineupPlayer("GK", "Keeper", 1, 85.0)],
        )
        md = b.to_markdown()
        assert "GK" in md
        assert "Keeper" in md

    def test_to_markdown_with_prediction(self):
        b = PreMatchBriefing(
            home_team="H",
            away_team="A",
            prediction=MatchPrediction(
                home_score=1.5,
                away_score=0.8,
                win_probability=0.55,
                draw_probability=0.25,
                loss_probability=0.20,
                btts_probability=0.60,
                confidence="medium",
            ),
        )
        md = b.to_markdown()
        assert "55%" in md
        assert "25%" in md
        assert "60%" in md

    def test_to_markdown_no_injuries_shows_message(self):
        b = PreMatchBriefing(
            home_team="H",
            away_team="A",
            our_team="H",
            our_injuries=[],
            our_suspensions=[],
        )
        md = b.to_markdown()
        assert "No injury or suspension concerns" in md

    def test_to_html_basic(self):
        b = PreMatchBriefing(home_team="Home", away_team="Away", competition="Cup")
        html = b.to_html()
        assert 'class="manager-briefing"' in html
        assert "Home vs Away" in html
        assert "Cup" in html

    def test_to_html_with_weather(self):
        b = PreMatchBriefing(
            home_team="H",
            away_team="A",
            weather_condition="Rainy",
            weather_temperature=12.0,
        )
        html = b.to_html()
        assert "Rainy" in html
        assert "12" in html

    def test_to_html_grid_structure(self):
        b = PreMatchBriefing(home_team="H", away_team="A")
        html = b.to_html()
        assert "briefing-grid" in html
        assert "briefing-column" in html

    def test_kickoff_formatted(self):
        b = PreMatchBriefing(home_team="H", away_team="A", kickoff="15:00")
        md = b.to_markdown()
        assert "15:00" in md

    def test_referee_in_markdown_title(self):
        b = PreMatchBriefing(home_team="H", away_team="A", referee_name="Mike Dean")
        md = b.to_markdown()
        assert "Mike Dean" in md

    def test_h2h_in_markdown(self):
        b = PreMatchBriefing(
            home_team="H",
            away_team="A",
            h2h_total_meetings=10,
            h2h_home_wins=6,
            h2h_away_wins=2,
            h2h_draws=2,
            h2h_avg_goals=2.5,
            h2h_recent=[
                {"date": "2025-01-01", "home": "H", "away": "A", "home_score": 2, "away_score": 1},
            ],
        )
        md = b.to_markdown()
        assert "10" in md
        assert "2.50" in md

    def test_tactical_in_markdown(self):
        b = PreMatchBriefing(
            home_team="H",
            away_team="A",
            tactical=TacticalRecommendations(
                suggested_formation="4-4-2",
                pressing_strategy="High",
                key_battles=[KeyBattle("Kante", "De Bruyne", "high", True)],
                notes=["Watch corners"],
            ),
        )
        md = b.to_markdown()
        assert "4-4-2" in md
        assert "High" in md
        assert "Kante" in md
        assert "De Bruyne" in md
        assert "Watch corners" in md


# ── PreMatchBriefingService tests ─────────────────────────────────


class TestPreMatchBriefingService:
    def test_available_property(self, service):
        assert service.available is True

    def test_generate_basic(self, service):
        b = service.generate(
            home_team="Home FC",
            away_team="Away FC",
            our_side="home",
            competition="League",
        )
        assert isinstance(b, PreMatchBriefing)
        assert b.home_team == "Home FC"
        assert b.away_team == "Away FC"
        assert b.our_team == "Home FC"
        assert b.competition == "League"

    def test_generate_our_team_away_side(self, service):
        b = service.generate(
            home_team="Home FC",
            away_team="Away FC",
            our_side="away",
        )
        assert b.our_team == "Away FC"

    def test_generate_with_form(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            our_form_data=_FORM_SAMPLE,
        )
        assert len(b.our_form) == 5
        assert b.our_form[0].result == "W"
        assert b.our_form[0].opponent == "Team A"

    def test_generate_streak_winning(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            our_form_data=[{"result": "W"}, {"result": "W"}, {"result": "W"}],
        )
        assert "3 winning" in b.our_streak or "winning" in b.our_streak

    def test_generate_streak_losing(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            our_form_data=[{"result": "L"}, {"result": "L"}],
        )
        assert "2 losing" in b.our_streak or "losing" in b.our_streak

    def test_generate_streak_single_result(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            our_form_data=[{"result": "D"}],
        )
        assert "Last: D" in b.our_streak or "D" in b.our_streak

    def test_generate_with_injuries(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            our_injuries_data=[
                {
                    "player": "P1",
                    "injury": "Hamstring",
                    "expected_return": "2 weeks",
                    "severity": "medium",
                },
            ],
        )
        assert len(b.our_injuries) == 1
        assert b.our_injuries[0].player == "P1"

    def test_generate_with_suspensions(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            our_suspensions_data=["Player X", "Player Y"],
        )
        assert b.our_suspensions == ["Player X", "Player Y"]

    def test_generate_predicted_lineup(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            our_predicted_lineup_data=[
                {"position": "GK", "player_name": "Keeper", "number": 1, "rating": 90.0},
            ],
        )
        assert len(b.our_predicted_lineup) == 1
        assert b.our_predicted_lineup[0].position == "GK"

    def test_generate_opponent_data(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            opponent_form_data=_FORM_SAMPLE[:3],
            opponent_preferred_formation="3-5-2",
            opponent_pressing="High intensity",
            opponent_build_up="Short passes",
            opponent_key_players_data=[
                {"name": "Star A", "threat_score": 92, "key_stat": "10 goals"},
            ],
        )
        assert len(b.opponent_form) == 3
        assert b.opponent_preferred_formation == "3-5-2"
        assert b.opponent_pressing == "High intensity"
        assert len(b.opponent_key_players) == 1

    def test_generate_h2h(self, service):
        b = service.generate(
            home_team="Home",
            away_team="Away",
            our_side="home",
            h2h_data=[
                {
                    "home": "Home",
                    "away": "Away",
                    "home_score": 2,
                    "away_score": 1,
                    "date": "2025-01-01",
                },
                {
                    "home": "Home",
                    "away": "Away",
                    "home_score": 1,
                    "away_score": 1,
                    "date": "2025-06-01",
                },
                {
                    "home": "Away",
                    "away": "Home",
                    "home_score": 0,
                    "away_score": 3,
                    "date": "2024-12-01",
                },
            ],
        )
        assert b.h2h_total_meetings == 3
        assert b.h2h_home_wins > 0
        assert b.h2h_avg_goals > 0

    def test_generate_h2h_away_perspective(self, service):
        b = service.generate(
            home_team="Home",
            away_team="Away",
            our_side="away",
            h2h_data=[
                {
                    "home": "Home",
                    "away": "Away",
                    "home_score": 2,
                    "away_score": 1,
                    "date": "2025-01-01",
                },
            ],
        )
        # "Home" won 2-1, but our_side is "away", so that counts as away win for us
        assert b.h2h_away_wins == 1

    def test_generate_h2h_empty(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home")
        assert b.h2h_total_meetings == 0
        assert b.h2h_avg_goals == 0.0

    def test_generate_tactical(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            suggested_formation="4-2-3-1",
            pressing_strategy="Mid-block",
            attacking_focus="Left flank overloads",
            defensive_focus="Compact central shape",
            set_piece_plans=["Short corners", "Near-post on crosses"],
            key_battles_data=[
                {
                    "our_player": "LB",
                    "opponent_player": "RW",
                    "importance": "high",
                    "our_advantage": False,
                },
            ],
            tactical_notes=["Avoid early yellow cards"],
        )
        t = b.tactical
        assert t.suggested_formation == "4-2-3-1"
        assert t.pressing_strategy == "Mid-block"
        assert len(t.set_piece_plans) == 2
        assert len(t.key_battles) == 1
        assert t.key_battles[0].our_advantage is False
        assert "yellow cards" in t.notes[0]

    def test_generate_prediction(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            home_score_pred=2.0,
            away_score_pred=1.0,
            win_prob=0.6,
            draw_prob=0.2,
            loss_prob=0.2,
            btts_prob=0.55,
            prediction_confidence="high",
        )
        p = b.prediction
        assert p.home_score == 2.0
        assert p.away_score == 1.0
        assert p.win_probability == 0.6
        assert p.confidence == "high"

    def test_generate_referee(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            referee_name="Mike Dean",
            referee_avg_cards=4.5,
            referee_home_bias=0.08,
            referee_foul_threshold=12.0,
            referee_inconsistency=0.15,
        )
        assert b.referee_name == "Mike Dean"
        assert b.referee_avg_cards == 4.5

    def test_generate_weather(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            weather_condition="Rainy",
            weather_temperature=10.0,
            weather_wind=25.0,
            weather_precipitation=3.0,
        )
        assert b.weather_condition == "Rainy"
        assert b.weather_temperature == 10.0

    def test_generate_opponent_vulnerabilities_and_strengths(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            opponent_vulnerabilities=["Slow CBs", "High defensive line"],
            opponent_strengths=["Counter attacks", "Set pieces"],
        )
        assert "Slow CBs" in b.opponent_vulnerabilities
        assert "Counter attacks" in b.opponent_strengths

    def test_generate_opponent_set_piece_tendencies(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            opponent_set_piece_tendencies=["Short corners", "Zonal marking"],
        )
        assert "Short corners" in b.opponent_set_piece_tendencies

    def test_generate_opponent_predicted_lineup(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            opponent_predicted_lineup_data=[
                {"position": "CF", "player_name": "Opp Striker", "number": 9, "rating": 88.0},
            ],
        )
        assert len(b.opponent_predicted_lineup) == 1
        assert b.opponent_predicted_lineup[0].player_name == "Opp Striker"

    def test_generate_top_scorer_and_possession(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            our_top_scorer="Lewandowski",
            our_avg_possession=58.7,
        )
        assert b.our_top_scorer == "Lewandowski"
        assert b.our_avg_possession == 58.7

    def test_generate_all_kickoff_match_date_venue(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            match_date="2026-07-15",
            kickoff="20:00",
            venue="Camp Nou",
        )
        assert b.match_date == "2026-07-15"
        assert b.kickoff == "20:00"
        assert b.venue == "Camp Nou"

    def test_generate_streak_empty_no_error(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home")
        assert b.our_streak == ""

    def test_generate_form_fields_normalized(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            our_form_data=[
                {"result": None, "home_score": None, "away_score": "3"},
            ],
        )
        assert b.our_form[0].result == "None"
        assert b.our_form[0].home_score == 0
        assert b.our_form[0].away_score == 3

    def test_generate_h2h_avg_goals_zero_on_no_h2h(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home")
        assert b.h2h_avg_goals == 0.0

    def test_generate_briefing_has_generated_at(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home")
        assert b.generated_at

    def test_generate_briefing_has_briefing_id(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home")
        assert len(b.briefing_id) == 12


# ── integration-ish: service → object round-trip ──────────────────


class TestRoundTrip:
    def test_full_briefing_to_dict_and_back(self, service):
        b = service.generate(
            home_team="Barcelona",
            away_team="Real Madrid",
            our_side="home",
            competition="La Liga",
            venue="Camp Nou",
            match_date="2026-07-15",
            kickoff="21:00",
            referee_name="Mike Dean",
            our_form_data=_FORM_SAMPLE,
            our_injuries_data=[
                {
                    "player": "P1",
                    "injury": "Knee",
                    "expected_return": "3 weeks",
                    "severity": "high",
                },
            ],
            our_suspensions_data=["P2"],
            our_top_scorer="Messi",
            our_formation="4-3-3",
            our_predicted_lineup_data=[
                {"position": "GK", "player_name": "Ter Stegen", "number": 1, "rating": 89.0},
                {"position": "CF", "player_name": "Messi", "number": 10, "rating": 94.0},
            ],
            our_avg_possession=62.0,
            opponent_form_data=_FORM_SAMPLE[:3],
            opponent_preferred_formation="4-4-2",
            opponent_pressing="Medium",
            opponent_build_up="Long ball",
            opponent_key_players_data=[
                {"name": "Vinicius", "threat_score": 85, "key_stat": "8 assists"},
            ],
            opponent_vulnerabilities=["High line"],
            opponent_strengths=["Counter attack"],
            h2h_data=[
                {
                    "home": "Barcelona",
                    "away": "Real Madrid",
                    "home_score": 2,
                    "away_score": 1,
                    "date": "2025-03-01",
                },
                {
                    "home": "Real Madrid",
                    "away": "Barcelona",
                    "home_score": 1,
                    "away_score": 1,
                    "date": "2025-10-01",
                },
            ],
            suggested_formation="4-3-3",
            pressing_strategy="High press",
            attacking_focus="Right side",
            defensive_focus="Stay compact",
            set_piece_plans=["Short corners"],
            key_battles_data=[
                {
                    "our_player": "Messi",
                    "opponent_player": "Carvajal",
                    "importance": "high",
                    "our_advantage": True,
                },
            ],
            tactical_notes=["Avoid early card"],
            home_score_pred=2.3,
            away_score_pred=1.1,
            win_prob=0.65,
            draw_prob=0.20,
            loss_prob=0.15,
            btts_prob=0.60,
            prediction_confidence="high",
            weather_condition="Clear",
            weather_temperature=22.0,
        )

        d = b.to_dict()
        assert d["match_info"]["home_team"] == "Barcelona"
        assert d["our_team"]["name"] == "Barcelona"
        assert d["our_team"]["top_scorer"] == "Messi"
        assert len(d["our_team"]["predicted_lineup"]) == 2
        assert d["opponent"]["preferred_formation"] == "4-4-2"
        assert len(d["head_to_head"]["recent"]) == 2
        assert d["referee"]["name"] == "Mike Dean"
        assert d["weather"]["temperature"] == 22.0
        assert d["tactical"]["suggested_formation"] == "4-3-3"
        assert abs(d["prediction"]["win_probability"] - 0.65) < 0.001
        assert d["prediction"]["confidence"] == "high"
        assert b.briefing_id and len(b.briefing_id) == 12

    def test_markdown_renders_all_sections(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            competition="League",
            venue="Stadium",
            our_form_data=_FORM_SAMPLE,
            opponent_form_data=_FORM_SAMPLE[:3],
            our_injuries_data=[
                {"player": "P1", "injury": "Knee", "expected_return": "2w", "severity": "medium"}
            ],
            our_predicted_lineup_data=[
                {"position": "GK", "player_name": "Keeper", "number": 1, "rating": 85}
            ],
            opponent_key_players_data=[{"name": "Star", "threat_score": 90, "key_stat": "10g"}],
            h2h_data=[
                {"home": "H", "away": "A", "home_score": 1, "away_score": 0, "date": "2025-01-01"}
            ],
            referee_name="Ref",
            suggested_formation="4-4-2",
            pressing_strategy="High",
            home_score_pred=2.0,
            away_score_pred=0.5,
            win_prob=0.7,
            draw_prob=0.2,
            loss_prob=0.1,
            btts_prob=0.4,
            prediction_confidence="high",
            weather_condition="Rainy",
        )
        md = b.to_markdown()
        expected_sections = [
            "Form Guide",
            "Team News",
            "Opponent",
            "Head-to-Head",
            "Tactical Plan",
            "Match Prediction",
        ]
        for section in expected_sections:
            assert section in md, f"Missing section: {section}"
        assert "Rainy" in md
        assert "Ref" in md

    def test_html_renders_grid(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home")
        html = b.to_html()
        assert "manager-briefing" in html
        assert "briefing-grid" in html
        assert "briefing-column" in html


# ── edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_form_list(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home", our_form_data=[])
        assert b.our_form == []
        assert b.our_streak == ""

    def test_injury_entry_missing_fields(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            our_injuries_data=[{"player": "P1"}],
        )
        assert b.our_injuries[0].player == "P1"
        assert b.our_injuries[0].injury == ""

    def test_suspensions_empty_list(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            our_suspensions_data=[],
        )
        assert b.our_suspensions == []

    def test_h2h_null_scores(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            h2h_data=[{"home_score": None, "away_score": None}],
        )
        assert b.h2h_total_meetings == 1
        assert b.h2h_avg_goals == 0.0

    def test_opponent_key_players_empty(self, service):
        b = service.generate(
            home_team="H",
            away_team="A",
            our_side="home",
            opponent_key_players_data=[],
        )
        assert b.opponent_key_players == []

    def test_no_injuries_or_suspensions_in_markdown(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home")
        md = b.to_markdown()
        assert "No injury or suspension concerns" in md

    def test_tactical_empty(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home")
        assert b.tactical.suggested_formation == ""
        assert b.tactical.key_battles == []
        assert b.tactical.notes == []

    def test_prediction_defaults(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home")
        assert b.prediction.home_score == 0.0
        assert b.prediction.confidence == "medium"

    def test_weather_empty(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home")
        assert b.weather_condition == ""

    def test_long_form_streak(self, service):
        data = [{"result": "W"}] * 10
        b = service.generate(home_team="H", away_team="A", our_side="home", our_form_data=data)
        assert "10 winning" in b.our_streak or f"{'winning'}" in b.our_streak

    def test_referee_matches_markdown(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home", referee_name="M.D.")
        md = b.to_markdown()
        assert "M.D." in md

    def test_briefing_id_unique(self):
        b1 = PreMatchBriefing()
        b2 = PreMatchBriefing()
        assert b1.briefing_id != b2.briefing_id

    def test_to_dict_roundtrip_no_errors(self, service):
        b = service.generate(home_team="H", away_team="A", our_side="home")
        d = b.to_dict()
        assert isinstance(d, dict)
        assert d["match_info"]["home_team"] == "H"

    def test_service_available_flag(self):
        s = PreMatchBriefingService()
        assert s.available is True
