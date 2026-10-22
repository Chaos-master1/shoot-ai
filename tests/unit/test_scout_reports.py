"""Tests for Scout Report Generation."""

from kawkab.core.scout_reports import (
    SCOUT_METRIC_MEANS,
    SCOUT_METRICS,
    STAT_MEANS,
    _z_to_percentile,
    generate_scout_report,
)


def _make_stats(values: dict) -> dict:
    return {k: float(v) for k, v in values.items()}


def _default_similar_players() -> list[dict]:
    return [
        {
            "name": "Player A",
            "similarity_score": 0.92,
            "club": "Club A",
            "league": "Premier League",
            "key_comparison_metrics": {
                "xG_per_90": 0.35,
                "xA_per_90": 0.18,
            },
        },
    ]


STAR_STATS = {
    "xG_per_90": 1.2,
    "xA_per_90": 0.5,
    "shots_per_90": 5.0,
    "pass_completion_pct": 92.0,
    "progressive_passes_per_90": 12.0,
    "touches_in_box_per_90": 8.0,
    "pressures_per_90": 25.0,
    "tackles_per_90": 6.0,
    "interceptions_per_90": 4.5,
    "aerial_win_pct": 78.0,
    "dribbles_completed_pct": 80.0,
    "key_passes_per_90": 4.0,
    "passes_into_final_third_per_90": 18.0,
}

POOR_STATS = {
    "xG_per_90": 0.01,
    "xA_per_90": 0.01,
    "shots_per_90": 0.2,
    "pass_completion_pct": 50.0,
    "progressive_passes_per_90": 0.5,
    "touches_in_box_per_90": 0.1,
    "pressures_per_90": 2.0,
    "tackles_per_90": 0.3,
    "interceptions_per_90": 0.2,
    "aerial_win_pct": 20.0,
    "dribbles_completed_pct": 25.0,
    "key_passes_per_90": 0.1,
    "passes_into_final_third_per_90": 0.5,
}

AVERAGE_STATS = {k: STAT_MEANS[k] for k in SCOUT_METRICS if k in STAT_MEANS}
for k in SCOUT_METRICS:
    if k not in AVERAGE_STATS:
        AVERAGE_STATS[k] = SCOUT_METRIC_MEANS.get(k, 0.0)


class TestZToPercentile:
    def test_zero_z_is_50th_percentile(self):
        assert _z_to_percentile(0.0) == 50.0

    def test_positive_z_above_50(self):
        assert _z_to_percentile(1.0) > 50.0
        assert _z_to_percentile(2.0) > 95.0

    def test_negative_z_below_50(self):
        assert _z_to_percentile(-1.0) < 50.0
        assert _z_to_percentile(-2.0) < 5.0


class TestGenerateScoutReport:
    def test_empty_stats_returns_no_strengths_or_weaknesses(self):
        report = generate_scout_report(
            player_id="1",
            player_name="Empty Player",
            player_stats={},
            squad_stats={},
            league_stats={},
            similar_players=[],
        )
        assert report.strengths == []
        assert report.weaknesses == []
        assert report.comparison_table == {}
        assert report.similar_players == []
        assert report.recommendation != ""

    def test_no_similar_players(self):
        report = generate_scout_report(
            player_id="1",
            player_name="Lonely Player",
            player_stats=STAR_STATS,
            squad_stats={},
            league_stats={},
            similar_players=[],
        )
        assert report.similar_players == []

    def test_star_player_identifies_strengths(self):
        report = generate_scout_report(
            player_id="1",
            player_name="Star Player",
            player_stats=STAR_STATS,
            squad_stats={},
            league_stats={},
            similar_players=_default_similar_players(),
        )
        assert len(report.strengths) > 0
        assert len(report.weaknesses) == 0

    def test_poor_player_identifies_weaknesses(self):
        report = generate_scout_report(
            player_id="2",
            player_name="Poor Player",
            player_stats=POOR_STATS,
            squad_stats={},
            league_stats={},
            similar_players=_default_similar_players(),
        )
        assert len(report.weaknesses) > 0
        assert len(report.strengths) == 0

    def test_average_player_no_strengths_or_weaknesses(self):
        report = generate_scout_report(
            player_id="3",
            player_name="Average Player",
            player_stats=AVERAGE_STATS,
            squad_stats={},
            league_stats={},
            similar_players=_default_similar_players(),
        )
        assert report.strengths == []
        assert report.weaknesses == []

    def test_similar_players_included_in_report(self):
        sims = [
            {
                "name": "P1",
                "similarity_score": 0.95,
                "club": "C1",
                "league": "L1",
                "key_comparison_metrics": {},
            },
            {
                "name": "P2",
                "similarity_score": 0.88,
                "club": "C2",
                "league": "L2",
                "key_comparison_metrics": {},
            },
            {
                "name": "P3",
                "similarity_score": 0.82,
                "club": "C3",
                "league": "L3",
                "key_comparison_metrics": {},
            },
        ]
        report = generate_scout_report(
            player_id="1",
            player_name="Target",
            player_stats=STAR_STATS,
            squad_stats={},
            league_stats={},
            similar_players=sims,
        )
        assert len(report.similar_players) == 3
        assert report.similar_players[0]["name"] == "P1"
        assert report.similar_players[0]["similarity_score"] == 0.95

    def test_comparison_table_contains_all_scout_metrics(self):
        report = generate_scout_report(
            player_id="1",
            player_name="Full Data",
            player_stats=STAR_STATS,
            squad_stats={},
            league_stats={},
            similar_players=_default_similar_players(),
        )
        for metric in SCOUT_METRICS:
            assert metric in report.comparison_table, f"Missing {metric} in comparison table"
            entry = report.comparison_table[metric]
            assert "player_value" in entry
            assert "league_average" in entry
            assert "percentile" in entry
            assert "is_strength" in entry
            assert "is_weakness" in entry

    def test_squad_averages_appear_in_comparison_table(self):
        squad = {m: float(v * 0.8) for m, v in STAR_STATS.items()}
        report = generate_scout_report(
            player_id="1",
            player_name="Squad Compare",
            player_stats=STAR_STATS,
            squad_stats=squad,
            league_stats={},
            similar_players=[],
        )
        for metric in SCOUT_METRICS:
            assert report.comparison_table[metric]["squad_average"] == squad[metric]

    def test_report_metadata_fields(self):
        report = generate_scout_report(
            player_id="42",
            player_name="Messi",
            player_stats=STAR_STATS,
            squad_stats={},
            league_stats={},
            similar_players=[],
        )
        assert report.target_player_id == "42"
        assert report.target_player_name == "Messi"
        assert report.report_date != ""
        assert report.recommendation != ""
        assert report.comparable_rating != ""

    def test_recommendation_scale(self):
        star = generate_scout_report("1", "Star", STAR_STATS, {}, {}, [])
        poor = generate_scout_report("2", "Poor", POOR_STATS, {}, {}, [])
        assert "Highly recommended" in star.recommendation
        assert "Development project" in poor.recommendation

    def test_comparable_rating_scale(self):
        star = generate_scout_report("1", "Star", STAR_STATS, {}, {}, [])
        poor = generate_scout_report("2", "Poor", POOR_STATS, {}, {}, [])
        assert star.comparable_rating in ("Elite level", "Premier League level")
        assert poor.comparable_rating in ("Development level", "League One level")

    def test_star_player_has_high_percentiles(self):
        report = generate_scout_report("1", "Star", STAR_STATS, {}, {}, [])
        for entry in report.comparison_table.values():
            assert entry["percentile"] > 50.0

    def test_poor_player_has_low_percentiles(self):
        report = generate_scout_report("2", "Poor", POOR_STATS, {}, {}, [])
        for entry in report.comparison_table.values():
            assert entry["percentile"] < 50.0

    def test_league_stats_override_defaults(self):
        league = {"xG_per_90": 0.5, "xA_per_90": 0.3}
        report = generate_scout_report(
            "1",
            "Test",
            STAR_STATS,
            {},
            league,
            [],
        )
        assert report.comparison_table["xG_per_90"]["league_average"] == 0.5
        assert report.comparison_table["xA_per_90"]["league_average"] == 0.3
