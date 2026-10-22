"""Tests for MultiMatchAnalysisService."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from tests.conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


class FakeRow:
    """Simulates sqlite3.Row with dict-like access."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data.items())

    def keys(self):
        return self._data.keys()

    def get(self, key, default=None):
        return self._data.get(key, default)


def _make_row(data: dict):
    return FakeRow(data)


@pytest.fixture(scope="module")
def mm_mod():
    return load_service_module(
        "kawkab.services.multi_match_analysis_service",
        "multi_match_analysis_service.py",
    )


class TestMultiMatchAnalysisService:
    def _mock_db(self, svc, rows_by_query: dict, param_matcher=None):
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def _set_result(result):
            if result is None:
                cursor.fetchone.return_value = None
                cursor.fetchall.return_value = []
            elif isinstance(result, list):
                cursor.fetchall.return_value = [
                    _make_row(r) if r is not None else None for r in result
                ]
                cursor.fetchone.return_value = (
                    _make_row(result[0]) if result and result[0] is not None else None
                )
            elif isinstance(result, dict):
                cursor.fetchone.return_value = _make_row(result)

        def _execute(sql, params=None):
            # Try param-based matcher first (use 'in' for sql to handle whitespace)
            if param_matcher is not None:
                for (sql_snippet, p), result in param_matcher.items():
                    if sql_snippet in sql and tuple(params) == p:
                        _set_result(result)
                        return cursor
            for snippet, result in rows_by_query.items():
                if snippet in sql:
                    _set_result(result)
                    break
            return cursor

        cursor.execute = _execute
        svc._conn = conn
        return svc

    def test_dataclass_season_summary(self, mm_mod):
        ss = mm_mod.SeasonSummary(
            season_id=1,
            season_name="Test",
            matches_played=10,
            wins=5,
            draws=2,
            losses=3,
            goals_for=20,
            goals_against=10,
            avg_possession=55.0,
            avg_pass_accuracy=82.0,
            avg_shots_per_match=12.5,
            total_distance_km=1200.0,
            avg_max_speed_kmh=30.0,
            formations_used=["4-3-3", "4-4-2"],
            most_common_formation="4-3-3",
        )
        assert ss.season_id == 1
        assert ss.most_common_formation == "4-3-3"
        assert ss.matches_played == 10

    def test_dataclass_player_trend(self, mm_mod):
        pt = mm_mod.PlayerTrend(
            player_id=1,
            player_name="P1",
            metric_name="distance_covered_m",
            values=[("2024-01-01", 5000.0)],
            trend_direction="improving",
            trend_slope=0.5,
            avg_value=5000.0,
            best_value=5000.0,
            worst_value=5000.0,
        )
        assert pt.trend_direction == "improving"
        assert pt.avg_value == 5000.0

    def test_dataclass_match_comparison(self, mm_mod):
        mc = mm_mod.MatchComparison(
            match_1_id=1,
            match_1_name="M1",
            match_2_id=2,
            match_2_name="M2",
            possession_diff={"match_1": 50, "match_2": 60, "delta": 10},
            shots_diff={"match_1": 5, "match_2": 8, "delta": 3},
            passes_diff={},
            formation_diff={},
            line_height_diff={},
            ppda_diff={},
            xg_diff={},
            key_differences=["test"],
            tactical_evolution="test",
        )
        assert mc.tactical_evolution == "test"

    def test_dataclass_team_evolution(self, mm_mod):
        te = mm_mod.TeamEvolution(
            period="season",
            matches_analyzed=5,
            formation_trend=[("4-3-3", 3)],
            possession_trend=[],
            ppda_trend=[],
            line_height_trend=[],
            shot_volume_trend=[],
            pass_accuracy_trend=[],
            overall_direction="stable",
        )
        assert te.matches_analyzed == 5

    @pytest.mark.asyncio
    async def test_get_season_summary(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "FROM seasons WHERE id = ?": {"id": 1, "name": "2024 Season"},
                "FROM matches": [
                    {
                        "matches": 10,
                        "gf": 25,
                        "ga": 12,
                        "avg_poss": 55.5,
                        "avg_passes": 400.0,
                        "avg_shots": 12.0,
                        "score_home": 3,
                        "score_away": 1,
                    },
                    {"score_home": 2, "score_away": 2},
                    {"score_home": 1, "score_away": 0},
                    {"score_home": 0, "score_away": 2},
                    {"score_home": 2, "score_away": 1},
                    {"score_home": 1, "score_away": 1},
                    {"score_home": 3, "score_away": 0},
                    {"score_home": 2, "score_away": 3},
                    {"score_home": 4, "score_away": 1},
                    {"score_home": 1, "score_away": 2},
                ],
                "JOIN matches m ON e.match_id = m.id": [{"avg_acc": 0.85}],
                "FROM analysis_results ar": [
                    {
                        "full_data": json.dumps(
                            {
                                "formations": {
                                    "home": {"formation": "4-3-3"},
                                    "away": {"formation": "4-4-2"},
                                }
                            }
                        )
                    },
                ],
            },
        )
        result = await svc.get_season_summary(1)
        assert result.season_id == 1
        assert result.season_name == "2024 Season"
        assert result.matches_played == 10
        assert result.most_common_formation == "4-3-3"

    @pytest.mark.asyncio
    async def test_get_season_summary_no_season(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "FROM seasons WHERE id = ?": None,
                "FROM matches": [
                    {
                        "matches": 0,
                        "gf": None,
                        "ga": None,
                        "avg_poss": None,
                        "avg_passes": None,
                        "avg_shots": None,
                        "score_home": 0,
                        "score_away": 0,
                    }
                ],
                "FROM analysis_results ar": [],
            },
        )
        result = await svc.get_season_summary(999)
        assert result.season_name == "Season 999"
        assert result.matches_played == 0

    @pytest.mark.asyncio
    async def test_get_season_summary_formation_parse_error(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "FROM seasons WHERE id = ?": {"id": 1, "name": "S1"},
                "FROM matches": [
                    {
                        "matches": 5,
                        "gf": 10,
                        "ga": 5,
                        "avg_poss": 50.0,
                        "avg_passes": 300.0,
                        "avg_shots": 8.0,
                        "score_home": 2,
                        "score_away": 1,
                    }
                ],
                "JOIN matches m ON e.match_id = m.id": [{"avg_acc": 0.75}],
                "FROM analysis_results ar": [{"full_data": "not valid json"}],
            },
        )
        result = await svc.get_season_summary(1)
        assert result.formations_used == []

    @pytest.mark.asyncio
    async def test_get_player_trend(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_match_links l": [
                    {
                        "display_name": "Player 1",
                        "match_date": "2024-01-01",
                        "distance_covered_m": 5000.0,
                        "max_speed_kmh": 30.0,
                        "avg_speed_kmh": 7.0,
                        "passes_attempted": 40,
                        "passes_completed": 35,
                        "shots": 3,
                        "tackles": 2,
                    },
                    {
                        "display_name": "Player 1",
                        "match_date": "2024-01-02",
                        "distance_covered_m": 5200.0,
                        "max_speed_kmh": 31.0,
                        "avg_speed_kmh": 7.5,
                        "passes_attempted": 45,
                        "passes_completed": 40,
                        "shots": 4,
                        "tackles": 1,
                    },
                    {
                        "display_name": "Player 1",
                        "match_date": "2024-01-03",
                        "distance_covered_m": 5500.0,
                        "max_speed_kmh": 32.0,
                        "avg_speed_kmh": 8.0,
                        "passes_attempted": 50,
                        "passes_completed": 45,
                        "shots": 5,
                        "tackles": 0,
                    },
                ],
            },
        )
        result = await svc.get_player_trend(1, metric="distance_covered_m", min_matches=2)
        assert result is not None
        assert result.player_name == "Player 1"
        assert len(result.values) == 3
        assert result.trend_direction in ("improving", "declining", "stable")

    @pytest.mark.asyncio
    async def test_get_player_trend_not_enough_matches(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_match_links l": [
                    {
                        "display_name": "P1",
                        "match_date": "2024-01-01",
                        "distance_covered_m": 1000.0,
                        "max_speed_kmh": 25.0,
                        "avg_speed_kmh": 5.0,
                        "passes_attempted": 10,
                        "passes_completed": 8,
                        "shots": 1,
                        "tackles": 0,
                    }
                ],
            },
        )
        result = await svc.get_player_trend(1, min_matches=5)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_player_trend_no_rows(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_match_links l": [],
            },
        )
        result = await svc.get_player_trend(999)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_player_trend_pass_accuracy(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "FROM player_match_links l": [
                    {
                        "display_name": "P1",
                        "match_date": "2024-01-01",
                        "distance_covered_m": 5000.0,
                        "max_speed_kmh": 30.0,
                        "avg_speed_kmh": 7.0,
                        "passes_attempted": 40,
                        "passes_completed": 30,
                        "shots": 3,
                        "tackles": 2,
                    },
                    {
                        "display_name": "P1",
                        "match_date": "2024-01-02",
                        "distance_covered_m": 5200.0,
                        "max_speed_kmh": 31.0,
                        "avg_speed_kmh": 7.5,
                        "passes_attempted": 50,
                        "passes_completed": 45,
                        "shots": 4,
                        "tackles": 1,
                    },
                    {
                        "display_name": "P1",
                        "match_date": "2024-01-03",
                        "distance_covered_m": 5500.0,
                        "max_speed_kmh": 32.0,
                        "avg_speed_kmh": 8.0,
                        "passes_attempted": 60,
                        "passes_completed": 55,
                        "shots": 5,
                        "tackles": 0,
                    },
                ],
            },
        )
        result = await svc.get_player_trend(1, metric="pass_accuracy")
        assert result is not None
        assert result.values[0][1] == 75.0
        assert result.values[1][1] == 90.0

    @pytest.mark.asyncio
    async def test_compare_matches(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {},
            param_matcher={
                ("FROM analysis_results WHERE match_id = ?", (1,)): {
                    "match_id": 1,
                    "full_data": json.dumps(
                        {
                            "possession_home": 45,
                            "shots_home": 5,
                            "passes_home": 300,
                            "formations": {"home": {"formation": "4-4-2"}},
                        }
                    ),
                },
                ("FROM analysis_results WHERE match_id = ?", (2,)): {
                    "match_id": 2,
                    "full_data": json.dumps(
                        {
                            "possession_home": 55,
                            "shots_home": 8,
                            "passes_home": 400,
                            "formations": {"home": {"formation": "4-3-3"}},
                        }
                    ),
                },
                ("FROM matches WHERE id = ?", (1,)): {"id": 1, "name": "Match 1"},
                ("FROM matches WHERE id = ?", (2,)): {"id": 2, "name": "Match 2"},
            },
        )
        result = await svc.compare_matches(1, 2)
        assert result.match_1_id == 1
        assert result.match_2_id == 2
        assert result.possession_diff["delta"] == 10.0
        assert len(result.key_differences) >= 1

    @pytest.mark.asyncio
    async def test_compare_matches_big_diffs(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {},
            param_matcher={
                ("FROM analysis_results WHERE match_id = ?", (1,)): {
                    "match_id": 1,
                    "full_data": json.dumps(
                        {
                            "possession_home": 30,
                            "shots_home": 2,
                            "passes_home": 150,
                            "formations": {"home": {"formation": "5-4-1"}},
                        }
                    ),
                },
                ("FROM analysis_results WHERE match_id = ?", (2,)): {
                    "match_id": 2,
                    "full_data": json.dumps(
                        {
                            "possession_home": 70,
                            "shots_home": 15,
                            "passes_home": 600,
                            "formations": {"home": {"formation": "4-3-3"}},
                        }
                    ),
                },
                ("FROM matches WHERE id = ?", (1,)): {"id": 1, "name": "M1"},
                ("FROM matches WHERE id = ?", (2,)): {"id": 2, "name": "M2"},
            },
        )
        result = await svc.compare_matches(1, 2)
        assert abs(result.possession_diff["delta"]) > 10
        assert abs(result.shots_diff["delta"]) >= 3
        assert "Formation changed" in result.key_differences[-1]

    @pytest.mark.asyncio
    async def test_compare_matches_no_analysis_data(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {},
            param_matcher={
                ("FROM analysis_results WHERE match_id = ?", (1,)): None,
                ("FROM analysis_results WHERE match_id = ?", (2,)): None,
                ("FROM matches WHERE id = ?", (1,)): {"id": 1, "name": "M1"},
                ("FROM matches WHERE id = ?", (2,)): {"id": 2, "name": "M2"},
            },
        )
        result = await svc.compare_matches(1, 2)
        assert result.tactical_evolution == "No significant tactical evolution detected"

    @pytest.mark.asyncio
    async def test_get_team_evolution_by_season(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "FROM analysis_results ar": [
                    {
                        "match_id": 1,
                        "full_data": json.dumps(
                            {
                                "formations": {
                                    "home": {"formation": "4-3-3", "line_height_m": 40.0}
                                },
                                "possession_home": 55,
                                "pressing_intensity": 8.0,
                                "shots_home": 10,
                                "home_team": {"pass_accuracy": 0.82},
                            }
                        ),
                        "match_date": "2024-01-01",
                        "name": "M1",
                    },
                    {
                        "match_id": 2,
                        "full_data": json.dumps(
                            {
                                "formations": {
                                    "home": {"formation": "4-3-3", "line_height_m": 45.0}
                                },
                                "possession_home": 58,
                                "pressing_intensity": 7.0,
                                "shots_home": 12,
                                "home_team": {"pass_accuracy": 0.85},
                            }
                        ),
                        "match_date": "2024-02-01",
                        "name": "M2",
                    },
                ],
            },
        )
        result = await svc.get_team_evolution(season_id=1)
        assert result.matches_analyzed == 2
        assert len(result.formation_trend) >= 1
        assert len(result.possession_trend) == 2
        assert len(result.ppda_trend) == 2

    @pytest.mark.asyncio
    async def test_get_team_evolution_by_match_ids(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "WHERE ar.match_id IN": [
                    {
                        "match_id": 1,
                        "full_data": json.dumps(
                            {
                                "formations": {"home": {"formation": "4-4-2"}},
                                "possession_home": 50,
                                "pressing_intensity": 6.0,
                                "shots_home": 8,
                                "home_team": {"pass_accuracy": 0.80},
                            }
                        ),
                        "match_date": "2024-01-01",
                        "name": "M1",
                    },
                ],
            },
        )
        result = await svc.get_team_evolution(match_ids=[1])
        assert result.matches_analyzed == 1

    @pytest.mark.asyncio
    async def test_get_team_evolution_no_filter(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "ORDER BY m.match_date ASC": [
                    {"match_id": 1, "full_data": "{}", "match_date": "2024-01-01", "name": "M1"},
                ],
            },
        )
        result = await svc.get_team_evolution()
        assert result.matches_analyzed == 1

    @pytest.mark.asyncio
    async def test_get_team_evolution_with_overall_direction(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "FROM analysis_results ar": [
                    {
                        "match_id": 1,
                        "full_data": json.dumps(
                            {
                                "formations": {"home": {"formation": "4-3-3"}},
                                "possession_home": 40,
                                "pressing_intensity": 10.0,
                                "shots_home": 5,
                                "home_team": {},
                            }
                        ),
                        "match_date": "2024-01-01",
                        "name": "M1",
                    },
                    {
                        "match_id": 2,
                        "full_data": json.dumps(
                            {
                                "formations": {"home": {"formation": "4-3-3"}},
                                "possession_home": 55,
                                "pressing_intensity": 5.0,
                                "shots_home": 12,
                                "home_team": {},
                            }
                        ),
                        "match_date": "2024-02-01",
                        "name": "M2",
                    },
                ],
            },
        )
        result = await svc.get_team_evolution(season_id=1)
        assert "more possession" in result.overall_direction
        assert (
            "more pressing" in result.overall_direction
            or "less pressing" in result.overall_direction
        )

    @pytest.mark.asyncio
    async def test_get_leaderboard_with_season(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "WHERE m.season_id = ?": [
                    {
                        "id": 1,
                        "display_name": "P1",
                        "jersey_number": 10,
                        "preferred_position": "FW",
                        "avg_distance": 5000.0,
                        "avg_max_speed": 30.0,
                        "avg_speed": 7.0,
                        "total_shots": 15,
                        "total_passes": 200,
                        "total_passes_completed": 180,
                        "matches": 5,
                    },
                    {
                        "id": 2,
                        "display_name": "P2",
                        "jersey_number": 4,
                        "preferred_position": "DF",
                        "avg_distance": 4500.0,
                        "avg_max_speed": 28.0,
                        "avg_speed": 6.5,
                        "total_shots": 2,
                        "total_passes": 250,
                        "total_passes_completed": 230,
                        "matches": 5,
                    },
                ],
            },
        )
        result = await svc.get_leaderboard(season_id=1, metric="distance_covered_m", top_n=10)
        assert len(result) == 2
        assert result[0]["name"] == "P1"
        assert result[0]["pass_accuracy"] > 0

    @pytest.mark.asyncio
    async def test_get_leaderboard_no_season(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "GROUP BY p.id": [
                    {
                        "id": 1,
                        "display_name": "P1",
                        "jersey_number": 10,
                        "preferred_position": "MF",
                        "avg_distance": 4800.0,
                        "avg_max_speed": 29.0,
                        "avg_speed": 6.8,
                        "total_shots": 8,
                        "total_passes": 300,
                        "total_passes_completed": 270,
                        "matches": 3,
                    },
                ],
            },
        )
        result = await svc.get_leaderboard(top_n=5)
        assert len(result) == 1
        assert result[0]["position"] == "MF"

    @pytest.mark.asyncio
    async def test_get_leaderboard_zero_passes(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        svc = self._mock_db(
            svc,
            {
                "GROUP BY p.id": [
                    {
                        "id": 3,
                        "display_name": "P3",
                        "jersey_number": 1,
                        "preferred_position": "GK",
                        "avg_distance": 2000.0,
                        "avg_max_speed": 20.0,
                        "avg_speed": 4.0,
                        "total_shots": 0,
                        "total_passes": 0,
                        "total_passes_completed": 0,
                        "matches": 2,
                    },
                ],
            },
        )
        result = await svc.get_leaderboard()
        assert result[0]["pass_accuracy"] == 0.0

    @pytest.mark.asyncio
    async def test_close_clears_connection(self, mm_mod):
        svc = mm_mod.MultiMatchAnalysisService()
        conn = MagicMock()
        svc._conn = conn
        await svc.close()
        conn.close.assert_called_once()
        assert svc._conn is None
