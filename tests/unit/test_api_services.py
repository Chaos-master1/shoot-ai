"""Tests for all 7 external API service modules.

Covers: ApiFootball, Bzzoiro, FootballData, StatsBomb, TheSportsDB,
OpenFootballData, and RoboflowSportsService.

Each service gets at least 5 tests covering:
  - successful request returns expected data structure
  - empty / missing-key response handled
  - HTTP error status handled gracefully
  - invalid credentials (where applicable)
  - network timeout / RequestError handled
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

_apifb = load_service_module("apifb_test", "api_football_service.py")
_bzzoiro = load_service_module("bzzoiro_test", "bzzoiro_service.py")
_fdata = load_service_module("fdata_test", "football_data_service.py")
_sbomb = load_service_module("sbomb_test", "statsbomb_service.py")
_tsdb = load_service_module("tsdb_test", "thesportsdb_service.py")
_ofb = load_service_module("ofb_test", "openfootball_service.py")
_rfs = load_service_module("rfs_test", "roboflow_sports_service.py")

ApiFootballService = _apifb.ApiFootballService
BzzoiroService = _bzzoiro.BzzoiroService
FootballDataService = _fdata.FootballDataService
StatsBombService = _sbomb.StatsBombService
TheSportsDBService = _tsdb.TheSportsDBService
OpenFootballDataService = _ofb.OpenFootballDataService
RoboflowSportsService = _rfs.RoboflowSportsService


def _mock_response(status=200, json_data=None):
    m = MagicMock(spec=["status_code", "json"])
    m.status_code = status
    m.json = Mock(return_value=json_data or {})
    return m


@pytest.fixture
def mock_httpx():
    with patch("httpx.AsyncClient") as cls_mock:
        client_instance = AsyncMock()
        cls_mock.return_value = client_instance
        client_instance.__aenter__.return_value = client_instance
        yield client_instance


# ──────────────────────────────────────────────
# ApiFootballService
# ──────────────────────────────────────────────


class TestApiFootballService:
    @pytest.mark.asyncio
    async def test_successful_search_team(self, mock_httpx):
        svc = ApiFootballService(api_key="test_key")
        mock_httpx.get.return_value = _mock_response(
            200,
            {
                "response": [
                    {
                        "team": {"id": 1, "name": "KACM", "code": "KACM"},
                        "venue": {"name": "Stade", "city": "Marrakech", "capacity": 45000},
                    }
                ]
            },
        )
        result = await svc.search_team("KACM")
        assert len(result) == 1
        assert result[0]["name"] == "KACM"
        assert result[0]["venue_city"] == "Marrakech"

    @pytest.mark.asyncio
    async def test_empty_api_key_returns_empty(self, mock_httpx):
        svc = ApiFootballService(api_key=None)
        result = await svc.search_team("KACM")
        assert result == []

    @pytest.mark.asyncio
    async def test_403_marks_unavailable(self, mock_httpx):
        svc = ApiFootballService(api_key="bad")
        mock_httpx.get.return_value = _mock_response(403)
        data = await svc.check_status()
        assert data["available"] is False

    @pytest.mark.asyncio
    async def test_httpx_timeout_returns_empty(self, mock_httpx):
        svc = ApiFootballService(api_key="k")
        mock_httpx.get.side_effect = _apifb.httpx.RequestError("timeout")
        result = await svc.search_team("KACM")
        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty(self, mock_httpx):
        svc = ApiFootballService(api_key="k")
        resp = _mock_response(200)
        resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        mock_httpx.get.return_value = resp
        result = await svc.search_team("KACM")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_standings_parses_rows(self, mock_httpx):
        svc = ApiFootballService(api_key="k")
        mock_httpx.get.return_value = _mock_response(
            200,
            {
                "response": [
                    {
                        "league": {
                            "standings": [
                                [
                                    {
                                        "rank": 1,
                                        "team": {"id": 1, "name": "A", "logo": None},
                                        "points": 10,
                                        "goalsDiff": 5,
                                        "all": {
                                            "played": 4,
                                            "win": 3,
                                            "draw": 1,
                                            "lose": 0,
                                            "goals": {"for": 8, "against": 3},
                                        },
                                        "form": "WWWD",
                                    }
                                ]
                            ]
                        }
                    }
                ]
            },
        )
        standings = await svc.get_standings(200, 2024)
        assert len(standings) == 1
        assert standings[0]["rank"] == 1
        assert standings[0]["points"] == 10

    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self, mock_httpx):
        svc = ApiFootballService(api_key="k")
        await svc._ensure_client()
        assert svc._client is not None
        await svc.shutdown()
        assert svc._client is None


# ──────────────────────────────────────────────
# BzzoiroService
# ──────────────────────────────────────────────


class TestBzzoiroService:
    @pytest.mark.asyncio
    async def test_successful_team_search(self, mock_httpx):
        svc = BzzoiroService(api_key="tk")
        mock_httpx.get.return_value = _mock_response(
            200, {"results": [{"id": 1, "name": "Wydad", "country": "Morocco", "logo": "logo.png"}]}
        )
        result = await svc.search_team("Wydad")
        assert len(result) == 1
        assert result[0]["name"] == "Wydad"
        assert result[0]["country"] == "Morocco"

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self, mock_httpx):
        svc = BzzoiroService(api_key=None)
        result = await svc.search_team("Wydad")
        assert result == []

    @pytest.mark.asyncio
    async def test_500_returns_empty(self, mock_httpx):
        svc = BzzoiroService(api_key="k")
        mock_httpx.get.return_value = _mock_response(500)
        result = await svc.get_live_events()
        assert result == []

    @pytest.mark.asyncio
    async def test_httpx_error_returns_empty(self, mock_httpx):
        svc = BzzoiroService(api_key="k")
        mock_httpx.get.side_effect = _bzzoiro.httpx.RequestError("conn err")
        result = await svc.get_live_events()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_standings_parses(self, mock_httpx):
        svc = BzzoiroService(api_key="k")
        mock_httpx.get.return_value = _mock_response(
            200,
            {
                "standings": [
                    {
                        "position": 1,
                        "team_id": 1,
                        "team_name": "A",
                        "played": 3,
                        "wins": 2,
                        "draws": 1,
                        "losses": 0,
                        "goals_for": 5,
                        "goals_against": 2,
                        "goal_diff": 3,
                        "points": 7,
                    }
                ]
            },
        )
        rows = await svc.get_standings(1)
        assert len(rows) == 1
        assert rows[0]["points"] == 7

    @pytest.mark.asyncio
    async def test_get_leagues_parses(self, mock_httpx):
        svc = BzzoiroService(api_key="k")
        mock_httpx.get.return_value = _mock_response(
            200, {"results": [{"id": 1, "name": "Botola", "country": "Morocco", "is_active": True}]}
        )
        leagues = await svc.get_leagues()
        assert len(leagues) == 1
        assert leagues[0]["name"] == "Botola"

    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self, mock_httpx):
        svc = BzzoiroService(api_key="k")
        await svc._ensure_client()
        assert svc._client is not None
        await svc.shutdown()
        assert svc._client is None


# ──────────────────────────────────────────────
# FootballDataService
# ──────────────────────────────────────────────


class TestFootballDataService:
    @pytest.mark.asyncio
    async def test_successful_check_status(self, mock_httpx):
        svc = FootballDataService(api_key="k")
        mock_httpx.get.return_value = _mock_response(
            200, {"count": 2, "competitions": [{"id": 1}, {"id": 2}]}
        )
        data = await svc.check_status()
        assert data["available"] is True
        assert data["competitions_count"] == 2

    @pytest.mark.asyncio
    async def test_no_api_key_still_uses_env_value(self, mock_httpx):
        svc = FootballDataService(api_key=None)
        mock_httpx.get.return_value = _mock_response(200, {"count": 0, "competitions": []})
        data = await svc.check_status()
        assert data["available"] is True

    @pytest.mark.asyncio
    async def test_429_retries_then_succeeds(self, mock_httpx):
        svc = FootballDataService(api_key="k")
        resp_429 = _mock_response(429)
        resp_ok = _mock_response(200, {"competitions": [{"id": 1}]})
        mock_httpx.get.side_effect = [resp_429, resp_ok]
        result = await svc.check_status()
        assert result["available"] is True

    @pytest.mark.asyncio
    async def test_403_marks_unavailable(self, mock_httpx):
        svc = FootballDataService(api_key="bad")
        mock_httpx.get.return_value = _mock_response(403)
        data = await svc.check_status()
        assert data["available"] is False

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self, mock_httpx):
        svc = FootballDataService(api_key="k")
        mock_httpx.get.side_effect = _fdata.httpx.TimeoutException("timeout")
        result = await svc.get_competitions()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_team_returns_parsed(self, mock_httpx):
        svc = FootballDataService(api_key="k")
        mock_httpx.get.return_value = _mock_response(
            200, {"id": 1, "name": "KACM", "squad": [{"id": 10, "name": "P1", "position": "FW"}]}
        )
        team = await svc.get_team(1)
        assert team is not None
        assert team["name"] == "KACM"

    @pytest.mark.asyncio
    async def test_shutdown_closes_client(self, mock_httpx):
        svc = FootballDataService(api_key="k")
        await svc._ensure_client()
        assert svc._client is not None
        await svc.shutdown()
        assert svc._client is None


# ──────────────────────────────────────────────
# StatsBombService
# ──────────────────────────────────────────────


class TestStatsBombService:
    @pytest.mark.asyncio
    async def test_successful_competitions(self, mock_httpx):
        svc = StatsBombService()
        mock_httpx.get.return_value = _mock_response(
            200,
            [
                {
                    "competition_id": 1,
                    "season_id": 2,
                    "competition_name": "PL",
                    "country_name": "England",
                    "season_name": "2024",
                }
            ],
        )
        comps = await svc.get_competitions()
        assert len(comps) == 1
        assert comps[0].competition_name == "PL"

    @pytest.mark.asyncio
    async def test_non_200_returns_empty_list(self, mock_httpx):
        svc = StatsBombService()
        mock_httpx.get.return_value = _mock_response(404)
        comps = await svc.get_competitions()
        assert comps == []

    @pytest.mark.asyncio
    async def test_request_error_returns_empty(self, mock_httpx):
        svc = StatsBombService()
        mock_httpx.get.side_effect = _sbomb.httpx.RequestError("fail")
        comps = await svc.get_competitions()
        assert comps == []

    @pytest.mark.asyncio
    async def test_malformed_json_propagates(self, mock_httpx):
        svc = StatsBombService()
        resp = _mock_response(200)
        resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        mock_httpx.get.return_value = resp
        with pytest.raises(json.JSONDecodeError):
            await svc.get_competitions()

    @pytest.mark.asyncio
    async def test_get_events_returns_parsed(self, mock_httpx):
        svc = StatsBombService()
        mock_httpx.get.return_value = _mock_response(
            200,
            [
                {
                    "id": "e1",
                    "type": {"name": "Pass"},
                    "period": 1,
                    "minute": 10,
                    "second": 0,
                    "timestamp": "00:00:10",
                    "team": {"name": "Home"},
                    "player": {"name": "P1"},
                    "location": [1.0, 2.0],
                    "outcome": "Success",
                },
                {
                    "id": "e2",
                    "type": {"name": "Shot"},
                    "period": 2,
                    "minute": 50,
                    "second": 10,
                    "timestamp": "00:50:10",
                    "team": {"name": "Away"},
                    "player": {"name": "P2"},
                    "location": [10.0, 20.0],
                    "shot": {"statsbomb_xg": 0.45},
                },
            ],
        )
        events = await svc.get_events(123)
        assert len(events) == 2
        assert events[0].event_type == "Pass"
        assert events[1].xg == 0.45

    @pytest.mark.asyncio
    async def test_close_method(self, mock_httpx):
        svc = StatsBombService()
        svc._client = mock_httpx
        await svc.close()
        mock_httpx.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_team_matches(self, mock_httpx):
        svc = StatsBombService()
        comp_resp = _mock_response(
            200,
            [
                {
                    "competition_id": 1,
                    "season_id": 2,
                    "competition_name": "PL",
                    "country_name": "England",
                    "season_name": "2024",
                }
            ],
        )
        match_resp = _mock_response(
            200,
            [
                {
                    "match_id": 1,
                    "competition_id": 1,
                    "season_id": 2,
                    "home_team": {"home_team_name": "Arsenal"},
                    "away_team": {"away_team_name": "Chelsea"},
                    "home_score": 2,
                    "away_score": 1,
                    "match_date": "2024-08-16",
                    "competition_stage": "regular",
                    "stadium": {},
                    "referee": {},
                }
            ],
        )
        mock_httpx.get.side_effect = [comp_resp, match_resp]
        matches = await svc.search_team_matches("Arsenal")
        assert len(matches) >= 1


# ──────────────────────────────────────────────
# TheSportsDBService
# ──────────────────────────────────────────────


class TestTheSportsDBService:
    @pytest.mark.asyncio
    async def test_successful_team_search(self, mock_httpx):
        svc = TheSportsDBService(api_key="3")
        mock_httpx.get.return_value = _mock_response(
            200, {"teams": [{"idTeam": "1", "strTeam": "KACM", "strLeague": "Botola"}]}
        )
        results = await svc.search_teams("KACM")
        assert len(results) == 1
        assert results[0].name == "KACM"

    @pytest.mark.asyncio
    async def test_empty_teams_key_returns_empty(self, mock_httpx):
        svc = TheSportsDBService(api_key="3")
        mock_httpx.get.return_value = _mock_response(200, {})
        results = await svc.search_teams("KACM")
        assert results == []

    @pytest.mark.asyncio
    async def test_500_returns_none(self, mock_httpx):
        svc = TheSportsDBService(api_key="3")
        mock_httpx.get.return_value = _mock_response(500)
        result = await svc.get_team("1")
        assert result is None

    @pytest.mark.asyncio
    async def test_request_error_returns_empty(self, mock_httpx):
        svc = TheSportsDBService(api_key="3")
        mock_httpx.get.side_effect = _tsdb.httpx.RequestError("fail")
        results = await svc.search_teams("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_standings_empty_response(self, mock_httpx):
        svc = TheSportsDBService(api_key="3")
        mock_httpx.get.return_value = _mock_response(200, {})
        rows = await svc.get_standings("123", "2024")
        assert rows == []

    @pytest.mark.asyncio
    async def test_get_league_returns_none_on_missing(self, mock_httpx):
        svc = TheSportsDBService(api_key="3")
        mock_httpx.get.return_value = _mock_response(200, {})
        league = await svc.get_league("123")
        assert league is None

    @pytest.mark.asyncio
    async def test_get_all_leagues_empty(self, mock_httpx):
        svc = TheSportsDBService(api_key="3")
        mock_httpx.get.return_value = _mock_response(200, {})
        leagues = await svc.get_all_leagues()
        assert leagues == []


# ──────────────────────────────────────────────
# OpenFootballDataService
# ──────────────────────────────────────────────


class TestOpenFootballDataService:
    @pytest.mark.asyncio
    async def test_successful_matches(self, mock_httpx):
        svc = OpenFootballDataService()
        mock_httpx.get.return_value = _mock_response(
            200,
            {
                "matches": [
                    {
                        "round": "1",
                        "date": "2024-08-16",
                        "team1": "Arsenal",
                        "team2": "Wolves",
                        "score": {"ft": [2, 1]},
                    }
                ]
            },
        )
        matches = await svc.get_matches("en.1", "2024-25")
        assert len(matches) == 1
        assert matches[0].home_team == "Arsenal"
        assert matches[0].home_score == 2

    @pytest.mark.asyncio
    async def test_unknown_league_returns_empty(self, mock_httpx):
        svc = OpenFootballDataService()
        matches = await svc.get_matches("xx.1", "2024-25")
        assert matches == []

    @pytest.mark.asyncio
    async def test_no_matches_key_returns_empty(self, mock_httpx):
        svc = OpenFootballDataService()
        mock_httpx.get.return_value = _mock_response(200, {})
        matches = await svc.get_matches("en.1", "2024-25")
        assert matches == []

    @pytest.mark.asyncio
    async def test_404_returns_empty(self, mock_httpx):
        svc = OpenFootballDataService()
        mock_httpx.get.return_value = _mock_response(404)
        matches = await svc.get_matches("en.1", "2024-25")
        assert matches == []

    @pytest.mark.asyncio
    async def test_httpx_error_caught(self, mock_httpx):
        svc = OpenFootballDataService()
        mock_httpx.get.side_effect = _ofb.httpx.RequestError("fail")
        matches = await svc.get_matches("en.1", "2024-25")
        assert matches == []

    @pytest.mark.asyncio
    async def test_worldcup_matches(self, mock_httpx):
        svc = OpenFootballDataService()
        mock_httpx.get.return_value = _mock_response(
            200,
            {
                "matches": [
                    {
                        "round": "Final",
                        "date": "2026-07-19",
                        "team1": "A",
                        "team2": "B",
                        "score": {"ft": [3, 1]},
                    }
                ]
            },
        )
        matches = await svc.get_worldcup_matches(2026)
        assert len(matches) == 1
        assert matches[0].competition == "worldcup"

    @pytest.mark.asyncio
    async def test_search_team_matches(self, mock_httpx):
        svc = OpenFootballDataService()
        mock_httpx.get.return_value = _mock_response(
            200,
            {
                "matches": [
                    {
                        "round": "1",
                        "date": "2024-08-16",
                        "team1": "Liverpool",
                        "team2": "Chelsea",
                        "score": {"ft": [3, 0]},
                    }
                ]
            },
        )
        results = await svc.search_team_matches("Liverpool", "en.1")
        assert len(results) >= 1
        assert results[0].home_team == "Liverpool"


# ──────────────────────────────────────────────
# RoboflowSportsService
# ──────────────────────────────────────────────


class TestRoboflowSportsService:
    def test_not_available_when_package_missing(self):
        svc = RoboflowSportsService()
        assert svc.available is False
        assert svc.draw_pitch() is None
        assert svc.draw_points_on_pitch(None) is None
        assert svc.create_ball_annotator(5) is None
        assert svc.create_ball_tracker() is None
        assert svc.create_team_classifier() is None
        assert svc.create_view_transformer(None, None) is None

    def test_properties_graceful(self):
        svc = RoboflowSportsService()
        assert svc.has_team_classifier is False
        assert svc.has_view_transformer is False

    def test_draw_voronoi_returns_none_when_unavailable(self):
        svc = RoboflowSportsService()
        import numpy as np

        result = svc.draw_voronoi(
            team_1_xy=np.array([[1.0, 2.0]]),
            team_2_xy=np.array([[3.0, 4.0]]),
        )
        assert result is None

    def test_draw_paths_on_pitch_returns_none_when_unavailable(self):
        svc = RoboflowSportsService()
        import numpy as np

        result = svc.draw_paths_on_pitch([np.array([[1.0, 2.0]])])
        assert result is None

    def test_draw_points_on_pitch_returns_none_when_unavailable(self):
        svc = RoboflowSportsService()
        import numpy as np

        result = svc.draw_points_on_pitch(xy=np.array([[1.0, 2.0]]))
        assert result is None

    def test_create_ball_annotator_returns_none(self):
        svc = RoboflowSportsService()
        assert svc.create_ball_annotator(radius=5) is None

    def test_create_ball_tracker_returns_none(self):
        svc = RoboflowSportsService()
        assert svc.create_ball_tracker() is None

    def test_create_team_classifier_returns_none(self):
        svc = RoboflowSportsService()
        assert svc.create_team_classifier() is None

    def test_create_view_transformer_returns_none(self):
        svc = RoboflowSportsService()
        import numpy as np

        assert svc.create_view_transformer(source=np.eye(3), target=np.eye(3)) is None
