"""Tests for all 7 external API services.

Covers: ApiFootball, Bzzoiro, FootballData, StatsBomb, TheSportsDB,
OpenFootballData, and WeatherService (Open-Meteo API).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()

_apifb = load_service_module("apifb_test", "api_football_service.py")
_bzzoiro = load_service_module("bzzoiro_test", "bzzoiro_service.py")
_fdata = load_service_module("fdata_test", "football_data_service.py")
_sbomb = load_service_module("sbomb_test", "statsbomb_service.py")
_tsdb = load_service_module("tsdb_test", "thesportsdb_service.py")
_ofb = load_service_module("ofb_test", "openfootball_service.py")
_wtr = load_service_module("wtr_test", "weather_service.py")

ApiFootballService = _apifb.ApiFootballService
BzzoiroService = _bzzoiro.BzzoiroService
FootballDataService = _fdata.FootballDataService
StatsBombService = _sbomb.StatsBombService
TheSportsDBService = _tsdb.TheSportsDBService
OpenFootballDataService = _ofb.OpenFootballDataService
WeatherService = _wtr.WeatherService


def _mock_response(status=200, json_data=None):
    """Build a Mock that looks like httpx.Response.

    NOTE: httpx.Response.json() is synchronous, so we use a regular Mock.
    """
    m = MagicMock(spec=["status_code", "json"])
    m.status_code = status
    m.json = Mock(return_value=json_data or {})
    return m


@pytest.fixture
def mock_httpx():
    """Mock httpx.AsyncClient so no real network calls are made.

    Yields the client instance returned by httpx.AsyncClient().
    """
    with patch("httpx.AsyncClient") as cls_mock:
        client_instance = AsyncMock()
        cls_mock.return_value = client_instance
        client_instance.__aenter__.return_value = client_instance
        yield client_instance


# ──────────────────────────────────────────────
# ApiFootballService
# ──────────────────────────────────────────────


class TestApiFootballService:
    """ApiFootballService – api-sports.io v3."""

    @pytest.mark.asyncio
    async def test_successful_request(self, mock_httpx):
        svc = ApiFootballService(api_key="test_key")
        mock_httpx.get.return_value = _mock_response(
            200, {"response": [{"team": {"id": 1, "name": "KACM", "code": "KACM"}}]}
        )
        result = await svc.search_team("KACM")
        assert len(result) == 1
        assert result[0]["name"] == "KACM"
        assert svc._available is True

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_none(self, mock_httpx):
        svc = ApiFootballService(api_key=None)
        result = await svc.search_team("KACM")
        assert result == []

    @pytest.mark.asyncio
    async def test_429_rate_limit_retries(self, mock_httpx):
        svc = ApiFootballService(api_key="k")
        svc._requests_today = 0
        resp_429 = _mock_response(429)
        resp_ok = _mock_response(200, {"response": [{"team": {"id": 1, "name": "OK"}}]})
        mock_httpx.get.side_effect = [resp_429, resp_ok]
        result = await svc.search_team("OK")
        assert len(result) == 1
        assert result[0]["name"] == "OK"

    @pytest.mark.asyncio
    async def test_non_200_returns_none(self, mock_httpx):
        svc = ApiFootballService(api_key="k")
        mock_httpx.get.return_value = _mock_response(403)
        data = await svc.check_status()
        assert data["available"] is False

    @pytest.mark.asyncio
    async def test_httpx_timeout_caught(self, mock_httpx):
        svc = ApiFootballService(api_key="k")
        mock_httpx.get.side_effect = _apifb.httpx.RequestError("timeout")
        result = await svc.search_team("KACM")
        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self, mock_httpx):
        svc = ApiFootballService(api_key="k")
        resp = _mock_response(200)
        resp.json.side_effect = json.JSONDecodeError("bad json", "", 0)
        mock_httpx.get.return_value = resp
        result = await svc.search_team("KACM")
        assert result == []

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
    """BzzoiroService – sports.bzzoiro.com v2."""

    @pytest.mark.asyncio
    async def test_successful_request(self, mock_httpx):
        svc = BzzoiroService(api_key="tk")
        mock_httpx.get.return_value = _mock_response(
            200, {"results": [{"id": 1, "name": "KACM", "country": "Morocco"}]}
        )
        result = await svc.search_team("KACM")
        assert len(result) == 1
        assert result[0]["name"] == "KACM"

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_none(self, mock_httpx):
        svc = BzzoiroService(api_key=None)
        result = await svc.search_team("KACM")
        assert result == []

    @pytest.mark.asyncio
    async def test_non_200_returns_none(self, mock_httpx):
        svc = BzzoiroService(api_key="k")
        mock_httpx.get.return_value = _mock_response(500)
        result = await svc.search_team("KACM")
        assert result == []

    @pytest.mark.asyncio
    async def test_httpx_request_error_caught(self, mock_httpx):
        svc = BzzoiroService(api_key="k")
        mock_httpx.get.side_effect = _bzzoiro.httpx.RequestError("conn err")
        result = await svc.search_team("KACM")
        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self, mock_httpx):
        svc = BzzoiroService(api_key="k")
        resp = _mock_response(200)
        resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        mock_httpx.get.return_value = resp
        result = await svc.search_team("KACM")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_predictions_returns_raw_data(self, mock_httpx):
        svc = BzzoiroService(api_key="k")
        mock_httpx.get.return_value = _mock_response(200, {"prediction": "home_win"})
        result = await svc.get_predictions(42)
        assert result == {"prediction": "home_win"}

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
    """FootballDataService – football-data.org v4."""

    @pytest.mark.asyncio
    async def test_successful_request(self, mock_httpx):
        svc = FootballDataService(api_key="k")
        mock_httpx.get.return_value = _mock_response(
            200, {"count": 1, "competitions": [{"id": 1, "name": "PL"}]}
        )
        result = await svc.check_status()
        assert result["available"] is True
        assert result["competitions_count"] == 1

    @pytest.mark.asyncio
    async def test_429_rate_limit_retries(self, mock_httpx):
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
    async def test_httpx_timeout_caught(self, mock_httpx):
        svc = FootballDataService(api_key="k")
        mock_httpx.get.side_effect = _fdata.httpx.TimeoutException("timeout")
        result = await svc.get_competitions()
        assert result == []

    @pytest.mark.asyncio
    async def test_non_200_returns_none(self, mock_httpx):
        svc = FootballDataService(api_key="k")
        mock_httpx.get.return_value = _mock_response(500)
        result = await svc.get_competitions()
        assert result == []

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
    """StatsBombService – raw GitHub open data."""

    @pytest.mark.asyncio
    async def test_successful_request(self, mock_httpx):
        svc = StatsBombService()
        mock_httpx.get.return_value = _mock_response(
            200, [{"competition_id": 1, "season_id": 2, "competition_name": "PL"}]
        )
        comps = await svc.get_competitions()
        assert len(comps) == 1
        assert comps[0].competition_name == "PL"
        assert svc.available is True

    @pytest.mark.asyncio
    async def test_non_200_returns_empty(self, mock_httpx):
        svc = StatsBombService()
        mock_httpx.get.return_value = _mock_response(404)
        comps = await svc.get_competitions()
        assert comps == []

    @pytest.mark.asyncio
    async def test_httpx_request_error_caught(self, mock_httpx):
        svc = StatsBombService()
        mock_httpx.get.side_effect = _sbomb.httpx.RequestError("fail")
        comps = await svc.get_competitions()
        assert comps == []

    @pytest.mark.asyncio
    async def test_malformed_json_raises(self, mock_httpx):
        """StatsBombService only catches httpx.RequestError, so JSONDecodeError propagates."""
        svc = StatsBombService()
        resp = _mock_response(200)
        resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        mock_httpx.get.return_value = resp
        with pytest.raises(json.JSONDecodeError):
            await svc.get_competitions()

    @pytest.mark.asyncio
    async def test_get_shots_filters_by_type(self, mock_httpx):
        svc = StatsBombService()
        mock_httpx.get.return_value = _mock_response(
            200,
            [
                {"id": "1", "type": {"name": "Shot"}, "minute": 10},
                {"id": "2", "type": {"name": "Pass"}, "minute": 11},
                {"id": "3", "type": {"name": "Shot"}, "minute": 12},
            ],
        )
        shots = await svc.get_shots(123)
        assert len(shots) == 2

    @pytest.mark.asyncio
    async def test_get_player_events_filters(self, mock_httpx):
        svc = StatsBombService()
        mock_httpx.get.return_value = _mock_response(
            200,
            [
                {"id": "1", "player": {"name": "Messi"}, "team": {"name": "ARG"}},
                {"id": "2", "player": {"name": "Ronaldo"}, "team": {"name": "POR"}},
            ],
        )
        events = await svc.get_player_events(1, "Messi")
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_close_method_handles_none(self, mock_httpx):
        svc = StatsBombService()
        svc._client = mock_httpx
        await svc.close()
        mock_httpx.aclose.assert_awaited_once()


# ──────────────────────────────────────────────
# TheSportsDBService
# ──────────────────────────────────────────────


class TestTheSportsDBService:
    """TheSportsDBService – thesportsdb.com v1."""

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
    async def test_no_teams_key_returns_empty(self, mock_httpx):
        svc = TheSportsDBService(api_key="3")
        mock_httpx.get.return_value = _mock_response(200, {})
        results = await svc.search_teams("KACM")
        assert results == []

    @pytest.mark.asyncio
    async def test_non_200_returns_none(self, mock_httpx):
        svc = TheSportsDBService(api_key="3")
        mock_httpx.get.return_value = _mock_response(500)
        result = await svc.get_team("1")
        assert result is None

    @pytest.mark.asyncio
    async def test_httpx_request_error_caught(self, mock_httpx):
        svc = TheSportsDBService(api_key="3")
        mock_httpx.get.side_effect = _tsdb.httpx.RequestError("fail")
        result = await svc.search_teams("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_standings_table(self, mock_httpx):
        svc = TheSportsDBService(api_key="3")
        mock_httpx.get.return_value = _mock_response(200, {})
        rows = await svc.get_standings("123", "2024")
        assert rows == []

    @pytest.mark.asyncio
    async def test_close_method(self, mock_httpx):
        svc = TheSportsDBService(api_key="test123")
        svc._client = mock_httpx
        await svc.close()
        mock_httpx.aclose.assert_awaited_once()


# ──────────────────────────────────────────────
# OpenFootballDataService
# ──────────────────────────────────────────────


class TestOpenFootballDataService:
    """OpenFootballDataService – openfootball repos."""

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
    async def test_non_200_returns_none(self, mock_httpx):
        svc = OpenFootballDataService()
        mock_httpx.get.return_value = _mock_response(404)
        matches = await svc.get_matches("en.1", "2024-25")
        assert matches == []

    @pytest.mark.asyncio
    async def test_httpx_request_error_caught(self, mock_httpx):
        svc = OpenFootballDataService()
        mock_httpx.get.side_effect = _ofb.httpx.RequestError("fail")
        matches = await svc.get_matches("en.1", "2024-25")
        assert matches == []

    @pytest.mark.asyncio
    async def test_worldcup_matches(self, mock_httpx):
        svc = OpenFootballDataService()
        mock_httpx.get.return_value = _mock_response(
            200, {"matches": [{"round": "Final", "date": "2026-07-19", "team1": "A", "team2": "B"}]}
        )
        matches = await svc.get_worldcup_matches(2026)
        assert len(matches) == 1
        assert matches[0].competition == "worldcup"

    @pytest.mark.asyncio
    async def test_close_method(self, mock_httpx):
        svc = OpenFootballDataService()
        svc._client = mock_httpx
        await svc.close()
        mock_httpx.aclose.assert_awaited_once()

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

    def test_get_competitions_returns_static_list(self):
        svc = OpenFootballDataService()
        comps = svc.get_competitions()
        assert len(comps) == 6
        assert comps[0].id == "en.1"


# ──────────────────────────────────────────────
# WeatherService (API portion)
# ──────────────────────────────────────────────


class TestWeatherServiceAPI:
    """WeatherService – Open-Meteo API calls."""

    @pytest.mark.asyncio
    async def test_fetch_conditions_success(self, mock_httpx):
        svc = WeatherService()
        mock_httpx.get.return_value = _mock_response(
            200,
            {
                "hourly": {
                    "time": ["2024-01-15T15:00"],
                    "temperature_2m": [12.5],
                    "precipitation": [0.0],
                    "wind_speed_10m": [5.0],
                    "wind_direction_10m": [180.0],
                    "relative_humidity_2m": [60.0],
                    "cloud_cover": [20.0],
                    "is_day": [1],
                }
            },
        )
        cond = await svc.fetch_conditions(31.6, -8.0, "2024-01-15")
        assert cond is not None
        assert cond.temperature_c == 12.5
        assert cond.conditions == "clear"

    @pytest.mark.asyncio
    async def test_fetch_conditions_non_200(self, mock_httpx):
        svc = WeatherService()
        mock_httpx.get.return_value = _mock_response(500)
        cond = await svc.fetch_conditions(31.6, -8.0, "2024-01-15")
        assert cond is None

    @pytest.mark.asyncio
    async def test_fetch_conditions_missing_hourly(self, mock_httpx):
        svc = WeatherService()
        mock_httpx.get.return_value = _mock_response(200, {})
        cond = await svc.fetch_conditions(31.6, -8.0, "2024-01-15")
        assert cond is None

    @pytest.mark.asyncio
    async def test_fetch_conditions_empty_times(self, mock_httpx):
        svc = WeatherService()
        mock_httpx.get.return_value = _mock_response(
            200, {"hourly": {"time": [], "temperature_2m": []}}
        )
        cond = await svc.fetch_conditions(31.6, -8.0, "2024-01-15")
        assert cond is None

    @pytest.mark.asyncio
    async def test_fetch_conditions_httpx_error(self, mock_httpx):
        svc = WeatherService()
        mock_httpx.get.side_effect = _wtr.httpx.RequestError("fail")
        cond = await svc.fetch_conditions(31.6, -8.0, "2024-01-15")
        assert cond is None

    @pytest.mark.asyncio
    async def test_fetch_forecast(self, mock_httpx):
        svc = WeatherService()
        mock_httpx.get.return_value = _mock_response(
            200,
            {
                "hourly": {
                    "time": ["2024-06-15T15:00"],
                    "temperature_2m": [28.0],
                    "precipitation": [0.0],
                    "wind_speed_10m": [10.0],
                    "wind_direction_10m": [90.0],
                    "relative_humidity_2m": [40.0],
                    "cloud_cover": [10.0],
                    "is_day": [1],
                }
            },
        )
        cond = await svc.fetch_conditions(40.4, -3.7, "2024-06-15", is_forecast=True)
        assert cond is not None
        assert cond.temperature_c == 28.0

    @pytest.mark.asyncio
    async def test_close_method(self, mock_httpx):
        svc = WeatherService()
        svc._owns_client = True
        svc._client = mock_httpx
        await svc.close()
        mock_httpx.aclose.assert_awaited_once()
