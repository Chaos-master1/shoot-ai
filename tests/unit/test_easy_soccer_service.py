"""Tests for EasySoccerData wrapper service."""

from unittest.mock import MagicMock, patch

import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.easy_soccer_service import EasySoccerService


@pytest.fixture
def mock_client():
    service = EasySoccerService()
    fake = MagicMock()
    fake.get_events.return_value = [
        {
            "id": 1,
            "home_team": "A",
            "away_team": "B",
            "home_score": 1,
            "away_score": 0,
            "status": "live",
            "league_name": "PL",
            "current_minute": 30,
        }
    ]
    fake.get_event.return_value = {
        "id": 1,
        "home_team": "A",
        "away_team": "B",
        "home_score": 1,
        "away_score": 0,
        "status": "finished",
        "venue": "Stadium",
        "league_name": "PL",
        "event_date": "2024-01-01",
    }
    fake.get_match_incidents.return_value = [
        {
            "type": "goal",
            "team": "home",
            "player": "P1",
            "minute": 15,
            "score_home": 1,
            "score_away": 0,
        }
    ]
    fake.get_player.return_value = {
        "id": 1,
        "name": "Player1",
        "position": "FW",
        "jersey_number": 9,
        "nationality": "Country",
        "date_of_birth": "2000-01-01",
    }
    service._client = fake
    service._available = True
    return service


class TestEasySoccerService:
    def test_get_live_events(self, mock_client):
        events = mock_client.get_live_events()
        assert len(events) == 1
        assert events[0]["home_team"] == "A"

    def test_get_event(self, mock_client):
        ev = mock_client.get_event(1)
        assert ev is not None
        assert ev["home_team"] == "A"

    def test_get_event_not_found(self, mock_client):
        mock_client._client.get_event.return_value = None
        ev = mock_client.get_event(999)
        assert ev is None

    def test_get_match_incidents(self, mock_client):
        incidents = mock_client.get_match_incidents(1)
        assert len(incidents) == 1
        assert incidents[0]["type"] == "goal"

    def test_get_player(self, mock_client):
        p = mock_client.get_player(1)
        assert p is not None
        assert p["name"] == "Player1"

    def test_search_events(self, mock_client):
        events = mock_client.search_events("2024-01-01")
        assert len(events) == 1

    def test_check_available(self, mock_client):
        assert mock_client.check_available() is True

    def test_unavailable_no_client(self):
        # Mock esd so the real module (which creates a Playwright event loop) is never loaded
        with patch.dict("sys.modules", {"esd": MagicMock()}):
            import importlib

            import kawkab.services.easy_soccer_service as _ess_mod

            importlib.reload(_ess_mod)
            service = _ess_mod.EasySoccerService()
            assert service.get_live_events() == []
            # With esd mocked, SofascoreClient() raises ImportError at import time,
            # so _get_client returns None and check_available returns False.
            assert service.check_available() is False

    def test_exception_returns_empty(self, mock_client):
        mock_client._client.get_events.side_effect = Exception("fail")
        assert mock_client.get_live_events() == []

    def test_exception_get_event_returns_none(self, mock_client):
        mock_client._client.get_event.side_effect = Exception("fail")
        assert mock_client.get_event(1) is None
