"""Async service for sports.bzzoiro.com API v2.

Free tier: unlimited requests, 65 leagues, live scores, predictions,
shotmaps, odds, Botola Pro (Morocco) coverage.
Silent degradation: returns empty results on failure.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

CACHE_TTL_SHORT = 300
CACHE_TTL_MEDIUM = 3600
CACHE_TTL_LONG = 86400

BASE_URL = "https://sports.bzzoiro.com/api/v2"


class BzzoiroService:
    """Async wrapper for sports.bzzoiro.com API v2.

    Provides team search, live matches, standings, fixtures, predictions,
    shotmaps, and squad data. All methods return empty results on failure.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, tuple[float, Any]] = {}
        self._available = False

    async def _ensure_client(self) -> None:
        if self._client is None:
            headers = {"Authorization": f"Token {self.api_key}"} if self.api_key else {}
            self._client = httpx.AsyncClient(base_url=BASE_URL, headers=headers, timeout=15.0)

    def _cache_get(self, key: str) -> Any | None:
        if key in self._cache:
            expires, data = self._cache[key]
            if time.monotonic() < expires:
                return data
            del self._cache[key]
        return None

    def _cache_set(self, key: str, data: Any, ttl: int) -> None:
        self._cache[key] = (time.monotonic() + ttl, data)

    async def _get(self, path: str, ttl: int = CACHE_TTL_SHORT) -> dict | list | None:
        cache_key = f"bzzoiro:{path}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self.api_key:
            return None

        await self._ensure_client()
        assert self._client is not None
        try:
            r = await self._client.get(path)
            if r.status_code == 200:
                data = r.json()
                self._cache_set(cache_key, data, ttl)
                self._available = True
                return data
            else:
                logger.warning(f"Bzzoiro API error {r.status_code}: {path}")
                return None
        except httpx.RequestError as e:
            logger.warning(f"Bzzoiro API request failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"Bzzoiro API unexpected error: {e}")
            return None

    async def check_status(self) -> dict:
        """Check if API is available by fetching live events."""
        data = await self._get("events/live/", ttl=CACHE_TTL_SHORT)
        if data is None:
            return {"available": False, "error": "API not available"}
        count = len(data.get("events", [])) if isinstance(data, dict) else 0
        return {"available": True, "live_matches": count}

    async def search_team(self, query: str) -> list[dict]:
        """Search teams by name. Returns list of {id, name, country, logo}."""

        data = await self._get(f"teams/search/?q={query}", ttl=CACHE_TTL_LONG)
        if data is None:
            return []
        results = data.get("results", data) if isinstance(data, dict) else data
        teams = []
        for t in results or []:
            teams.append(
                {
                    "id": t.get("id"),
                    "name": t.get("name", ""),
                    "country": t.get("country", ""),
                    "logo": t.get("logo"),
                }
            )
        return teams

    async def get_team_matches(
        self, team_id: int, date_from: str | None = None, date_to: str | None = None
    ) -> list[dict]:
        """Get matches for a team."""
        path = f"events/?team_ids={team_id}&limit=20"
        if date_from:
            path += f"&date_from={date_from}"
        if date_to:
            path += f"&date_to={date_to}"
        data = await self._get(path, ttl=CACHE_TTL_SHORT)
        if data is None:
            return []
        events = data.get("events", data) if isinstance(data, dict) else data
        matches = []
        for m in events or []:
            matches.append(
                {
                    "id": m.get("id"),
                    "home_team": m.get("home_team", ""),
                    "away_team": m.get("away_team", ""),
                    "home_team_id": m.get("home_team_id"),
                    "away_team_id": m.get("away_team_id"),
                    "home_score": m.get("home_score"),
                    "away_score": m.get("away_score"),
                    "status": m.get("status", ""),
                    "event_date": m.get("event_date", ""),
                    "league_name": m.get("league_name", ""),
                    "league_id": m.get("league_id"),
                    "period": m.get("period"),
                    "current_minute": m.get("current_minute"),
                }
            )
        return matches

    async def get_team_squad(self, team_id: int) -> list[dict]:
        """Get squad for a team. Returns list of {id, name, position, jersey_number}."""
        data = await self._get(f"teams/{team_id}/squad/", ttl=CACHE_TTL_LONG)
        if data is None:
            return []
        squad = data.get("players", data) if isinstance(data, dict) else data
        players = []
        for p in squad or []:
            players.append(
                {
                    "id": p.get("id"),
                    "name": p.get("name", ""),
                    "position": p.get("position", ""),
                    "jersey_number": p.get("jersey_number"),
                    "nationality": p.get("nationality"),
                    "date_of_birth": p.get("date_of_birth"),
                }
            )
        return players

    async def get_match_detail(self, event_id: int) -> dict | None:
        """Get detailed match info including stats and incidents."""
        data = await self._get(f"events/{event_id}/", ttl=CACHE_TTL_SHORT)
        if data is None:
            return None
        event = data.get("event", data) if isinstance(data, dict) else data
        if not isinstance(event, dict):
            return None
        return {
            "id": event.get("id"),
            "home_team": event.get("home_team", ""),
            "away_team": event.get("away_team", ""),
            "home_score": event.get("home_score"),
            "away_score": event.get("away_score"),
            "home_score_ht": event.get("home_score_ht"),
            "away_score_ht": event.get("away_score_ht"),
            "status": event.get("status", ""),
            "period": event.get("period"),
            "current_minute": event.get("current_minute"),
            "event_date": event.get("event_date", ""),
            "league_name": event.get("league_name", ""),
        }

    async def get_standings(self, league_id: int) -> list[dict]:
        """Get standings table for a league."""
        data = await self._get(f"leagues/{league_id}/standings/", ttl=CACHE_TTL_MEDIUM)
        if data is None:
            return []
        standings = data.get("standings", data) if isinstance(data, dict) else data
        rows = []
        for s in standings or []:
            rows.append(
                {
                    "position": s.get("position"),
                    "team_id": s.get("team_id"),
                    "team_name": s.get("team_name", ""),
                    "played": s.get("played"),
                    "wins": s.get("wins"),
                    "draws": s.get("draws"),
                    "losses": s.get("losses"),
                    "goals_for": s.get("goals_for"),
                    "goals_against": s.get("goals_against"),
                    "goal_diff": s.get("goal_diff"),
                    "points": s.get("points"),
                }
            )
        return rows

    async def get_leagues(self) -> list[dict]:
        """Get list of available leagues."""
        data = await self._get("leagues/", ttl=CACHE_TTL_LONG)
        if data is None:
            return []
        results = data.get("results", data) if isinstance(data, dict) else data
        leagues = []
        for result in results or []:
            leagues.append(
                {
                    "id": result.get("id"),
                    "name": result.get("name", ""),
                    "country": result.get("country", ""),
                    "is_active": result.get("is_active", False),
                }
            )
        return leagues

    async def get_live_events(self) -> list[dict]:
        """Get currently live matches."""
        data = await self._get("events/live/", ttl=CACHE_TTL_SHORT)
        if data is None:
            return []
        events = data.get("events", data) if isinstance(data, dict) else data
        matches = []
        for m in events or []:
            matches.append(
                {
                    "id": m.get("id"),
                    "home_team": m.get("home_team", ""),
                    "away_team": m.get("away_team", ""),
                    "home_score": m.get("home_score"),
                    "away_score": m.get("away_score"),
                    "status": m.get("status", ""),
                    "league_name": m.get("league_name", ""),
                    "current_minute": m.get("current_minute"),
                    "period": m.get("period"),
                }
            )
        return matches

    async def get_predictions(self, event_id: int) -> dict | None:
        """Get AI predictions for a match (CatBoost)."""
        data = await self._get(f"predictions/{event_id}/", ttl=CACHE_TTL_MEDIUM)
        if isinstance(data, dict):
            return data
        return None

    async def get_match_stats(self, event_id: int) -> list[dict] | None:
        """Get per-shot xG stats for a match."""
        data = await self._get(f"events/{event_id}/stats/", ttl=CACHE_TTL_SHORT)
        if isinstance(data, list):
            return data
        return None

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
