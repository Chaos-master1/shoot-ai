"""Async service for API-Football (api-sports.io) v3.

Free tier: 100 requests/day, historical seasons 2022-2024.
Covers Botola Pro (league 200), Kawkab Marrakech (team 971), and
1,233+ leagues. Squad data, standings, predictions, fixtures.
Silent degradation: returns empty results on failure.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

BASE_URL = "https://v3.football.api-sports.io"
CACHE_TTL_SHORT = 300
CACHE_TTL_MEDIUM = 3600
CACHE_TTL_LONG = 86400
DAILY_LIMIT = 100


class ApiFootballService:
    """Async wrapper for API-Football (api-sports.io) v3.

    Provides team/squad data, standings, fixtures, predictions.
    All methods return empty results on failure (silent degrade).
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, tuple[float, Any]] = {}
        self._requests_today = 0
        self._day_start = time.monotonic()
        self._available = False

    async def _ensure_client(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=15.0)

    async def _rate_limit(self) -> None:
        now = time.monotonic()
        if now - self._day_start > 86400:
            self._requests_today = 0
            self._day_start = now
        while self._requests_today >= DAILY_LIMIT:
            await asyncio.sleep(10)
            now = time.monotonic()
            if now - self._day_start > 86400:
                self._requests_today = 0
                self._day_start = now
                break
        self._requests_today += 1

    async def _request(self, path: str, ttl: int = CACHE_TTL_SHORT) -> dict | None:
        ckey = f"apifb:{path}"
        if ckey in self._cache:
            expires, data = self._cache[ckey]
            if time.monotonic() < expires:
                return data

        if not self.api_key:
            return None

        await self._rate_limit()
        try:
            await self._ensure_client()
            assert self._client is not None
            headers = {"x-apisports-key": self.api_key}
            resp = await self._client.get(path, headers=headers)
            if resp.status_code == 429:
                logger.warning("API-Football rate limited, sleeping 30s")
                await asyncio.sleep(30)
                return await self._request(path, ttl)
            if resp.status_code != 200:
                return None
            data = resp.json()
            self._cache[ckey] = (time.monotonic() + ttl, data)
            self._available = True
            return data
        except httpx.RequestError as e:
            logger.warning(f"API-Football request failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"API-Football error: {e}")
            return None

    async def check_status(self) -> dict:
        """Validate API key and return availability info."""
        data = await self._request("/status", ttl=CACHE_TTL_SHORT)
        if data and "response" in data:
            sub = data.get("response", {}).get("subscription", {})
            return {
                "available": True,
                "plan": sub.get("plan", "free"),
                "requests_left": max(0, DAILY_LIMIT - self._requests_today),
                "daily_limit": DAILY_LIMIT,
            }
        return {"available": False, "error": "API key invalid"}

    async def search_team(self, query: str) -> list[dict]:
        """Search teams by name."""

        data = await self._request(f"/teams?search={query}", ttl=CACHE_TTL_LONG)
        if not data:
            return []
        results = []
        for item in data.get("response", []):
            t = item.get("team", {})
            v = item.get("venue", {})
            results.append(
                {
                    "id": t.get("id"),
                    "name": t.get("name", ""),
                    "code": t.get("code", ""),
                    "country": t.get("country", ""),
                    "logo": t.get("logo"),
                    "founded": t.get("founded"),
                    "venue_name": v.get("name"),
                    "venue_city": v.get("city"),
                    "venue_capacity": v.get("capacity"),
                }
            )
        return results

    async def get_team_squad(self, team_id: int) -> list[dict]:
        """Get squad for a team. Returns list of {id, name, position, jersey_number, photo, age}."""
        data = await self._request(f"/players/squads?team={team_id}", ttl=CACHE_TTL_LONG)
        if not data:
            return []
        players = []
        for item in data.get("response", []):
            for p in item.get("players", []):
                players.append(
                    {
                        "id": p.get("id"),
                        "name": p.get("name", ""),
                        "position": p.get("position", ""),
                        "jersey_number": p.get("number"),
                        "age": p.get("age"),
                        "photo": p.get("photo"),
                    }
                )
        return players

    async def get_standings(self, league_id: int, season: int = 2024) -> list[dict]:
        """Get league standings."""
        data = await self._request(
            f"/standings?league={league_id}&season={season}", ttl=CACHE_TTL_MEDIUM
        )
        if not data:
            return []
        rows = []
        for item in data.get("response", []):
            for standing_list in item.get("league", {}).get("standings", []):
                for s in standing_list:
                    rows.append(
                        {
                            "rank": s.get("rank"),
                            "team_id": s.get("team", {}).get("id"),
                            "team_name": s.get("team", {}).get("name", ""),
                            "team_logo": s.get("team", {}).get("logo"),
                            "points": s.get("points"),
                            "goalsDiff": s.get("goalsDiff"),
                            "played": s.get("all", {}).get("played"),
                            "wins": s.get("all", {}).get("win"),
                            "draws": s.get("all", {}).get("draw"),
                            "losses": s.get("all", {}).get("lose"),
                            "goals_for": s.get("all", {}).get("goals", {}).get("for"),
                            "goals_against": s.get("all", {}).get("goals", {}).get("against"),
                            "form": s.get("form"),
                            "description": s.get("description"),
                            "group": s.get("group"),
                        }
                    )
        return rows

    async def get_fixtures(
        self, team_id: int, season: int = 2024, last: int | None = None, next: int | None = None
    ) -> list[dict]:
        """Get fixtures for a team."""
        path = f"/fixtures?team={team_id}&season={season}"
        if last is not None:
            path += f"&last={last}"
        if next is not None:
            path += f"&next={next}"
        data = await self._request(path, ttl=CACHE_TTL_SHORT)
        if not data:
            return []
        matches = []
        for item in data.get("response", []):
            f = item.get("fixture", {})
            teams = item.get("teams", {})
            goals = item.get("goals", {})
            league = item.get("league", {})
            _ = item.get("score", {})
            matches.append(
                {
                    "id": f.get("id"),
                    "date": f.get("date", ""),
                    "status": f.get("status", {}).get("long", ""),
                    "short_status": f.get("status", {}).get("short", ""),
                    "elapsed": f.get("status", {}).get("elapsed"),
                    "venue": f.get("venue", {}).get("name"),
                    "home_team": teams.get("home", {}).get("name", ""),
                    "home_team_id": teams.get("home", {}).get("id"),
                    "home_logo": teams.get("home", {}).get("logo"),
                    "away_team": teams.get("away", {}).get("name", ""),
                    "away_team_id": teams.get("away", {}).get("id"),
                    "away_logo": teams.get("away", {}).get("logo"),
                    "home_score": goals.get("home"),
                    "away_score": goals.get("away"),
                    "league_name": league.get("name", ""),
                    "league_id": league.get("id"),
                    "round": league.get("round"),
                }
            )
        return matches

    async def get_predictions(self, fixture_id: int) -> dict | None:
        """Get AI predictions for a fixture."""
        data = await self._request(f"/predictions?fixture={fixture_id}", ttl=CACHE_TTL_MEDIUM)
        if not data:
            return None
        response = data.get("response", [])
        if not response:
            return None
        return response[0].get("predictions", {})

    async def get_leagues(self, search: str | None = None) -> list[dict]:
        """Get available leagues, optionally filtered by search."""
        path = "/leagues"
        if search:
            path += f"?search={search}"
        data = await self._request(path, ttl=CACHE_TTL_LONG)
        if not data:
            return []
        leagues = []
        for item in data.get("response", []):
            league = item.get("league", {})
            c = item.get("country", {})
            seasons = item.get("seasons", [])
            current_season = None
            for s in seasons:
                if s.get("current"):
                    current_season = s.get("year")
                    break
            leagues.append(
                {
                    "id": league.get("id"),
                    "name": league.get("name", ""),
                    "type": league.get("type", ""),
                    "logo": league.get("logo"),
                    "country": c.get("name", ""),
                    "country_code": c.get("code"),
                    "country_flag": c.get("flag"),
                    "current_season": current_season,
                }
            )
        return leagues

    async def get_live_fixtures(self) -> list[dict]:
        """Get currently live fixtures."""
        return await self.get_fixtures_team(0, live="all")

    async def get_fixtures_team(
        self, team_id: int | None = None, live: str | None = None
    ) -> list[dict]:
        """Get fixtures, optionally filtered by team or live status."""
        path = "/fixtures?"
        params = []
        if team_id:
            params.append(f"team={team_id}")
        if live:
            params.append(f"live={live}")
        path += "&".join(params)
        data = await self._request(path, ttl=CACHE_TTL_SHORT)
        if not data:
            return []
        matches = []
        for item in data.get("response", []):
            f = item.get("fixture", {})
            teams = item.get("teams", {})
            goals = item.get("goals", {})
            league = item.get("league", {})
            matches.append(
                {
                    "id": f.get("id"),
                    "date": f.get("date", ""),
                    "status": f.get("status", {}).get("long", ""),
                    "short_status": f.get("status", {}).get("short", ""),
                    "elapsed": f.get("status", {}).get("elapsed"),
                    "home_team": teams.get("home", {}).get("name", ""),
                    "home_team_id": teams.get("home", {}).get("id"),
                    "away_team": teams.get("away", {}).get("name", ""),
                    "away_team_id": teams.get("away", {}).get("id"),
                    "home_score": goals.get("home"),
                    "away_score": goals.get("away"),
                    "league_name": league.get("name", ""),
                    "league_id": league.get("id"),
                }
            )
        return matches

    async def get_fixture_detail(self, fixture_id: int) -> dict | None:
        """Get detailed fixture info."""
        data = await self._request(f"/fixtures?id={fixture_id}", ttl=CACHE_TTL_SHORT)
        if not data:
            return None
        response = data.get("response", [])
        if not response:
            return None
        item = response[0]
        f = item.get("fixture", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        league = item.get("league", {})
        return {
            "id": f.get("id"),
            "date": f.get("date", ""),
            "status": f.get("status", {}).get("long", ""),
            "elapsed": f.get("status", {}).get("elapsed"),
            "venue": f.get("venue", {}).get("name"),
            "home_team": teams.get("home", {}).get("name", ""),
            "away_team": teams.get("away", {}).get("name", ""),
            "home_score": goals.get("home"),
            "away_score": goals.get("away"),
            "league_name": league.get("name", ""),
            "league_id": league.get("id"),
            "season": league.get("season"),
            "round": league.get("round"),
        }

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
