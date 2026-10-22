"""Async service for football-data.org API v4.

Free tier: 10 req/min, TIER_ONE competitions (PL, BL1, SA, PD, FL1, CL, WC, ...).
Silent degradation: if API is offline/rate-limited, returns empty results.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

CACHE_TTL_SHORT = 300  # 5 min for matches/scores
CACHE_TTL_MEDIUM = 3600  # 1 h for standings
CACHE_TTL_LONG = 86400  # 24 h for team/squad/competition data


@dataclass
class TeamSearchResult:
    id: int
    name: str
    short_name: str
    tla: str
    crest: str | None
    competition_name: str | None = None
    competition_code: str | None = None
    area_name: str | None = None


class FootballDataService:
    """Async wrapper for football-data.org API v4.

    Provides team search, squad import, match verification, and standings.
    All methods return empty results on failure (silent degrade).
    """

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None) -> None:
        self.api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY")
        self._client: httpx.AsyncClient | None = None
        self._tokens = 10.0
        self._last_refill = time.monotonic()
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_path: Path | None = cache_dir
        self._available = False

    async def _ensure_client(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=15.0)

    async def _rate_limit(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(10.0, self._tokens + elapsed * (10.0 / 60.0))
        self._last_refill = now
        while self._tokens < 1.0:
            await asyncio.sleep(0.1)
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(10.0, self._tokens + elapsed * (10.0 / 60.0))
            self._last_refill = now
        self._tokens -= 1.0

    def _cache_key(self, endpoint: str, params: dict | None) -> str:
        return f"{endpoint}_{json.dumps(params or {}, sort_keys=True, default=str)}"

    def _cache_ttl(self, endpoint: str) -> int:
        if endpoint.startswith("/teams/") and not endpoint.endswith("/matches"):
            return CACHE_TTL_LONG
        if endpoint.startswith("/competitions/") and (
            "/standings" in endpoint or "/teams" in endpoint
        ):
            return CACHE_TTL_MEDIUM
        if endpoint == "/teams":
            return CACHE_TTL_LONG
        if endpoint.startswith("/competitions"):
            return CACHE_TTL_LONG
        return CACHE_TTL_SHORT

    async def _request(self, endpoint: str, params: dict | None = None) -> dict | None:
        ckey = self._cache_key(endpoint, params)
        if ckey in self._cache:
            expires, data = self._cache[ckey]
            if time.monotonic() < expires:
                return data
        await self._rate_limit()
        try:
            await self._ensure_client()
            assert self._client is not None
            headers = {"X-Auth-Token": self.api_key} if self.api_key else {}
            resp = await self._client.get(
                f"{self.BASE_URL}{endpoint}", params=params, headers=headers
            )
            if resp.status_code == 429:
                await asyncio.sleep(3)
                return await self._request(endpoint, params)
            if resp.status_code == 403:
                self._available = False
                return None
            if resp.status_code != 200:
                return None
            data = resp.json()
            ttl = self._cache_ttl(endpoint)
            self._cache[ckey] = (time.monotonic() + ttl, data)
            self._available = True
            return data
        except httpx.TimeoutException:
            logger.warning("football-data.org timeout")
            return None
        except httpx.RequestError as e:
            logger.warning(f"football-data.org request failed: {e}")
            return None
        except Exception as e:
            logger.warning(f"football-data.org error: {e}")
            return None

    async def check_status(self) -> dict:
        """Validate API key and return availability info."""
        data = await self._request("/competitions", {"limit": 1})
        if data and "competitions" in data:
            return {
                "available": True,
                "competitions_count": data.get("count", 0),
                "rate_remaining": max(0, int(self._tokens)),
            }
        return {"available": False, "error": "API key invalid or rate limited"}

    async def search_team(self, query: str) -> list[dict]:
        """Search teams by name across all cached competitions."""
        query_lower = query.lower().strip()
        if not query_lower:
            return []
        teams_data = await self._request("/teams")
        results: list[dict] = []
        if teams_data:
            for t in teams_data.get("teams", []):
                if (
                    query_lower in t.get("name", "").lower()
                    or query_lower in t.get("shortName", "").lower()
                ):
                    results.append(
                        {
                            "id": t["id"],
                            "name": t["name"],
                            "short_name": t.get("shortName", ""),
                            "tla": t.get("tla", ""),
                            "crest": t.get("crest"),
                            "area_name": t.get("area", {}).get("name") if t.get("area") else None,
                        }
                    )
        if not results:
            comp_data = await self._request("/competitions")
            if comp_data:
                for comp in comp_data.get("competitions", []):
                    code = comp.get("code")
                    if not code:
                        continue
                    ct = await self._request(f"/competitions/{code}/teams")
                    if not ct:
                        continue
                    for t in ct.get("teams", []):
                        if (
                            query_lower in t.get("name", "").lower()
                            or query_lower in t.get("shortName", "").lower()
                        ):
                            results.append(
                                {
                                    "id": t["id"],
                                    "name": t["name"],
                                    "short_name": t.get("shortName", ""),
                                    "tla": t.get("tla", ""),
                                    "crest": t.get("crest"),
                                    "area_name": t.get("area", {}).get("name")
                                    if t.get("area")
                                    else None,
                                    "competition_name": comp.get("name"),
                                    "competition_code": code,
                                }
                            )
        dedup = {r["id"]: r for r in results}
        return list(dedup.values())

    async def get_team(self, team_id: int) -> dict | None:
        """Get full team details including squad."""
        return await self._request(f"/teams/{team_id}")

    async def get_team_matches(
        self,
        team_id: int,
        date_from: str | None = None,
        date_to: str | None = None,
        status: str | None = None,
    ) -> list[dict]:
        """Get matches for a team."""
        params = {}
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        if status:
            params["status"] = status
        data = await self._request(f"/teams/{team_id}/matches", params)
        if not data:
            return []
        return data.get("matches", [])

    async def get_match(self, match_id: int, unfold: bool = False) -> dict | None:
        """Get full match details with optional lineup/goal unfold."""
        extra_headers = {}
        if unfold:
            extra_headers = {
                "X-Unfold-Lineups": "true",
                "X-Unfold-Goals": "true",
                "X-Unfold-Bookings": "true",
                "X-Unfold-Subs": "true",
            }
        ckey = self._cache_key(f"/matches/{match_id}", {"unfold": unfold})
        if ckey in self._cache:
            expires, data = self._cache[ckey]
            if time.monotonic() < expires:
                return data
        await self._rate_limit()
        try:
            await self._ensure_client()
            assert self._client is not None
            headers = {"X-Auth-Token": self.api_key} if self.api_key else {}
            headers.update(extra_headers)
            resp = await self._client.get(f"{self.BASE_URL}/matches/{match_id}", headers=headers)
            if resp.status_code == 429:
                await asyncio.sleep(3)
                return await self.get_match(match_id, unfold)
            if resp.status_code != 200:
                return None
            data = resp.json()
            ttl = CACHE_TTL_MEDIUM
            self._cache[ckey] = (time.monotonic() + ttl, data)
            return data
        except Exception as e:
            logger.warning(f"get_match({match_id}) failed: {e}")
            return None

    async def get_standings(self, competition_code: str) -> list[dict] | None:
        """Get standings for a competition (TOTAL, HOME, AWAY)."""
        data = await self._request(f"/competitions/{competition_code}/standings")
        if not data:
            return None
        return data.get("standings", [])

    async def get_competition_teams(self, competition_code: str) -> list[dict]:
        """Get all teams in a competition."""
        data = await self._request(f"/competitions/{competition_code}/teams")
        if not data:
            return []
        return data.get("teams", [])

    async def get_competitions(self) -> list[dict]:
        """Get all available competitions."""
        data = await self._request("/competitions")
        if not data:
            return []
        return data.get("competitions", [])

    async def get_competition(self, competition_code: str) -> dict | None:
        """Get a single competition."""
        return await self._request(f"/competitions/{competition_code}")

    async def get_competition_matches(
        self, competition_code: str, matchday: int | None = None, status: str | None = None
    ) -> list[dict]:
        """Get matches for a competition."""
        params: dict[str, int | str] = {}
        if matchday:
            params["matchday"] = matchday
        if status:
            params["status"] = status
        data = await self._request(f"/competitions/{competition_code}/matches", params)
        if not data:
            return []
        return data.get("matches", [])

    async def import_team_squad(self, team_id: int, side: str = "home") -> list[dict]:
        """Fetch squad and return list of player dicts ready for profile creation."""
        team = await self.get_team(team_id)
        if not team:
            return []
        squad = team.get("squad", [])
        result = []
        for player in squad:
            shirt = player.get("shirtNumber")
            if shirt is None:
                continue
            result.append(
                {
                    "display_name": player.get("name", ""),
                    "jersey_number": shirt,
                    "preferred_position": player.get("position", ""),
                    "nationality": player.get("nationality"),
                    "date_of_birth": player.get("dateOfBirth"),
                    "team": side,
                    "football_data_person_id": player.get("id"),
                    "football_data_team_id": team_id,
                }
            )
        return result

    async def verify_match(
        self, api_match_id: int, detected_score_home: int, detected_score_away: int
    ) -> dict | None:
        """Compare detected score with API match data."""
        match = await self.get_match(api_match_id, unfold=False)
        if not match:
            return None
        score = match.get("score") or {}
        ft = score.get("fullTime") or {}
        api_home = ft.get("home")
        api_away = ft.get("away")
        if api_home is None or api_away is None:
            return {
                "verified": False,
                "reason": "Match not finished or no score data",
                "api_score": None,
                "status": match.get("status"),
            }
        match_result = api_home == detected_score_home and api_away == detected_score_away
        return {
            "verified": match_result,
            "api_score": {"home": api_home, "away": api_away},
            "detected_score": {"home": detected_score_home, "away": detected_score_away},
            "status": match.get("status"),
            "competition": (match.get("competition") or {}).get("name"),
            "matchday": match.get("matchday"),
        }

    async def get_todays_matches(self) -> list[dict]:
        """Get all matches scheduled for today."""
        data = await self._request("/matches")
        if not data:
            return []
        return data.get("matches", [])

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def close(self) -> None:
        await self.shutdown()
