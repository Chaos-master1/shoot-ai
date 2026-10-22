"""Async service for TheSportsDB API v1 (free tier).

Free tier: 30 req/min, no daily cap, public API key '123'.
Covers Botola Pro standings, fixtures, team info, venues.
Squad/player endpoint returns 404 on free tier.
Silent degradation: returns empty results on failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

CACHE_TTL_SHORT = 300
CACHE_TTL_MEDIUM = 3600
CACHE_TTL_LONG = 86400

BASE_URL = "https://www.thesportsdb.com/api/v1/json"


@dataclass
class TeamResult:
    id: str
    name: str
    alternate_name: str
    league_id: str
    league_name: str
    badge_url: str
    formed_year: str
    stadium: str
    stadium_capacity: str
    location: str
    description: str
    api_football_id: str
    raw: dict = field(default_factory=dict)


@dataclass
class StandingEntry:
    rank: int
    team_id: str
    team_name: str
    badge_url: str
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    goal_diff: int
    points: int
    form: str
    description: str
    raw: dict = field(default_factory=dict)


@dataclass
class EventResult:
    id: str
    event_name: str
    home_team: str
    away_team: str
    home_team_id: str
    away_team_id: str
    home_score: int | None
    away_score: int | None
    round: str
    season: str
    date: str
    time: str
    league_id: str
    league_name: str
    status: str
    raw: dict = field(default_factory=dict)


class TheSportsDBService:
    """Async wrapper for TheSportsDB API v1 (free tier).

    Provides team search, standings, fixtures, event details, and venue info.
    All methods return empty results on failure.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or "123"
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, tuple[float, Any]] = {}
        self._available = False

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=f"{BASE_URL}/{self.api_key}", timeout=15.0)
        assert (
            self._client is not None
        )  # narrowed for mypy (https://github.com/python/mypy/issues/11011)
        return self._client

    def _cache_get(self, key: str) -> Any | None:
        if key in self._cache:
            expires, data = self._cache[key]
            if time.monotonic() < expires:
                return data
            del self._cache[key]
        return None

    def _cache_set(self, key: str, data: Any, ttl: int) -> None:
        self._cache[key] = (time.monotonic() + ttl, data)

    async def _get(self, path: str, ttl: int = CACHE_TTL_SHORT) -> dict[str, Any] | None:
        """Fetch a TheSportsDB v1 endpoint (JSON object) with caching.

        The v1 API always returns a JSON object keyed by result-set name
        ("teams", "events", ...); a list return was never real, and the
        ``| list`` union previously poisoned every ``data["teams"]`` site
        with a spurious list-indexing error.
        """
        cache_key = f"tsdb:{path}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            result: dict[str, Any] = cached
            return result

        client = self._ensure_client()
        try:
            r = await client.get(path)
            if r.status_code == 200:
                data: dict[str, Any] = r.json()
                self._cache_set(cache_key, data, ttl)
                self._available = True
                return data
            else:
                logger.warning(f"TheSportsDB API error {r.status_code}: {path}")
                return None
        except httpx.RequestError as e:
            logger.warning(f"TheSportsDB API request failed: {e}")
            return None

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Team search
    # ------------------------------------------------------------------

    async def search_teams(self, query: str) -> list[TeamResult]:
        """Search teams by name (partial match)."""
        data = await self._get(f"searchteams.php?t={query}", ttl=CACHE_TTL_MEDIUM)
        if not data or "teams" not in data or not data["teams"]:
            return []

        teams = data["teams"]
        if not isinstance(teams, list):
            return []

        results = []
        for t in teams:
            if not isinstance(t, dict):
                continue
            results.append(
                TeamResult(
                    id=str(t.get("idTeam", "")),
                    name=str(t.get("strTeam", "")),
                    alternate_name=str(t.get("strTeamAlternate", "")),
                    league_id=str(t.get("idLeague", "")),
                    league_name=str(t.get("strLeague", "")),
                    badge_url=str(t.get("strBadge", "") or t.get("strTeamBadge", "")),
                    formed_year=str(t.get("intFormedYear", "")),
                    stadium=str(t.get("strStadium", "")),
                    stadium_capacity=str(t.get("intStadiumCapacity", "")),
                    location=str(t.get("strLocation", "")),
                    description=str(t.get("strDescriptionEN", "")),
                    api_football_id=str(t.get("idAPIfootball", "")),
                    raw=t,
                )
            )
        return results

    async def get_team(self, team_id: str) -> TeamResult | None:
        """Get team by ID."""
        data = await self._get(f"lookupteam.php?id={team_id}", ttl=CACHE_TTL_MEDIUM)
        if not data or "teams" not in data or not data["teams"]:
            return None
        t = data["teams"][0]
        if not isinstance(t, dict):
            return None
        return TeamResult(
            id=str(t.get("idTeam", "")),
            name=str(t.get("strTeam", "")),
            alternate_name=str(t.get("strTeamAlternate", "")),
            league_id=str(t.get("idLeague", "")),
            league_name=str(t.get("strLeague", "")),
            badge_url=str(t.get("strBadge", "") or t.get("strTeamBadge", "")),
            formed_year=str(t.get("intFormedYear", "")),
            stadium=str(t.get("strStadium", "")),
            stadium_capacity=str(t.get("intStadiumCapacity", "")),
            location=str(t.get("strLocation", "")),
            description=str(t.get("strDescriptionEN", "")),
            api_football_id=str(t.get("idAPIfootball", "")),
            raw=t,
        )

    # ------------------------------------------------------------------
    # League lookup
    # ------------------------------------------------------------------

    async def get_league(self, league_id: str) -> dict | None:
        """Get league info by ID."""
        data = await self._get(f"lookupleague.php?id={league_id}", ttl=CACHE_TTL_LONG)
        if data and "leagues" in data and data["leagues"]:
            league: dict = data["leagues"][0]
            return league
        return None

    # ------------------------------------------------------------------
    # Standings / table
    # ------------------------------------------------------------------

    async def get_standings(self, league_id: str, season: str = "") -> list[StandingEntry]:
        """Get league standings."""
        path = f"lookuptable.php?l={league_id}"
        if season:
            path += f"&s={season}"
        data = await self._get(path, ttl=CACHE_TTL_SHORT)
        if not data or "table" not in data or not data["table"]:
            return []

        entries = data["table"]
        if not isinstance(entries, list):
            return []

        results = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            results.append(
                StandingEntry(
                    rank=int(e.get("intRank", 0)),
                    team_id=str(e.get("idTeam", "")),
                    team_name=str(e.get("strTeam", "")),
                    badge_url=str(e.get("strBadge", "")),
                    played=int(e.get("intPlayed", 0)),
                    won=int(e.get("intWin", 0)),
                    drawn=int(e.get("intDraw", 0)),
                    lost=int(e.get("intLoss", 0)),
                    goals_for=int(e.get("intGoalsFor", 0)),
                    goals_against=int(e.get("intGoalsAgainst", 0)),
                    goal_diff=int(e.get("intGoalDifference", 0)),
                    points=int(e.get("intPoints", 0)),
                    form=str(e.get("strForm", "")),
                    description=str(e.get("strDescription", "")),
                    raw=e,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Events / fixtures
    # ------------------------------------------------------------------

    async def get_team_events_last(self, team_id: str) -> list[EventResult]:
        """Get last (recent) events for a team."""
        data = await self._get(f"eventslast.php?id={team_id}", ttl=CACHE_TTL_SHORT)
        if not data or "results" not in data or not data["results"]:
            return []
        return self._parse_events(data["results"])

    async def get_team_events_next(self, team_id: str) -> list[EventResult]:
        """Get next (upcoming) events for a team."""
        data = await self._get(f"eventsnext.php?id={team_id}", ttl=CACHE_TTL_SHORT)
        if not data or "events" not in data or not data["events"]:
            return []
        return self._parse_events(data["events"])

    async def get_round_events(
        self, league_id: str, season: str, round_num: str
    ) -> list[EventResult]:
        """Get all events for a specific round."""
        data = await self._get(
            f"eventsround.php?id={league_id}&s={season}&r={round_num}",
            ttl=CACHE_TTL_SHORT,
        )
        if not data or "events" not in data or not data["events"]:
            return []
        return self._parse_events(data["events"])

    async def get_event(self, event_id: str) -> EventResult | None:
        """Get event detail by ID."""
        data = await self._get(f"lookupevent.php?id={event_id}", ttl=CACHE_TTL_SHORT)
        if not data or "events" not in data or not data["events"]:
            return None
        events = self._parse_events(data["events"])
        return events[0] if events else None

    def _parse_events(self, raw_events: list) -> list[EventResult]:
        results = []
        for e in raw_events:
            if not isinstance(e, dict):
                continue
            home_score = e.get("intHomeScore")
            away_score = e.get("intAwayScore")
            results.append(
                EventResult(
                    id=str(e.get("idEvent", "")),
                    event_name=str(e.get("strEvent", "")),
                    home_team=str(e.get("strHomeTeam", "")),
                    away_team=str(e.get("strAwayTeam", "")),
                    home_team_id=str(e.get("idHomeTeam", "")),
                    away_team_id=str(e.get("idAwayTeam", "")),
                    home_score=int(home_score)
                    if home_score and home_score != "0"
                    else (int(home_score) if home_score else None),
                    away_score=int(away_score)
                    if away_score and away_score != "0"
                    else (int(away_score) if away_score else None),
                    round=str(e.get("intRound", "")),
                    season=str(e.get("strSeason", "")),
                    date=str(e.get("dateEvent", "")),
                    time=str(e.get("strTime", "")),
                    league_id=str(e.get("idLeague", "")),
                    league_name=str(e.get("strLeague", "")),
                    status=str(e.get("strStatus", "scheduled")),
                    raw=e,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Venue lookup
    # ------------------------------------------------------------------

    async def get_venue(self, venue_id: str) -> dict | None:
        """Get venue info by ID."""
        data = await self._get(f"lookupvenue.php?id={venue_id}", ttl=CACHE_TTL_LONG)
        if data and "venues" in data and data["venues"]:
            venue: dict = data["venues"][0]
            return venue
        return None

    # ------------------------------------------------------------------
    # Search all leagues
    # ------------------------------------------------------------------

    async def get_all_leagues(self) -> list[dict]:
        """Get all leagues (returns only id + name + sport)."""
        data = await self._get("all_leagues.php", ttl=CACHE_TTL_LONG)
        if data and "leagues" in data and data["leagues"]:
            leagues: list[dict] = data["leagues"]
            return leagues
        return []
