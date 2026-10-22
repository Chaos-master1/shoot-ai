"""Async service for openfootball data repos (football.json + worldcup).

Free public domain data available as JSON on GitHub raw URLs.
No API key needed. CC0 license.
Covers Big 5 leagues (2010-2026) and World Cups (1930-2026).
Silent degradation: returns empty results on failure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

CACHE_TTL_LONG = 86400

FOOTBALL_JSON_BASE = "https://raw.githubusercontent.com/openfootball/football.json/master"
WORLDCUP_BASE = "https://raw.githubusercontent.com/openfootball/worldcup/master"

LEAGUE_MAP: dict[str, str] = {
    "en.1": "English Premier League",
    "en.2": "English Championship",
    "de.1": "Deutsche Bundesliga",
    "es.1": "Spanish La Liga",
    "it.1": "Italian Serie A",
    "fr.1": "French Ligue 1",
}

SEASONS: list[str] = [
    "2010-11",
    "2011-12",
    "2012-13",
    "2013-14",
    "2014-15",
    "2015-16",
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

WORLDCUP_YEARS: list[int] = [
    1930,
    1934,
    1938,
    1950,
    1954,
    1958,
    1962,
    1966,
    1970,
    1974,
    1978,
    1982,
    1986,
    1990,
    1994,
    1998,
    2002,
    2006,
    2010,
    2014,
    2018,
    2022,
    2026,
]


@dataclass
class MatchResult:
    competition: str
    season: str
    round: str
    date: str
    time: str
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    half_time_home: int | None
    half_time_away: int | None
    raw: dict = field(default_factory=dict)


@dataclass
class CompetitionInfo:
    id: str
    name: str
    seasons: list[str]


class OpenFootballDataService:
    """Async wrapper for openfootball data repos.

    Provides historical match fixtures and results from football.json (Big 5 leagues)
    and worldcup (1930-2026). All methods return empty results on failure.
    """

    def __init__(
        self,
        football_json_base: str = FOOTBALL_JSON_BASE,
        worldcup_base: str = WORLDCUP_BASE,
    ) -> None:
        self.football_json_base = football_json_base
        self.worldcup_base = worldcup_base
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, tuple[float, Any]] = {}
        self._available = False

    async def _ensure_client(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=20.0)

    def _cache_get(self, key: str) -> Any | None:
        if key in self._cache:
            expires, data = self._cache[key]
            if time.monotonic() < expires:
                return data
            del self._cache[key]
        return None

    def _cache_set(self, key: str, data: Any, ttl: int) -> None:
        self._cache[key] = (time.monotonic() + ttl, data)

    async def _get(self, url: str, ttl: int = CACHE_TTL_LONG) -> dict | None:
        cache_key = f"ofb:{url}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        await self._ensure_client()
        assert self._client is not None
        try:
            r = await self._client.get(url)
            if r.status_code == 200:
                data = r.json()
                self._cache_set(cache_key, data, ttl)
                self._available = True
                return data
            else:
                logger.warning(f"openfootball {r.status_code}: {url}")
                return None
        except httpx.RequestError as e:
            logger.warning(f"openfootball request failed: {e}")
            return None

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Competitions / leagues
    # ------------------------------------------------------------------

    def get_competitions(self) -> list[CompetitionInfo]:
        """Return the static list of supported competitions + seasons."""
        return [
            CompetitionInfo(id=lid, name=name, seasons=list(SEASONS))
            for lid, name in LEAGUE_MAP.items()
        ]

    # ------------------------------------------------------------------
    # League matches
    # ------------------------------------------------------------------

    async def get_matches(self, competition_id: str, season: str) -> list[MatchResult]:
        """Fetch matches for a league (competition_id like 'en.1') and season."""
        if competition_id not in LEAGUE_MAP:
            return []
        url = f"{self.football_json_base}/{season}/{competition_id}.json"
        data = await self._get(url)
        if not data or "matches" not in data:
            return []
        return [self._parse_match(m, competition_id, season) for m in data["matches"]]

    def _parse_match(self, raw: dict, competition_id: str, season: str) -> MatchResult:
        score = raw.get("score")
        home_score, away_score, ht_home, ht_away = None, None, None, None
        if isinstance(score, dict):
            ft = score.get("ft")
            if isinstance(ft, list) and len(ft) == 2:
                home_score = ft[0] if ft[0] is not None else None
                away_score = ft[1] if ft[1] is not None else None
            ht = score.get("ht")
            if isinstance(ht, list) and len(ht) == 2:
                ht_home = ht[0] if ht[0] is not None else None
                ht_away = ht[1] if ht[1] is not None else None
        elif isinstance(score, list) and len(score) == 2:
            home_score = score[0] if score[0] is not None else None
            away_score = score[1] if score[1] is not None else None
        return MatchResult(
            competition=LEAGUE_MAP.get(competition_id, competition_id),
            season=season,
            round=raw.get("round", ""),
            date=raw.get("date", ""),
            time=raw.get("time", ""),
            home_team=raw.get("team1", ""),
            away_team=raw.get("team2", ""),
            home_score=home_score,
            away_score=away_score,
            half_time_home=ht_home,
            half_time_away=ht_away,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Team search across leagues
    # ------------------------------------------------------------------

    async def search_team_matches(
        self, team_name: str, competition_id: str | None = None
    ) -> list[MatchResult]:
        """Search all matches containing a team across leagues/seasons."""
        results: list[MatchResult] = []
        comps = [competition_id] if competition_id else list(LEAGUE_MAP.keys())
        needle = team_name.lower()
        for comp in comps:
            for season in SEASONS:
                matches = await self.get_matches(comp, season)
                for m in matches:
                    if needle in m.home_team.lower() or needle in m.away_team.lower():
                        results.append(m)
        return results

    # ------------------------------------------------------------------
    # World Cup
    # ------------------------------------------------------------------

    def get_all_worldcup_years(self) -> list[int]:
        return list(WORLDCUP_YEARS)

    async def get_worldcup_matches(self, year: int) -> list[MatchResult]:
        """Fetch World Cup matches for a given year."""
        url = f"{self.worldcup_base}/{year}/worldcup.json"
        data = await self._get(url)
        if not data or "matches" not in data:
            return []
        return [self._parse_match(m, "worldcup", str(year)) for m in data["matches"]]
