"""Async service for StatsBomb open data (free, attribution required).

Provides event-level football data: passes, shots (with xG), 360 positions, lineups.
Covers top leagues + UCL + World Cup. No Botola/Morocco coverage.
Data lives at github.com/statsbomb/open-data.
Terms: must credit StatsBomb and use their logo when publishing analysis.

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

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


@dataclass
class SbCompetition:
    competition_id: int
    season_id: int
    competition_name: str
    country_name: str
    season_name: str
    competition_gender: str
    competition_international: bool
    competition_youth: bool
    has_360: bool
    raw: dict = field(default_factory=dict)


@dataclass
class SbMatch:
    match_id: int
    competition_id: int
    season_id: int
    competition_name: str
    season_name: str
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None
    match_date: str
    competition_stage: str
    stadium: str
    referee: str
    has_360: bool
    raw: dict = field(default_factory=dict)


@dataclass
class SbEvent:
    event_id: str
    match_id: int
    index: int
    period: int
    minute: int
    second: int
    timestamp: str
    event_type: str
    team: str
    player: str
    player_id: int | None
    position: str
    location_x: float | None
    location_y: float | None
    outcome: str
    possession: int | None
    xg: float | None
    shot_type: str
    shot_body_part: str
    pass_target: str
    raw: dict = field(default_factory=dict)


@dataclass
class SbLineup:
    team_name: str
    team_id: int
    players: list[dict]


class StatsBombService:
    """Async wrapper for StatsBomb open data.

    All methods return empty results on failure.
    """

    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, tuple[float, Any]] = {}
        self._available = False

    async def _ensure_client(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)

    def _cache_get(self, key: str) -> Any | None:
        if key in self._cache:
            expires, data = self._cache[key]
            if time.monotonic() < expires:
                return data
            del self._cache[key]
        return None

    def _cache_set(self, key: str, data: Any, ttl: int) -> None:
        self._cache[key] = (time.monotonic() + ttl, data)

    async def _get(self, path: str, ttl: int = CACHE_TTL_LONG) -> Any | None:
        url = f"{self.base_url}/{path}"
        cache_key = f"sb:{path}"
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
                logger.warning(f"StatsBomb {r.status_code}: {path}")
                return None
        except httpx.RequestError as e:
            logger.warning(f"StatsBomb request failed: {e}")
            return None

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Competitions
    # ------------------------------------------------------------------

    async def get_competitions(self) -> list[SbCompetition]:
        data = await self._get("competitions.json")
        if not data or not isinstance(data, list):
            return []
        return [self._parse_competition(c) for c in data]

    def _parse_competition(self, raw: dict) -> SbCompetition:
        return SbCompetition(
            competition_id=int(raw.get("competition_id", 0)),
            season_id=int(raw.get("season_id", 0)),
            competition_name=str(raw.get("competition_name", "")),
            country_name=str(raw.get("country_name", "")),
            season_name=str(raw.get("season_name", "")),
            competition_gender=str(raw.get("competition_gender", "")),
            competition_international=bool(raw.get("competition_international", False)),
            competition_youth=bool(raw.get("competition_youth", False)),
            has_360=raw.get("match_available_360") is not None,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Matches
    # ------------------------------------------------------------------

    async def get_matches(self, competition_id: int, season_id: int) -> list[SbMatch]:
        path = f"matches/{competition_id}/{season_id}.json"
        data = await self._get(path)
        if not data or not isinstance(data, list):
            return []
        return [self._parse_match(m) for m in data]

    def _parse_match(self, raw: dict) -> SbMatch:
        home = raw.get("home_team", {}) or {}
        away = raw.get("away_team", {}) or {}
        score = raw.get("home_score")
        away_score = raw.get("away_score")
        return SbMatch(
            match_id=int(raw.get("match_id", 0)),
            competition_id=int(raw.get("competition", raw.get("competition_id", 0))),
            season_id=int(raw.get("season", raw.get("season_id", 0))),
            competition_name=str(raw.get("competition_name", "")),
            season_name=str(raw.get("season_name", "")),
            home_team=str(home.get("home_team_name", "")),
            away_team=str(away.get("away_team_name", "")),
            home_score=int(score) if score is not None else None,
            away_score=int(away_score) if away_score is not None else None,
            match_date=str(raw.get("match_date", "")),
            competition_stage=str(raw.get("competition_stage", "")),
            stadium=str(
                raw.get("stadium", {}).get("name", "")
                if isinstance(raw.get("stadium"), dict)
                else raw.get("stadium", "")
            ),
            referee=str(
                raw.get("referee", {}).get("name", "")
                if isinstance(raw.get("referee"), dict)
                else raw.get("referee", "")
            ),
            has_360=raw.get("match_available_360") is not None,
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    async def get_events(self, match_id: int) -> list[SbEvent]:
        path = f"events/{match_id}.json"
        data = await self._get(path)
        if not data or not isinstance(data, list):
            return []
        return [self._parse_event(e, match_id, i) for i, e in enumerate(data)]

    def _parse_event(self, raw: dict, match_id: int, index: int) -> SbEvent:
        loc = raw.get("location", [])
        loc_x = loc[0] if isinstance(loc, list) and len(loc) >= 1 else None
        loc_y = loc[1] if isinstance(loc, list) and len(loc) >= 2 else None
        outcome_map = {"Success": "success", "Failure": "failure", "Incomplete": "incomplete"}
        pos = raw.get("position", {}) or {}
        shot = raw.get("shot", {}) or {}
        pass_ = raw.get("pass", {}) or {}
        return SbEvent(
            event_id=str(raw.get("id", "")),
            match_id=match_id,
            index=index,
            period=int(raw.get("period", 0)),
            minute=int(raw.get("minute", 0)),
            second=int(raw.get("second", 0)),
            timestamp=str(raw.get("timestamp", "")),
            event_type=str(raw.get("type", {}).get("name", "")),
            team=str(raw.get("team", {}).get("name", "")),
            player=str(raw.get("player", {}).get("name", "")),
            player_id=raw.get("player", {}).get("id") if raw.get("player") else None,
            position=str(pos.get("name", "")),
            location_x=float(loc_x) if loc_x is not None else None,
            location_y=float(loc_y) if loc_y is not None else None,
            outcome=outcome_map.get(shot.get("outcome", {}).get("name", ""), ""),
            possession=int(raw.get("possession", 0)) if raw.get("possession") is not None else None,
            xg=float(shot.get("statsbomb_xg", 0)) if shot.get("statsbomb_xg") is not None else None,
            shot_type=str(shot.get("type", {}).get("name", "")),
            shot_body_part=str(shot.get("body_part", {}).get("name", "")),
            pass_target=str(pass_.get("recipient", {}).get("name", ""))
            if pass_.get("recipient")
            else "",
            raw=raw,
        )

    async def get_shots(self, match_id: int) -> list[SbEvent]:
        events = await self.get_events(match_id)
        return [e for e in events if e.event_type == "Shot"]

    async def get_team_events(self, match_id: int, team_name: str) -> list[SbEvent]:
        events = await self.get_events(match_id)
        needle = team_name.lower()
        return [e for e in events if e.team.lower() == needle]

    async def get_player_events(self, match_id: int, player_name: str) -> list[SbEvent]:
        events = await self.get_events(match_id)
        needle = player_name.lower()
        return [e for e in events if e.player.lower() == needle]

    # ------------------------------------------------------------------
    # Lineups
    # ------------------------------------------------------------------

    async def get_lineups(self, match_id: int) -> list[SbLineup]:
        path = f"lineups/{match_id}.json"
        data = await self._get(path)
        if not data or not isinstance(data, list):
            return []
        result: list[SbLineup] = []
        for entry in data:
            team = entry.get("team", {}) or {}
            players = []
            for p in entry.get("players", []) or []:
                player = p.get("player", {}) or {}
                positions = [pp.get("position", "") for pp in p.get("positions", [])]
                players.append(
                    {
                        "name": player.get("name", ""),
                        "player_id": player.get("id"),
                        "jersey_number": player.get("jersey_number"),
                        "country": player.get("country", {}).get("name", ""),
                        "positions": positions,
                    }
                )
            result.append(
                SbLineup(
                    team_name=team.get("name", ""),
                    team_id=team.get("id", 0),
                    players=players,
                )
            )
        return result

    # ------------------------------------------------------------------
    # 360 data
    # ------------------------------------------------------------------

    async def get_three_sixty(self, match_id: int) -> list[dict]:
        path = f"three-sixty/{match_id}.json"
        data = await self._get(path)
        if not data or not isinstance(data, list):
            return []
        return data

    # ------------------------------------------------------------------
    # Team search
    # ------------------------------------------------------------------

    async def get_raw_events(self, match_id: int) -> list[dict]:
        """Fetch raw StatsBomb event dicts (not SbEvent objects)."""
        path = f"events/{match_id}.json"
        data = await self._get(path)
        if not data or not isinstance(data, list):
            return []
        return data

    async def import_match_to_db(self, match_id: int, storage_service) -> int:
        """Fetch events from StatsBomb and import into local storage.

        Returns number of events imported. Requires an initialized storage_service.
        """
        raw_events = await self.get_raw_events(match_id)
        if not raw_events:
            return 0
        from kawkab.services.data_import_service import DataImportService

        converter = DataImportService(storage_service)
        events = []
        for item in raw_events:
            event = converter._statsbomb_to_event(item, str(match_id))
            if event:
                events.append(event)
        count = await storage_service.save_events_bulk(match_id, events)
        logger.info(f"Imported {count}/{len(events)} StatsBomb events from match {match_id} to DB")
        return count

    async def search_team_matches(self, team_name: str) -> list[SbMatch]:
        """Find all matches for a team across all competitions/seasons."""
        comps = await self.get_competitions()
        results: list[SbMatch] = []
        needle = team_name.lower()
        for c in comps:
            matches = await self.get_matches(c.competition_id, c.season_id)
            for m in matches:
                if needle in m.home_team.lower() or needle in m.away_team.lower():
                    results.append(m)
        return results
