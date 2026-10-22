from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

WYSCOUT_RATE_LIMIT = 5


@dataclass
class WyscoutMatch:
    match_id: str
    competition_id: str
    season_id: str
    home_team: str
    away_team: str
    home_score: int | None = None
    away_score: int | None = None
    match_date: str = ""


@dataclass
class WyscoutEvent:
    event_id: str
    match_id: str
    team_id: str
    player_id: str
    event_type: str
    minute: int = 0
    second: int = 0
    x: float = 0.0
    y: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0
    tags: list[str] = field(default_factory=list)


class WyscoutImporter:
    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key
        self._last_request_time = 0.0
        self._request_window: list[float] = []

    def _check_rate_limit(self) -> None:
        now = time.monotonic()
        self._request_window = [t for t in self._request_window if now - t < 60.0]
        if len(self._request_window) >= WYSCOUT_RATE_LIMIT:
            wait = 60.0 - (now - self._request_window[0])
            if wait > 0:
                logger.info(f"Rate limit reached, waiting {wait:.1f}s")
                time.sleep(wait)
                self._request_window = [
                    t for t in self._request_window if time.monotonic() - t < 60.0
                ]
        self._request_window.append(time.monotonic())

    def import_match(
        self, match_id: str
    ) -> tuple[WyscoutMatch | None, list[WyscoutEvent], list[dict]]:
        raise NotImplementedError(
            "Wyscout API key required for live data — use import_local(path) for offline files"
        )

    def import_competition(self, competition_id: str, season_id: str) -> list[WyscoutMatch]:
        raise NotImplementedError(
            "Wyscout API key required for live data — use import_local(path) for offline files"
        )

    def _parse_events(self, raw_json: dict) -> list[WyscoutEvent]:
        if not raw_json or not isinstance(raw_json, dict):
            return []
        events: list[WyscoutEvent] = []
        for item in raw_json.get("events", raw_json.get("match_events", [])):
            if not isinstance(item, dict):
                continue
            try:
                event = WyscoutEvent(
                    event_id=str(item.get("id", "")),
                    match_id=str(item.get("matchId", "")),
                    team_id=str(item.get("teamId", "")),
                    player_id=str(item.get("playerId", "")),
                    event_type=str(item.get("eventName", "")),
                    minute=int(item.get("minute", 0)),
                    second=int(item.get("second", 0)),
                    x=self._safe_coord(item, "x"),
                    y=self._safe_coord(item, "y"),
                    end_x=self._safe_coord(item, "endX"),
                    end_y=self._safe_coord(item, "endY"),
                    tags=[str(t) for t in item.get("tags", []) if isinstance(t, (str, int))],
                )
                events.append(event)
            except Exception as exc:
                logger.warning(f"Skipping Wyscout event due to parse error: {exc}")
        return events

    def _parse_lineups(self, raw_json: dict) -> list[dict]:
        if not raw_json or not isinstance(raw_json, dict):
            return []
        lineups: list[dict] = []
        for entry in raw_json.get("lineups", raw_json.get("match_lineups", [])):
            if not isinstance(entry, dict):
                continue
            try:
                team_id = str(entry.get("teamId", ""))
                players = []
                for p in entry.get("players", []):
                    if isinstance(p, dict):
                        players.append(
                            {
                                "player_id": str(p.get("playerId", "")),
                                "name": str(p.get("name", "")),
                                "shirt_number": int(p.get("shirtNumber", 0))
                                if p.get("shirtNumber") is not None
                                else 0,
                                "position": str(p.get("position", "")),
                            }
                        )
                lineups.append(
                    {
                        "team_id": team_id,
                        "team_name": str(entry.get("teamName", "")),
                        "formation": str(entry.get("formation", "")),
                        "players": players,
                    }
                )
            except Exception as exc:
                logger.warning(f"Skipping Wyscout lineup due to parse error: {exc}")
        return lineups

    def import_local(
        self, path: str | Path
    ) -> tuple[WyscoutMatch | None, list[WyscoutEvent], list[dict]]:
        try:
            with open(str(path), encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error(f"Failed to read Wyscout file {path}: {exc}")
            return None, [], []
        if not isinstance(data, dict):
            logger.warning(f"Unexpected Wyscout file format in {path}")
            return None, [], []
        match = self._parse_match(data)
        events = self._parse_events(data)
        lineups = self._parse_lineups(data)
        return match, events, lineups

    def _parse_match(self, raw: dict) -> WyscoutMatch | None:
        try:
            match_data = raw.get("match", raw.get("match_info", raw))
            if not isinstance(match_data, dict):
                match_data = raw
            return WyscoutMatch(
                match_id=str(match_data.get("matchId", match_data.get("id", ""))),
                competition_id=str(match_data.get("competitionId", "")),
                season_id=str(match_data.get("seasonId", "")),
                home_team=str(
                    match_data.get("home", match_data.get("homeTeam", {})).get("name", "")
                ),
                away_team=str(
                    match_data.get("away", match_data.get("awayTeam", {})).get("name", "")
                ),
                home_score=self._safe_int(match_data, "homeScore"),
                away_score=self._safe_int(match_data, "awayScore"),
                match_date=str(match_data.get("date", match_data.get("matchDate", ""))),
            )
        except Exception as exc:
            logger.warning(f"Failed to parse Wyscout match: {exc}")
            return None

    @staticmethod
    def _safe_coord(item: dict, key: str) -> float:
        val = item.get(key)
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _safe_int(item: dict, key: str) -> int | None:
        val = item.get(key)
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None
