from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_HOTKEYS = {
    "1": "goal",
    "2": "shot",
    "3": "pass",
    "4": "tackle",
    "5": "foul",
    "6": "corner",
    "7": "save",
    "8": "substitution",
    "9": "offside",
    "0": "card_yellow",
    "q": "throw_in",
    "w": "free_kick",
    "e": "cross",
    "r": "dribble",
    "t": "clearance",
    "z": "card_red",
    "x": "interception",
    "c": "foul_drawn",
    "v": "possession_change",
    "b": "missed_shot",
    "n": "blocked_shot",
    "m": "counter_attack",
}


@dataclass
class LiveTag:
    id: int = 0
    event_type: str = ""
    timestamp_s: float = 0.0
    team: str = ""
    player_track_id: int = 0
    period: int = 1
    x: float | None = None
    y: float | None = None
    notes: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.event_type,
            "t": round(self.timestamp_s, 1),
            "team": self.team,
            "player_id": self.player_track_id,
            "period": self.period,
            "x": self.x,
            "y": self.y,
            "notes": self.notes,
        }


@dataclass
class LiveMatchStats:
    events_by_type: dict = field(default_factory=Counter)
    home_goals: int = 0
    away_goals: int = 0
    home_shots: int = 0
    away_shots: int = 0
    home_possession_pct: float = 50.0
    tags_count: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self):
        return {
            "events_by_type": dict(self.events_by_type),
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "home_shots": self.home_shots,
            "away_shots": self.away_shots,
            "home_possession_pct": round(self.home_possession_pct, 1),
            "tags_count": self.tags_count,
            "elapsed_s": round(self.elapsed_seconds, 1),
        }


class LiveTaggingService:
    def __init__(self):
        self._tags: list[LiveTag] = []
        self._next_id = 1
        self._session_active = False
        self._session_start: float = 0.0
        self._home_team = "Home"
        self._away_team = "Away"
        self._current_period = 1
        self._stats = LiveMatchStats()
        self._hotkeys = dict(DEFAULT_HOTKEYS)

    def start_session(self, home_team: str = "Home", away_team: str = "Away") -> str:
        try:
            self._tags = []
            self._next_id = 1
            self._session_active = True
            self._session_start = time.time()
            self._home_team = home_team
            self._away_team = away_team
            self._current_period = 1
            self._stats = LiveMatchStats()
            return json.dumps({"ok": True, "message": "Live tagging session started"})
        except Exception as e:
            logger.error(f"start_session failed: {e}")
            return json.dumps({"error": str(e)})

    def stop_session(self) -> str:
        try:
            self._session_active = False
            total_tags = len(self._tags)
            return json.dumps(
                {
                    "ok": True,
                    "total_tags": total_tags,
                    "message": f"Session stopped. {total_tags} tags recorded.",
                }
            )
        except Exception as e:
            logger.error(f"stop_session failed: {e}")
            return json.dumps({"error": str(e)})

    def tag_event(
        self,
        event_type: str,
        team: str = "",
        player_id: int = 0,
        notes: str = "",
        x: float | None = None,
        y: float | None = None,
    ) -> str:
        try:
            if not self._session_active:
                return json.dumps({"error": "No active session"})
            tag = LiveTag(
                id=self._next_id,
                event_type=event_type,
                timestamp_s=time.time() - self._session_start,
                team=team,
                player_track_id=player_id,
                period=self._current_period,
                x=x,
                y=y,
                notes=notes,
            )
            self._next_id += 1
            self._tags.append(tag)
            self._stats.events_by_type[event_type] += 1
            self._stats.tags_count = len(self._tags)
            if event_type == "goal":
                if team == self._home_team:
                    self._stats.home_goals += 1
                elif team == self._away_team:
                    self._stats.away_goals += 1
            if event_type == "shot":
                if team == self._home_team:
                    self._stats.home_shots += 1
                elif team == self._away_team:
                    self._stats.away_shots += 1
            return json.dumps({"tag": tag.to_dict(), "ok": True})
        except Exception as e:
            logger.error(f"tag_event failed: {e}")
            return json.dumps({"error": str(e)})

    def set_period(self, period: int) -> str:
        try:
            self._current_period = period
            return json.dumps({"ok": True, "period": period})
        except Exception as e:
            logger.error(f"set_period failed: {e}")
            return json.dumps({"error": str(e)})

    def get_stats(self) -> str:
        try:
            if self._session_active:
                self._stats.elapsed_seconds = time.time() - self._session_start
            return json.dumps({"stats": self._stats.to_dict()})
        except Exception as e:
            logger.error(f"get_stats failed: {e}")
            return json.dumps({"error": str(e)})

    def get_all_tags(self) -> str:
        try:
            return json.dumps({"tags": [t.to_dict() for t in self._tags], "total": len(self._tags)})
        except Exception as e:
            logger.error(f"get_all_tags failed: {e}")
            return json.dumps({"error": str(e)})

    def clear_tags(self) -> str:
        try:
            self._tags = []
            self._stats = LiveMatchStats()
            return json.dumps({"ok": True})
        except Exception as e:
            logger.error(f"clear_tags failed: {e}")
            return json.dumps({"error": str(e)})

    def get_hotkeys(self) -> str:
        return json.dumps({"hotkeys": self._hotkeys})

    def export_tags(self) -> str:
        try:
            return json.dumps(
                {
                    "tags": [t.to_dict() for t in self._tags],
                    "stats": self._stats.to_dict(),
                    "home_team": self._home_team,
                    "away_team": self._away_team,
                    "total": len(self._tags),
                }
            )
        except Exception as e:
            logger.error(f"export_tags failed: {e}")
            return json.dumps({"error": str(e)})
