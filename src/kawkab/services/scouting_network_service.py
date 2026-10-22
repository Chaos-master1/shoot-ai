"""Shared scouting network — discover players shared across the community, with anonymized ratings."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

from kawkab.core import paths as kawkab_paths
from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NetworkPlayer:
    id: str
    name: str
    position: str = ""
    age: int = 0
    club: str = ""
    league: str = ""
    nationality: str = ""
    estimated_value: float = 0.0
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    rating: float = 0.0
    scout_notes: str = ""
    tags: list[str] = field(default_factory=list)
    submitted_by: str = ""
    contact_info: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class ScoutingNetworkService:
    """Community-driven scouting network — share and discover player profiles."""

    def __init__(self, data_file: str | None = None) -> None:
        self._players: dict[str, NetworkPlayer] = {}
        # Per-user app-data file, NOT the source tree (same reason as
        # OpponentDatabaseService: runtime user data used to land inside
        # the repo under src/data/).
        if data_file is None:
            data_file = str(kawkab_paths.get_paths().appdata / "data" / "scouting_network.json")
        self._data_file = data_file
        self._load_data()

    def _load_data(self) -> None:
        os.makedirs(os.path.dirname(self._data_file), exist_ok=True)
        try:
            if os.path.exists(self._data_file):
                with open(self._data_file, encoding="utf-8") as f:
                    data = json.load(f)
                for p in data:
                    self._players[p["id"]] = NetworkPlayer(**p)
        except Exception as e:
            logger.warning(f"Failed to load scouting network: {e}")

    def _save_data(self) -> None:
        data = [vars(p) for p in self._players.values()]
        with open(self._data_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def search_players(
        self,
        query: str = "",
        position: str = "",
        min_age: int = 0,
        max_age: int = 99,
        league: str = "",
        min_rating: float = 0.0,
    ) -> list[dict]:
        query = query.lower().strip()
        results = []
        for p in self._players.values():
            if query and query not in p.name.lower() and query not in p.club.lower():
                continue
            if position and p.position.lower() != position.lower():
                continue
            if p.age < min_age or p.age > max_age:
                continue
            if league and league.lower() not in p.league.lower():
                continue
            if p.rating < min_rating:
                continue
            results.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "position": p.position,
                    "age": p.age,
                    "club": p.club,
                    "league": p.league,
                    "nationality": p.nationality,
                    "estimated_value": p.estimated_value,
                    "rating": p.rating,
                    "tags": p.tags,
                    "strengths": p.strengths[:3],
                    "scout_notes": p.scout_notes[:100] if p.scout_notes else "",
                }
            )
        results.sort(key=lambda x: float(x["rating"]), reverse=True)  # type: ignore[arg-type,return-value]
        return results

    def get_player(self, player_id: str) -> dict | None:
        p = self._players.get(player_id)
        if not p:
            return None
        return {
            "id": p.id,
            "name": p.name,
            "position": p.position,
            "age": p.age,
            "club": p.club,
            "league": p.league,
            "nationality": p.nationality,
            "estimated_value": p.estimated_value,
            "strengths": p.strengths,
            "weaknesses": p.weaknesses,
            "rating": p.rating,
            "scout_notes": p.scout_notes,
            "tags": p.tags,
            "submitted_by": p.submitted_by,
        }

    def add_player(
        self,
        name: str,
        position: str = "",
        club: str = "",
        league: str = "",
        rating: float = 0.0,
        strengths: list[str] | None = None,
        weaknesses: list[str] | None = None,
        scout_notes: str = "",
        submitted_by: str = "",
        tags: list[str] | None = None,
    ) -> dict:
        import uuid

        pid = str(uuid.uuid4())[:8]
        player = NetworkPlayer(
            id=pid,
            name=name,
            position=position,
            club=club,
            league=league,
            rating=rating,
            strengths=strengths or [],
            weaknesses=weaknesses or [],
            scout_notes=scout_notes,
            submitted_by=submitted_by,
            tags=tags or [],
        )
        self._players[pid] = player
        self._save_data()
        return {"id": pid, "name": name}

    def delete_player(self, player_id: str) -> bool:
        if player_id in self._players:
            del self._players[player_id]
            self._save_data()
            return True
        return False

    def list_tags(self) -> list[str]:
        tags = set()
        for p in self._players.values():
            tags.update(p.tags)
        return sorted(tags)

    def get_stats(self) -> dict:
        total = len(self._players)
        if total == 0:
            return {"total": 0, "avg_rating": 0, "by_position": {}, "by_league": {}}
        avg_rating = sum(p.rating for p in self._players.values()) / total
        by_position: dict[str, int] = {}
        by_league: dict[str, int] = {}
        for p in self._players.values():
            pos = p.position or "Unknown"
            by_position[pos] = by_position.get(pos, 0) + 1
            lg = p.league or "Unknown"
            by_league[lg] = by_league.get(lg, 0) + 1
        return {
            "total": total,
            "avg_rating": round(avg_rating, 1),
            "by_position": dict(sorted(by_position.items(), key=lambda x: -x[1])),
            "by_league": dict(sorted(by_league.items(), key=lambda x: -x[1])),
        }
