from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ProviderMatch:
    match_id: str
    home_team: str
    away_team: str
    competition: str
    season: str
    date: datetime
    home_score: int | None = None
    away_score: int | None = None
    status: str = "scheduled"


@dataclass
class ProviderEvent:
    event_id: str
    match_id: str
    timestamp: float
    type: str
    team: str
    player: str = ""
    x: float = 0.0
    y: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0
    outcome: bool = True
    body_part: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class ProviderLineup:
    match_id: str
    team: str
    formation: str
    players: list[dict] = field(default_factory=list)


class BaseDataProvider(ABC):
    """Abstract base for all external data provider integrations."""

    @abstractmethod
    def get_provider_name(self) -> str: ...

    @abstractmethod
    async def search_matches(
        self,
        team: str | None = None,
        competition: str | None = None,
        season: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> list[ProviderMatch]: ...

    @abstractmethod
    async def get_match_events(self, match_id: str) -> list[ProviderEvent]: ...

    @abstractmethod
    async def get_match_lineups(self, match_id: str) -> list[ProviderLineup]: ...

    async def import_match_to_db(self, match_id: str, db: Any) -> int:
        """Default import: fetches events + lineups, inserts into storage DB."""
        events = await self.get_match_events(match_id)
        lineups = await self.get_match_lineups(match_id)
        internal_id = 0
        if events:
            internal_id = self._store_events(events, db)
        if lineups:
            self._store_lineups(lineups, db)
        return internal_id

    def _store_events(self, events: list[ProviderEvent], db: Any) -> int:
        return 0

    def _store_lineups(self, lineups: list[ProviderLineup], db: Any) -> None:  # noqa: B027  (optional hook)
        pass

    @abstractmethod
    def get_rate_limit_info(self) -> dict:
        return {"requests_per_min": 0, "daily_limit": 0}


class DataProviderRegistry:
    """Registry of available data providers."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseDataProvider] = {}

    def register(self, provider: BaseDataProvider) -> None:
        name = provider.get_provider_name()
        self._providers[name] = provider

    def get(self, name: str) -> BaseDataProvider | None:
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def unregister(self, name: str) -> None:
        self._providers.pop(name, None)
