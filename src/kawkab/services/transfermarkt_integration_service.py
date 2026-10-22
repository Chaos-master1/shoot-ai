"""Transfermarkt integration — market values, squad data, and player search.

Honesty note: there is currently NO live Transfermarkt data provider wired
in (Transfermarkt offers no free public API, and scraping is against their
terms of service). Earlier versions of this service returned hardcoded
demo data ("Demo FC", "Academy FC", invented market values) for every
query — fabricated data rendered in a live recruitment UI, indistinguishable
from real scouting output. This service now returns explicit
``available: False`` / ``data_available: False`` states instead, so the UI
can render "no provider configured" honestly.

Every result dict carries a ``provenance`` field:

- ``"cache"``        — previously imported by the user (trusted as their own input)
- ``"none"``         — no data exists; the caller must show an honest empty state

Wire a real provider by implementing :meth:`_provider_search` /
:meth:`_provider_details` / :meth:`_provider_squad` in a subclass and
passing it to the constructor; nothing else needs to change.
"""

from __future__ import annotations

import contextlib
import json
import os
from typing import Any

from kawkab.core import paths as kawkab_paths
from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class TransfermarktIntegrationService:
    """Player/club market data access with honest no-provider semantics."""

    def __init__(self, cache_dir: str | None = None, provider: Any = None) -> None:
        # Per-user app-data cache, NOT the source tree (same reason as
        # OpponentDatabaseService: runtime writes used to land in src/data/).
        if cache_dir is None:
            cache_dir = str(kawkab_paths.get_paths().appdata / "data" / "transfermarkt")
        self._cache_dir = cache_dir
        self._cache: dict[str, Any] = {}
        self._provider = provider  # optional real data provider (none ships by default)
        self._load_cache()

    # ── cache plumbing (unchanged semantics) ────────────────────────────

    def _cache_path(self, key: str) -> str:
        os.makedirs(self._cache_dir, exist_ok=True)
        return os.path.join(self._cache_dir, f"{key}.json")

    def _load_cache(self) -> None:
        cache_file = self._cache_path("_index")
        try:
            if os.path.exists(cache_file):
                with open(cache_file, encoding="utf-8") as f:
                    self._cache = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load TM cache: {e}")

    def _save_cache(self) -> None:
        with open(self._cache_path("_index"), "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2, ensure_ascii=False, default=str)

    def _get_cached(self, key: str) -> Any:
        return self._cache.get(key)

    def _set_cached(self, key: str, data: Any) -> None:
        self._cache[key] = data
        self._save_cache()

    # ── provider hooks (no live provider ships; subclasses may add one) ─

    def _provider_search(self, name: str) -> list[dict] | None:
        return None

    def _provider_details(self, player_id: int) -> dict | None:
        return None

    def _provider_squad(self, club_name: str) -> list[dict] | None:
        return None

    # ── public API (honest states) ──────────────────────────────────────

    def search_player(self, name: str) -> list[dict]:
        """Search players. Returns cached (user-imported) hits, provider
        results when a provider is configured, else an honest empty list."""
        cached = self._get_cached(f"search:{name.lower()}")
        if cached:
            return cached

        if self._provider is not None:
            results = self._provider_search(name)
            if results:
                for r in results:
                    r["provenance"] = "provider"
                self._set_cached(f"search:{name.lower()}", results)
                return results

        # No provider, no cache: write the explicit empty state to the
        # cache index (preserves the cache-persist test contract) and
        # return it. Never fabricate players or market values.
        empty: list[dict] = []
        self._set_cached(f"search:{name.lower()}", empty)
        return empty

    def provider_status(self) -> dict:
        """Honest capability report for the UI."""
        return {
            "available": self._provider is not None,
            "provider": type(self._provider).__name__ if self._provider else None,
            "note": (
                "Live provider configured."
                if self._provider
                else "No live Transfermarkt provider configured. Transfermarkt offers no free public API; import data manually or configure a provider."
            ),
        }

    def get_player_details(self, player_id: int) -> dict:
        """Player details. Returns an explicit no-data state — never an
        invented profile (the old path fabricated stats and a market-value
        history for any ID)."""
        cached = self._get_cached(f"player:{player_id}")
        if cached:
            return cached

        if self._provider is not None:
            details = self._provider_details(player_id)
            if details:
                details["provenance"] = "provider"
                self._set_cached(f"player:{player_id}", details)
                return details

        no_data: dict[str, Any] = {
            "id": player_id,
            "data_available": False,
            "provenance": "none",
            "note": "No data on file for this player. Import player data or configure a live provider.",
        }
        self._set_cached(f"player:{player_id}", no_data)
        return no_data

    def get_club_squad(self, club_name: str) -> list[dict]:
        """Club squad. Returns an honest empty list when nothing is on
        file — the old path returned a synthetic 6-player squad for any
        club name."""
        cached = self._get_cached(f"squad:{club_name.lower()}")
        if cached:
            return cached

        if self._provider is not None:
            squad = self._provider_squad(club_name)
            if squad:
                for p in squad:
                    p["provenance"] = "provider"
                self._set_cached(f"squad:{club_name.lower()}", squad)
                return squad

        empty: list[dict] = []
        self._set_cached(f"squad:{club_name.lower()}", empty)
        return empty

    def get_market_value(self, player_name: str) -> dict:
        results = self.search_player(player_name)
        if results:
            p = results[0]
            return {
                "name": p.get("name", player_name),
                "value": p.get("market_value", 0),
                "currency": "EUR",
                "provenance": p.get("provenance", "cache"),
            }
        return {
            "name": player_name,
            "value": 0,
            "currency": "EUR",
            "data_available": False,
            "provenance": "none",
        }

    def clear_cache(self) -> None:
        self._cache = {}
        self._save_cache()
        for f in os.listdir(self._cache_dir):
            if f.endswith(".json"):
                with contextlib.suppress(Exception):
                    os.remove(os.path.join(self._cache_dir, f))
