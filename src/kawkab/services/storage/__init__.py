"""Shared storage helpers.

This package once hosted the "specialised storage classes" split out of
StorageService (MatchStorage, EventStorage, ...). Those classes had zero
production constructors — only tests exercised them, and they silently
drifted from the real adapters — so they were removed. Both backends
(StorageService, PostgresStorageAdapter) are the single source of truth
for storage behavior; only the shared row-parsing helper survives here.
"""

from kawkab.services.storage.base import parse_metadata_json

__all__ = ["parse_metadata_json"]
