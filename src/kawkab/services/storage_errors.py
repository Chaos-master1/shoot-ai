"""Typed storage errors — A1 of the elite plan ("nothing lies").

History: storage methods swallowed every DB failure into 0/[]/False.
The proven damage: save_coding_tag's FK violation surfaced to the UI as
"Tag rejected: event_type/tag_type required" — the user was told their
input was wrong when the real cause was a missing match row (CI failure,
2026-09-19), and a Postgres deployment once returned empty data for every
call because the adapter's no-pool fallback silently fed [] to all reads.

Contract (applied cluster-by-cluster, coding tags first):

- "Not initialized / not connected" raises StorageNotInitialized — the
  caller must know the operation never ran.
- "The database refused / failed the operation" raises StorageWriteError
  or StorageReadError with the method name as context. Writes raise;
  reads raise too — an empty list is reserved for legitimate no-rows.
- "Rejected input" (invalid tag dict, no-op update) keeps the legacy
  falsy return so existing callers distinguish it from failure —
  rejection is a *contract*, failure is an *exception*.
"""

from __future__ import annotations

import sqlite3


class StorageError(Exception):
    """Base class for all storage honesty errors."""


class StorageNotInitializedError(StorageError):
    """The storage backend has no usable connection (initialize/close)."""

    def __init__(self, operation: str) -> None:
        self.operation = operation
        super().__init__(
            f"storage not initialized: {operation} called with no open connection "
            "(service not initialized, already closed, or pool never came up)"
        )


class StorageWriteError(StorageError):
    """A write operation failed at the database level."""

    def __init__(self, operation: str, cause: Exception | str) -> None:
        self.operation = operation
        self.cause = cause
        super().__init__(f"{operation} failed: {cause}")


class StorageDuplicateError(StorageWriteError):
    """A write hit a UNIQUE/PK constraint — the row already exists.

    Deliberately a *subclass* of StorageWriteError: callers that treat any
    failed write as failure keep catching duplicates unchanged, while
    callers that treat duplicates as an expected outcome (import dedup,
    idempotent re-imports) can catch only this. Duplicates are a *contract
    outcome*, not a malfunction — raising them as opaque failures here once
    broke the StatsBomb importer's designed skip path.
    """


def is_duplicate_violation(exc: Exception) -> bool:
    """Classify a driver exception as a UNIQUE/PK constraint violation.

    Uses the driver's structured code when available (sqlite extended
    result codes 1555/2067, Postgres SQLSTATE 23505) and falls back to the
    message text only for drivers that expose neither. FK violations must
    NOT classify as duplicates — they stay hard StorageWriteErrors.
    """
    if isinstance(exc, sqlite3.IntegrityError) and getattr(exc, "sqlite_errorcode", None) in (
        1555,
        2067,
    ):
        return True
    sqlstate = getattr(exc, "sqlstate", None) or getattr(exc, "pgcode", None)
    if sqlstate == "23505":
        return True
    text = str(exc).lower()
    return "unique constraint" in text or "duplicate key" in text


class StorageReadError(StorageError):
    """A read operation failed at the database level."""

    def __init__(self, operation: str, cause: Exception | str) -> None:
        self.operation = operation
        self.cause = cause
        super().__init__(f"{operation} failed: {cause}")
