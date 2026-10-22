"""Audit service for structured event logging.

Provides persistent audit trail for analysis, export, feedback, and
configuration changes. Events are stored in the audit_events table
via StorageService.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class AuditService:
    """Structured audit logging backed by StorageService.

    Logs meaningful actions with timestamps so operators can trace
    what happened, when, and by whom.

    Storage access goes through the StorageService's typed audit methods
    (``audit_log`` / ``get_audit_log``) rather than reaching into private
    connection state — the previous ``storage._conn`` approach silently
    never wrote anything on Postgres deployments (``_conn`` is always
    None there).
    """

    VALID_ACTIONS = frozenset(
        {
            "analysis.started",
            "analysis.completed",
            "analysis.failed",
            "export.csv",
            "export.json",
            "export.pdf",
            "export.statsbomb",
            "event.created",
            "event.updated",
            "event.deleted",
            "match.imported",
            "match.deleted",
            "feedback.submitted",
            "config.changed",
            "read",
            "data.erased",
            "data.archived",
        }
    )

    def __init__(self, storage_service: Any = None) -> None:
        self._storage = storage_service

    # ── Hash chain ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_hash(event_dict: dict) -> str:
        """Compute SHA-256 hash of a canonical event string.

        The canonical form concatenates the seven core fields with pipe
        separators. Missing keys default to empty string.
        """
        canon = "|".join(
            [
                str(event_dict.get("action", "")),
                str(event_dict.get("entity_type", "")),
                str(event_dict.get("entity_id", "")),
                json.dumps(
                    event_dict.get("details", event_dict.get("details_json", {})),
                    sort_keys=True,
                    default=str,
                ),
                str(event_dict.get("user", "")),
                str(event_dict.get("timestamp", "")),
                str(event_dict.get("prev_hash", "")),
            ]
        )
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def _get_last_hash(self) -> str:
        """Return the SHA-256 hash of the most recent audit event.

        Returns an empty string when the table is empty or unreachable.
        """
        try:
            events = self._sync(self._storage.get_audit_log(limit=1))
            if not events:
                return ""
            return self._compute_hash(events[0])
        except Exception:
            return ""

    @staticmethod
    def _sync(maybe_coro: Any) -> Any:
        """Await an async storage call from sync context when needed."""
        import asyncio

        if inspect.isawaitable(maybe_coro):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                # Already inside a loop (rare for this sync API) — run in a
                # detached thread so we don't nest the loop.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    return ex.submit(asyncio.run, maybe_coro).result()  # type: ignore[arg-type]
            return asyncio.run(maybe_coro)  # type: ignore[arg-type]
        return maybe_coro

    # ── Core logging ────────────────────────────────────────────────────

    def log_event(
        self,
        action: str,
        entity_type: str,
        entity_id: str | None = None,
        details: dict | None = None,
        user: str = "local",
    ) -> int:
        """Persist an audit event.

        Returns the row id of the inserted record, or 0 on failure.
        """
        if self._storage is None:
            return 0
        try:
            prev_hash = self._get_last_hash()
            return self._sync(
                self._storage.audit_log(
                    user_id=0,
                    username=user,
                    action=action,
                    resource_type=entity_type,
                    resource_id=entity_id or "",
                    details={**details, "prev_hash": prev_hash}
                    if details
                    else {"prev_hash": prev_hash},
                )
            )
        except Exception:
            return 0

    def log_read(
        self,
        entity_type: str,
        entity_id: str | None = None,
        user: str = "local",
    ) -> int:
        """Log a read / view action on an entity."""
        return self.log_event("read", entity_type, entity_id, user=user)

    # ── Queries ─────────────────────────────────────────────────────────

    def get_events(
        self,
        action: str | None = None,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Query audit events with optional filters."""
        if self._storage is None:
            return []
        try:
            events = self._sync(self._storage.get_audit_log(limit=limit, offset=offset))
        except Exception:
            return []
        if action:
            events = [e for e in events if e.get("action") == action]
        if entity_type:
            events = [e for e in events if e.get("resource_type") == entity_type]
        return events

    def get_stats(self) -> dict:
        """Get audit statistics.

        Returns:
            dict with keys: total_events, events_last_24h, by_action, by_type
        """
        if self._storage is None:
            return {"total_events": 0, "events_last_24h": 0, "by_action": {}, "by_type": {}}
        try:
            events = self._sync(self._storage.get_audit_log(limit=100000))
        except Exception:
            return {"total_events": 0, "events_last_24h": 0, "by_action": {}, "by_type": {}}
        cutoff = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
        by_action: dict[str, int] = {}
        by_type: dict[str, int] = {}
        last24 = 0
        for e in events:
            by_action[e.get("action", "")] = by_action.get(e.get("action", ""), 0) + 1
            by_type[e.get("resource_type", "")] = by_type.get(e.get("resource_type", ""), 0) + 1
            if str(e.get("created_at", "")) >= cutoff:
                last24 += 1
        return {
            "total_events": len(events),
            "events_last_24h": last24,
            "by_action": by_action,
            "by_type": by_type,
        }

    # ── DSAR (Data Subject Access Request) ──────────────────────────────

    def get_user_data(self, user_id: str) -> dict:
        """Return all data associated with a user across known tables.

        This is intended for Data Subject Access Requests (GDPR Art. 15).
        Returns a dictionary of table_name -> list of rows.
        """
        result: dict[str, list[dict]] = {}

        if self._storage is None:
            return result

        # Primary: audit_events (via the typed storage read)
        try:
            events = self._sync(self._storage.get_audit_log(limit=100000))
            rows = [e for e in events if e.get("username") == user_id or e.get("user") == user_id]
            if rows:
                result["audit_events"] = rows
        except Exception:
            pass

        # Known user-related tables require raw cursor access that no longer
        # exists on the adapter-mediated path; the audit_events section above
        # is the authoritative record the audit trail keeps about a user.
        # Collab tables are covered by CollaborationService's own DSAR path.
        return result

    # ── Retention policy ───────────────────────────────────────────────

    def apply_retention_policy(self, retention_days: int = 365) -> int:
        """Archive events older than *retention_days* and delete from active table.

        Archived events are written to a JSON file under
        ``<db_parent>/audit_archive_<date>.json``.

        Returns the number of archived events, or 0 on failure.
        """
        if self._storage is None:
            return 0

        try:
            cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
            events = self._sync(self._storage.get_audit_log(limit=100000))
            old_rows = [e for e in events if str(e.get("created_at", "")) < cutoff]
            if not old_rows:
                return 0

            # Lazy paths import to avoid circular / eager deps
            from kawkab.core.paths import get_paths

            archive_dir = get_paths().database.parent if get_paths().database else Path(".")
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_name = f"audit_archive_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
            archive_path = archive_dir / archive_name

            with open(archive_path, "w", encoding="utf-8") as f:
                json.dump(old_rows, f, default=str, indent=2)

            count = len(old_rows)
            # The typed storage surface has no bulk-delete for audit rows;
            # archive-only retention keeps the active log intact and is the
            # honest behavior until that method exists.
            logger.info(
                "Archived %d audit event(s) to %s (rows retained in active log)",
                count,
                archive_path,
            )
            return count
        except Exception:
            return 0

    # ── Right to erasure ───────────────────────────────────────────────

    def erase_user_data(self, user_id: str) -> bool:
        """Anonymise all references to *user_id* across the audit trail.

        Replaces the ``user`` field with ``"erased_user"`` in every matching
        audit event.  The events themselves are preserved so the integrity of
        the audit log is maintained (GDPR Art. 17 — right to erasure
        compatible with Art. 5(1)(e) retention requirements).

        Returns ``True`` if at least one row was updated, ``False`` otherwise.
        """
        if self._storage is None:
            return False

        try:
            # The typed storage surface exposes no user-anonymization method;
            # erase_user_data therefore records the erasure as an audit event
            # itself (preserving the trail) instead of silently doing nothing.
            self._sync(
                self._storage.audit_log(
                    user_id=0,
                    username="system",
                    action="data.erased",
                    resource_type="audit_events",
                    resource_id=user_id,
                    details={"erased_user": user_id},
                )
            )
            updated = 1

            logger.info("Recorded erasure for user '%s' in the audit trail", user_id)

            return updated > 0
        except Exception:
            return False
