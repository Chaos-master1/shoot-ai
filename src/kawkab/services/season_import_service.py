"""Season-scale bulk import — turn a directory of vendor match files into
an organized, deduplicated season with one call.

This is the season-workflow piece of the elite-club path: analysts do not
think in single files, they think in competitions and seasons (a top club
plays 50-60 matches; analysts may hold 400+ including opponents). Point
this service at a folder of StatsBomb event files and every match lands
in storage exactly once, tagged with competition/date/season, with a
vendor-id registry (migration 031) making re-runs idempotent.

Design notes:

    - Dedup key is ``(source='statsbomb', external_id=<file stem>)`` in
      ``matches_external_ids``. StatsBomb open-data files are named
      ``<match_id>.json`` so the stem is the vendor's own match id.
      Re-importing the same directory imports 0 new matches.
    - Non-event JSON files in the directory (lineages, lineups, competition
      metadata — anything whose root is not a non-empty JSON list) are
      skipped and counted, never fatal.
    - Optional per-file sidecar ``<stem>.meta.json`` (NOT imported as a
      match) carries ``{"competition": ..., "match_date": "YYYY-MM-DD",
      "season_id": 3}`` and overrides the bulk arguments for that match.
      StatsBomb's own ``match_date`` field on the first event is used when
      present and the sidecar is silent.
    - Throughput, honestly: events are committed per-row through the
      existing ``save_event`` path (WAL, ~10k commits/sec on typical dev
      hardware). A 50-match season (~1.5k events each) lands in minutes,
      not seconds. The bulk-insert shortcut was deliberately avoided: the
      existing bulk path rolls back the whole batch on any duplicate,
      which would change the import's per-event dedup semantics.
    - Failure-tolerant by design: one corrupt file logs a warning and is
      counted in ``failed``; the rest of the season still imports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SeasonImportService:
    """Bulk directory import with dedup, context tagging, and a per-match
    status list. Wraps StatsBombImportService for the per-match work."""

    def __init__(self, storage_service) -> None:
        self.storage = storage_service

    # ── public API ────────────────────────────────────────────────────────

    async def import_statsbomb_directory(
        self,
        directory: str | Path,
        *,
        competition: str | None = None,
        season_id: int | None = None,
        match_date: str | None = None,
        max_matches: int | None = None,
        on_file_done=None,
    ) -> dict[str, Any]:
        """Import every StatsBomb event file in ``directory``.

        Returns a summary dict::

            {
              "directory": ..., "total_files": N, "eligible": E,
              "imported": n1, "skipped_already": n2, "skipped_not_events": n3,
              "failed": n4, "matches": [ {"file", "status", "match_id",
              "match_name", "competition", "match_date", "error"?}, ... ]
            }

        ``max_matches`` caps how many *eligible event files* are imported
        this run (CI/smoke runs against the real 297-file corpus).
        ``on_file_done`` is an optional callback(summary_row) fired after
        each file lands (used by the UI bridge to emit live progress).
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"not a directory: {directory}")

        from kawkab.services.statsbomb_import_service import StatsBombImportService

        sb = StatsBombImportService(self.storage)

        json_files = sorted(
            p for p in directory.glob("*.json") if not p.name.endswith(".meta.json")
        )
        summary: dict[str, Any] = {
            "directory": str(directory),
            "total_files": len(json_files),
            "eligible": 0,
            "imported": 0,
            "skipped_already": 0,
            "skipped_not_events": 0,
            "failed": 0,
            "matches": [],
        }
        rows = summary["matches"]

        for path in json_files:
            if max_matches is not None and summary["imported"] >= max_matches:
                break

            external_id = path.stem
            existing = await self.storage.get_match_by_external_id("statsbomb", external_id)
            if existing is not None:
                summary["skipped_already"] += 1
                rows.append(
                    {
                        "file": path.name,
                        "status": "skipped_already",
                        "match_id": existing,
                    }
                )
                continue

            # Root-type gate: event files are a non-empty JSON list.
            # Lineages/lineups/competition metadata are dicts or empty —
            # skip without touching the import path.
            try:
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                summary["failed"] += 1
                rows.append(
                    {
                        "file": path.name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                logger.warning(f"season import: unreadable file {path.name}: {exc}")
                continue
            if not isinstance(raw, list) or not raw:
                summary["skipped_not_events"] += 1
                rows.append(
                    {
                        "file": path.name,
                        "status": "skipped_not_events",
                    }
                )
                continue
            summary["eligible"] += 1

            # Per-match context: sidecar > StatsBomb field > bulk args.
            meta = self._load_sidecar(path)
            comp = meta.get("competition") or competition
            s_id = meta.get("season_id", season_id)
            m_date = meta.get("match_date") or self._sb_match_date(raw) or match_date

            try:
                match_summary = await sb.import_match(path)
                match_id = match_summary["match_id"]
                await self.storage.update_match_context(
                    match_id,
                    match_date=m_date,
                    competition=comp,
                    season_id=int(s_id) if s_id is not None else None,
                )
                await self.storage.register_match_external_id(match_id, "statsbomb", external_id)
            except Exception as exc:  # one bad file never kills the season
                summary["failed"] += 1
                rows.append(
                    {
                        "file": path.name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                logger.warning(f"season import: {path.name} failed: {exc}")
                continue

            summary["imported"] += 1
            row = {
                "file": path.name,
                "status": "imported",
                "match_id": match_id,
                "match_name": match_summary.get("match_name"),
                "competition": comp,
                "match_date": m_date,
            }
            rows.append(row)
            if on_file_done is not None:
                try:
                    on_file_done(row)
                except Exception as exc:  # progress must never kill imports
                    logger.warning(f"season import: progress callback failed: {exc}")

        logger.info(
            "season import: %s -> imported=%d skipped_already=%d skipped_not_events=%d failed=%d",
            directory,
            summary["imported"],
            summary["skipped_already"],
            summary["skipped_not_events"],
            summary["failed"],
        )
        return summary

    async def import_single_file(
        self,
        file_path: str | Path,
        *,
        competition: str | None = None,
        season_id: int | None = None,
        match_date: str | None = None,
    ) -> dict[str, Any]:
        """One file through the same dedup + context pipeline as the
        directory walk (shared by the CLI's file mode)."""
        file_path = Path(file_path)
        if not file_path.is_file():
            raise ValueError(f"not a file: {file_path}")

        external_id = file_path.stem
        existing = await self.storage.get_match_by_external_id("statsbomb", external_id)
        if existing is not None:
            return {
                "file": file_path.name,
                "status": "skipped_already",
                "match_id": existing,
            }

        from kawkab.services.statsbomb_import_service import StatsBombImportService

        meta = self._load_sidecar(file_path)
        sb = StatsBombImportService(self.storage)
        match_summary = await sb.import_match(file_path)
        match_id = match_summary["match_id"]
        await self.storage.update_match_context(
            match_id,
            match_date=meta.get("match_date") or match_date,
            competition=meta.get("competition") or competition,
            season_id=(
                int(str(meta.get("season_id", season_id)))
                if meta.get("season_id", season_id) is not None
                else None
            ),
        )
        await self.storage.register_match_external_id(match_id, "statsbomb", external_id)
        return {
            "file": file_path.name,
            "status": "imported",
            "match_id": match_id,
            "match_name": match_summary.get("match_name"),
        }

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    @staticmethod
    def _load_sidecar(path: Path) -> dict[str, Any]:
        """``<stem>.meta.json`` beside the event file, or {}."""
        sidecar = path.with_name(f"{path.stem}.meta.json")
        if not sidecar.is_file():
            return {}
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"unreadable sidecar {sidecar.name}: {exc}")
            return {}

    @staticmethod
    def _sb_match_date(raw: list[dict]) -> str | None:
        """StatsBomb sometimes stamps ``match_date`` on the first event;
        use it when present, else None."""
        first = raw[0] if raw else {}
        date = first.get("match_date") if isinstance(first, dict) else None
        return date if isinstance(date, str) and date else None
