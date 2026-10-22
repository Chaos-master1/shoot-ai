"""Tests for the season-scale bulk import service (elite workflow phase).

Runs against a REAL SQLite StorageService with the REAL migration chain
(001-031) — same convention as test_statsbomb_import_service.py: synthetic
schemas have masked real bugs here before.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS = PROJECT_ROOT / "data" / "statsbomb_corpus"


@pytest.fixture()
def storage(tmp_path):
    """Real SQLite storage with the REAL 001-031 migration chain."""
    import sqlite3

    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.storage_service import StorageService

    scratch_db = tmp_path / "test_season.db"
    migrations_dir = PROJECT_ROOT / "src" / "kawkab" / "migrations"
    MigrationManager(scratch_db, migrations_dir).migrate()
    svc = StorageService()
    svc._db_path = scratch_db
    svc._conn = sqlite3.connect(str(scratch_db))
    svc._conn.row_factory = sqlite3.Row
    return svc


def _real_event_file():
    files = sorted(CORPUS.glob("*.json"))
    if not files:
        pytest.skip("statsbomb corpus not on this machine")
    return files[0]


def _copy_with_sidecar(src: Path, dest_dir: Path, name: str, meta: dict | None):
    dest = dest_dir / name
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    if meta is not None:
        sidecar = dest_dir / f"{dest.stem}.meta.json"
        sidecar.write_text(json.dumps(meta), encoding="utf-8")
    return dest


def _match_row(storage, match_id: int) -> dict:
    row = storage._conn.execute(
        "SELECT name, competition, season_id, match_date FROM matches WHERE id = ?",
        (match_id,),
    ).fetchone()
    return dict(row) if row else {}


def _external_ids(storage) -> set[tuple[str, str]]:
    rows = storage._conn.execute("SELECT source, external_id FROM matches_external_ids").fetchall()
    return {(r["source"], r["external_id"]) for r in rows}


class TestSeasonImport:
    @pytest.mark.asyncio
    async def test_dedup_second_run_imports_zero(self, storage, tmp_path):
        """The core elite-workflow guarantee: re-running a season import
        must not duplicate matches (external-ID registry dedup)."""
        from kawkab.services.season_import_service import SeasonImportService

        src = _real_event_file()
        season_dir = tmp_path / "season"
        season_dir.mkdir()
        _copy_with_sidecar(src, season_dir, "100001.json", None)
        _copy_with_sidecar(src, season_dir, "100002.json", None)

        svc = SeasonImportService(storage)
        first = await svc.import_statsbomb_directory(season_dir)
        assert first["imported"] == 2
        assert first["skipped_already"] == 0
        assert first["failed"] == 0

        second = await svc.import_statsbomb_directory(season_dir)
        assert second["imported"] == 0
        assert second["skipped_already"] == 2
        assert second["failed"] == 0

        ids = _external_ids(storage)
        assert ("statsbomb", "100001") in ids
        assert ("statsbomb", "100002") in ids

    @pytest.mark.asyncio
    async def test_non_event_json_skipped_not_fatal(self, storage, tmp_path):
        """Lineups/lineages (dict root) and empty files are counted as
        skipped, never crash the season run."""
        from kawkab.services.season_import_service import SeasonImportService

        src = _real_event_file()
        season_dir = tmp_path / "season"
        season_dir.mkdir()
        _copy_with_sidecar(src, season_dir, "100010.json", None)
        (season_dir / "lineups.json").write_text(json.dumps({"team": "x"}), encoding="utf-8")
        (season_dir / "empty.json").write_text("[]", encoding="utf-8")

        svc = SeasonImportService(storage)
        summary = await svc.import_statsbomb_directory(season_dir)

        assert summary["total_files"] == 3
        assert summary["imported"] == 1
        assert summary["skipped_not_events"] == 2
        assert summary["failed"] == 0

    @pytest.mark.asyncio
    async def test_sidecar_and_bulk_context_tagging(self, storage, tmp_path):
        """Competition/date/season land on the match row: sidecar wins
        over bulk args, StatsBomb's own match_date fills the rest."""
        from kawkab.services.season_import_service import SeasonImportService

        src = _real_event_file()
        season_dir = tmp_path / "season"
        season_dir.mkdir()
        _copy_with_sidecar(
            src,
            season_dir,
            "100020.json",
            {"competition": "UCL", "season_id": 7, "match_date": "2025-05-31"},
        )
        _copy_with_sidecar(src, season_dir, "100021.json", None)

        svc = SeasonImportService(storage)
        summary = await svc.import_statsbomb_directory(
            season_dir,
            competition="La Liga",
            season_id=3,
        )
        assert summary["imported"] == 2

        by_file = {m["file"]: m for m in summary["matches"]}
        sidecar_row = _match_row(storage, by_file["100020.json"]["match_id"])
        assert sidecar_row["competition"] == "UCL"
        assert sidecar_row["season_id"] == 7
        assert sidecar_row["match_date"] == "2025-05-31"

        bulk_row = _match_row(storage, by_file["100021.json"]["match_id"])
        assert bulk_row["competition"] == "La Liga"
        assert bulk_row["season_id"] == 3

    @pytest.mark.asyncio
    async def test_corrupt_file_counted_as_failed_not_fatal(self, storage, tmp_path):
        """One corrupt file logs + counts as failed; the season still lands."""
        from kawkab.services.season_import_service import SeasonImportService

        src = _real_event_file()
        season_dir = tmp_path / "season"
        season_dir.mkdir()
        _copy_with_sidecar(src, season_dir, "100030.json", None)
        (season_dir / "broken.json").write_text("{not json", encoding="utf-8")

        svc = SeasonImportService(storage)
        summary = await svc.import_statsbomb_directory(season_dir)

        assert summary["imported"] == 1
        assert summary["failed"] == 1
        failed = [m for m in summary["matches"] if m["status"] == "failed"]
        assert failed and failed[0]["file"] == "broken.json"

    @pytest.mark.asyncio
    async def test_max_matches_caps_eligible_imports(self, storage, tmp_path):
        from kawkab.services.season_import_service import SeasonImportService

        src = _real_event_file()
        season_dir = tmp_path / "season"
        season_dir.mkdir()
        for i in range(3):
            _copy_with_sidecar(src, season_dir, f"10004{i}.json", None)

        svc = SeasonImportService(storage)
        summary = await svc.import_statsbomb_directory(season_dir, max_matches=1)

        assert summary["imported"] == 1
        assert summary["eligible"] == 1  # capped before the walk finished

    @pytest.mark.asyncio
    async def test_empty_directory(self, storage, tmp_path):
        from kawkab.services.season_import_service import SeasonImportService

        svc = SeasonImportService(storage)
        summary = await svc.import_statsbomb_directory(tmp_path)
        assert summary["total_files"] == 0
        assert summary["imported"] == 0

    @pytest.mark.asyncio
    async def test_not_a_directory_raises(self, storage, tmp_path):
        from kawkab.services.season_import_service import SeasonImportService

        svc = SeasonImportService(storage)
        with pytest.raises(ValueError):
            await svc.import_statsbomb_directory(tmp_path / "missing")

    @pytest.mark.asyncio
    async def test_single_file_mode_dedups(self, storage, tmp_path):
        from kawkab.services.season_import_service import SeasonImportService

        src = _real_event_file()
        f = _copy_with_sidecar(src, tmp_path, "100040.json", None)

        svc = SeasonImportService(storage)
        first = await svc.import_single_file(f, competition="Copa")
        assert first["status"] == "imported"
        assert first["match_id"] > 0

        second = await svc.import_single_file(f)
        assert second["status"] == "skipped_already"
        assert second["match_id"] == first["match_id"]

        row = _match_row(storage, first["match_id"])
        assert row["competition"] == "Copa"

    @pytest.mark.asyncio
    async def test_meta_sidecar_never_imported_as_match(self, storage, tmp_path):
        """<stem>.meta.json sits in the same directory but is excluded
        from the import walk even when it is the only extra file."""
        from kawkab.services.season_import_service import SeasonImportService

        src = _real_event_file()
        season_dir = tmp_path / "season"
        season_dir.mkdir()
        _copy_with_sidecar(
            src,
            season_dir,
            "100050.json",
            {"competition": "Serie A"},
        )

        svc = SeasonImportService(storage)
        summary = await svc.import_statsbomb_directory(season_dir)

        names = [m["file"] for m in summary["matches"]]
        assert "100050.meta.json" not in names
        assert summary["total_files"] == 1


class TestExternalIdStorage:
    @pytest.mark.asyncio
    async def test_register_round_trip_and_duplicate_detection(self, storage):

        match_id = await storage.save_match("t", "")
        assert await storage.register_match_external_id(match_id, "statsbomb", "999") is True
        # duplicate registration is a no-op returning False
        assert await storage.register_match_external_id(match_id, "statsbomb", "999") is False
        assert await storage.get_match_by_external_id("statsbomb", "999") == match_id
        assert await storage.get_match_by_external_id("statsbomb", "nope") is None

    @pytest.mark.asyncio
    async def test_update_match_context_partial_writes(self, storage):
        match_id = await storage.save_match("ctx", "")
        await storage.update_match_context(match_id, competition="Ligue 1")
        row = _match_row(storage, match_id)
        assert row["competition"] == "Ligue 1"
        assert row["season_id"] is None  # untouched

        await storage.update_match_context(match_id, season_id=5)
        row = _match_row(storage, match_id)
        assert row["competition"] == "Ligue 1"  # not clobbered
        assert row["season_id"] == 5

    @pytest.mark.asyncio
    async def test_external_id_survives_across_matches(self, storage):
        """Same external id from two sources is allowed; same source is not."""
        m1 = await storage.save_match("a", "")
        m2 = await storage.save_match("b", "")
        assert await storage.register_match_external_id(m1, "opta", "f1") is True
        assert await storage.register_match_external_id(m2, "wyscout", "f1") is True
        assert await storage.get_match_by_external_id("opta", "f1") == m1
        assert await storage.get_match_by_external_id("wyscout", "f1") == m2
