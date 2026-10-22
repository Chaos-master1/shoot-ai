"""Tests for the StatsBomb import service (elite interop path).

Runs against a REAL SQLite StorageService with the REAL migration chain
(001-029) — not a hand-written synthetic schema — because the CLAUDE.md
storage-divergence history shows synthetic schemas mask real bugs
(the get_match_players `confidence` column bug shipped that way).
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
    """Real SQLite storage with the REAL 001-029 migration chain.

    Mirrors the scratch-DB pattern test_api_v1.py uses (migrate + connect
    manually rather than svc.initialize(), which resolves the DB path via
    the app's global get_paths()).
    """
    import sqlite3

    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.storage_service import StorageService

    scratch_db = tmp_path / "test_import.db"
    migrations_dir = PROJECT_ROOT / "src" / "kawkab" / "migrations"
    MigrationManager(scratch_db, migrations_dir).migrate()
    svc = StorageService()
    svc._db_path = scratch_db
    svc._conn = sqlite3.connect(str(scratch_db))
    svc._conn.row_factory = sqlite3.Row
    return svc


@pytest.fixture()
def sample_file():
    files = sorted(CORPUS.glob("*.json"))
    if not files:
        pytest.skip("statsbomb corpus not on this machine")
    return files[0]


async def fetch_all_events(storage, match_id: int) -> list[dict]:
    """Page through get_match_events (its default limit is 200) and parse
    the metadata JSON strings into dicts — the shape downstream readers
    get after a real read."""
    out: list[dict] = []
    offset = 0
    while True:
        page = await storage.get_match_events(match_id, limit=500, offset=offset)
        if not page:
            break
        for row in page:
            e = dict(row)
            if isinstance(e.get("metadata"), str):
                try:
                    e["metadata"] = json.loads(e["metadata"])
                except (json.JSONDecodeError, TypeError):
                    e["metadata"] = {}
            out.append(e)
        if len(page) < 500:
            break
        offset += 500
    return out


class TestImportMatch:
    @pytest.mark.asyncio
    async def test_import_creates_match_and_events(self, storage, sample_file):
        from kawkab.services.statsbomb_import_service import StatsBombImportService

        svc = StatsBombImportService(storage)
        summary = await svc.import_match(sample_file)

        assert summary["match_id"] > 0
        assert summary["events_imported"] > 100  # real matches have thousands
        assert summary["players_registered"] >= 11
        assert summary["shots"] >= 0
        assert "xg_total" in summary

    @pytest.mark.asyncio
    async def test_events_round_trip_from_storage(self, storage, sample_file):
        """Imported events must be readable by get_match_events with the
        fields downstream models actually read (event_type, team, metadata)."""
        from kawkab.services.statsbomb_import_service import StatsBombImportService

        svc = StatsBombImportService(storage)
        summary = await svc.import_match(sample_file)
        events = await fetch_all_events(storage, summary["match_id"])

        assert len(events) == summary["events_imported"]
        types = {e.get("event_type") for e in events}
        assert "pass" in types  # every real match has passes
        teams = {e.get("team") for e in events}
        assert teams <= {"home", "away"}
        # metadata carries pitch coords in meters
        pass_events = [
            e
            for e in events
            if e.get("event_type") == "pass"
            and isinstance(e.get("metadata"), dict)
            and "start_x" in e["metadata"]
        ]
        assert pass_events, "no pass events carry spatial metadata"
        for e in pass_events[:50]:
            assert 0.0 <= e["metadata"]["start_x"] <= 105.0
            assert 0.0 <= e["metadata"]["start_y"] <= 68.0

    @pytest.mark.asyncio
    async def test_shot_metadata_has_trained_xg(self, storage, sample_file):
        """Every imported shot must carry xg computed by the ACTIVE model."""
        from kawkab.services.statsbomb_import_service import StatsBombImportService

        svc = StatsBombImportService(storage)
        summary = await svc.import_match(sample_file)
        events = await fetch_all_events(storage, summary["match_id"])
        shots = [e for e in events if e.get("event_type") == "shot"]
        assert shots, "sample match has no shots?"
        for s in shots[:20]:
            meta = s.get("metadata") or {}
            assert "xg" in meta, f"shot missing xg: {meta.keys()}"
            assert 0.0 <= meta["xg"] <= 1.0
            assert 0.0 <= meta.get("statsbomb_xg", 0.0) <= 1.0

    @pytest.mark.asyncio
    async def test_players_registered_with_teams(self, storage, sample_file):
        from kawkab.services.statsbomb_import_service import StatsBombImportService

        svc = StatsBombImportService(storage)
        summary = await svc.import_match(sample_file)
        players = await storage.get_match_players(summary["match_id"])
        assert len(players) >= 11
        teams = {p.get("team") for p in players}
        assert teams <= {"home", "away"}
        assert len(teams) == 2  # both squads present

    @pytest.mark.asyncio
    async def test_reimport_creates_new_match(self, storage, sample_file):
        """Importing twice must not corrupt anything — new match row each time."""
        from kawkab.services.statsbomb_import_service import StatsBombImportService

        svc = StatsBombImportService(storage)
        s1 = await svc.import_match(sample_file)
        s2 = await svc.import_match(sample_file)
        assert s1["match_id"] != s2["match_id"]
        e1 = await fetch_all_events(storage, s1["match_id"])
        e2 = await fetch_all_events(storage, s2["match_id"])
        assert len(e1) == len(e2)

    @pytest.mark.asyncio
    async def test_rejects_bad_file(self, storage, tmp_path):
        from kawkab.services.statsbomb_import_service import StatsBombImportService

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"not": "a list"}))
        svc = StatsBombImportService(storage)
        with pytest.raises(ValueError):
            await svc.import_match(bad)

    @pytest.mark.asyncio
    async def test_downstream_models_run_on_imported_match(self, storage, sample_file):
        """The point of interop: the existing analytics stack must work
        on imported data with zero extra plumbing. Spot-check three."""
        from kawkab.core.pass_network import compute_pass_network
        from kawkab.core.pressing_efficiency import PressingEfficiencyAnalyzer
        from kawkab.core.tactical_shape_analyzer import TacticalShapeAnalyzer
        from kawkab.services.statsbomb_import_service import StatsBombImportService

        svc = StatsBombImportService(storage)
        summary = await svc.import_match(sample_file)
        events = await fetch_all_events(storage, summary["match_id"])

        net = compute_pass_network(events, team="home")
        assert net is not None

        analyzer = PressingEfficiencyAnalyzer()
        trap_rate = analyzer.compute_trap_to_shot_rate(events)
        assert isinstance(trap_rate, dict)

        shapes = TacticalShapeAnalyzer()
        result = shapes.analyze_shapes(events, team="home")
        assert result is not None
