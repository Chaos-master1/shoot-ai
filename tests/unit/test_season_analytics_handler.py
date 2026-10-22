"""Tests for the SeasonAnalyticsHandler (cross-match season blocks)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS = PROJECT_ROOT / "data" / "statsbomb_corpus"


@pytest.fixture()
def multi_match_store(tmp_path):
    """Real store with 3 real corpus matches imported (season context)."""
    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.statsbomb_import_service import StatsBombImportService
    from kawkab.services.storage_service import StorageService

    files = sorted(CORPUS.glob("*.json"))[:3]
    if len(files) < 3:
        pytest.skip("statsbomb corpus not on this machine")
    db = tmp_path / "season.db"
    MigrationManager(db, PROJECT_ROOT / "src" / "kawkab" / "migrations").migrate()
    svc = StorageService()
    svc._db_path = db
    svc._conn = sqlite3.connect(str(db))
    svc._conn.row_factory = sqlite3.Row
    imp = StatsBombImportService(svc)

    async def _import_all():
        return [await imp.import_match(f) for f in files]

    summaries = asyncio.run(_import_all())
    return svc, summaries


class TestSeasonProReport:
    def test_report_shape(self, multi_match_store):
        from kawkab.ui.bridge_handlers.bridge_season_analytics import SeasonAnalyticsHandler

        svc, _ = multi_match_store
        handler = SeasonAnalyticsHandler(bridge=None, services={"storage_service": svc})
        r = json.loads(asyncio.run(handler.get_season_pro_report()))
        assert r["success"] is True
        assert r["n_matches"] == 3
        assert set(r["blocks"].keys()) == {"formation_trends", "discipline", "fixture_difficulty"}
        for name, block in r["blocks"].items():
            assert isinstance(block.get("data_available"), bool), name
            if not block["data_available"]:
                assert block.get("reason"), f"{name} unavailable without reason"

    def test_fixture_difficulty_computes(self, multi_match_store):
        """With 3 real imported matches, opponent strength is inferable
        from stored results, so the difficulty block must compute."""
        from kawkab.ui.bridge_handlers.bridge_season_analytics import SeasonAnalyticsHandler

        svc, _ = multi_match_store
        handler = SeasonAnalyticsHandler(bridge=None, services={"storage_service": svc})
        r = json.loads(asyncio.run(handler.get_season_pro_report()))
        block = r["blocks"]["fixture_difficulty"]
        assert block["data_available"] is True
        assert block["n_fixtures"] == 3

    def test_formation_trends_honest_method_label(self, multi_match_store):
        """When shapes ARE approximated, the method label must say so —
        approximation honesty is part of the contract."""
        from kawkab.ui.bridge_handlers.bridge_season_analytics import SeasonAnalyticsHandler

        svc, _ = multi_match_store
        handler = SeasonAnalyticsHandler(bridge=None, services={"storage_service": svc})
        r = json.loads(asyncio.run(handler.get_season_pro_report()))
        block = r["blocks"]["formation_trends"]
        if block["data_available"]:
            assert "approximat" in block.get("method", "").lower()

    def test_single_match_store_returns_honest_failure(self, tmp_path):
        from kawkab.core.migration_manager import MigrationManager
        from kawkab.services.statsbomb_import_service import StatsBombImportService
        from kawkab.services.storage_service import StorageService
        from kawkab.ui.bridge_handlers.bridge_season_analytics import SeasonAnalyticsHandler

        files = sorted(CORPUS.glob("*.json"))[:1]
        if not files:
            pytest.skip("corpus not on this machine")
        db = tmp_path / "one.db"
        MigrationManager(db, PROJECT_ROOT / "src" / "kawkab" / "migrations").migrate()
        svc = StorageService()
        svc._db_path = db
        svc._conn = sqlite3.connect(str(db))
        svc._conn.row_factory = sqlite3.Row
        imp = StatsBombImportService(svc)
        asyncio.run(_import_one(imp, files[0]))
        handler = SeasonAnalyticsHandler(bridge=None, services={"storage_service": svc})
        r = json.loads(asyncio.run(handler.get_season_pro_report()))
        assert r["success"] is False
        assert "at least 2" in r["reason"]


async def _import_one(imp, path):
    return await imp.import_match(path)
