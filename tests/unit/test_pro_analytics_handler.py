"""Tests for the Pro Analytics handler (elite module aggregation).

Runs against a REAL migration-chained SQLite store fed by a REAL StatsBomb
import (the interop path), then asserts each block computes or reports an
honest data_available=False with a reason — never fabricated numbers.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS = PROJECT_ROOT / "data" / "statsbomb_corpus"

ALL_BLOCKS = [
    "epv",
    "obv",
    "off_ball",
    "pass_flow",
    "pressing_clusters",
    "duels",
    "ball_recovery",
    "box_entries",
    "switch_of_play",
    "crossing",
    "set_pieces",
    "through_balls",
    # tranche 2
    "carry_xt",
    "xg_chain",
    "game_state",
    "flank_analysis",
    "defensive_xt",
    "corner_xg",
    "crossing_xg",
    "expected_pass",
    "passing_triangles",
    "scoreline",
    "velocity",
    "influence_map",
    "lineup_optimizer",
]


@pytest.fixture()
def imported_match(tmp_path):
    """Real SQLite + real migrations + one real corpus import → (storage, match_id)."""
    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.statsbomb_import_service import StatsBombImportService
    from kawkab.services.storage_service import StorageService

    files = sorted(CORPUS.glob("*.json"))
    if not files:
        pytest.skip("statsbomb corpus not on this machine")
    db = tmp_path / "pro_analytics.db"
    MigrationManager(db, PROJECT_ROOT / "src" / "kawkab" / "migrations").migrate()
    svc = StorageService()
    svc._db_path = db
    svc._conn = sqlite3.connect(str(db))
    svc._conn.row_factory = sqlite3.Row
    imp = StatsBombImportService(svc)
    import asyncio

    async def _do():
        return await imp.import_match(files[0])

    summary = asyncio.run(_do())
    return svc, summary["match_id"]


class TestProAnalyticsReport:
    def test_report_shape(self, imported_match):
        import asyncio

        from kawkab.ui.bridge_handlers.bridge_pro_analytics import ProAnalyticsHandler

        svc, match_id = imported_match
        handler = ProAnalyticsHandler(bridge=None, services={"storage_service": svc})

        async def _run():
            return await handler.get_pro_analytics_report(match_id)

        r = json.loads(asyncio.run(_run()))
        assert r["success"] is True
        assert r["match_id"] == match_id
        assert r["n_events"] > 100
        # every block present, each with an honest availability flag
        assert set(r["blocks"].keys()) == set(ALL_BLOCKS)
        for name, block in r["blocks"].items():
            assert isinstance(block.get("data_available"), bool), name
            if not block["data_available"]:
                assert block.get("reason"), f"{name} unavailable without reason"

    def test_event_blocks_compute_on_imported_match(self, imported_match):
        """The point of the interop path: imported event data must produce
        real numbers in the event-only blocks (at least EPV + pass flow +
        pressing clusters + box entries + set pieces)."""
        import asyncio

        from kawkab.ui.bridge_handlers.bridge_pro_analytics import ProAnalyticsHandler

        svc, match_id = imported_match
        handler = ProAnalyticsHandler(bridge=None, services={"storage_service": svc})

        async def _run():
            return await handler.get_pro_analytics_report(match_id)

        r = json.loads(asyncio.run(_run()))
        blocks = r["blocks"]
        assert blocks["epv"]["data_available"] is True
        assert "home_total" in blocks["epv"]
        assert blocks["pass_flow"]["data_available"] is True
        assert blocks["pass_flow"]["n_links"] > 0
        assert blocks["pressing_clusters"]["data_available"] is True
        assert blocks["ball_recovery"]["data_available"] is True
        assert blocks["box_entries"]["data_available"] is True
        assert blocks["set_pieces"]["data_available"] is True
        # tranche-2 event blocks
        assert blocks["carry_xt"]["data_available"] is True
        assert blocks["xg_chain"]["data_available"] is True
        assert blocks["game_state"]["data_available"] is True
        assert blocks["flank_analysis"]["data_available"] is True
        assert blocks["defensive_xt"]["data_available"] is True
        assert blocks["expected_pass"]["data_available"] is True
        assert blocks["scoreline"]["data_available"] is True

    def test_tracking_blocks_honest_when_no_frames(self, imported_match):
        """OBV/off-ball must say WHY they're unavailable for an event-only
        import — never fabricate. The reason must mention tracking frames."""
        import asyncio

        from kawkab.ui.bridge_handlers.bridge_pro_analytics import ProAnalyticsHandler

        svc, match_id = imported_match
        handler = ProAnalyticsHandler(bridge=None, services={"storage_service": svc})

        async def _run():
            return await handler.get_pro_analytics_report(match_id)

        r = json.loads(asyncio.run(_run()))
        assert r["tracking_frames_available"] is False
        for name in ("obv", "off_ball"):
            block = r["blocks"][name]
            assert block["data_available"] is False
            assert "tracking" in block["reason"].lower()

    def test_obv_block_computes_with_synthetic_frames(self, imported_match, tmp_path):
        """When tracking frames DO exist, the OBV block must delegate and
        return the OffBallValuator's report (schema smoke)."""

        from kawkab.ui.bridge_handlers.bridge_pro_analytics import ProAnalyticsHandler

        svc, match_id = imported_match
        # Inject synthetic frames in the OBV schema via the handler's converter:
        handler = ProAnalyticsHandler(bridge=None, services={"storage_service": svc})
        raw_frames = [
            {
                "timestamp": float(t),
                "player_detections": [
                    {
                        "track_id": i,
                        "class_name": "person",
                        "confidence": 0.9,
                        "bbox": [10.0 * i, 20.0, 10.0 * i + 5.0, 30.0],
                    }
                    for i in range(6)
                ],
                "ball_detections": [
                    {
                        "track_id": 99,
                        "class_name": "sports ball",
                        "confidence": 0.8,
                        "bbox": [40.0, 24.0, 45.0, 29.0],
                    }
                ],
            }
            for t in range(30)
        ]
        frames = handler._frames_to_obv_schema(raw_frames)
        assert len(frames) == 30
        block = handler._obv_block(frames, [])
        assert block["data_available"] is True
        block_ob = handler._offball_block(frames)
        assert block_ob["data_available"] is True

    def test_empty_store_returns_honest_failure(self, tmp_path):
        import asyncio

        from kawkab.core.migration_manager import MigrationManager
        from kawkab.services.storage_service import StorageService
        from kawkab.ui.bridge_handlers.bridge_pro_analytics import ProAnalyticsHandler

        db = tmp_path / "empty.db"
        MigrationManager(db, PROJECT_ROOT / "src" / "kawkab" / "migrations").migrate()
        svc = StorageService()
        svc._db_path = db
        svc._conn = sqlite3.connect(str(db))
        svc._conn.row_factory = sqlite3.Row
        handler = ProAnalyticsHandler(bridge=None, services={"storage_service": svc})

        async def _run():
            return await handler.get_pro_analytics_report(1)

        r = json.loads(asyncio.run(_run()))
        assert r["success"] is False
        assert r["data_available"] is False
        assert "no events" in r["reason"]

    def test_none_coordinate_events_never_crash_blocks(self, tmp_path):
        """Regression: x=None (json_extract NULL) previously crashed EPV and
        ball_recovery with TypeError. Feed unlocated events directly."""
        from kawkab.core.ball_recovery import BallRecoveryAnalyzer
        from kawkab.core.epv import EPVModel

        events = [
            {"type": "pass", "team": "home", "x": None, "y": None, "end_x": None, "end_y": None},
            {"type": "pass", "team": "home", "x": None, "y": None},
            {"type": "shot", "team": "home", "x": None, "y": None, "is_goal": False},
            {"type": "interception", "team": "away", "x": None, "y": None},
            {"type": "pass", "team": "away", "x": 60.0, "y": 30.0},
        ]
        model = EPVModel()
        report = model.compute_match_epv(events)  # must not raise
        assert report is not None

        analyzer = BallRecoveryAnalyzer()
        out = analyzer.analyze_recoveries(events, "away")  # must not raise
        assert isinstance(out, dict)

    def test_none_events_never_crash_tranche2_modules(self):
        """Regression for the tranche-2 crashes found by real-import testing:
        defensive_xt (zone math on None), passing_triangles (sorted() over a
        set containing None), scoreline (sum over xg=None)."""
        import numpy as np

        from kawkab.core.defensive_xt import compute_defensive_xt
        from kawkab.core.passing_triangles import PassingTriangleAnalyzer
        from kawkab.core.scoreline_distribution import ScorelineDistribution

        events = [
            {
                "type": "pass",
                "team": "home",
                "timestamp": 10.0,
                "x": None,
                "y": None,
                "from_track_id": 1,
                "to_track_id": 2,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp": 20.0,
                "x": None,
                "y": None,
                "from_track_id": 2,
                "to_track_id": 3,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp": 30.0,
                "x": None,
                "y": None,
                "from_track_id": 3,
                "to_track_id": 1,
                "completed": True,
            },
            {
                "type": "interception",
                "team": "away",
                "timestamp": 35.0,
                "x": None,
                "y": None,
                "start_x": None,
                "start_y": None,
            },
            {
                "type": "shot",
                "team": "home",
                "timestamp": 40.0,
                "is_goal": True,
                "x": None,
                "y": None,
                "xg": None,
            },
        ]
        # defensive_xt: unlocated events are skipped, located ones valued
        grid = np.zeros((4, 4))
        actions = compute_defensive_xt(events, grid, xT_rows=4, xT_cols=4)
        assert isinstance(actions, list)

        # passing_triangles: None track_ids filtered before sorted()
        tris = PassingTriangleAnalyzer().detect_passing_triangles(events)
        assert isinstance(tris, list)

        # scoreline: xg=None treated as 0
        result = ScorelineDistribution().compute_scoreline_probabilities(events, n_sims=100)
        assert isinstance(result, dict)
