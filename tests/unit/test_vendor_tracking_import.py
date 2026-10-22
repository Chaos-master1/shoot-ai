"""Tests for the vendor tracking import service (elite interop path).

Follows test_statsbomb_import_service.py's convention: a REAL SQLite
StorageService with the REAL migration chain (001-030), because synthetic
schemas mask real bugs (documented in CLAUDE.md's storage-divergence
history).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def storage(tmp_path):
    from kawkab.core.migration_manager import MigrationManager
    from kawkab.services.storage_service import StorageService

    scratch_db = tmp_path / "test_tracking_import.db"
    migrations_dir = PROJECT_ROOT / "src" / "kawkab" / "migrations"
    MigrationManager(scratch_db, migrations_dir).migrate()
    svc = StorageService()
    svc._db_path = scratch_db
    svc._conn = sqlite3.connect(str(scratch_db))
    svc._conn.row_factory = sqlite3.Row
    return svc


@pytest.fixture()
def skillcorner_file(tmp_path):
    """Minimal SkillCorner-shaped feed: 3 frames, 4 players, metadata."""
    data = {
        "fps": 25,
        "players": [
            {"track_id": 1, "name": "Alice", "side": "home", "position": "GK"},
            {"track_id": 2, "name": "Betty", "side": "home", "position": "CB"},
            {"track_id": 101, "name": "Carla", "side": "away", "position": "ST"},
            {"track_id": 102, "name": "Dana", "side": "away", "position": "CM"},
        ],
        "frames": [
            {
                "frame_id": 0,
                "time": 0,
                "period": 1,
                "players": [
                    {"track_id": 1, "x": -0.9, "y": 0.0},
                    {"track_id": 2, "x": -0.5, "y": -0.2},
                    {"track_id": 101, "x": 0.7, "y": 0.1},
                    {"track_id": 102, "x": 0.2, "y": 0.3},
                ],
                "ball": {"x": -0.4, "y": 0.0, "z": 0.0},
            },
            {
                "frame_id": 1,
                "time": 40,
                "period": 1,  # 40ms
                "players": [
                    {"track_id": 1, "x": -0.9, "y": 0.0},
                    {"track_id": 2, "x": -0.48, "y": -0.2},
                    {"track_id": 101, "x": 0.71, "y": 0.1},
                    {"track_id": 102, "x": 0.21, "y": 0.3},
                ],
                "ball": {"x": -0.3, "y": 0.01, "z": 0.0},
            },
            {
                "frame_id": 2,
                "time": 80,
                "period": 1,
                "players": [
                    {"track_id": 1, "x": -0.9, "y": 0.0},
                    {"track_id": 2, "x": -0.46, "y": -0.21},
                    {"track_id": 101, "x": 0.72, "y": 0.11},
                    {"track_id": 102, "x": 0.22, "y": 0.31},
                ],
                "ball": {"x": -0.2, "y": 0.02, "z": 0.1},
            },
        ],
    }
    path = tmp_path / "skillcorner_match.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _frame_rows(storage, match_id, limit=5000):
    return asyncio_get(storage.get_tracking_frames(match_id, limit=limit))


def asyncio_get(coro):
    import asyncio

    return asyncio.run(coro)


class TestCoordinateNormalization:
    def test_skillcorner_unit_pm_converts_to_meters(self):
        from kawkab.services.vendor_tracking_import_service import to_kawkab_meters

        x, y = to_kawkab_meters(-1.0, 1.0, convention="skillcorner")
        assert x == pytest.approx(0.0)
        assert y == pytest.approx(68.0)

        x, y = to_kawkab_meters(1.0, -1.0, convention="skillcorner")
        assert x == pytest.approx(105.0)
        assert y == pytest.approx(0.0)

        x, y = to_kawkab_meters(0.0, 0.0, convention="skillcorner")
        assert x == pytest.approx(52.5)
        assert y == pytest.approx(34.0)

    def test_normalized_100_convention(self):
        from kawkab.services.vendor_tracking_import_service import to_kawkab_meters

        x, y = to_kawkab_meters(50.0, 50.0, convention="normalized_100")
        assert x == pytest.approx(52.5)
        assert y == pytest.approx(34.0)

    def test_auto_detects_each_convention(self):
        from kawkab.services.vendor_tracking_import_service import to_kawkab_meters

        # ±1 territory -> skillcorner
        x, _ = to_kawkab_meters(0.5, 0.5, convention="auto")
        assert x == pytest.approx(78.75)
        # anything larger is treated as meters (documented: 0-100 feeds
        # MUST pass convention="normalized_100" explicitly — see service)
        x, _ = to_kawkab_meters(52.5, 34.0, convention="auto")
        assert x == pytest.approx(52.5)
        x, _ = to_kawkab_meters(50.0, 50.0, convention="normalized_100")
        assert x == pytest.approx(52.5)


class TestSkillCornerImport:
    def test_full_import_round_trip(self, storage, skillcorner_file):
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        svc = VendorTrackingImportService(storage)
        summary = asyncio_get(
            svc.import_tracking_file(
                skillcorner_file,
                home_team="Home FC",
                away_team="Away FC",
            )
        )

        assert summary["match_id"] > 0
        assert summary["vendor"] == "skillcorner"
        assert summary["deduplicated"] is False
        assert summary["frames_imported"] == 3
        assert summary["players_registered"] == 4

        # Frames persisted through the shared tracking_frames plumbing
        frames = _frame_rows(storage, summary["match_id"])
        assert len(frames) == 3
        # SkillCorner ±1 coords -> meters: ball at x=-0.4 -> 31.5m
        ball = frames[0]["ball_detections"][0]
        assert ball["x"] == pytest.approx(31.5, abs=0.1)

        # Players registered with vendor metadata
        players = asyncio_get(storage.get_match_players(summary["match_id"]))
        by_name = {p["name"]: p for p in players}
        assert by_name["Alice"]["team"] == "home"
        assert by_name["Carla"]["team"] == "away"

        # Provenance row written (migration 030)
        imports = asyncio_get(storage.get_tracking_imports(summary["match_id"]))
        assert len(imports) == 1
        assert imports[0]["vendor"] == "skillcorner"
        assert imports[0]["frame_count"] == 3
        assert imports[0]["fps"] == pytest.approx(25.0)
        assert imports[0]["checksum"]

        # Headline metric cached
        # (matches video-pipeline convention: advanced_metrics category=import)

    def test_reimport_same_file_is_deduplicated(self, storage, skillcorner_file):
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        svc = VendorTrackingImportService(storage)
        first = asyncio_get(svc.import_tracking_file(skillcorner_file))
        # Re-import attached to the SAME match -> deduplicated
        second = asyncio_get(
            svc.import_tracking_file(
                skillcorner_file,
                match_id=first["match_id"],
            )
        )

        assert first["deduplicated"] is False
        assert second["deduplicated"] is True
        assert second["frames_imported"] == 0
        # Frames unchanged after dedup skip
        frames = _frame_rows(storage, first["match_id"])
        assert len(frames) == 3


class TestTrackingDataQuality:
    """The import-time data-quality report (garbage-in defense)."""

    def test_quality_report_flags_synthetic_defects(self, storage, tmp_path):
        """A feed with a frame gap, a duplicate timestamp and an out-of-
        bounds player must have every defect surfaced by the report."""
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        data = {
            "fps": 10,
            "players": [
                {"track_id": 1, "name": "A", "side": "home"},
                {"track_id": 2, "name": "B", "side": "home"},
            ],
            "frames": [],
        }
        for i in range(10):
            ts_ms = (0.0 + (i if i < 5 else i + 3.0)) * 1000.0  # 3 s gap
            if i == 7:
                ts_ms = 8.0 * 1000.0  # duplicate of i=5 (clock glitch)
            players = [
                {"track_id": 1, "x": ((10.0 + i) / 105.0) * 2 - 1, "y": (30.0 / 68.0) * 2 - 1},
            ]
            if i == 3:
                players.append(
                    {"track_id": 2, "x": (500.0 / 105.0) * 2 - 1, "y": (-900.0 / 68.0) * 2 - 1}
                )
            else:
                players.append(
                    {"track_id": 2, "x": ((20.0 + i) / 105.0) * 2 - 1, "y": (40.0 / 68.0) * 2 - 1}
                )
            data["frames"].append(
                {
                    "frame_id": i,
                    "time": ts_ms,
                    "period": 1,
                    "players": players,
                    "ball": {"x": 0.0, "y": 0.0, "z": 0.0},
                }
            )
        f_path = tmp_path / "defective.json"
        f_path.write_text(json.dumps(data), encoding="utf-8")

        svc = VendorTrackingImportService(storage)
        summary = asyncio_get(svc.import_tracking_file(f_path))
        q = summary["quality"]
        assert q["frames"] == 10
        assert q["frame_gaps"] >= 1  # the 3 s hole
        assert q["duplicate_timestamps"] >= 1
        assert q["player_count_min"] == 2 and q["player_count_max"] == 2
        assert q["out_of_bounds_positions"] >= 1
        assert q["missing_ball_pct"] == 0.0

    def test_real_metrica_quality_is_plausible(self, storage):
        """The real fixture's report: few gaps, low out-of-bounds rate —
        real vendor numbers, not synthetic perfection."""
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        home = PROJECT_ROOT / "tests" / "fixtures" / "tracking" / "metrica_sample2_home.csv"
        away = PROJECT_ROOT / "tests" / "fixtures" / "tracking" / "metrica_sample2_away.csv"
        if not (home.exists() and away.exists()):
            pytest.skip("Metrica fixture files missing")

        svc = VendorTrackingImportService(storage)
        summary = asyncio_get(svc.import_tracking_file(home, vendor="metrica", away_csv=away))
        q = summary["quality"]
        assert q["frames"] == 1999
        assert q["frame_gaps"] <= 5  # near-clean feed
        assert q["duplicate_timestamps"] == 0
        assert q["player_count_min"] >= 18  # real feeds dip sometimes
        assert q["out_of_bounds_pct"] < 1.0
        assert 0.0 < q["missing_ball_pct"] < 15.0

        # Quality is stored in the provenance record for downstream reads
        imports = asyncio_get(storage.get_tracking_imports(summary["match_id"]))
        assert imports[0]["metadata"]["quality"]["frames"] == 1999

    def test_checksum_recorded_for_dedup_detection(self, storage, skillcorner_file):
        """Importing the same feed into a DIFFERENT (new) match is allowed
        (caller's choice) — but the identical checksum is recorded on both
        provenance rows so tooling can detect cross-match duplication."""
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        svc = VendorTrackingImportService(storage)
        a = asyncio_get(svc.import_tracking_file(skillcorner_file))
        b = asyncio_get(svc.import_tracking_file(skillcorner_file))
        assert a["match_id"] != b["match_id"]
        assert a["checksum"] == b["checksum"]
        rows_a = asyncio_get(storage.get_tracking_imports(a["match_id"]))
        rows_b = asyncio_get(storage.get_tracking_imports(b["match_id"]))
        assert rows_a[0]["checksum"] == rows_b[0]["checksum"]

    def test_attach_to_existing_match(self, storage, skillcorner_file):
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        existing_id = asyncio_get(storage.save_match("Existing", ""))
        svc = VendorTrackingImportService(storage)
        summary = asyncio_get(
            svc.import_tracking_file(
                skillcorner_file,
                match_id=existing_id,
                home_team="Home FC",
                away_team="Away FC",
            )
        )
        assert summary["match_id"] == existing_id


class TestEPTSImport:
    def test_epts_xml_import(self, storage, tmp_path):
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
            parse_epts,
        )

        # A deliberately non-FIFA-standard-but-representative EPTS-shaped
        # document: namespace-wrapped, Frame elements with Player/Ball
        # children in meters.
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<ns:TrainingSession xmlns:ns="http://schemas.fifa.com/epts/2021"
                    xmlns:schemaLocation="x">
  <ns:DeviceInfo><ns:FPS>25</ns:FPS></ns:DeviceInfo>
  <ns:Frame id="0" utc="1720000000000">
    <ns:Section>1</ns:Section>
    <ns:Player PlayerId="10" x="10.0" y="5.0" speed="1.2" team="home"/>
    <ns:Player PlayerId="20" x="90.0" y="60.0" speed="0.5" team="away"/>
    <ns:Ball x="50.0" y="34.0" z="0.0"/>
  </ns:Frame>
  <ns:Frame id="1" utc="1720000000040">
    <ns:Section>1</ns:Section>
    <ns:Player PlayerId="10" x="11.0" y="5.5" speed="1.5" team="home"/>
    <ns:Player PlayerId="20" x="89.5" y="60.2" speed="0.6" team="away"/>
    <ns:Ball x="51.0" y="34.5" z="0.2"/>
  </ns:Frame>
</ns:TrainingSession>"""
        path = tmp_path / "epts_match.xml"
        path.write_text(xml, encoding="utf-8")

        frames, meta, fps = parse_epts(path)
        assert fps == pytest.approx(25.0)
        assert len(frames) == 2
        # meters passthrough (auto convention)
        assert frames[0]["player_detections"][0]["x"] == pytest.approx(10.0)
        assert frames[0]["ball"]["y"] == pytest.approx(34.0)
        assert meta[10]["team"] == "home"

        svc = VendorTrackingImportService(storage)
        summary = asyncio_get(svc.import_tracking_file(path, vendor="epts"))
        assert summary["frames_imported"] == 2
        assert summary["players_registered"] == 2

    def test_epts_bad_file_raises(self, tmp_path):
        from kawkab.services.vendor_tracking_import_service import parse_epts

        path = tmp_path / "bad.xml"
        path.write_text("<html><body>not tracking</body></html>", encoding="utf-8")
        with pytest.raises(ValueError):
            parse_epts(path)


class TestMetricaImport:
    def test_metrica_requires_two_files(self, storage, tmp_path):
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        home = tmp_path / "home.csv"
        home.write_text("Period,Frame,Time [s]\n1,1,0.0\n", encoding="utf-8")
        svc = VendorTrackingImportService(storage)
        with pytest.raises(ValueError, match="away_csv"):
            asyncio_get(svc.import_tracking_file(home, vendor="metrica"))


class TestEventAlignment:
    def test_align_events_writes_frame_links(self, storage, skillcorner_file):
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        svc = VendorTrackingImportService(storage)
        summary = asyncio_get(svc.import_tracking_file(skillcorner_file))
        match_id = summary["match_id"]

        # Events at 0.04s and 0.08s — exactly on frames 1 and 2
        ev1 = asyncio_get(
            storage.save_event(
                match_id,
                {
                    "type": "pass",
                    "timestamp": 0.04,
                    "team": "home",
                },
            )
        )
        ev2 = asyncio_get(
            storage.save_event(
                match_id,
                {
                    "type": "shot",
                    "timestamp": 0.08,
                    "team": "away",
                },
            )
        )
        ev3 = asyncio_get(
            storage.save_event(
                match_id,
                {
                    "type": "pass",
                    "timestamp": 500.0,
                    "team": "home",  # no frame here
                },
            )
        )

        result = asyncio_get(svc.align_events(match_id, window_frames=1, fps=25.0))
        assert result["events_total"] == 3
        assert result["events_matched"] == 2
        assert result["links_written"] == 2

        links = asyncio_get(storage.get_event_frame_links(match_id))
        by_event = {link["event_id"]: link["frame_number"] for link in links}
        assert by_event[ev1] == 1
        assert by_event[ev2] == 2
        assert ev3 not in by_event  # unmatched events are reported, not linked

    def test_align_events_empty_match(self, storage):
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        match_id = asyncio_get(storage.save_match("Empty", ""))
        svc = VendorTrackingImportService(storage)
        result = asyncio_get(svc.align_events(match_id))
        assert result["events_total"] == 0
        assert result["links_written"] == 0


class TestVendorDetection:
    def test_detect_vendor_by_extension_and_content(self, tmp_path):
        from kawkab.services.vendor_tracking_import_service import detect_vendor

        json_file = tmp_path / "m.json"
        json_file.write_text('{"frames": []}', encoding="utf-8")
        assert detect_vendor(json_file) == "skillcorner"

        epts_file = tmp_path / "m.xml"
        epts_file.write_text('<TrainingSession xmlns="epts"/>', encoding="utf-8")
        assert detect_vendor(epts_file) == "epts"

        plain_xml = tmp_path / "other.xml"
        plain_xml.write_text("<root/>", encoding="utf-8")
        assert detect_vendor(plain_xml) is None

    def test_unsupported_vendor_rejected(self, storage, tmp_path):
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        f = tmp_path / "x.json"
        f.write_text("{}", encoding="utf-8")
        svc = VendorTrackingImportService(storage)
        with pytest.raises(ValueError, match="unsupported vendor"):
            asyncio_get(svc.import_tracking_file(f, vendor="tracab"))


class TestRealMetricaFixture:
    """End-to-end import of GENUINE vendor data: the first 2,000 frames of
    Metrica Sports' open Sample Game 2 (see tests/fixtures/tracking/README.md).
    Everything above runs on synthetic files; this is the proof the importer
    handles real vendor output."""

    FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "tracking"

    @pytest.fixture()
    def metrica_files(self):
        home = self.FIXTURE_DIR / "metrica_sample2_home.csv"
        away = self.FIXTURE_DIR / "metrica_sample2_away.csv"
        if not (home.exists() and away.exists()):
            pytest.skip("Metrica fixture files missing")
        return home, away

    def test_real_metrica_import_end_to_end(self, storage, metrica_files):
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        home_csv, away_csv = metrica_files
        svc = VendorTrackingImportService(storage)
        summary = asyncio_get(
            svc.import_tracking_file(
                home_csv,
                vendor="metrica",
                away_csv=away_csv,
                match_name="Metrica Sample Game 2 (fixture)",
                home_team="Home",
                away_team="Away",
            )
        )

        # The raw file's third line is a repeated header row the loader
        # (correctly) drops, so 2,000 CSV rows -> 1,999 genuine frames of
        # the first half's opening ~80 seconds.
        assert summary["frames_imported"] == 1999
        assert summary["players_registered"] == 22  # 11 home + 11 away
        assert summary["fps"] == pytest.approx(25.0)

        # Frames stored with real player positions in Kawkab meters
        frames = _frame_rows(storage, summary["match_id"])
        assert len(frames) == 1999
        first = frames[0]
        assert first["player_detections"], "frame must carry player positions"
        xs = [p["x"] for p in first["player_detections"]]
        ys = [p["y"] for p in first["player_detections"]]
        # Real vendor feeds carry slight out-of-bounds noise (this fixture
        # has an away player at y=-1.5m in frame 1) — assert within a
        # 3 m tolerance of the 105x68 pitch instead of strict bounds.
        assert all(-3.0 <= x <= 108.0 for x in xs)
        assert all(-3.0 <= y <= 71.0 for y in ys)

        # Tracking-import provenance record saved (migration 030 path)
        imports = asyncio_get(storage.get_tracking_imports(summary["match_id"]))
        assert len(imports) == 1
        assert imports[0]["vendor"] == "metrica"
        assert imports[0]["frame_count"] == 1999

    def test_real_metrica_is_deduplicated_on_reimport(self, storage, metrica_files):
        from kawkab.services.vendor_tracking_import_service import (
            VendorTrackingImportService,
        )

        home_csv, away_csv = metrica_files
        svc = VendorTrackingImportService(storage)
        first = asyncio_get(svc.import_tracking_file(home_csv, vendor="metrica", away_csv=away_csv))
        # Re-import attached to the SAME match -> deduplicated (dedup is
        # per-match by design; a new match import is the caller's choice).
        second = asyncio_get(
            svc.import_tracking_file(
                home_csv,
                vendor="metrica",
                away_csv=away_csv,
                match_id=first["match_id"],
            )
        )
        assert first["deduplicated"] is False
        assert second["deduplicated"] is True
        assert second["frames_imported"] == 0
