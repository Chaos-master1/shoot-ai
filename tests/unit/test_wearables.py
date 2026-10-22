"""Tests for the wearables ingestion sub-package.

Covers:
- ``WearableSession`` / ``WearableDataPoint`` models (finalisation, to_dict).
- ``CatapultCsvParser`` against a synthetic 10 Hz sensor export.
- ``StatsportsGpxParser`` against a synthetic GPX with HR + speed extensions.
- ``PolarHrCsvParser`` against both Polar Flow (``---`` separator) and plain
  numeric layouts.
- ``detect_parser`` auto-detection across extensions + content sniffing.
- ``WearableImportService`` facade (both legacy JSON API and structured API).

Fixtures write synthetic files to a tmp_path so no binary/proprietary data is
required to run these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kawkab.services.wearables import (
    BaseWearableParser,
    CatapultCsvParser,
    PolarHrCsvParser,
    StatsportsGpxParser,
    WearableDataPoint,
    WearableImportService,
    WearableSession,
    detect_parser,
)

# ---------------------------------------------------------------------------
# Fixture writers — synthesize the kind of files each vendor produces.
# ---------------------------------------------------------------------------


def _write_catapult_csv(path: Path, n: int = 1000) -> None:
    """Write a Catapult OpenField 10 Hz sensor-export style CSV."""
    import csv as _csv

    headers = [
        "Athlete Id",
        "Athlete",
        "Date",
        "Timestamp (s)",
        "Latitude",
        "Longitude",
        "Speed (m/s)",
        "Acceleration (m/s²)",
        "Heart Rate (bpm)",
        "Distance (m)",
        "Cadence (rpm)",
        "Body Load",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(headers)
        for i in range(n):
            writer.writerow(
                [
                    "ATH-001",
                    "Salah, Mohamed",  # contains a comma — csv.writer quotes it
                    "2026-07-05",
                    f"{i * 0.1:.2f}",
                    f"{53.430 + i * 1e-6:.6f}",
                    f"{-2.960 + i * 1e-6:.6f}",
                    f"{3.0 + (i % 20) * 0.1:.2f}",
                    f"{(i % 5) - 2:.2f}",
                    f"{140 + (i % 30)}",
                    f"{i * 0.3:.2f}",
                    f"{80 + (i % 10)}",
                    f"{0.5 + (i % 4) * 0.1:.2f}",
                ]
            )


def _write_gpx(path: Path, n: int = 50) -> None:
    """Write a STATSports-style GPX with HR + speed extensions."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx xmlns="http://www.topografix.com/GPX/1/1" '
        'xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1" '
        'xmlns:speed="http://www.garmin.com/xmlschemas/SpeedExtension/v1">',
        "  <trk><trkseg>",
    ]
    _ = 1751700000  # arbitrary epoch (unused; kept for reference)
    for i in range(n):
        lines.append(f'    <trkpt lat="{53.430 + i * 1e-5:.6f}" lon="{-2.960 + i * 1e-5:.6f}">')
        lines.append(f"      <ele>{10.0 + i * 0.01:.2f}</ele>")
        lines.append(f"      <time>2026-07-05T10:{i // 60:02d}:{i % 60:02d}Z</time>")
        lines.append("      <extensions>")
        lines.append(
            "        <gpxtpx:TrackPointExtension>"
            f"<gpxtpx:hr>{145 + (i % 20)}</gpxtpx:hr></gpxtpx:TrackPointExtension>"
        )
        lines.append(f"        <speed:speed>{5.0 + (i % 10) * 0.2:.2f}</speed:speed>")
        lines.append("      </extensions>")
        lines.append("    </trkpt>")
    lines.append("  </trkseg></trk></gpx>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_polar_flow_csv(path: Path, n: int = 60) -> None:
    """Polar Flow export: header row, ``---`` separator, then data."""
    with path.open("w", encoding="utf-8") as f:
        f.write("Time,HR (bpm),Speed (m/s)\n")
        f.write("---\n")
        for i in range(n):
            mm, ss = divmod(i, 60)
            f.write(f"00:{mm:02d}:{ss:02d},{130 + (i % 30)},{4.0 + (i % 8) * 0.1:.1f}\n")


def _write_polar_plain_csv(path: Path, n: int = 30) -> None:
    """Plain numeric: ``timestamp_s,hr[,speed]`` no header."""
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(f"{i * 1.0:.1f},{120 + i % 25},{3.0 + (i % 5) * 0.2:.1f}\n")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    def test_data_point_defaults(self):
        dp = WearableDataPoint()
        assert dp.timestamp_s == 0.0
        assert dp.heart_rate_bpm is None
        assert dp.speed_ms is None
        assert dp.extras == {}

    def test_session_finalize_empty(self):
        s = WearableSession()
        s.finalize()
        assert s.duration_s == 0.0
        assert s.sample_rate_hz == 0.0

    def test_session_finalize_computes_duration_and_rate(self):
        s = WearableSession(device_type="catapult")
        # 100 points spanning 0.0 → 9.9s at 10 Hz
        s.data = [WearableDataPoint(timestamp_s=i * 0.1) for i in range(100)]
        s.finalize()
        assert s.duration_s == pytest.approx(9.9, abs=1e-6)
        assert s.sample_rate_hz == pytest.approx(10.0, abs=1e-3)

    def test_session_to_dict_handles_empty(self):
        d = WearableSession().to_dict()
        assert d["point_count"] == 0
        assert d["avg_hr"] == 0.0
        assert d["total_distance_m"] == 0.0

    def test_session_to_dict_aggregates(self):
        s = WearableSession(device_type="polar")
        s.data = [
            WearableDataPoint(timestamp_s=0.0, heart_rate_bpm=140, speed_ms=4.0, distance_m=4.0),
            WearableDataPoint(timestamp_s=1.0, heart_rate_bpm=160, speed_ms=6.0, distance_m=6.0),
        ]
        s.finalize()
        d = s.to_dict()
        assert d["point_count"] == 2
        assert d["avg_hr"] == 150.0
        assert d["max_hr"] == 160.0
        assert d["min_hr"] == 140.0
        assert d["max_speed_ms"] == 6.0
        assert d["avg_speed_ms"] == 5.0
        assert d["total_distance_m"] == 10.0
        assert d["duration_s"] == 1.0
        assert d["sample_rate_hz"] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Catapult CSV parser
# ---------------------------------------------------------------------------


class TestCatapultCsvParser:
    def test_parses_full_session(self, tmp_path):
        f = tmp_path / "catapult.csv"
        _write_catapult_csv(f, n=1000)

        session = CatapultCsvParser().parse(str(f))

        assert session.device_type == "catapult"
        assert len(session.data) == 1000
        assert session.duration_s == pytest.approx(99.9, abs=1e-3)
        assert session.sample_rate_hz == pytest.approx(10.0, abs=1e-3)
        assert session.athlete_id == "ATH-001"
        assert session.athlete_name == "Salah, Mohamed"
        # First row spot-check
        assert session.data[0].speed_ms == pytest.approx(3.0)
        assert session.data[0].heart_rate_bpm == 140
        assert session.data[0].latitude is not None
        # extras carries Body Load
        assert "body_load" in session.data[0].extras

    def test_sensor_export_flag(self, tmp_path):
        f = tmp_path / "catapult.csv"
        _write_catapult_csv(f, n=1000)
        session = CatapultCsvParser().parse(str(f))
        assert session.metadata["is_sensor_export"] is True

    def test_ctr_short_export_flag_false(self, tmp_path):
        f = tmp_path / "ctr.csv"
        _write_catapult_csv(f, n=10)  # too few rows to be a 10Hz sensor export
        session = CatapultCsvParser().parse(str(f))
        assert session.metadata["is_sensor_export"] is False

    def test_skips_malformed_rows(self, tmp_path):
        f = tmp_path / "bad.csv"
        f.write_text(
            "Timestamp (s),Speed (m/s),Heart Rate (bpm)\n"
            "0.0,3.0,140\n"
            "not_a_number,3.0,141\n"  # bad timestamp → skipped
            "1.0,,142\n"  # missing speed → still parsed (None)
            "2.0,4.0,143\n",
            encoding="utf-8",
        )
        session = CatapultCsvParser().parse(str(f))
        # 1 row dropped (bad ts); 3 retained
        assert len(session.data) == 3
        assert session.data[1].speed_ms is None  # missing speed was tolerated

    def test_missing_timestamp_column_raises(self, tmp_path):
        f = tmp_path / "no_ts.csv"
        f.write_text("Speed (m/s),Heart Rate (bpm)\n3.0,140\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing a timestamp column"):
            CatapultCsvParser().parse(str(f))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            CatapultCsvParser().parse("does_not_exist.csv")

    def test_supports_extension(self):
        assert CatapultCsvParser().supports("foo.csv") is True
        assert CatapultCsvParser().supports("foo.gpx") is False


# ---------------------------------------------------------------------------
# STATSports GPX parser
# ---------------------------------------------------------------------------


class TestStatsportsGpxParser:
    def test_parses_gpx_with_extensions(self, tmp_path):
        f = tmp_path / "statsports.gpx"
        _write_gpx(f, n=50)

        session = StatsportsGpxParser().parse(str(f))

        assert session.device_type == "statsports"
        assert len(session.data) == 50
        assert session.start_time is not None
        # HR pulled from gpxtpx:hr extension
        assert session.data[0].heart_rate_bpm == 145
        # Speed pulled from speed:speed extension
        assert session.data[0].speed_ms == pytest.approx(5.0)
        # Elevation parsed
        assert session.data[0].altitude_m is not None
        assert session.data[0].latitude is not None

    def test_parses_minimal_gpx_no_extensions(self, tmp_path):
        f = tmp_path / "minimal.gpx"
        f.write_text(
            '<?xml version="1.0"?>'
            '<gpx xmlns="http://www.topografix.com/GPX/1/1">'
            "<trk><trkseg>"
            '<trkpt lat="53.43" lon="-2.96">'
            "<time>2026-07-05T10:00:00Z</time>"
            "</trkpt>"
            '<trkpt lat="53.431" lon="-2.961">'
            "<time>2026-07-05T10:00:01Z</time>"
            "</trkpt>"
            "</trkseg></trk></gpx>",
            encoding="utf-8",
        )
        session = StatsportsGpxParser().parse(str(f))
        assert len(session.data) == 2
        assert session.data[0].latitude == pytest.approx(53.43)
        # No extensions → HR/speed stay None
        assert session.data[0].heart_rate_bpm is None
        assert session.duration_s == pytest.approx(1.0, abs=1e-3)

    def test_handles_invalid_time_format(self, tmp_path):
        f = tmp_path / "bad_time.gpx"
        f.write_text(
            '<?xml version="1.0"?>'
            '<gpx xmlns="http://www.topografix.com/GPX/1/1">'
            "<trk><trkseg>"
            '<trkpt lat="53.43" lon="-2.96"><time>nonsense</time></trkpt>'
            "</trkseg></trk></gpx>",
            encoding="utf-8",
        )
        session = StatsportsGpxParser().parse(str(f))
        assert len(session.data) == 1
        assert session.data[0].timestamp_s == 0.0  # unchanged default

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            StatsportsGpxParser().parse("missing.gpx")


# ---------------------------------------------------------------------------
# Polar HR CSV parser
# ---------------------------------------------------------------------------


class TestPolarHrCsvParser:
    def test_parses_polar_flow_with_separator(self, tmp_path):
        f = tmp_path / "polar_flow.csv"
        _write_polar_flow_csv(f, n=60)

        session = PolarHrCsvParser().parse(str(f))

        assert session.device_type == "polar"
        assert len(session.data) == 60
        # First row: 00:00:00 → ts 0.0 ; HR 130 ; speed 4.0
        assert session.data[0].timestamp_s == 0.0
        assert session.data[0].heart_rate_bpm == 130
        assert session.data[0].speed_ms == pytest.approx(4.0)
        # Last row: 00:00:59 → ts 59 ; 1 Hz sampling
        assert session.data[-1].timestamp_s == pytest.approx(59.0)
        assert session.sample_rate_hz == pytest.approx(1.0, abs=1e-3)

    def test_parses_plain_numeric_csv(self, tmp_path):
        f = tmp_path / "plain.csv"
        _write_polar_plain_csv(f, n=30)

        session = PolarHrCsvParser().parse(str(f))
        assert len(session.data) == 30
        assert session.data[0].timestamp_s == 0.0
        assert session.data[0].heart_rate_bpm == 120
        assert session.data[5].speed_ms is not None

    def test_skips_blank_rows(self, tmp_path):
        f = tmp_path / "blanks.csv"
        f.write_text(
            "Time,HR (bpm)\n---\n00:00:00,140\n\n,\n00:00:01,142\n",
            encoding="utf-8",
        )
        session = PolarHrCsvParser().parse(str(f))
        assert len(session.data) == 2


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


class TestAutoDetect:
    def test_gpx_dispatches_to_statsports(self):
        p = detect_parser("foo.gpx")
        assert isinstance(p, StatsportsGpxParser)

    def test_catapult_csv_dispatches_correctly(self, tmp_path):
        f = tmp_path / "catapult.csv"
        _write_catapult_csv(f, n=10)
        p = detect_parser(str(f))
        assert isinstance(p, CatapultCsvParser)

    def test_polar_flow_dispatches_correctly(self, tmp_path):
        f = tmp_path / "polar.csv"
        _write_polar_flow_csv(f, n=5)
        p = detect_parser(str(f))
        assert isinstance(p, PolarHrCsvParser)

    def test_unknown_extension_returns_none(self):
        assert detect_parser("foo.xyz") is None

    def test_unknown_csv_falls_back_to_catapult(self, tmp_path):
        f = tmp_path / "mystery.csv"
        f.write_text("foo,bar,baz\n1,2,3\n", encoding="utf-8")
        p = detect_parser(str(f))
        # Fallback: Catapult (most permissive alias table)
        assert isinstance(p, CatapultCsvParser)


# ---------------------------------------------------------------------------
# WearableImportService facade
# ---------------------------------------------------------------------------


class TestWearableImportService:
    def test_import_catapult_csv_returns_json(self, tmp_path):
        f = tmp_path / "catapult.csv"
        _write_catapult_csv(f, n=20)
        out = WearableImportService().import_catapult_csv(str(f))
        payload = json.loads(out)
        assert payload["ok"] is True
        assert payload["session"]["device_type"] == "catapult"
        assert payload["session"]["point_count"] == 20

    def test_import_statsports_gpx_returns_json(self, tmp_path):
        f = tmp_path / "statsports.gpx"
        _write_gpx(f, n=10)
        out = WearableImportService().import_statsports_gpx(str(f))
        payload = json.loads(out)
        assert payload["ok"] is True
        assert payload["session"]["device_type"] == "statsports"

    def test_import_polar_hr_csv_returns_json(self, tmp_path):
        f = tmp_path / "polar.csv"
        _write_polar_flow_csv(f, n=10)
        out = WearableImportService().import_polar_hr_csv(str(f))
        payload = json.loads(out)
        assert payload["ok"] is True

    def test_import_auto_dispatches(self, tmp_path):
        f = tmp_path / "catapult.csv"
        _write_catapult_csv(f, n=10)
        out = WearableImportService().import_auto(str(f))
        payload = json.loads(out)
        assert payload["ok"] is True

    def test_import_auto_unknown_format(self, tmp_path):
        out = WearableImportService().import_auto(str(tmp_path / "nope.xyz"))
        payload = json.loads(out)
        assert "error" in payload

    def test_import_session_structured(self, tmp_path):
        f = tmp_path / "catapult.csv"
        _write_catapult_csv(f, n=10)
        session = WearableImportService().import_session(str(f))
        assert session is not None
        assert isinstance(session, WearableSession)
        assert session.device_type == "catapult"
        assert len(session.data) == 10

    def test_import_session_returns_none_on_failure(self):
        assert WearableImportService().import_session("missing.csv") is None

    def test_import_handles_missing_file_json(self):
        out = WearableImportService().import_catapult_csv("missing.csv")
        payload = json.loads(out)
        assert "error" in payload

    def test_back_compat_shim_reexports(self):
        # The old import path must still work
        from kawkab.services.wearable_import_service import (
            WearableDataPoint as ShimDataPoint,
        )
        from kawkab.services.wearable_import_service import (
            WearableImportService as ShimService,
        )
        from kawkab.services.wearable_import_service import (
            WearableSession as ShimSession,
        )

        assert ShimService is WearableImportService
        assert ShimSession is WearableSession
        assert ShimDataPoint is WearableDataPoint


# ---------------------------------------------------------------------------
# Base parser contract
# ---------------------------------------------------------------------------


class TestBaseParserContract:
    def test_base_is_abstract(self):
        with pytest.raises(TypeError):
            BaseWearableParser()  # type: ignore[abstract]

    def test_parse_float_helper(self):
        assert BaseWearableParser._parse_float("3.5") == 3.5
        assert BaseWearableParser._parse_float("") is None
        assert BaseWearableParser._parse_float(None) is None
        assert BaseWearableParser._parse_float("abc") is None

    def test_parse_int_helper(self):
        assert BaseWearableParser._parse_int("140") == 140
        assert BaseWearableParser._parse_int("140.7") == 140  # coerces float-string
        assert BaseWearableParser._parse_int("") is None


# ---------------------------------------------------------------------------
# FIT binary parser (via fitdecode mock/stub)
# ---------------------------------------------------------------------------


class TestFitParser:
    def test_supports_extension(self):
        from kawkab.services.wearables.fit_parser import FitParser

        assert FitParser().supports("session.fit") is True
        assert FitParser().supports("data.fit") is True
        assert FitParser().supports("data.csv") is False

    def test_parse_raises_on_missing_file(self):
        from kawkab.services.wearables.fit_parser import FitParser

        with pytest.raises(FileNotFoundError):
            FitParser().parse("no_such_file.fit")


# ---------------------------------------------------------------------------
# TCX XML parser
# ---------------------------------------------------------------------------


class TestTcxParser:
    def test_supports_extension(self):
        from kawkab.services.wearables.tcx_parser import TcxParser

        assert TcxParser().supports("activity.tcx") is True
        assert TcxParser().supports("data.csv") is False

    def test_parse_raises_on_missing_file(self):
        from kawkab.services.wearables.tcx_parser import TcxParser

        with pytest.raises(FileNotFoundError):
            TcxParser().parse("no_such_file.tcx")


# ---------------------------------------------------------------------------
# STATSports CSV aggregate parser
# ---------------------------------------------------------------------------


class TestStatsportsCsvParser:
    def test_supports_extension(self):
        from kawkab.services.wearables.statsports_csv import StatsportsCsvParser

        assert StatsportsCsvParser().supports("sonra.csv") is True
        assert StatsportsCsvParser().supports("data.gpx") is False

    def test_parse_raises_on_missing_file(self):
        from kawkab.services.wearables.statsports_csv import StatsportsCsvParser

        with pytest.raises(FileNotFoundError):
            StatsportsCsvParser().parse("no_such_file.csv")


# ---------------------------------------------------------------------------
# Storage / save
# ---------------------------------------------------------------------------


class TestWearableSave:
    @pytest.mark.asyncio
    async def test_save_session_no_storage(self):
        s = WearableSession(device_type="catapult")
        result = await WearableImportService().save_session(s)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_save_session_with_mock_storage(self):
        class MockStorage:
            def save_wearable_session(self, row):
                return 42  # simulated sync session_id

        s = WearableSession(
            device_type="catapult", athlete_id="ATH-001", athlete_name="Test Player"
        )
        s.data = [WearableDataPoint(timestamp_s=0.0, heart_rate_bpm=140, speed_ms=4.0)]
        s.finalize()
        result = await WearableImportService().save_session(s, storage_service=MockStorage())
        assert result.get("ok") is True
        assert result.get("session_id") == 42
