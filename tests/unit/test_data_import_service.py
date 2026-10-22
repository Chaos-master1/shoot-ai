"""Tests for DataImportService — CSV, JSON, and StatsBomb import."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from tests.conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.data_import_service import DataImportService  # noqa: E402


@pytest.fixture
def storage_mock():
    s = MagicMock()
    s.save_events_bulk.return_value = 3
    return s


@pytest.fixture
def validator_mock():
    v = MagicMock()
    v.clamp_x.side_effect = lambda x: max(0.0, min(105.0, x))
    v.clamp_y.side_effect = lambda y: max(0.0, min(68.0, y))
    return v


@pytest.fixture
def service(storage_mock, validator_mock):
    return DataImportService(storage_service=storage_mock, coordinate_validator=validator_mock)


def _csv_content(rows: list[tuple]) -> str:
    header = "type,team,player_name,x,y,end_x,end_y,timestamp,xg,xa,xt"
    lines = [header]
    for r in rows:
        lines.append(",".join(str(v) for v in r))
    return "\n".join(lines)


class TestDataImportService:
    def test_import_csv_valid_events(self, service):
        content = _csv_content(
            [
                ("pass", "home", "Player A", 50, 30, 70, 40, 120.5, "", "", ""),
                ("shot", "home", "Player B", 95, 40, "", "", 245.0, 0.25, "", ""),
                ("goal", "home", "Player C", 90, 35, "", "", 300.0, 0.45, "", ""),
            ]
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            result = service.import_file(path, match_id="match_1")
            assert result["imported_count"] == 3
            assert result["total_errors"] == 0
        finally:
            os.unlink(path)

    def test_import_csv_malformed_row_tracks_error(self, service):
        content = _csv_content(
            [
                ("pass", "home", "Player A", 50, 30, 70, 40, 120.5, "", "", ""),
                ("bad_row", "", "", "", "", "", "", "not_a_number", "", "", ""),
            ]
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            result = service.import_file(path, match_id="match_1")
            assert result["total_errors"] > 0
            assert len(result["errors"]) > 0
        finally:
            os.unlink(path)

    def test_import_empty_csv(self, service):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write("type,team,player_name,x,y,end_x,end_y,timestamp,xg,xa,xt\n")
            f.flush()
            path = f.name
        try:
            result = service.import_file(path, match_id="match_1")
            assert result["imported_count"] == 0
            assert result["total_errors"] == 0
        finally:
            os.unlink(path)

    def test_import_generic_json_array(self, service):
        events = [
            {"type": "pass", "timestamp": 10.0, "team": "home", "x": 50.0, "y": 30.0},
            {"type": "shot", "timestamp": 20.0, "team": "home", "x": 90.0, "y": 40.0, "xg": 0.35},
            {"type": "goal", "timestamp": 30.0, "team": "home", "x": 95.0, "y": 35.0},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(events, f)
            f.flush()
            path = f.name
        try:
            result = service.import_file(path, match_id="match_1")
            assert result["imported_count"] == 3
            assert result["total_errors"] == 0
        finally:
            os.unlink(path)

    def test_import_statsbomb_json_format(self, service):
        data = {
            "events": [
                {
                    "type": {"name": "Pass"},
                    "location": [50.0, 30.0],
                    "pass": {"end_location": [70.0, 40.0]},
                    "team": {"name": "home"},
                    "player": {"id": 101, "name": "Player A"},
                    "timestamp": "00:01:00.000",
                    "period": 1,
                },
                {
                    "type": {"name": "Shot"},
                    "location": [95.0, 40.0],
                    "shot": {"statsbomb_xg": 0.45},
                    "team": {"name": "home"},
                    "player": {"id": 102, "name": "Player B"},
                    "timestamp": "00:02:00.000",
                    "period": 1,
                },
            ]
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            result = service.import_file(path, match_id="match_1")
            assert result["imported_count"] > 0
            assert result["total_errors"] == 0
        finally:
            os.unlink(path)

    def test_detect_csv_format_by_content(self):
        s = DataImportService()
        content = _csv_content([("pass", "home", "A", 50, 30, 70, 40, 120.5, "", "", "")])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            fmt = s.detect_format(path)
            assert fmt == "csv"
        finally:
            os.unlink(path)

    def test_detect_statsbomb_format_by_structure(self):
        s = DataImportService()
        data = {"events": [{"type": {"name": "Pass"}, "location": [0, 0]}]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            fmt = s.detect_format(path)
            assert fmt == "statsbomb_json"
        finally:
            os.unlink(path)

    def test_unsupported_extension_raises_value_error(self, service):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write("<events></events>")
            f.flush()
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                service.import_file(path, match_id="m1")
        finally:
            os.unlink(path)

    def test_file_not_found_raises_filenotfound(self, service):
        with pytest.raises(FileNotFoundError):
            service.import_file("C:/nonexistent_file_xyz.json", match_id="m1")

    def test_coordinate_validator_clamps_out_of_bounds(self, service, validator_mock):
        content = _csv_content(
            [
                ("pass", "home", "A", -10, 200, 150, -20, 100.0, "", "", ""),
            ]
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            result = service.import_file(path, match_id="match_1")
            assert result["imported_count"] > 0
        finally:
            os.unlink(path)
        assert validator_mock.clamp_x.call_count >= 2
        assert validator_mock.clamp_y.call_count >= 2

    def test_import_json_with_statsbomb_data_key(self, service):
        data = {
            "data": [
                {
                    "type": {"name": "Pass"},
                    "location": [50, 30],
                    "team": {"name": "away"},
                    "player": {"id": 1},
                }
            ]
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            result = service.import_file(path, match_id="m1")
            assert result["imported_count"] > 0
        finally:
            os.unlink(path)

    def test_detect_generic_json_format(self):
        s = DataImportService()
        data = [{"type": "pass", "timestamp": 1.0}]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            fmt = s.detect_format(path)
            assert fmt == "generic_json"
        finally:
            os.unlink(path)

    def test_csv_missing_type_skips_row(self, service):
        content = "type,team,player_name,x,y,end_x,end_y,timestamp,xg,xa,xt\n"
        content += ",".join(["", "home", "A", "50", "30", "70", "40", "100", "", "", ""])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            result = service.import_file(path, match_id="m1")
            assert result["imported_count"] == 0
        finally:
            os.unlink(path)

    def test_no_storage_returns_zero_imported(self, validator_mock):
        s = DataImportService(storage_service=None, coordinate_validator=validator_mock)
        content = _csv_content([("pass", "home", "A", 50, 30, 70, 40, 100, "", "", "")])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            path = f.name
        try:
            result = s.import_file(path, match_id="m1")
            assert result["imported_count"] == 0
        finally:
            os.unlink(path)
