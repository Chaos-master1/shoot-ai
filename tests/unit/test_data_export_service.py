"""Tests for DataExportService."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


@pytest.fixture(scope="module")
def de_mod():
    return load_service_module("kawkab.services.data_export_service", "data_export_service.py")


class FakeRow:
    """Simulates sqlite3.Row with dict-like access."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data.items())

    def keys(self):
        return self._data.keys()

    def get(self, key, default=None):
        return self._data.get(key, default)


def _make_row(data: dict) -> FakeRow:
    return FakeRow(data)


def _mock_conn(mocker, rows_by_query: dict[str, list[dict] | None]):
    """Return a mock connection that returns given rows per query snippet."""
    conn = MagicMock()
    cursor = MagicMock()

    def _execute(sql, params=None):
        for snippet, result in rows_by_query.items():
            if snippet in sql:
                if result is None:
                    cursor.fetchone.return_value = None
                    cursor.fetchall.return_value = []
                else:
                    fake_rows = [_make_row(r) for r in result]
                    if "LIMIT" in sql or "WHERE id = ?" in sql and "ORDER BY" not in sql:
                        cursor.fetchone.return_value = fake_rows[0] if fake_rows else None
                        cursor.fetchall.return_value = fake_rows
                    else:
                        cursor.fetchone.return_value = fake_rows[0] if fake_rows else None
                        cursor.fetchall.return_value = fake_rows
                break
        return cursor

    conn.cursor.return_value = cursor
    cursor.execute = _execute
    conn.commit = MagicMock()
    conn.close = MagicMock()
    return conn


class TestDataExportService:
    @pytest.mark.asyncio
    async def test_export_match_csv_success(self, de_mod, mocker):
        svc = de_mod.DataExportService()
        _ = tempfile.mktemp(suffix=".db")
        try:
            match_row = {
                "id": 1,
                "name": "Test Match",
                "home_team": "Home",
                "away_team": "Away",
                "match_date": "2024-01-01",
                "duration_seconds": 5400,
                "fps": 30,
                "total_frames": 162000,
            }
            events = [
                {
                    "id": 1,
                    "event_type": "pass",
                    "timestamp": 10.0,
                    "from_track_id": 1,
                    "to_track_id": 2,
                    "team": "home",
                    "completed": 1,
                    "confidence": 0.9,
                    "metadata": "{}",
                },
            ]
            players = [
                {
                    "id": 1,
                    "track_id": 1,
                    "jersey_number": 10,
                    "name": "P1",
                    "team": "home",
                    "position": "FW",
                    "distance_covered_m": 5000.0,
                    "max_speed_kmh": 30.0,
                    "avg_speed_kmh": 7.0,
                    "passes_attempted": 40,
                    "passes_completed": 35,
                    "shots": 3,
                    "tackles": 2,
                },
            ]

            conn = MagicMock()
            cursor = MagicMock()
            conn.cursor.return_value = cursor

            def _execute(sql, params=None):
                if "FROM matches WHERE id = ?" in sql:
                    cursor.fetchone.return_value = _make_row(match_row)
                elif "FROM events WHERE match_id = ?" in sql:
                    cursor.fetchall.return_value = [_make_row(e) for e in events]
                elif "FROM players WHERE match_id = ?" in sql:
                    cursor.fetchall.return_value = [_make_row(p) for p in players]
                return cursor

            cursor.execute = _execute
            svc._conn = conn

            svc._exports_dir = Path(tempfile.mkdtemp())
            result = await svc.export_match_csv(1)
            assert result.exists()
            assert (result / "summary.csv").exists()
            assert (result / "events.csv").exists()
            assert (result / "players.csv").exists()

            with open(result / "summary.csv") as f:
                lines = f.readlines()
                assert len(lines) == 2
                assert "Test Match" in lines[1]

            with open(result / "events.csv") as f:
                lines = f.readlines()
                assert len(lines) == 2
                assert "pass" in lines[1]

            with open(result / "players.csv") as f:
                lines = f.readlines()
                assert len(lines) == 2
                assert "P1" in lines[1]
        finally:
            pass

    @pytest.mark.asyncio
    async def test_export_match_csv_match_not_found(self, de_mod):
        svc = de_mod.DataExportService()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        cursor.execute.return_value = cursor
        conn.cursor.return_value = cursor
        svc._conn = conn

        with pytest.raises(ValueError, match="Match 999 not found"):
            await svc.export_match_csv(999)

    @pytest.mark.asyncio
    async def test_export_match_csv_no_events(self, de_mod):
        svc = de_mod.DataExportService()
        match_row = {
            "id": 2,
            "name": "No Events",
            "home_team": "H",
            "away_team": "A",
            "match_date": "2024-01-01",
            "duration_seconds": 5400,
            "fps": 30,
            "total_frames": 162000,
        }
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def _execute(sql, params=None):
            if "FROM matches WHERE id = ?" in sql:
                cursor.fetchone.return_value = _make_row(match_row)
            elif (
                "FROM events WHERE match_id = ?" in sql or "FROM players WHERE match_id = ?" in sql
            ):
                cursor.fetchall.return_value = []
            return cursor

        cursor.execute = _execute
        svc._conn = conn
        svc._exports_dir = Path(tempfile.mkdtemp())
        result = await svc.export_match_csv(2, include_players=False)
        assert result.exists()
        assert (result / "summary.csv").exists()
        assert not (result / "players.csv").exists()

    @pytest.mark.asyncio
    async def test_export_match_json_success(self, de_mod):
        svc = de_mod.DataExportService()
        match_row = {
            "id": 1,
            "name": "JSON Match",
            "home_team": "H",
            "away_team": "A",
            "match_date": "2024-01-01",
            "duration_seconds": 5400,
            "fps": 30,
            "total_frames": 162000,
        }
        analysis_row = {"id": 1, "match_id": 1, "full_data": "{}"}
        players = [
            {
                "id": 1,
                "track_id": 1,
                "jersey_number": 10,
                "name": "P1",
                "team": "home",
                "position": "FW",
                "distance_covered_m": 5000.0,
                "max_speed_kmh": 30.0,
                "avg_speed_kmh": 7.0,
                "passes_attempted": 40,
                "passes_completed": 35,
                "shots": 3,
                "tackles": 2,
            }
        ]
        events = [
            {
                "id": 1,
                "event_type": "pass",
                "timestamp": 10.0,
                "from_track_id": 1,
                "to_track_id": 2,
                "team": "home",
                "completed": 1,
                "confidence": 0.9,
                "metadata": '{"key": "val"}',
            }
        ]

        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def _execute(sql, params=None):
            if "FROM matches WHERE id = ?" in sql:
                cursor.fetchone.return_value = _make_row(match_row)
            elif "FROM analysis_results WHERE match_id = ?" in sql:
                cursor.fetchone.return_value = _make_row(analysis_row)
            elif "FROM players WHERE match_id = ?" in sql:
                cursor.fetchall.return_value = [_make_row(p) for p in players]
            elif "FROM events WHERE match_id = ?" in sql:
                cursor.fetchall.return_value = [_make_row(e) for e in events]
            return cursor

        cursor.execute = _execute
        svc._conn = conn
        svc._exports_dir = Path(tempfile.mkdtemp())
        result = await svc.export_match_json(1)
        assert result.exists()
        with open(result) as f:
            data = json.load(f)
        assert data["match"]["name"] == "JSON Match"
        assert len(data["players"]) == 1
        assert len(data["events"]) == 1
        assert data["events"][0]["metadata"]["key"] == "val"
        assert data["export_version"] == "1.0"

    @pytest.mark.asyncio
    async def test_export_match_json_no_analysis(self, de_mod):
        svc = de_mod.DataExportService()
        match_row = {
            "id": 3,
            "name": "No Analysis",
            "home_team": "H",
            "away_team": "A",
            "match_date": "2024-01-01",
            "duration_seconds": 5400,
            "fps": 30,
            "total_frames": 162000,
        }
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def _execute(sql, params=None):
            if "FROM matches WHERE id = ?" in sql:
                cursor.fetchone.return_value = _make_row(match_row)
            elif "FROM analysis_results WHERE match_id = ?" in sql:
                cursor.fetchone.return_value = None
            elif (
                "FROM players WHERE match_id = ?" in sql or "FROM events WHERE match_id = ?" in sql
            ):
                cursor.fetchall.return_value = []
            return cursor

        cursor.execute = _execute
        svc._conn = conn
        svc._exports_dir = Path(tempfile.mkdtemp())
        result = await svc.export_match_json(3)
        with open(result) as f:
            data = json.load(f)
        assert data["analysis"] == {}

    @pytest.mark.asyncio
    async def test_export_match_json_bad_metadata(self, de_mod):
        svc = de_mod.DataExportService()
        match_row = {
            "id": 4,
            "name": "Bad Meta",
            "home_team": "H",
            "away_team": "A",
            "match_date": "2024-01-01",
            "duration_seconds": 5400,
            "fps": 30,
            "total_frames": 162000,
        }
        events = [
            {
                "id": 1,
                "event_type": "pass",
                "timestamp": 10.0,
                "from_track_id": 1,
                "to_track_id": 2,
                "team": "home",
                "completed": 1,
                "confidence": 0.9,
                "metadata": "{invalid",
            }
        ]
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def _execute(sql, params=None):
            if "FROM matches WHERE id = ?" in sql:
                cursor.fetchone.return_value = _make_row(match_row)
            elif "FROM analysis_results WHERE match_id = ?" in sql:
                cursor.fetchone.return_value = None
            elif "FROM players WHERE match_id = ?" in sql:
                cursor.fetchall.return_value = []
            elif "FROM events WHERE match_id = ?" in sql:
                cursor.fetchall.return_value = [_make_row(e) for e in events]
            return cursor

        cursor.execute = _execute
        svc._conn = conn
        svc._exports_dir = Path(tempfile.mkdtemp())
        result = await svc.export_match_json(4)
        with open(result) as f:
            data = json.load(f)
        assert data["events"][0]["metadata"] == {}

    @pytest.mark.asyncio
    async def test_export_statsbomb_compatible(self, de_mod):
        svc = de_mod.DataExportService()
        match_row = {
            "id": 1,
            "name": "SB Match",
            "home_team": "H",
            "away_team": "A",
            "match_date": "2024-01-01",
            "duration_seconds": 5400,
            "fps": 30,
            "total_frames": 162000,
        }
        events = [
            {
                "id": 1,
                "event_type": "pass",
                "timestamp": 10.0,
                "from_track_id": 1,
                "to_track_id": 2,
                "team": "home",
                "completed": 1,
                "confidence": 0.9,
                "metadata": "{}",
            },
            {
                "id": 2,
                "event_type": "shot",
                "timestamp": 20.0,
                "from_track_id": 3,
                "to_track_id": None,
                "team": "away",
                "completed": 1,
                "confidence": 0.7,
                "metadata": "{}",
            },
        ]
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def _execute(sql, params=None):
            if "FROM matches WHERE id = ?" in sql:
                cursor.fetchone.return_value = _make_row(match_row)
            elif "FROM events WHERE match_id = ?" in sql:
                cursor.fetchall.return_value = [_make_row(e) for e in events]
            return cursor

        cursor.execute = _execute
        svc._conn = conn
        svc._exports_dir = Path(tempfile.mkdtemp())
        result = await svc.export_statsbomb_compatible(1)
        with open(result) as f:
            data = json.load(f)
        assert len(data["events"]) == 2
        assert data["events"][0]["type"]["name"] == "Pass"
        assert data["events"][0]["pass"]["recipient"]["id"] == 2
        assert data["events"][1]["type"]["name"] == "Shot"
        assert "shot" in data["events"][1]

    # ── StatsBomb-specific fixes ────────────────────────────────────────────

    async def _sb_export(self, de_mod, events, match_row=None):
        """Helper: run export_statsbomb_compatible and return parsed data."""
        svc = de_mod.DataExportService()
        if match_row is None:
            match_row = {
                "id": 1,
                "name": "SB Match",
                "home_team": "H",
                "away_team": "A",
                "match_date": "2024-01-01",
                "duration_seconds": 5400,
                "fps": 30,
                "total_frames": 162000,
            }
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def _execute(sql, params=None):
            if "FROM matches WHERE id = ?" in sql:
                cursor.fetchone.return_value = _make_row(match_row)
            elif "FROM events WHERE match_id = ?" in sql:
                cursor.fetchall.return_value = [_make_row(e) for e in events]
            return cursor

        cursor.execute = _execute
        svc._conn = conn
        svc._exports_dir = Path(tempfile.mkdtemp())
        out_path = await svc.export_statsbomb_compatible(1)
        with open(out_path) as f:
            return json.load(f)

    @pytest.mark.asyncio
    async def test_statsbomb_period_first_half(self, de_mod):
        events = [
            {
                "id": 1,
                "event_type": "pass",
                "timestamp": 120.0,
                "from_track_id": 1,
                "to_track_id": 2,
                "team": "home",
                "completed": 1,
                "confidence": 0.9,
                "metadata": "{}",
            },
        ]
        data = await self._sb_export(de_mod, events)
        assert data["events"][0]["period"] == 1

    @pytest.mark.asyncio
    async def test_statsbomb_period_second_half(self, de_mod):
        events = [
            {
                "id": 1,
                "event_type": "pass",
                "timestamp": 3600.0,
                "from_track_id": 1,
                "to_track_id": 2,
                "team": "home",
                "completed": 1,
                "confidence": 0.9,
                "metadata": "{}",
            },
        ]
        data = await self._sb_export(de_mod, events)
        assert data["events"][0]["period"] == 2

    @pytest.mark.asyncio
    async def test_statsbomb_period_from_metadata(self, de_mod):
        events = [
            {
                "id": 1,
                "event_type": "pass",
                "timestamp": 5000.0,
                "from_track_id": 1,
                "to_track_id": 2,
                "team": "home",
                "completed": 1,
                "confidence": 0.9,
                "metadata": '{"period": 3}',
            },
        ]
        data = await self._sb_export(de_mod, events)
        assert data["events"][0]["period"] == 3

    @pytest.mark.asyncio
    async def test_statsbomb_shot_outcome_goal(self, de_mod):
        events = [
            {
                "id": 1,
                "event_type": "shot",
                "timestamp": 30.0,
                "from_track_id": 3,
                "to_track_id": None,
                "team": "home",
                "completed": 1,
                "confidence": 0.9,
                "metadata": '{"is_goal": true}',
            },
        ]
        data = await self._sb_export(de_mod, events)
        assert data["events"][0]["shot"]["outcome"]["id"] == 97
        assert data["events"][0]["shot"]["outcome"]["name"] == "Goal"

    @pytest.mark.asyncio
    async def test_statsbomb_shot_outcome_saved(self, de_mod):
        events = [
            {
                "id": 1,
                "event_type": "shot",
                "timestamp": 30.0,
                "from_track_id": 3,
                "to_track_id": None,
                "team": "home",
                "completed": 1,
                "confidence": 0.9,
                "metadata": '{"is_saved": true}',
            },
        ]
        data = await self._sb_export(de_mod, events)
        assert data["events"][0]["shot"]["outcome"]["id"] == 95

    @pytest.mark.asyncio
    async def test_statsbomb_shot_outcome_blocked(self, de_mod):
        events = [
            {
                "id": 1,
                "event_type": "shot",
                "timestamp": 30.0,
                "from_track_id": 3,
                "to_track_id": None,
                "team": "home",
                "completed": 1,
                "confidence": 0.9,
                "metadata": '{"blocked": true}',
            },
        ]
        data = await self._sb_export(de_mod, events)
        assert data["events"][0]["shot"]["outcome"]["id"] == 96

    @pytest.mark.asyncio
    async def test_statsbomb_shot_outcome_missed(self, de_mod):
        events = [
            {
                "id": 1,
                "event_type": "shot",
                "timestamp": 30.0,
                "from_track_id": 3,
                "to_track_id": None,
                "team": "home",
                "completed": 1,
                "confidence": 0.9,
                "metadata": '{"off_target": true}',
            },
        ]
        data = await self._sb_export(de_mod, events)
        assert data["events"][0]["shot"]["outcome"]["id"] == 94

    @pytest.mark.asyncio
    async def test_statsbomb_xg_from_event_xg(self, de_mod):
        events = [
            {
                "id": 1,
                "event_type": "shot",
                "timestamp": 30.0,
                "from_track_id": 3,
                "to_track_id": None,
                "team": "home",
                "completed": 1,
                "confidence": 0.9,
                "metadata": '{"xg": 0.45}',
            },
        ]
        data = await self._sb_export(de_mod, events)
        assert data["events"][0]["shot"]["xG"] == 0.45

    @pytest.mark.asyncio
    async def test_statsbomb_pass_length_and_angle(self, de_mod):
        events = [
            {
                "id": 1,
                "event_type": "pass",
                "timestamp": 10.0,
                "from_track_id": 1,
                "to_track_id": 2,
                "team": "home",
                "completed": 1,
                "confidence": 0.9,
                "metadata": '{"start_x": 0, "start_y": 34, "end_x": 30, "end_y": 40}',
            },
        ]
        data = await self._sb_export(de_mod, events)
        length = data["events"][0]["pass"]["length"]
        _ = data["events"][0]["pass"]["angle"]
        assert length > 0
        import math

        expected_length = math.hypot(30, 6)
        assert abs(length - expected_length) < 0.1

    @pytest.mark.asyncio
    async def test_export_statsbomb_match_not_found(self, de_mod):
        svc = de_mod.DataExportService()
        conn = MagicMock()
        cursor = MagicMock()
        cursor.fetchone.return_value = None
        cursor.execute.return_value = cursor
        conn.cursor.return_value = cursor
        svc._conn = conn
        with pytest.raises(ValueError, match="Match 999 not found"):
            await svc.export_statsbomb_compatible(999)

    @pytest.mark.asyncio
    async def test_map_event_type_known(self, de_mod):
        svc = de_mod.DataExportService()
        assert svc._map_event_type("pass") == (30, "Pass")
        assert svc._map_event_type("shot") == (16, "Shot")
        assert svc._map_event_type("tackle") == (70, "Tackle")
        assert svc._map_event_type("goal") == (16, "Goal")
        assert svc._map_event_type("corner") == (6, "Corner Kick")

    @pytest.mark.asyncio
    async def test_map_event_type_unknown(self, de_mod):
        svc = de_mod.DataExportService()
        assert svc._map_event_type("nonexistent") == (1, "Unknown")

    @pytest.mark.asyncio
    async def test_export_season_csv(self, de_mod):
        svc = de_mod.DataExportService()
        season_row = {"id": 1, "name": "2024 Season"}
        matches = [
            {
                "id": 1,
                "name": "M1",
                "match_date": "2024-01-01",
                "home_team": "H",
                "away_team": "A",
                "score_home": 2,
                "score_away": 1,
                "duration_seconds": 5400,
                "match_type": "league",
            },
            {
                "id": 2,
                "name": "M2",
                "match_date": "2024-01-02",
                "home_team": "H2",
                "away_team": "A2",
                "score_home": 0,
                "score_away": 0,
                "duration_seconds": 5400,
                "match_type": "cup",
            },
        ]
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def _execute(sql, params=None):
            if "FROM seasons WHERE id = ?" in sql:
                cursor.fetchone.return_value = _make_row(season_row)
            else:
                cursor.fetchall.return_value = [_make_row(m) for m in matches]
            return cursor

        cursor.execute = _execute
        svc._conn = conn
        svc._exports_dir = Path(tempfile.mkdtemp())
        result = await svc.export_season_csv(1)
        assert result.exists()
        with open(result) as f:
            lines = f.readlines()
        assert len(lines) == 3
        assert "M1" in lines[1]
        assert "M2" in lines[2]

    @pytest.mark.asyncio
    async def test_export_season_csv_no_season_name(self, de_mod):
        svc = de_mod.DataExportService()
        matches = [
            {
                "id": 5,
                "name": "M5",
                "match_date": "2024-01-01",
                "home_team": "H",
                "away_team": "A",
                "score_home": 1,
                "score_away": 1,
                "duration_seconds": 5400,
                "match_type": "league",
            },
        ]
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor

        def _execute(sql, params=None):
            if "FROM seasons WHERE id = ?" in sql:
                cursor.fetchone.return_value = None
            else:
                cursor.fetchall.return_value = [_make_row(m) for m in matches]
            return cursor

        cursor.execute = _execute
        svc._conn = conn
        svc._exports_dir = Path(tempfile.mkdtemp())
        result = await svc.export_season_csv(999)
        assert result.exists()

    @pytest.mark.asyncio
    async def test_close_clears_connection(self, de_mod):
        svc = de_mod.DataExportService()
        conn = MagicMock()
        svc._conn = conn
        await svc.close()
        conn.close.assert_called_once()
        assert svc._conn is None

    @pytest.mark.asyncio
    async def test_get_conn_creates_connection(self, de_mod):
        svc = de_mod.DataExportService()
        svc._db_path = Path(tempfile.mktemp(suffix=".db"))
        conn = svc._get_conn()
        assert conn is not None
        await svc.close()
