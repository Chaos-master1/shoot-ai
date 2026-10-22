"""Tests for the vendor event import service (Opta F24 / Wyscout)
and the validation report service (Phase 2).

Real SQLite storage with the real migration chain — same convention as
test_statsbomb_import_service.py.
"""

from __future__ import annotations

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

    scratch_db = tmp_path / "test_event_import.db"
    migrations_dir = PROJECT_ROOT / "src" / "kawkab" / "migrations"
    MigrationManager(scratch_db, migrations_dir).migrate()
    svc = StorageService()
    svc._db_path = scratch_db
    svc._conn = sqlite3.connect(str(scratch_db))
    svc._conn.row_factory = sqlite3.Row
    return svc


def aio(coro):
    import asyncio

    return asyncio.run(coro)


F24_XML = """<?xml version="1.0" encoding="UTF-8"?>
<soccerFeed xmlns="http://feed.elasticstats.com/schema/soccer/soccer-v1-0.xsd">
  <soccerDocument>
    <MatchData uID="m123">
      <MatchInfo>
        <Competition><Name>Premier League</Name></Competition>
        <Date>2026-08-01T15:00:00</Date>
      </MatchInfo>
      <GameEvents>
        <Event id="e1" type_id="1" team_id="t100" player_id="p11"
               time_min="12" time_sec="30" x="50.0" y="50.0"
               end_x="70.0" end_y="60.0" outcome="1"/>
        <Event id="e2" type_id="9" team_id="t200" player_id="p21"
               time_min="50" time_sec="15" x="85.0" y="50.0"
               end_x="90.0" end_y="52.0" outcome="0"/>
        <Event id="e3" type_id="11" team_id="t100" player_id="p11"
               time_min="80" time_sec="0" x="88.0" y="50.0"
               end_x="99.0" end_y="50.0" outcome="1"/>
      </GameEvents>
    </MatchData>
  </soccerDocument>
</soccerFeed>"""

F7_XML = """<?xml version="1.0" encoding="UTF-8"?>
<soccerFeed xmlns="http://feed.elasticstats.com/schema/soccer/soccer-v1-0.xsd">
  <soccerDocument>
    <MatchData uID="m123">
      <MatchInfo>
        <Competition><Name>Premier League</Name></Competition>
        <Season><Name>2026/27</Name></Season>
        <Date>2026-08-01T15:00:00</Date>
      </MatchInfo>
      <Team TeamRef="t100" uID="t100"><Name>Alpha United</Name>
        <Player uID="p11"><Name>Paul passer</Name>
          <Position PositionRef="mid">Midfielder</Position>
          <ShirtNumber Number="8">8</ShirtNumber>
        </Player>
      </Team>
      <Team TeamRef="t200" uID="t200"><Name>Beta City</Name>
        <Player uID="p21"><Name>Sam striker</Name>
          <Position PositionRef="fwd">Forward</Position>
          <ShirtNumber Number="9">9</ShirtNumber>
        </Player>
      </Team>
    </MatchData>
  </soccerDocument>
</soccerFeed>"""

WYSCOUT_JSON = """{
  "match": {"matchId": "w9", "homeTeam": {"name": "Alpha United"},
             "awayTeam": {"name": "Beta City"}},
  "lineups": [
    {"teamId": "100", "teamName": "Alpha United", "formation": "4-3-3",
     "players": [{"playerId": "5001", "name": "Paul passer", "shirtNumber": 8, "position": "midfielder"}]},
    {"teamId": "200", "teamName": "Beta City", "formation": "4-4-2",
     "players": [{"playerId": "6001", "name": "Sam striker", "shirtNumber": 9, "position": "forward"}]}
  ],
  "events": [
    {"id": "1", "matchId": "w9", "teamId": "100", "playerId": "5001",
     "eventName": "Passes", "minute": 10, "second": 5,
     "x": 50.0, "y": 50.0, "endX": 60.0, "endY": 55.0, "tags": ["accurate"]},
    {"id": "2", "matchId": "w9", "teamId": "200", "playerId": "6001",
     "eventName": "Shots", "minute": 55, "second": 30,
     "x": 88.0, "y": 50.0, "endX": 99.0, "endY": 50.0, "tags": ["goal"]}
  ]
}"""


class TestOptaImport:
    def test_f24_with_f7_resolves_names(self, storage, tmp_path):
        from kawkab.services.vendor_event_import_service import (
            VendorEventImportService,
        )

        f24 = tmp_path / "f24.xml"
        f24.write_text(F24_XML, encoding="utf-8")
        f7 = tmp_path / "f7.xml"
        f7.write_text(F7_XML, encoding="utf-8")

        svc = VendorEventImportService(storage)
        summary = aio(svc.import_opta_f24(f24, f7))

        assert summary["match_id"] > 0
        assert summary["vendor"] == "opta"
        assert summary["match_name"].startswith("Alpha United vs Beta City")
        # e1 pass + e2 shot(9->shot family? type_id 9 maps via parser) + e3 goal
        assert summary["events_imported"] == 3
        assert summary["players_registered"] == 2
        assert summary["shots"] >= 1
        assert summary["goals"] >= 1

        players = aio(storage.get_match_players(summary["match_id"]))
        names = {p["name"] for p in players}
        assert "Paul passer" in names, f"player names not resolved: {names}"
        assert "Sam striker" in names

    def test_shot_metadata_normalized(self, storage, tmp_path):
        from kawkab.services.vendor_event_import_service import (
            VendorEventImportService,
        )

        f24 = tmp_path / "f24.xml"
        f24.write_text(F24_XML, encoding="utf-8")
        f7 = tmp_path / "f7.xml"
        f7.write_text(F7_XML, encoding="utf-8")

        svc = VendorEventImportService(storage)
        summary = aio(svc.import_opta_f24(f24, f7))
        events = aio(storage.get_match_events(summary["match_id"], limit=500))
        shots = [
            e
            for e in events
            if (e["event_type"] if isinstance(e, dict) else e.get("event_type")) == "shot"
            or _safe_meta(e).get("is_goal")
        ]
        assert shots, "no shot events imported"
        goal_meta = [_safe_meta(e) for e in events if _safe_meta(e).get("is_goal")]
        assert goal_meta, "goal metadata missing"
        assert goal_meta[0]["shot_outcome"] == "goal"

    def test_f24_without_f7_uses_uids(self, storage, tmp_path):
        from kawkab.services.vendor_event_import_service import (
            VendorEventImportService,
        )

        f24 = tmp_path / "f24.xml"
        f24.write_text(F24_XML, encoding="utf-8")
        svc = VendorEventImportService(storage)
        summary = aio(svc.import_opta_f24(f24))
        players = aio(storage.get_match_players(summary["match_id"]))
        names = {p["name"] for p in players}
        assert any(name.startswith("Opta ") or name.startswith("Player ") for name in names)


def _safe_meta(event: dict) -> dict:
    import json as _json

    meta = event.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = _json.loads(meta)
        except (_json.JSONDecodeError, TypeError):
            meta = {}
    return meta


class TestWyscoutImport:
    def test_full_wyscout_import(self, storage, tmp_path):
        from kawkab.services.vendor_event_import_service import (
            VendorEventImportService,
        )

        path = tmp_path / "wyscout.json"
        path.write_text(WYSCOUT_JSON, encoding="utf-8")
        svc = VendorEventImportService(storage)
        summary = aio(svc.import_wyscout(path))

        assert summary["match_id"] > 0
        assert summary["vendor"] == "wyscout"
        assert summary["match_name"].startswith("Alpha United vs Beta City")
        assert summary["events_imported"] == 2
        assert summary["players_registered"] == 2
        assert summary["shots"] == 1
        assert summary["goals"] == 1

        players = aio(storage.get_match_players(summary["match_id"]))
        by_name = {p["name"]: p for p in players}
        assert by_name["Paul passer"]["jersey_number"] == 8
        assert by_name["Paul passer"]["team"] == "home"
        assert by_name["Sam striker"]["team"] == "away"

    def test_bad_wyscout_file_raises(self, storage, tmp_path):
        from kawkab.services.vendor_event_import_service import (
            VendorEventImportService,
        )

        path = tmp_path / "bad.json"
        path.write_text("[]", encoding="utf-8")  # not a dict -> no events
        svc = VendorEventImportService(storage)
        with pytest.raises(ValueError):
            aio(svc.import_wyscout(path))


class TestValidationReportService:
    def test_model_cards_section_counts_all(self):
        from kawkab.services.validation_report_service import ValidationReportService

        svc = ValidationReportService()
        section = svc.model_cards_section()
        assert section.status == "evaluated"
        assert section.metrics["registered"] >= 17
        names = [c["name"] for c in section.metrics["cards"]]
        assert "xG" in names and "PSxG" in names

    def test_xt_grid_section_evaluates(self):
        from kawkab.services.validation_report_service import ValidationReportService

        svc = ValidationReportService()
        section = svc.xt_grid_section()
        # The trained grid ships in-repo, so this must evaluate, not skip
        assert section.status == "evaluated"
        assert section.metrics["all_nonnegative"] is True

    def test_build_report_structure(self):
        from kawkab.services.validation_report_service import ValidationReportService

        svc = ValidationReportService()
        report = svc.build_report()
        assert "generated_at" in report
        assert len(report["sections"]) >= 4
        assert "summary" in report

    def test_markdown_and_write_round_trip(self, tmp_path):
        from kawkab.services.validation_report_service import ValidationReportService

        svc = ValidationReportService()
        _ = svc.write_report(out_dir=tmp_path)
        md_path = tmp_path / "validation_report.md"
        json_path = tmp_path / "validation_report.json"
        assert md_path.exists() and json_path.exists()
        md = md_path.read_text(encoding="utf-8")
        assert "# Kawkab Validation Report" in md
        assert "## model_cards" in md
