"""Tests for bridge handler modules: Export, External, Lifecycle, Storage, Video."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.ui.bridge_handlers.bridge_export import ExportHandler
from kawkab.ui.bridge_handlers.bridge_external import ExternalHandler
from kawkab.ui.bridge_handlers.bridge_lifecycle import LifecycleHandler
from kawkab.ui.bridge_handlers.bridge_storage import StorageHandler
from kawkab.ui.bridge_handlers.bridge_video import VideoHandler

# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def mock_bridge():
    return MagicMock()


@pytest.fixture
def mock_data_export_service():
    svc = MagicMock()
    svc.export_match_csv = AsyncMock(return_value=Path("/tmp/export.csv"))
    svc.export_match_json = AsyncMock(return_value=Path("/tmp/export.json"))
    svc.export_statsbomb_compatible = AsyncMock(return_value=Path("/tmp/statsbomb_export.json"))
    return svc


@pytest.fixture
def mock_clip_service():
    svc = MagicMock()
    svc.extract_event_clips = AsyncMock(return_value=["clip1.mp4", "clip2.mp4"])
    return svc


@pytest.fixture
def mock_storage():
    svc = MagicMock()
    svc.get_match = AsyncMock(
        return_value={
            "id": 1,
            "name": "Test Match",
            "match_date": "2025-01-01",
            "video_path": "/videos/test.mp4",
            "home_team": "Home",
            "away_team": "Away",
        }
    )
    svc.get_match_events = AsyncMock(
        return_value=[
            {"type": "shot", "team": "home", "on_target": True, "timestamp": 120.0},
            {"type": "shot", "team": "away", "on_target": False, "timestamp": 240.0},
            {"type": "pass", "team": "home"},
            {"type": "goal", "team": "home"},
            {"type": "goal", "team": "home"},
        ]
    )
    svc.get_reports = AsyncMock(return_value=[{"report_text": "Good performance"}])
    svc.update_event = AsyncMock(return_value=True)
    svc.delete_event = AsyncMock(return_value=True)
    svc.update_match_football_data = AsyncMock()
    svc.update_match_bzzoiro = AsyncMock()
    svc.update_match_apifootball = AsyncMock()
    return svc


@pytest.fixture
def mock_feedback_service():
    svc = MagicMock()
    svc.submit_feedback = AsyncMock(return_value=42)
    svc.submit_issue = AsyncMock(return_value=99)
    svc.get_summary_stats = AsyncMock(return_value={"total": 10, "avg_rating": 4.2})
    return svc


@pytest.fixture
def mock_player_profile_service():
    svc = MagicMock()
    svc.get_all_profiles = AsyncMock(return_value=[])
    svc.create_profile = AsyncMock()
    profile = MagicMock()
    profile.id = 100
    svc.create_profile.return_value = profile
    return svc


@pytest.fixture
def export_handler(mock_bridge, mock_data_export_service, mock_clip_service, mock_storage):
    services = {
        "data_export_service": mock_data_export_service,
        "clip_service": mock_clip_service,
        "storage_service": mock_storage,
    }
    return ExportHandler(mock_bridge, services)


@pytest.fixture
def external_football_handler(mock_bridge, mock_storage, mock_player_profile_service):
    svc = MagicMock()
    svc.check_status = AsyncMock(return_value={"available": True})
    svc.search_team = AsyncMock(return_value=[{"id": 1, "name": "FC Test"}])
    svc.import_team_squad = AsyncMock(return_value=[])
    svc.verify_match = AsyncMock(return_value={"api_home": 1, "api_away": 0, "verified": True})
    svc.get_standings = AsyncMock(return_value=[{"position": 1, "team": "FC Test"}])
    svc.get_competitions = AsyncMock(return_value=[{"id": "CL", "name": "Champions League"}])
    svc.get_team_matches = AsyncMock(return_value=[{"id": 1, "home": "A", "away": "B"}])

    services = {
        "football_data_service": svc,
        "player_profile_service": mock_player_profile_service,
        "storage_service": mock_storage,
    }
    return ExternalHandler(mock_bridge, services)


@pytest.fixture
def external_bzzoiro_handler(mock_bridge, mock_storage, mock_player_profile_service):
    svc = MagicMock()
    svc.check_status = AsyncMock(return_value={"available": True})
    svc.search_team = AsyncMock(return_value=[{"id": 1, "name": "BZ Team"}])
    svc.get_team_squad = AsyncMock(
        return_value=[
            {
                "id": 10,
                "name": "Player A",
                "jersey_number": 7,
                "position": "FW",
                "nationality": "BR",
                "date_of_birth": "1995-01-01",
            }
        ]
    )
    svc.get_match_detail = AsyncMock(
        return_value={
            "home_score": 2,
            "away_score": 0,
            "home_team": "Home",
            "away_team": "Away",
        }
    )
    svc.get_standings = AsyncMock(return_value=[{"rank": 1, "team": "FC"}])
    svc.get_leagues = AsyncMock(return_value=[{"id": "L1", "name": "League 1"}])
    svc.get_team_matches = AsyncMock(return_value=[{"id": 1}])
    svc.get_live_events = AsyncMock(return_value=[{"id": 99, "home": "A", "away": "B"}])
    svc.get_predictions = AsyncMock(return_value={"winner": "home"})
    svc.get_match_stats = AsyncMock(return_value={"possession": 60})

    services = {
        "bzzoiro_service": svc,
        "player_profile_service": mock_player_profile_service,
        "storage_service": mock_storage,
    }
    return ExternalHandler(mock_bridge, services)


@pytest.fixture
def external_easy_handler(mock_bridge):
    svc = MagicMock()
    svc.check_available = MagicMock(return_value=True)
    svc.get_live_events = MagicMock(return_value=[{"id": 1}])
    svc.get_event = MagicMock(return_value={"id": 1, "home": "A", "away": "B"})
    svc.get_match_incidents = MagicMock(return_value=[{"type": "goal"}])
    svc.get_player = MagicMock(return_value={"id": 1, "name": "P1"})
    svc.search_events = MagicMock(return_value=[{"id": 2}])

    services = {"easy_soccer_service": svc}
    return ExternalHandler(mock_bridge, services)


@pytest.fixture
def external_apifb_handler(mock_bridge, mock_storage, mock_player_profile_service):
    svc = MagicMock()
    svc.check_status = AsyncMock(return_value={"available": True})
    svc.search_team = AsyncMock(return_value=[{"id": 1, "name": "API Team"}])
    svc.get_team_squad = AsyncMock(
        return_value=[
            {
                "id": 20,
                "name": "APlayer",
                "jersey_number": 10,
                "position": "MF",
            }
        ]
    )
    svc.get_standings = AsyncMock(return_value=[{"rank": 1}])
    svc.get_fixtures = AsyncMock(return_value=[{"id": 1}])
    svc.get_fixture_detail = AsyncMock(
        return_value={
            "home_score": 2,
            "away_score": 0,
            "home_team": "Home",
            "away_team": "Away",
        }
    )
    svc.get_predictions = AsyncMock(return_value={"winner": "home"})
    svc.get_live_fixtures = AsyncMock(return_value=[{"id": 2}])

    services = {
        "api_football_service": svc,
        "player_profile_service": mock_player_profile_service,
        "storage_service": mock_storage,
    }
    return ExternalHandler(mock_bridge, services)


@pytest.fixture
def external_tsdb_handler(mock_bridge):
    svc = MagicMock()
    svc.get_all_leagues = AsyncMock(return_value=[{"id": 1}])
    tsdb_team = SimpleNamespace(
        id=1,
        name="TS Team",
        league_name="L1",
        league_id=10,
        badge_url="/badge.png",
        stadium="S",
        location="L",
        formed_year=1900,
        api_football_id=100,
    )
    svc.search_teams = AsyncMock(return_value=[tsdb_team])

    tsdb_standing = SimpleNamespace(
        rank=1,
        team_name="TS",
        team_id=1,
        badge_url="",
        played=10,
        won=5,
        drawn=3,
        lost=2,
        goals_for=20,
        goals_against=10,
        goal_diff=10,
        points=18,
        form="WDL",
        description="",
    )
    svc.get_standings = AsyncMock(return_value=[tsdb_standing])

    tsdb_event = SimpleNamespace(
        id=1,
        event_name="E1",
        home_team="H",
        away_team="A",
        home_score=2,
        away_score=1,
        round="1",
        date="2025-01-01",
        time="15:00",
        league_name="L1",
    )
    svc.get_team_events_last = AsyncMock(return_value=[tsdb_event])
    svc.get_team_events_next = AsyncMock(return_value=[])

    tsdb_team_info = SimpleNamespace(
        id=1,
        name="TS",
        alternate_name="",
        league_name="L1",
        league_id=10,
        badge_url="",
        stadium="S",
        stadium_capacity=50000,
        location="L",
        formed_year=1900,
        description="A club",
        api_football_id=100,
    )
    svc.get_team = AsyncMock(return_value=tsdb_team_info)

    services = {"thesportsdb_service": svc}
    return ExternalHandler(mock_bridge, services)


@pytest.fixture
def external_statsbomb_handler(mock_bridge, mock_storage):
    svc = MagicMock()
    svc.get_competitions = AsyncMock(
        return_value=[
            MagicMock(
                competition_id=1,
                season_id=2,
                competition_name="UCL",
                country_name="Europe",
                season_name="2024-25",
                competition_gender="male",
                competition_international=True,
                competition_youth=False,
                has_360=False,
            ),
        ]
    )
    svc.get_matches = AsyncMock(
        return_value=[
            MagicMock(
                match_id=10,
                home_team="H",
                away_team="A",
                home_score=2,
                away_score=0,
                match_date="2025-01-01",
                competition_stage="Group",
                stadium="S",
                has_360=False,
            ),
        ]
    )
    svc.get_events = AsyncMock(
        return_value=[
            MagicMock(
                event_type="Shot",
                team="H",
                player="P1",
                xg=0.5,
                minute=30,
                outcome="Goal",
                shot_body_part="Right",
                shot_type="Open",
            ),
        ]
    )
    svc.get_lineups = AsyncMock(
        return_value=[
            MagicMock(team_name="H", team_id=1, players=["P1", "P2"]),
        ]
    )
    svc.import_match_to_db = AsyncMock(return_value=100)
    svc.search_team_matches = AsyncMock(return_value=[])

    services = {"statsbomb_service": svc, "storage_service": mock_storage}
    return ExternalHandler(mock_bridge, services)


@pytest.fixture
def external_of_handler(mock_bridge):
    svc = MagicMock()
    svc.get_matches = AsyncMock(
        return_value=[
            SimpleNamespace(
                competition="PL",
                round="1",
                date="2025-01-01",
                time="15:00",
                home_team="H",
                away_team="A",
                home_score=2,
                away_score=1,
                half_time_home=1,
                half_time_away=0,
            ),
        ]
    )
    svc.get_competitions = MagicMock(
        return_value=[
            SimpleNamespace(id="en.1", name="Premier League", seasons=["2024-25"]),
        ]
    )
    svc.search_team_matches = AsyncMock(
        return_value=[
            SimpleNamespace(
                competition="PL",
                season="2024-25",
                round="1",
                date="2025-01-01",
                home_team="H",
                away_team="A",
                home_score=2,
                away_score=1,
            ),
        ]
    )
    svc.get_worldcup_matches = AsyncMock(
        return_value=[
            SimpleNamespace(
                round="Final",
                date="2026-07-15",
                home_team="BRA",
                away_team="ARG",
                home_score=3,
                away_score=1,
            ),
        ]
    )
    svc.get_all_worldcup_years = MagicMock(return_value=[2018, 2022, 2026])

    services = {"openfootball_service": svc}
    return ExternalHandler(mock_bridge, services)


@pytest.fixture
def external_empty_bridge(mock_bridge):
    return ExternalHandler(mock_bridge, {})


@pytest.fixture
def lifecycle_handler(mock_bridge):
    svc = MagicMock()
    svc._system_info = {"gpu_name": "NVIDIA RTX 4090", "ram_gb": 32}
    svc.classify_gpu_tier = MagicMock(return_value="high")
    svc.recommend_settings = MagicMock(return_value={"model": "m", "skip": 3})

    profiler = MagicMock()
    profiler.report = MagicMock()
    report = MagicMock()
    report.to_dict = MagicMock(return_value={"total_calls": 42, "total_time": 1.5})
    profiler.report.return_value = report

    cv = MagicMock()
    cv.model_size = "m"

    services = {"benchmark_service": svc, "profiler": profiler, "cv_service": cv, "frame_skip": 5}
    return LifecycleHandler(mock_bridge, services)


@pytest.fixture
def lifecycle_handler_minimal(mock_bridge):
    return LifecycleHandler(mock_bridge, {})


@pytest.fixture
def storage_handler(mock_bridge, mock_storage, mock_feedback_service):
    services = {"storage_service": mock_storage, "feedback_service": mock_feedback_service}
    return StorageHandler(mock_bridge, services)


@pytest.fixture
def storage_handler_no_feedback(mock_bridge):
    services = {"storage_service": MagicMock()}
    return StorageHandler(mock_bridge, services)


@pytest.fixture
def video_handler(mock_bridge):
    svc = MagicMock()
    svc.target_fps = 30
    svc.buffer_size = 256
    svc._alert_rules = [1, 2]
    svc._subscribers = [MagicMock()]

    services = {"realtime_service": svc}
    return VideoHandler(mock_bridge, services)


@pytest.fixture
def video_handler_no_realtime(mock_bridge):
    return VideoHandler(mock_bridge, {})


# ========================================================================
# ExportHandler Tests
# ========================================================================


class TestExportHandler:
    """ExportHandler — export_match_csv, export_match_json, export_report_pdf, extract_event_clips."""

    def test_init(self, export_handler):
        assert export_handler._bridge is not None
        assert export_handler._services is not None

    @pytest.mark.asyncio
    async def test_export_csv_success(self, export_handler):
        result = json.loads(await export_handler.export_match_csv("1"))
        assert result["success"] is True
        assert "path" in result

    @pytest.mark.asyncio
    async def test_export_csv_no_service(self, export_handler):
        export_handler._services.pop("data_export_service")
        result = json.loads(await export_handler.export_match_csv("1"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_export_csv_exception(self, export_handler, mock_data_export_service):
        mock_data_export_service.export_match_csv.side_effect = RuntimeError("boom")
        result = json.loads(await export_handler.export_match_csv("1"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_export_json_success(self, export_handler):
        result = json.loads(await export_handler.export_match_json("1"))
        assert result["success"] is True
        assert "path" in result

    @pytest.mark.asyncio
    async def test_export_json_no_service(self, export_handler):
        export_handler._services.pop("data_export_service")
        result = json.loads(await export_handler.export_match_json("1"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_export_report_pdf_success(self, export_handler):
        result = json.loads(await export_handler.export_report_pdf("1", "en"))
        assert result["success"] is True
        assert "path" in result

    @pytest.mark.asyncio
    async def test_export_report_pdf_arabic(self, export_handler):
        result = json.loads(await export_handler.export_report_pdf("1", "ar"))
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_export_report_pdf_no_match(self, export_handler, mock_storage):
        mock_storage.get_match.return_value = None
        result = json.loads(await export_handler.export_report_pdf("1", "en"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_extract_event_clips_success(self, export_handler):
        result = json.loads(await export_handler.extract_event_clips("1"))
        assert result["success"] is True
        assert "clips" in result

    @pytest.mark.asyncio
    async def test_extract_event_clips_no_video(self, export_handler, mock_storage):
        mock_storage.get_match.return_value = {"id": 1, "name": "No Video"}
        result = json.loads(await export_handler.extract_event_clips("1"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_extract_event_clips_no_shots(self, export_handler, mock_storage):
        mock_storage.get_match_events.return_value = [{"type": "pass"}]
        result = json.loads(await export_handler.extract_event_clips("1"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_extract_event_clips_no_service(self, export_handler):
        export_handler._services.pop("clip_service")
        result = json.loads(await export_handler.extract_event_clips("1"))
        assert "error" in result

    # ── StatsBomb export ──

    @pytest.mark.asyncio
    async def test_export_statsbomb_success(
        self, export_handler, mock_data_export_service, tmp_path
    ):
        src = tmp_path / "src_statsbomb.json"
        src.write_text('{"match_id": 1, "events": []}', encoding="utf-8")
        mock_data_export_service.export_statsbomb_compatible.return_value = src
        output = tmp_path / "statsbomb_test.json"
        result = json.loads(await export_handler.export_match_statsbomb("1", str(output)))
        assert result["success"] is True
        assert "path" in result
        assert output.exists()

    @pytest.mark.asyncio
    async def test_export_statsbomb_exception(self, export_handler, mock_data_export_service):
        mock_data_export_service.export_statsbomb_compatible.side_effect = RuntimeError(
            "export failed"
        )
        result = json.loads(await export_handler.export_match_statsbomb("1", "/tmp/test.json"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_export_statsbomb_no_service(self, export_handler):
        export_handler._services.pop("data_export_service")
        result = json.loads(await export_handler.export_match_statsbomb("1", "/tmp/test.json"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_export_statsbomb_missing_match(self, export_handler, mock_data_export_service):
        mock_data_export_service.export_statsbomb_compatible.side_effect = ValueError(
            "Match 999 not found"
        )
        result = json.loads(await export_handler.export_match_statsbomb("999", "/tmp/test.json"))
        assert "error" in result


# ========================================================================
# ExternalHandler Tests
# ========================================================================


class TestExternalHandler:
    """ExternalHandler — 51+ methods across 7 API providers."""

    def test_init(self, mock_bridge):
        h = ExternalHandler(mock_bridge, {})
        assert h._bridge is not None

    # ── service-unavailable (no services dict) ──

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method,args",
        [
            ("check_football_data_status", []),
            ("check_bzzoiro_status", []),
            ("check_easy_soccer_status", []),
            ("check_apifootball_status", []),
            ("check_thesportsdb_status", []),
            ("check_statsbomb_status", []),
            ("check_openfootball_status", []),
            ("search_football_team", ["test"]),
            ("search_bzzoiro_team", ["test"]),
            ("search_apifootball_team", ["test"]),
            ("get_football_competitions", []),
            ("get_bzzoiro_leagues", []),
            ("get_bzzoiro_live", []),
            ("get_apifootball_live", []),
            ("get_easy_soccer_live", []),
            ("get_statsbomb_competitions", []),
            ("get_openfootball_competitions", []),
        ],
    )
    async def test_service_unavailable_returns_default(self, external_empty_bridge, method, args):
        h = external_empty_bridge
        result = json.loads(await getattr(h, method)(*args))
        assert isinstance(result, dict)

    # ── football-data.org ──

    @pytest.mark.asyncio
    async def test_football_check_status(self, external_football_handler):
        r = json.loads(await external_football_handler.check_football_data_status())
        assert r.get("available") is True

    @pytest.mark.asyncio
    async def test_football_search_team(self, external_football_handler):
        r = json.loads(await external_football_handler.search_football_team("test"))
        assert len(r["teams"]) == 1

    @pytest.mark.asyncio
    async def test_football_import_squad(self, external_football_handler, mock_storage):
        r = json.loads(
            await external_football_handler.import_football_team_squad("1", "42", "home")
        )
        assert r["success"] is True
        mock_storage.update_match_football_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_football_import_squad_bad_side(self, external_football_handler):
        r = json.loads(await external_football_handler.import_football_team_squad("1", "42", "bad"))
        assert r["success"] is False

    @pytest.mark.asyncio
    async def test_football_verify_match(self, external_football_handler, mock_storage):
        r = json.loads(await external_football_handler.verify_match_with_api("1", "100"))
        assert r["success"] is True
        mock_storage.update_match_football_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_football_standings(self, external_football_handler):
        r = json.loads(await external_football_handler.get_football_standings("CL"))
        assert r["success"] is True

    @pytest.mark.asyncio
    async def test_football_team_matches(self, external_football_handler):
        r = json.loads(
            await external_football_handler.get_football_team_matches(
                "1", "2025-01-01", "2025-12-31"
            )
        )
        assert len(r["matches"]) == 1

    @pytest.mark.asyncio
    async def test_football_verify_match_not_found(self, external_football_handler, mock_storage):
        mock_storage.get_match.return_value = None
        r = json.loads(await external_football_handler.verify_match_with_api("1", "100"))
        assert "error" in r

    @pytest.mark.asyncio
    async def test_football_verify_match_api_fails(self, external_football_handler):
        svc = external_football_handler._services["football_data_service"]
        svc.verify_match.return_value = None
        r = json.loads(await external_football_handler.verify_match_with_api("1", "100"))
        assert "error" in r

    # ── Bzzoiro ──

    @pytest.mark.asyncio
    async def test_bzzoiro_check_status(self, external_bzzoiro_handler):
        r = json.loads(await external_bzzoiro_handler.check_bzzoiro_status())
        assert r.get("available") is True

    @pytest.mark.asyncio
    async def test_bzzoiro_import_squad(self, external_bzzoiro_handler, mock_storage):
        r = json.loads(await external_bzzoiro_handler.import_bzzoiro_team_squad("1", "42", "away"))
        assert r["success"] is True
        assert len(r["created"]) == 1
        mock_storage.update_match_bzzoiro.assert_called_once()

    @pytest.mark.asyncio
    async def test_bzzoiro_import_squad_skipped(
        self, external_bzzoiro_handler, mock_player_profile_service
    ):
        existing = MagicMock()
        existing.jersey_number = 7
        mock_player_profile_service.get_all_profiles.return_value = [existing]
        r = json.loads(await external_bzzoiro_handler.import_bzzoiro_team_squad("1", "42", "home"))
        assert r["skipped"] == 1

    @pytest.mark.asyncio
    async def test_bzzoiro_verify_match_ok(self, external_bzzoiro_handler, mock_storage):
        r = json.loads(await external_bzzoiro_handler.verify_match_bzzoiro("1", "200"))
        assert r["success"] is True
        assert r["match_ok"] is True
        mock_storage.update_match_bzzoiro.assert_called_once()

    @pytest.mark.asyncio
    async def test_bzzoiro_verify_match_not_found(self, external_bzzoiro_handler, mock_storage):
        mock_storage.get_match.return_value = None
        r = json.loads(await external_bzzoiro_handler.verify_match_bzzoiro("1", "200"))
        assert "error" in r

    @pytest.mark.asyncio
    async def test_bzzoiro_standings(self, external_bzzoiro_handler):
        r = json.loads(await external_bzzoiro_handler.get_bzzoiro_standings("L1"))
        assert len(r["standings"]) == 1

    @pytest.mark.asyncio
    async def test_bzzoiro_predictions(self, external_bzzoiro_handler):
        r = json.loads(await external_bzzoiro_handler.get_bzzoiro_predictions("200"))
        assert "predictions" in r

    @pytest.mark.asyncio
    async def test_bzzoiro_predictions_none(self, external_bzzoiro_handler):
        svc = external_bzzoiro_handler._services["bzzoiro_service"]
        svc.get_predictions.return_value = None
        r = json.loads(await external_bzzoiro_handler.get_bzzoiro_predictions("200"))
        assert "error" in r

    @pytest.mark.asyncio
    async def test_bzzoiro_match_stats(self, external_bzzoiro_handler):
        r = json.loads(await external_bzzoiro_handler.get_bzzoiro_match_stats("200"))
        assert "stats" in r

    @pytest.mark.asyncio
    async def test_bzzoiro_team_matches(self, external_bzzoiro_handler):
        r = json.loads(
            await external_bzzoiro_handler.get_bzzoiro_team_matches("1", "2025-01-01", "2025-12-31")
        )
        assert len(r["matches"]) == 1

    # ── EasySoccer ──

    @pytest.mark.asyncio
    async def test_easy_check_status(self, external_easy_handler):
        r = json.loads(await external_easy_handler.check_easy_soccer_status())
        assert r["available"] is True

    @pytest.mark.asyncio
    async def test_easy_get_event(self, external_easy_handler):
        r = json.loads(await external_easy_handler.get_easy_soccer_event("1"))
        assert "event" in r

    @pytest.mark.asyncio
    async def test_easy_get_event_not_found(self, external_easy_handler):
        svc = external_easy_handler._services["easy_soccer_service"]
        svc.get_event.return_value = None
        r = json.loads(await external_easy_handler.get_easy_soccer_event("1"))
        assert "error" in r

    @pytest.mark.asyncio
    async def test_easy_incidents(self, external_easy_handler):
        r = json.loads(await external_easy_handler.get_easy_soccer_incidents("1"))
        assert len(r["incidents"]) == 1

    @pytest.mark.asyncio
    async def test_easy_player(self, external_easy_handler):
        r = json.loads(await external_easy_handler.get_easy_soccer_player("1"))
        assert "player" in r

    @pytest.mark.asyncio
    async def test_easy_search_events(self, external_easy_handler):
        r = json.loads(await external_easy_handler.search_easy_soccer_events("2025-01-01"))
        assert len(r["events"]) == 1

    # ── API-Football ──

    @pytest.mark.asyncio
    async def test_apifb_check_status(self, external_apifb_handler):
        r = json.loads(await external_apifb_handler.check_apifootball_status())
        assert r.get("available") is True

    @pytest.mark.asyncio
    async def test_apifb_import_squad(self, external_apifb_handler, mock_storage):
        r = json.loads(await external_apifb_handler.import_apifootball_squad("1", "42", "home"))
        assert r["success"] is True
        mock_storage.update_match_apifootball.assert_called_once()

    @pytest.mark.asyncio
    async def test_apifb_fixture_detail(self, external_apifb_handler):
        r = json.loads(await external_apifb_handler.get_apifootball_fixture_detail("500"))
        assert "fixture" in r

    @pytest.mark.asyncio
    async def test_apifb_fixture_detail_not_found(self, external_apifb_handler):
        svc = external_apifb_handler._services["api_football_service"]
        svc.get_fixture_detail.return_value = None
        r = json.loads(await external_apifb_handler.get_apifootball_fixture_detail("500"))
        assert "error" in r

    @pytest.mark.asyncio
    async def test_apifb_predictions(self, external_apifb_handler):
        r = json.loads(await external_apifb_handler.get_apifootball_predictions("500"))
        assert "predictions" in r

    @pytest.mark.asyncio
    async def test_apifb_verify_match_ok(self, external_apifb_handler, mock_storage):
        r = json.loads(await external_apifb_handler.verify_match_apifootball("1", "500"))
        assert r["success"] is True
        assert r["match_ok"] is True

    @pytest.mark.asyncio
    async def test_apifb_verify_match_mismatch(self, external_apifb_handler, mock_storage):
        svc = external_apifb_handler._services["api_football_service"]
        svc.get_fixture_detail.return_value = {
            "home_score": 5,
            "away_score": 0,
            "home_team": "H",
            "away_team": "A",
        }
        r = json.loads(await external_apifb_handler.verify_match_apifootball("1", "500"))
        assert r["match_ok"] is False

    @pytest.mark.asyncio
    async def test_apifb_fixtures(self, external_apifb_handler):
        r = json.loads(await external_apifb_handler.get_apifootball_fixtures("1", 2024))
        assert len(r["matches"]) == 1

    # ── TheSportsDB ──

    @pytest.mark.asyncio
    async def test_tsdb_check_status(self, external_tsdb_handler):
        r = json.loads(await external_tsdb_handler.check_thesportsdb_status())
        assert r["available"] is True

    @pytest.mark.asyncio
    async def test_tsdb_search_team(self, external_tsdb_handler):
        r = json.loads(await external_tsdb_handler.search_thesportsdb_team("test"))
        assert len(r["teams"]) == 1
        assert r["teams"][0]["name"] == "TS Team"

    @pytest.mark.asyncio
    async def test_tsdb_standings(self, external_tsdb_handler):
        r = json.loads(await external_tsdb_handler.get_thesportsdb_standings("10"))
        assert len(r["standings"]) == 1
        assert r["standings"][0]["rank"] == 1

    @pytest.mark.asyncio
    async def test_tsdb_team_events_last(self, external_tsdb_handler):
        r = json.loads(await external_tsdb_handler.get_thesportsdb_team_events_last("1"))
        assert len(r["events"]) == 1

    @pytest.mark.asyncio
    async def test_tsdb_team_events_next(self, external_tsdb_handler):
        r = json.loads(await external_tsdb_handler.get_thesportsdb_team_events_next("1"))
        assert r["events"] == []

    @pytest.mark.asyncio
    async def test_tsdb_team_info(self, external_tsdb_handler):
        r = json.loads(await external_tsdb_handler.get_thesportsdb_team_info("1"))
        assert r["team"]["name"] == "TS"

    @pytest.mark.asyncio
    async def test_tsdb_team_info_not_found(self, external_tsdb_handler):
        svc = external_tsdb_handler._services["thesportsdb_service"]
        svc.get_team.return_value = None
        r = json.loads(await external_tsdb_handler.get_thesportsdb_team_info("1"))
        assert r["team"] is None

    # ── StatsBomb ──

    @pytest.mark.asyncio
    async def test_statsbomb_check_status(self, external_statsbomb_handler):
        r = json.loads(await external_statsbomb_handler.check_statsbomb_status())
        assert r["available"] is True

    @pytest.mark.asyncio
    async def test_statsbomb_competitions(self, external_statsbomb_handler):
        r = json.loads(await external_statsbomb_handler.get_statsbomb_competitions())
        assert len(r["competitions"]) == 1

    @pytest.mark.asyncio
    async def test_statsbomb_matches(self, external_statsbomb_handler):
        r = json.loads(await external_statsbomb_handler.get_statsbomb_matches(1, 2))
        assert len(r["matches"]) == 1

    @pytest.mark.asyncio
    async def test_statsbomb_events(self, external_statsbomb_handler):
        r = json.loads(await external_statsbomb_handler.get_statsbomb_events(10))
        assert "summary" in r
        assert r["summary"]["shots"] == 1
        assert r["summary"]["total_xg"] == 0.5

    @pytest.mark.asyncio
    async def test_statsbomb_lineups(self, external_statsbomb_handler):
        r = json.loads(await external_statsbomb_handler.get_statsbomb_lineups(10))
        assert len(r["lineups"]) == 1

    @pytest.mark.asyncio
    async def test_statsbomb_import_match(self, external_statsbomb_handler):
        r = json.loads(await external_statsbomb_handler.import_statsbomb_match(10))
        assert r["imported"] == 100

    @pytest.mark.asyncio
    async def test_statsbomb_import_no_storage(self, mock_bridge):
        h = ExternalHandler(mock_bridge, {"statsbomb_service": MagicMock()})
        r = json.loads(await h.import_statsbomb_match(10))
        assert r["imported"] == 0

    # ── OpenFootball ──

    @pytest.mark.asyncio
    async def test_of_check_status(self, external_of_handler):
        r = json.loads(await external_of_handler.check_openfootball_status())
        assert r["available"] is True

    @pytest.mark.asyncio
    async def test_of_competitions(self, external_of_handler):
        r = json.loads(await external_of_handler.get_openfootball_competitions())
        assert len(r["competitions"]) == 1

    @pytest.mark.asyncio
    async def test_of_matches(self, external_of_handler):
        r = json.loads(await external_of_handler.get_openfootball_matches("en.1", "2024-25"))
        assert len(r["matches"]) == 1

    @pytest.mark.asyncio
    async def test_of_search_team(self, external_of_handler):
        r = json.loads(await external_of_handler.search_openfootball_team("test"))
        assert len(r["matches"]) == 1

    @pytest.mark.asyncio
    async def test_of_search_team_empty(self, external_of_handler):
        r = json.loads(await external_of_handler.search_openfootball_team(""))
        assert r["matches"] == []

    @pytest.mark.asyncio
    async def test_of_worldcup(self, external_of_handler):
        r = json.loads(await external_of_handler.get_openfootball_worldcup(2026))
        assert len(r["matches"]) == 1
        assert "years" in r

    # ── Exception handling (parametrized across providers) ──

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method,args,service_key",
        [
            ("search_football_team", ["x"], "football_data_service"),
            ("search_bzzoiro_team", ["x"], "bzzoiro_service"),
            ("search_apifootball_team", ["x"], "api_football_service"),
            ("get_football_competitions", [], "football_data_service"),
            ("get_bzzoiro_leagues", [], "bzzoiro_service"),
            ("get_easy_soccer_live", [], "easy_soccer_service"),
            ("get_apifootball_fixtures", ["1", 2024], "api_football_service"),
            ("get_statsbomb_competitions", [], "statsbomb_service"),
            ("get_openfootball_matches", ["en.1", "2024-25"], "openfootball_service"),
        ],
    )
    async def test_exception_returns_json_error(
        self, mock_bridge, mock_storage, method, args, service_key
    ):
        svc = MagicMock()
        svc_mock = (
            AsyncMock()
            if method
            in ("check_football_data_status", "check_apifootball_status", "check_statsbomb_status")
            else MagicMock()
        )
        if isinstance(svc_mock, AsyncMock):
            svc_mock.side_effect = RuntimeError("api down")
        else:
            setattr(
                svc_mock,
                svc_mock._method_map_.get(
                    method.replace("search_", "search_").replace("get_", "get_"), "_"
                ),
                MagicMock(side_effect=RuntimeError("api down")),
            ) if False else None

        # Simpler: make the real method fail by patching the service method
        handler = ExternalHandler(mock_bridge, {service_key: svc, "storage_service": mock_storage})
        fn = getattr(handler, method)
        # Patch so ANY call to the service raises
        mock_method = (
            AsyncMock(side_effect=RuntimeError("boom"))
            if method
            in (
                "check_football_data_status",
                "check_apifootball_status",
                "check_statsbomb_status",
            )
            else MagicMock(side_effect=RuntimeError("boom"))
        )

        if (
            method == "search_football_team"
            or method == "search_bzzoiro_team"
            or method == "search_apifootball_team"
        ):
            svc.search_team = mock_method
        elif method == "get_football_competitions":
            svc.get_competitions = mock_method
        elif method == "get_bzzoiro_leagues":
            svc.get_leagues = mock_method
        elif method == "get_easy_soccer_live":
            svc.get_live_events = mock_method
        elif method == "get_apifootball_fixtures":
            svc.get_fixtures = mock_method
        elif method == "get_statsbomb_competitions":
            svc.get_competitions = mock_method
        elif method == "get_openfootball_matches":
            svc.get_matches = mock_method

        r = json.loads(await fn(*args))
        assert isinstance(r, dict)


# ========================================================================
# LifecycleHandler Tests
# ========================================================================


class TestLifecycleHandler:
    """LifecycleHandler — get_gpu_info, profiler_status, profiler_reset, metrics_text."""

    def test_init(self, lifecycle_handler):
        assert lifecycle_handler._bridge is not None

    def test_get_gpu_info_with_benchmark(self, lifecycle_handler):
        r = json.loads(lifecycle_handler.get_gpu_info())
        assert r["gpu_name"] == "NVIDIA RTX 4090"
        assert r["tier"] == "high"
        assert r["current_settings"]["frame_skip"] == 5

    def test_get_gpu_info_no_benchmark(self, lifecycle_handler_minimal):
        r = json.loads(lifecycle_handler_minimal.get_gpu_info())
        assert r["gpu_name"] == "unknown"

    def test_profiler_status(self, lifecycle_handler):
        r = json.loads(lifecycle_handler.profiler_status())
        assert r["total_calls"] == 42

    def test_profiler_status_no_service(self, lifecycle_handler_minimal):
        r = json.loads(lifecycle_handler_minimal.profiler_status())
        assert "error" in r

    def test_profiler_reset(self, lifecycle_handler):
        r = json.loads(lifecycle_handler.profiler_reset())
        assert r["ok"] is True

    def test_profiler_reset_no_service(self, lifecycle_handler_minimal):
        r = json.loads(lifecycle_handler_minimal.profiler_reset())
        assert "error" in r

    def test_metrics_text(self, lifecycle_handler):
        text = lifecycle_handler.metrics_text()
        assert isinstance(text, str)

    def test_metrics_text_minimal(self, lifecycle_handler_minimal):
        text = lifecycle_handler_minimal.metrics_text()
        assert isinstance(text, str)

    def test_get_gpu_info_exception(self, lifecycle_handler):
        svc = lifecycle_handler._services["benchmark_service"]
        svc._system_info = {"gpu_name": "NVIDIA RTX 4090"}
        # Make `classify_gpu_tier` raise to trigger error path
        import kawkab.services.benchmark_service as bm

        orig = bm.BenchmarkService.classify_gpu_tier
        bm.BenchmarkService.classify_gpu_tier = MagicMock(side_effect=RuntimeError("boom"))
        try:
            r = json.loads(lifecycle_handler.get_gpu_info())
            assert "error" in r
        finally:
            bm.BenchmarkService.classify_gpu_tier = orig


# ========================================================================
# StorageHandler Tests
# ========================================================================


class TestStorageHandler:
    """StorageHandler — update_event, delete_event, submit_feedback, submit_issue, get_feedback_stats."""

    def test_init(self, storage_handler):
        assert storage_handler._bridge is not None

    @pytest.mark.asyncio
    async def test_update_event(self, storage_handler):
        r = json.loads(await storage_handler.update_event("1", json.dumps({"type": "goal"})))
        assert r["success"] is True

    @pytest.mark.asyncio
    async def test_update_event_bad_json(self, storage_handler):
        r = json.loads(await storage_handler.update_event("1", "not-json"))
        assert "error" in r

    @pytest.mark.asyncio
    async def test_delete_event(self, storage_handler):
        r = json.loads(await storage_handler.delete_event("1"))
        assert r["success"] is True

    @pytest.mark.asyncio
    async def test_delete_event_fails(self, storage_handler, mock_storage):
        mock_storage.delete_event.return_value = False
        r = json.loads(await storage_handler.delete_event("1"))
        assert r["success"] is False

    @pytest.mark.asyncio
    async def test_submit_feedback(self, storage_handler):
        fb = json.dumps({"coach_id": "c1", "match_id": 1, "overall_rating": 5, "comments": "Great"})
        r = json.loads(await storage_handler.submit_feedback(fb))
        assert r["feedback_id"] == 42
        assert r["status"] == "saved"

    @pytest.mark.asyncio
    async def test_submit_feedback_no_service(self, storage_handler_no_feedback):
        r = json.loads(await storage_handler_no_feedback.submit_feedback("{}"))
        assert "error" in r

    @pytest.mark.asyncio
    async def test_submit_feedback_bad_json(self, storage_handler):
        r = json.loads(await storage_handler.submit_feedback("not-json"))
        assert "error" in r

    @pytest.mark.asyncio
    async def test_submit_issue(self, storage_handler):
        issue = json.dumps({"category": "tracking", "severity": "high", "description": "Bug"})
        r = json.loads(await storage_handler.submit_issue(issue))
        assert r["issue_id"] == 99
        assert r["status"] == "saved"

    @pytest.mark.asyncio
    async def test_submit_issue_no_service(self, storage_handler_no_feedback):
        r = json.loads(await storage_handler_no_feedback.submit_issue("{}"))
        assert "error" in r

    @pytest.mark.asyncio
    async def test_get_feedback_stats(self, storage_handler):
        r = json.loads(await storage_handler.get_feedback_stats())
        assert r["total"] == 10
        assert r["avg_rating"] == 4.2

    @pytest.mark.asyncio
    async def test_get_feedback_stats_no_service(self, storage_handler_no_feedback):
        r = json.loads(await storage_handler_no_feedback.get_feedback_stats())
        assert "error" in r

    @pytest.mark.asyncio
    async def test_storage_exception(self, storage_handler, mock_storage):
        mock_storage.update_event.side_effect = RuntimeError("db fail")
        r = json.loads(await storage_handler.update_event("1", '{"x": 1}'))
        assert "error" in r


# ========================================================================
# VideoHandler Tests
# ========================================================================


class TestVideoHandler:
    """VideoHandler — sync_*, trim, reel_*, realtime_*."""

    def test_init(self, video_handler):
        assert video_handler._bridge is not None
        assert video_handler._sync_service is not None
        assert video_handler._highlight_reel is not None

    def test_sync_load(self, video_handler):
        with patch.object(
            video_handler._sync_service, "load_videos", return_value='{"ok": true}'
        ) as mock_load:
            r = video_handler.sync_load('["/v1.mp4", "/v2.mp4"]')
            assert r == '{"ok": true}'
            mock_load.assert_called_once()

    def test_sync_set_offset(self, video_handler):
        with patch.object(
            video_handler._sync_service, "set_offset", return_value='{"ok": true}'
        ) as _:
            r = video_handler.sync_set_offset(0, 1.5)
            assert r == '{"ok": true}'

    def test_sync_positions(self, video_handler):
        with patch.object(
            video_handler._sync_service, "get_sync_positions", return_value='{"positions": []}'
        ) as _:
            r = video_handler.sync_positions(120.0)
            assert r == '{"positions": []}'

    def test_sync_state(self, video_handler):
        with patch.object(
            video_handler._sync_service, "get_state", return_value='{"state": "ready"}'
        ) as _:
            r = video_handler.sync_state()
            assert r == '{"state": "ready"}'

    def test_sync_clear(self, video_handler):
        with patch.object(video_handler._sync_service, "clear", return_value='{"ok": true}') as _:
            r = video_handler.sync_clear()
            assert r == '{"ok": true}'

    def test_sync_load_bad_json(self, video_handler):
        r = json.loads(video_handler.sync_load("not-json"))
        assert "error" in r

    @pytest.mark.parametrize(
        "method,args",
        [
            ("sync_set_offset", (0, 1.5)),
            ("sync_positions", (120.0,)),
            ("sync_state", ()),
            ("sync_clear", ()),
        ],
    )
    def test_sync_methods_exception(self, video_handler, method, args):
        attr = {
            "sync_set_offset": "set_offset",
            "sync_positions": "get_sync_positions",
            "sync_state": "get_state",
            "sync_clear": "clear",
        }[method]
        with patch.object(video_handler._sync_service, attr, side_effect=RuntimeError("sync fail")):
            r = json.loads(getattr(video_handler, method)(*args))
            assert "error" in r

    def test_trim_video(self, video_handler):
        with (
            patch("kawkab.services.clip_service.ClipExtractionService") as mock_cls,
            patch(
                "kawkab.ui.bridge_handlers.bridge_video.SecurityValidator.validate_video_path",
                return_value=Path("/v/test.mp4"),
            ),
        ):
            instance = MagicMock()
            mock_cls.return_value = instance

            async def mock_extract(*a, **kw):
                return "/out/trim.mp4"

            instance.extract_clip = mock_extract
            r = json.loads(video_handler.trim_video("/v/test.mp4", 10.0, 20.0, "my_trim.mp4"))
            assert r["ok"] is True
            assert r["output"] == "/out/trim.mp4"

    def test_trim_video_invalid_path(self, video_handler):
        r = json.loads(video_handler.trim_video("", 0, 1))
        assert "error" in r

    def test_trim_video_exception(self, video_handler):
        with (
            patch("kawkab.services.clip_service.ClipExtractionService") as mock_cls,
            patch(
                "kawkab.ui.bridge_handlers.bridge_video.SecurityValidator.validate_video_path",
                return_value=Path("/v/test.mp4"),
            ),
        ):
            instance = MagicMock()
            mock_cls.return_value = instance

            async def mock_raise(*a, **kw):
                raise RuntimeError("trim fail")

            instance.extract_clip = mock_raise
            r = json.loads(video_handler.trim_video("/v/test.mp4", 0, 1))
            assert "error" in r

    def test_reel_compose(self, video_handler):
        clips_json = json.dumps(
            [{"video_path": "/v/a.mp4", "start_s": 0, "end_s": 10, "label": "goal"}]
        )
        with patch.object(
            video_handler._highlight_reel, "compose_reel", return_value='{"path": "/reel.mp4"}'
        ) as _:
            r = video_handler.reel_compose(clips_json, "reel.mp4")
            assert r == '{"path": "/reel.mp4"}'

    def test_reel_compose_bad_json(self, video_handler):
        r = json.loads(video_handler.reel_compose("not-json", "x.mp4"))
        assert "error" in r

    def test_reel_from_events(self, video_handler):
        events_json = json.dumps([{"type": "goal", "timestamp": 120, "team": "home"}])
        with patch.object(
            video_handler._highlight_reel,
            "make_reel_from_events",
            return_value='{"path": "/reel.mp4"}',
        ) as _:
            r = json.loads(video_handler.reel_from_events(1, events_json, "/v/test.mp4"))
            assert r["path"] == "/reel.mp4"
            assert "reel_id" in r

    def test_reel_from_events_bad_json(self, video_handler):
        r = json.loads(video_handler.reel_from_events(1, "bad", "/v/test.mp4"))
        assert "error" in r

    def test_realtime_status(self, video_handler):
        r = json.loads(video_handler.realtime_status())
        assert r["available"] is True
        assert r["target_fps"] == 30

    def test_realtime_status_no_service(self, video_handler_no_realtime):
        r = json.loads(video_handler_no_realtime.realtime_status())
        assert r["available"] is False

    def test_realtime_cancel(self, video_handler):
        r = json.loads(video_handler.realtime_cancel())
        assert r["ok"] is True

    def test_realtime_cancel_no_service(self, video_handler_no_realtime):
        r = json.loads(video_handler_no_realtime.realtime_cancel())
        assert "error" in r

    def test_realtime_subscribe_console(self, video_handler):
        with patch("kawkab.services.realtime_service.ConsoleSubscriber") as mock_cls:
            instance = MagicMock()
            mock_cls.return_value = instance
            r = json.loads(video_handler.realtime_subscribe_console())
            assert r["ok"] is True

    def test_realtime_subscribe_console_no_service(self, video_handler_no_realtime):
        r = json.loads(video_handler_no_realtime.realtime_subscribe_console())
        assert "error" in r

    def test_realtime_status_exception(self, video_handler):
        svc = video_handler._services["realtime_service"]
        type(svc).target_fps = PropertyMock(side_effect=RuntimeError("sensor fail"))
        r = json.loads(video_handler.realtime_status())
        assert "error" in r
