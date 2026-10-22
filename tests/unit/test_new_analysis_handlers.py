"""Tests for the bridge_analysis.py handlers added while fixing 6 broken
frontend<->Bridge calls (get_dashboard_stats, compare_players,
get_player_stats, get_match_players, search_external_player) plus
get_season_form (Task 8: wiring FormAnalyzer into the Dashboard). See
CLAUDE.md for the audit these came out of.

Uses a from-scratch AnalysisHandler with a mock services dict rather than
load_service_module, since bridge_handlers/ isn't a service module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler
from kawkab.ui.bridge_handlers.bridge_cloud import CloudCollabHandler
from kawkab.ui.bridge_handlers.bridge_domain import DomainHandler
from kawkab.ui.bridge_handlers.bridge_match_intel import MatchIntelHandler
from kawkab.ui.bridge_handlers.bridge_physical import PhysicalHandler
from kawkab.ui.bridge_handlers.bridge_recruitment import RecruitmentHandler


class MockStorageService:
    def __init__(self, matches=None, events_by_match=None, players_by_match=None):
        self._matches = matches or []
        self._events_by_match = events_by_match or {}
        self._players_by_match = players_by_match or {}

    async def get_all_matches(self):
        return self._matches

    async def get_match_events(self, match_id):
        return self._events_by_match.get(match_id, [])

    async def get_match_players(self, match_id):
        return self._players_by_match.get(match_id, [])


class MockPlayerProfileService:
    def __init__(self, profiles=None, appearances=None):
        self._profiles = profiles or {}
        self._appearances = appearances or {}

    async def get_profile(self, profile_id):
        return self._profiles.get(profile_id)

    async def get_profile_appearances(self, profile_id):
        return self._appearances.get(profile_id, [])


def _goal_event(team, timestamp=10.0):
    return {"event_type": "goal", "team": team, "timestamp": timestamp}


def _handler(storage=None, player_profiles=None):
    services = {
        "storage_service": storage or MockStorageService(),
        "player_profile_service": player_profiles,
    }
    return AnalysisHandler(bridge=None, services=services, rate_limiter=None)


_MOVED = {
    "get_xa_report": MatchIntelHandler,
    "get_pressing_report": MatchIntelHandler,
    "get_match_quality_score": MatchIntelHandler,
    "compute_goals_added": MatchIntelHandler,
    "get_gps_sessions": PhysicalHandler,
    "get_gps_samples": PhysicalHandler,
    "get_player_gps_summary": PhysicalHandler,
    "get_player_acwr": PhysicalHandler,
    "import_gps_file": PhysicalHandler,
    "generate_briefing": PhysicalHandler,
    "check_setpiece_status": DomainHandler,
    "analyze_setpieces": DomainHandler,
    "analyze_goalkeeper": DomainHandler,
    "analyze_goalkeeper_advanced": DomainHandler,
    "compute_xgot": DomainHandler,
    "analyze_substitutions": DomainHandler,
    "analyze_possession": DomainHandler,
    "analyze_match_psychology": DomainHandler,
    "get_law_summary": DomainHandler,
    "classify_event_rule": DomainHandler,
    "check_offside": DomainHandler,
    "infer_cards_tactically": DomainHandler,
    "fetch_external_cards": DomainHandler,
}


def _handler_for(method, storage=None, player_profiles=None, extra=None):
    """Build the handler that now owns `method` (post Phase-C1 split)."""
    services = {
        "storage_service": storage or MockStorageService(),
        "player_profile_service": player_profiles,
    }
    if extra:
        services.update(extra)
    cls = _MOVED.get(method, AnalysisHandler)
    return cls(bridge=None, services=services, rate_limiter=None)


class TestGetDashboardStats:
    @pytest.mark.asyncio
    async def test_empty_season(self):
        handler = _handler()
        result = json.loads(await handler.get_dashboard_stats())
        assert result == {
            "match_count": 0,
            "total_events": 0,
            "total_xg": 0.0,
            "home_wins": 0,
            "away_wins": 0,
            "draws": 0,
        }

    @pytest.mark.asyncio
    async def test_counts_wins_by_team_side_not_team_name(self):
        # Match 1: home wins 2-1. Match 2: away wins 0-3. Match 3: draw 1-1.
        matches = [{"id": 1}, {"id": 2}, {"id": 3}]
        events_by_match = {
            1: [_goal_event("home"), _goal_event("home"), _goal_event("away")],
            2: [_goal_event("away"), _goal_event("away"), _goal_event("away")],
            3: [_goal_event("home"), _goal_event("away")],
        }
        handler = _handler(MockStorageService(matches, events_by_match))
        result = json.loads(await handler.get_dashboard_stats())
        assert result["match_count"] == 3
        assert result["home_wins"] == 1
        assert result["away_wins"] == 1
        assert result["draws"] == 1
        assert result["total_events"] == 3 + 3 + 2

    @pytest.mark.asyncio
    async def test_total_xg_is_raw_sum_not_averaged(self):
        matches = [{"id": 1}, {"id": 2}]
        events_by_match = {
            1: [{"event_type": "shot", "metadata": {"xg": 0.3}}],
            2: [{"event_type": "shot", "metadata": {"xg": 0.5}}],
        }
        handler = _handler(MockStorageService(matches, events_by_match))
        result = json.loads(await handler.get_dashboard_stats())
        assert result["total_xg"] == pytest.approx(0.8)


class TestGetXaReport:
    @pytest.mark.asyncio
    async def test_no_events(self):
        handler = _handler_for("get_xa_report")
        result = json.loads(await handler.get_xa_report(1))
        assert result["home"] == 0.0
        assert result["away"] == 0.0

    @pytest.mark.asyncio
    async def test_computes_home_and_away_xa(self):
        events_by_match = {
            1: [
                {
                    "type": "pass",
                    "team": "home",
                    "pass_type": "through_ball_assist",
                    "start_x": 80.0,
                    "start_y": 34.0,
                    "end_x": 95.0,
                    "end_y": 34.0,
                    "is_progressive": True,
                    "timestamp": 10.0,
                },
                {
                    "type": "pass",
                    "team": "away",
                    "pass_type": "standard",
                    "start_x": 40.0,
                    "start_y": 20.0,
                    "end_x": 45.0,
                    "end_y": 22.0,
                    "timestamp": 20.0,
                },
                {"type": "shot", "team": "home", "timestamp": 12.0},
            ],
        }
        handler = _handler_for("get_xa_report", MockStorageService(events_by_match=events_by_match))
        result = json.loads(await handler.get_xa_report(1))
        assert result["home"] > 0.0
        assert result["away"] >= 0.0
        assert result["home"] > result["away"]
        # "total" is round(home_xa + away_xa, 3) computed from the unrounded
        # values, so it can differ slightly from summing the two already-
        # rounded display fields -- a wider tolerance avoids asserting on
        # double-rounding behavior rather than real functionality.
        assert result["total"] == pytest.approx(result["home"] + result["away"], abs=0.01)


class TestGetPressingReport:
    @pytest.mark.asyncio
    async def test_no_events(self):
        handler = _handler_for("get_pressing_report")
        result = json.loads(await handler.get_pressing_report(1))
        assert result["home"]["traps"] == 0.0
        assert result["home"]["conversion_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_trap_converted_to_shot(self):
        events_by_match = {
            1: [
                {"type": "tackle", "team": "home", "timestamp": 0.0, "x": 70.0, "y": 34.0},
                {"type": "pass", "team": "home", "timestamp": 1.0},
                {"type": "shot", "team": "home", "timestamp": 2.0, "is_goal": False},
            ],
        }
        handler = _handler_for(
            "get_pressing_report", MockStorageService(events_by_match=events_by_match)
        )
        result = json.loads(await handler.get_pressing_report(1))
        assert result["home"]["traps"] == 1.0
        assert result["home"]["shots_from_traps"] == 1.0
        assert result["home"]["conversion_rate"] == 1.0
        assert "high_press_index" in result["home"]


class TestGetSeasonForm:
    @pytest.mark.asyncio
    async def test_no_matches(self):
        handler = _handler()
        result = json.loads(await handler.get_season_form())
        assert result["streak_type"] == "none"
        assert result["streak_length"] == 0

    @pytest.mark.asyncio
    async def test_win_streak_uses_chronological_order(self):
        # get_all_matches() returns newest-first (id 3, 2, 1); the handler
        # must reverse this before computing streaks, or a losing run would
        # be read backwards as a winning one.
        matches = [{"id": 3}, {"id": 2}, {"id": 1}]  # newest-first, as stored
        events_by_match = {
            1: [_goal_event("home")],  # oldest: W
            2: [_goal_event("home"), _goal_event("home")],  # W
            3: [_goal_event("away")],  # newest: L
        }
        handler = _handler(MockStorageService(matches, events_by_match))
        result = json.loads(await handler.get_season_form())
        # Chronological W, W, L -> current streak is a 1-match loss.
        assert result["streak_type"] == "L"
        assert result["streak_length"] == 1
        assert result["last_5_results"] == "WWL"

    @pytest.mark.asyncio
    async def test_ppg_and_points_computed(self):
        matches = [{"id": 2}, {"id": 1}]
        events_by_match = {
            1: [_goal_event("home")],  # W = 3 pts
            2: [],  # D = 1 pt (0-0)
        }
        handler = _handler(MockStorageService(matches, events_by_match))
        result = json.loads(await handler.get_season_form())
        assert result["total_points"] == 4
        assert result["ppg_last_5"] == pytest.approx(2.0)


class TestGetPlayerStats:
    @pytest.mark.asyncio
    async def test_no_profile_service_configured(self):
        handler = _handler(player_profiles=None)
        result = json.loads(await handler.get_player_stats(1))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_player_not_found(self):
        handler = _handler(player_profiles=MockPlayerProfileService())
        result = json.loads(await handler.get_player_stats(999))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_aggregates_across_appearances(self):
        class FakeProfile:
            id = 1
            display_name = "Test Player"
            jersey_number = 9
            preferred_position = "FW"

        class FakeAppearance:
            def __init__(self, passes_completed, shots, tackles, distance, xg):
                self.passes_completed = passes_completed
                self.shots = shots
                self.tackles = tackles
                self.distance_covered_m = distance
                self.xg = xg

        pp = MockPlayerProfileService(
            profiles={1: FakeProfile()},
            appearances={
                1: [
                    FakeAppearance(10, 2, 1, 5000.0, 0.3),
                    FakeAppearance(15, 1, 0, 6000.0, 0.1),
                ]
            },
        )
        handler = _handler(player_profiles=pp)
        result = json.loads(await handler.get_player_stats(1))
        assert result["name"] == "Test Player"
        assert result["matches_played"] == 2
        assert result["passes"] == 25
        assert result["shots"] == 3
        assert result["tackles"] == 1
        assert result["distance"] == pytest.approx(11000.0)
        assert result["xg"] == pytest.approx(0.4)
        assert result["sprints"] == 0  # honestly not computed, see docstring


class TestComparePlayers:
    @pytest.mark.asyncio
    async def test_combines_two_get_player_stats_calls(self):
        class FakeProfile:
            def __init__(self, pid, name):
                self.id = pid
                self.display_name = name
                self.jersey_number = 1
                self.preferred_position = "MF"

        pp = MockPlayerProfileService(
            profiles={1: FakeProfile(1, "Player A"), 2: FakeProfile(2, "Player B")},
            appearances={1: [], 2: []},
        )
        handler = _handler(player_profiles=pp)
        result = json.loads(await handler.compare_players(1, 2))
        assert result["player_a"]["name"] == "Player A"
        assert result["player_b"]["name"] == "Player B"


class TestGetMatchPlayers:
    @pytest.mark.asyncio
    async def test_wraps_storage_result(self):
        players = [{"track_id": 1, "name": "P1", "team": "home"}]
        handler = _handler(MockStorageService(players_by_match={5: players}))
        result = json.loads(await handler.get_match_players(5))
        assert result["players"] == players


class TestSearchExternalPlayer:
    @pytest.mark.asyncio
    async def test_returns_empty_players_honestly(self):
        # No configured provider supports player-name search (see
        # docstring on the handler) -- confirms it degrades to an empty,
        # well-formed result rather than raising.
        # search_external_player moved to RecruitmentHandler in the Phase 6
        # handler split (it is part of the recruitment/scouting surface).
        handler = RecruitmentHandler(bridge=None, services={}, rate_limiter=None)
        result = json.loads(await handler.search_external_player("Messi"))
        assert result["players"] == []


class TestGpsAcwrHandlers:
    """Regression test: import_gps_file/get_gps_sessions/get_gps_samples/
    get_player_gps_summary/get_player_acwr all called SecurityValidator
    .validate_int(...), which didn't exist anywhere on SecurityValidator --
    an unconditional AttributeError on every call to the entire GPS/ACWR
    feature, caught by each method's own broad except and returned as a
    generic {"error": ...} response."""

    async def test_get_gps_sessions_no_longer_errors_on_validate_int(self):
        handler = _handler_for("get_gps_sessions", MockStorageService())

        async def _empty(mid):
            return []

        handler.storage_service.get_gps_sessions = _empty
        result = json.loads(await handler.get_gps_sessions("5"))
        assert result == {"success": True, "sessions": []}

    async def test_get_player_acwr_no_longer_errors_on_validate_int(self):
        handler = _handler_for("get_gps_sessions", MockStorageService())

        async def _empty(pid):
            return []

        handler.storage_service.get_player_acwr = _empty
        result = json.loads(await handler.get_player_acwr("3"))
        assert result == {"success": True, "acwr": []}

    async def test_get_gps_samples_no_longer_errors_on_validate_int(self):
        handler = _handler_for("get_gps_sessions", MockStorageService())

        async def _empty(sid):
            return []

        handler.storage_service.get_gps_samples = _empty
        result = json.loads(await handler.get_gps_samples("9"))
        assert result == {"success": True, "samples": []}

    async def test_gps_read_slots_await_the_real_async_storage(self):
        """v0.13.2 regression pin: PhysicalHandler called the (async) real
        StorageService without await, so every GPS/ACWR read slot returned
        {"error": "Object of type coroutine is not JSON serializable"} in
        the actual app -- invisible to the sync stubs used above, which is
        why they were rewritten to exercise the async contract too."""
        from unittest.mock import AsyncMock, MagicMock

        handler = PhysicalHandler(
            bridge=None,
            services={
                "storage_service": MagicMock(
                    get_player_acwr=AsyncMock(return_value=[{"date": "2026-01-01", "acwr": 1.1}]),
                    get_gps_sessions=AsyncMock(return_value=[{"id": 3}]),
                    get_gps_samples=AsyncMock(return_value=[{"speed": 7.2}]),
                    get_player_gps_summary=AsyncMock(return_value=[{"total_distance": 10400.0}]),
                )
            },
            rate_limiter=None,
        )
        acwr = json.loads(await handler.get_player_acwr("7"))
        assert acwr == {"success": True, "acwr": [{"date": "2026-01-01", "acwr": 1.1}]}
        sessions = json.loads(await handler.get_gps_sessions("5"))
        assert sessions == {"success": True, "sessions": [{"id": 3}]}
        samples = json.loads(await handler.get_gps_samples("9"))
        assert samples == {"success": True, "samples": [{"speed": 7.2}]}
        summary = json.loads(await handler.get_player_gps_summary("3"))
        assert summary == {"success": True, "sessions": [{"total_distance": 10400.0}]}


class MockAIAssistantV2Service:
    """Avoids the real AIAssistantV2Service, which reads/writes
    data/ai_conversations.json on construction -- not something a unit
    test should touch."""

    def __init__(self):
        self._convs = {}

    def create_conversation(self, match_id, title="New Chat"):
        conv = type("Conv", (), {"id": str(len(self._convs) + 1), "title": title})()
        self._convs[conv.id] = (match_id, title)
        return conv

    def list_conversations(self, match_id=None):
        return [
            {"id": cid, "match_id": mid, "title": title}
            for cid, (mid, title) in self._convs.items()
            if match_id is None or mid == match_id
        ]


class TestAiV2Conversations:
    """Regression test for a bug found during the audit: ai_v2_list_convs's
    body had been orphaned onto the tail of an unrelated method
    (transfermarkt_squad), losing its own `def` line entirely -- so calling
    it raised AttributeError (bridge.py's @Slot has no handler method to
    delegate to), even though app-ai.js calls it on every match load."""

    @pytest.mark.asyncio
    async def test_list_convs_is_a_real_method_not_orphaned_dead_code(self):
        handler = CloudCollabHandler(
            bridge=None,
            services={"ai_assistant_v2_service": MockAIAssistantV2Service()},
            rate_limiter=None,
        )
        result = json.loads(await handler.ai_v2_list_convs(match_id="7"))
        assert result == {"success": True, "conversations": []}

    @pytest.mark.asyncio
    async def test_create_then_list_round_trip(self):
        handler = CloudCollabHandler(
            bridge=None,
            services={"ai_assistant_v2_service": MockAIAssistantV2Service()},
            rate_limiter=None,
        )
        created = json.loads(await handler.ai_v2_create_conv(match_id="7", title="Post-match"))
        assert created["success"] is True

        listed = json.loads(await handler.ai_v2_list_convs(match_id="7"))
        assert len(listed["conversations"]) == 1
        assert listed["conversations"][0]["title"] == "Post-match"

    @pytest.mark.asyncio
    async def test_list_convs_filters_by_match_id(self):
        handler = CloudCollabHandler(
            bridge=None,
            services={"ai_assistant_v2_service": MockAIAssistantV2Service()},
            rate_limiter=None,
        )
        await handler.ai_v2_create_conv(match_id="7", title="Match 7 chat")
        await handler.ai_v2_create_conv(match_id="9", title="Match 9 chat")

        listed = json.loads(await handler.ai_v2_list_convs(match_id="7"))
        assert len(listed["conversations"]) == 1
        assert listed["conversations"][0]["title"] == "Match 7 chat"


class TestComputeGoalsAdded:
    """compute_goals_added imported a nonexistent kawkab.core.goals_added
    .compute_g_plus (the real function is compute_goals_added(player_id,
    match_stats, position)) and was itself a plain `def` calling
    self.storage_service.get_match_events(...) -- an async method --
    without await, so even fixing the import alone would have left it
    aggregating over a coroutine object instead of a list of events."""

    @pytest.mark.asyncio
    async def test_computes_real_per_player_goals_added(self):
        events = [
            {"event_type": "shot", "team": "home", "from_track_id": 7, "timestamp": 10.0},
            {"event_type": "tackle", "team": "home", "from_track_id": 7, "timestamp": 20.0},
        ]
        players = [{"track_id": 7, "position": "FWD"}]
        storage = MockStorageService(events_by_match={1: events}, players_by_match={1: players})
        handler = _handler_for("compute_goals_added", storage)
        result = json.loads(await handler.compute_goals_added(1))
        assert result["success"] is True
        assert result["count"] == 1
        report = result["result"]["7"]
        assert report["total_g_plus"] > 0
        assert report["components"]["xg_contribution"] > 0
        assert report["components"]["defensive_contribution"] > 0

    @pytest.mark.asyncio
    async def test_no_players_returns_empty_not_error(self):
        handler = _handler_for("compute_goals_added", MockStorageService())
        result = json.loads(await handler.compute_goals_added(1))
        assert result == {"success": True, "result": {}, "count": 0}


def _acwr_stub(history):
    """An async storage_service.get_player_acwr replacement -- production
    code awaits this call, so a plain lambda returning a list (as the other
    Mock*/lambda stubs in this file use for sync handler methods) won't
    work here; this returns a real coroutine."""

    async def _get(track_id, limit=1):
        return history

    return _get


class TestSquadInjuryReportBridgeHandler:
    """AnalysisHandler defined get_squad_injury_report twice -- Python keeps
    only the second, same-named method in a class body, so the first
    (whose risk_category/key_factors field names app-squad.js actually
    reads) was silently dead code, and the second -- fabricating ACWR from
    a synthetic sawtooth formula and returning differently-named
    risk_level/factors fields instead -- was the one that actually ran.
    Merged into one method using real GPS-backed ACWR data
    (storage_service.get_player_acwr); these tests pin the field names the
    frontend depends on and the honest no-data path."""

    @pytest.mark.asyncio
    async def test_player_without_acwr_data_is_honest_not_fabricated(self):
        players = [{"track_id": 7, "team": "home", "name": "No Data Player", "position": "MID"}]
        storage = MockStorageService(players_by_match={1: players})
        storage.get_player_acwr = _acwr_stub([])
        handler = _handler(storage)
        result = json.loads(await handler.get_squad_injury_report(1))
        assert result["success"] is True
        entry = result["home_players"][0]
        assert entry["risk_category"] == "insufficient_data"
        assert entry["key_factors"] == []
        assert entry["acwr"] == 0.0

    @pytest.mark.asyncio
    async def test_player_with_real_acwr_uses_the_frontends_field_names(self):
        players = [{"track_id": 7, "team": "home", "name": "Real Data Player", "position": "MID"}]
        storage = MockStorageService(players_by_match={1: players})
        storage.get_player_acwr = _acwr_stub(
            [{"acwr": 1.8, "date": "2026-08-18", "load_category": "high"}]
        )
        handler = _handler(storage)
        result = json.loads(await handler.get_squad_injury_report(1))
        entry = result["home_players"][0]
        # app-squad.js reads risk_category/key_factors, not risk_level/factors
        # (see renderSquadHealthPlayers in app-squad.js) -- the shadowed,
        # live version before this fix returned the wrong names.
        assert entry["acwr"] == 1.8
        assert "risk_category" in entry
        assert "key_factors" in entry
        assert "risk_level" not in entry
        assert "factors" not in entry
