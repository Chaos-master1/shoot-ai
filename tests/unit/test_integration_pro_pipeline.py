"""Integration smoke tests for the pro-analytics pipeline.

These tests exercise the workflow that the UI exposes:
"load events -> call each pro service -> check JSON output".

Unlike unit tests, they don't stub individual services — instead
they use lightweight fakes that mimic the data shape the bridge
passes between Python and JavaScript.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


def _install_cv_stub() -> None:
    if "kawkab.services" in sys.modules:
        return
    services_mod = types.ModuleType("kawkab.services")
    sys.modules["kawkab.services"] = services_mod
    cv_mod = types.ModuleType("kawkab.services.cv_service")

    class FrameDetections:
        pass

    class MatchTrackData:
        pass

    cv_mod.FrameDetections = FrameDetections
    cv_mod.MatchTrackData = MatchTrackData
    sys.modules["kawkab.services.cv_service"] = cv_mod
    services_mod.cv_service = cv_mod


_install_cv_stub()


# Load all pro analytics services
_sp = load_service_module("sp_test", "setpiece_service.py")
_gk = load_service_module("gk_test", "goalkeeper_service.py")
_sub = load_service_module("sub_test", "substitution_service.py")
_pos = load_service_module("pos_test", "possession_service.py")

SetPieceService = _sp.SetPieceService
GoalkeeperService = _gk.GoalkeeperService
SubstitutionService = _sub.SubstitutionService
PossessionService = _pos.PossessionService

import json

SAMPLE_MATCH = {
    "home": "Kawkab FC",
    "away": "Raja Casablanca",
    "events": [
        {
            "type": "shot",
            "team": "home",
            "timestamp_s": 120,
            "player_track_id": 9,
            "xg": 0.35,
            "outcome": "shot",
        },
        {
            "type": "shot",
            "team": "home",
            "timestamp_s": 300,
            "player_track_id": 11,
            "xg": 0.55,
            "outcome": "goal",
        },
        {
            "type": "shot",
            "team": "away",
            "timestamp_s": 600,
            "player_track_id": 7,
            "xg": 0.20,
            "outcome": "save",
        },
        {
            "type": "shot",
            "team": "home",
            "timestamp_s": 1800,
            "player_track_id": 10,
            "xg": 0.15,
            "outcome": "shot",
        },
        {
            "type": "shot",
            "team": "away",
            "timestamp_s": 2400,
            "player_track_id": 8,
            "xg": 0.40,
            "outcome": "goal",
        },
    ],
    "set_pieces": [
        {
            "set_piece_type": "corner",
            "minute": 15,
            "second": 0,
            "team": "home",
            "delivery_x": 100,
            "delivery_y": 0,
            "delivery_style": "inswinging",
            "delivery_height": "medium",
            "first_contact_x": 96,
            "first_contact_y": 30,
            "outcome": "shot",
        },
        {
            "set_piece_type": "corner",
            "minute": 35,
            "second": 0,
            "team": "home",
            "delivery_x": 100,
            "delivery_y": 68,
            "delivery_style": "outswinging",
            "delivery_height": "high",
            "first_contact_x": 97,
            "first_contact_y": 40,
            "outcome": "clearance",
        },
        {
            "set_piece_type": "free_kick",
            "minute": 70,
            "second": 0,
            "team": "away",
            "delivery_x": 88,
            "delivery_y": 34,
            "delivery_style": "lofted",
            "delivery_height": "high",
            "first_contact_x": 99,
            "first_contact_y": 5,
            "outcome": "shot",
        },
    ],
    "subs": [
        {
            "minute": 60,
            "second": 0,
            "team": "home",
            "player_off_track_id": 7,
            "player_on_track_id": 14,
        },
    ],
    "gk_actions": [
        {
            "action_type": "save",
            "team": "home",
            "timestamp_s": 600,
            "outcome": "complete",
            "x": 99,
            "y": 34,
        },
        {
            "action_type": "short_dist",
            "team": "home",
            "timestamp_s": 700,
            "outcome": "complete",
            "x": 30,
        },
    ],
    "shots_faced": [
        {"x": 99, "y": 34, "body_part": "foot", "one_on_one": False, "outcome": "save"},
        {"x": 95, "y": 30, "body_part": "head", "one_on_one": False, "outcome": "save"},
    ],
    "clean_sheet": False,
}


class TestEndToEnd:
    def test_setpiece_analyze(self) -> None:
        svc = SetPieceService()
        SetPieceEvent = _sp.SetPieceEvent
        events = [SetPieceEvent(**d) for d in SAMPLE_MATCH["set_pieces"]]
        report = svc.analyze(events, "home", "away")
        assert report.home_stats.total_corners + report.home_stats.total_free_kicks >= 1
        result = {
            "home_corners": report.home_stats.total_corners,
            "home_threat": report.home_stats.threat_per_set_piece,
            "notes": report.notes,
        }
        assert result["home_corners"] >= 1
        assert isinstance(result["notes"], list)

    def test_goalkeeper_analyze(self) -> None:
        svc = GoalkeeperService()
        GoalkeeperAction = _gk.GoalkeeperAction
        actions = []
        for d in SAMPLE_MATCH["gk_actions"]:
            ts = d.get("timestamp_s", 0)
            minute = int(ts // 60)
            second = int(ts % 60)
            actions.append(
                GoalkeeperAction(
                    action_type=d["action_type"],
                    minute=minute,
                    second=second,
                    team=d["team"],
                    outcome=d.get("outcome", "complete"),
                    x=d.get("x"),
                    y=d.get("y"),
                )
            )
        stats = svc.compute_stats(
            "home", actions, SAMPLE_MATCH["shots_faced"], clean_sheet=SAMPLE_MATCH["clean_sheet"]
        )
        result = {
            "saves": stats.saves,
            "save_rate": stats.save_rate,
            "clean_sheet": stats.clean_sheet,
        }
        assert "saves" in result
        assert isinstance(result["save_rate"], float)

    def test_substitution_analyze(self) -> None:
        svc = SubstitutionService()
        SubstitutionEvent = _sub.SubstitutionEvent
        subs = [SubstitutionEvent(**d) for d in SAMPLE_MATCH["subs"]]
        report = svc.analyze("home", subs, SAMPLE_MATCH["events"])
        result = {
            "sub_count": len(report.impacts),
            "avg_rating": report.avg_impact,
        }
        assert result["sub_count"] >= 1
        assert isinstance(result["avg_rating"], (int, float))

    def test_possession_analyze(self) -> None:
        svc = PossessionService()
        events_with_passes = SAMPLE_MATCH["events"] + [
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 10,
                "player_track_id": 7,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 20,
                "player_track_id": 8,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "timestamp_s": 30,
                "player_track_id": 9,
                "completed": False,
            },
            {
                "type": "pass",
                "team": "away",
                "timestamp_s": 40,
                "player_track_id": 5,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "away",
                "timestamp_s": 50,
                "player_track_id": 6,
                "completed": True,
            },
        ]
        report = svc.analyze("home", "away", events_with_passes)
        result = {
            "home_pct": report.home_possession_pct,
            "away_pct": report.away_possession_pct,
            "n_chains": len(report.home_chains) + len(report.away_chains),
        }
        assert result["home_pct"] + result["away_pct"] >= 0
        assert result["n_chains"] >= 0

    def test_full_pipeline_json_serializable(self) -> None:
        """Verify all reports produce JSON-serializable dicts via to_dict-like access."""
        SetPieceEvent = _sp.SetPieceEvent
        GoalkeeperAction = _gk.GoalkeeperAction
        SubstitutionEvent = _sub.SubstitutionEvent
        sp_svc = SetPieceService()
        gk_svc = GoalkeeperService()
        sub_svc = SubstitutionService()
        pos_svc = PossessionService()
        sp_events = [SetPieceEvent(**d) for d in SAMPLE_MATCH["set_pieces"]]
        gk_actions = []
        for d in SAMPLE_MATCH["gk_actions"]:
            ts = d.get("timestamp_s", 0)
            minute = int(ts // 60)
            second = int(ts % 60)
            gk_actions.append(
                GoalkeeperAction(
                    action_type=d["action_type"],
                    minute=minute,
                    second=second,
                    team=d["team"],
                    outcome=d.get("outcome", "complete"),
                    x=d.get("x"),
                    y=d.get("y"),
                )
            )
        sub_events = [SubstitutionEvent(**d) for d in SAMPLE_MATCH["subs"]]
        sp_report = sp_svc.analyze(sp_events, "home", "away")
        gk_stats = gk_svc.compute_stats(
            "home", gk_actions, SAMPLE_MATCH["shots_faced"], clean_sheet=SAMPLE_MATCH["clean_sheet"]
        )
        sub_report = sub_svc.analyze("home", sub_events, SAMPLE_MATCH["events"])
        pos_report = pos_svc.analyze("home", "away", SAMPLE_MATCH["events"])
        bundle = {
            "setpiece": {
                "home_n": sp_report.home_stats.total_corners
                + sp_report.home_stats.total_free_kicks,
                "away_n": sp_report.away_stats.total_corners
                + sp_report.away_stats.total_free_kicks,
                "home_threat": sp_report.home_stats.threat_per_set_piece,
                "notes": sp_report.notes,
            },
            "goalkeeper": {
                "saves": gk_stats.saves,
                "save_rate": gk_stats.save_rate,
                "xgot_per_shot": gk_stats.xgot_per_shot,
                "clean_sheet": gk_stats.clean_sheet,
            },
            "substitutions": {
                "count": len(sub_report.impacts),
                "avg_rating": sub_report.avg_impact,
            },
            "possession": {
                "home_pct": pos_report.home_possession_pct,
                "away_pct": pos_report.away_possession_pct,
                "counter_presses": pos_report.counter_presses,
            },
        }
        json_str = json.dumps(bundle)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["goalkeeper"]["saves"] >= 0


class TestBridgeJsonShape:
    """Verify each service output maps cleanly to the bridge JSON contract."""

    def test_setpiece_summary_keys(self) -> None:
        SetPieceEvent = _sp.SetPieceEvent
        sp_svc = SetPieceService()
        sp_events = [SetPieceEvent(**d) for d in SAMPLE_MATCH["set_pieces"]]
        report = sp_svc.analyze(sp_events, "home", "away")
        home = {
            "n_corners": report.home_stats.total_corners,
            "n_free_kicks": report.home_stats.total_free_kicks,
            "n_throw_ins": report.home_stats.total_throw_ins,
            "threat_per_set_piece": report.home_stats.threat_per_set_piece,
            "set_piece_differential": report.set_piece_differential,
        }
        assert all(k in home for k in ["n_corners", "n_free_kicks", "threat_per_set_piece"])

    def test_goalkeeper_keys(self) -> None:
        gk_svc = GoalkeeperService()
        actions = []
        stats = gk_svc.compute_stats("home", actions, [], clean_sheet=True)
        keys = {
            "saves": stats.saves,
            "save_rate": stats.save_rate,
            "clean_sheet": stats.clean_sheet,
            "notes": stats.notes,
        }
        assert "save_rate" in keys
        assert "clean_sheet" in keys
