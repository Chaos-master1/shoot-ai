"""Supplementary tests for 8 low-coverage modules in kawkab.core.

Adds edge-case, error-handling, and corner-case coverage beyond what
existing test files exercise. Each class targets one module.
"""

from __future__ import annotations

import platform
from pathlib import Path

import pytest

# ── Module 1: fatigue_model ────────────────────────────────────────────────
from kawkab.core.fatigue_model import (
    PlayerFatigueProfile,
    compute_fatigue,
)


class TestFatigueModelExtras:
    """7 supplementary tests → total 12."""

    def test_player_id_fallback(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "player_id": 10,
                "start_x": 0,
                "start_y": 34,
                "end_x": 40,
                "end_y": 34,
                "timestamp": 0,
            },
            {
                "type": "pass",
                "team": "home",
                "player_id": 10,
                "start_x": 40,
                "start_y": 34,
                "end_x": 70,
                "end_y": 34,
                "timestamp": 300,
            },
        ]
        report = compute_fatigue(events, 90)
        assert len(report.home_fatigue) == 1
        assert report.home_fatigue[0]["track_id"] == 10

    def test_skips_none_track(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 0,
                "start_y": 34,
                "end_x": 40,
                "end_y": 34,
                "timestamp": 0,
            },
            {
                "type": "run",
                "team": "home",
                "start_x": 40,
                "start_y": 34,
                "end_x": 70,
                "end_y": 34,
                "timestamp": 300,
            },
        ]
        report = compute_fatigue(events, 90)
        assert len(report.home_fatigue) == 0
        assert len(report.away_fatigue) == 0

    def test_zero_distance_not_high_intensity(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
                "timestamp": 0,
            },
        ]
        report = compute_fatigue(events, 90)
        assert report.home_fatigue[0]["high_intensity_actions"] == 0
        assert report.home_fatigue[0]["distance_covered_m"] == 0

    def test_fatigue_index_clamped(self):
        pf = PlayerFatigueProfile(fatigue_index=1.5, speed_decline_pct=30.0)
        d = pf.to_dict()
        # to_dict preserves the raw value; the model clamps in compute_fatigue
        assert d["fatigue_index"] == 1.5  # stored as-is; compute clamps
        # Actually test compute clamping via extreme values
        events = [
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 0,
                "start_y": 34,
                "end_x": 100,
                "end_y": 34,
                "timestamp": 0,
            },
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 100,
                "start_y": 34,
                "end_x": 0,
                "end_y": 34,
                "timestamp": 5400,
            },
            # Many high-intensity actions
            {
                "type": "shot",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
                "timestamp": 60,
                "is_goal": True,
            },
            {
                "type": "tackle",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
                "timestamp": 120,
            },
        ]
        report = compute_fatigue(events, 90)
        fi = report.home_fatigue[0]["fatigue_index"]
        assert 0.0 <= fi <= 1.0

    def test_carry_type_adds_distance(self):
        events = [
            {
                "type": "carry",
                "team": "away",
                "track_id": 5,
                "start_x": 30,
                "start_y": 34,
                "end_x": 80,
                "end_y": 34,
                "timestamp": 10,
            },
        ]
        report = compute_fatigue(events, 90)
        assert len(report.away_fatigue) == 1
        assert report.away_fatigue[0]["distance_covered_m"] > 0

    def test_substitution_with_track_id(self):
        events = [
            {
                "type": "substitution",
                "team": "away",
                "track_id": 22,
                "player_out": 7,
                "timestamp": 1800,
            },
        ]
        report = compute_fatigue(events, 90)
        assert len(report.substitutions) == 1
        assert report.substitutions[0]["track_id_in"] == 22

    def test_multiple_substitutions(self):
        events = [
            {
                "type": "substitution",
                "team": "home",
                "player_in": 11,
                "player_out": 7,
                "timestamp": 2700,
            },
            {
                "type": "substitution",
                "team": "home",
                "player_in": 15,
                "player_out": 9,
                "timestamp": 3600,
            },
            {
                "type": "substitution",
                "team": "away",
                "player_in": 20,
                "player_out": 3,
                "timestamp": 1800,
            },
        ]
        report = compute_fatigue(events, 90)
        assert len(report.substitutions) == 3


# ── Module 2: game_state ──────────────────────────────────────────────────

from kawkab.core.game_state import GameStateMetrics, analyze_game_state


class TestGameStateExtras:
    """6 supplementary tests → total 10."""

    def test_possession_and_nonpossession_frames(self):
        events = [{"type": "pass", "timestamp": 5.0, "team": "home", "completed": True}]
        frames = [
            {
                "timestamp": t,
                "possession": t < 5,
                "home_positions": [(50, 34)],
                "away_positions": [(70, 34)],
            }
            for t in range(10)
        ]
        result = analyze_game_state(events, frames)
        assert 0 < result.drawing.possession_pct < 100

    def test_custom_home_team_name(self):
        events = [
            {"type": "shot", "timestamp": 10.0, "team": "FC Barcelona", "is_goal": True},
            {"type": "pass", "timestamp": 15.0, "team": "FC Barcelona", "completed": True},
        ]
        frames = [
            {
                "timestamp": t,
                "possession": t < 15,
                "home_positions": [(50, 34)],
                "away_positions": [(70, 34)],
            }
            for t in range(0, 20)
        ]
        result = analyze_game_state(events, frames, home_team_name="FC Barcelona")
        assert result.home_winning.duration_s > 0
        assert result.drawing.duration_s > 0

    def test_missing_completed_default_true(self):
        events = [{"type": "pass", "timestamp": 5.0, "team": "home"}]
        frames = [
            {
                "timestamp": t,
                "possession": True,
                "home_positions": [(50, 34)],
                "away_positions": [(70, 34)],
            }
            for t in range(10)
        ]
        result = analyze_game_state(events, frames)
        assert result.drawing.pass_completion_pct == 100.0

    def test_empty_home_positions(self):
        events = [{"type": "pass", "timestamp": 5.0, "team": "home", "completed": True}]
        frames = [
            {"timestamp": 5.0, "possession": True, "home_positions": [], "away_positions": []}
        ]
        result = analyze_game_state(events, frames)
        assert result.drawing.defensive_line_height_m == 0.0

    def test_state_to_dict_rounding(self):
        result = analyze_game_state([], [])
        d = result.to_dict()
        assert isinstance(d["home_winning"]["possession_pct"], float)
        assert isinstance(d["drawing"]["shots_per_10min"], float)

    def test_game_state_metrics_defaults(self):
        m = GameStateMetrics()
        assert m.possession_pct == 0.0
        assert m.duration_s == 0.0
        d = m.to_dict()
        assert d["possession_pct"] == 0.0


# ── Module 3: logging ─────────────────────────────────────────────────────

from kawkab.core.logging import get_logger, setup_logging


class TestLoggingExtras:
    """4 supplementary tests → total 8."""

    def test_setup_logging_debug(self):
        setup_logging(debug=True)
        logger = get_logger("test_debug")
        logger.debug("debug message should not crash")

    def test_get_logger_no_name(self):
        logger = get_logger()
        assert logger is not None
        logger.info("no-name logger works")

    def test_setup_logging_twice(self):
        setup_logging()
        setup_logging()
        logger = get_logger("double_setup")
        logger.info("called setup_logging twice, no crash")

    def test_logger_info_level_by_default(self):
        setup_logging(debug=False)
        logger = get_logger()
        assert logger is not None


# ── Module 4: match_timeline ──────────────────────────────────────────────

from kawkab.core.match_timeline import compute_xg_timeline


class TestMatchTimelineExtras:
    """5 supplementary tests → total 10."""

    def test_mixed_home_away_shots(self):
        events = [
            {"type": "shot", "team": "home", "xg": 0.3, "timestamp": 600, "is_goal": False},
            {"type": "shot", "team": "away", "xg": 0.7, "timestamp": 1200, "is_goal": True},
        ]
        report = compute_xg_timeline(events, 90)
        assert report.home_total == 0.3
        assert report.away_total == 0.7
        assert report.home_goals == 0
        assert report.away_goals == 1

    def test_shot_without_xg_key(self):
        events = [{"type": "shot", "team": "home", "timestamp": 900, "is_goal": False}]
        report = compute_xg_timeline(events, 90)
        assert report.home_total == 0.0

    def test_multiple_goals(self):
        events = [
            {"type": "shot", "team": "home", "xg": 0.2, "timestamp": 600, "is_goal": True},
            {"type": "shot", "team": "away", "xg": 0.5, "timestamp": 1800, "is_goal": True},
            {"type": "shot", "team": "home", "xg": 0.1, "timestamp": 3000, "is_goal": True},
        ]
        report = compute_xg_timeline(events, 90)
        assert report.home_goals == 2
        assert report.away_goals == 1
        assert report.home_total == pytest.approx(0.3, abs=1e-6)
        assert report.away_total == pytest.approx(0.5, abs=1e-6)

    def test_match_duration_affects_end_point(self):
        events = []
        report = compute_xg_timeline(events, 120.0)
        assert report.points[-1]["minute"] == 120.0

    def test_cumulative_values_after_goal(self):
        events = [
            {"type": "shot", "team": "home", "xg": 0.4, "timestamp": 600, "is_goal": True},
            {"type": "shot", "team": "home", "xg": 0.3, "timestamp": 1800, "is_goal": False},
        ]
        report = compute_xg_timeline(events, 90)
        goal_point = report.points[1]
        shot_point = report.points[2]
        assert goal_point["home_cumulative"] == 0.4
        assert shot_point["home_cumulative"] == 0.7
        assert goal_point["event_type"] == "goal"
        assert shot_point["event_type"] == "shot"


# ── Module 5: pass_flow ────────────────────────────────────────────────────

from kawkab.core.pass_flow import compute_pass_flow


class TestPassFlowExtras:
    """5 supplementary tests → total 10."""

    def test_away_team_filter(self):
        events = [
            {
                "type": "pass",
                "team": "away",
                "start_x": 10,
                "start_y": 34,
                "end_x": 50,
                "end_y": 40,
                "completed": True,
                "timestamp": 10,
            },
        ]
        result = compute_pass_flow(events, "away")
        assert len(result) == 1

    def test_grid_cells_one(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 5,
                "start_y": 5,
                "end_x": 100,
                "end_y": 60,
                "completed": True,
                "timestamp": 10,
            },
            {
                "type": "pass",
                "team": "home",
                "start_x": 20,
                "start_y": 30,
                "end_x": 80,
                "end_y": 40,
                "completed": False,
                "timestamp": 20,
            },
        ]
        result = compute_pass_flow(events, "home", grid_cells=1)
        assert len(result) == 1
        assert result[0]["count"] == 2

    def test_default_coords_when_missing(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 10,
                "end_x": 50,
                "completed": True,
                "timestamp": 5,
            },
        ]
        result = compute_pass_flow(events, "home")
        assert len(result) >= 1
        assert result[0]["count"] == 1

    def test_sort_by_count_descending(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 10,
                "start_y": 10,
                "end_x": 50,
                "end_y": 10,
                "completed": True,
                "timestamp": 1,
            },
            {
                "type": "pass",
                "team": "home",
                "start_x": 10,
                "start_y": 10,
                "end_x": 50,
                "end_y": 10,
                "completed": True,
                "timestamp": 2,
            },
            {
                "type": "pass",
                "team": "home",
                "start_x": 60,
                "start_y": 50,
                "end_x": 90,
                "end_y": 50,
                "completed": False,
                "timestamp": 3,
            },
        ]
        result = compute_pass_flow(events, "home")
        assert result[0]["count"] >= result[-1]["count"]

    def test_avg_progress_computed(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 10,
                "start_y": 34,
                "end_x": 50,
                "end_y": 34,
                "completed": True,
                "timestamp": 5,
            },
        ]
        result = compute_pass_flow(events, "home")
        assert result[0]["avg_progress"] == 40.0


# ── Module 6: pass_sonars ──────────────────────────────────────────────────

from kawkab.core.pass_sonars import compute_pass_sonars


class TestPassSonarsExtras:
    """5 supplementary tests → total 10."""

    def test_none_track_id_uses_question_mark(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "start_x": 50,
                "start_y": 34,
                "end_x": 70,
                "end_y": 40,
                "completed": True,
            },
        ]
        result = compute_pass_sonars(events)
        assert len(result) == 1
        assert result[0]["track_id"] == "?"

    def test_fewer_sectors(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 70,
                "end_y": 34,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 30,
                "end_y": 34,
                "completed": True,
            },
        ]
        result = compute_pass_sonars(events, sectors=4)
        assert len(result) == 1
        assert len(result[0]["sectors"]) == 4

    def test_team_preserved_in_output(self):
        events = [
            {
                "type": "pass",
                "team": "away",
                "track_id": 5,
                "start_x": 50,
                "start_y": 34,
                "end_x": 70,
                "end_y": 40,
                "completed": True,
            },
        ]
        result = compute_pass_sonars(events)
        assert result[0]["team"] == "away"

    def test_default_end_coords(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "completed": True,
            },
        ]
        result = compute_pass_sonars(events)
        assert len(result) == 1

    def test_sector_avg_distance_accuracy(self):
        events = [
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 70,
                "end_y": 34,
                "completed": True,
            },
            {
                "type": "pass",
                "team": "home",
                "track_id": 1,
                "start_x": 50,
                "start_y": 34,
                "end_x": 70,
                "end_y": 34,
                "completed": False,
            },
        ]
        result = compute_pass_sonars(events, sectors=12)
        # both passes go right (angle ~0°), sector 0
        sector0 = result[0]["sectors"][0]
        assert sector0["count"] == 2
        assert sector0["completed"] == 1
        assert sector0["accuracy"] == 0.5


# ── Module 7: paths ────────────────────────────────────────────────────────

import importlib.util as _iu

import kawkab.core.paths as _paths_mod

# conftest may have stubbed kawkab.core.paths — load real module attrs
_src_pkg = Path(__file__).resolve().parent.parent.parent / "src"
_src_paths = _src_pkg / "kawkab" / "core" / "paths.py"
_spec = _iu.spec_from_file_location("kawkab.core.paths.real", str(_src_paths))
_real_paths = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_real_paths)
# Copy real functions into the module reference used here
for _attr in (
    "_get_appdata_dir",
    "_get_localappdata_dir",
    "_get_documents_dir",
    "Paths",
    "get_paths",
):
    setattr(_paths_mod, _attr, getattr(_real_paths, _attr))


class TestPathsExtras:
    """6 supplementary tests → total 10."""

    def test_config_file_property(self, tmp_path, monkeypatch):
        monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "userprofile"))
        paths = _paths_mod.Paths()
        assert paths.config_file == paths.appdata / "config.json"
        assert paths.config_file.suffix == ".json"

    def test_migrations_property_finds_package_dir(self):
        paths = _paths_mod.get_paths()
        mig_path = paths.migrations
        assert isinstance(mig_path, Path)

    def test_get_appdata_dir_fallback(self, monkeypatch):
        monkeypatch.delenv("APPDATA", raising=False)
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        path = _paths_mod._get_appdata_dir()
        assert "Roaming" in str(path)
        assert "KawkabAI" in str(path)

    def test_get_localappdata_dir_fallback(self, monkeypatch):
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        path = _paths_mod._get_localappdata_dir()
        assert "Local" in str(path)
        assert "KawkabAI" in str(path)

    def test_get_documents_dir_fallback(self, monkeypatch):
        monkeypatch.delenv("USERPROFILE", raising=False)
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        path = _paths_mod._get_documents_dir()
        assert "Documents" in str(path)
        assert "KawkabAI" in str(path)

    def test_macos_paths(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        monkeypatch.setenv("HOME", "/Users/testuser")
        appdata = _paths_mod._get_appdata_dir()
        assert "Application Support" in str(appdata)
        localapp = _paths_mod._get_localappdata_dir()
        assert "Caches" in str(localapp)


# ── Module 8: trap_transition_linkage ─────────────────────────────────────

from kawkab.core.pressing_traps import PressingTrap
from kawkab.core.transitions import PhaseTransition
from kawkab.core.trap_transition_linkage import (
    TrapTransitionAnalysis,
    TrapTransitionLink,
    _find_trap_recovery_events,
    _infer_team_for_trap,
    _zone_midpoint,
    analyze_trap_transitions,
    summarize_trap_transition,
)


def _ev(etype, team, ts, x, y, **kw):
    ev = {"type": etype, "team": team, "timestamp": ts, "x": x, "y": y}
    ev.update(kw)
    return ev


def _trap(zone_name, x_range, y_range, regains=0):
    return PressingTrap(
        zone_name=zone_name,
        zone_x_range=x_range,
        zone_y_range=y_range,
        trigger_event_type="backward_pass",
        defensive_actions_in_zone=5,
        opponent_passes_into_zone=3,
        regain_possession_count=regains,
        success_rate=regains / 5.0,
        intensity=2.0,
        trap_rating=0.5,
    )


def _trans(team, ts, sx, sy):
    return PhaseTransition(
        timestamp=ts,
        team=team,
        transition_type="counter_attack",
        start_x=sx,
        start_y=sy,
        duration_s=3.0,
        speed_mps=8.0,
        outcome="shot",
        ended_in_final_third=True,
    )


# central_mid zone ranges
CEN_XR = (34.65, 70.35)
CEN_YR = (17.0, 51.0)
CEN_XM = 52.5
CEN_YM = 34.0


class TestTrapTransitionLinkageExtras:
    """8 supplementary tests."""

    def test_missing_keys_in_events(self):
        events = [{"type": "tackle", "timestamp": 1.0}]
        traps = [_trap("central_mid", CEN_XR, CEN_YR, regains=1)]
        transitions = [_trans("home", 5.0, CEN_XM + 5, CEN_YM)]
        analysis = analyze_trap_transitions(traps, transitions, events)
        assert analysis.total_traps == 1

    def test_events_without_coordinates(self):
        events = [
            {"type": "tackle", "team": "home", "timestamp": 3.0},
            {"type": "pass", "team": "home", "timestamp": 4.0},
        ]
        traps = [_trap("central_mid", CEN_XR, CEN_YR, regains=1)]
        analysis = analyze_trap_transitions(traps, [], events)
        # no x/y so events are skipped; trap has regains>0 but no link
        assert analysis.total_traps == 1

    def test_multiple_traps_some_successful(self):
        events = [
            _ev("pass", "away", 1.0, 50, 34),
            _ev("tackle", "home", 3.0, 52, 34),
            _ev("pass", "home", 4.0, 55, 34),
        ]
        traps = [
            _trap("central_mid", CEN_XR, CEN_YR, regains=1),  # successful
            _trap("left_defensive", (0, 34.65), (0, 34), regains=0),  # not successful
        ]
        transitions = [_trans("home", 5.0, CEN_XM + 5, CEN_YM)]
        analysis = analyze_trap_transitions(traps, transitions, events)
        assert analysis.total_traps == 2
        assert analysis.successful_traps == 1

    def test_summarize_full_conversion(self):
        analysis = TrapTransitionAnalysis(
            total_traps=3,
            successful_traps=3,
            transitions_from_traps=[
                TrapTransitionLink(0, 0, 1.0, 5.0, True, True),
            ],
            conversion_rate=1.0,
            goal_conversion_rate=0.333,
            avg_transition_time=1.0,
        )
        summary = summarize_trap_transition(analysis)
        assert "100%" in summary["chance_conversion"]
        assert "1.0 seconds" in summary["avg_transition_time"]

    def test_infer_team_no_matching_events(self):
        events = [
            {"type": "pass", "team": "home", "x": 50, "y": 34},
        ]
        trap = _trap("central_mid", CEN_XR, CEN_YR)
        team = _infer_team_for_trap(events, trap)
        assert team == "home"  # default fallback

    def test_find_recovery_no_recovery(self):
        events = [
            _ev("tackle", "home", 3.0, 52, 34),
        ]
        trap = _trap("central_mid", CEN_XR, CEN_YR)
        times = _find_trap_recovery_events(events, trap, "home")
        assert times == []

    def test_zone_midpoint(self):
        trap = _trap("central_mid", CEN_XR, CEN_YR)
        mx, my = _zone_midpoint(trap)
        assert mx == pytest.approx(CEN_XM, abs=0.01)
        assert my == pytest.approx(CEN_YM, abs=0.01)

    def test_trap_transition_link_dataclass(self):
        link = TrapTransitionLink(
            trap_index=0,
            transition_index=1,
            time_delta=2.5,
            spatial_distance=12.0,
            goal_scored=False,
            shot_created=True,
        )
        assert link.trap_index == 0
        assert link.transition_index == 1
        assert link.time_delta == 2.5
        assert not link.goal_scored
        assert link.shot_created
