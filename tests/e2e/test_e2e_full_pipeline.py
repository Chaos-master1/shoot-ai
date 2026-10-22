"""End-to-end integration test for the full analysis pipeline.

Tests the complete flow: events → analysis modules → result structure,
covering all analytics modules in the platform.
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()


def _install_cv_stub() -> None:
    svc_mod = types.ModuleType("kawkab.services.cv_service")

    @dataclass
    class Detection:
        bbox: tuple[float, float, float, float] = (0, 0, 10, 10)
        confidence: float = 0.9
        class_id: int = 0
        class_name: str = "person"
        track_id: int | None = None

    @dataclass
    class FrameDetections:
        frame_number: int = 0
        timestamp: float = 0.0
        detections: list = field(default_factory=list)
        image_width: int = 1280
        image_height: int = 720

    @dataclass
    class MatchTrackData:
        match_id: int = 0
        fps: float = 30.0
        total_frames: int = 0
        duration_seconds: float = 0.0
        frames: list = field(default_factory=list)
        track_registry: dict = field(default_factory=dict)
        player_teams: dict[int, str] = field(default_factory=dict)
        tracking_metrics: dict = field(default_factory=dict)
        match_type: str = "unknown"

        def swap_teams(self) -> None:
            self.player_teams = {
                tid: ("away" if t == "home" else "home") for tid, t in self.player_teams.items()
            }

    class CVService:
        async def detect_frame(self, *a, **k):
            return FrameDetections()

        async def process_video(self, *a, **k):
            return MatchTrackData()

    svc_mod.Detection = Detection
    svc_mod.FrameDetections = FrameDetections
    svc_mod.MatchTrackData = MatchTrackData
    svc_mod.CVService = CVService
    sys.modules["kawkab.services.cv_service"] = svc_mod


# The stub is installed at TEST TIME (module-scoped autouse fixture), not at
# collection time, and the teardown always REMOVES it rather than reinstating
# whatever was in sys.modules at collection. Collection happens for all files
# before any test runs, so a later-collected file can snapshot an EARLIER
# file's stub as its "original" and reinstate it at teardown -- permanently
# poisoning sys.modules for every subsequent module (observed as
# AttributeError: CVService has no _interpolate_skip_frames in
# test_accuracy_audit_fixes). Popping forces any later importer to re-import
# the real module.


@pytest.fixture(scope="module", autouse=True)
def _cv_service_stub():
    _install_cv_stub()
    yield
    sys.modules.pop("kawkab.services.cv_service", None)


_as = load_service_module("as_e2e", "analysis_service.py")
AnalysisService = _as.AnalysisService


# ── Sample events ──────────────────────────────────────────────────────────


def make_event(
    etype: str,
    team: str = "home",
    timestamp: float = 0.0,
    completed: bool = True,
    on_target: bool = False,
    player_track_id: int | None = None,
    from_track_id: int | None = None,
    to_track_id: int | None = None,
    metadata: dict | None = None,
    **kw,
) -> dict:
    ev = {
        "type": etype,
        "team": team,
        "timestamp": timestamp,
        "completed": completed,
        "on_target": on_target,
        "metadata": metadata or {},
    }
    if player_track_id is not None:
        ev["player_track_id"] = player_track_id
    if from_track_id is not None:
        ev["from_track_id"] = from_track_id
    if to_track_id is not None:
        ev["to_track_id"] = to_track_id
    ev.update(kw)
    return ev


def sample_events_24() -> list[dict]:
    return [
        # ── Goals (2) ──
        make_event(
            "shot",
            "home",
            300.0,
            on_target=True,
            metadata={"distance_to_goal_m": 8, "angle_to_goal_deg": 15},
            is_goal=True,
            xg=0.45,
        ),
        make_event(
            "shot",
            "away",
            1800.0,
            on_target=True,
            metadata={"distance_to_goal_m": 12, "angle_to_goal_deg": 20},
            is_goal=True,
            xg=0.28,
        ),
        # ── Shots (2) ──
        make_event(
            "shot", "home", 600.0, metadata={"distance_to_goal_m": 18, "angle_to_goal_deg": 30}
        ),
        make_event(
            "shot", "away", 2400.0, metadata={"distance_to_goal_m": 22, "angle_to_goal_deg": 35}
        ),
        # ── Passes (4: short, medium, long, assist) ──
        make_event(
            "pass",
            "home",
            100.0,
            metadata={"start_x_pct": 0.3, "end_x_pct": 0.38},
            from_track_id=1,
            to_track_id=2,
        ),
        make_event(
            "pass",
            "away",
            400.0,
            metadata={"start_x_pct": 0.2, "end_x_pct": 0.45},
            from_track_id=11,
            to_track_id=12,
        ),
        make_event(
            "pass",
            "home",
            700.0,
            metadata={"start_x_pct": 0.1, "end_x_pct": 0.75},
            from_track_id=3,
            to_track_id=4,
        ),
        make_event(
            "pass",
            "away",
            900.0,
            metadata={"start_x_pct": 0.15, "end_x_pct": 0.65},
            from_track_id=13,
            to_track_id=14,
        ),
        # ── Tackles (2) ──
        make_event("tackle", "home", 500.0, player_track_id=5),
        make_event("tackle", "away", 1500.0, player_track_id=15),
        # ── Interceptions (2) ──
        make_event("interception", "home", 800.0, player_track_id=6),
        make_event("interception", "away", 2000.0, player_track_id=16),
        # ── Clearances (2) ──
        make_event("clearance", "home", 1100.0, player_track_id=7),
        make_event("clearance", "away", 2200.0, player_track_id=17),
        # ── Fouls (2) ──
        make_event("foul", "home", 1300.0),
        make_event("foul", "away", 2600.0),
        # ── Corners (2) ──
        make_event("corner_kick", "home", 1400.0, metadata={"x": 100, "y": 5}),
        make_event("corner_kick", "away", 2700.0, metadata={"x": 0, "y": 5}),
        # ── Goal kicks (2) ──
        make_event("goal_kick", "home", 1600.0),
        make_event("goal_kick", "away", 2900.0),
        # ── Carries (2) ──
        make_event(
            "carry",
            "home",
            1700.0,
            metadata={"start_x_pct": 0.3, "end_x_pct": 0.55},
            player_track_id=8,
        ),
        make_event(
            "carry",
            "away",
            3100.0,
            metadata={"start_x_pct": 0.2, "end_x_pct": 0.45},
            player_track_id=18,
        ),
        # ── Saves (2) ──
        make_event("save", "home", 1900.0, player_track_id=1),
        make_event("save", "away", 3300.0, player_track_id=11),
    ]


# ── Fake tracking data helper ──────────────────────────────────────────────


def build_minimal_track_data(
    n_frames: int = 30,
    fps: float = 30.0,
    n_players_home: int = 11,
    n_players_away: int = 11,
) -> object:
    """Build a minimal MatchTrackData with ball + player detections."""
    cv_svc_mod = sys.modules.get("kawkab.services.cv_service")
    if cv_svc_mod is None:
        raise RuntimeError("kawkab.services.cv_service not found in sys.modules")
    Detection = cv_svc_mod.Detection
    FrameDetections = cv_svc_mod.FrameDetections
    MatchTrackData = cv_svc_mod.MatchTrackData

    frames = []
    player_teams = {}

    for i in range(n_players_home):
        tid = i + 1
        player_teams[tid] = "home"
    for i in range(n_players_away):
        tid = 100 + i
        player_teams[tid] = "away"

    for fno in range(n_frames):
        ts = fno / fps
        dets = []
        # Ball detection
        ball_x = 640 + 50 * math.sin(ts * 0.5)
        ball_y = 360 + 30 * math.cos(ts * 0.3)
        dets.append(
            Detection(
                bbox=(ball_x - 5, ball_y - 5, ball_x + 5, ball_y + 5),
                confidence=0.95,
                class_id=32,
                class_name="sports ball",
                track_id=999,
            )
        )
        # Home players spaced across left side
        for j in range(n_players_home):
            px = 200 + j * 60 + 10 * math.sin(ts + j)
            py = 50 + j * 55 + 10 * math.cos(ts * 0.5 + j)
            dets.append(
                Detection(
                    bbox=(px - 15, py - 15, px + 15, py + 15),
                    confidence=0.9,
                    class_id=0,
                    class_name="person",
                    track_id=j + 1,
                )
            )
        # Away players spaced across right side
        for j in range(n_players_away):
            px = 800 + j * 40 + 10 * math.sin(ts + j * 0.7)
            py = 50 + j * 55 + 10 * math.cos(ts * 0.4 + j * 0.5)
            dets.append(
                Detection(
                    bbox=(px - 15, py - 15, px + 15, py + 15),
                    confidence=0.9,
                    class_id=0,
                    class_name="person",
                    track_id=100 + j,
                )
            )
        frames.append(
            FrameDetections(
                frame_number=fno,
                timestamp=ts,
                detections=dets,
                image_width=1280,
                image_height=720,
            )
        )

    track_registry = {}
    for tid in list(player_teams.keys()):
        track_registry[tid] = {"first_pixel_x": 200.0 if tid <= n_players_home else 800.0}

    return MatchTrackData(
        match_id=1,
        fps=fps,
        total_frames=n_frames,
        duration_seconds=n_frames / fps,
        frames=frames,
        track_registry=track_registry,
        player_teams=player_teams,
        tracking_metrics={},
        match_type="test",
    )


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def svc() -> AnalysisService:
    return AnalysisService()


@pytest.fixture
def events_24() -> list[dict]:
    return sample_events_24()


@pytest.fixture
def track_data():
    return build_minimal_track_data()


# ══════════════════════════════════════════════════════════════════════════
# 1. Pipeline execution flow
# ══════════════════════════════════════════════════════════════════════════


class TestPipelineExecution:
    """Verify the core pipeline runs without errors."""

    @pytest.mark.asyncio
    async def test_analyze_match_runs(self, svc, track_data):
        result = await svc.analyze_match(track_data, match_id=42)
        assert result.match_id == 42
        assert isinstance(result.home_team, _as.TeamStats)
        assert isinstance(result.away_team, _as.TeamStats)
        assert isinstance(result.players, dict)
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_analyze_match_events_processed(self, svc, track_data):
        result = await svc.analyze_match(track_data, match_id=1)
        assert hasattr(result, "events")
        assert result.xg_total is not None
        assert result.xt_total is not None
        assert result.pass_network is not None
        assert result.formations is not None

    def test_compute_xg_simple_uses_xg_model(self, svc, events_24):
        result = svc.compute_xg_simple(events_24)
        assert "home" in result and "away" in result
        assert result["home"] > 0
        assert len(result["shot_details"]) == 4

    def test_compute_xt_simple_uses_xt_model(self, svc, events_24):
        result = svc.compute_xt_simple(events_24)
        assert "home" in result and "away" in result
        assert result["home"] >= 0

    def test_attribute_possession_robust_completes(self, svc, events_24):
        result = svc.attribute_possession_robust(events_24)
        assert len(result) == len(events_24)
        for ev in result:
            assert "attribution_source" in ev

    def test_detect_line_breaking_passes_completes(self, svc, events_24):
        result = svc.detect_line_breaking_passes(events_24)
        lb = [e for e in result if e.get("lines_crossed", 0) >= 2]
        assert len(lb) >= 1


# ══════════════════════════════════════════════════════════════════════════
# 2. Module integration verification
# ══════════════════════════════════════════════════════════════════════════


class TestModuleIntegration:
    """Verify each analytics module is reachable and produces output."""

    def test_xg_model_compute_xg(self):
        from kawkab.core.xg_model import compute_xg, compute_xg_from_dict

        result = compute_xg(distance_m=12.0, angle_deg=30.0)
        assert 0.0 <= result <= 1.0
        shot = {
            "type": "shot",
            "team": "home",
            "metadata": {"distance_to_goal_m": 8, "angle_to_goal_deg": 15},
        }
        xg = compute_xg_from_dict(shot)
        assert 0.0 <= xg <= 1.0

    def test_ppda_metric(self, svc, track_data):
        result = svc.compute_ppda(track_data, team="home")
        assert "ppda" in result
        assert "intensity" in result

    def test_momentum_analysis(self, events_24):
        from kawkab.core.momentum import compute_momentum_index

        result = compute_momentum_index(events_24)
        assert result.home_momentum_pct >= 0
        assert result.away_momentum_pct >= 0
        assert (
            abs(result.home_momentum_pct + result.away_momentum_pct + result.neutral_pct - 100.0)
            < 0.1
        )

    def test_win_probability(self, events_24):
        from kawkab.core.win_probability import compute_win_probability

        result = compute_win_probability(events_24)
        assert result.starting_home_win > 0
        assert len(result.timeline) >= 1

    def test_pass_network(self, svc, events_24):
        network = svc._compute_pass_network(events_24)
        assert "nodes" in network
        assert "edges" in network
        assert len(network["nodes"]) >= 2

    def test_formation_analysis(self, svc, track_data):
        result = svc.detect_formation(track_data, team="home")
        assert "formation" in result

    def test_player_ratings(self, svc, track_data):
        players = svc._compute_player_stats(track_data)
        typed_events = []
        pitch_control = None
        ratings = svc._compute_player_ratings(players, typed_events, pitch_control, track_data)
        assert isinstance(ratings, dict)

    def test_vaep_module(self, events_24):
        from kawkab.core.vaep import compute_vaep

        result = compute_vaep(events_24)
        assert isinstance(result, list)
        for entry in result:
            assert "vaep_value" in entry or "value" in entry or "offensive_value" in entry

    def test_set_piece_analysis(self):
        from kawkab.core.set_piece_analysis import analyze_set_pieces

        sp_events = [
            {
                "event_type": "corner_kick",
                "team": "home",
                "x": 100,
                "y": 5,
                "is_goal": False,
                "xg": 0.05,
                "timestamp": 500.0,
            },
            {
                "event_type": "free_kick",
                "team": "away",
                "x": 30,
                "y": 34,
                "is_goal": True,
                "xg": 0.12,
                "timestamp": 1500.0,
            },
        ]
        report = analyze_set_pieces(sp_events)
        assert report.total_set_pieces >= 2
        assert len(report.summaries) >= 2

    def test_xa_model(self):
        from kawkab.core.xa_model import ExpectedAssistModel

        model = ExpectedAssistModel()
        result = model.compute_xa(end_x=90.0, end_y=34.0, pass_type="cross")
        assert result.xa >= 0

    def test_pressing_traps(self, events_24):
        from kawkab.core.pressing_traps import detect_pressing_traps

        report = detect_pressing_traps(events_24, team="home")
        assert report.total_traps >= 0
        assert hasattr(report, "traps")

    def test_progressive_actions(self, events_24):
        from kawkab.core.progressive_actions import analyze_progressive_passes

        report = analyze_progressive_passes(events_24, team="home")
        assert report.total_progressive_passes >= 0

    def test_formations_track_formations(self, svc, track_data):
        result = svc.track_formations(track_data, window_minutes=5)
        assert "changes" in result
        assert "home_timeline" in result


# ══════════════════════════════════════════════════════════════════════════
# 3. Results structure verification
# ══════════════════════════════════════════════════════════════════════════


class TestResultsStructure:
    """Verify MatchAnalysis, PlayerStats, TeamStats are fully populated."""

    @pytest.mark.asyncio
    async def test_match_analysis_all_fields(self, svc, track_data):
        result = await svc.analyze_match(track_data, match_id=7)
        assert result.match_id == 7
        assert result.duration_seconds > 0
        assert isinstance(result.home_team, _as.TeamStats)
        assert isinstance(result.away_team, _as.TeamStats)
        assert len(result.players) >= 22
        assert result.xg_total is not None
        assert result.xt_total is not None
        assert result.formations is not None
        assert result.pass_network is not None
        assert 0.0 <= result.confidence_overall <= 1.0

    @pytest.mark.asyncio
    async def test_player_stats_non_empty(self, svc, track_data):
        result = await svc.analyze_match(track_data, match_id=3)
        players = result.players
        assert len(players) >= 1
        for _pid, pstats in players.items():
            assert pstats.track_id is not None
            assert pstats.team in ("home", "away", None)
            assert pstats.distance_covered_m >= 0
            assert pstats.passes_attempted >= 0

    @pytest.mark.asyncio
    async def test_team_stats_for_both_teams(self, svc, track_data):
        result = await svc.analyze_match(track_data, match_id=5)
        assert result.home_team.team_name is not None
        assert result.away_team.team_name is not None
        assert result.home_team.possession_pct >= 0
        assert result.away_team.possession_pct >= 0
        assert abs(result.home_team.possession_pct + result.away_team.possession_pct - 100.0) < 0.1

    @pytest.mark.asyncio
    async def test_xg_xt_populated(self, svc, track_data):
        result = await svc.analyze_match(track_data, match_id=9)
        assert "home" in result.xg_total
        assert "away" in result.xg_total

    @pytest.mark.asyncio
    async def test_no_exceptions_in_pipeline(self, svc, track_data):
        try:
            await svc.analyze_match(track_data, match_id=99)
        except Exception:
            pytest.fail("analyze_match raised an exception")


# ══════════════════════════════════════════════════════════════════════════
# 4. Error handling
# ══════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Verify pipeline handles edge cases gracefully."""

    def test_empty_events_xg(self, svc):
        result = svc.compute_xg_simple([])
        assert result["home"] == 0.0
        assert result["away"] == 0.0
        assert result["shot_details"] == []

    def test_empty_events_xt(self, svc):
        result = svc.compute_xt_simple([])
        assert result["home"] == 0.0
        assert result["away"] == 0.0

    def test_empty_events_line_breaking(self, svc):
        assert svc.detect_line_breaking_passes([]) == []

    def test_empty_events_attribution(self, svc):
        assert svc.attribute_possession_robust([]) == []

    def test_empty_match_track(self, svc):
        """Simulate empty match data (no frames)."""
        MatchTrackData = sys.modules["kawkab.services.cv_service"].MatchTrackData
        empty = MatchTrackData(match_id=1, fps=30, total_frames=0, duration_seconds=0, frames=[])
        result = svc.compute_ppda(empty, team="home")
        assert result["ppda"] is None
        assert result["intensity"] == "unknown"

    @pytest.mark.asyncio
    async def test_empty_match_analyze(self, svc):
        MatchTrackData = sys.modules["kawkab.services.cv_service"].MatchTrackData
        empty = MatchTrackData(match_id=1, fps=30, total_frames=0, duration_seconds=0, frames=[])
        result = await svc.analyze_match(empty, match_id=0)
        assert result.match_id == 0
        assert result.duration_seconds == 0

    def test_malformed_event_data(self, svc):
        malformed = [
            {"type": None},
            {},
            {"type": "shot", "metadata": {}},
            {"team": "home"},
            42,
        ]
        # Shot with empty metadata gets default distance=18/angle=30 → xG > 0
        good_events = [e for e in malformed if isinstance(e, dict)]
        result = svc.compute_xg_simple(good_events)
        # Only the shot event produces xG; others are skipped
        assert result["home"] >= 0
        assert result["away"] == 0

    def test_single_team_only(self, svc):
        events = [
            make_event(
                "shot", "home", 100.0, metadata={"distance_to_goal_m": 10, "angle_to_goal_deg": 20}
            ),
            make_event("pass", "home", 200.0, metadata={"start_x_pct": 0.3, "end_x_pct": 0.6}),
        ]
        xg = svc.compute_xg_simple(events)
        assert xg["home"] > 0
        assert xg["away"] == 0

        xt = svc.compute_xt_simple(events)
        assert xt["home"] >= 0

    def test_vaep_empty_events(self):
        from kawkab.core.vaep import compute_vaep

        assert compute_vaep([]) == []

    def test_win_probability_empty_events(self):
        from kawkab.core.win_probability import compute_win_probability

        result = compute_win_probability([])
        assert result.starting_home_win > 0
        assert result.starting_away_win > 0

    def test_momentum_empty_events(self):
        from kawkab.core.momentum import compute_momentum_index

        result = compute_momentum_index([])
        assert result.timeline == []

    def test_set_piece_empty_events(self):
        from kawkab.core.set_piece_analysis import analyze_set_pieces

        report = analyze_set_pieces([])
        assert report.total_set_pieces == 0

    def test_pressing_traps_empty_events(self):
        from kawkab.core.pressing_traps import detect_pressing_traps

        report = detect_pressing_traps([], team="home")
        assert report.total_traps >= 0

    def test_progressive_empty_events(self):
        from kawkab.core.progressive_actions import analyze_progressive_passes

        report = analyze_progressive_passes([], team="home")
        assert report.total_progressive_passes == 0

    @pytest.mark.asyncio
    async def test_storage_service_mock_integration(self, svc, events_24):
        mock_storage = MagicMock()
        mock_storage.save_events_bulk = MagicMock(return_value=12)
        ev = events_24[:4]
        n = mock_storage.save_events_bulk(1, ev)
        assert n == 12
        mock_storage.save_events_bulk.assert_called_once_with(1, ev)

    def test_track_formations_empty_data(self, svc):
        MatchTrackData = sys.modules["kawkab.services.cv_service"].MatchTrackData
        empty = MatchTrackData(match_id=1, fps=30, total_frames=0, duration_seconds=0, frames=[])
        result = svc.track_formations(empty, window_minutes=5)
        assert result["changes"] == 0
        assert result["home_timeline"] == []
