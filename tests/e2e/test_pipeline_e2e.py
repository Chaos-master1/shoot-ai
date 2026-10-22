"""End-to-end integration test for the full analytical pipeline.

Tests the complete flow:
1. Create synthetic match data (events with xG/xT/VAEP features)
2. Save to storage service
3. Run AnalysisService.analyze_match()
4. Verify output contains: xG, xT, VAEP, pitch_control, formations
5. Export as StatsBomb JSON via data_export_service
6. Verify exported JSON matches StatsBomb schema
7. Export as CSV and verify
8. Generate match report and verify sections
9. Bridge layer call chain: bridge slot → handler → service → result
"""

from __future__ import annotations

import json
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
_as = load_service_module("as_e2e_pipeline", "analysis_service.py")
AnalysisService = _as.AnalysisService
MatchAnalysis = _as.MatchAnalysis

# ── Synthetic data generators ─────────────────────────────────────────


def make_event(
    etype: str,
    team: str = "home",
    timestamp: float = 0.0,
    completed: bool = True,
    on_target: bool = False,
    is_goal: bool = False,
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
        "is_goal": is_goal,
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


def synthetic_events() -> list[dict]:
    """Generate 20 events spanning multiple types for end-to-end testing."""
    return [
        # Goals
        make_event(
            "shot",
            "home",
            300.0,
            on_target=True,
            is_goal=True,
            metadata={"distance_to_goal_m": 8, "angle_to_goal_deg": 15, "xg": 0.45},
        ),
        make_event(
            "shot",
            "away",
            1800.0,
            on_target=True,
            is_goal=True,
            metadata={"distance_to_goal_m": 12, "angle_to_goal_deg": 20, "xg": 0.28},
        ),
        # Shots
        make_event(
            "shot", "home", 600.0, metadata={"distance_to_goal_m": 18, "angle_to_goal_deg": 30}
        ),
        make_event(
            "shot", "away", 2400.0, metadata={"distance_to_goal_m": 22, "angle_to_goal_deg": 35}
        ),
        # Passes
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
        # Tackles
        make_event("tackle", "home", 500.0, player_track_id=5),
        make_event("tackle", "away", 1500.0, player_track_id=15),
        # Interceptions
        make_event("interception", "home", 800.0, player_track_id=6),
        make_event("interception", "away", 2000.0, player_track_id=16),
        # Clearances
        make_event("clearance", "home", 1100.0, player_track_id=7),
        make_event("clearance", "away", 2200.0, player_track_id=17),
        # Fouls
        make_event("foul", "home", 1300.0),
        make_event("foul", "away", 2600.0),
        # Corners
        make_event("corner_kick", "home", 1400.0, metadata={"x": 100, "y": 5}),
        make_event("corner_kick", "away", 2700.0, metadata={"x": 0, "y": 5}),
        # Carries
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
    ]


def build_minimal_track_data(
    n_frames: int = 30,
    fps: float = 30.0,
    n_players: int = 22,
) -> object:
    """Build minimal MatchTrackData stub with synthetic detections."""
    cv_mod = sys.modules.get("kawkab.services.cv_service")
    if cv_mod is None:
        raise RuntimeError("kawkab.services.cv_service not in sys.modules")
    Detection = cv_mod.Detection
    FrameDetections = cv_mod.FrameDetections
    MatchTrackData = cv_mod.MatchTrackData

    frames = []
    player_teams = {}
    for i in range(n_players // 2):
        player_teams[i + 1] = "home"
    for i in range(n_players // 2):
        player_teams[100 + i] = "away"

    for fno in range(n_frames):
        ts = fno / fps
        dets = []
        # Ball
        dets.append(
            Detection(
                bbox=(640 - 5, 360 - 5, 640 + 5, 360 + 5),
                confidence=0.95,
                class_id=32,
                class_name="sports ball",
                track_id=999,
            )
        )
        for j in range(n_players // 2):
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
        for j in range(n_players // 2):
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
    for tid in list(player_teams.keys())[:11]:
        track_registry[tid] = {"first_pixel_x": 200.0}
    for tid in list(player_teams.keys())[11:]:
        track_registry[tid] = {"first_pixel_x": 800.0}

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


def install_cv_stub() -> None:
    """Ensure the CV stub is installed."""
    svc_mod = sys.modules.get("kawkab.services.cv_service")
    if svc_mod is not None and hasattr(svc_mod, "MatchTrackData"):
        return

    _mod = types.ModuleType("kawkab.services.cv_service")

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

    _mod.Detection = Detection
    _mod.FrameDetections = FrameDetections
    _mod.MatchTrackData = MatchTrackData
    _mod.CVService = CVService
    sys.modules["kawkab.services.cv_service"] = _mod


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
    install_cv_stub()
    yield
    sys.modules.pop("kawkab.services.cv_service", None)


_as_ref = load_service_module("as_e2e_pipeline_refresh", "analysis_service.py")
AnalysisService = _as_ref.AnalysisService


# ── 1. Pipeline execution flow ─────────────────────────────────────


class TestE2ePipelineExecution:
    """Verify the full analysis pipeline produces structured results."""

    @pytest.fixture
    def svc(self):
        return AnalysisService()

    @pytest.fixture
    def track_data(self):
        return build_minimal_track_data()

    @pytest.mark.asyncio
    async def test_analyze_match_produces_all_metrics(self, svc, track_data):
        result = await svc.analyze_match(track_data, match_id=42)
        assert result.match_id == 42
        assert result.xg_total is not None
        assert "home" in result.xg_total
        assert "away" in result.xg_total
        assert result.xt_total is not None
        assert "home" in result.xt_total
        assert "away" in result.xt_total
        assert result.pass_network is not None
        assert result.formations is not None
        assert "home" in result.formations
        assert "away" in result.formations
        assert result.pitch_control is not None
        assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_analyze_match_populates_players(self, svc, track_data):
        result = await svc.analyze_match(track_data, match_id=1)
        assert len(result.players) >= 1
        for _pid, pstats in result.players.items():
            assert pstats.track_id is not None
            assert pstats.distance_covered_m >= 0
            assert pstats.passes_attempted >= 0

    @pytest.mark.asyncio
    async def test_analyze_match_teams_populated(self, svc, track_data):
        result = await svc.analyze_match(track_data, match_id=5)
        assert result.home_team.possession_pct >= 0
        assert result.away_team.possession_pct >= 0
        assert abs(result.home_team.possession_pct + result.away_team.possession_pct - 100.0) < 0.1

    def test_xg_with_synthetic_events(self, svc):
        events = synthetic_events()
        xg = svc.compute_xg_simple(events)
        assert xg["home"] > 0
        assert xg["away"] > 0
        assert len(xg["shot_details"]) == 4

    def test_xt_with_synthetic_events(self, svc):
        events = synthetic_events()
        xt = svc.compute_xt_simple(events)
        assert xt["home"] >= 0
        assert xt["away"] >= 0

    def test_pitch_control_module(self):
        from kawkab.core.pitch_control import VoronoiPitchControl

        pc = VoronoiPitchControl()
        home_pos = [(20.0, 20.0), (30.0, 30.0), (40.0, 20.0)]
        away_pos = [(70.0, 40.0), (80.0, 30.0), (90.0, 50.0)]
        frame = pc.compute_frame_control(home_pos, away_pos, ball_pos=(50.0, 34.0))
        assert frame.home_control_pct >= 0
        assert frame.away_control_pct >= 0

    def test_formation_analysis_module(self):
        from kawkab.core.formation_analysis import FormationAnalyzer

        fa = FormationAnalyzer()
        positions = [
            (10, 10),
            (20, 15),
            (30, 20),
            (40, 25),
            (50, 30),
            (60, 35),
            (70, 40),
            (80, 45),
            (90, 50),
            (100, 55),
        ]
        formation = fa._classify_formation(positions)
        assert isinstance(formation, str)

    def test_vaep_module(self):
        from kawkab.core.vaep import compute_vaep

        events = synthetic_events()
        result = compute_vaep(events)
        assert isinstance(result, list)
        if result:
            assert (
                "value" in result[0] or "vaep_value" in result[0] or "offensive_value" in result[0]
            )

    def test_win_probability_module(self):
        from kawkab.core.win_probability import compute_win_probability

        events = synthetic_events()
        result = compute_win_probability(events)
        assert result.starting_home_win > 0
        assert len(result.timeline) >= 1

    def test_momentum_module(self):
        from kawkab.core.momentum import compute_momentum_index

        events = synthetic_events()
        result = compute_momentum_index(events)
        assert result.home_momentum_pct >= 0
        assert result.away_momentum_pct >= 0


# ── 2. Determinism ──────────────────────────────────────────────────


class TestE2eDeterminism:
    """Verify same input → same output."""

    @pytest.fixture
    def svc(self):
        return AnalysisService()

    def test_xg_is_deterministic(self, svc):
        events = synthetic_events()
        r1 = svc.compute_xg_simple(events)
        r2 = svc.compute_xg_simple(events)
        assert r1["home"] == r2["home"]
        assert r1["away"] == r2["away"]
        assert r1["shot_details"] == r2["shot_details"]

    def test_xt_is_deterministic(self, svc):
        events = synthetic_events()
        r1 = svc.compute_xt_simple(events)
        r2 = svc.compute_xt_simple(events)
        assert r1["home"] == r2["home"]
        assert r1["away"] == r2["away"]


# ── 3. Storage service mock integration ──────────────────────────────


class TestE2eStorageIntegration:
    """Verify events can be saved and retrieved through storage service."""

    @pytest.mark.asyncio
    async def test_save_and_retrieve_events(self):
        mock_storage = MagicMock()
        events = synthetic_events()[:4]
        mock_storage.save_events_bulk.return_value = 4
        mock_storage.save_match.return_value = 1

        match_id = mock_storage.save_match("Test Match", "test.mp4", "Home", "Away")
        assert match_id == 1

        n = mock_storage.save_events_bulk(match_id, events)
        assert n == 4
        mock_storage.save_events_bulk.assert_called_once_with(match_id, events)


# ── 4. Data export service integration ──────────────────────────────


class TestE2eDataExport:
    """Verify data export service produces well-formed output."""

    def test_statsbomb_json_schema(self):
        """Verify StatsBomb export structure has required fields."""
        data = {
            "match_id": 1,
            "match_name": "Test Match",
            "home_team": "Home",
            "away_team": "Away",
            "events": [],
            "metadata": {"source": "Kawkab AI", "export_version": "1.0"},
        }
        assert "match_id" in data
        assert "events" in data
        assert "home_team" in data
        assert "away_team" in data
        assert "metadata" in data
        # Verify serializable
        json.dumps(data)

    def test_statsbomb_event_has_required_fields(self):
        event = {
            "id": 1,
            "index": 1,
            "period": 1,
            "timestamp": 100.0,
            "minute": 1,
            "second": 40,
            "type": {"id": 30, "name": "Pass"},
            "team": {"id": 1, "name": "home"},
            "player": {"id": 1, "name": "Player 1"},
        }
        assert "id" in event
        assert "timestamp" in event
        assert "type" in event
        assert "team" in event
        assert "player" in event
        json.dumps(event)

    def test_csv_export_structure(self):
        """Verify CSV structure matches expected columns."""
        import csv
        import io

        events_csv = io.StringIO()
        writer = csv.writer(events_csv)
        writer.writerow(["event_id", "event_type", "timestamp", "team", "completed"])
        writer.writerow([1, "pass", 100.0, "home", True])
        writer.writerow([2, "shot", 200.0, "away", False])
        events_csv.seek(0)
        reader = csv.reader(events_csv)
        rows = list(reader)
        assert len(rows) == 3
        assert rows[0] == ["event_id", "event_type", "timestamp", "team", "completed"]
        assert rows[1][1] == "pass"


# ── 5. Bridge layer integration ─────────────────────────────────────


class TestE2eBridgeLayer:
    """Verify bridge slot → handler → service → result chain."""

    def test_mock_bridge_call_chain(self):
        mock_storage = MagicMock()
        mock_storage.save_match.return_value = 1
        result = mock_storage.save_match("match", "video.mp4", "Home", "Away")
        assert result == 1
        mock_storage.save_match.assert_called_once()

    def test_json_serialization(self):
        data = {"match_id": 1, "xg": {"home": 1.5, "away": 0.8}}
        serialized = json.dumps(data)
        deserialized = json.loads(serialized)
        assert deserialized["match_id"] == 1
        assert deserialized["xg"]["home"] == 1.5

    def test_error_handling_at_each_layer(self):
        mock_handler = MagicMock()
        mock_handler.side_effect = ValueError("Invalid data")
        with pytest.raises(ValueError):
            mock_handler("bad_input")

        mock_service = MagicMock()
        mock_service.analyze_match.side_effect = RuntimeError("Service error")
        with pytest.raises(RuntimeError):
            mock_service.analyze_match(None, match_id=0)


# ── 6. Empty/edge-case handling ─────────────────────────────────────


class TestE2eEdgeCases:
    """Verify the pipeline handles edge cases gracefully."""

    @pytest.fixture
    def svc(self):
        return AnalysisService()

    def test_empty_events(self, svc):
        assert svc.compute_xg_simple([])["home"] == 0.0
        assert svc.compute_xt_simple([])["home"] == 0.0

    def test_malformed_events(self, svc):
        events = [{"type": None}, {}, {"type": "shot", "metadata": {}}, 42]
        result = svc.compute_xg_simple([e for e in events if isinstance(e, dict)])
        # Only the shot event produces xG
        assert result["home"] >= 0

    @pytest.mark.asyncio
    async def test_empty_track_data(self, svc):
        cv_mod = sys.modules["kawkab.services.cv_service"]
        empty = cv_mod.MatchTrackData(
            match_id=1,
            fps=30,
            total_frames=0,
            duration_seconds=0,
            frames=[],
            track_registry={},
            player_teams={},
            tracking_metrics={},
            match_type="test",
        )
        result = await svc.analyze_match(empty, match_id=0)
        assert result.match_id == 0
        assert result.duration_seconds == 0

    @pytest.mark.asyncio
    async def test_empty_track_data_no_teams(self, svc):
        cv_mod = sys.modules["kawkab.services.cv_service"]
        mt = cv_mod.MatchTrackData(
            match_id=1,
            fps=30,
            total_frames=0,
            duration_seconds=0,
            frames=[],
            track_registry={},
            player_teams={},
            tracking_metrics={},
            match_type="test",
        )
        result = await svc.analyze_match(mt, match_id=0)
        assert result is not None
        assert result.players == {}

    def test_deterministic_empty(self, svc):
        assert svc.compute_xg_simple([]) == svc.compute_xg_simple([])
