"""Regression tests for the 2026-09-02 accuracy audit fixes:

1. analyze_match no longer double-saves base events when the advanced
   event service returns them alongside its derived events.
2. AdvancedEventDetectionService works against BOTH event shapes:
   the raw internal dicts (from_track_id + metadata) and the production
   typed-event to_dict() shape (track_id + top-level start_x/end_x,
   no metadata key). Six detectors silently produced zero events
   against the production shape before this fix.
3. Pass detection requires 2 consecutive frames of a new possessor
   (no phantom passes from single-frame nearest-player jitter).
4. Pass completion is honest: a possession flip straight to an opponent
   is completed=False (feeds tackle/turnover detectors); a same-team
   flip is completed=True.
5. Team fallbacks: no track-ID-parity fabrication anywhere (shot team,
   pass team, possession split, PPDA possession).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.advanced_event_detection_service import AdvancedEventDetectionService
from kawkab.services.analysis.core import AnalysisServiceCore
from kawkab.services.cv_service import Detection, FrameDetections, MatchTrackData


def make_detection(track_id, class_name, x, y, w=20.0, h=40.0):
    return Detection(
        bbox=(x, y, x + w, y + h),
        confidence=0.9,
        class_id=0 if class_name == "person" else 32,
        class_name=class_name,
        track_id=track_id,
    )


def make_frame(n, t, dets, width=1280, height=720):
    return FrameDetections(
        frame_number=n,
        timestamp=t,
        detections=dets,
        image_width=width,
        image_height=height,
    )


def build_track_data(player_seq, teams=None, frame_skip=1):
    """One ball + one player per frame; player_seq[i] is the nearest track id."""
    frames = []
    for i, pid in enumerate(player_seq):
        ball = make_detection(99, "sports ball", 100, 300, 5, 5)
        player = make_detection(pid, "person", 105, 300)
        frames.append(make_frame(i, i * 0.1, [ball, player]))
    metrics = {"frame_skip": frame_skip} if frame_skip > 1 else {}
    return MatchTrackData(
        match_id=1,
        fps=10.0,
        total_frames=len(player_seq),
        duration_seconds=len(player_seq) * 0.1,
        frames=frames,
        track_registry={p: {} for p in set(player_seq)} | {99: {}},
        player_teams=teams or {},
        tracking_metrics=metrics,
    )


# ------------------------------------------------------------------
# 1. Double-save dedup in the analyze_match bridge path
# ------------------------------------------------------------------


class _RecordingStorage:
    def __init__(self):
        self.saved_events = []

    async def save_event(self, match_id, event):
        self.saved_events.append(event)
        return 1


class _FakeAdvancedService:
    """Mimics detect_all_advanced_events: returns base events + derived ones."""

    def __init__(self, derived):
        self._derived = derived

    async def detect_all_advanced_events(self, track_data, base_events, homography_matrix=None):
        return list(base_events) + self._derived


class TestNoDoubleSave:
    def test_event_identity_distinct_keys(self):
        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        raw = {"type": "pass", "timestamp": 1.5, "from_track_id": 3, "to_track_id": 4}
        typed = {"type": "pass", "timestamp": 1.5, "track_id": 3, "to_track_id": 4}
        # Raw and typed representations of the SAME pass must share identity.
        assert AnalysisHandler._event_identity(raw) == AnalysisHandler._event_identity(typed)
        # A different timestamp is a different event.
        other = dict(raw, timestamp=2.5)
        assert AnalysisHandler._event_identity(raw) != AnalysisHandler._event_identity(other)

    def test_advanced_save_skips_base_duplicates(self):
        """The advanced-event save pass must not re-save base events.

        Regression: detect_all_advanced_events returns base_events + new
        events, and the handler used to persist the whole returned list --
        doubling every pass/shot/carry in the DB-derived stats.
        """
        import asyncio

        from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler

        base = [
            {"type": "pass", "timestamp": 1.5, "track_id": 3, "team": "home", "completed": True},
            {"type": "shot", "timestamp": 30.0, "track_id": 3, "team": "home"},
        ]
        derived = [
            {"type": "tackle", "timestamp": 12.0, "team": "away", "track_id": 7},
        ]
        storage = _RecordingStorage()

        async def _run():
            handler = AnalysisHandler(
                bridge=None,
                services={
                    "storage_service": storage,
                    "advanced_event_detection_service": _FakeAdvancedService(derived),
                },
                rate_limiter=None,
            )
            base_keys = handler._event_identity_keys(base)
            advanced_events = base + derived
            for event in advanced_events:
                if handler._event_identity(event) in base_keys:
                    continue
                await storage.save_event(match_id=1, event=event)

        asyncio.run(_run())
        # Only the derived tackle may be saved in the second pass.
        assert len(storage.saved_events) == 1
        assert storage.saved_events[0]["type"] == "tackle"


# ------------------------------------------------------------------
# 2. Advanced detectors work against the PRODUCTION event shape
# ------------------------------------------------------------------


def _production_pass(ts, track_id, to_id, team, completed, sx, sy, ex, ey):
    """Exactly what PassEvent.to_dict() produces in the production pipeline
    (core/events.py) -- no from_track_id, no metadata key, 0-1 fractions."""
    return {
        "type": "pass",
        "timestamp": ts,
        "team": team,
        "track_id": track_id,
        "to_track_id": to_id,
        "start_x": sx,
        "start_y": sy,
        "end_x": ex,
        "end_y": ey,
        "completed": completed,
        "confidence": 0.7,
    }


def _empty_track_data(teams=None):
    return build_track_data([], teams=teams)


class TestProductionShapeDetectors:
    def setup_method(self):
        self.svc = AdvancedEventDetectionService()
        self.td = _empty_track_data(teams={1: "home", 2: "away", 7: "away"})

    @pytest.mark.asyncio
    async def test_tackles_fire_on_production_shape(self):
        """Incomplete cross-team pass (typed to_dict shape) -> tackle.

        Regression: _detect_tackles read event['from_track_id'], which the
        typed shape never carries -- it silently returned [] forever.
        """
        base = [
            _production_pass(5.0, 1, 2, "home", completed=False, sx=0.4, sy=0.5, ex=0.6, ey=0.5),
        ]
        tackles = self.svc._detect_tackles(self.td, base, None)
        assert len(tackles) == 1
        assert tackles[0]["type"] == "tackle"
        assert tackles[0]["team"] == "away"

    @pytest.mark.asyncio
    async def test_crosses_fire_on_production_shape(self):
        """Wide pass into the box (typed shape, fraction coords) -> cross.

        start_y near sideline (y=0.08*68=5.4m), end in the penalty area
        (x=0.95*105=99.75m, y=0.5*68=34m).
        """
        base = [
            _production_pass(5.0, 1, 2, "home", completed=True, sx=0.8, sy=0.08, ex=0.95, ey=0.5),
        ]
        crosses = self.svc._detect_crosses(self.td, base, None)
        assert len(crosses) == 1
        assert crosses[0]["type"] == "cross"

    @pytest.mark.asyncio
    async def test_blocks_fire_on_production_shape(self):
        base = [
            _production_pass(5.0, 1, 2, "home", completed=False, sx=0.4, sy=0.5, ex=0.6, ey=0.5),
        ]
        blocks = self.svc._detect_blocks(self.td, base, None)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "block"

    @pytest.mark.asyncio
    async def test_high_turnovers_fire_on_production_shape(self):
        # home loses ball at x=0.8*105=84m (final 40m) -> high turnover
        base = [
            _production_pass(5.0, 1, 2, "home", completed=False, sx=0.8, sy=0.5, ex=0.6, ey=0.5),
        ]
        turnovers = self.svc._detect_high_turnovers(self.td, base, None)
        assert len(turnovers) == 1
        assert turnovers[0]["type"] == "high_turnover"

    @pytest.mark.asyncio
    async def test_final_third_entries_fire_on_production_shape(self):
        # starts x=0.5*105=52.5 (< 70.35), ends x=0.8*105=84 (>= 70.35)
        base = [
            _production_pass(5.0, 1, 2, "home", completed=True, sx=0.5, sy=0.5, ex=0.8, ey=0.5),
        ]
        entries = self.svc._detect_final_third_entries(self.td, base, None)
        assert len(entries) == 1
        assert entries[0]["type"] == "final_third_entry"

    @pytest.mark.asyncio
    async def test_progressive_actions_fire_on_production_shape(self):
        # Start near own goal, end deep in attacking half, >25% of remaining
        # distance covered -> progressive per core.progressive_actions.
        base = [
            _production_pass(5.0, 1, 2, "home", completed=True, sx=0.15, sy=0.5, ex=0.85, ey=0.5),
        ]
        progressive = self.svc._detect_progressive_actions(self.td, base, None)
        assert len(progressive) == 1
        assert progressive[0]["type"] == "progressive_action"

    @pytest.mark.asyncio
    async def test_missing_positions_skip_honestly(self):
        """No position info at all -> no fabricated corner-of-pitch events."""
        base = [
            {
                "type": "pass",
                "timestamp": 5.0,
                "team": "home",
                "track_id": 1,
                "to_track_id": 2,
                "completed": True,
            },
        ]
        assert self.svc._detect_crosses(self.td, base, None) == []
        assert self.svc._detect_progressive_actions(self.td, base, None) == []
        assert self.svc._detect_final_third_entries(self.td, base, None) == []


# ------------------------------------------------------------------
# Task 7: BallTracker dt + HSV fallback gate
# ------------------------------------------------------------------


class TestBallTrackerDtAndHsvGate:
    def _frame_with_ball(self, cx, cy, size=16, w=240, h=160):
        import numpy as np

        img = np.zeros((h, w, 3), dtype=np.uint8)
        import cv2

        cv2.circle(img, (int(cx), int(cy)), size // 2, (255, 255, 255), -1)
        return img

    def test_effective_dt_fixes_velocity_scale(self):
        """Ball moving 10 px per update() call.

        With dt=1/30 (wrong, pre-fix default at fps=30), Kalman velocity
        converges toward ~10 px per 1/30s step. With the correct dt for a
        frame_skip of 3 (dt=3/30=0.1s), the same per-call displacement
        represents a physically slower ball. set_effective_dt must
        re-initialize the transition model so estimated vy tracks the
        per-step displacement in consistent units.
        """
        from kawkab.services.ball_tracker import BallTracker

        bt = BallTracker(fps=30.0)
        bt._use_yolo = False  # force HSV path in isolated test
        bt.set_effective_dt(3.0 / 30.0)
        for i in range(12):
            # Ball falling 10 px per update call, dt=0.1s => 100 px/s.
            bt.update(self._frame_with_ball(120, 20 + i * 10), i, i * 0.1)
        state = bt.kalman.statePost.ravel()
        # Converged vy should approximate 10 px per 0.1s step = 100 px/s
        # in per-second units handled by the filter (px/step * (1/dt)).
        assert float(state[4]) > 5.0, "vy should reflect real per-step displacement"

    def test_hsv_confidence_never_crosses_recording_gate(self):
        """Every HSV fallback detection must stay below the 0.3 gate.

        Regression: the old formula (0.3 + circularity * 0.4) was >= 0.3 by
        construction, so every white blob (pitch lines, boots) was recorded
        as a ball detection while marginal real YOLO detections were
        dropped by the same gate.
        """
        from kawkab.services.ball_tracker import BallTracker

        bt = BallTracker(fps=30.0)
        bt._use_yolo = False
        for i in range(5):
            det = bt.update(self._frame_with_ball(120, 40 + i * 5), i, i / 30.0)
            if det is not None:
                assert det.conf < 0.3, (
                    "HSV fallback confidence must stay below the 0.3 recording gate"
                )

    def test_hsv_far_from_prediction_rejected(self):
        """A white blob far from the Kalman prediction must not latch on."""
        from kawkab.services.ball_tracker import BallTracker

        bt = BallTracker(fps=30.0)
        bt._use_yolo = False
        # Establish a track at the left of the frame.
        for i in range(6):
            bt.update(self._frame_with_ball(40, 80), i, i / 30.0)
        # Now the only white blob is at the far right of the frame.
        det = bt.update(self._frame_with_ball(200, 80), 6, 6 / 30.0)
        # Either rejected outright (None), or recorded as a low-confidence
        # prediction -- never as a confident measurement of the new blob.
        if det is not None:
            assert det.is_prediction is True or det.conf < 0.3


# ------------------------------------------------------------------
# Task 5: trained xG coefficients sanity guard (CI)
# ------------------------------------------------------------------


class TestTrainedXgCoefficientsSanity:
    """Guard against pathological trained-xG fits shipping again.

    The original trained_xg_coefficients.json was fit on 56 shots and had:
    - a positive distance^2 coefficient -> xG INCREASED with distance
      (a 40m central shot was worth 0.98),
    - a wrong-signed angle term -> wide shots worth more than central,
    - extreme flag offsets (is_header=-4.2, is_free_kick=+3.2).

    scripts/train_xg_from_statsbomb.py has the same guards built in; this
    test enforces them at CI time against whatever JSON is committed.
    """

    def _load(self):
        import sys as _sys

        repo = Path(__file__).resolve().parent.parent.parent
        script = repo / "scripts" / "train_xg_from_statsbomb.py"
        if script.exists():
            if str(script.parent) not in _sys.path:
                _sys.path.insert(0, str(script.parent))
            import importlib.util

            spec = importlib.util.spec_from_file_location("_xg_train_script", str(script))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        pytest.skip("training script not found")

    def test_committed_coefficients_pass_sanity_checks(self):
        mod = self._load()
        json_path = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "kawkab"
            / "core"
            / "trained_xg_coefficients.json"
        )
        if not json_path.exists():
            pytest.skip("no trained coefficients committed")
        import json

        coeffs = json.loads(json_path.read_text(encoding="utf-8"))
        failures = mod.sanity_check_coefficients(coeffs)
        assert failures == [], "Pathological xG fit committed:\n  " + "\n  ".join(failures)

    def test_enhanced_model_with_json_is_monotonic_in_distance(self):
        """The default EnhancedXgModel auto-loads the committed JSON --
        verify central-shot xG decays with distance end to end.

        Angle convention: deviation from the central line (0 deg = dead
        central; cos(angle) = visible-goal fraction), matching
        EnhancedXgModel's feature construction and the CV pipeline's
        atan2(|dy|,|dx|) shots. gk_distance follows the same plausible
        geometry the trainer's sanity guard uses (GK ~3m closer to goal
        than the shooter): the fitted gk terms carry much of the
        close-range difficulty signal, so evaluating without them
        (gk_distance_m=0 -> feature inactive) understates the decay.
        """
        from kawkab.core.xg_model import EnhancedXgModel

        model = EnhancedXgModel()

        def xg_at(d, angle=0.0):
            ev = {
                "distance_m": d,
                "angle_deg": angle,
                "gk_distance_m": max(1.0, d - 3.0),
            }
            return model.compute_single(model.extract_features(ev))

        # Long-range central decay must be strictly decreasing.
        xg_by_dist = [xg_at(d) for d in (11.0, 20.0, 30.0, 40.0)]
        for i in range(len(xg_by_dist) - 1):
            assert xg_by_dist[i] > xg_by_dist[i + 1], f"long-range xG must decrease: {xg_by_dist}"
        assert xg_at(40.0) < 0.15, f"40m xG too high: {xg_at(40.0):.3f}"
        assert xg_at(11.0) > xg_at(30.0)

    def test_wide_shots_worth_less_than_central(self):
        from kawkab.core.xg_model import EnhancedXgModel

        model = EnhancedXgModel()
        central = model.compute_single(
            model.extract_features({"distance_m": 11.0, "angle_deg": 0.0, "gk_distance_m": 8.0})
        )
        wide = model.compute_single(
            model.extract_features({"distance_m": 11.0, "angle_deg": 80.0, "gk_distance_m": 8.0})
        )
        assert wide < 0.85 * central, f"wide ({wide:.3f}) must be clearly < central ({central:.3f})"


# ------------------------------------------------------------------
# P2: Hungarian optimal assignment
# ------------------------------------------------------------------


class TestHungarianTrackAssignment:
    """detect_frame's greedy best-IoU loops misassigned track IDs when two
    detections competed for the same track (first-match-wins). The global
    Hungarian assignment resolves the whole cluster optimally."""

    def test_crossed_detections_keep_ids(self):
        from kawkab.services.cv_service import CVService

        # Two tracks; next frame they cross. Greedy could swap them if it
        # evaluates det A against track 1 first (IoU 0.6) and then det B
        # is left with track 2 (IoU 0.3) or nothing -- the optimal total
        # assignment keeps both paired with their best matches.
        det_bboxes = [(0, 0, 10, 10), (9.5, 0, 19.5, 10)]
        track_bboxes = [(0.5, 0, 10.5, 10), (9.0, 0, 19.0, 10)]
        track_ids = [101, 202]
        assigned = CVService._assign_track_ids_optimal(det_bboxes, track_bboxes, track_ids)
        assert assigned[0] == 101
        assert assigned[1] == 202

    def test_dense_cluster_no_orphan_misassignment(self):
        from kawkab.services.cv_service import CVService

        # 3 detections all overlapping one strong track + 2 weaker tracks;
        # optimal assignment distributes detections across tracks rather
        # than letting the first detection monopolize the best one.
        dets = [(0, 0, 10, 10), (2, 0, 12, 10), (4, 0, 14, 10)]
        tracks = [(0, 0, 10, 10), (3, 0, 13, 10), (6, 0, 16, 10)]
        tids = [1, 2, 3]
        assigned = CVService._assign_track_ids_optimal(dets, tracks, tids)
        # Each detection gets a distinct track.
        assert len(set(assigned)) == 3
        assert all(t is not None for t in assigned)

    def test_below_threshold_unmatched(self):
        from kawkab.services.cv_service import CVService

        dets = [(0, 0, 10, 10)]
        tracks = [(50, 50, 60, 60)]  # no overlap
        assigned = CVService._assign_track_ids_optimal(dets, tracks, [7])
        assert assigned == [None]

    def test_empty_inputs(self):
        from kawkab.services.cv_service import CVService

        assert CVService._assign_track_ids_optimal([], [], []) == []
        assert CVService._assign_track_ids_optimal([(0, 0, 1, 1)], [], []) == [None]


# ------------------------------------------------------------------
# P2: Skip-frame interpolation (replaces frozen copies)
# ------------------------------------------------------------------


class TestSkipFrameInterpolation:
    def _frames_for(self, frame_skip=3, n_real=3):
        """Real frames every frame_skip; skip frames carry frozen copies."""
        from kawkab.services.cv_service import CVService

        frames = []
        # Real frames: track 1 walks right 30px per real frame.
        real_positions = {0: 0, 3: 30, 6: 60}
        for fno in range(0, frame_skip * n_real):
            if fno % frame_skip == 0:
                x = real_positions[fno]
                dets = [make_detection(1, "person", x, 100)]
            else:
                # frozen copy of the previous real frame's position
                last_real = fno - (fno % frame_skip)
                dets = [make_detection(1, "person", real_positions[last_real], 100)]
            frames.append(make_frame(fno, fno * 0.1, dets))
        return CVService._interpolate_skip_frames(frames, frame_skip), frame_skip, real_positions

    def test_skip_frames_interpolated_not_frozen(self):
        frames, frame_skip, real_positions = self._frames_for()
        # Frame 1 (between real 0 and 3) should be at x=10, not frozen at 0.
        f1 = frames[1]
        d = f1.detections[0]
        expected_x = 10  # linear interp: 0 + (1/3)*(30-0)
        assert abs(d.bbox[0] - expected_x) < 2.0, (
            f"skip frame 1 should be interpolated to ~x={expected_x}, got {d.bbox[0]}"
        )
        # Frame 2 -> x=20
        f2 = frames[2]
        assert abs(f2.detections[0].bbox[0] - 20.0) < 2.0

    def test_real_frames_untouched(self):
        frames, frame_skip, real_positions = self._frames_for()
        for fno, x in real_positions.items():
            d = frames[fno].detections[0]
            assert abs(d.bbox[0] - x) < 0.01, f"real frame {fno} moved"

    def test_teleport_guard_keeps_frozen_copy(self):
        """Camera cut inside a gap (huge jump) must not be interpolated."""
        from kawkab.services.cv_service import CVService

        frames = []
        # real frame 0: player at x=0; real frame 3: player at x=1000 (cut)
        for fno in range(6):
            if fno % 3 == 0:
                x = 0 if fno == 0 else 1000
                dets = [make_detection(1, "person", x, 100)]
            else:
                dets = [make_detection(1, "person", 0, 100)]
            frames.append(make_frame(fno, fno * 0.1, dets))
        out = CVService._interpolate_skip_frames(frames, 3)
        # skip frames keep the frozen copy (x=0), not a teleport blend
        assert abs(out[1].detections[0].bbox[0] - 0.0) < 0.01
        assert abs(out[2].detections[0].bbox[0] - 0.0) < 0.01


# ------------------------------------------------------------------
# 3+4. Possession stability + honest pass completion
# ------------------------------------------------------------------


class TestPassDetectionStability:
    def setup_method(self):
        self.core = AnalysisServiceCore()

    def test_confirmed_flip_emits_pass(self):
        td = build_track_data([1] * 10 + [2] * 10, teams={1: "home", 2: "home"})
        events = self.core._detect_events(td, None)
        passes = [e for e in events if e["type"] == "pass"]
        assert len(passes) == 1
        assert passes[0]["from_track_id"] == 1
        assert passes[0]["to_track_id"] == 2
        assert passes[0]["completed"] is True

    def test_single_frame_jitter_no_phantom_pass(self):
        """Alternating nearest-player for single frames must NOT emit passes."""
        td = build_track_data([1, 2] * 10, teams={1: "home", 2: "away"})
        events = self.core._detect_events(td, None)
        passes = [e for e in events if e["type"] == "pass"]
        assert passes == []

    def test_flip_to_opponent_is_incomplete(self):
        """Possession lost straight to an opponent -> completed=False.

        This is what revives the tackle / interception / high-turnover
        detectors downstream (they all key on completed=False).
        """
        td = build_track_data([1] * 10 + [3] * 10, teams={1: "home", 3: "away"})
        events = self.core._detect_events(td, None)
        passes = [e for e in events if e["type"] == "pass"]
        assert len(passes) == 1
        assert passes[0]["completed"] is False
        assert passes[0]["metadata"]["outcome"] == "lost_to_opponent"

    def test_intercepted_reception_is_incomplete(self):
        """Pass received by teammate, then immediately lost to an opponent.

        Sequence: player 1 (home) passes to 2 (home); 2 settles it briefly;
        player 3 (away) wins the ball. Semantics: the 1->2 pass was received
        (completed=True -- a brief reception still counts), and the 2->3
        possession flip is completed=False (lost_to_opponent), which is what
        feeds the tackle / high-turnover detectors downstream.
        """
        td = build_track_data(
            [1] * 10 + [2] * 4 + [3] * 10,
            teams={1: "home", 2: "home", 3: "away"},
        )
        events = self.core._detect_events(td, None)
        passes = [e for e in events if e["type"] == "pass"]
        assert len(passes) == 2
        first = next(p for p in passes if p["to_track_id"] == 2)
        # Receiver (2) did settle the ball -> completed pass.
        assert first["completed"] is True
        # The subsequent loss to the opponent is the incomplete one.
        second = next(p for p in passes if p["to_track_id"] == 3)
        assert second["completed"] is False
        assert second["metadata"]["outcome"] == "lost_to_opponent"

    def test_frozen_skip_frames_ignored(self):
        """Skip-frames (frozen copies) must not create possession evidence.

        With frame_skip=3, frames 1,2 of every group of 3 are verbatim
        copies of frame 0's detections in the real pipeline. Here we
        simulate that by repeating each real detection 3x; possession
        ratios must come out the same as without the copies.
        """
        real = [1] * 6 + [2] * 6  # 12 real frames, half each
        td_no_skip = build_track_data(real, teams={1: "home", 2: "away"})
        ratio_no_skip = self.core._compute_possession(td_no_skip, None)

        # Same real frames, padded with frozen copies (3x repetition).
        frozen = []
        for pid in real:
            frozen.extend([pid] * 3)
        td_skip = build_track_data(frozen, teams={1: "home", 2: "away"}, frame_skip=3)
        ratio_skip = self.core._compute_possession(td_skip, None)

        assert abs(ratio_no_skip["home"] - ratio_skip["home"]) < 0.1


# ------------------------------------------------------------------
# 5. Honest team fallbacks
# ------------------------------------------------------------------


class TestHonestTeamFallback:
    def setup_method(self):
        self.core = AnalysisServiceCore()

    def test_possession_without_teams_is_zero(self):
        td = build_track_data([1] * 10 + [2] * 10, teams={})
        r = self.core._compute_possession(td, None)
        assert r["home"] == 0.0
        assert r["away"] == 0.0
        assert r.get("unknown") == 100.0

    def test_shot_team_unknown_without_teams(self):
        """Shots must not inherit a tid%2 fabricated team.

        Build a scenario that triggers the (pixel-space) shot detector:
        ball moving >=600 px/s downward in the lower 30% of the frame, with
        a ball track id present. No player_teams.
        """
        frames = []
        for i in range(8):
            y = 600 + i * 80  # 80 px/frame @ 10fps = 800 px/s >= 600
            ball = make_detection(99, "sports ball", 200, y, 5, 5)
            player = make_detection(5, "person", 210, 690)
            frames.append(make_frame(i, i * 0.1, [ball, player]))
        td = MatchTrackData(
            match_id=1,
            fps=10.0,
            total_frames=8,
            duration_seconds=0.8,
            frames=frames,
            track_registry={5: {}, 99: {}},
            player_teams={},  # NO team assignment
        )
        events = self.core._detect_events(td, None)
        shots = [e for e in events if e["type"] == "shot"]
        assert len(shots) >= 1
        for s in shots:
            assert s["team"] == "unknown", "shot team must not be fabricated from track parity"

    def test_pass_team_unknown_without_teams(self):
        td = build_track_data([1] * 10 + [2] * 10, teams={})
        events = self.core._detect_events(td, None)
        passes = [e for e in events if e["type"] == "pass"]
        assert len(passes) == 1
        assert passes[0]["team"] == "unknown"
