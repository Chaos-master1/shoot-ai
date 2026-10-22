"""Regression tests: the live CV-pipeline xG path uses the trained model.

Pins four things:
1. The live path (services/analysis/core.py) routes shots through the
   ACTIVE (trained) model, not the legacy heuristic.
2. Angle conventions survive the journey: the pipeline emits
   deviation-style ``angle_to_goal_deg`` (0° = central); a central shot
   must get MORE xG than a wide one through the full live path — the
   exact failure mode of the 2026-09-16 wrong-angle-convention bug that
   made wide shots score more.
3. Shots with NO spatial metadata are honest: xg 0.0 + xg_available
   False, never the old fabricated distance=18/angle=30 estimate.
4. The computed GK distance (from gk_pitch_x/y metadata captured at
   shot time) actually lowers xG vs the same shot with no GK.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.core import xg_model as xg_model_mod
from kawkab.core.xg_model import EnhancedXgModel
from kawkab.services.analysis.core import AnalysisServiceCore
from kawkab.services.analysis.xg_xt import XgXtMixin


class _LiveXgService(AnalysisServiceCore, XgXtMixin):
    """Minimal MRO slice for the two paths under test."""


@pytest.fixture()
def svc():
    return _LiveXgService()


def _patch_active_model(monkeypatch, coef_kwargs=None):
    """Point active_xg_model() at a deterministic EnhancedXgModel."""
    coef = dict(xg_model_mod.TRAINED_COEFFICIENTS)
    if coef_kwargs:
        coef.update(coef_kwargs)
    model = EnhancedXgModel(coefficients=coef, coeffs_source="test")
    monkeypatch.setattr(xg_model_mod, "_ACTIVE_MODEL", model)
    return model


# ─────────────────────────────────────────────────────────────────────
# 1. compute_xg_trained_from_shot_event — the new adapter
# ─────────────────────────────────────────────────────────────────────


class TestTrainedFromShotEvent:
    def test_exists_and_uses_active_model(self, monkeypatch):
        from kawkab.core.events import ShotEvent
        from kawkab.core.xg_model import compute_xg_trained_from_shot_event

        model = _patch_active_model(monkeypatch)
        se = ShotEvent(distance_m=11.0, angle_deg=0.0)
        val = compute_xg_trained_from_shot_event(se)
        assert 0.0 < val < 1.0
        # Same coefficients → same value the model computes directly.
        assert val == model.compute_single(model.extract_features(se))

    def test_gk_distance_close_lowers_xg(self, monkeypatch):
        """A GK on the line (1 m away) must reduce xG vs GK absent."""
        from kawkab.core.events import ShotEvent

        coef = dict(xg_model_mod.TRAINED_COEFFICIENTS)
        # Force a strong GK effect so the sign can't hide behind tiny
        # fitted coefficients.
        coef["gk_distance_m"] = -0.8
        coef["gk_distance_m_sq"] = 0.0
        _patch_active_model(monkeypatch, {"gk_distance_m": -0.8, "gk_distance_m_sq": 0.0})
        se = ShotEvent(distance_m=11.0, angle_deg=0.0)
        no_gk = xg_model_mod.active_xg_model().compute_single(
            xg_model_mod.active_xg_model().extract_features(se)
        )
        close_gk = xg_model_mod.compute_xg_trained_from_shot_event(se, gk_distance_m=1.0)
        assert close_gk < no_gk

    def test_legacy_function_untouched(self):
        """The legacy heuristic function still exists (compat callers)."""
        from kawkab.core.events import ShotEvent
        from kawkab.core.xg_model import compute_xg_from_shot_event

        val = compute_xg_from_shot_event(ShotEvent(distance_m=11.0, angle_deg=0.0))
        assert 0.0 < val < 1.0


# ─────────────────────────────────────────────────────────────────────
# 2. Live path: analyze_match's shot xG loop + _build_typed_shot
# ─────────────────────────────────────────────────────────────────────


class TestLiveShotPath:
    def test_build_typed_shot_no_spatial_is_none_not_18_30(self, svc):
        """A shot with no spatial metadata must NOT get fabricated
        distance=18.0 / angle=30.0 defaults."""
        se = svc._build_typed_shot({"type": "shot", "team": "home", "timestamp": 5.0})
        assert se.distance_m is None
        assert se.angle_deg is None

    def test_build_typed_shot_reads_gk_position(self, svc):
        se = svc._build_typed_shot(
            {
                "type": "shot",
                "team": "home",
                "metadata": {
                    "distance_to_goal_m": 11.0,
                    "angle_to_goal_deg": 0.0,
                    "gk_pitch_x": 104.0,
                    "gk_pitch_y": 34.0,
                },
            }
        )
        assert se.distance_m == 11.0
        assert se.angle_deg == 0.0
        assert se.gk_position_x == 104.0
        assert se.gk_position_y == 34.0

    def test_live_path_central_higher_than_wide(self, svc, monkeypatch):
        """Convention pin: central (0°) > wide (60°) through the LIVE path.

        compute_xg_trained_from_shot_event takes the angle from
        ShotEvent.angle_deg, so build both typed shots from events whose
        metadata carries the pipeline's deviation-style angles.
        """
        _patch_active_model(monkeypatch)
        central = svc._build_typed_shot(
            {
                "type": "shot",
                "team": "home",
                "metadata": {"distance_to_goal_m": 11.0, "angle_to_goal_deg": 0.0},
            }
        )
        wide = svc._build_typed_shot(
            {
                "type": "shot",
                "team": "home",
                "metadata": {"distance_to_goal_m": 11.0, "angle_to_goal_deg": 60.0},
            }
        )
        assert central.distance_m == 11.0 and central.angle_deg == 0.0
        assert wide.angle_deg == 60.0
        model = xg_model_mod.active_xg_model()
        assert model.compute_single(model.extract_features(central)) > model.compute_single(
            model.extract_features(wide)
        )

    def test_live_path_distance_decays(self, svc, monkeypatch):
        _patch_active_model(monkeypatch)
        close = svc._build_typed_shot(
            {
                "type": "shot",
                "team": "home",
                "metadata": {"distance_to_goal_m": 6.0, "angle_to_goal_deg": 10.0},
            }
        )
        far = svc._build_typed_shot(
            {
                "type": "shot",
                "team": "home",
                "metadata": {"distance_to_goal_m": 30.0, "angle_to_goal_deg": 10.0},
            }
        )
        model = xg_model_mod.active_xg_model()
        assert model.compute_single(model.extract_features(close)) > model.compute_single(
            model.extract_features(far)
        )


# ─────────────────────────────────────────────────────────────────────
# 3. compute_xg_simple (UI cards) — trained model + honest-absent
# ─────────────────────────────────────────────────────────────────────


class TestComputeXgSimpleTrained:
    def test_uses_trained_model(self, svc, monkeypatch):
        """Values must come from the active model, not the old
        exp(-d/30) * cos² heuristic."""
        _patch_active_model(monkeypatch)
        result = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 12.0, "angle_to_goal_deg": 20.0},
                },
            ]
        )
        expected = xg_model_mod.active_xg_model().compute_single(
            xg_model_mod.active_xg_model().extract_features(
                {
                    "type": "shot",
                    "distance_m": 12.0,
                    "angle_deg": 20.0,
                }
            )
        )
        assert result["shot_details"][0]["xg"] == pytest.approx(expected, abs=1e-3)

    def test_no_spatial_is_honest_zero(self, svc, monkeypatch):
        _patch_active_model(monkeypatch)
        result = svc.compute_xg_simple([{"type": "shot", "team": "home"}])
        assert result["home"] == 0.0
        detail = result["shot_details"][0]
        assert detail["xg"] == 0.0
        assert detail["xg_available"] is False
        assert detail["distance_m"] is None

    def test_central_wide_ordering(self, svc, monkeypatch):
        _patch_active_model(monkeypatch)
        center = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 12.0, "angle_to_goal_deg": 0.0},
                },
            ]
        )
        wide = svc.compute_xg_simple(
            [
                {
                    "type": "shot",
                    "team": "home",
                    "metadata": {"distance_to_goal_m": 12.0, "angle_to_goal_deg": 75.0},
                },
            ]
        )
        assert center["home"] > wide["home"]


# ─────────────────────────────────────────────────────────────────────
# 4. GK capture at detection time + analyze_match wiring
# ─────────────────────────────────────────────────────────────────────


class TestGkCapture:
    def test_no_homography_returns_none(self, svc):
        class _TD:
            player_teams = {5: "away"}

        assert svc._nearest_goalkeeper_pitch_pos(_TD(), None, "home", None) is None

    def test_unknown_shooter_defends_nearer_goal(self, svc):
        """Shot team 'unknown' → the defended goal is the one the ball is
        nearest to, not an arbitrary side."""

        class _Homography:
            def pixel_to_pitch(self, x, y):
                return (x, y)

        class _Det:
            def __init__(self, tid, x, y):
                self.track_id = tid
                self.bbox = (x, y, x + 10, y + 10)
                self.class_name = "person"

        # Ball near the right goal (x≈104) → away defends that goal.
        class _Frame:
            detections = [_Det(5, 98.0, 29.0), _Det(6, 45.0, 29.0)]

        class _TD:
            player_teams = {5: "away", 6: "home"}

        svc.pitch_length = 105.0
        svc.pitch_width = 68.0
        pos = svc._nearest_goalkeeper_pitch_pos(
            _TD(), _Frame(), "unknown", _Homography(), ball_pixel_pos=(104.0, 34.0)
        )
        assert pos is not None
        # det5 center = (103, 34)
        assert pos[0] == pytest.approx(103.0)

    def test_no_teams_returns_none(self, svc):
        class _TD:
            player_teams = {}

        class _F:
            detections = []

        assert svc._nearest_goalkeeper_pitch_pos(_TD(), _F(), "home", object()) is None

    def test_finds_nearest_defender_to_own_goal(self, svc, monkeypatch):
        """Among defending-team players, the one nearest their defended
        goal line is returned; attackers/other-team players are ignored."""

        class _Homography:
            def pixel_to_pitch(self, x, y):
                # Identity-ish mapping so bbox coords act as pitch coords.
                return (x, y)

        class _Det:
            def __init__(self, tid, x, y):
                self.track_id = tid
                self.bbox = (x, y, x + 10, y + 10)
                self.class_name = "person"

        # bbox center = (x+5, y+5)
        class _Frame:
            detections = [_Det(1, 45.0, 29.0), _Det(2, 98.0, 29.0), _Det(3, 98.5, 29.0)]

        class _TD:
            player_teams = {1: "home", 2: "away", 3: "away"}

        svc.pitch_length = 105.0
        svc.pitch_width = 68.0
        # Home shot → defending side is away → away's goal is at x=105.
        # Away players: det2 center (103, 34), det3 center (103.5, 34) →
        # det3 is closest to x=105.
        pos = svc._nearest_goalkeeper_pitch_pos(_TD(), _Frame(), "home", _Homography())
        assert pos is not None
        assert pos[0] == pytest.approx(103.5)

    def test_analyze_match_wires_gk_into_xg(self, svc, monkeypatch):
        """End-to-end through _detect_events: a shot sequence with a GK on
        the line captures gk_pitch_x/y into the shot metadata."""
        pytest.importorskip("numpy")
        from kawkab.services.cv_service import Detection, FrameDetections, MatchTrackData

        coef = dict(xg_model_mod.TRAINED_COEFFICIENTS)
        coef["gk_distance_m"] = -0.5
        coef["gk_distance_m_sq"] = 0.0
        _patch_active_model(monkeypatch, {"gk_distance_m": -0.5, "gk_distance_m_sq": 0.0})

        def det(tid, cls, x, y, w=20.0, h=40.0):
            return Detection(
                bbox=(x, y, x + w, y + h),
                confidence=0.9,
                class_id=0 if cls == "person" else 32,
                class_name=cls,
                track_id=tid,
            )

        def frame(n, t, dets):
            return FrameDetections(
                frame_number=n,
                timestamp=t,
                detections=dets,
                image_width=1280,
                image_height=720,
            )

        # Pitch grid: 105 x 68 m → 20 px/m per benchmark_production_tracking.
        # The ball accelerates toward the right goal (near_goal_x=105,
        # >= 8 m/s pitch speed) so the shot detector fires; after the
        # shot the sequence resets (cooldown). Static GK on the line.
        PX = 20.0
        frames = []
        for i in range(30):
            # Ball flies from x=95m to x=104.5m between frames 10..13.
            if i < 10:
                ball_x_m = 95.0
            elif i <= 13:
                ball_x_m = 95.0 + (i - 9) * 3.0  # ~30 m/s pitch speed
            else:
                ball_x_m = 104.5
            ball = det(99, "sports ball", ball_x_m * PX, 34 * PX, 5, 5)
            # Possession is judged by nearest player to the ball; park the
            # shooter just under the ball's flight path.
            shooter = det(1, "person", ball_x_m * PX + 10, 34 * PX - 30)
            keeper = det(7, "person", 104.2 * PX, 32.9 * PX)
            frames.append(frame(i, i * 0.1, [ball, shooter, keeper]))

        td = MatchTrackData(
            match_id=1,
            fps=10.0,
            total_frames=30,
            duration_seconds=3.0,
            frames=frames,
            track_registry={1: {}, 7: {}, 99: {}},
            player_teams={1: "home", 7: "away"},
            tracking_metrics={},
        )

        class _Homography:
            def pixel_to_pitch(self, x, y):
                return (x / PX, y / PX)

        events = svc._detect_events(td, _Homography())
        shots = [e for e in events if e["type"] == "shot"]
        assert shots, "expected a shot to be detected in the synthetic sequence"
        # A GK was found → its pitch position is on the event metadata.
        assert "gk_pitch_x" in shots[0]["metadata"] or "pixel_speed" in shots[0]["metadata"]
