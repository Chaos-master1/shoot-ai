"""Tests for the xG model."""

from kawkab.core.events import BodyPart, ShotEvent, ShotType
from kawkab.core.xg_model import (
    XG_COEFFICIENTS,
    batch_compute_xg,
    compute_xg,
    compute_xg_from_dict,
    compute_xg_from_shot_event,
)


class TestComputeXg:
    def test_default_mid_range(self):
        xg = compute_xg(18, 0)
        assert 0.05 <= xg <= 0.20, f"xG at 18m/0deg out of range: {xg}"

    def test_close_shot_higher_xg(self):
        close = compute_xg(5, 0)
        far = compute_xg(30, 0)
        assert close > far, "Close shot should have higher xG"

    def test_central_higher_than_wide(self):
        central = compute_xg(11, 0)
        wide = compute_xg(11, 60)
        assert central > wide, "Central shot should have higher xG"

    def test_header_penalty(self):
        foot = compute_xg(11, 0, body_part="right_foot")
        header = compute_xg(11, 0, body_part="head")
        assert foot > header, "Header should be penalized vs foot"

    def test_one_on_one_bonus(self):
        normal = compute_xg(11, 0)
        one_on_one = compute_xg(11, 0, is_one_on_one=True)
        assert one_on_one > normal, "One-on-one should increase xG"

    def test_pressure_penalty(self):
        normal = compute_xg(11, 0)
        pressed = compute_xg(11, 0, is_pressed=True)
        assert normal > pressed, "Pressure should decrease xG"

    def test_through_ball_assist_bonus(self):
        standard = compute_xg(18, 0, assist_type="standard")
        through = compute_xg(18, 0, assist_type="through_ball")
        assert through > standard, "Through ball assist should increase xG"

    def test_xg_clamped_zero_to_one(self):
        xg = compute_xg(150, 90)
        assert 0.0 <= xg <= 1.0
        xg = compute_xg(0, 0, is_one_on_one=True)
        assert 0.0 <= xg <= 1.0

    def test_left_foot_vs_right_foot(self):
        left = compute_xg(18, 0, body_part="left_foot")
        right = compute_xg(18, 0, body_part="right_foot")
        assert abs(left - right) < 0.001

    def test_xg_monotonic_with_distance(self):
        values = [compute_xg(d, 0) for d in [5, 10, 15, 20, 30, 40]]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] or abs(values[i] - values[i + 1]) < 0.001

    def test_volley_vs_open_play(self):
        open_play = compute_xg(18, 0, shot_type="open_play")
        volley = compute_xg(18, 0, shot_type="volley")
        assert volley > open_play, "Volley should have slight bonus"

    def test_free_kick_vs_open_play(self):
        open_play = compute_xg(25, 0, shot_type="open_play")
        fk = compute_xg(25, 0, shot_type="free_kick")
        assert fk > open_play, "Free kick should have slight bonus"


class TestComputeXgFromShotEvent:
    def test_from_shot_event(self):
        event = ShotEvent(
            timestamp=10.0,
            team="home",
            track_id=1,
            distance_m=12.0,
            angle_deg=0.0,
            body_part=BodyPart.RIGHT_FOOT,
        )
        xg = compute_xg_from_shot_event(event)
        assert 0.07 <= xg <= 0.30

    def test_from_shot_event_with_all_features(self):
        event = ShotEvent(
            timestamp=10.0,
            team="home",
            track_id=1,
            distance_m=8.0,
            angle_deg=10.0,
            body_part=BodyPart.HEAD,
            shot_type=ShotType.VOLLEY,
            is_one_on_one=True,
            was_pressed=False,
        )
        xg = compute_xg_from_shot_event(event)
        assert 0.0 <= xg <= 1.0


class TestBatchAndCompat:
    def test_compute_xg_from_dict(self):
        d = {
            "type": "shot",
            "timestamp": 10.0,
            "team": "home",
            "distance_m": 15.0,
            "angle_deg": 0.0,
        }
        xg = compute_xg_from_dict(d)
        assert 0.0 <= xg <= 1.0

    def test_batch_compute_xg_mixed(self):
        shot = ShotEvent(timestamp=10.0, team="home", track_id=1, distance_m=12.0, angle_deg=0.0)
        results = batch_compute_xg([shot, {"type": "shot", "distance_m": 20.0}])
        assert len(results) == 2
        assert all(0.0 <= x <= 1.0 for x in results)

    def test_batch_compute_xg_skips_non_shot(self):
        results = batch_compute_xg([{"type": "pass"}])
        assert results == [0.0]

    def test_coefficients_have_expected_keys(self):
        expected = {
            "intercept",
            "distance_m",
            "distance_m_sq",
            "angle_deg_sin",
            "is_header",
            "is_pressed",
            "is_one_on_one",
            "is_penalty",
        }
        assert expected.issubset(XG_COEFFICIENTS.keys())

    def test_from_side_penalty(self):
        normal = compute_xg(15, 0)
        far_side = compute_xg(15, 0, from_side=True)
        assert normal > far_side, "Far side should have penalty"

    def test_cross_assist(self):
        normal = compute_xg(15, 0, assist_type="standard")
        cross = compute_xg(15, 0, assist_type="cross")
        assert cross != normal

    def test_half_volley_bonus(self):
        open_play = compute_xg(18, 0, shot_type="open_play")
        half_volley = compute_xg(18, 0, shot_type="half_volley")
        assert half_volley > open_play, "Half volley should have bonus"

    def test_extreme_distance_clamping(self):
        xg = compute_xg(0, 0)
        assert 0.0 <= xg <= 1.0

    def test_shot_event_defaults(self):
        event = ShotEvent(timestamp=10.0, team="home", track_id=1, distance_m=12.0, angle_deg=0.0)
        xg = compute_xg_from_shot_event(event)
        assert 0.0 <= xg <= 1.0


class TestEnhancedXgModel:
    """Tests for the enhanced xG model (EnhancedXgModel)."""

    def _enhanced_model_for_test(self):
        from kawkab.core.xg_model import ENHANCED_COEFFICIENTS, EnhancedXgModel

        return EnhancedXgModel(coefficients=ENHANCED_COEFFICIENTS, coeffs_source="test")

    def test_central_higher_than_wide(self):
        from kawkab.core.xg_model import EnhancedXgFeatures

        model = self._enhanced_model_for_test()
        central = model.compute_single(
            EnhancedXgFeatures(
                distance_m=12.0,
                angle_deg=0.0,
                is_penalty=False,
            )
        )
        wide = model.compute_single(
            EnhancedXgFeatures(
                distance_m=12.0,
                angle_deg=60.0,
                is_penalty=False,
            )
        )
        assert central > wide, "Central shot should have higher xG than wide shot"

    def test_monotonic_decreasing_with_angle(self):
        """xG must never increase as shooting angle widens, all else equal."""
        from kawkab.core.xg_model import EnhancedXgFeatures

        model = self._enhanced_model_for_test()
        angles = [0, 10, 20, 30, 45, 60, 75, 89]
        xgs = []
        for a in angles:
            xg = model.compute_single(
                EnhancedXgFeatures(
                    distance_m=15.0,
                    angle_deg=float(a),
                    is_penalty=False,
                )
            )
            xgs.append(xg)
        for i in range(len(xgs) - 1):
            assert xgs[i] >= xgs[i + 1] - 1e-9, (
                f"xG increased from {xgs[i]:.4f} at {angles[i]}° to "
                f"{xgs[i + 1]:.4f} at {angles[i + 1]}°"
            )

    def test_near_distance_penalty(self):
        from kawkab.core.xg_model import EnhancedXgFeatures

        model = self._enhanced_model_for_test()
        close = model.compute_single(
            EnhancedXgFeatures(
                distance_m=3.0,
                angle_deg=0.0,
                is_penalty=False,
            )
        )
        far = model.compute_single(
            EnhancedXgFeatures(
                distance_m=30.0,
                angle_deg=0.0,
                is_penalty=False,
            )
        )
        assert close > far, "Close shot should have higher xG"

    def test_penalty_constant(self):
        from kawkab.core.xg_model import PENALTY_XG, EnhancedXgFeatures

        model = self._enhanced_model_for_test()
        xg = model.compute_single(
            EnhancedXgFeatures(
                distance_m=12.0,
                angle_deg=0.0,
                is_penalty=True,
            )
        )
        assert xg == PENALTY_XG

    def test_one_on_one_bonus(self):
        from kawkab.core.xg_model import EnhancedXgFeatures

        model = self._enhanced_model_for_test()
        normal = model.compute_single(
            EnhancedXgFeatures(
                distance_m=15.0,
                angle_deg=10.0,
                is_penalty=False,
            )
        )
        one_on_one = model.compute_single(
            EnhancedXgFeatures(
                distance_m=15.0,
                angle_deg=10.0,
                is_penalty=False,
                is_one_on_one=True,
            )
        )
        assert one_on_one > normal, "One-on-one should increase xG"

    def test_batch_monotonicity(self):
        from kawkab.core.events import BodyPart, ShotEvent, ShotType

        model = self._enhanced_model_for_test()
        angles = [0, 30, 60, 89]
        events = [
            ShotEvent(
                timestamp=float(i),
                team="home",
                track_id=1,
                distance_m=15.0,
                angle_deg=float(a),
                body_part=BodyPart.RIGHT_FOOT,
                shot_type=ShotType.OPEN_PLAY,
            )
            for i, a in enumerate(angles)
        ]
        xgs = model.batch_compute(events)
        for i in range(len(xgs) - 1):
            assert xgs[i] >= xgs[i + 1] - 1e-9, (
                f"Batch xG increased from {xgs[i]:.4f} at {angles[i]}° to "
                f"{xgs[i + 1]:.4f} at {angles[i + 1]}°"
            )
