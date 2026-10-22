"""Tests for Expected Pass Completion (EP) probability model."""

import math

import numpy as np

from kawkab.core.expected_pass import (
    EP_COEFFICIENTS,
    ExpectedPassResult,
    _classify_difficulty,
    _feature_vector,
    _is_progressive,
    compute_ep,
    compute_ep_batch,
)

# ── Difficulty classification ─────────────────────────────────────────────────


class TestClassifyDifficulty:
    def test_easy(self):
        assert _classify_difficulty(0.85) == "easy"
        assert _classify_difficulty(0.92) == "easy"
        assert _classify_difficulty(1.0) == "easy"

    def test_moderate(self):
        assert _classify_difficulty(0.65) == "moderate"
        assert _classify_difficulty(0.75) == "moderate"
        assert _classify_difficulty(0.84) == "moderate"

    def test_difficult(self):
        assert _classify_difficulty(0.40) == "difficult"
        assert _classify_difficulty(0.55) == "difficult"
        assert _classify_difficulty(0.64) == "difficult"

    def test_very_difficult(self):
        assert _classify_difficulty(0.0) == "very_difficult"
        assert _classify_difficulty(0.20) == "very_difficult"
        assert _classify_difficulty(0.39) == "very_difficult"


# ── Progressive pass detection ────────────────────────────────────────────────


class TestIsProgressive:
    def test_forward_pass_long_enough(self):
        assert _is_progressive(50.0, 80.0, 30.0, attacking_direction=1) is True

    def test_forward_pass_shorter_than_threshold(self):
        assert _is_progressive(50.0, 55.0, 5.0, attacking_direction=1) is False

    def test_backward_pass_not_progressive(self):
        assert _is_progressive(60.0, 40.0, 20.0, attacking_direction=1) is False

    def test_lateral_pass_not_progressive(self):
        assert _is_progressive(50.0, 50.0, 0.0, attacking_direction=1) is False

    def test_attacking_direction_negative(self):
        assert _is_progressive(60.0, 30.0, 30.0, attacking_direction=-1) is True

    def test_forward_progress_meets_25_percent_rule(self):
        # 105 * 0.25 = 26.25, so 27m forward counts even if distance is small
        assert _is_progressive(50.0, 77.0, 10.0, attacking_direction=1) is True

    def test_forward_progress_below_25_percent_and_short(self):
        assert _is_progressive(50.0, 75.0, 10.0, attacking_direction=1) is False


# ── Single-pass EP computation ────────────────────────────────────────────────


class TestComputeEP:
    def test_default_empty_dict(self):
        result = compute_ep({})
        assert 0.0 <= result.ep <= 1.0
        assert isinstance(result, ExpectedPassResult)
        assert "intercept" in result.factors

    def test_short_pass_high_probability(self):
        result = compute_ep({"distance_m": 5.0})
        assert result.ep >= 0.70
        assert result.difficulty in ("easy", "moderate")

    def test_long_ball_reduces_probability(self):
        short = compute_ep({"distance_m": 10.0, "is_long_ball": False})
        long_ = compute_ep({"distance_m": 10.0, "is_long_ball": True})
        assert long_.ep < short.ep

    def test_through_ball_reduces_probability(self):
        normal = compute_ep({"distance_m": 15.0, "is_through_ball": False})
        through = compute_ep({"distance_m": 15.0, "is_through_ball": True})
        assert through.ep < normal.ep

    def test_cross_reduces_probability(self):
        normal = compute_ep({"distance_m": 20.0, "is_cross": False})
        cross = compute_ep({"distance_m": 20.0, "is_cross": True})
        assert cross.ep < normal.ep

    def test_pressure_reduces_probability(self):
        free = compute_ep({"distance_m": 12.0, "is_pressured": False})
        pressured = compute_ep({"distance_m": 12.0, "is_pressured": True})
        assert pressured.ep < free.ep

    def test_headed_reduces_probability(self):
        foot = compute_ep({"distance_m": 10.0, "is_headed": False})
        head = compute_ep({"distance_m": 10.0, "is_headed": True})
        assert head.ep < foot.ep

    def test_receiver_pressured_reduces_probability(self):
        free = compute_ep({"distance_m": 10.0, "receiver_pressured": False})
        pressed = compute_ep({"distance_m": 10.0, "receiver_pressured": True})
        assert pressed.ep < free.ep

    def test_angle_reduces_probability(self):
        straight = compute_ep({"distance_m": 15.0, "angle_deg": 0.0})
        angled = compute_ep({"distance_m": 15.0, "angle_deg": 45.0})
        assert angled.ep < straight.ep

    def test_start_x_advantage(self):
        deep_own = compute_ep({"distance_m": 10.0, "start_x": 10.0})
        deep_opp = compute_ep({"distance_m": 10.0, "start_x": 95.0})
        # Higher start_x (closer to opponent goal) gives higher EP via +0.15 coefficient
        assert deep_opp.ep > deep_own.ep

    def test_ep_range_always_zero_to_one(self):
        for dist in [0.0, 50.0, 100.0]:
            for angle in [0.0, 90.0]:
                for through in [False, True]:
                    for cross in [False, True]:
                        for pressured in [False, True]:
                            r = compute_ep(
                                {
                                    "distance_m": dist,
                                    "angle_deg": angle,
                                    "is_through_ball": through,
                                    "is_cross": cross,
                                    "is_pressured": pressured,
                                }
                            )
                            assert 0.0 <= r.ep <= 1.0, f"EP {r.ep} out of range"

    def test_zero_distance_pass(self):
        result = compute_ep({"distance_m": 0.0, "start_x": 50.0, "end_x": 50.0})
        assert result.ep > 0.80
        assert result.is_progressive is False
        assert result.difficulty == "moderate"

    def test_extreme_long_distance(self):
        result = compute_ep({"distance_m": 100.0})
        assert result.ep < 0.3
        assert result.difficulty in ("difficult", "very_difficult")

    def test_negative_coordinates(self):
        result = compute_ep(
            {
                "distance_m": 10.0,
                "start_x": -10.0,
                "end_x": 5.0,
            }
        )
        assert 0.0 <= result.ep <= 1.0
        # start_x clamped to 0, so still valid

    def test_very_difficult_all_negatives(self):
        result = compute_ep(
            {
                "distance_m": 40.0,
                "is_through_ball": True,
                "is_cross": True,
                "is_long_ball": True,
                "is_pressured": True,
                "is_headed": True,
                "receiver_pressured": True,
                "angle_deg": 90.0,
            }
        )
        assert result.ep < 0.40
        assert result.difficulty == "very_difficult"

    def test_progressive_flag_set_correctly(self):
        prog = compute_ep(
            {
                "distance_m": 30.0,
                "start_x": 40.0,
                "end_x": 75.0,
                "attacking_direction": 1,
            }
        )
        assert prog.is_progressive is True

        non_prog = compute_ep(
            {
                "distance_m": 5.0,
                "start_x": 50.0,
                "end_x": 52.0,
                "attacking_direction": 1,
            }
        )
        assert non_prog.is_progressive is False

    def test_factor_contributions_present(self):
        result = compute_ep({"distance_m": 12.0, "is_pressured": True})
        assert "intercept" in result.factors
        assert "distance_m" in result.factors
        assert "distance_m_sq" in result.factors
        assert "is_pressured" in result.factors
        assert "is_through_ball" in result.factors
        assert result.factors["is_pressured"] < 0

    def test_factor_sum_matches_logit(self):
        result = compute_ep({"distance_m": 15.0, "angle_deg": 30.0})
        total = sum(result.factors.values())
        logit = total
        expected_ep = 1.0 / (1.0 + math.exp(-min(logit, 20.0)))
        assert abs(result.ep - expected_ep) < 1e-4

    def test_attacking_direction_negative_progressive(self):
        result = compute_ep(
            {
                "distance_m": 30.0,
                "start_x": 70.0,
                "end_x": 35.0,
                "attacking_direction": -1,
            }
        )
        assert result.is_progressive is True

    def test_progressive_false_for_backward_pass(self):
        result = compute_ep(
            {
                "distance_m": 30.0,
                "start_x": 70.0,
                "end_x": 35.0,
                "attacking_direction": 1,
            }
        )
        assert result.is_progressive is False

    def test_classify_difficulty_from_ep(self):
        moderate = compute_ep({"distance_m": 2.0})
        assert moderate.difficulty == "moderate"

        hard = compute_ep({"distance_m": 50.0, "is_pressured": True})
        assert hard.difficulty == "very_difficult"


# ── Batch computation ─────────────────────────────────────────────────────────


class TestComputeEPBatch:
    def test_empty_batch(self):
        assert compute_ep_batch([]) == []

    def test_multiple_passes(self):
        passes = [
            {"distance_m": 5.0},
            {"distance_m": 25.0},
            {"distance_m": 50.0, "is_pressured": True},
        ]
        results = compute_ep_batch(passes)
        assert len(results) == 3
        assert all(isinstance(r, ExpectedPassResult) for r in results)
        # EP should decrease with distance
        assert results[0].ep > results[1].ep > results[2].ep

    def test_batch_factors(self):
        passes = [
            {"distance_m": 10.0, "is_cross": True},
            {"distance_m": 20.0, "is_through_ball": True},
        ]
        results = compute_ep_batch(passes)
        for r in results:
            assert "intercept" in r.factors
            assert "distance_m" in r.factors
            assert r.factors["distance_m"] < 0
        assert results[0].factors["is_cross"] < 0
        assert results[1].factors["is_through_ball"] < 0

    def test_batch_progressive_flags(self):
        passes = [
            {"distance_m": 5.0, "start_x": 50.0, "end_x": 52.0},
            {"distance_m": 30.0, "start_x": 40.0, "end_x": 75.0},
        ]
        results = compute_ep_batch(passes)
        assert results[0].is_progressive is False
        assert results[1].is_progressive is True

    def test_batch_range(self):
        passes = [{"distance_m": d, "is_pressured": True} for d in [0.0, 10.0, 50.0, 100.0]]
        results = compute_ep_batch(passes)
        for r in results:
            assert 0.0 <= r.ep <= 1.0

    def test_batch_round_trip(self):
        single = compute_ep({"distance_m": 15.0, "is_cross": True})
        batch = compute_ep_batch([{"distance_m": 15.0, "is_cross": True}])
        assert len(batch) == 1
        assert batch[0].ep == single.ep
        assert batch[0].is_progressive == single.is_progressive
        assert batch[0].difficulty == single.difficulty
        assert batch[0].factors == single.factors


# ── Feature vector internal ───────────────────────────────────────────────────


class TestFeatureVector:
    def test_feature_vector_shape(self):
        fv = _feature_vector({})
        assert fv.shape == (11,)
        assert fv.dtype == np.float64

    def test_feature_vector_defaults(self):
        fv = _feature_vector({})
        # intercept = 1.0, distance_m = 15.0, distance_m_sq = 225.0
        assert fv[0] == 1.0  # intercept
        assert fv[1] == 15.0  # distance_m
        assert fv[2] == 225.0  # distance_m_sq
        assert fv[3] == 0.0  # is_through_ball
        assert fv[4] == 0.0  # is_cross
        assert fv[5] == 0.0  # is_long_ball
        assert fv[6] == 0.0  # is_pressured
        assert fv[7] == 0.0  # is_headed
        assert fv[8] == 0.0  # receiver_pressured
        assert fv[9] == 52.5 / 105.0  # start_x_norm
        assert fv[10] == 0.0  # angle_deg

    def test_feature_vector_flags(self):
        fv = _feature_vector(
            {
                "is_through_ball": True,
                "is_cross": True,
                "is_long_ball": True,
                "is_pressured": True,
                "is_headed": True,
                "receiver_pressured": True,
            }
        )
        assert fv[3] == 1.0
        assert fv[4] == 1.0
        assert fv[5] == 1.0
        assert fv[6] == 1.0
        assert fv[7] == 1.0
        assert fv[8] == 1.0

    def test_feature_vector_start_x_clamp(self):
        fv_neg = _feature_vector({"start_x": -100.0})
        assert fv_neg[9] == 0.0

        fv_over = _feature_vector({"start_x": 200.0})
        assert fv_over[9] == 1.0


# ── EP_COEFFICIENTS contract ──────────────────────────────────────────────────


class TestCoefficients:
    def test_all_coefficients_have_matching_feature(self):
        from kawkab.core.expected_pass import _FEATURE_NAMES

        for name in _FEATURE_NAMES:
            assert name in EP_COEFFICIENTS, f"{name} missing from EP_COEFFICIENTS"

    def test_intercept_positive(self):
        assert EP_COEFFICIENTS["intercept"] > 0

    def test_penalty_coefficients_negative(self):
        for key in [
            "is_through_ball",
            "is_cross",
            "is_long_ball",
            "is_pressured",
            "is_headed",
            "receiver_pressured",
            "distance_m",
            "distance_m_sq",
            "angle_deg",
        ]:
            assert EP_COEFFICIENTS[key] < 0, f"{key} should be negative"
