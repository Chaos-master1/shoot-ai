"""Tests for Finishing Analysis — shot quality tiers, streaks, placement."""

from kawkab.core.finishing_analysis import (
    DEFAULT_TIER_THRESHOLDS,
    _compute_placement_skill,
    _detect_streaks,
    _tier_for_xg,
    analyze_finishing,
)


class TestTierForXG:
    def test_big_chance(self):
        assert _tier_for_xg(0.5, DEFAULT_TIER_THRESHOLDS) == "big_chance"

    def test_big_chance_boundary(self):
        assert _tier_for_xg(0.35, DEFAULT_TIER_THRESHOLDS) == "big_chance"

    def test_half_chance(self):
        assert _tier_for_xg(0.20, DEFAULT_TIER_THRESHOLDS) == "half_chance"

    def test_half_chance_low_boundary(self):
        assert _tier_for_xg(0.10, DEFAULT_TIER_THRESHOLDS) == "half_chance"

    def test_half_chance_high_boundary(self):
        assert _tier_for_xg(0.349, DEFAULT_TIER_THRESHOLDS) == "half_chance"

    def test_low_chance(self):
        assert _tier_for_xg(0.05, DEFAULT_TIER_THRESHOLDS) == "low_chance"

    def test_zero_xg(self):
        assert _tier_for_xg(0.0, DEFAULT_TIER_THRESHOLDS) == "low_chance"


class TestDetectStreaks:
    def test_hot_streak_detected(self):
        shots = [{"xG": 0.1, "goal": True}] * 3 + [{"xG": 0.1, "goal": True}] * 2
        result = _detect_streaks(shots, 5)
        assert result["hot_streak"] is True

    def test_cold_streak_detected(self):
        shots = [{"xG": 0.25, "goal": False}] * 5
        result = _detect_streaks(shots, 5)
        assert result["cold_streak"] is True

    def test_no_streak_empty(self):
        result = _detect_streaks([], 5)
        assert result["hot_streak"] is False
        assert result["cold_streak"] is False

    def test_cold_streak_not_enough_xg(self):
        shots = [{"xG": 0.1, "goal": False}] * 5
        result = _detect_streaks(shots, 5)
        assert result["cold_streak"] is False

    def test_hot_streak_recent_only(self):
        shots = [{"xG": 0.1, "goal": False}] * 10 + [{"xG": 0.5, "goal": True}] * 3
        result = _detect_streaks(shots, 5)
        assert result["hot_streak"] is True


class TestPlacementSkill:
    def test_all_corner_placements(self):
        shots = [{"placement_x": 3.66, "placement_y": 2.44, "xG": 0.3, "goal": True}] * 3
        skill = _compute_placement_skill(shots)
        assert -1.0 <= skill <= 1.0

    def test_all_center_placements(self):
        shots = [{"placement_x": 0.0, "placement_y": 0.0, "xG": 0.3, "goal": False}] * 3
        skill = _compute_placement_skill(shots)
        assert skill <= 0.0

    def test_no_placement_data(self):
        shots = [{"xG": 0.3, "goal": True}] * 3
        assert _compute_placement_skill(shots) == 0.0

    def test_mixed_placements(self):
        shots = [
            {"placement_x": 3.5, "placement_y": 2.0, "xG": 0.4, "goal": True},
            {"placement_x": 1.0, "placement_y": 1.0, "xG": 0.2, "goal": False},
        ]
        skill = _compute_placement_skill(shots)
        assert -1.0 <= skill <= 1.0


class TestAnalyzeFinishing:
    def test_empty_shots(self):
        r = analyze_finishing("p1", [])
        assert r.player_id == "p1"
        assert r.total_goals == 0
        assert r.total_xg == 0.0
        assert r.conversion_rate == 0.0
        assert r.xg_per_shot == 0.0

    def test_all_goals(self):
        shots = [{"xG": 0.5, "goal": True, "distance": 5, "angle": 0}] * 4
        r = analyze_finishing("p1", shots)
        assert r.total_goals == 4
        assert r.total_xg == 2.0
        assert r.finishing_delta == 2.0

    def test_no_goals(self):
        shots = [{"xG": 0.5, "goal": False, "distance": 15, "angle": 30}] * 3
        r = analyze_finishing("p1", shots)
        assert r.total_goals == 0
        assert r.finishing_delta == -1.5

    def test_tier_distribution(self):
        shots = [
            {"xG": 0.5, "goal": True, "distance": 5, "angle": 0},
            {"xG": 0.2, "goal": False, "distance": 12, "angle": 20},
            {"xG": 0.05, "goal": False, "distance": 25, "angle": 40},
        ]
        r = analyze_finishing("p1", shots)
        assert r.shot_tiers["big_chance"]["shots"] == 1
        assert r.shot_tiers["half_chance"]["shots"] == 1
        assert r.shot_tiers["low_chance"]["shots"] == 1

    def test_conversion_rate(self):
        shots = [{"xG": 0.3, "goal": True, "distance": 10, "angle": 15}] * 2 + [
            {"xG": 0.3, "goal": False, "distance": 10, "angle": 15}
        ] * 2
        r = analyze_finishing("p1", shots)
        assert r.conversion_rate == 0.5

    def test_custom_tier_thresholds(self):
        custom = {"easy": (0.5, None), "hard": (0.0, 0.5)}
        shots = [{"xG": 0.6, "goal": True, "distance": 5, "angle": 0}]
        r = analyze_finishing("p1", shots, tier_thresholds=custom)
        assert r.shot_tiers["easy"]["shots"] == 1
        assert r.shot_tiers["hard"]["shots"] == 0

    def test_streak_data_in_report(self):
        shots = [{"xG": 0.1, "goal": True}] * 5
        r = analyze_finishing("p1", shots)
        assert "hot_streak" in r.streak_data
        assert "cold_streak" in r.streak_data

    def test_placement_skill_in_report(self):
        shots = [{"xG": 0.3, "goal": True, "placement_x": 3.0, "placement_y": 2.0}]
        r = analyze_finishing("p1", shots)
        assert -1.0 <= r.placement_skill <= 1.0

    def test_xg_per_shot(self):
        shots = [{"xG": 0.5, "goal": False}, {"xG": 0.3, "goal": True}]
        r = analyze_finishing("p1", shots)
        assert r.xg_per_shot == 0.4

    def test_single_shot(self):
        shots = [{"xG": 0.8, "goal": True, "distance": 3, "angle": 5}]
        r = analyze_finishing("p1", shots)
        assert r.total_goals == 1
        assert r.shot_tiers["big_chance"]["shots"] == 1

    def test_streak_window_parameter(self):
        shots = [{"xG": 0.2, "goal": True}] * 7
        r = analyze_finishing("p1", shots, streak_window=3)
        assert r.streak_data["window"] == 3
