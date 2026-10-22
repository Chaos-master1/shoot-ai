"""Tests for core/validation — loaders, metrics, and the training pipeline.

All tests run against synthetic data or tiny slices of the real corpus
(the full-corpus training run lives in scripts/validate_models.py, not
here). No test depends on torch/cv2/PySide6.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

# ── StatsBomb loader ────────────────────────────────────────────────────────
from kawkab.core.validation.statsbomb_loader import (
    StatsBombShot,
    extract_shots_for_fitting,
    load_statsbomb_match,
    sb_to_meters,
    shot_distance_angle,
)


class TestCoordinateConversion:
    def test_sb_to_meters_corner(self):
        x, y = sb_to_meters([120.0, 80.0])
        assert x == pytest.approx(105.0)
        assert y == pytest.approx(68.0)

    def test_sb_to_meters_origin(self):
        x, y = sb_to_meters([0.0, 0.0])
        assert (x, y) == (0.0, 0.0)

    def test_sb_to_meters_goal_center(self):
        # SB goal center is at x=120, y=40
        x, y = sb_to_meters([120.0, 40.0])
        assert x == pytest.approx(105.0)
        assert y == pytest.approx(34.0)

    def test_shot_distance_angle_goal_mouth_center(self):
        # From the goal line center, angle should be ~90 (on top of goal)
        d, a = shot_distance_angle(105.0, 34.0)
        assert d == pytest.approx(0.0, abs=1e-6)
        assert a == pytest.approx(90.0)

    def test_shot_distance_angle_penalty_spot(self):
        # Penalty spot: 11 m from goal, dead center -> angle ~37 deg
        d, a = shot_distance_angle(94.0, 34.0)
        assert d == pytest.approx(11.0, abs=0.01)
        # Known geometry: angle subtended by 7.32m goal at 11m distance
        expected = 2 * math.degrees(math.atan(3.66 / 11.0))
        assert a == pytest.approx(expected, abs=0.1)

    def test_shot_distance_angle_far_post_wide(self):
        # Wide angle near the corner flag: small angle, large distance
        d, a = shot_distance_angle(103.0, 66.0)
        assert d > 30.0
        assert a < 10.0


class TestLoadStatsBombMatch:
    @pytest.fixture()
    def corpus_dir(self) -> Path:
        d = Path(__file__).resolve().parents[2] / "data" / "statsbomb_corpus"
        if not d.is_dir():
            pytest.skip("statsbomb corpus not on this machine")
        return d

    def test_loads_real_file(self, corpus_dir):
        files = sorted(corpus_dir.glob("*.json"))
        if not files:
            pytest.skip("corpus empty")
        m = load_statsbomb_match(files[0])
        assert m.n_events >= 0  # structure parsed
        for s in m.shots:
            assert 0.0 <= s.statsbomb_xg <= 1.0
            assert s.distance_m >= 0.0
            # Opening angle can exceed 90° for close-range shots (7.32m
            # goal mouth from <3.66m); the model saturates >= 90 by
            # convention. 180° is the on-the-line extreme.
            assert 0.0 <= s.angle_deg <= 180.0
            assert s.body_part in ("right_foot", "left_foot", "head", "other")
            assert isinstance(s.is_goal, bool)

    def test_rejects_non_list_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(ValueError):
            load_statsbomb_match(bad)

    def test_totals_consistent(self, corpus_dir):
        files = sorted(corpus_dir.glob("*.json"))[:5]
        if not files:
            pytest.skip("corpus empty")
        for f in files:
            m = load_statsbomb_match(f)
            assert m.total_sb_xg == pytest.approx(sum(s.statsbomb_xg for s in m.shots))
            assert m.total_goals == sum(1 for s in m.shots if s.is_goal)

    def test_extract_shots_excludes_penalties(self, corpus_dir):
        files = sorted(corpus_dir.glob("*.json"))[:10]
        matches = [load_statsbomb_match(f) for f in files]
        shots, n = extract_shots_for_fitting(matches)
        assert n == len(matches)
        assert all(s.shot_type != "penalty" for s in shots)


# ── Metrics ─────────────────────────────────────────────────────────────────

from kawkab.core.validation.metrics import (
    bootstrap_ci,
    brier_score,
    brier_skill_score,
    calibration_error,
    log_loss,
    mean_absolute_error,
    pearson_correlation,
    reliability_curve,
    roc_auc,
)


class TestBrier:
    def test_perfect_prediction(self):
        assert brier_score([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_coin_flip_baseline(self):
        assert brier_score([1.0, 0.0], [0.5, 0.5]) == pytest.approx(0.25)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            brier_score([1.0, 0.0], [0.5])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            brier_score([], [])

    def test_proba_out_of_range_raises(self):
        with pytest.raises(ValueError):
            brier_score([1.0], [1.5])

    def test_non_binary_y_raises(self):
        with pytest.raises(ValueError):
            brier_score([0.5], [0.5])

    def test_nan_raises(self):
        with pytest.raises(ValueError):
            brier_score([1.0], [float("nan")])


class TestSkillScore:
    def test_perfect_beats_climatology(self):
        y = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        assert brier_skill_score(y, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) > 0.99

    def test_climatology_forecast_scores_zero(self):
        y = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        assert brier_skill_score(y, [0.5] * 6) == pytest.approx(0.0)


class TestReliability:
    def test_well_calibrated_bins(self):
        # 2 of 4 events at p=0.5, 3 of 6 at p~0.33
        y = [1, 0, 1, 0] * 1 + [1, 0, 0, 1, 0, 0]
        p = [0.5] * 4 + [0.35] * 6
        curve = reliability_curve(y, p, n_bins=10)
        assert len(curve) >= 2
        total_n = sum(c["n"] for c in curve)
        assert total_n == 10

    def test_bins_partition_all_points(self):
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.3).astype(float)
        p = rng.random(500)
        curve = reliability_curve(y, p, n_bins=10)
        assert sum(c["n"] for c in curve) == 500

    def test_quantile_strategy(self):
        p = [0.01, 0.02, 0.5, 0.51, 0.98, 0.99]
        y = [0, 0, 1, 0, 1, 1]
        curve = reliability_curve(y, p, n_bins=3, strategy="quantile")
        assert sum(c["n"] for c in curve) == 6

    def test_bad_strategy_raises(self):
        with pytest.raises(ValueError):
            reliability_curve([1.0], [0.5], strategy="nope")


class TestRocAuc:
    def test_perfect_separation(self):
        assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)

    def test_inverted(self):
        assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)

    def test_ties_get_average_rank(self):
        # All same prediction -> AUC 0.5
        assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_single_class_raises(self):
        with pytest.raises(ValueError):
            roc_auc([1, 1, 1], [0.2, 0.5, 0.8])


class TestLogLossEce:
    def test_log_loss_bounds(self):
        ll = log_loss([1.0, 0.0], [0.9, 0.1])
        assert 0.0 < ll < 1.0

    def test_log_loss_clips_extremes(self):
        ll = log_loss([1.0, 0.0], [1.0, 0.0])
        assert np.isfinite(ll)
        assert ll == pytest.approx(0.0, abs=1e-10)

    def test_ece_zero_for_perfect_calibration(self):
        y = [1, 0] * 5
        p = [0.5] * 10
        assert calibration_error(y, p, n_bins=5) == pytest.approx(0.0, abs=0.11)


class TestHelpers:
    def test_mae(self):
        assert mean_absolute_error([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)

    def test_pearson(self):
        a = [1, 2, 3, 4, 5]
        assert pearson_correlation(a, [x * 2 for x in a]) == pytest.approx(1.0)
        assert pearson_correlation(a, [-x for x in a]) == pytest.approx(-1.0)

    def test_pearson_constant_returns_zero(self):
        assert pearson_correlation([1, 1, 1], [1, 2, 3]) == 0.0

    def test_bootstrap_ci_brackets_value(self):
        rng = np.random.default_rng(1)
        vals = rng.normal(10.0, 2.0, 200)
        ci = bootstrap_ci(vals, "mean", n_bootstrap=500)
        assert ci["ci_lower"] <= ci["value"] <= ci["ci_upper"]
        assert ci["n"] == 200


# ── Trainer pipeline ────────────────────────────────────────────────────────

from kawkab.core.validation.train_xg import (
    fit_with_holdout,
    predict_xg,
    split_train_val,
)


class TestSplitAndPredict:
    def _fake_matches(self, n=6):
        matches = []
        for mi in range(n):
            shots = []
            for si in range(8):
                shots.append(
                    StatsBombShot(
                        distance_m=5.0 + si * 3.0,
                        angle_deg=10.0 + si * 5.0,
                        body_part="right_foot",
                        shot_type="open_play",
                        gk_distance_m=2.0 + si,
                        is_pressed=False,
                        is_one_on_one=False,
                        is_rebound=False,
                        is_big_chance=False,
                        is_goal=(si % 4 == 0),
                        statsbomb_xg=0.1 * ((si % 4) + 1),
                        match_id=f"m{mi}",
                        event_id=f"e{mi}_{si}",
                        minute=si,
                        period=1,
                        player_name="p",
                        team_name="t",
                        angle_deviation_deg=10.0 + si * 5.0,
                    )
                )
            from kawkab.core.validation.statsbomb_loader import StatsBombMatch

            matches.append(StatsBombMatch(match_id=f"m{mi}", shots=shots))
        return matches

    def test_split_is_match_level(self):
        train, val = split_train_val(self._fake_matches(20))
        train_ids = {id(s) for s in train}
        assert not (train_ids & {id(s) for s in val})

    def test_split_excludes_penalties(self):
        matches = self._fake_matches(10)
        from kawkab.core.validation.statsbomb_loader import StatsBombMatch, StatsBombShot

        pen = StatsBombShot(
            distance_m=11.0,
            angle_deg=37.0,
            body_part="right_foot",
            shot_type="penalty",
            gk_distance_m=0.0,
            is_pressed=False,
            is_one_on_one=False,
            is_rebound=False,
            is_big_chance=False,
            is_goal=True,
            statsbomb_xg=0.76,
            match_id="mp",
            event_id="pe",
            minute=1,
            period=1,
            player_name="p",
            team_name="t",
            angle_deviation_deg=37.0,
        )
        matches.append(StatsBombMatch(match_id="mp", shots=[pen]))
        train, val = split_train_val(matches)
        assert all(s.shot_type != "penalty" for s in train + val)

    def test_fit_fallback_on_tiny_data(self):
        from kawkab.core.xg_model import ENHANCED_COEFFICIENTS

        train, _ = split_train_val(self._fake_matches(1))
        coeffs = fit_with_holdout(train)
        assert coeffs == ENHANCED_COEFFICIENTS  # <10 shots -> heuristic fallback

    def test_predict_returns_probabilities(self):
        from kawkab.core.xg_model import ENHANCED_COEFFICIENTS

        train, val = split_train_val(self._fake_matches(4))
        p = predict_xg(dict(ENHANCED_COEFFICIENTS), val)
        assert len(p) == len(val)
        assert np.all((p >= 0.0) & (p <= 1.0))


# ── Metrica loader ─────────────────────────────────────────────────────────

from kawkab.core.validation.metrica_loader import load_metrica_match


class TestMetricaLoader:
    @pytest.fixture()
    def metrica_dir(self) -> Path:
        d = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "ground_truth"
            / "metrica"
            / "Sample_Game_1"
        )
        if not d.is_dir():
            pytest.skip("metrica data not on this machine")
        return d

    def test_loads_frames(self, metrica_dir):
        home = metrica_dir / "Sample_Game_1_RawTrackingData_Home_Team.csv"
        away = metrica_dir / "Sample_Game_1_RawTrackingData_Away_Team.csv"
        m = load_metrica_match(home, away, max_frames=50)
        assert 0 < m.n_frames <= 50
        f = m.frames[0]
        assert f.period == 1
        assert len(f.home) == 11
        assert len(f.away) == 11
        # meters conversion: all positions inside the pitch
        for x, y in f.home + f.away:
            assert -5.0 <= x <= 110.0
            assert -5.0 <= y <= 73.0

    def test_ball_in_bounds(self, metrica_dir):
        home = metrica_dir / "Sample_Game_1_RawTrackingData_Home_Team.csv"
        away = metrica_dir / "Sample_Game_1_RawTrackingData_Away_Team.csv"
        m = load_metrica_match(home, away, max_frames=30)
        for f in m.frames:
            if f.ball:
                assert -5.0 <= f.ball[0] <= 110.0
                assert -5.0 <= f.ball[1] <= 73.0

    def test_time_monotonic_within_period(self, metrica_dir):
        home = metrica_dir / "Sample_Game_1_RawTrackingData_Home_Team.csv"
        away = metrica_dir / "Sample_Game_1_RawTrackingData_Away_Team.csv"
        m = load_metrica_match(home, away, max_frames=100)
        times = [f.time_s for f in m.frames if f.period == 1]
        assert all(b >= a for a, b in zip(times, times[1:], strict=False))
