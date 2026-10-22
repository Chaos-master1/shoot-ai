"""Tests for the xG trainer — numpy logistic regression."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from kawkab.core.xg_model import ENHANCED_COEFFICIENTS, EnhancedXgModel
from kawkab.core.xg_trainer import (
    FEATURE_NAMES,
    FitShot,
    _build_feature_matrix,
    _sigmoid,
    batch_gradient_descent,
    fit_from_events,
    fit_from_shots,
    generate_synthetic_training_data,
    load_coefficients,
    save_coefficients,
)


class TestFitShot:
    def test_defaults(self):
        fs = FitShot()
        assert fs.distance_m == 18.0
        assert fs.is_goal is False


class TestBuildFeatureMatrix:
    def test_single_shot(self):
        shots = [FitShot(distance_m=10.0, angle_deg=45.0, is_goal=True)]
        X, y = _build_feature_matrix(shots)
        assert X.shape == (1, len(FEATURE_NAMES))
        assert y[0] == 1.0
        assert X[0, 0] == 1.0  # intercept

    def test_multiple_shots(self):
        shots = [
            FitShot(distance_m=5.0, angle_deg=10.0, is_goal=True),
            FitShot(distance_m=25.0, angle_deg=80.0, is_goal=False),
        ]
        X, y = _build_feature_matrix(shots)
        assert X.shape == (2, len(FEATURE_NAMES))
        assert list(y) == [1.0, 0.0]

    def test_header_flag(self):
        shots = [FitShot(is_header=True, is_goal=False)]
        X, y = _build_feature_matrix(shots)
        assert X[0, 5] == 1.0

    def test_gk_distance(self):
        shots = [FitShot(gk_distance_m=5.0)]
        X, y = _build_feature_matrix(shots)
        assert X[0, 12] == 5.0
        assert X[0, 13] == 25.0

    def test_no_gk_distance(self):
        shots = [FitShot(gk_distance_m=0.0)]
        X, y = _build_feature_matrix(shots)
        assert X[0, 12] == 0.0
        assert X[0, 13] == 0.0

    def test_empty_shots(self):
        X, y = _build_feature_matrix([])
        assert X.shape == (0, len(FEATURE_NAMES))


class TestSigmoid:
    def test_zero(self):
        assert _sigmoid(np.array([0.0]))[0] == 0.5

    def test_large_positive(self):
        assert _sigmoid(np.array([100.0]))[0] > 0.9999

    def test_large_negative(self):
        assert _sigmoid(np.array([-100.0]))[0] < 0.0001


class TestBatchGradientDescent:
    def test_converges_on_synthetic_data(self):
        shots = generate_synthetic_training_data(n_shots=2000, seed=42)
        X, y = _build_feature_matrix(shots)
        theta, losses = batch_gradient_descent(X, y, epochs=4000, lr=0.005, verbose=False)
        assert len(losses) > 0
        preds = _sigmoid(X @ theta)
        acc = float(np.mean((preds > 0.5) == y))
        assert acc > 0.50

    def test_intercept_only_data(self):
        X = np.ones((100, 1), dtype=np.float64)
        y = np.ones(100, dtype=np.float64)
        theta, losses = batch_gradient_descent(X, y, epochs=2000, lr=0.1)
        p = _sigmoid(X @ theta)
        assert float(np.mean(p)) > 0.6


class TestGenerateSyntheticData:
    def test_returns_correct_count(self):
        shots = generate_synthetic_training_data(n_shots=5000, seed=0)
        assert len(shots) == 5000

    def test_all_have_valid_goals(self):
        shots = generate_synthetic_training_data(n_shots=1000, seed=0)
        assert any(s.is_goal for s in shots)
        assert any(not s.is_goal for s in shots)

    def test_consistent_with_seed(self):
        a = generate_synthetic_training_data(n_shots=100, seed=42)
        b = generate_synthetic_training_data(n_shots=100, seed=42)
        for sa, sb in zip(a, b, strict=False):
            assert sa.is_goal == sb.is_goal


class TestFitFromShots:
    def test_returns_coefficients(self):
        shots = generate_synthetic_training_data(n_shots=2000, seed=42)
        coeffs = fit_from_shots(shots)
        assert len(coeffs) >= len(FEATURE_NAMES)
        assert coeffs.get("_n_shots") == 2000
        assert "intercept" in coeffs

    def test_fewer_than_10_returns_defaults(self):
        shots = [FitShot() for _ in range(5)]
        coeffs = fit_from_shots(shots)
        assert coeffs["intercept"] == ENHANCED_COEFFICIENTS["intercept"]

    def test_trained_model_accepts_coefficients(self):
        shots = generate_synthetic_training_data(n_shots=1000, seed=42)
        coeffs = fit_from_shots(shots)
        model = EnhancedXgModel(coefficients=coeffs, coeffs_source="trained")
        assert model.coeffs_source == "trained"
        xg = model.compute_single(
            model.extract_features(
                {
                    "type": "shot",
                    "distance_m": 12.0,
                    "angle_deg": 30.0,
                    "is_goal": False,
                }
            )
        )
        assert 0.0 < xg < 1.0

    def test_goal_rate_reported(self):
        shots = generate_synthetic_training_data(n_shots=5000, seed=0)
        coeffs = fit_from_shots(shots)
        assert coeffs["_goal_rate"] > 0.0


class TestFitFromEvents:
    def test_from_shot_dicts(self):
        events = [
            {
                "type": "shot",
                "distance_m": 5.0,
                "angle_deg": 10.0,
                "is_goal": True,
                "body_part": "right_foot",
                "shot_type": "open_play",
            },
            {
                "type": "shot",
                "distance_m": 25.0,
                "angle_deg": 75.0,
                "is_goal": False,
                "body_part": "left_foot",
                "shot_type": "open_play",
            },
        ]
        coeffs = fit_from_events(events)
        assert "intercept" in coeffs

    def test_non_shot_filtered(self):
        events = [
            {"type": "pass", "timestamp": 10.0},
            {
                "type": "shot",
                "distance_m": 10.0,
                "angle_deg": 20.0,
                "is_goal": False,
                "body_part": "right_foot",
                "shot_type": "open_play",
            },
        ]
        coeffs = fit_from_events(events)
        assert "intercept" in coeffs


class TestSaveLoadCoefficients:
    def test_save_load_roundtrip(self):
        coeffs = {"intercept": -1.5, "distance_m": -0.12, "_model_name": "test"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
            save_coefficients(coeffs, path)
        loaded = load_coefficients(path)
        assert loaded["intercept"] == -1.5
        assert loaded["distance_m"] == -0.12
        assert loaded["_model_name"] == "test"
        Path(path).unlink(missing_ok=True)


class TestEnhancedXgModelLoadTrained:
    def test_load_trained(self):
        coeffs = dict(ENHANCED_COEFFICIENTS)
        coeffs["_model_name"] = "test_model"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(coeffs, f)
            path = f.name
        model = EnhancedXgModel.load_trained(path)
        assert model.coeffs_source == path
        assert abs(model.coef["intercept"] - ENHANCED_COEFFICIENTS["intercept"]) < 0.01
        Path(path).unlink(missing_ok=True)

    def test_trained_model_computes_same_xg(self):
        coeffs = dict(ENHANCED_COEFFICIENTS)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(coeffs, f)
            path = f.name
        trained = EnhancedXgModel.load_trained(path)
        heuristic = EnhancedXgModel()
        ev = {
            "type": "shot",
            "distance_m": 12.0,
            "angle_deg": 30.0,
            "is_goal": False,
            "body_part": "right_foot",
            "shot_type": "open_play",
        }
        assert abs(trained.compute(ev) - heuristic.compute(ev)) < 0.5
        Path(path).unlink(missing_ok=True)
