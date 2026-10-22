"""Tests for deep-learning xG model — neural network + heuristic fallback."""

import numpy as np

from kawkab.core.dl_xg_model import (
    FI_ANGLE,
    FI_ANGLE_SIN,
    FI_IS_HEADER,
    N_FEATURES,
    DenseLayer,
    DLXgModel,
    predict_dl_xg,
)


class TestDenseLayer:
    def test_forward_output_shape(self):
        layer = DenseLayer(12, 16, seed=42)
        x = np.ones((5, 12), dtype=np.float64)
        out = layer.forward(x)
        assert out.shape == (5, 16)

    def test_relu_activation(self):
        layer = DenseLayer(12, 16, seed=42)
        x = np.ones((3, 12), dtype=np.float64)
        out = layer.forward(x)
        assert np.all(out >= 0.0)

    def test_zero_input_no_nan(self):
        layer = DenseLayer(12, 16, seed=42)
        x = np.zeros((2, 12), dtype=np.float64)
        out = layer.forward(x)
        assert not np.any(np.isnan(out))


class TestDLXgModel:
    def test_forward_output_shape(self):
        model = DLXgModel(seed=42)
        x = np.ones((5, N_FEATURES), dtype=np.float64)
        preds = model.forward(x)
        assert preds.shape == (5,)

    def test_forward_output_range(self):
        model = DLXgModel(seed=42)
        x = np.random.randn(20, N_FEATURES).astype(np.float64)
        preds = model.forward(x)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

    def test_predict_returns_same_as_forward(self):
        model = DLXgModel(seed=42)
        x = np.random.randn(3, N_FEATURES).astype(np.float64)
        np.testing.assert_array_equal(model.predict(x), model.forward(x))

    def test_extract_features_empty(self):
        model = DLXgModel()
        features = model.extract_features([])
        assert features.shape == (0, N_FEATURES)

    def test_extract_features_shape(self):
        model = DLXgModel()
        shots = [
            {"distance_m": 12.0, "angle_deg": 30.0},
            {"distance_m": 5.0, "angle_deg": 10.0, "is_header": True},
        ]
        features = model.extract_features(shots)
        assert features.shape == (2, N_FEATURES)

    def test_extract_features_header_flag(self):
        model = DLXgModel()
        shots = [
            {"distance_m": 12.0, "angle_deg": 30.0},
            {"distance_m": 12.0, "angle_deg": 30.0, "is_header": True, "body_part": "head"},
        ]
        features = model.extract_features(shots)
        assert features[0, FI_IS_HEADER] == 0.0
        assert features[1, FI_IS_HEADER] == 1.0

    def test_compute_single_returns_float(self):
        model = DLXgModel()
        event = {"distance_m": 15.0, "angle_deg": 25.0}
        xg = model.compute_single(event)
        assert isinstance(xg, float)
        assert 0.0 <= xg <= 1.0

    def test_train_reduces_loss(self):
        model = DLXgModel(seed=42)
        x = np.random.randn(50, N_FEATURES).astype(np.float64)
        y = (np.random.rand(50) > 0.5).astype(np.float64)
        history = model.train(x, y, epochs=10, batch_size=16, lr=0.01)
        loss_start = history["loss"][0]
        loss_end = history["loss"][-1]
        assert loss_end <= loss_start + 0.01

    def test_train_with_validation(self):
        model = DLXgModel(seed=42)
        x = np.random.randn(50, N_FEATURES).astype(np.float64)
        y = (np.random.rand(50) > 0.5).astype(np.float64)
        history = model.train(x, y, epochs=5, batch_size=16, validation_split=0.2)
        assert "loss" in history
        assert "val_loss" in history
        assert len(history["val_loss"]) > 0

    def test_save_load_roundtrip(self, tmp_path):
        model = DLXgModel(seed=42)
        path = str(tmp_path / "xg_weights.json")
        model.save(path)
        model2 = DLXgModel(seed=99)
        model2.load(path)
        x = np.random.randn(5, N_FEATURES).astype(np.float64)
        np.testing.assert_array_almost_equal(model.predict(x), model2.predict(x), decimal=5)

    def test_generate_example_data_shapes(self):
        model = DLXgModel()
        features, labels = model.generate_example_data(n_samples=100)
        assert features.shape == (100, N_FEATURES)
        assert labels.shape == (100,)

    def test_generate_calibrated_data_shapes(self):
        model = DLXgModel()
        features, labels = model.generate_calibrated_data(n_samples=200)
        assert features.shape == (200, N_FEATURES)
        assert labels.shape == (200,)

    def test_close_shot_higher_than_far_shot_after_training(self):
        model = DLXgModel(seed=42)
        feats, labels = model.generate_example_data(n_samples=200)
        model.train(feats, labels, epochs=50, batch_size=32, lr=0.005)
        close = model.compute_single({"distance_m": 2.0, "angle_deg": 10.0})
        far = model.compute_single({"distance_m": 35.0, "angle_deg": 30.0})
        assert close > far

    def test_penalty_high_value(self):
        xg = predict_dl_xg(distance_m=11.0, angle_deg=0.0, shot_type="penalty")
        assert xg > 0.50

    def test_header_lower_than_foot_after_training(self):
        model = DLXgModel(seed=42)
        feats, labels = model.generate_example_data(n_samples=200)
        model.train(feats, labels, epochs=50, batch_size=32, lr=0.005)
        foot = model.compute_single({"distance_m": 10.0, "angle_deg": 15.0, "is_header": False})
        header = model.compute_single({"distance_m": 10.0, "angle_deg": 15.0, "is_header": True})
        assert header < foot

    def test_one_on_one_higher_than_normal(self):
        normal = predict_dl_xg(distance_m=10.0, angle_deg=15.0)
        one_on_one = predict_dl_xg(distance_m=10.0, angle_deg=15.0, is_one_on_one=True)
        assert one_on_one > normal

    def test_pressed_lower_than_unpressed_after_training(self):
        model = DLXgModel(seed=42)
        feats, labels = model.generate_example_data(n_samples=200)
        model.train(feats, labels, epochs=50, batch_size=32, lr=0.005)
        unpressed = model.compute_single(
            {"distance_m": 12.0, "angle_deg": 20.0, "was_pressed": False}
        )
        pressed = model.compute_single({"distance_m": 12.0, "angle_deg": 20.0, "was_pressed": True})
        assert pressed < unpressed

    def test_big_chance_higher(self):
        normal = predict_dl_xg(distance_m=8.0, angle_deg=10.0)
        big = predict_dl_xg(distance_m=8.0, angle_deg=10.0, is_big_chance=True)
        assert big > normal

    def test_rebound_higher(self):
        normal = predict_dl_xg(distance_m=8.0, angle_deg=10.0)
        rebound = predict_dl_xg(distance_m=8.0, angle_deg=10.0, is_rebound=True)
        assert rebound > normal

    def test_zero_distance(self):
        xg = predict_dl_xg(distance_m=0.5, angle_deg=0.0)
        assert 0.0 <= xg <= 1.0

    def test_distance_monotonic_after_training(self):
        model = DLXgModel(seed=42)
        feats, labels = model.generate_example_data(n_samples=200)
        model.train(feats, labels, epochs=50, batch_size=32, lr=0.005)
        xgs = [model.compute_single({"distance_m": d, "angle_deg": 15.0}) for d in [5, 10, 20, 30]]
        for i in range(len(xgs) - 1):
            assert xgs[i] >= xgs[i + 1] - 1e-6, f"xG not monotonic at index {i}"

    def test_default_model_gives_reasonable_values_without_training(self):
        xg = predict_dl_xg(distance_m=11.0, angle_deg=0.0, shot_type="penalty")
        assert 0.0 <= xg <= 1.0

    def test_angle_feature_extracted_correctly(self):
        model = DLXgModel()
        shots = [
            {"distance_m": 15.0, "angle_deg": 5.0},
            {"distance_m": 15.0, "angle_deg": 40.0},
        ]
        feats = model.extract_features(shots)
        # Central shot has lower angle_sin (closer to goal-facing)
        assert feats[0, FI_ANGLE_SIN] < feats[1, FI_ANGLE_SIN]
        # Central shot has lower raw angle
        assert feats[0, FI_ANGLE] < feats[1, FI_ANGLE]

    def test_gk_distance_affects_xg(self):
        # Heuristic weight is positive: farther GK → higher xG
        close_gk = predict_dl_xg(distance_m=15.0, angle_deg=20.0, gk_distance_m=1.0)
        far_gk = predict_dl_xg(distance_m=15.0, angle_deg=20.0, gk_distance_m=6.0)
        assert close_gk != far_gk
        assert 0.0 <= close_gk <= 1.0
        assert 0.0 <= far_gk <= 1.0

    def test_extreme_values_no_error(self):
        xg = predict_dl_xg(distance_m=100.0, angle_deg=90.0, is_header=True)
        assert 0.0 <= xg <= 1.0
        xg2 = predict_dl_xg(distance_m=0.1, angle_deg=0.0)
        assert 0.0 <= xg2 <= 1.0
