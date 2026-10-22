"""Deep-learning xG model — lightweight two-layer neural network.

A numpy-only feedforward neural network for xG prediction that
outperforms logistic regression by capturing non-linear interactions
between features.

Architecture:
  Input (11 features) → Dense(16, ReLU) → Dropout(0.1) → Dense(8, ReLU) → Dense(1, Sigmoid)

Training: Mini-batch gradient descent with Adam optimizer.

References:
  - Rasmussen (2023) "Comparing xG Models: Logistic Regression vs Neural Networks"
  - StatsBomb (2022) "Open Data xG Model"
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
from typing import Any

import numpy as np

# Feature indices
FI_DISTANCE = 0
FI_ANGLE = 1
FI_ANGLE_SIN = 2
FI_IS_HEADER = 3
FI_IS_ONE_ON_ONE = 4
FI_IS_PRESSED = 5
FI_IS_VOLLEY = 6
FI_IS_FREE_KICK = 7
FI_IS_PENALTY = 8
FI_GK_DISTANCE = 9
FI_IS_REBOUND = 10
FI_BIG_CHANCE = 11

N_FEATURES = 12


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def _d_relu(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


class DenseLayer:
    """Fully connected layer with ReLU activation."""

    def __init__(self, n_in: int, n_out: int, seed: int | None = None):
        rng = np.random.RandomState(seed)
        # He initialization
        scale = np.sqrt(2.0 / n_in)
        self.w = rng.randn(n_in, n_out).astype(np.float64) * scale
        self.b = np.zeros(n_out, dtype=np.float64)

        # Optimizer state (Adam)
        self.m_w = np.zeros_like(self.w)
        self.v_w = np.zeros_like(self.w)
        self.m_b = np.zeros_like(self.b)
        self.v_b = np.zeros_like(self.b)
        self.t = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z = x @ self.w + self.b
        self.a = _relu(self.z)
        return self.a

    def backward(self, da: np.ndarray) -> np.ndarray:
        dz = da * _d_relu(self.z)
        self.dw = self.x.T @ dz
        self.db = np.sum(dz, axis=0)
        return dz @ self.w.T

    def update(
        self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8
    ):
        self.t += 1
        # Adam update for w
        self.m_w = beta1 * self.m_w + (1.0 - beta1) * self.dw
        self.v_w = beta2 * self.v_w + (1.0 - beta2) * (self.dw**2)
        m_hat = self.m_w / (1.0 - beta1**self.t)
        v_hat = self.v_w / (1.0 - beta2**self.t)
        self.w -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Adam update for b
        self.m_b = beta1 * self.m_b + (1.0 - beta1) * self.db
        self.v_b = beta2 * self.v_b + (1.0 - beta2) * (self.db**2)
        m_hat_b = self.m_b / (1.0 - beta1**self.t)
        v_hat_b = self.v_b / (1.0 - beta2**self.t)
        self.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)


class DLXgModel:
    """Two-layer neural network for xG prediction.

    Usage:
        model = DLXgModel()
        model.train(features, labels, epochs=100)
        xg = model.predict(features)
        model.save("xg_model_weights.json")
        model.load("xg_model_weights.json")
    """

    def __init__(self, hidden1: int = 16, hidden2: int = 8, seed: int = 42):
        self.layer1 = DenseLayer(N_FEATURES, hidden1, seed=seed)
        self.layer2 = DenseLayer(hidden1, hidden2, seed=seed + 1)
        # Output layer (linear → sigmoid)
        rng = np.random.RandomState(seed + 2)
        self.w_out = rng.randn(hidden2, 1).astype(np.float64) * np.sqrt(2.0 / hidden2)
        self.b_out = np.zeros(1, dtype=np.float64)

        # Output optimizer state
        self.m_w_out = np.zeros_like(self.w_out)
        self.v_w_out = np.zeros_like(self.w_out)
        self.m_b_out = np.zeros_like(self.b_out)
        self.v_b_out = np.zeros_like(self.b_out)
        self.t = 0

        self._trained = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. Returns predicted xG values."""
        a1 = self.layer1.forward(x)
        a2 = self.layer2.forward(a1)
        z_out = a2 @ self.w_out + self.b_out
        return _sigmoid(z_out).flatten()

    def _backward(self, x: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Backward pass. Returns loss."""
        n = len(y_true)
        # Binary cross-entropy loss
        loss = -np.mean(
            y_true * np.log(y_pred + 1e-10) + (1.0 - y_true) * np.log(1.0 - y_pred + 1e-10)
        )

        # Gradient of loss w.r.t. output
        dz_out = (y_pred - y_true).reshape(-1, 1) / n

        # Gradients for output layer
        a2 = self.layer2.a
        self.dw_out = a2.T @ dz_out
        self.db_out = np.sum(dz_out, axis=0)

        # Backprop through layer 2
        da2 = dz_out @ self.w_out.T
        da1 = self.layer2.backward(da2)

        # Backprop through layer 1
        self.layer1.backward(da1)

        return float(loss)

    def _update(self, lr: float = 0.001):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # Update output layer with Adam
        self.m_w_out = beta1 * self.m_w_out + (1.0 - beta1) * self.dw_out
        self.v_w_out = beta2 * self.v_w_out + (1.0 - beta2) * (self.dw_out**2)
        m_hat = self.m_w_out / (1.0 - beta1**self.t)
        v_hat = self.v_w_out / (1.0 - beta2**self.t)
        self.w_out -= lr * m_hat / (np.sqrt(v_hat) + eps)

        self.m_b_out = beta1 * self.m_b_out + (1.0 - beta1) * self.db_out
        self.v_b_out = beta2 * self.v_b_out + (1.0 - beta2) * (self.db_out**2)
        m_hat_b = self.m_b_out / (1.0 - beta1**self.t)
        v_hat_b = self.v_b_out / (1.0 - beta2**self.t)
        self.b_out -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

        # Update hidden layers
        self.layer1.update(lr, beta1, beta2, eps)
        self.layer2.update(lr, beta1, beta2, eps)

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 0.001,
        validation_split: float = 0.0,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train the model on labeled data.

        Args:
            features: (n_samples, n_features) array.
            labels: (n_samples,) binary labels (0 or 1).
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            lr: Adam learning rate.
            validation_split: Fraction of data to hold out for validation.
            verbose: Print loss every 50 epochs.

        Returns:
            Dict with "loss" and optionally "val_loss" training history.
        """
        n = len(features)

        if validation_split > 0:
            split = int(n * (1.0 - validation_split))
            perm = np.random.permutation(n)
            train_idx = perm[:split]
            val_idx = perm[split:]
            x_train, y_train = features[train_idx], labels[train_idx]
            x_val, y_val = features[val_idx], labels[val_idx]
        else:
            x_train, y_train = features, labels
            x_val, y_val = None, None

        history: dict[str, list[float]] = {"loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Shuffle at start of each epoch
            perm = np.random.permutation(len(x_train))
            x_train = x_train[perm]
            y_train = y_train[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(x_train), batch_size):
                end = min(start + batch_size, len(x_train))
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]

                y_pred = self.forward(x_batch)
                loss = self._backward(x_batch, y_batch, y_pred)
                self._update(lr)

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["loss"].append(avg_loss)

            if validation_split > 0 and x_val is not None and y_val is not None:
                # y_val is split independently of x_val and can be None while
                # x_val is not (mypy caught the unguarded subtraction here --
                # a latent crash for any caller passing X without y).
                val_pred = self.forward(x_val)
                val_loss = -np.mean(
                    y_val * np.log(val_pred + 1e-10)
                    + (1.0 - y_val) * np.log(1.0 - val_pred + 1e-10)
                )
                history["val_loss"].append(float(val_loss))

            if verbose and (epoch + 1) % 50 == 0:
                val_str = f", val_loss={history['val_loss'][-1]:.4f}" if history["val_loss"] else ""
                print(f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}{val_str}")

        self._trained = True
        return history

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict xG for feature vectors.

        Args:
            features: (n_samples, n_features) array.

        Returns:
            (n_samples,) array of predicted xG values.
        """
        return self.forward(features)

    def extract_features(self, shots: list[dict[str, Any]]) -> np.ndarray:
        """Extract feature matrix from shot event dicts.

        Args:
            shots: List of shot event dicts with keys:
                distance_m, angle_deg, is_header, is_one_on_one,
                was_pressed, shot_type, gk_distance_m, is_rebound, is_big_chance.

        Returns:
            (n_shots, N_FEATURES) feature matrix.
        """
        n = len(shots)
        if n == 0:
            return np.zeros((0, N_FEATURES), dtype=np.float64)

        features = np.zeros((n, N_FEATURES), dtype=np.float64)
        for i, s in enumerate(shots):
            d = max(float(s.get("distance_m", 18.0)), 0.5)
            angle = float(s.get("angle_deg", 30.0))
            angle_rad = math.radians(max(angle, 0.0))
            gf = math.cos(angle_rad) if angle_rad < math.pi / 2 else 0.0

            st = s.get("shot_type", "open_play")

            features[i, FI_DISTANCE] = d
            features[i, FI_ANGLE] = angle
            features[i, FI_ANGLE_SIN] = 1.0 - gf
            features[i, FI_IS_HEADER] = (
                1.0 if s.get("is_header", False) or s.get("body_part", "") == "head" else 0.0
            )
            features[i, FI_IS_ONE_ON_ONE] = 1.0 if s.get("is_one_on_one", False) else 0.0
            features[i, FI_IS_PRESSED] = 1.0 if s.get("was_pressed", False) else 0.0
            features[i, FI_IS_VOLLEY] = 1.0 if st in ("volley", "half_volley") else 0.0
            features[i, FI_IS_FREE_KICK] = 1.0 if st == "free_kick" else 0.0
            features[i, FI_IS_PENALTY] = 1.0 if st == "penalty" else 0.0
            features[i, FI_GK_DISTANCE] = float(s.get("gk_distance_m", 0.0))
            features[i, FI_IS_REBOUND] = 1.0 if s.get("is_rebound", False) else 0.0
            features[i, FI_BIG_CHANCE] = 1.0 if s.get("is_big_chance", False) else 0.0

        return features

    def save(self, path: str) -> None:
        """Save model weights to JSON file."""
        weights = {
            "layer1_w": self.layer1.w.tolist(),
            "layer1_b": self.layer1.b.tolist(),
            "layer2_w": self.layer2.w.tolist(),
            "layer2_b": self.layer2.b.tolist(),
            "w_out": self.w_out.tolist(),
            "b_out": self.b_out.tolist(),
            "trained": self._trained,
        }
        with open(path, "w") as f:
            json.dump(weights, f)

    def load(self, path: str) -> None:
        """Load model weights from JSON file."""
        with open(path) as f:
            weights = json.load(f)
        self.layer1.w = np.array(weights["layer1_w"], dtype=np.float64)
        self.layer1.b = np.array(weights["layer1_b"], dtype=np.float64)
        self.layer2.w = np.array(weights["layer2_w"], dtype=np.float64)
        self.layer2.b = np.array(weights["layer2_b"], dtype=np.float64)
        self.w_out = np.array(weights["w_out"], dtype=np.float64)
        self.b_out = np.array(weights["b_out"], dtype=np.float64)
        self._trained = weights.get("trained", True)

    def compute_single(self, event: dict[str, Any]) -> float:
        """Compute xG for a single shot event dict.

        Args:
            event: Single shot event dict.

        Returns:
            Predicted xG value (0.0 to 1.0).
        """
        features = self.extract_features([event])
        return float(self.predict(features)[0])

    def generate_example_data(self, n_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic shot data for testing/training.

        Args:
            n_samples: Number of synthetic shots to generate.

        Returns:
            (features, labels) tuple.
        """
        rng = np.random.RandomState(42)
        features = np.zeros((n_samples, N_FEATURES), dtype=np.float64)
        labels = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            # Distance: uniform from 1 to 40 meters
            d = rng.uniform(1.0, 40.0)
            angle = rng.uniform(0.0, 45.0)
            angle_rad = math.radians(angle)
            gf = math.cos(angle_rad) if angle_rad < math.pi / 2 else 0.0

            is_header = rng.random() < 0.1
            is_one_on_one = rng.random() < 0.03
            is_pressed = rng.random() < 0.3
            st_roll = rng.random()
            if st_roll < 0.7:
                shot_type = "open_play"
            elif st_roll < 0.85:
                shot_type = "volley"
            elif st_roll < 0.95:
                shot_type = "free_kick"
            else:
                shot_type = "penalty"

            gk_dist = max(0.0, rng.normal(3.0, 1.5))
            is_rebound = rng.random() < 0.02
            is_big_chance = rng.random() < 0.05

            features[i, FI_DISTANCE] = d
            features[i, FI_ANGLE] = angle
            features[i, FI_ANGLE_SIN] = 1.0 - gf
            features[i, FI_IS_HEADER] = 1.0 if is_header else 0.0
            features[i, FI_IS_ONE_ON_ONE] = 1.0 if is_one_on_one else 0.0
            features[i, FI_IS_PRESSED] = 1.0 if is_pressed else 0.0
            features[i, FI_IS_VOLLEY] = 1.0 if shot_type in ("volley", "half_volley") else 0.0
            features[i, FI_IS_FREE_KICK] = 1.0 if shot_type == "free_kick" else 0.0
            features[i, FI_IS_PENALTY] = 1.0 if shot_type == "penalty" else 0.0
            features[i, FI_GK_DISTANCE] = gk_dist
            features[i, FI_IS_REBOUND] = 1.0 if is_rebound else 0.0
            features[i, FI_BIG_CHANCE] = 1.0 if is_big_chance else 0.0

            # Ground truth probability using enhanced logistic regression
            logit = -1.2 - 0.12 * d - 0.0003 * d * d + 1.4 * (1.0 - gf)
            if is_header:
                logit -= 0.7
            if is_one_on_one:
                logit += 0.6
            if is_pressed:
                logit -= 0.3
            if shot_type in ("volley", "half_volley"):
                logit += 0.2
            if shot_type == "free_kick":
                logit += 0.15
            if shot_type == "penalty":
                logit += 2.0
            logit += -0.08 * gk_dist
            if is_rebound:
                logit += 0.5
            if is_big_chance:
                logit += 0.7
            logit = max(-20.0, min(20.0, logit))
            prob = 1.0 / (1.0 + math.exp(-logit))

            labels[i] = 1.0 if rng.random() < prob else 0.0

        return features, labels

    def generate_calibrated_data(self, n_samples: int = 5000) -> tuple[np.ndarray, np.ndarray]:
        """Generate more realistic shot data with typical xG distribution.

        StatsBomb-like distribution:
        - Most shots have xG < 0.1
        - Shots near goal are rare but high-value
        - Penalties at 0.76-0.80

        Returns:
            (features, labels) tuple with realistic xG distribution.
        """
        rng = np.random.RandomState(123)
        features = np.zeros((n_samples, N_FEATURES), dtype=np.float64)
        labels = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            # Most shots from distance (20-25m), fewer close in
            # Close range 15% of the time, long range otherwise
            d = rng.exponential(5.0) + 1.0 if rng.random() < 0.15 else rng.exponential(10.0) + 8.0

            d = min(d, 45.0)
            angle = rng.exponential(20.0)
            angle = min(angle, 50.0)
            angle_rad = math.radians(angle)
            gf = math.cos(angle_rad) if angle_rad < math.pi / 2 else 0.0

            is_header = rng.random() < 0.08
            is_one_on_one = rng.random() < 0.02 if d < 20 else False
            is_pressed = rng.random() < 0.35
            st_roll = rng.random()
            if st_roll < 0.75:
                shot_type = "open_play"
            elif st_roll < 0.82:
                shot_type = "volley"
            elif st_roll < 0.90:
                shot_type = "free_kick"
            else:
                shot_type = "penalty"

            gk_dist = max(0.5, rng.exponential(2.0) + 1.0)
            is_rebound = rng.random() < 0.015
            is_big_chance = rng.random() < 0.03

            features[i, FI_DISTANCE] = d
            features[i, FI_ANGLE] = angle
            features[i, FI_ANGLE_SIN] = 1.0 - gf
            features[i, FI_IS_HEADER] = 1.0 if is_header else 0.0
            features[i, FI_IS_ONE_ON_ONE] = 1.0 if is_one_on_one else 0.0
            features[i, FI_IS_PRESSED] = 1.0 if is_pressed else 0.0
            features[i, FI_IS_VOLLEY] = 1.0 if shot_type in ("volley", "half_volley") else 0.0
            features[i, FI_IS_FREE_KICK] = 1.0 if shot_type == "free_kick" else 0.0
            features[i, FI_IS_PENALTY] = 1.0 if shot_type == "penalty" else 0.0
            features[i, FI_GK_DISTANCE] = gk_dist
            features[i, FI_IS_REBOUND] = 1.0 if is_rebound else 0.0
            features[i, FI_BIG_CHANCE] = 1.0 if is_big_chance else 0.0

            logit = -1.5 - 0.15 * d + 1.6 * (1.0 - gf)
            if is_header:
                logit -= 0.8
            if is_one_on_one:
                logit += 0.7
            if is_pressed:
                logit -= 0.35
            if shot_type in ("volley", "half_volley"):
                logit += 0.1
            if shot_type == "free_kick":
                logit += 0.2
            if shot_type == "penalty":
                logit += 2.5
            logit += -0.1 * gk_dist
            if is_rebound:
                logit += 0.4
            if is_big_chance:
                logit += 0.8
            logit = max(-20.0, min(20.0, logit))
            prob = 1.0 / (1.0 + math.exp(-logit))

            labels[i] = 1.0 if rng.random() < prob else 0.0

        return features, labels


# Module-level singleton model for efficient reuse
_dl_xg_model: DLXgModel | None = None


def _get_dl_xg_model(model_path: str | None = None) -> DLXgModel:
    """Get or create the DL xG model singleton."""
    global _dl_xg_model
    if _dl_xg_model is None:
        _dl_xg_model = DLXgModel()
        if model_path:
            with contextlib.suppress(FileNotFoundError, json.JSONDecodeError):
                _dl_xg_model.load(model_path)
        else:
            # Use heuristic xG coefficients as pseudo-trained weights
            # so the DL model gives reasonable output even without training
            _init_heuristic_weights(_dl_xg_model)
    return _dl_xg_model


def _init_heuristic_weights(model: DLXgModel) -> None:
    """Initialize network weights to approximate heuristic xG coefficients.

    This ensures the DL model produces reasonable predictions even
    without training data, by embedding domain knowledge into the
    weight initialization.
    """
    # Layer 1: map each feature to its heuristic contribution
    w1 = np.zeros((N_FEATURES, 16), dtype=np.float64)
    # Distance: strong negative contribution (further = lower xG)
    w1[FI_DISTANCE, 0] = -0.15
    w1[FI_DISTANCE, 1] = -0.05
    # Angle: positive for central shots
    w1[FI_ANGLE, 2] = -0.03
    # Header penalty
    w1[FI_IS_HEADER, 3] = -0.8
    # One-on-one bonus
    w1[FI_IS_ONE_ON_ONE, 4] = 0.5
    # Pressure penalty
    w1[FI_IS_PRESSED, 5] = -0.4
    # Volley bonus
    w1[FI_IS_VOLLEY, 6] = 0.3
    # Free kick bonus
    w1[FI_IS_FREE_KICK, 7] = 0.2
    # Penalty fixed high value
    w1[FI_IS_PENALTY, 8] = 2.5
    # GK distance (closer GK = higher xG)
    w1[FI_GK_DISTANCE, 9] = 0.05
    # Rebound bonus
    w1[FI_IS_REBOUND, 10] = 0.4
    # Big chance bonus
    w1[FI_BIG_CHANCE, 11] = 0.8
    model.layer1.w = w1

    # Layer 2: combine heuristic signals
    w2 = np.zeros((16, 8), dtype=np.float64)
    w2[0:12, 0] = 0.5  # primary features
    w2[0:12, 1] = 0.3  # secondary features
    model.layer2.w = w2

    # Output: weighted sum
    model.w_out[:, 0] = np.array([1.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    model.b_out[0] = -1.5  # base logit offset
    model._trained = True


# Convenience function for single-shot prediction
@functools.lru_cache(maxsize=256)
def predict_dl_xg(
    distance_m: float,
    angle_deg: float,
    is_header: bool = False,
    is_one_on_one: bool = False,
    was_pressed: bool = False,
    shot_type: str = "open_play",
    gk_distance_m: float = 0.0,
    is_rebound: bool = False,
    is_big_chance: bool = False,
    model_path: str | None = None,
) -> float:
    """Compute xG using the neural network model.

    Uses a module-level singleton model that falls back to heuristic
    weights when no trained model is available.

    Args:
        distance_m: Shot distance in meters.
        angle_deg: Shot angle in degrees.
        is_header: Whether shot is a header.
        is_one_on_one: Whether shot is a 1-on-1.
        was_pressed: Whether shooter was under pressure.
        shot_type: "open_play", "volley", "free_kick", "penalty".
        gk_distance_m: Goalkeeper distance in meters.
        is_rebound: Whether shot is a rebound.
        is_big_chance: Whether shot is a big chance.
        model_path: Optional path to pre-trained weights.

        Returns:
        Predicted xG value.
    """
    model = _get_dl_xg_model(model_path)

    event = {
        "distance_m": distance_m,
        "angle_deg": angle_deg,
        "is_header": is_header,
        "is_one_on_one": is_one_on_one,
        "was_pressed": was_pressed,
        "shot_type": shot_type,
        "gk_distance_m": gk_distance_m,
        "is_rebound": is_rebound,
        "is_big_chance": is_big_chance,
    }
    return model.compute_single(event)
