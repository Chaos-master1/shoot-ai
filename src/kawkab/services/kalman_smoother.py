"""Kalman smoother for player tracking positions.

Conservative parameters tuned for amateur broadcast footage
with frame_skip (dt up to 0.04s at 50fps with frame_skip=2).

Applies 3-frame median pre-filter + constant-velocity Kalman
to produce smooth position sequences for distance/speed
computations.

References:
  - Typical Kalman filter (Kalman, 1960)
  - Constant velocity model for sports tracking
"""

from __future__ import annotations

from collections import deque

import numpy as np


class PlayerPositionSmoother:
    """Kalman smoother for a single player's position trajectory.

    State: [x, y, vx, vy] (4D)
    Measurement: [x, y] (position from bbox center -> pitch meters)

    Conservative parameters (max sprint up to 9 m/s = 32 km/h):
      - process_noise_std = 1.0  m/s^2 (acceleration uncertainty)
      - measurement_noise_std = 0.5  m (bbox jitter)
    """

    def __init__(
        self,
        process_noise_std: float = 0.3,
        measurement_noise_std: float = 0.8,
    ) -> None:
        self.q_std = process_noise_std
        self.r_std = measurement_noise_std

        self._initialized = False
        self._state: np.ndarray | None = None
        self._cov: np.ndarray | None = None

        self._median_buffer: deque[tuple[float, float]] = deque(maxlen=3)

    def reset(self) -> None:
        self._initialized = False
        self._state = None
        self._cov = None
        self._median_buffer.clear()

    @property
    def initialized(self) -> bool:
        return self._initialized and self._state is not None

    def _init_state(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0) -> None:
        self._state = np.array([x, y, vx, vy], dtype=np.float64)
        self._cov = np.eye(4, dtype=np.float64) * 5.0
        self._initialized = True

    def _velocity_from_buffer(self) -> tuple[float, float]:
        items = list(self._median_buffer)
        if len(items) < 2:
            return 0.0, 0.0
        vx_sum = 0.0
        vy_sum = 0.0
        n = len(items)
        for i in range(1, n):
            dt = 0.04
            vx_sum += (items[i][0] - items[i - 1][0]) / dt
            vy_sum += (items[i][1] - items[i - 1][1]) / dt
        n_pairs = n - 1
        return vx_sum / n_pairs, vy_sum / n_pairs

    def _median_filter(self, x: float, y: float) -> tuple[float, float]:
        self._median_buffer.append((x, y))
        if len(self._median_buffer) < 3:
            return x, y
        xs = [p[0] for p in self._median_buffer]
        ys = [p[1] for p in self._median_buffer]
        return float(np.median(xs)), float(np.median(ys))

    def _predict(self, dt: float) -> None:
        if self._state is None or self._cov is None:
            return
        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        dt2 = dt * dt
        G = np.array(
            [
                [dt2 / 2, 0],
                [0, dt2 / 2],
                [dt, 0],
                [0, dt],
            ],
            dtype=np.float64,
        )
        self._state = F @ self._state
        Q = G @ G.T * (self.q_std * self.q_std)
        Q = np.maximum(Q, np.eye(4) * 1e-6)
        self._cov = F @ self._cov @ F.T + Q
        self._cov = np.maximum(self._cov, np.eye(4) * 1e-6)

    def _update(self, x: float, y: float) -> None:
        if self._state is None or self._cov is None:
            return
        z = np.array([x, y], dtype=np.float64)
        H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float64,
        )
        R = np.eye(2, dtype=np.float64) * (self.r_std * self.r_std)
        y_vec = z - H @ self._state
        S = H @ self._cov @ H.T + R
        try:
            K = self._cov @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self._state = self._state + K @ y_vec
        IKH = np.eye(4) - K @ H
        self._cov = IKH @ self._cov
        self._cov = np.maximum(self._cov, np.eye(4) * 1e-6)

    def update(self, x: float, y: float, dt: float) -> None:
        """Feed a new (x, y) measurement, dt seconds since last measurement."""
        if not self._initialized:
            self._init_state(x, y)
            self._median_buffer.append((x, y))
            return

        state = self._state
        assert state is not None  # _init_state ran above (early return otherwise)
        mx, my = self._median_filter(x, y)
        diff = np.sqrt((mx - state[0]) ** 2 + (my - state[1]) ** 2)
        if diff > 2.0:
            mx, my = state[0], state[1]

        self._predict(dt)
        self._update(mx, my)

    def predict_only(self, dt: float) -> None:
        """Run prediction step without measurement update.

        Used when measurement is rejected (high innovation).
        Keeps the state estimate ticking forward.
        """
        if not self._initialized:
            return
        self._predict(dt)

    def get_position(self) -> tuple[float, float]:
        if self._state is None:
            return 0.0, 0.0
        return float(self._state[0]), float(self._state[1])

    def get_velocity(self) -> tuple[float, float]:
        if self._state is None:
            return 0.0, 0.0
        return float(self._state[2]), float(self._state[3])

    def get_speed_mps(self) -> float:
        vx, vy = self.get_velocity()
        return float(np.sqrt(vx * vx + vy * vy))

    def get_position_innovation(self, x: float, y: float) -> float:
        if self._state is None:
            return 0.0
        px, py = self.get_position()
        return float(np.sqrt((x - px) ** 2 + (y - py) ** 2))
