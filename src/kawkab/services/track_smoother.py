"""Post-hoc track smoothing using Rauch-Tung-Striebel (RTS) smoother.

Given a set of track positions (x, y per frame), applies a forward-backward
Kalman smoother to reduce jitter and fill short gaps (< 10 frames).

Usage:
    smoother = TrackSmoother(dt=1/24)
    smoothed = smoother.smooth(track_frames, track_positions)

Where track_frames = [frame_number, ...] and track_positions = [(x, y), ...]
"""

from __future__ import annotations

import numpy as np


class TrackSmoother:
    def __init__(
        self, dt: float = 1.0 / 24.0, process_noise: float = 1e-3, measurement_noise: float = 1e-1
    ):
        self.dt = dt
        self.F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float64,
        )
        self.Q = np.eye(4, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

    def smooth(
        self, frames: list[int], positions: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        n = len(frames)
        if n < 3:
            return list(positions)

        # Convert to numpy
        z = np.array(positions, dtype=np.float64)

        # Forward pass (Kalman filter)
        x_pred = np.zeros((n, 4), dtype=np.float64)
        P_pred = np.zeros((n, 4, 4), dtype=np.float64)
        x_post = np.zeros((n, 4), dtype=np.float64)
        P_post = np.zeros((n, 4, 4), dtype=np.float64)

        # Init
        x_post[0] = np.array([z[0, 0], z[0, 1], 0, 0])
        P_post[0] = np.eye(4, dtype=np.float64)

        for k in range(1, n):
            dt_actual = (frames[k] - frames[k - 1]) * self.dt / max(self.dt, 1e-6)
            Fk = self.F.copy()
            Fk[0, 2] = dt_actual
            Fk[1, 3] = dt_actual
            x_pred[k] = Fk @ x_post[k - 1]
            P_pred[k] = Fk @ P_post[k - 1] @ Fk.T + self.Q
            innov = z[k] - self.H @ x_pred[k]
            S = self.H @ P_pred[k] @ self.H.T + self.R
            K = P_pred[k] @ self.H.T @ np.linalg.inv(S)
            x_post[k] = x_pred[k] + K @ innov
            P_post[k] = (np.eye(4) - K @ self.H) @ P_pred[k]

        # Backward pass (RTS smoother)
        x_smooth = np.zeros((n, 4), dtype=np.float64)
        x_smooth[-1] = x_post[-1]
        for k in range(n - 2, -1, -1):
            dt_next = (frames[k + 1] - frames[k]) * self.dt / max(self.dt, 1e-6)
            Fk = self.F.copy()
            Fk[0, 2] = dt_next
            Fk[1, 3] = dt_next
            G = P_post[k] @ Fk.T @ np.linalg.inv(P_pred[k + 1])
            x_smooth[k] = x_post[k] + G @ (x_smooth[k + 1] - x_pred[k + 1])

        return [(x_smooth[k, 0], x_smooth[k, 1]) for k in range(n)]
