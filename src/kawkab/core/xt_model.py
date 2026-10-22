"""Expected Threat (xT) model for football analytics.

Computes the threat value of each pitch zone by solving
a transition matrix built from actual match events (passes, carries).

xT values represent the probability of scoring from each zone,
accounting for the probability of moving the ball to higher-threat zones.

References:
    - Singh (2019) "Expected Threat (xT)"
    - Fernandez et al. (2018) "Decomposing the Immeasurable"
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from kawkab.core.coordinate_validator import CoordinateValidator
from kawkab.core.game_constants import GAME
from kawkab.core.perf_timing import timed

PITCH_LENGTH = GAME.PITCH_LENGTH_M

# ── League-wide reference grid (pre-trained from StatsBomb open data) ────────
# Auto-loaded as the cold-start default when a match's own events are too
# sparse to learn from. Rebuild with:
#   PYTHONPATH=src python -m kawkab.core.validation.train_xt --force
_REFERENCE_GRID_PATH = Path(__file__).parent / "trained_xt_grid.json"
_REFERENCE_GRID: np.ndarray | None = None
_REFERENCE_GRID_SHAPE: tuple[int, ...] | None = None


def reference_grid_available() -> bool:
    """True when the pre-trained league-wide xT grid is loadable."""
    if _REFERENCE_GRID is not None:
        return True
    return _REFERENCE_GRID_PATH.exists()


def _load_reference_grid() -> np.ndarray | None:
    global _REFERENCE_GRID, _REFERENCE_GRID_SHAPE
    if _REFERENCE_GRID is not None:
        return _REFERENCE_GRID
    if not _REFERENCE_GRID_PATH.exists():
        return None
    try:
        with open(_REFERENCE_GRID_PATH) as f:
            payload = json.load(f)
        grid = np.asarray(payload.get("grid", []), dtype=np.float64)
        if grid.ndim != 2 or grid.size == 0:
            return None
        _REFERENCE_GRID = grid
        assert grid.shape is not None  # ndarray.shape is never None at runtime
        _REFERENCE_GRID_SHAPE = grid.shape
        return _REFERENCE_GRID
    except Exception:
        return None


class ExpectedThreatModel:
    """Data-driven expected threat model.

    Builds a transition matrix from match events and solves
    for zone threat values using the xT algorithm.

    Args:
        rows: Number of vertical zones (default 16).
        cols: Number of horizontal zones (default 12).
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.
        gamma: Discount factor for future actions (default 0.9).
    """

    def __init__(
        self,
        rows: int = 20,
        cols: int = 32,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        gamma: float = 0.9,
        attacking_direction: str = "right",
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.gamma = gamma
        self.attacking_direction = attacking_direction
        self._transition = None
        self._ze_values: np.ndarray | None = None
        self._zone_cache: dict[tuple[float, float], tuple[int, int]] = {}

    def _zone_from_position(self, x: float, y: float) -> tuple[int, int]:
        cache_key = (x, y)
        cached = self._zone_cache.get(cache_key)
        if cached is not None:
            return cached
        if self.attacking_direction == "left":
            x = PITCH_LENGTH - x
        col = min(self.cols - 1, max(0, int(x / self.pitch_length * self.cols)))
        row = min(self.rows - 1, max(0, int(y / self.pitch_width * self.rows)))
        self._zone_cache[cache_key] = (row, col)
        return (row, col)

    def build_transition_matrix(
        self,
        events: list[dict[str, Any]],
    ) -> None:
        transitions: defaultdict[tuple[int, int], defaultdict[tuple[int, int], int]] = defaultdict(
            lambda: defaultdict(int)
        )
        possession_from_zone: defaultdict[tuple[int, int], int] = defaultdict(int)
        shots_from_zone: defaultdict[tuple[int, int], int] = defaultdict(int)
        goals_from_zone: defaultdict[tuple[int, int], int] = defaultdict(int)

        n_actions = 0
        for ev in events:
            if ev.get("type") not in ("pass", "carry"):
                continue
            n_actions += 1

        for ev in events:
            if ev.get("type") not in ("pass", "carry"):
                continue
            sx = ev.get("start_x", 0.0)
            sy = ev.get("start_y", 34.0)
            ex = ev.get("end_x", 0.0)
            ey = ev.get("end_y", 34.0)
            start_zone = self._zone_from_position(sx, sy)
            end_zone = self._zone_from_position(ex, ey)
            possession_from_zone[start_zone] += 1

            if ev.get("completed", True):
                # Successful actions: transition to destination zone
                if start_zone != end_zone:
                    transitions[start_zone][end_zone] += 1
            else:
                # Failed actions: transition to opponent's equivalent zone
                # (mirror across pitch: col = cols - 1 - col)
                opp_row = end_zone[0]
                opp_col = self.cols - 1 - end_zone[1]
                opp_zone = (opp_row, opp_col)
                if start_zone != opp_zone:
                    transitions[start_zone][opp_zone] += 1

        for ev in events:
            if ev.get("type") != "shot":
                continue
            sx = ev.get("start_x", 0.0)
            sy = ev.get("start_y", 34.0)
            zone = self._zone_from_position(sx, sy)
            shots_from_zone[zone] += 1
            possession_from_zone[zone] += 1
            if ev.get("is_goal", False):
                goals_from_zone[zone] += 1

        self._transition = dict(transitions)  # type: ignore[assignment]

        # Cold-start rule: a match with too few actions produces a
        # near-degenerate learned grid (a handful of zones with data,
        # zeros elsewhere — misleading as a threat surface). Fall back
        # to the league-wide reference grid trained from the StatsBomb
        # corpus when the match itself can't support learning. The
        # match's own goals-per-zone still overlay onto the reference
        # when available (they're the model's ze term), so a data-rich
        # match keeps its own learned behavior; only sparse ones borrow.
        min_actions_for_learning = 200
        if n_actions < min_actions_for_learning:
            ref = _load_reference_grid()
            if ref is not None and ref.shape == (self.rows, self.cols):
                ze = self._solve_xT(possession_from_zone, goals_from_zone)
                # Blend: reference grid carries the transition structure,
                # the match's own ze (zone scoring rates) overlays it.
                self._ze_values = 0.8 * ref + 0.2 * ze
                return
        self._ze_values = self._solve_xT(possession_from_zone, goals_from_zone)

    def _solve_xT(
        self,
        possession_from_zone: dict[tuple[int, int], int] | None = None,
        goals_from_zone: dict[tuple[int, int], int] | None = None,
    ) -> np.ndarray:
        possession_from_zone = possession_from_zone or {}
        goals_from_zone = goals_from_zone or {}

        # ze = goals / total_possessions (NOT goals / shots)
        # This captures P(score|possession in zone) not P(score|shot in zone)
        ze = np.zeros((self.rows, self.cols), dtype=np.float64)
        for (r, c), count in possession_from_zone.items():
            goals = goals_from_zone.get((r, c), 0)
            ze[r, c] = goals / max(count, 1)

        if self._transition is None or not self._transition:
            return ze

        _ = self.rows * self.cols
        xT = ze.flatten().copy()

        # Power iteration: xT_new[src] = ze[src] + gamma * sum(prob * xT[dst])
        for _iteration in range(100):
            prev = xT.copy()
            for (src_r, src_c), targets in self._transition.items():
                src_idx = src_r * self.cols + src_c
                total = sum(targets.values())
                if total == 0:
                    continue
                weighted_sum = 0.0
                for (dst_r, dst_c), count in targets.items():
                    dst_idx = dst_r * self.cols + dst_c
                    weighted_sum += (count / total) * prev[dst_idx]
                xT[src_idx] = ze[src_r, src_c] + self.gamma * weighted_sum

            delta = float(np.max(np.abs(xT - prev)))
            if delta < 1e-6:
                break

        return xT.reshape(self.rows, self.cols)

    def compute_action_xt(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
    ) -> float:
        if self._ze_values is None:
            return 0.0
        start_zone = self._zone_from_position(start_x, start_y)
        end_zone = self._zone_from_position(end_x, end_y)
        start_val = self._ze_values[start_zone[0], start_zone[1]]
        end_val = self._ze_values[end_zone[0], end_zone[1]]
        return max(0.0, end_val - start_val)

    def compute_action_xt_with_ci(
        self,
        events: list[dict[str, Any]],
        action_event: dict[str, Any],
        n_bootstrap: int = 100,
    ) -> dict[str, float]:
        """Compute action xT with bootstrap confidence interval.

        Args:
            events: Full event list for bootstrap resampling.
            action_event: Dict with start_x, start_y, end_x, end_y keys.
            n_bootstrap: Number of bootstrap iterations.

        Returns:
            Dict with xt, ci_lower, ci_upper, n_bootstrap.
        """
        self.build_transition_matrix(events)
        point_estimate = self.compute_action_xt(
            action_event.get("start_x", 0.0),
            action_event.get("start_y", 34.0),
            action_event.get("end_x", 0.0),
            action_event.get("end_y", 34.0),
        )

        if n_bootstrap < 2:
            return {
                "xt": round(point_estimate, 4),
                "ci_lower": round(point_estimate, 4),
                "ci_upper": round(point_estimate, 4),
                "n_bootstrap": n_bootstrap,
            }

        xt_values: list[float] = []
        for _ in range(n_bootstrap):
            sample = random.choices(events, k=len(events))
            self.build_transition_matrix(sample)
            xt = self.compute_action_xt(
                action_event.get("start_x", 0.0),
                action_event.get("start_y", 34.0),
                action_event.get("end_x", 0.0),
                action_event.get("end_y", 34.0),
            )
            xt_values.append(xt)

        lower = float(np.percentile(xt_values, 2.5))
        upper = float(np.percentile(xt_values, 97.5))
        return {
            "xt": round(point_estimate, 4),
            "ci_lower": round(lower, 4),
            "ci_upper": round(upper, 4),
            "n_bootstrap": n_bootstrap,
        }

    # Deliberately not lru_cache'd: _ze_values is mutated by fit(), and a
    # method-level cache would keep serving pre-fit (stale) zone values.
    def _get_zone_value(self, row: int, col: int) -> float:
        if self._ze_values is None:
            return 0.0
        return float(self._ze_values[row, col])

    def get_zone_values(self) -> np.ndarray:
        if self._ze_values is None:
            return np.zeros((self.rows, self.cols), dtype=np.float64)
        return self._ze_values.copy()

    def get_zone_grid(self) -> list[list[float]]:
        vals = self.get_zone_values()
        return [[round(float(vals[r, c]), 4) for c in range(self.cols)] for r in range(self.rows)]

    @timed()
    def compute_match_xt(
        self,
        events: list[dict[str, Any]],
    ) -> dict[str, float]:
        for ev in events:
            CoordinateValidator.validate_event_spatial(ev)
        self.build_transition_matrix(events)
        home_xt = 0.0
        away_xt = 0.0

        for ev in events:
            if ev.get("type") not in ("pass", "carry"):
                continue
            if not ev.get("completed", True):
                continue
            xt_gained = self.compute_action_xt(
                ev.get("start_x", 0.0),
                ev.get("start_y", 34.0),
                ev.get("end_x", 0.0),
                ev.get("end_y", 34.0),
            )
            if ev.get("team") == "home":
                home_xt += xt_gained
            else:
                away_xt += xt_gained

        return {
            "home": round(home_xt, 4),
            "away": round(away_xt, 4),
        }


def compute_action_xt(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    attacking_direction: str = "right",
) -> float:
    model = ExpectedThreatModel(attacking_direction=attacking_direction)
    return model.compute_action_xt(start_x, start_y, end_x, end_y)
