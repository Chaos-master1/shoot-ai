"""Formation shape tracking — in-possession vs out-of-possession shapes.

Tracks formations separately for attacking and defending phases,
computes compactness, width, depth, and line distance metrics over time.
Uses k-means clustering on player x-coordinates to detect formation lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kawkab.core.perf_timing import timed

# Known formation patterns: (line_counts,) -> formation label
# Each pattern sums to 10 (outfield players)
FORMATION_PATTERNS: dict[tuple[int, ...], str] = {
    (4, 3, 3): "4-3-3",
    (4, 2, 3, 1): "4-2-3-1",
    (4, 4, 2): "4-4-2",
    (3, 4, 3): "3-4-3",
    (3, 5, 2): "3-5-2",
    (5, 3, 2): "5-3-2",
    (4, 1, 4, 1): "4-1-4-1",
}


@dataclass
class FormationSnapshot:
    """Shape metrics for a single frame."""

    timestamp: float = 0.0
    formation: str = "unknown"
    width: float = 0.0
    depth: float = 0.0
    compactness: float = 0.0
    defensive_line_height: float = 0.0
    defensive_line_y: float = 0.0
    line_distance: float = 0.0
    possession: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": round(self.timestamp, 1),
            "f": self.formation,
            "w": round(self.width, 1),
            "d": round(self.depth, 1),
            "c": round(self.compactness, 2),
            "dl": round(self.defensive_line_height, 1),
            "ld": round(self.line_distance, 1),
            "pos": self.possession,
        }


@dataclass
class FormationMatchReport:
    """Aggregate formation metrics for a full match."""

    in_possession_formation: str = "unknown"
    out_possession_formation: str = "unknown"
    avg_width_in: float = 0.0
    avg_width_out: float = 0.0
    avg_depth_in: float = 0.0
    avg_depth_out: float = 0.0
    avg_compactness_in: float = 0.0
    avg_compactness_out: float = 0.0
    def_line_height_in: float = 0.0
    def_line_height_out: float = 0.0
    timeline: list[FormationSnapshot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_possession_formation": self.in_possession_formation,
            "out_possession_formation": self.out_possession_formation,
            "avg_width_in": round(self.avg_width_in, 1),
            "avg_width_out": round(self.avg_width_out, 1),
            "avg_depth_in": round(self.avg_depth_in, 1),
            "avg_depth_out": round(self.avg_depth_out, 1),
            "avg_compactness_in": round(self.avg_compactness_in, 2),
            "avg_compactness_out": round(self.avg_compactness_out, 2),
            "def_line_height_in": round(self.def_line_height_in, 1),
            "def_line_height_out": round(self.def_line_height_out, 1),
        }


class FormationAnalyzer:
    """Analyzes team formations and shape over time.

    Uses k-means clustering on player x-coordinates to detect
    defensive/midfield/forward lines and classify formation.

    Usage:
        fa = FormationAnalyzer()
        report = fa.analyze_team_shape(frames_data, team="home")
    """

    @staticmethod
    def _kmeans_1d(
        data: np.ndarray,
        k: int,
        max_iter: int = 30,
    ) -> tuple[np.ndarray, np.ndarray]:
        """1D k-means clustering. Returns (centroids, labels)."""
        n = len(data)
        sorted_data = np.sort(data)
        indices = np.linspace(0, n - 1, k, dtype=int)
        centroids = sorted_data[indices].astype(np.float64)

        for _ in range(max_iter):
            diffs = np.abs(data[:, np.newaxis] - centroids[np.newaxis, :])
            labels = np.argmin(diffs, axis=1)

            new_centroids = np.empty(k)
            for i in range(k):
                mask = labels == i
                if np.any(mask):
                    new_centroids[i] = np.mean(data[mask])
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids, atol=1e-3):
                break
            centroids = new_centroids

        return centroids, labels

    def _classify_formation(
        self,
        positions: list[tuple[float, float]],
        centroids3: np.ndarray | None = None,
        labels3: np.ndarray | None = None,
        centroids4: np.ndarray | None = None,
        labels4: np.ndarray | None = None,
    ) -> str:
        n = len(positions)
        if n < 8:
            return "unknown"

        xs = np.array([p[0] for p in positions[:11]])

        if xs.max() - xs.min() < 15.0:
            return "unknown"

        best_formation = "unknown"
        best_score = -1.0

        for k, pre_centroids, pre_labels in (
            (3, centroids3, labels3),
            (4, centroids4, labels4),
        ):
            if k >= n:
                continue
            if pre_centroids is not None:
                centroids = pre_centroids
                labels = pre_labels
            else:
                centroids, labels = self._kmeans_1d(xs, k)

            order = np.argsort(centroids)
            counts = tuple(int(np.sum(labels == idx)) for idx in order)

            formation = FORMATION_PATTERNS.get(counts, "unknown")
            if formation != "unknown":
                score = 3.0 if k == 3 else 2.0
                if sum(counts) == n:
                    score += 1.0
                if score > best_score:
                    best_score = score
                    best_formation = formation

        if best_formation != "unknown":
            return best_formation

        _ = sorted(positions, key=lambda p: p[0])
        n_def = max(2, min(5, round(n * 0.4)))
        n_att = max(1, min(4, round(n * 0.3)))
        n_mid = n - n_def - n_att
        if n_mid < 2:
            n_mid = 2
            n_def = max(2, n - n_att - n_mid)
        return f"{n_def}-{n_mid}-{n_att}"

    @staticmethod
    @timed()
    def _compute_compactness(positions: list[tuple[float, float]]) -> float:
        if len(positions) < 2:
            return 0.0
        n = min(len(positions), 11)
        arr = np.array(positions[:n], dtype=np.float64)
        diff = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        return float(np.mean(dist[mask])) if np.any(mask) else 0.0

    @staticmethod
    def _compute_width(positions: list[tuple[float, float]]) -> float:
        if len(positions) < 2:
            return 0.0
        xs = [p[1] for p in positions]
        return max(xs) - min(xs)

    @staticmethod
    def _compute_depth(positions: list[tuple[float, float]]) -> float:
        if len(positions) < 2:
            return 0.0
        ys = [p[0] for p in positions]
        return max(ys) - min(ys)

    def _compute_line_distance(
        self,
        positions: list[tuple[float, float]],
        centroids: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> float:
        n = len(positions)
        if n < 8:
            return 0.0

        xs = np.array([p[0] for p in positions[:11]])
        if xs.max() - xs.min() < 15.0:
            return 0.0

        if centroids is None:
            centroids, labels = self._kmeans_1d(xs, 3)
        order = np.argsort(centroids)
        ordered_centroids = centroids[order]

        return float(abs(ordered_centroids[0] - ordered_centroids[1]))

    def _defensive_line_height(
        self,
        positions: list[tuple[float, float]],
        centroids: np.ndarray | None = None,
        labels: np.ndarray | None = None,
    ) -> float:
        n = len(positions)
        if n < 3:
            return 0.0

        xs = np.array([p[0] for p in positions[:11]])
        if xs.max() - xs.min() < 15.0:
            return float(np.mean(xs))

        if centroids is None:
            centroids, labels = self._kmeans_1d(xs, 3)
        order = np.argsort(centroids)
        return float(centroids[order[0]])

    def analyze_team_shape(
        self,
        frames: list[dict[str, Any]],
        team: str = "home",
        pitch_length: float = 105.0,
    ) -> FormationMatchReport:
        if not frames:
            return FormationMatchReport()

        in_pos_snapshots: list[FormationSnapshot] = []
        out_pos_snapshots: list[FormationSnapshot] = []

        for fdata in frames:
            ts = fdata.get("timestamp", 0.0)
            possession = fdata.get("possession", False)
            positions = fdata.get(f"{team}_positions", [])
            if len(positions) < 3:
                continue

            xs = np.array([p[0] for p in positions[:11]])
            centroids3: np.ndarray | None
            labels3: np.ndarray | None
            centroids4: np.ndarray | None
            labels4: np.ndarray | None
            if len(xs) >= 3 and xs.max() - xs.min() >= 15.0:
                centroids3, labels3 = self._kmeans_1d(xs, 3)
                centroids4, labels4 = self._kmeans_1d(xs, 4) if len(xs) >= 4 else (None, None)
            else:
                centroids3 = labels3 = centroids4 = labels4 = None

            formation = self._classify_formation(
                positions, centroids3, labels3, centroids4, labels4
            )
            width = self._compute_width(positions)
            depth = self._compute_depth(positions)
            compactness = self._compute_compactness(positions)
            def_line = self._defensive_line_height(positions, centroids3, labels3)
            line_dist = self._compute_line_distance(positions, centroids3, labels3)

            snap = FormationSnapshot(
                timestamp=ts,
                formation=formation,
                width=width,
                depth=depth,
                compactness=compactness,
                defensive_line_height=def_line,
                line_distance=line_dist,
                possession=possession,
            )

            if possession:
                in_pos_snapshots.append(snap)
            else:
                out_pos_snapshots.append(snap)

        def _avg(attr: str, snaps: list[FormationSnapshot]) -> float:
            if not snaps:
                return 0.0
            return sum(getattr(s, attr) for s in snaps) / len(snaps)

        in_form = ""
        out_form = ""
        in_form_freq: dict[str, int] = {}
        out_form_freq: dict[str, int] = {}
        if in_pos_snapshots:
            for s in in_pos_snapshots:
                in_form_freq[s.formation] = in_form_freq.get(s.formation, 0) + 1
            in_form = (
                max(in_form_freq, key=lambda k: in_form_freq[k]) if in_form_freq else "unknown"
            )

        if out_pos_snapshots:
            for s in out_pos_snapshots:
                out_form_freq[s.formation] = out_form_freq.get(s.formation, 0) + 1
            out_form = (
                max(out_form_freq, key=lambda k: out_form_freq[k]) if out_form_freq else "unknown"
            )

        timeline = sorted(
            in_pos_snapshots + out_pos_snapshots,
            key=lambda s: s.timestamp,
        )

        return FormationMatchReport(
            in_possession_formation=in_form,
            out_possession_formation=out_form,
            avg_width_in=_avg("width", in_pos_snapshots),
            avg_width_out=_avg("width", out_pos_snapshots),
            avg_depth_in=_avg("depth", in_pos_snapshots),
            avg_depth_out=_avg("depth", out_pos_snapshots),
            avg_compactness_in=_avg("compactness", in_pos_snapshots),
            avg_compactness_out=_avg("compactness", out_pos_snapshots),
            def_line_height_in=_avg("defensive_line_height", in_pos_snapshots),
            def_line_height_out=_avg("defensive_line_height", out_pos_snapshots),
            timeline=timeline,
        )
