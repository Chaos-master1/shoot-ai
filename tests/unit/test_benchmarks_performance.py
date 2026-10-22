"""Performance benchmarks for core analytics modules."""

import time

from kawkab.core.heatmap import compute_player_heatmap
from kawkab.core.momentum import compute_momentum_index
from kawkab.core.pitch_control import VoronoiPitchControl
from kawkab.core.xg_model import compute_xg
from kawkab.core.xt_model import ExpectedThreatModel


class TestBenchmarkPerformance:
    """Timing tests ensuring core operations stay under thresholds."""

    def test_xg_throughput(self):
        start = time.perf_counter()
        n = 10000
        for i in range(n):
            compute_xg(15 + (i % 30), 20 + (i % 60), body_part="right_foot" if i % 2 else "head")
        elapsed = time.perf_counter() - start
        ops_per_sec = n / elapsed
        assert ops_per_sec > 1000, f"xG throughput {ops_per_sec:.0f} ops/s < 1000"

    def test_xt_build_transition_matrix(self):
        n_events = 5000
        events = []
        for i in range(n_events):
            team = "home" if i % 2 == 0 else "away"
            events.append(
                {
                    "type": "pass",
                    "team": team,
                    "start_x": float(10 + (i % 90)),
                    "start_y": float(10 + (i % 50)),
                    "end_x": float(20 + (i % 80)),
                    "end_y": float(10 + (i % 50)),
                    "completed": bool(i % 3),
                    "timestamp": float(i),
                }
            )
        model = ExpectedThreatModel()
        start = time.perf_counter()
        model.build_transition_matrix(events)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"xT build took {elapsed:.2f}s > 2.0s"

    def test_heatmap_kde_throughput(self):
        n = 5000
        import random

        positions = [(random.random() * 105, random.random() * 68) for _ in range(n)]
        start = time.perf_counter()
        for _ in range(5):
            compute_player_heatmap(
                positions, pitch_length=105, pitch_width=68, grid_rows=30, grid_cols=46
            )
        elapsed = time.perf_counter() - start
        avg = elapsed / 5
        assert avg < 1.0, f"KDE heatmap avg {avg:.2f}s > 1.0s"

    def test_pitch_control_throughput(self):
        n_players = 22
        home_pos = [(float(20 + (i % 80)), float(10 + (i % 50))) for i in range(n_players // 2)]
        away_pos = [(float(50 + (i % 50)), float(10 + (i % 50))) for i in range(n_players // 2)]
        model = VoronoiPitchControl()
        start = time.perf_counter()
        for _ in range(10):
            model.compute_frame_control(home_pos, away_pos, ball_pos=(50, 34))
        elapsed = time.perf_counter() - start
        avg = elapsed / 10
        assert avg < 0.5, f"Pitch control avg {avg:.3f}s > 0.5s"

    def test_momentum_throughput(self):
        n = 1000
        events = []
        for i in range(n):
            events.append(
                {
                    "timestamp": float(i),
                    "type": "shot" if i % 5 == 0 else "pass",
                    "team": "home" if i % 2 == 0 else "away",
                    "x": float(50 + (i % 50)),
                    "y": float(34 + (i % 30)),
                    "xg": 0.1 * (i % 10) / 10,
                    "is_goal": i % 20 == 0,
                    "completed": True,
                }
            )
        start = time.perf_counter()
        result = compute_momentum_index(events, window_minutes=5)
        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, f"Momentum took {elapsed:.2f}s > 3.0s"
        assert len(result.timeline) > 0
