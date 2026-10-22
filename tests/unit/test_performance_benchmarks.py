"""Performance benchmarks for all key analytical modules.

Each benchmark:
- Uses time.perf_counter() to measure wall time
- Runs N iterations (configurable per module)
- Reports: min, max, mean, median, p95
- Asserts time < threshold (soft failure — warns if exceeded)
- Stores results in a BenchmarkResult dataclass
"""

from __future__ import annotations

import random
import statistics
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

# ── Benchmark result dataclass ──────────────────────────────────────


@dataclass
class BenchmarkResult:
    module: str
    n_iterations: int
    times_ms: list[float] = field(default_factory=list)

    @property
    def min_ms(self) -> float:
        return min(self.times_ms) if self.times_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.times_ms) if self.times_ms else 0.0

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0.0
        sorted_t = sorted(self.times_ms)
        idx = min(int(len(sorted_t) * 0.95), len(sorted_t) - 1)
        return sorted_t[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "n_iterations": self.n_iterations,
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
        }


# ── Benchmark data generators ──────────────────────────────────────


def _generate_xg_events(n: int) -> list[dict]:
    return [
        {
            "type": "shot",
            "team": "home" if i % 2 == 0 else "away",
            "timestamp": float(i),
            "metadata": {
                "distance_to_goal_m": random.uniform(5, 40),
                "angle_to_goal_deg": random.uniform(0, 90),
                "xg": random.uniform(0.01, 0.5),
            },
            "on_target": i % 3 == 0,
        }
        for i in range(n)
    ]


def _generate_xt_events(n: int) -> list[dict]:
    return [
        {
            "type": "pass",
            "team": "home" if i % 2 == 0 else "away",
            "completed": True,
            "timestamp": float(i),
            "metadata": {
                "start_x_pct": random.uniform(0.0, 0.5),
                "end_x_pct": random.uniform(0.3, 1.0),
            },
        }
        for i in range(n)
    ]


def _generate_vaep_events(n: int) -> list[dict]:
    events = []
    for i in range(n):
        ts = float(i)
        team = "home" if i % 2 == 0 else "away"
        if i % 5 == 0:
            events.append(
                {
                    "type": "shot",
                    "team": team,
                    "timestamp": ts,
                    "x": random.uniform(50, 105),
                    "y": random.uniform(10, 60),
                    "is_goal": i % 20 == 0,
                    "xg": random.uniform(0.01, 0.5),
                }
            )
        elif i % 5 == 1:
            events.append(
                {
                    "type": "pass",
                    "team": team,
                    "timestamp": ts,
                    "x": random.uniform(0, 105),
                    "y": random.uniform(0, 68),
                    "completed": True,
                }
            )
        elif i % 5 == 2:
            events.append(
                {
                    "type": "tackle",
                    "team": team,
                    "timestamp": ts,
                    "x": random.uniform(0, 105),
                    "y": random.uniform(0, 68),
                }
            )
        else:
            events.append(
                {
                    "type": "carry",
                    "team": team,
                    "timestamp": ts,
                    "x": random.uniform(0, 105),
                    "y": random.uniform(0, 68),
                }
            )
    return events


def _generate_player_positions(n_players: int = 22) -> tuple[list, list, tuple]:
    home = [(random.uniform(0, 105), random.uniform(0, 68)) for _ in range(n_players // 2)]
    away = [(random.uniform(0, 105), random.uniform(0, 68)) for _ in range(n_players // 2)]
    ball = (random.uniform(0, 105), random.uniform(0, 68))
    return home, away, ball


def _generate_formation_events(n: int) -> list[tuple[float, float]]:
    return [(random.uniform(0, 105), random.uniform(0, 68)) for _ in range(n)]


def _generate_benchmark_data(module: str, n_events: int) -> Any:
    if module == "xg_model":
        return _generate_xg_events(n_events)
    elif module == "xt_model":
        return _generate_xt_events(n_events)
    elif module == "vaep":
        return _generate_vaep_events(n_events)
    elif module == "pitch_control":
        return _generate_player_positions()
    elif module == "formation_analysis":
        return _generate_formation_events(n_events)
    elif module == "win_probability":
        return _generate_vaep_events(n_events)
    elif module == "space_control":
        return _generate_player_positions()
    elif module == "through_ball":
        return _generate_xt_events(n_events)
    elif module == "role_classifier":
        return _generate_vaep_events(n_events)
    return []


# ── Benchmark runner ───────────────────────────────────────────────


class BenchmarkRunner:
    """Runs benchmarks and collects results."""

    def __init__(self, n_iterations: int = 5):
        self.n_iterations = n_iterations
        self.results: dict[str, BenchmarkResult] = {}
        self.thresholds: dict[str, float] = {
            "xg_model": 2000.0,
            "xt_model": 5000.0,
            "vaep": 5000.0,
            "pitch_control": 500.0,
            "formation_analysis": 2000.0,
            "win_probability": 10000.0,
            "space_control": 1000.0,
            "through_ball": 1000.0,
            "role_classifier": 1000.0,
        }

    def run_benchmark(self, module: str, n_events: int) -> BenchmarkResult:
        data = _generate_benchmark_data(module, n_events)
        times_ms = []

        for _ in range(self.n_iterations):
            start = time.perf_counter()
            if module == "xg_model":
                self._run_xg(data)
            elif module == "xt_model":
                self._run_xt(data)
            elif module == "vaep":
                self._run_vaep(data)
            elif module == "pitch_control":
                self._run_pitch_control(data)
            elif module == "formation_analysis":
                self._run_formation(data)
            elif module == "win_probability":
                self._run_win_probability(data)
            elif module == "space_control":
                self._run_space_control(data)
            elif module == "through_ball":
                self._run_through_ball(data)
            elif module == "role_classifier":
                self._run_role_classifier(data)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed_ms)

        result = BenchmarkResult(module=module, n_iterations=self.n_iterations, times_ms=times_ms)
        self.results[module] = result
        return result

    def _run_xg(self, events: list[dict]) -> None:
        from kawkab.core.xg_model import compute_xg

        for ev in events:
            meta = ev.get("metadata", {})
            dist = meta.get("distance_to_goal_m", 18.0)
            angle = meta.get("angle_to_goal_deg", 30.0)
            compute_xg(dist, angle)

    def _run_xt(self, events: list[dict]) -> None:
        from kawkab.core.xt_model import ExpectedThreatModel

        model = ExpectedThreatModel()
        model.build_transition_matrix(events)

    def _run_vaep(self, events: list[dict]) -> None:
        from kawkab.core.vaep import compute_vaep

        compute_vaep(events)

    def _run_pitch_control(self, data: tuple) -> None:
        from kawkab.core.pitch_control import VoronoiPitchControl

        home, away, ball = data
        pc = VoronoiPitchControl()
        pc.compute_frame_control(home, away, ball)

    def _run_formation(self, positions: list) -> None:
        from kawkab.core.formation_analysis import FormationAnalyzer

        fa = FormationAnalyzer()
        fa._classify_formation(positions)

    def _run_win_probability(self, events: list[dict]) -> None:
        from kawkab.core.win_probability import compute_win_probability

        compute_win_probability(events)

    def _run_space_control(self, data: tuple) -> None:
        from kawkab.core.space_control import compute_pitch_control_grid

        home, away, _ = data
        all_positions = [(x, y, i) for i, (x, y) in enumerate(home + away)]
        team_ids = [0] * len(home) + [1] * len(away)
        compute_pitch_control_grid(all_positions, team_ids)

    def _run_through_ball(self, events: list[dict]) -> None:
        from kawkab.core.through_ball import detect_through_balls

        detect_through_balls(events, [])

    def _run_role_classifier(self, events: list[dict]) -> None:
        from kawkab.core.role_classifier import classify_player_role

        # Filter events for a single player if track_id exists, else pass as-is
        player_events = [
            e for e in events if e.get("player_track_id") == 1 or e.get("from_track_id") == 1
        ]
        if not player_events:
            player_events = events[:10] if len(events) >= 10 else events
        classify_player_role(player_events)

    def report(self) -> dict[str, dict[str, Any]]:
        return {name: r.to_dict() for name, r in self.results.items()}

    def check_thresholds(self) -> list[tuple[str, float, float, str]]:
        """Check each result against thresholds. Returns [(module, actual, threshold, status)]."""
        checks = []
        for module, result in self.results.items():
            threshold = self.thresholds.get(module, float("inf"))
            mean = result.mean_ms
            status = "PASS" if mean < threshold else "WARN"
            if mean >= threshold:
                warnings.warn(
                    f"{module}: mean {mean:.1f}ms >= threshold {threshold}ms",
                    stacklevel=2,
                )
            checks.append((module, mean, threshold, status))
        return checks


# ── Tests ──────────────────────────────────────────────────────────

MODULES_TO_BENCHMARK = [
    "xg_model",
    "xt_model",
    "vaep",
    "pitch_control",
    "formation_analysis",
    "win_probability",
    "space_control",
    "through_ball",
    "role_classifier",
]

N_EVENTS_DEFAULT = {
    "xg_model": 1000,
    "xt_model": 100,
    "vaep": 100,
    "pitch_control": 22,
    "formation_analysis": 100,
    "win_probability": 100,
    "space_control": 22,
    "through_ball": 100,
    "role_classifier": 50,
}


class TestPerformanceBenchmarks:
    """Benchmark each analytical module and verify timing thresholds."""

    def _make_runner(self, n_iterations: int = 3) -> BenchmarkRunner:
        return BenchmarkRunner(n_iterations=n_iterations)

    def test_benchmark_result_dataclass(self):
        r = BenchmarkResult(module="test", n_iterations=5, times_ms=[10.0, 20.0, 30.0, 40.0, 50.0])
        assert r.min_ms == 10.0
        assert r.max_ms == 50.0
        assert r.mean_ms == 30.0
        assert r.median_ms == 30.0
        assert r.p95_ms == 50.0
        assert r.to_dict()["module"] == "test"

    def test_benchmark_result_empty(self):
        r = BenchmarkResult(module="empty", n_iterations=0)
        assert r.min_ms == 0.0
        assert r.max_ms == 0.0
        assert r.mean_ms == 0.0
        assert r.p95_ms == 0.0

    def test_data_generators(self):
        assert len(_generate_xg_events(100)) == 100
        assert len(_generate_xt_events(50)) == 50
        assert len(_generate_vaep_events(30)) == 30
        home, away, ball = _generate_player_positions(22)
        assert len(home) == 11
        assert len(away) == 11
        assert len(ball) == 2

    def test_benchmark_xg_model(self):
        runner = self._make_runner(n_iterations=3)
        result = runner.run_benchmark("xg_model", 1000)
        assert result.module == "xg_model"
        assert len(result.times_ms) == 3
        assert result.min_ms > 0

    def test_benchmark_xt_model(self):
        runner = self._make_runner(n_iterations=3)
        result = runner.run_benchmark("xt_model", 100)
        assert result.module == "xt_model"

    def test_benchmark_vaep(self):
        runner = self._make_runner(n_iterations=3)
        result = runner.run_benchmark("vaep", 100)
        assert result.module == "vaep"

    def test_benchmark_pitch_control(self):
        runner = self._make_runner(n_iterations=3)
        result = runner.run_benchmark("pitch_control", 22)
        assert result.module == "pitch_control"

    def test_benchmark_formation_analysis(self):
        runner = self._make_runner(n_iterations=3)
        result = runner.run_benchmark("formation_analysis", 100)
        assert result.module == "formation_analysis"

    def test_benchmark_win_probability(self):
        runner = self._make_runner(n_iterations=3)
        result = runner.run_benchmark("win_probability", 100)
        assert result.module == "win_probability"

    def test_benchmark_space_control(self):
        runner = self._make_runner(n_iterations=3)
        result = runner.run_benchmark("space_control", 22)
        assert result.module == "space_control"

    def test_benchmark_through_ball(self):
        runner = self._make_runner(n_iterations=3)
        result = runner.run_benchmark("through_ball", 100)
        assert result.module == "through_ball"

    def test_benchmark_role_classifier(self):
        runner = self._make_runner(n_iterations=3)
        result = runner.run_benchmark("role_classifier", 50)
        assert result.module == "role_classifier"

    def test_all_modules_benchmark(self):
        """Run all benchmarks and report results."""
        runner = self._make_runner(n_iterations=3)
        for module in MODULES_TO_BENCHMARK:
            n_events = N_EVENTS_DEFAULT[module]
            runner.run_benchmark(module, n_events)
        report = runner.report()
        assert len(report) == len(MODULES_TO_BENCHMARK)
        for module in MODULES_TO_BENCHMARK:
            assert module in report
            assert report[module]["mean_ms"] > 0

    def test_threshold_checks(self):
        runner = self._make_runner(n_iterations=3)
        for module in MODULES_TO_BENCHMARK:
            n_events = N_EVENTS_DEFAULT[module]
            runner.run_benchmark(module, n_events)
        checks = runner.check_thresholds()
        assert len(checks) == len(MODULES_TO_BENCHMARK)
        for _module, _mean, _threshold, status in checks:
            assert status in ("PASS", "WARN")

    def test_empty_data_all_modules(self):
        """Benchmarks should handle empty data gracefully."""
        runner = self._make_runner(n_iterations=1)
        for module in MODULES_TO_BENCHMARK:
            result = runner.run_benchmark(module, 0)
            assert result.mean_ms >= 0

    def test_report_includes_all_stats(self):
        runner = self._make_runner(n_iterations=3)
        runner.run_benchmark("xg_model", 100)
        report = runner.report()
        xg_report = report["xg_model"]
        assert "min_ms" in xg_report
        assert "max_ms" in xg_report
        assert "mean_ms" in xg_report
        assert "median_ms" in xg_report
        assert "p95_ms" in xg_report
        assert "n_iterations" in xg_report

    def test_p95_computation(self):
        r = BenchmarkResult(module="test", n_iterations=10, times_ms=list(range(1, 11)))
        assert r.p95_ms == 10.0


# ── Module-specific benchmark tests ────────────────────────────────


class TestXgModelBenchmark:
    def test_compute_xg_throughput(self):
        from kawkab.core.xg_model import compute_xg

        n = 10000
        start = time.perf_counter()
        for i in range(n):
            compute_xg(15 + (i % 30), 20 + (i % 60), body_part="right_foot" if i % 2 else "head")
        elapsed = time.perf_counter() - start
        ops_per_sec = n / elapsed
        assert ops_per_sec > 1000, f"xG throughput {ops_per_sec:.0f} ops/s < 1000"


class TestPitchControlBenchmark:
    def test_pitch_control_throughput(self):
        from kawkab.core.pitch_control import VoronoiPitchControl

        n_players = 22
        home_pos = [(float(20 + (i % 80)), float(10 + (i % 50))) for i in range(n_players // 2)]
        away_pos = [(float(50 + (i % 50)), float(10 + (i % 50))) for i in range(n_players // 2)]
        pc = VoronoiPitchControl()
        start = time.perf_counter()
        for _ in range(10):
            pc.compute_frame_control(home_pos, away_pos, ball_pos=(50, 34))
        elapsed = time.perf_counter() - start
        avg = elapsed / 10
        assert avg < 0.5, f"Pitch control avg {avg:.3f}s > 0.5s"


class TestMomentumBenchmark:
    def test_momentum_throughput(self):
        from kawkab.core.momentum import compute_momentum_index

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


class TestXTBuildBenchmark:
    def test_xt_build_transition_matrix(self):
        from kawkab.core.xt_model import ExpectedThreatModel

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
