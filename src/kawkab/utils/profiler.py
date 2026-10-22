"""Lightweight performance profiler for Kawkab AI.

Tracks per-stage timing during analysis and produces a structured
report with:
- Total elapsed time
- Per-stage breakdown (count, mean, total, p50, p95, p99)
- Bottleneck identification (stages above 5% of total)
- Memory delta (if psutil is available; otherwise skipped)

Usage:
    profiler = Profiler()
    profiler.start("cv_process")
    ... do work ...
    profiler.end("cv_process")
    ...
    report = profiler.report()
    print(report)
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StageStats:
    """Statistics for one named stage."""

    name: str = ""
    count: int = 0
    total_s: float = 0.0
    min_s: float = float("inf")
    max_s: float = 0.0
    samples: list[float] = field(default_factory=list)

    def record(self, elapsed_s: float) -> None:
        self.count += 1
        self.total_s += elapsed_s
        self.min_s = min(self.min_s, elapsed_s)
        self.max_s = max(self.max_s, elapsed_s)
        if len(self.samples) < 1000:
            self.samples.append(elapsed_s)

    def mean(self) -> float:
        return self.total_s / self.count if self.count > 0 else 0.0

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        k = (len(sorted_samples) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_samples[int(k)]
        return sorted_samples[f] * (c - k) + sorted_samples[c] * (k - f)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "total_s": round(self.total_s, 4),
            "mean_s": round(self.mean(), 4),
            "min_s": round(self.min_s, 4) if self.min_s != float("inf") else 0.0,
            "max_s": round(self.max_s, 4),
            "p50_s": round(self.percentile(0.5), 4),
            "p95_s": round(self.percentile(0.95), 4),
            "p99_s": round(self.percentile(0.99), 4),
        }


@dataclass
class ProfileReport:
    """Full profile report."""

    total_s: float
    stages: list[StageStats]
    bottlenecks: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_s": round(self.total_s, 3),
            "stages": [s.to_dict() for s in self.stages],
            "bottlenecks": self.bottlenecks,
            "notes": self.notes,
        }

    def __str__(self) -> str:
        lines = [f"Profile report — total {self.total_s:.2f}s"]
        for s in self.stages:
            lines.append(
                f"  {s.name:>24s}: {s.count:>5d}x  total {s.total_s:>7.2f}s  "
                f"mean {s.mean() * 1000:>7.1f}ms  p95 {s.percentile(0.95) * 1000:>7.1f}ms"
            )
        if self.bottlenecks:
            lines.append("Bottlenecks: " + "; ".join(self.bottlenecks))
        if self.notes:
            lines.append("Notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)


class Profiler:
    """Track per-stage timing across an analysis run."""

    def __init__(self) -> None:
        self._stages: dict[str, StageStats] = defaultdict(StageStats)
        self._open_starts: dict[str, float] = {}
        self._start_time: float | None = None
        self._stop_time: float | None = None

    def start(self, total: bool = True) -> None:
        if total:
            self._start_time = time.monotonic()

    def stop(self) -> None:
        self._stop_time = time.monotonic()

    def begin(self, name: str) -> None:
        if name in self._open_starts:
            logger.warning("Stage %s already running; nested begin()", name)
            return
        self._open_starts[name] = time.monotonic()

    def end(self, name: str) -> float:
        start = self._open_starts.pop(name, None)
        if start is None:
            logger.warning("Stage %s ended without matching begin", name)
            return 0.0
        elapsed = time.monotonic() - start
        stats = self._stages[name]
        if not stats.name:
            stats.name = name
        stats.record(elapsed)
        return elapsed

    @contextmanager
    def stage(self, name: str) -> Any:
        """Context-manager form: ``with profiler.stage("cv"): ...``"""
        self.begin(name)
        try:
            yield
        finally:
            self.end(name)

    def record(self, name: str, elapsed_s: float) -> None:
        stats = self._stages[name]
        if not stats.name:
            stats.name = name
        stats.record(elapsed_s)

    def report(self) -> ProfileReport:
        total = (self._stop_time or time.monotonic()) - (self._start_time or time.monotonic())
        stages = sorted(self._stages.values(), key=lambda s: -s.total_s)
        bottlenecks: list[str] = []
        if total > 0:
            for s in stages:
                pct = s.total_s / total * 100
                if pct >= 5.0:
                    bottlenecks.append(f"{s.name} ({pct:.1f}%)")
        notes: list[str] = []
        if not stages:
            notes.append("No stages recorded.")
        elif total == 0:
            notes.append("Total time is 0 — check profiler.start() and stop().")
        return ProfileReport(total_s=total, stages=stages, bottlenecks=bottlenecks, notes=notes)

    def reset(self) -> None:
        self._stages.clear()
        self._open_starts.clear()
        self._start_time = None
        self._stop_time = None

    def get_stage(self, name: str) -> StageStats | None:
        return self._stages.get(name)
