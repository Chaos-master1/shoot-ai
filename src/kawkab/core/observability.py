"""Lightweight Prometheus-style metrics collection for Kawkab AI.

Provides Counter, Gauge, and Histogram primitives with an exposition
format output compatible with the Prometheus text format.  All metrics
are in-process only (no HTTP endpoint) — designed for on-demand retrieval
via the bridge ``metrics_text`` slot.

Usage:
    from kawkab.core.observability import metrics

    metrics.counter("videos_processed").inc()
    metrics.gauge("gpu_memory_mb").set(2048.0)
    metrics.histogram("analysis_duration_s").observe(12.3)
    print(metrics.render())  # Prometheus exposition format
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# ── Metric primitives ────────────────────────────────────────────────


@dataclass
class Counter:
    """Monotonically increasing counter."""

    name: str
    value: float = 0.0
    help_text: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def reset(self) -> None:
        self.value = 0.0


@dataclass
class Gauge:
    """Point-in-time measurement."""

    name: str
    value: float = 0.0
    help_text: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    def set(self, v: float) -> None:
        self.value = v

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.value -= amount


@dataclass
class Histogram:
    """Sampled observations with configurable buckets."""

    name: str
    buckets: tuple[float, ...] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        float("inf"),
    )
    help_text: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    values: list[float] = field(default_factory=list)
    count: int = 0
    total: float = 0.0
    _min: float = float("inf")
    _max: float = -float("inf")

    def observe(self, value: float) -> None:
        self.values.append(value)
        if len(self.values) > 1000:
            self.values.pop(0)
        self.count += 1
        self.total += value
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

    def mean(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        sorted_vals = sorted(self.values)
        k = (len(sorted_vals) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

    def bucket_counts(self) -> dict[float, int]:
        counts: dict[float, int] = dict.fromkeys(self.buckets, 0)
        for v in self.values:
            for b in self.buckets:
                if v <= b:
                    counts[b] += 1
        return counts


# ── Collector ─────────────────────────────────────────────────────────


class MetricsCollector:
    """Holds all registered metrics and renders them in Prometheus format."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}

    # --- registration helpers ---

    def counter(
        self,
        name: str,
        help_text: str = "",
        labels: dict[str, str] | None = None,
    ) -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name=name, help_text=help_text, labels=labels or {})
        return self._counters[name]

    def gauge(
        self,
        name: str,
        help_text: str = "",
        labels: dict[str, str] | None = None,
    ) -> Gauge:
        if name not in self._gauges:
            self._gauges[name] = Gauge(name=name, help_text=help_text, labels=labels or {})
        return self._gauges[name]

    def histogram(
        self,
        name: str,
        help_text: str = "",
        buckets: tuple[float, ...] | None = None,
        labels: dict[str, str] | None = None,
    ) -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = Histogram(
                name=name,
                buckets=buckets or Histogram.buckets,
                help_text=help_text,
                labels=labels or {},
            )
        return self._histograms[name]

    # --- render ---

    def render(self) -> str:
        lines: list[str] = []
        for cnt in self._counters.values():
            lines.append(f"# HELP {cnt.name} {cnt.help_text}")
            lines.append(f"# TYPE {cnt.name} counter")
            labels = _fmt_labels(cnt.labels)
            lines.append(f"{cnt.name}{labels} {cnt.value}")
        for g in self._gauges.values():
            lines.append(f"# HELP {g.name} {g.help_text}")
            lines.append(f"# TYPE {g.name} gauge")
            labels = _fmt_labels(g.labels)
            lines.append(f"{g.name}{labels} {g.value}")
        for h in self._histograms.values():
            lines.append(f"# HELP {h.name} {h.help_text}")
            lines.append(f"# TYPE {h.name} histogram")
            labels = _fmt_labels(h.labels)
            bc = h.bucket_counts()
            for b, c in sorted(bc.items()):
                if b == float("inf"):
                    lines.append(f'{h.name}_bucket{labels}{{le="+Inf"}} {c}')
                else:
                    lines.append(f'{h.name}_bucket{labels}{{le="{b}"}} {c}')
            lines.append(f"{h.name}_count{labels} {h.count}")
            lines.append(f"{h.name}_sum{labels} {h.total}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "counters": {n: c.value for n, c in self._counters.items()},
            "gauges": {n: g.value for n, g in self._gauges.items()},
            "histograms": {
                n: {
                    "count": h.count,
                    "total": round(h.total, 4),
                    "mean": round(h.mean(), 4),
                    "min": round(h._min, 4) if h._min != float("inf") else 0.0,
                    "max": round(h._max, 4) if h._max != -float("inf") else 0.0,
                    "p50": round(h.percentile(0.5), 4),
                    "p95": round(h.percentile(0.95), 4),
                    "p99": round(h.percentile(0.99), 4),
                }
                for n, h in self._histograms.items()
            },
        }

    def reset(self) -> None:
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


def _fmt_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + parts + "}"


# Singleton
metrics = MetricsCollector()
