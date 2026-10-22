"""Tests for performance profiler."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_p = load_service_module("prof_test", "profiler.py", subdir="utils")
Profiler = _p.Profiler
ProfileReport = _p.ProfileReport
StageStats = _p.StageStats

import pytest


class TestProfiler:
    def test_empty_report(self) -> None:
        p = Profiler()
        p.start()
        p.stop()
        report = p.report()
        assert isinstance(report, ProfileReport)
        assert report.stages == []
        assert "No stages recorded" in report.notes[0]

    def test_single_stage(self) -> None:
        p = Profiler()
        p.start()
        p.begin("cv")
        time.sleep(0.03)
        p.end("cv")
        p.stop()
        report = p.report()
        assert len(report.stages) == 1
        assert report.stages[0].name == "cv"
        assert report.stages[0].count == 1
        assert isinstance(report.stages[0].total_s, float)

    def test_multiple_stages(self) -> None:
        p = Profiler()
        p.start()
        p.begin("cv")
        time.sleep(0.005)
        p.end("cv")
        p.begin("analysis")
        time.sleep(0.005)
        p.end("analysis")
        p.stop()
        report = p.report()
        assert len(report.stages) == 2

    def test_stage_context_manager(self) -> None:
        p = Profiler()
        p.start()
        with p.stage("cv"):
            time.sleep(0.005)
        p.stop()
        report = p.report()
        assert report.stages[0].name == "cv"

    def test_stage_count(self) -> None:
        p = Profiler()
        p.start()
        for _ in range(5):
            p.begin("cv")
            time.sleep(0.001)
            p.end("cv")
        p.stop()
        report = p.report()
        assert report.stages[0].count == 5

    def test_bottleneck_identification(self) -> None:
        p = Profiler()
        p.start()
        p.begin("fast")
        time.sleep(0.001)
        p.end("fast")
        p.begin("slow")
        time.sleep(0.05)
        p.end("slow")
        p.stop()
        report = p.report()
        assert any("slow" in b for b in report.bottlenecks)

    def test_recording_explicit_elapsed(self) -> None:
        p = Profiler()
        p.start()
        p.record("cv", 0.123)
        p.record("cv", 0.456)
        p.stop()
        report = p.report()
        s = report.stages[0]
        assert s.count == 2
        assert s.total_s == pytest.approx(0.579, 0.01)

    def test_stage_stats_percentile(self) -> None:
        s = StageStats(name="t")
        for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            s.record(v)
        assert s.percentile(0.5) == pytest.approx(0.55, 0.05)
        assert s.percentile(0.95) >= 0.9
        assert s.min_s == 0.1
        assert s.max_s == 1.0

    def test_stage_stats_to_dict(self) -> None:
        s = StageStats(name="t")
        s.record(0.5)
        d = s.to_dict()
        assert d["name"] == "t"
        assert d["count"] == 1
        assert d["total_s"] == 0.5

    def test_nested_begin_ignored(self) -> None:
        p = Profiler()
        p.start()
        p.begin("cv")
        p.begin("cv")
        p.end("cv")
        p.stop()
        report = p.report()
        assert len(report.stages) == 1
        assert report.stages[0].count == 1

    def test_end_without_begin_returns_zero(self) -> None:
        p = Profiler()
        result = p.end("never_began")
        assert result == 0.0

    def test_reset_clears_state(self) -> None:
        p = Profiler()
        p.start()
        p.begin("cv")
        p.end("cv")
        p.reset()
        p.start()
        p.stop()
        report = p.report()
        assert report.stages == []

    def test_get_stage(self) -> None:
        p = Profiler()
        p.start()
        p.begin("cv")
        time.sleep(0.001)
        p.end("cv")
        p.stop()
        s = p.get_stage("cv")
        assert s is not None
        assert s.count == 1
        assert p.get_stage("nonexistent") is None

    def test_report_to_dict(self) -> None:
        p = Profiler()
        p.start()
        p.begin("cv")
        time.sleep(0.001)
        p.end("cv")
        p.stop()
        d = p.report().to_dict()
        assert "total_s" in d
        assert "stages" in d
        assert "bottlenecks" in d
        assert isinstance(d["stages"], list)

    def test_report_str(self) -> None:
        p = Profiler()
        p.start()
        p.begin("cv")
        time.sleep(0.001)
        p.end("cv")
        p.stop()
        s = str(p.report())
        assert "Profile report" in s
        assert "cv" in s
