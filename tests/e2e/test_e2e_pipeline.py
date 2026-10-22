"""End-to-end integration scaffold for Kawkab AI.

Tests key cross-module workflows without requiring a GPU or real video:
- Metrics collection (observability)
- Profiler integration
- Bridge slot availability
- .po → .json compilation
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


def _get_mc():
    from kawkab.core.observability import MetricsCollector

    return MetricsCollector


# ── A2: observability ────────────────────────────────────────────────


class TestMetricsCollection:
    def test_counter_inc(self):
        MetricsCollector = _get_mc()
        m = MetricsCollector()
        c = m.counter("test_calls", help_text="Test counter")
        assert c.value == 0.0
        c.inc()
        assert c.value == 1.0
        c.inc(5)
        assert c.value == 6.0

    def test_gauge_set(self):
        MetricsCollector = _get_mc()
        m = MetricsCollector()
        g = m.gauge("test_gauge", help_text="Test gauge")
        g.set(42.0)
        assert g.value == 42.0
        g.inc(1)
        assert g.value == 43.0
        g.dec(3)
        assert g.value == 40.0

    def test_histogram_observe(self):
        MetricsCollector = _get_mc()
        m = MetricsCollector()
        h = m.histogram("test_hist", help_text="Test hist")
        h.observe(0.5)
        h.observe(1.0)
        h.observe(1.5)
        assert h.count == 3
        assert h.total == 3.0
        assert h.mean() == 1.0

    def test_render_prometheus_format(self):
        MetricsCollector = _get_mc()
        m = MetricsCollector()
        m.counter("videos_total", help_text="Total videos").inc(3)
        m.gauge("mem_mb", help_text="Memory").set(512)
        output = m.render()
        assert "# HELP videos_total Total videos" in output
        assert "# TYPE videos_total counter" in output
        assert "videos_total " in output
        assert "# HELP mem_mb Memory" in output
        assert "# TYPE mem_mb gauge" in output

    def test_to_dict(self):
        MetricsCollector = _get_mc()
        m = MetricsCollector()
        m.counter("c").inc(2)
        m.gauge("g").set(1)
        m.histogram("h").observe(3.0)
        d = m.to_dict()
        assert d["counters"]["c"] == 2.0
        assert d["gauges"]["g"] == 1.0
        assert "h" in d["histograms"]

    def test_reset(self):
        MetricsCollector = _get_mc()
        m = MetricsCollector()
        m.counter("c").inc(1)
        m.reset()
        assert len(m._counters) == 0

    def test_singleton_metrics(self):
        from kawkab.core.observability import metrics as _m1
        from kawkab.core.observability import metrics as _m2

        assert _m1 is _m2


# ── A1: Profiler integration ─────────────────────────────────────────


class TestProfilerIntegration:
    def test_profiler_basic_flow(self):
        from kawkab.utils.profiler import Profiler

        p = Profiler()
        p.start()
        p.begin("stage1")
        p.end("stage1")
        p.stop()
        r = p.report()
        assert r.total_s >= 0
        assert len(r.stages) == 1
        assert r.stages[0].name == "stage1"
        assert r.stages[0].count == 1

    def test_profiler_multiple_stages(self):
        from kawkab.utils.profiler import Profiler

        p = Profiler()
        p.start()
        for s in ["a", "b", "c"]:
            p.begin(s)
            p.end(s)
        p.stop()
        r = p.report()
        assert len(r.stages) == 3

    def test_profiler_report_dict(self):
        from kawkab.utils.profiler import Profiler

        p = Profiler()
        p.start()
        p.begin("x")
        p.end("x")
        p.stop()
        d = p.report().to_dict()
        assert "total_s" in d
        assert "stages" in d
        assert "bottlenecks" in d

    def test_profiler_context_manager(self):
        from kawkab.utils.profiler import Profiler

        p = Profiler()
        p.start()
        with p.stage("ctx"):
            pass
        p.stop()
        assert p.get_stage("ctx") is not None
        assert p.get_stage("ctx").count == 1


# ── A5: Bridge slot contract ─────────────────────────────────────────


BRIDGE_SOURCE = (
    Path(__file__).resolve().parent.parent.parent / "src" / "kawkab" / "ui" / "bridge.py"
)


class TestBridgeSlots:
    """Verify bridge slots exist via AST inspection (no import needed)."""

    BRIDGE_SLOTS = [
        "profiler_status",
        "profiler_reset",
        "metrics_text",
        "check_llm_availability",
        "get_gpu_info",
    ]

    def _get_async_methods(self) -> set[str]:
        with open(BRIDGE_SOURCE, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        methods: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
                methods.add(node.name)
        return methods

    def test_required_slots_exist(self):
        methods = self._get_async_methods()
        for slot in self.BRIDGE_SLOTS:
            assert slot in methods, f"Missing bridge slot method: {slot}"

    def test_profiler_status_returns_json_dict(self):
        methods = self._get_async_methods()
        assert "profiler_status" in methods

    def test_profiler_reset_returns_json_dict(self):
        methods = self._get_async_methods()
        assert "profiler_reset" in methods

    def test_metrics_text_returns_string(self):
        methods = self._get_async_methods()
        assert "metrics_text" in methods


# ── A3: .po → .json compilation ─────────────────────────────────────


class TestPoCompilation:
    def test_compile_script_runs(self):
        locales_dir = Path(__file__).resolve().parent.parent.parent / "locales"
        en_json = locales_dir / "en.json"
        ar_json = locales_dir / "ar.json"
        assert en_json.exists(), "en.json not found — run scripts/compile_i18n.py"
        assert ar_json.exists(), "ar.json not found — run scripts/compile_i18n.py"

        en_data = json.loads(en_json.read_text(encoding="utf-8"))
        ar_data = json.loads(ar_json.read_text(encoding="utf-8"))

        assert "uploadTitle" in en_data
        assert en_data["uploadTitle"] == "📹 Upload Match Video"
        assert "uploadTitle" in ar_data
        assert len(en_data) >= 70
        assert len(ar_data) >= 70

    def test_all_keys_have_arabic_translation(self):
        locales_dir = Path(__file__).resolve().parent.parent.parent / "locales"
        en = json.loads((locales_dir / "en.json").read_text(encoding="utf-8"))
        ar = json.loads((locales_dir / "ar.json").read_text(encoding="utf-8"))
        missing = [k for k in en if k not in ar]
        assert not missing, f"Keys missing from ar.json: {missing}"


# ── A4: CI coverage config ──────────────────────────────────────────


class TestCoverageConfig:
    def test_coverage_fail_under_matches_pyproject(self):
        # test.yml's coverage job runs plain `pytest --cov=src/kawkab`, with
        # no --cov-fail-under flag of its own -- pytest-cov reads the
        # threshold from pyproject.toml's [tool.coverage.report] fail_under
        # automatically, so there's nothing in test.yml's own text to check
        # a specific number against. What *is* worth asserting: the
        # threshold is a real, deliberately-chosen value, not silently
        # reverted to pytest-cov's un-set default (no failure at all).
        pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        assert "fail_under = 65" in text, (
            "Coverage threshold in pyproject.toml should be 65 "
            "(the real measured baseline as of the 2026-07-30 audit -- "
            "see CLAUDE.md)"
        )

    def test_coverage_job_does_not_ignore_fixed_audio_service_tests(self):
        # test_audio_service.py's 10 tests were genuinely failing (stale
        # API) before the 2026-07-30 audit fixed them; test.yml's coverage
        # job carried its own --ignore for it independently of ci.yml's
        # unit job (already cleaned up), so this specific flag survived
        # that pass. Now that the tests pass, ignoring them here just
        # under-counts coverage for no reason.
        ci = Path(__file__).resolve().parent.parent.parent / ".github" / "workflows" / "test.yml"
        text = ci.read_text(encoding="utf-8")
        assert "--ignore=tests/unit/test_audio_service.py" not in text
