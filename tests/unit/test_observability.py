"""Tests for metrics observability (Counter, Gauge, Histogram, MetricsCollector)."""

from kawkab.core.observability import Counter, Gauge, Histogram, MetricsCollector, _fmt_labels


class TestCounter:
    def test_inc_default(self):
        c = Counter(name="test")
        c.inc()
        assert c.value == 1.0

    def test_inc_amount(self):
        c = Counter(name="test")
        c.inc(5.0)
        assert c.value == 5.0

    def test_reset(self):
        c = Counter(name="test", value=10.0)
        c.reset()
        assert c.value == 0.0


class TestGauge:
    def test_set(self):
        g = Gauge(name="mem")
        g.set(2048.0)
        assert g.value == 2048.0

    def test_inc(self):
        g = Gauge(name="mem", value=10.0)
        g.inc(5.0)
        assert g.value == 15.0

    def test_dec(self):
        g = Gauge(name="mem", value=10.0)
        g.dec(3.0)
        assert g.value == 7.0


class TestHistogram:
    def test_observe(self):
        h = Histogram(name="dur")
        h.observe(1.5)
        assert h.count == 1
        assert h.total == 1.5
        assert h._min == 1.5
        assert h._max == 1.5

    def test_mean(self):
        h = Histogram(name="dur")
        h.observe(1.0)
        h.observe(3.0)
        assert h.mean() == 2.0

    def test_mean_empty(self):
        h = Histogram(name="dur")
        assert h.mean() == 0.0

    def test_percentile_50(self):
        h = Histogram(name="dur")
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            h.observe(v)
        assert h.percentile(0.5) == 3.0

    def test_percentile_95(self):
        h = Histogram(name="dur")
        for v in range(1, 101):
            h.observe(float(v))
        assert 94.0 <= h.percentile(0.95) <= 96.0

    def test_percentile_empty(self):
        h = Histogram(name="dur")
        assert h.percentile(0.5) == 0.0

    def test_bucket_counts(self):
        h = Histogram(name="dur", buckets=(1.0, 5.0, 10.0, float("inf")))
        h.observe(0.5)
        h.observe(3.0)
        h.observe(7.0)
        h.observe(15.0)
        bc = h.bucket_counts()
        assert bc[1.0] == 1
        assert bc[5.0] == 2
        assert bc[10.0] == 3
        assert bc[float("inf")] == 4

    def test_min_max_tracking(self):
        h = Histogram(name="dur")
        h.observe(10.0)
        h.observe(1.0)
        h.observe(5.0)
        assert h._min == 1.0
        assert h._max == 10.0

    def test_default_buckets(self):
        h = Histogram(name="dur")
        assert float("inf") in h.buckets
        assert h.buckets[0] == 0.005


class TestMetricsCollector:
    def test_counter_registration(self):
        mc = MetricsCollector()
        c = mc.counter("hits", "Number of hits")
        assert c.name == "hits"
        assert c.value == 0.0
        assert mc.counter("hits") is c  # idempotent

    def test_gauge_registration(self):
        mc = MetricsCollector()
        g = mc.gauge("temp", "Temperature")
        g.set(36.5)
        assert mc.gauge("temp").value == 36.5

    def test_histogram_registration(self):
        mc = MetricsCollector()
        h = mc.histogram("latency", "Request latency")
        h.observe(0.5)
        assert mc.histogram("latency").count == 1

    def test_render_empty(self):
        mc = MetricsCollector()
        assert mc.render() == ""

    def test_render_counter(self):
        mc = MetricsCollector()
        mc.counter("hits", "Request count").inc(3)
        out = mc.render()
        assert "# HELP hits Request count" in out
        assert "# TYPE hits counter" in out
        assert "hits 3.0" in out

    def test_render_gauge(self):
        mc = MetricsCollector()
        mc.gauge("mem", "Memory").set(512)
        out = mc.render()
        assert "# TYPE mem gauge" in out
        assert "mem 512" in out

    def test_render_histogram(self):
        mc = MetricsCollector()
        h = mc.histogram("dur", "Duration", buckets=(1.0, float("inf")))
        h.observe(0.5)
        out = mc.render()
        assert "# TYPE dur histogram" in out
        assert 'dur_bucket{le="1.0"}' in out
        assert 'dur_bucket{le="+Inf"}' in out
        assert "dur_count 1" in out
        assert "dur_sum 0.5" in out

    def test_render_with_labels(self):
        mc = MetricsCollector()
        mc.counter("api_calls", "API calls", labels={"method": "GET"}).inc()
        out = mc.render()
        assert 'api_calls{method="GET"} 1.0' in out

    def test_to_dict(self):
        mc = MetricsCollector()
        h = mc.histogram("dur", buckets=(1.0, float("inf")))
        h.observe(2.0)
        d = mc.to_dict()
        assert "histograms" in d
        assert d["histograms"]["dur"]["count"] == 1
        assert d["histograms"]["dur"]["total"] == 2.0

    def test_reset(self):
        mc = MetricsCollector()
        mc.counter("hits").inc(10)
        mc.reset()
        assert mc.render() == ""


def test_fmt_labels_empty():
    assert _fmt_labels({}) == ""


def test_fmt_labels_single():
    assert _fmt_labels({"key": "val"}) == '{key="val"}'


def test_fmt_labels_multiple():
    result = _fmt_labels({"a": "1", "b": "2"})
    assert 'a="1"' in result
    assert 'b="2"' in result
    assert result.startswith("{")
    assert result.endswith("}")
