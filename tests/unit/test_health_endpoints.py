from __future__ import annotations

import os

os.environ["KAWKAB_RATE_LIMIT_DISABLE"] = "1"
os.environ.setdefault("KAWKAB_JWT_SECRET", "test-secret-for-health-tests-32chars-min")

from fastapi.testclient import TestClient

from kawkab.cloud.server import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "version" in data
    assert "uptime_s" in data
    assert "python" in data


def test_health_ready():
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert "ready" in data


def test_health_live():
    resp = client.get("/health/live")
    assert resp.status_code == 200
    data = resp.json()
    assert data["alive"] is True


def test_metrics_endpoint():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "uptime_s" in data
    assert "python_version" in data
    assert "gc_count" in data
    assert "gc_threshold" in data


def test_metrics_values_are_reasonable():
    resp = client.get("/metrics")
    data = resp.json()
    assert data["uptime_s"] >= 0
    assert len(data["gc_threshold"]) == 3
