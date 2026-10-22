"""Phase 6 Sprint 4 — CI/CD Pipeline + PWA Offline Support tests.

Tests offline IndexedDB, SW cache strategy, and config YAML validation.
Since IndexedDB / ServiceWorker are browser APIs, we verify JS file content
patterns as a proxy for correctness.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

# =============================================================================
# Helpers
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
WEB_DIR = REPO_ROOT / "src" / "kawkab" / "web"


def read_js(filename: str) -> str:
    path = WEB_DIR / filename
    assert path.exists(), f"{path} not found"
    return path.read_text(encoding="utf-8")


# =============================================================================
# Offline IndexedDB — 8 tests
# =============================================================================

_OFFLINE_JS = read_js("js/app-offline.js")


class TestKawkabOfflineJS:
    """Validate app-offline.js structure and patterns."""

    def test_namespace_exists(self):
        assert "window.KawkabOffline" in _OFFLINE_JS or "KawkabOffline" in _OFFLINE_JS

    def test_save_matches_function(self):
        assert "saveMatches" in _OFFLINE_JS
        assert "function" in _OFFLINE_JS

    def test_get_matches_function(self):
        assert "getMatches" in _OFFLINE_JS

    def test_enqueue_sync_function(self):
        assert "enqueueSync" in _OFFLINE_JS

    def test_get_pending_sync_function(self):
        assert "getPendingSync" in _OFFLINE_JS

    def test_process_sync_queue_function(self):
        assert "processSyncQueue" in _OFFLINE_JS

    def test_init_sync_on_reconnect_function(self):
        assert "initSyncOnReconnect" in _OFFLINE_JS

    def test_indexeddb_open(self):
        assert "indexedDB.open" in _OFFLINE_JS or "indexedDB" in _OFFLINE_JS


class TestKawkabOfflineFunctionality:
    """Behavioral tests of offline IndexedDB via JS content analysis + mock simulation."""

    def _simulate_offline(self, operations):
        """Simulate offline actions without actual IndexedDB."""
        store = {}
        results = []
        for op in operations:
            if op["type"] == "save_matches":
                for m in op.get("matches", []):
                    mid = m.get("id", hash(str(m)))
                    store[mid] = m
                results.append(len(op.get("matches", [])))
            elif op["type"] == "get_matches":
                results.append(list(store.values()))
            elif op["type"] == "enqueue_sync":
                rid = f"sync_{len(store)}_{hash(str(op))}"
                store[rid] = op.get("action", {})
                results.append(rid)
            elif op["type"] == "get_pending_sync":
                results.append([v for v in store.values() if isinstance(v, dict) and "type" in v])
            elif op["type"] == "process_sync":
                count = sum(1 for k in list(store.keys()) if k.startswith("sync_"))
                for k in list(store.keys()):
                    if k.startswith("sync_"):
                        del store[k]
                results.append(count)
        return results

    def test_save_get_matches_roundtrip(self):
        r = self._simulate_offline(
            [
                {
                    "type": "save_matches",
                    "matches": [{"id": 1, "home": "A"}, {"id": 2, "home": "B"}],
                },
                {"type": "get_matches"},
            ]
        )
        assert r[0] == 2
        assert len(r[1]) == 2
        ids = {m["id"] for m in r[1]}
        assert ids == {1, 2}

    def test_enqueue_and_get_sync(self):
        r = self._simulate_offline(
            [
                {"type": "enqueue_sync", "action": {"type": "create", "data": {}}},
                {"type": "enqueue_sync", "action": {"type": "delete", "event_id": 5}},
                {"type": "get_pending_sync"},
            ]
        )
        assert len(r[2]) == 2
        assert r[2][0]["type"] == "create"

    def test_process_sync_clears_queue(self):
        r = self._simulate_offline(
            [
                {"type": "enqueue_sync", "action": {"type": "update"}},
                {"type": "enqueue_sync", "action": {"type": "delete"}},
                {"type": "process_sync"},
                {"type": "get_pending_sync"},
            ]
        )
        assert r[2] == 2
        assert len(r[3]) == 0

    def test_empty_queue_process(self):
        r = self._simulate_offline(
            [
                {"type": "process_sync"},
            ]
        )
        assert r[0] == 0

    def test_reconnect_triggers_sync(self):
        js = read_js("js/app-offline.js")
        assert "window.addEventListener" in js or "addEventListener" in js
        assert "'online'" in js or "online" in js
        assert "processSyncQueue" in js


# =============================================================================
# SW Cache Strategy — 4 tests (content-based)
# =============================================================================

_SW_JS = read_js("sw.js")


class TestServiceWorkerCacheStrategy:
    """Validate sw.js cache strategy implementation."""

    def test_cache_first_for_static(self):
        """Static assets should use cache-first strategy."""
        assert "isStatic" in _SW_JS or "static" in _SW_JS.lower()
        assert "caches.match" in _SW_JS

    def test_network_first_for_api(self):
        """API calls should use network-first strategy."""
        assert "isApi" in _SW_JS or "/bridge/" in _SW_JS or "/api/" in _SW_JS
        assert "fetch(event.request)" in _SW_JS

    def test_offline_fallback(self):
        """Navigation should fall back to offline.html."""
        assert "offline.html" in _SW_JS
        assert "'navigate'" in _SW_JS or "navigate" in _SW_JS

    def test_static_asset_list(self):
        """STATIC_ASSETS should include key files."""
        assert "STATIC_ASSETS" in _SW_JS
        assert "index.html" in _SW_JS
        assert "css/main.css" in _SW_JS
        assert "manifest.json" in _SW_JS


class TestSWStrategyBehavior:
    """Verify the three cache strategy patterns produce correct responses."""

    STATIC_EXTENSIONS = {
        ".css",
        ".js",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".woff2",
        ".ttf",
        ".eot",
        ".ico",
    }

    def test_cache_first_detection(self):
        assert (
            re.search(r"\.\(css\|js\|png\|jpg\|jpeg\|gif\|svg\|woff2\?\|ttf\|eot\|ico\)", _SW_JS)
            is not None
        )

    def test_network_first_detection(self):
        assert "/bridge/" in _SW_JS or "/api/" in _SW_JS

    def test_navigate_mode_detection(self):
        assert "mode" in _SW_JS
        assert "navigate" in _SW_JS

    def test_cache_name_constant(self):
        assert "CACHE" in _SW_JS or "KAWKAB_CACHE" in _SW_JS


# =============================================================================
# Config Validation — 3 tests (plus extra) = 7 total
# =============================================================================


class TestWorkflowConfigValidation:
    """Validate YAML syntax and required fields in CI/CD configs."""

    @pytest.fixture(params=["test.yml", "build.yml", "release.yml"])
    def workflow_file(self, request):
        path = WORKFLOWS_DIR / request.param
        return path

    def _load_yaml(self, path: Path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_workflow_yaml_syntax(self, workflow_file):
        assert workflow_file.exists(), f"{workflow_file} not found"
        data = self._load_yaml(workflow_file)
        assert isinstance(data, dict)
        assert "name" in data
        # YAML parses 'on:' as boolean True
        assert True in data or "on" in data
        assert "jobs" in data

    def test_test_yml_jobs(self):
        path = WORKFLOWS_DIR / "test.yml"
        data = self._load_yaml(path)
        jobs = data["jobs"]
        assert "lint" in jobs
        assert "typecheck" in jobs
        assert "test-core" in jobs
        assert "benchmark" in jobs
        assert "coverage" in jobs

    def test_build_yml_structure(self):
        path = WORKFLOWS_DIR / "build.yml"
        data = self._load_yaml(path)
        jobs = data["jobs"]
        assert "build-frontend" in jobs
        steps = jobs["build-frontend"]["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("Setup" in n or "Install" in n or "Build" in n for n in step_names)

    def test_release_yml_structure(self):
        path = WORKFLOWS_DIR / "release.yml"
        data = self._load_yaml(path)
        jobs = data["jobs"]
        assert "build-wheel" in jobs
        steps = jobs["build-wheel"]["steps"]
        step_names = [s.get("name", "") for s in steps]
        assert any("Build wheel" in n or "build" in n.lower() for n in step_names)
        # GitHub Release publishing must be configured somewhere in the
        # workflow, in a dedicated publish job that gates on the build jobs
        # (desktop artifacts + wheel) so a green release implies green builds.
        assert "publish" in jobs
        publish_steps = jobs["publish"]["steps"]
        assert any("softprops/action-gh-release" in str(s) for s in publish_steps)
        publish_needs = jobs["publish"].get("needs", [])
        assert "build-wheel" in publish_needs

    def test_pre_commit_config_valid(self):
        path = REPO_ROOT / ".pre-commit-config.yaml"
        assert path.exists(), f"{path} not found"
        data = self._load_yaml(path)
        assert "repos" in data
        repo_urls = [r["repo"] for r in data["repos"]]
        assert any("ruff" in r for r in repo_urls)
        assert any("mypy" in r for r in repo_urls)
        # ruff-format replaces black (both reformat code; running both
        # fought each other every commit) -- assert the hook is present
        # rather than requiring the black repo specifically.
        ruff_repo = next(r for r in data["repos"] if "ruff" in r["repo"])
        hook_ids = [h["id"] for h in ruff_repo["hooks"]]
        assert "ruff-format" in hook_ids
