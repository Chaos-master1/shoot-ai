"""Shared test helpers and stubs + StatsBomb open-data auto-fetch."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def pytest_addoption(parser):
    # Registers --run-load, used by tests/test_load.py's skipif marker.
    # pytest only discovers pytest_addoption in conftest.py (or a plugin) --
    # test_load.py used to define this hook itself, where pytest never
    # looked for it, so config.getoption('--run-load') always raised
    # "no option named '--run-load'" instead of skipping/running as intended.
    parser.addoption(
        "--run-load",
        action="store_true",
        default=False,
        help="Run load/benchmark tests",
    )


# ── StatsBomb auto-fetch ─────────────────────────────────────────────────

_STATSBOMB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events"
_SB_MATCHES = [15946, 18245, 18252, 19975, 20378]
_SB_FETCHED = False


def _fetch_statsbomb_data() -> None:
    global _SB_FETCHED
    if _SB_FETCHED:
        return
    try:
        import httpx
    except ImportError:
        return
    gt_dir = PROJECT_ROOT / "data" / "ground_truth" / "statsbomb" / "events"
    gt_dir.mkdir(parents=True, exist_ok=True)
    existing = list(gt_dir.glob("*.json"))
    if len(existing) >= len(_SB_MATCHES):
        _SB_FETCHED = True
        return
    client = httpx.Client(timeout=30.0, follow_redirects=True)
    for mid in _SB_MATCHES:
        dest = gt_dir / f"{mid}.json"
        if dest.exists():
            continue
        try:
            resp = client.get(f"{_STATSBOMB_BASE}/{mid}.json")
            resp.raise_for_status()
            data = resp.json()
            dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("Failed to fetch StatsBomb match %s: %s", mid, exc)
    client.close()
    _SB_FETCHED = True


def pytest_configure(config: pytest.Config) -> None:
    _fetch_statsbomb_data()


# ── Stub helpers ────────────────────────────────────────────────────────────


def _make_module(name: str, path: str | None = None) -> types.ModuleType:
    m = types.ModuleType(name)
    if path is not None:
        m.__path__ = [path]
    return m


def install_loguru_stub() -> None:
    if "loguru" in sys.modules:
        return
    try:
        import loguru  # noqa: F401

        return
    except ImportError:
        pass
    loguru_stub = types.ModuleType("loguru")

    class _Logger:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def bind(self, name=""):
            return _Logger()

        def remove(self, *a, **k):
            pass

        def add(self, *a, **k):
            pass

    loguru_stub.logger = _Logger()
    sys.modules["loguru"] = loguru_stub


def install_httpx_stub() -> None:
    if "httpx" in sys.modules:
        return
    try:
        import httpx  # noqa: F401

        return
    except ImportError:
        pass
    httpx_stub = types.ModuleType("httpx")

    class _AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            return None

        async def aclose(self):
            pass

    httpx_stub.AsyncClient = _AsyncClient
    sys.modules["httpx"] = httpx_stub


def _ensure_real_package(pkg_name: str) -> types.ModuleType | None:
    """Import a real package if not already in sys.modules. Returns the module or None."""
    if pkg_name in sys.modules:
        return sys.modules[pkg_name]
    try:
        return importlib.import_module(pkg_name)
    except ImportError:
        return None


def _make_logging_stub() -> types.ModuleType:
    mod = _make_module("kawkab.core.logging")

    class _ServiceLogger:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

        def exception(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def bind(self, name=""):
            return _ServiceLogger()

    mod.get_logger = lambda name="": _ServiceLogger()
    mod.setup_logging = lambda debug=False: None
    return mod


def _make_paths_stub() -> types.ModuleType:
    mod = _make_module("kawkab.core.paths")
    # Carry a real __file__: _ensure_package_loaded re-imports any cached
    # module whose __file__ is None, which would swap this stub for a fresh
    # real module MID-RUN. Two kawkab.core.paths objects then coexist with
    # two separate _paths caches -- one reset per test by the GUI e2e
    # helpers, one stale -- and databases silently cross-test-pollute.
    mod.__file__ = str(SRC_DIR / "kawkab" / "core" / "paths.py")

    class _Paths:
        def __init__(self):
            # A UNIQUE directory per pytest process, not a fixed shared one.
            # The old fixed gettempdir()/"kawkab_test" let databases, users,
            # tags and auth artifacts accumulate ACROSS runs -- so tests
            # silently depended on leftovers from previous runs (e.g. a match
            # id 1 that a prior run had happened to create made the FK pass
            # here but fail on fresh CI).
            tmp = Path(tempfile.mkdtemp(prefix="kawkab_test_"))
            self.appdata = tmp
            self.localappdata = tmp
            self.documents = tmp
            self.videos = tmp / "videos"
            self.exports = tmp / "exports"
            self.cache = tmp / "cache"
            self.logs = tmp / "logs"
            self.models = tmp / "models"
            self.migrations = SRC_DIR / "kawkab" / "migrations"
            self.database = tmp / "kawkab.db"
            self.config_file = tmp / "config.json"
            # knowledge_base is a separate path outside tmp (real YAML rules
            # live in the repo). It must be on the CLASS, not patched onto
            # one instance afterwards: per-test resets re-instantiate _Paths,
            # and an instance-patched attribute vanishes on the fresh copy.
            self.knowledge_base = SRC_DIR / "kawkab" / "knowledge"
            # Every instance creates its own directories: with the caching
            # contract, get_paths() may build a fresh instance at any time,
            # and code that writes into exports/ etc. must not depend on an
            # earlier instance having mkdir'd them.
            for _d in (self.videos, self.exports, self.cache, self.logs, self.models):
                _d.mkdir(parents=True, exist_ok=True)

    mod.Paths = _Paths
    mod._default_paths = _Paths()
    # Mirror the real module's caching contract (get_paths() returns a
    # cached instance until someone resets `mod._paths`). The old
    # hard-wired `lambda: mod._default_paths` meant every get_paths()
    # call in the process hit ONE directory forever, so any e2e file
    # that installs these stubs leaked its DB into every later GUI e2e
    # in the same pytest run (observed as cross-file wellness/match
    # row pollution). With the cache contract, the GUI tests' existing
    # `_reset_paths` (`paths_mod._paths = None` per test) isolates them.
    # Default to the SAME instance the private helpers (_get_appdata_dir
    # etc.) return: two live instances would disagree on directories, and
    # code seeded through one path helper while serving through another
    # fails with spurious storage_unavailable (observed as RBAC 503s).
    mod._paths: types.ModuleType | None = mod._default_paths

    def _get_paths():
        if mod._paths is None:
            mod._paths = _Paths()
        return mod._paths

    mod.get_paths = _get_paths

    # Private helpers expected by some tests
    def _get_appdata_dir():
        return mod._default_paths.appdata

    def _get_localappdata_dir():
        return mod._default_paths.localappdata

    def _get_documents_dir():
        return mod._default_paths.documents

    mod._get_appdata_dir = _get_appdata_dir
    mod._get_localappdata_dir = _get_localappdata_dir
    mod._get_documents_dir = _get_documents_dir

    return mod


def _make_migration_manager_stub() -> types.ModuleType:
    mod = _make_module("kawkab.core.migration_manager")

    class _MigrationManager:
        def __init__(self, db_path, migrations_dir):
            pass

        def migrate(self):
            pass

    mod.MigrationManager = _MigrationManager
    return mod


def install_kawkab_stubs() -> None:
    # Isolate from PostgreSQL — all tests use SQLite by default.
    # Opt in to the real-Postgres integration tests by setting BOTH
    # KAWKAB_DB_URL and KAWKAB_PG_TESTS=1 (see test_postgres_storage.py).
    if os.environ.get("KAWKAB_PG_TESTS") != "1":
        os.environ.pop("KAWKAB_DB_URL", None)

    install_loguru_stub()
    install_httpx_stub()

    # Stub core modules FIRST so real imports see them. Build the paths stub
    # only when it is not already installed: _make_paths_stub() creates a
    # fresh per-process directory, so calling it eagerly here would leak an
    # unused temp dir on every repeat invocation (setdefault discards it, but
    # the mkdir'd tree would still exist).
    if "kawkab.core.logging" not in sys.modules:
        sys.modules["kawkab.core.logging"] = _make_logging_stub()
    if "kawkab.core.paths" not in sys.modules:
        sys.modules["kawkab.core.paths"] = _make_paths_stub()

    # Real packages — import normally instead of stubbing
    for _pkg in ["kawkab", "kawkab.core", "kawkab.services"]:
        _ensure_real_package(_pkg)

    _kawkab = sys.modules.get("kawkab")
    if _kawkab is not None:
        _kawkab.core = sys.modules.get("kawkab.core")
        _kawkab.services = sys.modules.get("kawkab.services")
        _kawkab.core.paths = sys.modules.get("kawkab.core.paths")

    _stg_dir = SRC_DIR / "kawkab" / "services" / "storage"
    if _stg_dir.exists() and "kawkab.services.storage" not in sys.modules:
        stg_pkg = _make_module(
            "kawkab.services.storage", str(SRC_DIR / "kawkab" / "services" / "storage")
        )
        sys.modules["kawkab.services.storage"] = stg_pkg
        for _sf in sorted(_stg_dir.glob("*.py")):
            if _sf.name == "__init__.py":
                continue
            _smn = f"kawkab.services.storage.{_sf.stem}"
            if _smn not in sys.modules:
                _spec = importlib.util.spec_from_file_location(_smn, str(_sf))
                if _spec and _spec.loader:
                    _mod = importlib.util.module_from_spec(_spec)
                    sys.modules[_smn] = _mod
                    _spec.loader.exec_module(_mod)


# ── Package loading helpers ─────────────────────────────────────────────────


def _ensure_package_loaded(package_name: str, skip_import: frozenset[str] = frozenset()) -> None:
    parts = package_name.split(".")
    for i in range(1, len(parts) + 1):
        prefix = ".".join(parts[:i])
        if prefix in skip_import:
            if prefix not in sys.modules:
                sys.modules[prefix] = _make_module(prefix, str(SRC_DIR / prefix.replace(".", "/")))
            continue
        mod = sys.modules.get(prefix)
        if mod is not None:
            if not hasattr(mod, "__file__") or mod.__file__ is None:
                del sys.modules[prefix]
                try:
                    importlib.import_module(prefix)
                except ImportError:
                    if prefix not in sys.modules:
                        sys.modules[prefix] = _make_module(
                            prefix, str(SRC_DIR / prefix.replace(".", "/"))
                        )
        else:
            try:
                importlib.import_module(prefix)
            except ImportError:
                if prefix not in sys.modules:
                    sys.modules[prefix] = _make_module(
                        prefix, str(SRC_DIR / prefix.replace(".", "/"))
                    )


# ── Service module loader ───────────────────────────────────────────────────


def load_service_module(
    module_name: str, file_basename: str, subdir: str = "services"
) -> types.ModuleType:
    _ensure_package_loaded("kawkab")
    _ensure_package_loaded("kawkab.core")
    _ensure_package_loaded("kawkab.core.paths", skip_import=frozenset({"kawkab.core.paths"}))
    _ensure_package_loaded("kawkab.services", skip_import=frozenset({"kawkab.services"}))
    _ensure_package_loaded("kawkab.services.storage", skip_import=frozenset({"kawkab.services"}))
    if subdir == "services":
        file_path = SRC_DIR / "kawkab" / "services" / file_basename
    elif subdir == "utils":
        file_path = SRC_DIR / "kawkab" / "utils" / file_basename
    elif subdir == "i18n":
        file_path = SRC_DIR / "kawkab" / "i18n" / file_basename
    elif subdir == "core":
        file_path = SRC_DIR / "kawkab" / "core" / file_basename
    else:
        raise ValueError(f"Unknown subdir: {subdir}")
    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
