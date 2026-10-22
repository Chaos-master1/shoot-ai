"""v0.13.2 follow-ups: recruitment services must not write into the source tree.

Three services (OpponentDatabaseService, ScoutingNetworkService,
TransfermarktIntegrationService) used to persist JSON under
``src/kawkab/../../data`` — i.e. the repository itself — because they joined
``os.path.dirname(__file__)`` with a relative data path. Running the GUI e2e
audit produced untracked "Audit Rivals"/"Audit Scout" artifacts inside
``src/data/``. They now resolve through ``kawkab.core.paths.get_paths()``
(per-user appdata), which the GUI e2e already isolates via HOME/XDG env vars,
and accept an explicit path override for tests.

What is pinned here:
- default resolution never lands inside the source tree (the regression that
  shipped test-run artifacts into the repo), whatever Paths implementation
  the unit-stub environment has active;
- the explicit override is honored and gives a full persistence round trip;
- data written through one instance is visible to a fresh instance (the
  handler rebuilds these lazily per bridge session).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.services.opponent_database_service import (  # noqa: E402
    OpponentDatabaseService,
)
from kawkab.services.scouting_network_service import (  # noqa: E402
    ScoutingNetworkService,
)
from kawkab.services.transfermarkt_integration_service import (  # noqa: E402
    TransfermarktIntegrationService,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_TREE = REPO_ROOT / "src"


@pytest.fixture()
def isolated_appdata(tmp_path, monkeypatch):
    """Point the active Paths implementation at tmp_path, e2e-style.

    install_kawkab_stubs() replaces kawkab.core.paths with a static stub,
    but other test files may have re-imported the real module into
    sys.modules (or copied real attrs onto the stub), so reset every
    candidate singleton before the service resolves its default dir.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path / "xdg-docs"))
    for name, mod_obj in list(sys.modules.items()):
        if name == "kawkab.core.paths" and mod_obj is not None and hasattr(mod_obj, "_paths"):
            monkeypatch.setattr(mod_obj, "_paths", None)
    yield tmp_path


def _assert_outside_src(path: Path, what: str) -> None:
    resolved = path.resolve()
    assert SRC_TREE not in resolved.parents and resolved != SRC_TREE, (
        f"{what} wrote inside the source tree: {resolved}"
    )


def test_opponent_profiles_default_outside_src_and_roundtrip(isolated_appdata):
    svc = OpponentDatabaseService()
    created = svc.create_profile("Rivals FC", "Ligue 1", "France")
    assert created.get("id"), created

    written = Path(svc._data_dir) / "profiles.json"
    _assert_outside_src(written, "OpponentDatabaseService")
    assert json.loads(written.read_text(encoding="utf-8"))[0]["team_name"] == "Rivals FC"

    # Fresh instance (the handler rebuilds lazily per session) sees the data.
    assert OpponentDatabaseService().get_profile(created["id"]) is not None


def test_scouting_network_default_outside_src_and_roundtrip(isolated_appdata):
    svc = ScoutingNetworkService()
    added = svc.add_player(name="Audit Scout", position="CM", rating=7.5)
    assert added.get("id"), added

    written = Path(svc._data_file)
    _assert_outside_src(written, "ScoutingNetworkService")
    assert json.loads(written.read_text(encoding="utf-8"))[0]["name"] == "Audit Scout"

    assert ScoutingNetworkService().get_player(added["id"]) is not None


def test_transfermarkt_cache_default_outside_src(isolated_appdata):
    svc = TransfermarktIntegrationService()
    results = svc.search_player("mbappe")  # cache-miss path saves the index
    assert isinstance(results, list)

    written = Path(svc._cache_dir) / "_index.json"
    _assert_outside_src(written, "TransfermarktIntegrationService")
    assert "search:mbappe" in json.loads(written.read_text(encoding="utf-8"))


def test_explicit_override_honored_with_full_roundtrip(tmp_path):
    """The ctor override makes the storage location fully deterministic."""
    opponents_dir = tmp_path / "opp"
    svc = OpponentDatabaseService(data_dir=str(opponents_dir))
    created = svc.create_profile("Override FC", "", "")
    assert (opponents_dir / "profiles.json").exists()
    assert svc.get_profile(created["id"]) is not None
    assert OpponentDatabaseService(data_dir=str(opponents_dir)).get_profile(created["id"])

    net_file = tmp_path / "net.json"
    svc2 = ScoutingNetworkService(data_file=str(net_file))
    added = svc2.add_player(name="Override Scout", position="ST", rating=6.0)
    assert net_file.exists()
    assert ScoutingNetworkService(data_file=str(net_file)).get_player(added["id"])

    tm_dir = tmp_path / "tm"
    svc3 = TransfermarktIntegrationService(cache_dir=str(tm_dir))
    svc3.search_player("demo")
    assert (tm_dir / "_index.json").exists()
