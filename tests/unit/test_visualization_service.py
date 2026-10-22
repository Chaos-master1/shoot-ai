"""Tests for VisualizationService — chart and diagram generation.

NOTE: matplotlib and networkx are NOT installed in this environment.
We install stubs in sys.modules so the lazy imports succeed with mock objects.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import _ensure_package_loaded, install_kawkab_stubs, load_service_module

install_kawkab_stubs()

# Ensure core package is loaded before we install stubs that could shadow it
_ensure_package_loaded("kawkab")
_ensure_package_loaded("kawkab.core")
_ensure_package_loaded("kawkab.services")
_ensure_package_loaded("kawkab.services.storage")

# ---------------------------------------------------------------------------
# Stub matplotlib + networkx so lazy imports inside service methods resolve
# ---------------------------------------------------------------------------


def _install_stubs():
    # Build matplotlib.pyplot as a module with MagicMock for every attribute
    # the VisualizationService touches: subplots, savefig, close, etc.
    _mpl_pyplot = types.ModuleType("matplotlib.pyplot")
    _mpl_pyplot.subplots = MagicMock(return_value=(MagicMock(), MagicMock()))
    _mpl_pyplot.savefig = MagicMock()
    _mpl_pyplot.close = MagicMock()
    _mpl_pyplot.subplots_adjust = MagicMock()
    _mpl_pyplot.suptitle = MagicMock()

    _mpl = types.ModuleType("matplotlib")
    _mpl.pyplot = _mpl_pyplot
    sys.modules["matplotlib"] = _mpl
    sys.modules["matplotlib.pyplot"] = _mpl_pyplot

    if "daimon_runtime" not in sys.modules:
        _drm = types.ModuleType("daimon_runtime")
        _drm.setup_plot = MagicMock()
        sys.modules["daimon_runtime"] = _drm

    if "networkx" not in sys.modules:

        def _make_digraph():
            g = MagicMock()
            g.nodes = [1, 2, 3]
            return g

        _nx = types.ModuleType("networkx")
        _nx.DiGraph = _make_digraph
        _nx.spring_layout = MagicMock(return_value={})
        _nx.draw_networkx_edges = MagicMock(return_value=None)
        _nx.draw_networkx_nodes = MagicMock(return_value=None)
        _nx.draw_networkx_labels = MagicMock(return_value=None)
        sys.modules["networkx"] = _nx


# Stubs are installed at TEST TIME via the module-scoped autouse fixture
# below, not at collection time -- a permanent sys.modules stub leaks into
# every later file in the process (observed: torch._dynamo's import machinery
# raises `ValueError: networkx.__spec__ is None` when the stub shadows the
# real package during a later real-inference test). The fixture removes only
# the entries this file itself added, so a real or previously-installed
# module is left untouched.


@pytest.fixture(scope="module", autouse=True)
def _scoped_stubs():
    pre = {
        k: (k in sys.modules)
        for k in ("matplotlib", "matplotlib.pyplot", "daimon_runtime", "networkx")
    }
    _install_stubs()
    yield
    for key, existed in pre.items():
        if not existed:
            sys.modules.pop(key, None)


_mod = load_service_module("viz_test", "visualization_service.py")
VisualizationService = _mod.VisualizationService


# ===========================================================================
# Helpers
# ===========================================================================


def _make_pass_events(n: int = 5):
    return [
        {
            "type": "pass",
            "completed": True,
            "from_track_id": i % 2,
            "to_track_id": (i + 1) % 2,
            "metadata": {"end_x": 50 + i * 5, "end_y": 30 + i * 3},
        }
        for i in range(n)
    ]


def _make_player_positions():
    return {0: (20.0, 30.0), 1: (60.0, 35.0), 2: (80.0, 40.0)}


# ===========================================================================
# Tests
# ===========================================================================


class TestInit:
    def test_creates_exports_dir(self):
        svc = VisualizationService()
        assert svc._exports_dir.exists()

    def test_logs_initialization(self):
        svc = VisualizationService()
        assert "VisualizationService" in str(type(svc).__name__)


class TestGenerateHeatmap:
    @pytest.mark.asyncio
    async def test_heatmap_returns_path(self):
        svc = VisualizationService()
        positions = [(10 + i, 10 + j) for i in range(10) for j in range(10)]
        result = await svc.generate_heatmap(positions, output_name="test_hm.png")
        assert result is not None
        assert result.name == "test_hm.png"

    @pytest.mark.asyncio
    async def test_heatmap_none_on_no_positions(self):
        svc = VisualizationService()
        result = await svc.generate_heatmap([], output_name="empty.png")
        assert result is None

    @pytest.mark.asyncio
    async def test_heatmap_none_on_import_error(self):
        with patch.dict(
            "sys.modules", {"matplotlib": None, "matplotlib.pyplot": None, "daimon_runtime": None}
        ):
            svc = VisualizationService()
            result = await svc.generate_heatmap([(10.0, 20.0)], output_name="no_mpl.png")
        assert result is None


class TestGeneratePassNetwork:
    @pytest.mark.asyncio
    async def test_pass_network_returns_path(self):
        svc = VisualizationService()
        events = _make_pass_events()
        positions = _make_player_positions()
        result = await svc.generate_pass_network(events, positions, output_name="pn.png")
        assert result is not None
        assert result.name == "pn.png"

    @pytest.mark.asyncio
    async def test_pass_network_none_on_no_events(self):
        svc = VisualizationService()
        result = await svc.generate_pass_network([], {}, output_name="empty.png")
        assert result is None

    @pytest.mark.asyncio
    async def test_pass_network_none_on_import_error(self):
        with patch.dict("sys.modules", {"networkx": None, "matplotlib.pyplot": None}):
            svc = VisualizationService()
            result = await svc.generate_pass_network(
                _make_pass_events(), _make_player_positions(), output_name="no_dep.png"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_pass_network_handles_exception(self):
        svc = VisualizationService()
        with patch.object(
            sys.modules["networkx"], "DiGraph", MagicMock(side_effect=Exception("Graph error"))
        ):
            result = await svc.generate_pass_network(
                _make_pass_events(), _make_player_positions(), output_name="err.png"
            )
        assert result is None


class TestGeneratePassSonar:
    @pytest.mark.asyncio
    async def test_pass_sonar_returns_path(self):
        svc = VisualizationService()
        events = _make_pass_events()
        positions = _make_player_positions()
        result = await svc.generate_pass_sonar(events, positions, output_name="sonar.png")
        assert result is not None
        assert result.name == "sonar.png"

    @pytest.mark.asyncio
    async def test_pass_sonar_none_on_no_passes(self):
        svc = VisualizationService()
        result = await svc.generate_pass_sonar([], {}, output_name="empty.png")
        assert result is None

    @pytest.mark.asyncio
    async def test_pass_sonar_none_on_import_error(self):
        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            svc = VisualizationService()
            result = await svc.generate_pass_sonar(
                _make_pass_events(), _make_player_positions(), output_name="no_mpl.png"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_pass_sonar_handles_exception(self):
        svc = VisualizationService()
        with patch.object(
            sys.modules["matplotlib.pyplot"],
            "subplots",
            MagicMock(side_effect=Exception("Plot error")),
        ):
            result = await svc.generate_pass_sonar(
                _make_pass_events(), _make_player_positions(), output_name="err.png"
            )
        assert result is None


class TestGenerateFormationDiagram:
    @pytest.mark.asyncio
    async def test_formation_diagram_returns_path(self):
        svc = VisualizationService()
        formation = {
            "defenders": [1, 2, 3, 4],
            "midfielders": [5, 6, 7],
            "attackers": [8, 9],
            "line_height_m": 30,
            "formation": "4-3-2",
        }
        result = await svc.generate_formation_diagram(formation, output_name="form.png")
        assert result is not None
        assert result.name == "form.png"

    @pytest.mark.asyncio
    async def test_formation_diagram_handles_empty_groups(self):
        svc = VisualizationService()
        formation = {"defenders": [], "midfielders": [1, 2], "attackers": [], "formation": "0-2-0"}
        result = await svc.generate_formation_diagram(formation, output_name="min.png")
        assert result is not None

    @pytest.mark.asyncio
    async def test_formation_diagram_none_on_import_error(self):
        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            svc = VisualizationService()
            result = await svc.generate_formation_diagram(
                {"defenders": [1], "midfielders": [], "attackers": []}, output_name="no_mpl.png"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_formation_diagram_exception_handling(self):
        svc = VisualizationService()
        with patch.object(
            sys.modules["matplotlib.pyplot"],
            "subplots",
            MagicMock(side_effect=Exception("Subplots error")),
        ):
            result = await svc.generate_formation_diagram(
                {"defenders": [1], "midfielders": [], "attackers": []}, output_name="err.png"
            )
        assert result is None
