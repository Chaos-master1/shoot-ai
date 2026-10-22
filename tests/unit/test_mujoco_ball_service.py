"""Tests for MuJoCoBallService — ball trajectory simulation."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_mod = load_service_module("mujoco_ball_test", "mujoco_ball_service.py")
MuJoCoBallService = _mod.MuJoCoBallService
TrajectoryPoint = _mod.TrajectoryPoint
TrajectoryResult = _mod.TrajectoryResult


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def svc():
    return MuJoCoBallService()


# ===========================================================================
# Init — default fallback (mujoco not available)
# ===========================================================================


class TestInit:
    def test_available_property(self, svc):
        assert svc.available is True

    def test_uses_mujoco_false_by_default(self, svc):
        assert svc.uses_mujoco is False

    def test_mujoco_attribute_is_none(self, svc):
        assert svc._mujoco is None

    def test_model_and_data_are_none(self, svc):
        assert svc._model is None
        assert svc._data is None


# ===========================================================================
# Init — mujoco available (mocked)
# ===========================================================================


class TestInitWithMujoco:
    def test_mujoco_loads_successfully(self, monkeypatch):
        mock_mujoco = MagicMock()
        mock_mujoco.MjModel.from_xml_string.return_value = MagicMock()
        mock_mujoco.MjData.return_value = MagicMock()
        # Register mock in sys.modules so the import inside _try_load resolves
        monkeypatch.setitem(sys.modules, "mujoco", mock_mujoco)

        svc = MuJoCoBallService()
        assert svc.uses_mujoco is True
        assert svc._mujoco is mock_mujoco


# ===========================================================================
# Presets
# ===========================================================================


class TestPresets:
    def test_get_preset_setpieces_returns_list(self, svc):
        presets = svc.get_preset_setpieces()
        assert isinstance(presets, list)
        assert len(presets) == 4

    def test_each_preset_has_required_keys(self, svc):
        for p in svc.get_preset_setpieces():
            assert "name" in p
            assert "initial_speed" in p
            assert "launch_angle_deg" in p
            assert "spin_rps" in p
            assert "direction_deg" in p

    def test_preset_values_positive(self, svc):
        for p in svc.get_preset_setpieces():
            assert p["initial_speed"] > 0


# ===========================================================================
# TrajectoryResult dataclass
# ===========================================================================


class TestTrajectoryResult:
    def test_default_method(self):
        t = TrajectoryResult(
            points=[], landing_x=0, landing_y=0, max_height=0, duration_s=0, final_speed_mps=0
        )
        assert t.method == "analytical"

    def test_trajectory_point_fields(self):
        p = TrajectoryPoint(t=1.0, x=10.0, y=5.0, z=3.0)
        assert p.t == 1.0
        assert p.x == 10.0
        assert p.y == 5.0
        assert p.z == 3.0


# ===========================================================================
# Simulate — analytical fallback
# ===========================================================================


class TestSimulateAnalytical:
    @pytest.mark.asyncio
    async def test_default_params_returns_result(self, svc):
        result = await svc.simulate()
        assert isinstance(result, TrajectoryResult)
        assert len(result.points) > 0
        assert result.method == "analytical"

    @pytest.mark.asyncio
    async def test_landing_position_different_with_spin(self, svc):
        no_spin = await svc.simulate(initial_speed=25.0, launch_angle_deg=18.0, spin_rps=0.0)
        with_spin = await svc.simulate(initial_speed=25.0, launch_angle_deg=18.0, spin_rps=5.0)
        assert no_spin.landing_x != with_spin.landing_x or no_spin.landing_y != with_spin.landing_y

    @pytest.mark.asyncio
    async def test_max_height_increases_with_angle(self, svc):
        low = await svc.simulate(initial_speed=25.0, launch_angle_deg=10.0)
        high = await svc.simulate(initial_speed=25.0, launch_angle_deg=30.0)
        assert high.max_height > low.max_height

    @pytest.mark.asyncio
    async def test_positive_direction_shifts_landing_y(self, svc):
        straight = await svc.simulate(direction_deg=0.0)
        angled = await svc.simulate(direction_deg=15.0)
        # With positive direction, the ball should deviate in y
        assert angled.landing_y != straight.landing_y

    @pytest.mark.asyncio
    async def test_final_speed_less_than_initial(self, svc):
        result = await svc.simulate(initial_speed=25.0, duration_s=2.0)
        assert result.final_speed_mps < 25.0

    @pytest.mark.asyncio
    async def test_duration_positive(self, svc):
        result = await svc.simulate(duration_s=1.0)
        assert result.duration_s > 0

    @pytest.mark.asyncio
    async def test_points_have_increasing_time(self, svc):
        result = await svc.simulate()
        times = [p.t for p in result.points]
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    @pytest.mark.asyncio
    async def test_custom_params(self, svc):
        result = await svc.simulate(
            initial_speed=30.0,
            launch_angle_deg=25.0,
            spin_rps=3.0,
            direction_deg=10.0,
            duration_s=3.0,
            drag_coeff=0.3,
            magnus_coeff=0.0005,
            ball_mass=0.45,
            ball_radius=0.12,
        )
        assert result.method == "analytical"
        assert len(result.points) > 0


# ===========================================================================
# Simulate — via mujoco (mocked)
# ===========================================================================


class TestSimulateMujoco:
    @pytest.mark.asyncio
    async def test_mujoco_simulation_returns_result(self, monkeypatch):
        mock_mujoco = MagicMock()
        model = MagicMock()
        model.opt.timestep = 0.005
        data = MagicMock()
        data.qpos = [0.0, 0.0, 1.0]
        data.qvel = [10.0, 0.0, -2.0]
        data.qfrc_applied = [0.0, 0.0, 0.0]

        mock_mujoco.MjModel.from_xml_string.return_value = model
        mock_mujoco.MjData.return_value = data

        monkeypatch.setitem(sys.modules, "mujoco", mock_mujoco)

        svc = MuJoCoBallService()
        result = await svc.simulate(initial_speed=20.0, launch_angle_deg=15.0, duration_s=0.5)
        assert result.method == "mujoco"
        assert len(result.points) > 0

    @pytest.mark.asyncio
    async def test_mujoco_failure_falls_back_to_analytical(self, monkeypatch):
        mock_mujoco = MagicMock()
        mock_mujoco.MjModel.from_xml_string.return_value = MagicMock()
        mock_mujoco.MjData.return_value = MagicMock()
        mock_mujoco.mj_resetData.side_effect = RuntimeError("mj crash")

        monkeypatch.setitem(sys.modules, "mujoco", mock_mujoco)

        svc = MuJoCoBallService()
        result = await svc.simulate(initial_speed=20.0, launch_angle_deg=15.0)
        assert result.method == "analytical"
        assert len(result.points) > 0
