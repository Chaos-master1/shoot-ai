"""Tests for the optional services: MuJoCo ball, FluidX3D, RoboFlow Sports."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_mj = load_service_module("mj_test", "mujoco_ball_service.py")
MuJoCoBallService = _mj.MuJoCoBallService
TrajectoryResult = _mj.TrajectoryResult

_fx = load_service_module("fx_test", "fluidx3d_service.py")
FluidX3DService = _fx.FluidX3DService

_rf = load_service_module("rf_test", "roboflow_sports_service.py")
RoboflowSportsService = _rf.RoboflowSportsService

import pytest


@pytest.fixture
def mujoco_svc() -> MuJoCoBallService:
    return MuJoCoBallService()


@pytest.fixture
def fluidx3d_svc() -> FluidX3DService:
    return FluidX3DService()


@pytest.fixture
def roboflow_svc() -> RoboflowSportsService:
    return RoboflowSportsService()


class TestMuJoCoService:
    def test_available(self, mujoco_svc: MuJoCoBallService) -> None:
        assert mujoco_svc.available

    def test_uses_mujoco_false(self, mujoco_svc: MuJoCoBallService) -> None:
        assert not mujoco_svc.uses_mujoco

    def test_simulate_basic(self, mujoco_svc: MuJoCoBallService) -> None:
        result = asyncio.run(mujoco_svc.simulate(initial_speed=20, launch_angle_deg=15, spin_rps=2))
        assert isinstance(result, TrajectoryResult)
        assert result.method in ("mujoco", "analytical")
        assert len(result.points) > 0

    def test_simulate_zero_speed(self, mujoco_svc: MuJoCoBallService) -> None:
        result = asyncio.run(mujoco_svc.simulate(initial_speed=0, launch_angle_deg=0))
        assert len(result.points) >= 0

    def test_preset_setpieces(self, mujoco_svc: MuJoCoBallService) -> None:
        presets = mujoco_svc.get_preset_setpieces()
        assert len(presets) >= 3
        for p in presets:
            assert "name" in p
            assert "initial_speed" in p
            assert "launch_angle_deg" in p
            assert "spin_rps" in p

    def test_landing_x_affected_by_speed(self, mujoco_svc: MuJoCoBallService) -> None:
        slow = asyncio.run(mujoco_svc.simulate(initial_speed=10, launch_angle_deg=30))
        fast = asyncio.run(mujoco_svc.simulate(initial_speed=30, launch_angle_deg=30))
        assert fast.landing_x > slow.landing_x


class TestFluidX3DService:
    def test_available_no_binary(self, fluidx3d_svc: FluidX3DService) -> None:
        assert not fluidx3d_svc.available

    def test_license_notice(self, fluidx3d_svc: FluidX3DService) -> None:
        notice = fluidx3d_svc.license_notice
        assert "non-commercial" in notice.lower()

    def test_simulate_no_binary(self, fluidx3d_svc: FluidX3DService) -> None:
        result = asyncio.run(fluidx3d_svc.simulate_ball_aerodynamics(wind_speed=20))
        assert not result.success
        assert result.error is not None


class TestRoboflowSportsService:
    def test_create_ball_annotator_without_module(
        self, roboflow_svc: RoboflowSportsService
    ) -> None:
        result = roboflow_svc.create_ball_annotator(radius=10)
        assert result is None

    def test_create_ball_tracker_without_module(self, roboflow_svc: RoboflowSportsService) -> None:
        result = roboflow_svc.create_ball_tracker()
        assert result is None
