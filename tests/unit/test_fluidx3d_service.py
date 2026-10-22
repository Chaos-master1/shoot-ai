"""Tests for FluidX3DService — subprocess CFD wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_mod = load_service_module("fluidx3d_test", "fluidx3d_service.py")
FluidX3DService = _mod.FluidX3DService
CfdResult = _mod.CfdResult

# Reference to the loaded module for monkeypatching
_fluidx3d_mod = _mod


# ===========================================================================
# Init & binary checking
# ===========================================================================


class TestInit:
    def test_no_binary_path(self):
        svc = FluidX3DService()
        assert svc.available is False
        assert svc._binary_path is None

    def test_binary_path_not_found(self):
        svc = FluidX3DService(binary_path="/nonexistent/fluidx3d")
        assert svc.available is False

    def test_binary_path_not_executable(self, monkeypatch):
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr("os.access", lambda path, mode: False)
        svc = FluidX3DService(binary_path="/fake/fluidx3d")
        assert svc.available is False

    def test_binary_available(self, monkeypatch):
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr("os.access", lambda path, mode: True)
        svc = FluidX3DService(binary_path="/fake/fluidx3d")
        assert svc.available is True


# ===========================================================================
# Properties
# ===========================================================================


class TestProperties:
    def test_license_notice(self):
        svc = FluidX3DService()
        assert "non-commercial" in svc.license_notice
        assert "FluidX3D" in svc.license_notice


# ===========================================================================
# Simulate — not available
# ===========================================================================


class TestSimulateNotAvailable:
    @pytest.mark.asyncio
    async def test_returns_error_result(self):
        svc = FluidX3DService()
        result = await svc.simulate_ball_aerodynamics()
        assert result.success is False
        assert result.method == "none"
        assert "not configured" in result.notes
        assert result.velocity_field is None
        assert result.pressure_field is None
        assert result.drag_coefficient is None
        assert result.lift_coefficient is None

    @pytest.mark.asyncio
    async def test_returns_cfd_result_type(self):
        svc = FluidX3DService()
        result = await svc.simulate_ball_aerodynamics(
            ball_radius=0.11, wind_speed=5.0, spin_rps=2.0
        )
        assert isinstance(result, CfdResult)
        assert result.error is not None


# ===========================================================================
# Simulate — available (mocked subprocess)
# ===========================================================================


class TestSimulateAvailable:
    @pytest.fixture
    def available_svc(self, monkeypatch):
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr("os.access", lambda path, mode: True)
        return FluidX3DService(binary_path="/fake/fluidx3d")

    @pytest.mark.asyncio
    async def test_success(self, available_svc, monkeypatch):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"sim output", b"")
        monkeypatch.setattr(
            _fluidx3d_mod.asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=mock_proc),
        )
        result = await available_svc.simulate_ball_aerodynamics()
        assert result.success is True
        assert result.method == "fluidx3d"
        assert "complete" in result.notes
        assert result.error is None

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, available_svc, monkeypatch):
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate.return_value = (b"", b"sim failure")
        monkeypatch.setattr(
            _fluidx3d_mod.asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=mock_proc),
        )
        result = await available_svc.simulate_ball_aerodynamics()
        assert result.success is False
        assert "exited with code 1" in result.notes
        assert result.error == "sim failure"

    @pytest.mark.asyncio
    async def test_timeout(self, available_svc, monkeypatch):
        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = TimeoutError()
        monkeypatch.setattr(
            _fluidx3d_mod.asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=mock_proc),
        )
        result = await available_svc.simulate_ball_aerodynamics(timeout_s=0.01)
        assert result.success is False
        assert "timed out" in result.notes
        assert "0.01" in result.error

    @pytest.mark.asyncio
    async def test_execution_exception(self, available_svc, monkeypatch):
        monkeypatch.setattr(
            _fluidx3d_mod.asyncio,
            "create_subprocess_exec",
            AsyncMock(side_effect=RuntimeError("binary not found")),
        )
        result = await available_svc.simulate_ball_aerodynamics()
        assert result.success is False
        assert "execution failed" in result.notes
        assert "binary not found" in result.error

    @pytest.mark.asyncio
    async def test_output_dir_passed_cleanup_skipped(self, available_svc, monkeypatch, tmp_path):
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate.return_value = (b"ok", b"")
        monkeypatch.setattr(
            _fluidx3d_mod.asyncio,
            "create_subprocess_exec",
            AsyncMock(return_value=mock_proc),
        )
        out_dir = str(tmp_path / "fluid_out")
        result = await available_svc.simulate_ball_aerodynamics(output_dir=out_dir)
        assert result.success is True
        assert out_dir in result.notes
        assert Path(out_dir).exists()
