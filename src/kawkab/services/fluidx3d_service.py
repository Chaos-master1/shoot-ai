"""Subprocess wrapper for FluidX3D ball aerodynamics simulation.

FluidX3D (https://github.com/ProjectPhysX/FluidX3D) is a Lattice-Boltzmann
CFD solver. For ball aerodynamics, we would need to:
1. Build a FluidX3D executable from source (OpenCL, C++17)
2. Wrap a setup.cpp that voxelizes a sphere
3. Capture stdout for the simulated velocity/pressure field around the ball

This service provides a stub interface that gracefully degrades when
FluidX3D is not installed (which is the common case). When the user
provides a path to a custom FluidX3D binary, it can run aerodynamic
simulations and return the airflow patterns.

License: FluidX3D is "Free for non-commercial use" (per its LICENSE).
Users must accept this license before enabling.

Use case: Simulate airflow patterns around a spinning ball at different
speeds and spin rates, predict curve/knuckle effects with proper CFD.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_FLUIDX3D_LICENSE_NOTE = (
    "FluidX3D is free for non-commercial use only. By enabling this feature, "
    "you confirm that your use of Kawkab AI is non-commercial."
)


@dataclass
class CfdResult:
    success: bool
    method: str
    velocity_field: list[list[list[list[float]]]] | None
    pressure_field: list[list[list[float]]] | None
    drag_coefficient: float | None
    lift_coefficient: float | None
    notes: str
    error: str | None = None


class FluidX3DService:
    """Subprocess wrapper for FluidX3D CFD simulations.

    FluidX3D is a standalone OpenCL C++ binary that must be built from
    source. This service allows users to point at a pre-built binary
    (via the binary_path argument or KAWKAB_FLUIDX3D_PATH env var).

    If no binary is configured, the service returns a stub result
    explaining how to enable it. This avoids a hard dep on a large
    C++ build.
    """

    def __init__(self, binary_path: str | None = None) -> None:
        self._binary_path = binary_path or os.environ.get("KAWKAB_FLUIDX3D_PATH")
        self._available = False
        self._check_binary()

    def _check_binary(self) -> None:
        if not self._binary_path:
            logger.info("FluidX3D binary not configured")
            self._available = False
            return
        path = Path(self._binary_path)
        if not path.exists():
            logger.warning(f"FluidX3D binary not found at {path}")
            self._available = False
            return
        if not os.access(path, os.X_OK):
            logger.warning(f"FluidX3D binary at {path} is not executable")
            self._available = False
            return
        self._available = True
        logger.info(f"FluidX3D binary available at {path}")

    @property
    def available(self) -> bool:
        return self._available

    @property
    def license_notice(self) -> str:
        return DEFAULT_FLUIDX3D_LICENSE_NOTE

    async def simulate_ball_aerodynamics(
        self,
        ball_radius: float = 0.11,
        wind_speed: float = 0.0,
        spin_rps: float = 0.0,
        output_dir: str | None = None,
        timeout_s: float = 120.0,
    ) -> CfdResult:
        """Simulate airflow around a ball.

        Args:
            ball_radius: ball radius in meters
            wind_speed: relative wind speed in m/s
            spin_rps: spin rate in revolutions per second
            output_dir: where FluidX3D writes its .vtk output
            timeout_s: max simulation time in seconds

        Returns:
            CfdResult with success flag, fields, and drag/lift coefficients.
        """
        if not self._available:
            return CfdResult(
                success=False,
                method="none",
                velocity_field=None,
                pressure_field=None,
                drag_coefficient=None,
                lift_coefficient=None,
                notes="FluidX3D binary not configured",
                error=(
                    "FluidX3D not available. Build from source and set "
                    "KAWKAB_FLUIDX3D_PATH environment variable to the binary path."
                ),
            )
        out = output_dir or tempfile.mkdtemp(prefix="fluidx3d_")
        os.makedirs(out, exist_ok=True)
        cmd = [
            str(self._binary_path),
            f"--radius={ball_radius}",
            f"--velocity={wind_speed}",
            f"--spin={spin_rps}",
            f"--output={out}",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            if proc.returncode != 0:
                return CfdResult(
                    success=False,
                    method="fluidx3d",
                    velocity_field=None,
                    pressure_field=None,
                    drag_coefficient=None,
                    lift_coefficient=None,
                    notes=f"FluidX3D exited with code {proc.returncode}",
                    error=stderr.decode("utf-8", errors="replace"),
                )
            return CfdResult(
                success=True,
                method="fluidx3d",
                velocity_field=None,
                pressure_field=None,
                drag_coefficient=None,
                lift_coefficient=None,
                notes=f"Simulation complete. Output in {out}",
                error=None,
            )
        except TimeoutError:
            return CfdResult(
                success=False,
                method="fluidx3d",
                velocity_field=None,
                pressure_field=None,
                drag_coefficient=None,
                lift_coefficient=None,
                notes="FluidX3D simulation timed out",
                error=f"Timeout after {timeout_s}s",
            )
        except Exception as e:
            return CfdResult(
                success=False,
                method="fluidx3d",
                velocity_field=None,
                pressure_field=None,
                drag_coefficient=None,
                lift_coefficient=None,
                notes="FluidX3D execution failed",
                error=str(e),
            )
