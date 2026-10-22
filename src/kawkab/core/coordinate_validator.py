from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    clamped: bool = False


class CoordinateValidator:
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0

    @staticmethod
    def validate_x(x: float) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        clamped = False
        try:
            val = float(x)
        except (TypeError, ValueError):
            errors.append(f"x coordinate is not numeric: {x!r}")
            return ValidationResult(valid=False, errors=errors)
        if val < 0:
            warnings.append(f"x={val} clamped to 0 (below pitch boundary)")
            clamped = True
        elif val > CoordinateValidator.PITCH_LENGTH:
            warnings.append(
                f"x={val} clamped to {CoordinateValidator.PITCH_LENGTH} (exceeds pitch length)"
            )
            clamped = True
        return ValidationResult(valid=True, errors=[], warnings=warnings, clamped=clamped)

    @staticmethod
    def validate_y(y: float) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        clamped = False
        try:
            val = float(y)
        except (TypeError, ValueError):
            errors.append(f"y coordinate is not numeric: {y!r}")
            return ValidationResult(valid=False, errors=errors)
        if val < 0:
            warnings.append(f"y={val} clamped to 0 (below pitch boundary)")
            clamped = True
        elif val > CoordinateValidator.PITCH_WIDTH:
            warnings.append(
                f"y={val} clamped to {CoordinateValidator.PITCH_WIDTH} (exceeds pitch width)"
            )
            clamped = True
        return ValidationResult(valid=True, errors=[], warnings=warnings, clamped=clamped)

    @staticmethod
    def validate_point(x: float, y: float) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        clamped = False
        rx = CoordinateValidator.validate_x(x)
        errors.extend(rx.errors)
        warnings.extend(rx.warnings)
        clamped = clamped or rx.clamped
        ry = CoordinateValidator.validate_y(y)
        errors.extend(ry.errors)
        warnings.extend(ry.warnings)
        clamped = clamped or ry.clamped
        return ValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings, clamped=clamped
        )

    @staticmethod
    def validate_event_spatial(event: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        clamped = False
        spatial_fields = ["x", "y", "end_x", "end_y", "start_x", "start_y"]
        present = [f for f in spatial_fields if f in event and event[f] is not None]
        if not present:
            return ValidationResult(valid=True)
        for spatial_field in present:
            val = event[spatial_field]
            try:
                fval = float(val)
            except (TypeError, ValueError):
                errors.append(f"field '{spatial_field}' is not numeric: {val!r}")
                continue
            if spatial_field in ("x", "end_x", "start_x"):
                r = CoordinateValidator.validate_x(fval)
                if r.clamped:
                    event[spatial_field] = CoordinateValidator.clamp_x(fval)
                    clamped = True
            else:
                r = CoordinateValidator.validate_y(fval)
                if r.clamped:
                    event[spatial_field] = CoordinateValidator.clamp_y(fval)
                    clamped = True
            warnings.extend(r.warnings)
        return ValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings, clamped=clamped
        )

    @staticmethod
    def clamp_x(x: float) -> float:
        return max(0.0, min(CoordinateValidator.PITCH_LENGTH, x))

    @staticmethod
    def clamp_y(y: float) -> float:
        return max(0.0, min(CoordinateValidator.PITCH_WIDTH, y))
