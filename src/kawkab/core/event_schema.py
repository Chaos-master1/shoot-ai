"""Runtime schema validation for event dictionaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


EVENT_SCHEMA: dict[str, tuple] = {
    "type": (str,),
    "team": (str,),
    "timestamp": (float, int),
    "x": (float, (0, 105)),
    "y": (float, (0, 68)),
    "track_id": (int, str, type(None)),
    "end_x": (float, (0, 105), type(None)),
    "end_y": (float, (0, 68), type(None)),
    "event_type": (str, type(None)),
    "period": (int, type(None)),
    "outcome": (str, bool, type(None)),
    "completed": (bool, int, type(None)),
    "minute": (float, int, type(None)),
    "second": (float, int, type(None)),
    "start_x": (float, (0, 105), type(None)),
    "start_y": (float, (0, 68), type(None)),
    "body_part": (str, type(None)),
    "shot_type": (str, type(None)),
    "assist_type": (str, type(None)),
    "is_goal": (bool, int, type(None)),
    "distance_m": (float, int, type(None)),
    "angle_deg": (float, int, type(None)),
    "speed_mps": (float, int, type(None)),
    "confidence": (float, int, type(None)),
    "metadata": (dict, str, type(None)),
}

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def _check_bounds(value: Any, bounds: tuple) -> list[str]:
    errs: list[str] = []
    if (
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and all(isinstance(b, (int, float)) for b in bounds)
    ):
        lo, hi = bounds
        if value is not None:
            try:
                v = float(value)
                if v < lo or v > hi:
                    errs.append(f"Value {v} out of bounds [{lo}, {hi}]")
            except (ValueError, TypeError):
                errs.append(f"Cannot convert {value} to float for bounds check")
    return errs


def validate_event(event: dict) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(event, dict):
        return ValidationResult(valid=False, errors=["Event is not a dict"])

    for field_name, spec in EVENT_SCHEMA.items():
        if field_name not in event:
            continue

        value = event[field_name]
        if value is None:
            if type(None) not in spec:
                warnings.append(f"Field '{field_name}' is None but schema expects non-nullable")
            continue

        type_ok = False
        for st in spec:
            if (
                isinstance(st, tuple)
                and len(st) == 2
                and all(isinstance(b, (int, float)) for b in st)
            ):
                continue
            if isinstance(st, type) and isinstance(value, st):
                type_ok = True
                break

        if not type_ok:
            expected = [str(s) for s in spec if isinstance(s, type)]
            errors.append(
                f"Field '{field_name}': expected one of {expected}, got {type(value).__name__}"
            )

        for st in spec:
            if isinstance(st, tuple):
                errs = _check_bounds(value, st)
                errors.extend(errs)

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_events(events: list[dict]) -> list[ValidationResult]:
    return [validate_event(ev) for ev in events]
