"""Reasoning engine package: pattern checkers and metric extraction.

The ``ReasoningService`` class itself stays in
``kawkab.services.reasoning_service`` (import compatibility: it has
always lived there and many modules import it from there). This
package holds the pure computational pieces the engine dispatches to:

- ``checkers``: one confidence checker per knowledge-base pattern type
- ``metrics``: shared event-stat precomputation and hypothesis
  evidence extraction (threshold parsing, provenance records)

Re-exports of the service classes are lazy (module ``__getattr__``)
because ``reasoning_service`` imports this package's checkers — a
module-level re-export here would be a circular import.
"""

from __future__ import annotations


def __getattr__(name: str):
    if name in ("ReasoningService", "Diagnosis", "DiagnosisReport"):
        from kawkab.services import reasoning_service as _rs

        return getattr(_rs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
