"""Trust-layer bridge handler (Phase D).

Exposes the intervention-efficacy measurement and the LLM
groundedness gate. The gated report path (``generate_and_gate_report``)
is the production generation route: the LLM text and its grounding
verdict travel together, so an ungrounded claim is visible at the
exact place it would otherwise be consumed as truth.
"""

from __future__ import annotations

import json

from kawkab.core.logging import get_logger
from kawkab.core.security import ErrorSanitizer, SecurityValidator
from kawkab.services.groundedness_service import GroundednessService
from kawkab.services.intervention_efficacy_service import InterventionEfficacyService
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class TrustHandler(BridgeHandlerBase):
    """Slots for the trust layer (Phase D)."""

    async def measure_intervention(self, plan_id):
        """Measure whether a plan moved its diagnosed metrics by N+2.

        Sample-size honesty is part of the contract: fewer than two
        post-plan matches → ``insufficient_sample``; unobservable
        metrics → ``unmeasurable`` — never a fabricated verdict.
        """
        try:
            self._check_rate_limit("training")
            pid = SecurityValidator.validate_int(plan_id)
            kb = await self._knowledge()
            report = await InterventionEfficacyService(
                self._services["storage_service"], kb
            ).measure_intervention(pid)
            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logger.error(f"measure_intervention failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def register_match_evidence(self, match_id):
        """Register a durable evidence record for a match."""
        try:
            self._check_rate_limit("training")
            mid = SecurityValidator.validate_int(match_id)
            record = await GroundednessService(
                self._services["storage_service"]
            ).register_match_evidence(mid)
            return json.dumps(record, ensure_ascii=False)
        except Exception as e:
            logger.error(f"register_match_evidence failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def gate_report_claims(self, review_text: str, evidence_ids_json: str = ""):
        """Gate an arbitrary review text against cited evidence records.

        evidence_ids_json: JSON array of evidence record ids.
        """
        try:
            self._check_rate_limit("training")
            ids = json.loads(evidence_ids_json or "[]")
            if not isinstance(ids, list):
                return json.dumps({"error": "evidence_ids_json must be a JSON array"})
            gate = await GroundednessService(self._services["storage_service"]).gate_claims(
                str(review_text or ""), [int(i) for i in ids]
            )
            return json.dumps(gate, ensure_ascii=False)
        except Exception as e:
            logger.error(f"gate_report_claims failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def generate_and_gate_report(self, match_id, language: str = "en", summary: str = ""):
        """Generate an LLM report and gate it against fresh evidence.

        Returns the report text and its grounding verdict together —
        the honest generation path.
        """
        try:
            self._check_rate_limit("training")
            mid = SecurityValidator.validate_int(match_id)
            llm = self._services.get("llm_service")
            if llm is None:
                return json.dumps(
                    {
                        "error": "LLM provider not configured — no report generated (no fabricated text instead)",
                        "grounding": {"status": "not_applicable", "sentences": []},
                    }
                )
            result = await GroundednessService(
                self._services["storage_service"]
            ).generate_and_gate_report(llm, mid, str(language or "en"), str(summary or ""))
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"generate_and_gate_report failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
