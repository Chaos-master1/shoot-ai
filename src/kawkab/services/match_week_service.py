"""Match Week: the diagnosis -> plan -> re-check loop as one read model.

The coach's weekly cycle is: diagnose the last match, train on it for a
week or two, then check whether the diagnosed metric actually moved.
Each of those pieces exists already:

- diagnoses are persisted on the plan (``training_plans.priority_diagnoses``)
- the plan itself is a persisted, versioned document
- the before/after measurement is ``InterventionEfficacyService``

What did not exist was one place that composes them into the state of a
*match week* without inventing anything. This module is that place: it
reads persisted facts, delegates the measurement to the efficacy service
(no delta math here), and maps the result onto a small stage vocabulary
the UI can render.

It deliberately does not diagnose on the fly. Generating a plan is the
diagnosis step, and that is an explicit, persisted action — recomputing
diagnoses on every panel refresh would create a second, ephemeral
diagnosis that can disagree with the plan the coach is actually
following.

No I/O beyond the injected storage service; no Qt; no formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.services.intervention_efficacy_service import InterventionEfficacyService

logger = get_logger(__name__)

# Stage vocabulary. Ordered from "nothing to show" to "measured".
NO_ANALYSIS = "no_analysis"
NO_PLAN = "no_plan"
AWAITING_MATCHES = "awaiting_matches"
MEASURED = "measured"
NOT_OBSERVED = "not_observed"
ERROR = "error"


@dataclass(frozen=True)
class MatchWeekState:
    """Everything the Match Week panel needs, and nothing more."""

    match_id: int
    stage: str
    note: str = ""
    diagnoses: tuple[dict, ...] = ()
    plan: dict | None = None
    recheck: dict | None = None
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "stage": self.stage,
            "note": self.note,
            "diagnoses": list(self.diagnoses),
            "plan": self.plan,
            "recheck": self.recheck,
            "provenance": self.provenance,
        }


class MatchWeekService:
    """Composes the persisted plan, its diagnoses and its re-check."""

    def __init__(
        self,
        storage_service: Any,
        knowledge_service: Any,
        homography_service: Any = None,
    ) -> None:
        self._storage = storage_service
        self._kb = knowledge_service
        self._homography = homography_service

    async def state(self, match_id: int) -> MatchWeekState:
        match_id = int(match_id)

        try:
            events = await self._storage.get_match_events(match_id, limit=1)
        except Exception as e:
            logger.error(f"match week events lookup failed for {match_id}: {e}")
            return MatchWeekState(
                match_id=match_id,
                stage=ERROR,
                note="storage failure while checking this match's events",
            )

        if not events:
            return MatchWeekState(
                match_id=match_id,
                stage=NO_ANALYSIS,
                note="this match has no analysis events yet — run the analysis first",
                provenance=self._provenance(match_id),
            )

        try:
            plans = await self._storage.get_training_plans_for_match(match_id)
        except Exception as e:
            logger.error(f"match week plan lookup failed for {match_id}: {e}")
            return MatchWeekState(
                match_id=match_id,
                stage=ERROR,
                note="storage failure while checking this match's training plans",
            )

        if not plans:
            return MatchWeekState(
                match_id=match_id,
                stage=NO_PLAN,
                note="no training plan yet — generate one from this match's diagnoses",
                provenance=self._provenance(match_id),
            )

        # get_training_plans_for_match is newest-first.
        plan_id = int(plans[0]["id"])
        plan = await self._storage.get_training_plan(plan_id)
        if plan is None:
            return MatchWeekState(
                match_id=match_id,
                stage=ERROR,
                note="the stored plan could not be read back",
            )

        payload = plan.get("payload") if isinstance(plan.get("payload"), dict) else {}
        diagnoses = tuple(plan.get("priority_diagnoses") or payload.get("priority_diagnoses") or ())

        try:
            recheck = await InterventionEfficacyService(
                self._storage, self._kb
            ).measure_intervention(plan_id)
        except Exception as e:
            logger.error(f"match week efficacy failed for plan {plan_id}: {e}")
            recheck = {"plan_id": plan_id, "error": "efficacy measurement failed"}

        plan_summary = {
            "id": plan_id,
            "title": plan.get("title") or "",
            "status": plan.get("status") or "",
            "duration_weeks": plan.get("duration_weeks"),
            "source": plan.get("source") or "",
            "created_at": plan.get("created_at") or "",
            "priority_addressed": payload.get("priority_addressed") or [],
            "unresolved_drills": payload.get("unresolved_drills") or [],
            # The persisted plan document is the only source for the weeks
            # the coach is following, so the panel renders from here rather
            # than from the one-off generation response.
            "total_drills": payload.get("total_drills") or 0,
            "expected_overall_improvement": payload.get("expected_overall_improvement") or "",
            "weeks": payload.get("weeks") or [],
        }

        stage, note = self._classify(recheck)
        provenance = self._provenance(match_id)
        if isinstance(recheck, dict):
            provenance.update(
                {
                    "based_on_match_ids": recheck.get("based_on_match_ids") or [],
                    "before_window": recheck.get("before_window") or [],
                    "after_window": recheck.get("after_window") or [],
                    "after_window_complete": bool(recheck.get("after_window_complete")),
                }
            )

        return MatchWeekState(
            match_id=match_id,
            stage=stage,
            note=note,
            diagnoses=diagnoses,
            plan=plan_summary,
            recheck=recheck,
            provenance=provenance,
        )

    @staticmethod
    def _classify(recheck: dict) -> tuple[str, str]:
        """Map an efficacy report onto a stage without overclaiming."""
        if not isinstance(recheck, dict):
            return ERROR, "the measurement could not be computed"
        if recheck.get("error"):
            # Surface the service's own reason (e.g. a plan with no
            # diagnoses) instead of a generic failure message.
            return ERROR, str(recheck["error"])
        if not recheck.get("after_window_complete"):
            recorded = len(recheck.get("after_window") or [])
            return AWAITING_MATCHES, (
                f"{recorded}/2 post-plan matches recorded — "
                "the outcome is unknown until the next two matches exist"
            )

        rules = recheck.get("rules") or []
        measured = [r for r in rules if isinstance(r, dict) and r.get("state") == "measured"]
        if measured:
            improved = sum(1 for r in measured if r.get("improved") is True)
            return MEASURED, (
                f"{len(measured)} of {len(rules)} diagnosed metric(s) measured; "
                f"{improved} moved in the intended direction"
            )
        return NOT_OBSERVED, (
            "the window is complete but no diagnosed metric was observable — no delta is claimed"
        )

    def _provenance(self, match_id: int) -> dict:
        """Facts that decide how far the numbers can be trusted."""
        calibrated = None
        if self._homography is not None:
            try:
                calibrated = self._homography.load_calibration(match_id) is not None
            except Exception as e:  # a calibration read failure is unknown, not "no"
                logger.debug(f"calibration lookup failed for {match_id}: {e}")
                calibrated = None
        return {"calibrated": calibrated}
