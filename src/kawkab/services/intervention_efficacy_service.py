"""Intervention efficacy measurement (transformation Phase D / D4).

The question this service answers, honestly: **did the training plan
generated after match N measurably move the diagnosed metric by match
N+2?** It deliberately refuses to answer the question when the data
cannot support an answer — that refusal is the product, not a gap:

- ``insufficient_sample`` — fewer than 2 post-plan matches: a delta
  from one match is noise pretending to be signal.
- ``pending_matches`` — the N+2 window has not elapsed yet: the
  outcome is unknown, not negative.
- ``unmeasurable`` — the rule's metric was not observable from the
  event data on the window matches: no delta is computed, because
  "no change" and "no data" are different claims.

Before and after are measured by the **same evaluator**: for each
window match this service rebuilds the reasoning engine's own input
(``reasoning.inputs``) and evaluates the rule with
``ReasoningService.observe_rule``, then reads the declared metric out
of that rule's hypothesis criteria — exactly the number the diagnosis
was based on. It does not re-derive the metric from an event bucket,
because different code paths resolving the same metric name can
disagree, and a delta between two different measurements is not a
delta.
"""

from __future__ import annotations

from typing import Any

from kawkab.core.logging import get_logger
from kawkab.services.reasoning.inputs import (
    build_match_analysis,
    normalize_storage_events,
)
from kawkab.services.reasoning_service import ReasoningService

logger = get_logger(__name__)

# The before-window is the matches the plan itself was generated from
# (capped at 3); the after-window is N+1 and N+2 — the morphocycle's
# natural "did the week change anything" checkpoint.
_BEFORE_WINDOW = 3
_AFTER_WINDOW = 2


def _direction_for_op(op: str) -> str:
    """Map a hypothesis threshold operator to improvement direction.

    ``">3"`` diagnoses a *problem that is too high* → improvement is a
    lower value after the intervention. ``"<3"`` → improvement is a
    higher value. ``">="`` / ``"<="`` behave like their strict forms.
    """
    if op in (">", ">="):
        return "lower_after_is_better"
    if op in ("<", "<="):
        return "higher_after_is_better"
    return "unknown_direction"


def _declared_metric(rule: Any) -> tuple[str | None, str]:
    """Return the rule's first hypothesis metric and its threshold operator."""
    for hypothesis in getattr(rule, "hypotheses", None) or []:
        if not isinstance(hypothesis, dict):
            continue
        for requirement in hypothesis.get("evidence_required") or []:
            if isinstance(requirement, dict) and requirement.get("metric"):
                from kawkab.services.reasoning.metrics import parse_threshold

                parsed = parse_threshold(requirement.get("threshold"))
                operator = parsed[0] if parsed else ""
                return str(requirement["metric"]), operator
    return None, ""


def _observed_value(observation: dict | None, metric: str) -> float | None:
    """Read ``metric`` out of a rule observation's hypothesis criteria.

    ``observe_rule`` evaluates the rule's hypotheses with the same
    ``extract_metric`` the diagnosis used, so the value here is the
    diagnosis's own quantity. A criterion that was not observable
    reports ``observed: None`` and is returned as None — never 0.
    """
    if not observation:
        return None
    for hypothesis in observation.get("hypotheses") or []:
        for criterion in hypothesis.get("criteria") or []:
            if criterion.get("metric") != metric:
                continue
            value = criterion.get("observed")
            if isinstance(value, (int, float)):
                return float(value)
            return None
    return None


class InterventionEfficacyService:
    """Measures whether a persisted training plan moved its own target metrics."""

    def __init__(self, storage_service: Any, knowledge_service: Any) -> None:
        self._storage = storage_service
        self._kb = knowledge_service

    async def _observe_match(
        self, match_id: int, rule_id: str, metric: str, reasoner: ReasoningService
    ) -> tuple[float | None, str]:
        """Return ``(value, reason)`` for one match; value None = unknown."""
        try:
            rows = await self._storage.get_match_events(match_id, limit=100000)
        except Exception as e:
            logger.error(f"measure_intervention events failed for match {match_id}: {e}")
            return None, "storage failure while loading events"
        events = normalize_storage_events(rows)
        if not events:
            return None, "match has no stored events"

        match_data = await self._storage.get_match(match_id) or {}
        analysis = build_match_analysis(match_id, match_data, events)
        observation = await reasoner.observe_rule(rule_id, analysis, events, match_count=1)
        if observation is None:
            return None, "rule has no pattern checker in the reasoning engine"

        value = _observed_value(observation, metric)
        if value is None:
            return None, f"metric '{metric}' not observable from this match's events"
        return value, ""

    async def measure_intervention(self, plan_id: int) -> dict[str, Any]:
        """Measure metric deltas for one plan, with honest sample-size provenance."""
        try:
            plan = await self._storage.get_training_plan(plan_id)
        except Exception as e:  # storage failure is loud, not "no effect"
            logger.error(f"measure_intervention storage error: {e}")
            return {"error": f"storage failure while loading plan {plan_id}"}

        if plan is None:
            return {"error": f"plan not found: {plan_id}"}

        payload: dict[str, Any] = (
            plan.get("payload") if isinstance(plan.get("payload"), dict) else {}
        )

        # Two places can hold the diagnoses: the payload (the editable
        # plan document, and what hand-corrected plans carry) and the
        # training_plans.priority_diagnoses column (what the generator
        # persists). Prefer the payload when it has them, fall back to
        # the column. Reading the payload *only* -- the old behaviour --
        # missed every generated plan, which stores them in the column.
        diagnoses = payload.get("priority_diagnoses") or plan.get("priority_diagnoses") or []

        based_on: list[int] = []
        for raw in payload.get("based_on_match_ids") or []:
            try:
                based_on.append(int(raw))
            except (TypeError, ValueError):
                continue
        if not based_on and plan.get("match_id"):
            try:
                based_on = [int(plan["match_id"])]
            except (TypeError, ValueError):
                based_on = []

        if not based_on or not diagnoses:
            return {
                "plan_id": plan_id,
                "error": (
                    "plan lacks based_on_match_ids or priority_diagnoses — "
                    "cannot define a before/after window"
                ),
            }

        plan_match_n = max(based_on)

        try:
            matches = await self._storage.get_all_matches()
        except Exception as e:
            logger.error(f"measure_intervention match listing failed: {e}")
            return {"plan_id": plan_id, "error": "storage failure while listing matches"}

        try:
            match_ids = sorted(int(m["id"]) for m in matches)
        except (TypeError, ValueError, KeyError) as e:
            logger.error(f"measure_intervention match listing malformed: {e}")
            return {"plan_id": plan_id, "error": "match list could not be read"}

        before_ids = [m for m in match_ids if m < plan_match_n][-_BEFORE_WINDOW:]
        after_ids = [m for m in match_ids if m > plan_match_n][:_AFTER_WINDOW]

        reasoner = ReasoningService(self._kb)
        await reasoner.initialize()

        per_rule: list[dict[str, Any]] = []
        for diagnosis in diagnoses:
            if not isinstance(diagnosis, dict):
                continue
            rule_id = str(diagnosis.get("rule_id") or "")
            rule = self._kb.get_rule(rule_id) if rule_id else None
            if rule is None:
                per_rule.append(
                    {
                        "rule_id": rule_id,
                        "state": "rule_not_found",
                        "note": "rule missing from the knowledge base — no metric defined",
                    }
                )
                continue

            metric, threshold_op = _declared_metric(rule)
            if not metric:
                per_rule.append(
                    {
                        "rule_id": rule_id,
                        "rule_name": diagnosis.get("rule_name"),
                        "state": "no_declared_metric",
                        "note": "rule declares no evidence_required metric — efficacy not computable",
                    }
                )
                continue

            before_vals: list[dict[str, Any]] = []
            after_vals: list[dict[str, Any]] = []
            reasons: list[str] = []

            for mid in before_ids:
                value, reason = await self._observe_match(mid, rule_id, metric, reasoner)
                entry: dict[str, Any] = {"match_id": mid, "value": value}
                if value is None:
                    entry["reason"] = reason
                    reasons.append(reason)
                before_vals.append(entry)

            for mid in after_ids:
                value, reason = await self._observe_match(mid, rule_id, metric, reasoner)
                entry = {"match_id": mid, "value": value}
                if value is None:
                    entry["reason"] = reason
                    reasons.append(reason)
                after_vals.append(entry)

            known_before = [e["value"] for e in before_vals if e["value"] is not None]
            known_after = [e["value"] for e in after_vals if e["value"] is not None]

            entry_out: dict[str, Any] = {
                "rule_id": rule_id,
                "rule_name": diagnosis.get("rule_name"),
                "metric": metric,
                "direction": _direction_for_op(threshold_op)
                if threshold_op
                else "unknown_direction",
                "threshold_operator": threshold_op or "undeclared",
                "before": before_vals,
                "after": after_vals,
                "measured_by": (
                    "rule hypothesis criteria via the reasoning engine's own "
                    "observe_rule evaluation (same quantity as the diagnosis)"
                ),
            }

            if not known_before and not known_after:
                entry_out["state"] = "unmeasurable"
                entry_out["note"] = (
                    "metric not observable on any window match — no delta computed; "
                    "'no data' is not 'no change'"
                )
                if reasons:
                    entry_out["reasons"] = sorted(set(reasons))
            elif len(known_after) < _AFTER_WINDOW:
                entry_out["state"] = "insufficient_sample"
                entry_out["note"] = (
                    f"only {len(known_after)}/{_AFTER_WINDOW} post-plan matches measured — "
                    "a delta from one match is noise, not evidence"
                )
                if known_before and known_after:
                    entry_out["delta_observed_but_not_conclusive"] = sum(known_after) / len(
                        known_after
                    ) - sum(known_before) / len(known_before)
            else:
                mean_before = sum(known_before) / len(known_before) if known_before else 0.0
                mean_after = sum(known_after) / len(known_after)
                delta = mean_after - mean_before
                improved: bool | None = None
                if entry_out["direction"] == "lower_after_is_better":
                    improved = delta < 0
                elif entry_out["direction"] == "higher_after_is_better":
                    improved = delta > 0
                entry_out.update(
                    {
                        "state": "measured",
                        "mean_before": round(mean_before, 4),
                        "mean_after": round(mean_after, 4),
                        "delta": round(delta, 4),
                        "improved": improved,
                        "note": (
                            "raw event-level delta; single-club small-sample — "
                            "a directional signal to review, not a causal verdict"
                        ),
                    }
                )

            per_rule.append(entry_out)

        return {
            "plan_id": plan_id,
            "plan_match_n": plan_match_n,
            "based_on_match_ids": based_on,
            "before_window": before_ids,
            "after_window": after_ids,
            "after_window_complete": len(after_ids) >= _AFTER_WINDOW,
            "window_note": "after-window = the next two stored matches following plan generation (N+1, N+2 in play order)",
            "rules": per_rule,
            "provenance": {
                "metric_resolution": (
                    "rule-declared evidence_required metric, observed by the same "
                    "reasoning-engine evaluation the diagnosis used"
                ),
                "created_by": plan.get("created_by"),
                "plan_created_at": plan.get("created_at"),
            },
        }
