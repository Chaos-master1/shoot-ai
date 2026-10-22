"""Operating program service (Phase D): the club's weekly operating
rhythm, roles/RACI, and the KPI tree — versioned club documents.

Honesty contract:

- The program is *stored* (migration-033 ``program_documents``), not
  hardcoded: publishing creates a new version and supersedes the old,
  so "what did we commit to and when" is always answerable.
- ``instantiate_week_rituals`` materializes the stored rhythm into
  real ``staff_rituals`` rows for a match-week (the morphocycle
  vocabulary matches TrainingPlanGenerator.MORPHOCYCLE). If no rhythm
  document is stored, the state is explicit — nothing is invented.
- The KPI tree stores *intent* (metric, target, direction). Reading
  ``kpi_snapshot`` computes each KPI's current value from real stored
  data where the repo can, and reports ``no_data`` where it cannot —
  a KPI with no basis is never fabricated.
"""

from __future__ import annotations

from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

# Document kinds (mirrors migration 033 comment).
DOC_RHYTHM = "weekly_rhythm"
DOC_RACI = "raci_matrix"
DOC_KPI = "kpi_tree"


class OperatingProgramService:
    """Versioned operating program over the program_documents table."""

    def __init__(self, storage_service: Any) -> None:
        self._storage = storage_service

    # -- Publishing (versioned writes) ---------------------------------------

    async def publish_weekly_rhythm(
        self,
        week: list[dict[str, Any]],
        name: str = "in-season morphocycle week",
        created_by: str = "",
    ) -> dict[str, Any]:
        """Publish the weekly rhythm (one entry per day of the morphocycle).

        Each entry: {day_label (e.g. "MD-3"), rituals: [{ritual_type,
        checklist?}]} — ``ritual_type`` must use the staff_rituals
        vocabulary (wellness_huddle, post_match_review, staff_sync,
        md1_brief, analyst_prematch, analyst_postmatch, briefing_prep).
        """
        if not week or not isinstance(week, list):
            return {"error": "week must be a non-empty list of day entries"}
        bad = [
            w for w in week if not isinstance(w, dict) or not str(w.get("day_label") or "").strip()
        ]
        if bad:
            return {"error": "every day entry needs a day_label"}
        doc_id = await self._storage.save_program_document(
            DOC_RHYTHM, name, {"days": week}, created_by=created_by
        )
        return {"success": True, "document_id": doc_id, "days": len(week)}

    async def publish_raci(
        self, roles: list[dict[str, Any]], name: str = "core staff", created_by: str = ""
    ) -> dict[str, Any]:
        """Publish the roles/RACI matrix.

        Each role entry: {role, responsible_for: [decision areas],
        consulted: [roles], informed: [roles]}.
        """
        if not roles or not isinstance(roles, list):
            return {"error": "roles must be a non-empty list"}
        bad = [r for r in roles if not isinstance(r, dict) or not str(r.get("role") or "").strip()]
        if bad:
            return {"error": "every role entry needs a role name"}
        doc_id = await self._storage.save_program_document(
            DOC_RACI, name, {"roles": roles}, created_by=created_by
        )
        return {"success": True, "document_id": doc_id, "roles": len(roles)}

    async def publish_kpi_tree(
        self, kpis: list[dict[str, Any]], name: str = "club KPI tree", created_by: str = ""
    ) -> dict[str, Any]:
        """Publish the KPI tree.

        Each KPI: {key, label, metric (basis name from the KPI
        registry below), target, direction ("higher"|"lower"),
        parent (optional key for tree structure)}.
        """
        if not kpis or not isinstance(kpis, list):
            return {"error": "kpis must be a non-empty list"}
        known = {
            "training_plans_generated",
            "diagnosis_confirmation_rate",
            "wellness_capture_rate",
            "srpe_capture_rate",
            "drill_effectiveness",
            "availability_pct",
            "squad_minutes_balance",
        }
        bad = [
            k
            for k in kpis
            if not isinstance(k, dict)
            or not str(k.get("key") or "").strip()
            or str(k.get("metric") or "") not in known
        ]
        if bad:
            return {
                "error": "every KPI needs a key and a metric from the supported registry",
                "supported_metrics": sorted(known),
            }
        doc_id = await self._storage.save_program_document(
            DOC_KPI, name, {"kpis": kpis}, created_by=created_by
        )
        return {"success": True, "document_id": doc_id, "kpis": len(kpis)}

    async def current_program(self) -> dict[str, Any]:
        """The current version of all three program documents (soft read)."""
        out: dict[str, Any] = {}
        for kind, key in ((DOC_RHYTHM, "weekly_rhythm"), (DOC_RACI, "raci"), (DOC_KPI, "kpi_tree")):
            doc = await self._storage.get_current_program_document(kind)
            out[key] = (
                {
                    "version": doc["version"],
                    "payload": doc["payload"],
                    "created_at": doc["created_at"],
                }
                if doc
                else None
            )
        out["provenance"] = {"basis": "program_documents (migration 033), current versions only"}
        return out

    # -- Rhythm instantiation ---------------------------------------------------

    async def instantiate_week_rituals(
        self, week_start_date: str, completed_by: str = ""
    ) -> dict[str, Any]:
        """Materialize the stored weekly rhythm into staff_rituals rows.

        ``week_start_date`` is the Monday of the target week; ritual
        rows are created for each rhythm day (upsert per type+date, so
        re-running is idempotent).
        """
        rhythm = await self._storage.get_current_program_document(DOC_RHYTHM)
        if not rhythm:
            return {
                "state": "no_rhythm_document",
                "honesty_note": "publish a weekly rhythm first; none is stored",
            }
        days = (rhythm.get("payload") or {}).get("days") or []
        if not days:
            return {"error": "stored rhythm document has no days"}

        from datetime import date as _date
        from datetime import timedelta as _td

        try:
            start = _date.fromisoformat(str(week_start_date)[:10])
        except ValueError:
            return {"error": "week_start_date must be an ISO date (YYYY-MM-DD)"}

        created = 0
        for i, day in enumerate(days):
            day_date = (start + _td(days=i)).isoformat()
            for ritual in day.get("rituals") or []:
                rtype = str(ritual.get("ritual_type") or "").strip()
                if not rtype:
                    continue
                await self._storage.save_ritual(
                    rtype,
                    day_date,
                    checklist_state=ritual.get("checklist") or [],
                    notes=f"from rhythm v{rhythm['version']} ({day.get('day_label', '')})",
                )
                created += 1
        return {
            "success": True,
            "week_start_date": start.isoformat(),
            "rituals_scheduled": created,
            "rhythm_version": rhythm["version"],
            "provenance": {
                "basis": f"weekly_rhythm v{rhythm['version']}",
                "completed_by": completed_by,
            },
        }

    # -- KPI snapshot --------------------------------------------------------------

    async def kpi_snapshot(self, match_window_days: int = 28) -> dict[str, Any]:
        """Current value for each KPI in the stored tree, from real data.

        Every KPI value carries its basis and window; anything the repo
        cannot compute returns ``no_data`` rather than a guess.
        """
        tree = await self._storage.get_current_program_document(DOC_KPI)
        if not tree:
            return {
                "state": "no_kpi_tree",
                "honesty_note": "publish a KPI tree first; none is stored",
            }
        kpis = (tree.get("payload") or {}).get("kpis") or []

        values = await self._compute_kpi_values(match_window_days)
        out = []
        for k in kpis:
            metric = str(k.get("metric") or "")
            computed = values.get(
                metric, {"value": None, "state": "no_data", "basis": "unsupported metric"}
            )
            out.append(
                {
                    "key": k.get("key"),
                    "label": k.get("label") or k.get("key"),
                    "target": k.get("target"),
                    "direction": k.get("direction") or "higher",
                    "parent": k.get("parent"),
                    **computed,
                }
            )
        return {
            "kpis": out,
            "tree_version": tree["version"],
            "provenance": {
                "window_days": match_window_days,
                "basis": "training-OS tables + reasoning plans (see per-KPI basis)",
            },
        }

    async def _compute_kpi_values(self, window_days: int) -> dict[str, dict[str, Any]]:
        """Compute each registry metric from real stored data."""
        values: dict[str, dict[str, Any]] = {}

        # Plans generated (all time — count is cheap and unambiguous).
        try:
            plans = await self._storage.list_training_plans()
            values["training_plans_generated"] = {
                "value": len(plans),
                "state": "ok",
                "basis": "training_plans rows",
            }
        except Exception:
            values["training_plans_generated"] = {
                "value": None,
                "state": "no_data",
                "basis": "training_plans rows",
            }

        # Diagnosis confirmation rate over multi-match plans.
        confirmed = total = 0
        try:
            for p in plans:
                for d in p.get("priority_diagnoses") or []:
                    total += 1
                    if str(d.get("confirmation")) == "confirmed":
                        confirmed += 1
            values["diagnosis_confirmation_rate"] = {
                "value": round(confirmed / total, 3) if total else None,
                "state": "ok" if total else "no_data",
                "basis": f"{total} diagnoses across multi-match plans",
            }
        except Exception:
            values["diagnosis_confirmation_rate"] = {
                "value": None,
                "state": "no_data",
                "basis": "training_plans.priority_diagnoses",
            }

        # Wellness / sRPE capture rates over the window.
        from datetime import date as _date
        from datetime import timedelta as _td

        end = _date.today()
        start = end - _td(days=window_days)
        try:
            sessions = await self._storage.get_training_sessions(
                date_from=start.isoformat(), date_to=end.isoformat(), limit=1000
            )
            sids = [int(s["id"]) for s in sessions]
            rpe_rows = []
            for sid in sids:
                rpe_rows.extend(await self._storage.get_session_rpe(sid))
            n_sessions = len(sids)
            players_per_session = (len(rpe_rows) / n_sessions) if n_sessions else 0
            values["srpe_capture_rate"] = {
                "value": round(players_per_session, 2) if n_sessions else None,
                "state": "ok" if n_sessions else "no_data",
                "basis": f"{len(rpe_rows)} sRPE rows / {n_sessions} sessions (rows-per-session, squad-size-relative)",
                "honesty_note": "squad roster size is not stored per session; interpret as rows per session",
            }
        except Exception:
            values["srpe_capture_rate"] = {
                "value": None,
                "state": "no_data",
                "basis": "training_sessions + session_rpe",
            }
        values["wellness_capture_rate"] = {
            "value": None,
            "state": "no_data",
            "basis": "wellness table has no squad-size denominator; needs roster snapshot (honest gap)",
        }

        # Drill effectiveness mean.
        try:
            fb = await self._storage.get_drill_feedback("__all__")
        except Exception:
            fb = []
        effs = [float(f.get("effectiveness")) for f in fb if f.get("effectiveness") is not None]
        values["drill_effectiveness"] = {
            "value": round(sum(effs) / len(effs), 2) if effs else None,
            "state": "ok" if effs else "no_data",
            "basis": f"{len(effs)} drill_feedback rows",
        }

        # Availability: fraction of profiles with a fit/limited clearance.
        try:
            profiles = await self._storage.get_all_player_profiles(limit=1000)
            pids = [int(p["id"]) for p in profiles]
            fit = 0
            known = 0
            for pid in pids:
                clr = await self._storage.get_latest_clearance(pid)
                if clr:
                    known += 1
                    if str(clr.get("status")) == "fit":
                        fit += 1
            values["availability_pct"] = {
                "value": round(100.0 * fit / known, 1) if known else None,
                "state": "ok" if known else "no_data",
                "basis": f"{fit} fit of {known} cleared profiles",
            }
        except Exception:
            values["availability_pct"] = {
                "value": None,
                "state": "no_data",
                "basis": "medical_clearances",
            }

        # Squad minutes balance (share of minutes logged for the least-used player).
        try:
            summary = await self._storage.get_squad_minutes_summary()
            totals = [int(s.get("total_minutes") or 0) for s in summary if s.get("total_minutes")]
            if len(totals) >= 2 and sum(totals) > 0:
                share = totals[-1] / sum(totals)
                values["squad_minutes_balance"] = {
                    "value": round(share, 3),
                    "state": "ok",
                    "basis": f"lowest-logged player holds {share:.1%} of logged minutes ({len(totals)} players)",
                }
            else:
                values["squad_minutes_balance"] = {
                    "value": None,
                    "state": "no_data",
                    "basis": "minutes_log",
                }
        except Exception:
            values["squad_minutes_balance"] = {
                "value": None,
                "state": "no_data",
                "basis": "minutes_log",
            }

        return values
