"""Training OS bridge handler — daily wellness, sRPE, sessions, game
model, drill feedback, testing, IDPs, nutrition, psychology, rituals,
minutes, and medical clearances.

Every slot follows the Phase A honesty contract:

- inputs validated and clamped at the service boundary (1-5 wellness
  scales, 0-10 Borg RPE, clearance status vocabulary),
- reads fail soft with explicit ``count: 0``/empty payloads, writes
  fail loud (error JSON with sanitized message),
- provenance labels travel with the data (``source`` fields, plan
  ``source`` column, plan payload ``based_on_match_ids``),
- the multi-match plan slot refuses to silently fabricate a
  diagnosis when no match has any events (empty-pool honesty).
"""

from __future__ import annotations

import json
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.core.security import ErrorSanitizer, SecurityValidator
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)

# Rate-limit category: "training", not "analysis". The analysis bucket is
# sized for expensive per-match computations (5/min); the Training OS
# serves squad-scale daily rituals -- a 25-player squad submitting
# morning wellness plus session RPE bursts easily exceeds that. The
# "training" bucket uses the standard 60/min default.

_SCALE_MIN, _SCALE_MAX = 1, 5


def _clamp_scale(value: Any, name: str, default: int) -> int:
    """Clamp a 1-5 scale value; accept ints or numeric strings."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return max(_SCALE_MIN, min(_SCALE_MAX, v))


class TrainingHandler(BridgeHandlerBase):
    """Slots for the Training OS (migration-032 tables)."""

    # ------------------------------------------------------------------
    # Daily wellness
    # ------------------------------------------------------------------

    async def save_wellness(self, player_id, payload: str | dict):
        """Record (or upsert) one player's daily wellness questionnaire.

        payload keys: sleep_quality, fatigue, soreness, stress, mood
        (1-5, 5 = best), record_date (ISO), source. Scores are clamped
        to 1-5; the normalized mean is computed by storage.
        """
        try:
            self._check_rate_limit("training")
            pid = SecurityValidator.validate_int(player_id)
            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            record_date = str(data.get("record_date") or "").strip()
            if not record_date:
                return json.dumps({"error": "record_date is required (ISO date)"})
            wid = await self._services["storage_service"].save_wellness(
                player_id=pid,
                record_date=record_date,
                sleep_quality=_clamp_scale(data.get("sleep_quality"), "sleep_quality", 3),
                fatigue=_clamp_scale(data.get("fatigue"), "fatigue", 3),
                soreness=_clamp_scale(data.get("soreness"), "soreness", 3),
                stress=_clamp_scale(data.get("stress"), "stress", 3),
                mood=_clamp_scale(data.get("mood"), "mood", 3),
                source=str(data.get("source") or "manual"),
            )
            return json.dumps(
                {"success": True, "wellness_id": wid, "player_id": pid, "record_date": record_date}
            )
        except Exception as e:
            logger.error(f"save_wellness failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_squad_wellness(self, record_date: str):
        """Latest wellness entry per player for a date (morning ritual view)."""
        try:
            self._check_rate_limit("training")
            entries = await self._services["storage_service"].get_squad_wellness_latest(
                str(record_date or "").strip()
            )
            return json.dumps(
                {
                    "success": True,
                    "record_date": record_date,
                    "entries": entries,
                    "count": len(entries),
                }
            )
        except Exception as e:
            logger.error(f"get_squad_wellness failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_player_wellness(self, player_id, limit: int = 30):
        """A player's wellness history (readiness trend view)."""
        try:
            self._check_rate_limit("training")
            pid = SecurityValidator.validate_int(player_id)
            entries = await self._services["storage_service"].get_player_wellness(
                pid, limit=max(1, min(int(limit or 30), 365))
            )
            return json.dumps(
                {"success": True, "player_id": pid, "entries": entries, "count": len(entries)}
            )
        except Exception as e:
            logger.error(f"get_player_wellness failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ------------------------------------------------------------------
    # Training sessions and session RPE
    # ------------------------------------------------------------------

    async def create_training_session(
        self, session_date: str, session_type: str, md_offset: str = "", plan_id=None, match_id=None
    ):
        """Create a training session (the morphocycle day container)."""
        try:
            self._check_rate_limit("training")
            if not str(session_date or "").strip():
                return json.dumps({"error": "session_date is required (ISO date)"})
            sid = await self._services["storage_service"].save_training_session(
                session_date=str(session_date),
                session_type=str(session_type or "tactical"),
                md_offset=str(md_offset or ""),
                plan_id=int(plan_id) if plan_id else None,
                match_id=int(match_id) if match_id else None,
            )
            return json.dumps({"success": True, "session_id": sid})
        except Exception as e:
            logger.error(f"create_training_session failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_training_sessions(self, date_from: str = "", date_to: str = "", limit: int = 100):
        """List sessions in a date window (weekly rhythm view)."""
        try:
            self._check_rate_limit("training")
            sessions = await self._services["storage_service"].get_training_sessions(
                date_from=str(date_from or "").strip() or None,
                date_to=str(date_to or "").strip() or None,
                limit=max(1, min(int(limit or 100), 500)),
            )
            return json.dumps({"success": True, "sessions": sessions, "count": len(sessions)})
        except Exception as e:
            logger.error(f"get_training_sessions failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def submit_session_rpe(self, session_id, player_id, rpe, minutes_played):
        """Record one player's session RPE (Borg CR10 0-10) + minutes.

        Session load (sRPE = RPE x minutes) is computed and persisted
        by storage; this is the primary load-monitoring input.
        """
        try:
            self._check_rate_limit("training")
            sid = SecurityValidator.validate_int(session_id)
            pid = SecurityValidator.validate_int(player_id)
            rid = await self._services["storage_service"].save_session_rpe(
                session_id=sid,
                player_id=pid,
                rpe=float(rpe),
                minutes_played=int(minutes_played or 0),
            )
            return json.dumps({"success": True, "rpe_id": rid, "session_id": sid})
        except Exception as e:
            logger.error(f"submit_session_rpe failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_session_rpe(self, session_id):
        """All players' RPE rows for one session."""
        try:
            self._check_rate_limit("training")
            sid = SecurityValidator.validate_int(session_id)
            rows = await self._services["storage_service"].get_session_rpe(sid)
            return json.dumps(
                {"success": True, "session_id": sid, "rpe_rows": rows, "count": len(rows)}
            )
        except Exception as e:
            logger.error(f"get_session_rpe failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ------------------------------------------------------------------
    # Drill feedback (does the drill actually work?)
    # ------------------------------------------------------------------

    async def save_drill_feedback(self, drill_id: str, payload: str | dict):
        """Record coach feedback on a drill's effectiveness (1-5)."""
        try:
            self._check_rate_limit("training")
            did = str(drill_id or "").strip()
            if not did:
                return json.dumps({"error": "drill_id is required"})
            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            try:
                effectiveness = int(data["effectiveness"])
            except (TypeError, ValueError, KeyError):
                return json.dumps({"error": "effectiveness must be an integer 1-5"})
            if not 1 <= effectiveness <= 5:
                return json.dumps({"error": "effectiveness must be within 1-5"})
            fid = await self._services["storage_service"].save_drill_feedback(
                drill_id=did,
                effectiveness=effectiveness,
                session_id=(
                    int(data["session_id"]) if data.get("session_id") not in (None, "") else None
                ),
                player_scope=str(data.get("player_scope") or "squad"),
                observations=str(data.get("observations") or ""),
                targeted_metric=str(data.get("targeted_metric") or ""),
            )
            return json.dumps({"success": True, "feedback_id": fid, "drill_id": did})
        except Exception as e:
            logger.error(f"save_drill_feedback failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_drill_feedback(self, drill_id: str):
        """Feedback history for one drill (effectiveness evidence)."""
        try:
            self._check_rate_limit("training")
            rows = await self._services["storage_service"].get_drill_feedback(
                str(drill_id or "").strip()
            )
            rates = [r["effectiveness"] for r in rows if isinstance(r.get("effectiveness"), int)]
            return json.dumps(
                {
                    "success": True,
                    "drill_id": drill_id,
                    "feedback": rows,
                    "count": len(rows),
                    "mean_effectiveness": (round(sum(rates) / len(rates), 2) if rates else None),
                }
            )
        except Exception as e:
            logger.error(f"get_drill_feedback failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ------------------------------------------------------------------
    # Club game model
    # ------------------------------------------------------------------

    async def save_game_model(self, payload: str | dict):
        """Persist a new game-model version for a team.

        payload keys: team_id, in_possession, out_of_possession,
        transition_attack, transition_defence, non_negotiables, created_by.
        Inserting a new version deactivates previous ones (versioned).
        """
        try:
            self._check_rate_limit("training")
            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            team_id = SecurityValidator.validate_int(data.get("team_id"))
            in_pos = str(data.get("in_possession") or "").strip()
            out_pos = str(data.get("out_of_possession") or "").strip()
            if not in_pos and not out_pos:
                return json.dumps(
                    {"error": "game model needs at least one of in_possession / out_of_possession"}
                )
            nn = data.get("non_negotiables")
            nn_list = [str(x) for x in nn] if isinstance(nn, list) else []
            gid = await self._services["storage_service"].save_game_model(
                team_id=team_id,
                in_possession=in_pos,
                out_of_possession=out_pos,
                transition_attack=str(data.get("transition_attack") or ""),
                transition_defence=str(data.get("transition_defence") or ""),
                non_negotiables=nn_list,
                created_by=str(data.get("created_by") or ""),
            )
            return json.dumps({"success": True, "game_model_id": gid, "team_id": team_id})
        except Exception as e:
            logger.error(f"save_game_model failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_game_model(self, team_id):
        """The team's active game model (honest None when never set)."""
        try:
            self._check_rate_limit("training")
            tid = SecurityValidator.validate_int(team_id)
            gm = await self._services["storage_service"].get_active_game_model(tid)
            return json.dumps({"success": True, "team_id": tid, "game_model": gm})
        except Exception as e:
            logger.error(f"get_game_model failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ------------------------------------------------------------------
    # Sports science: testing battery, IDPs
    # ------------------------------------------------------------------

    async def save_testing_result(self, player_id, payload: str | dict):
        """Record a physical testing battery result (e.g. CMJ 38.5 cm)."""
        try:
            self._check_rate_limit("training")
            pid = SecurityValidator.validate_int(player_id)
            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            test_type = str(data.get("test_type") or "").strip()
            record_date = str(data.get("test_date") or "").strip()
            if not test_type or not record_date:
                return json.dumps({"error": "test_type and test_date are required"})
            try:
                value = float(data["value"])
            except (TypeError, ValueError, KeyError):
                return json.dumps({"error": "value must be numeric"})
            rid = await self._services["storage_service"].save_testing_result(
                player_id=pid,
                test_date=record_date,
                test_type=test_type,
                value=value,
                unit=str(data.get("unit") or ""),
                notes=str(data.get("notes") or ""),
            )
            return json.dumps({"success": True, "testing_id": rid})
        except Exception as e:
            logger.error(f"save_testing_result failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_player_idp(self, player_id, season: str = ""):
        """A player's individual development plan goals."""
        try:
            self._check_rate_limit("training")
            pid = SecurityValidator.validate_int(player_id)
            goals = await self._services["storage_service"].get_player_idp(
                pid, season=str(season or "").strip() or None
            )
            return json.dumps({"success": True, "player_id": pid, "goals": goals})
        except Exception as e:
            logger.error(f"get_player_idp failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def save_idp_goal(self, player_id, payload: str | dict):
        """Add an IDP goal for a player."""
        try:
            self._check_rate_limit("training")
            pid = SecurityValidator.validate_int(player_id)
            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            goal_text = str(data.get("goal_text") or "").strip()
            if not goal_text:
                return json.dumps({"error": "goal_text is required"})
            gid = await self._services["storage_service"].save_idp_goal(
                player_id=pid,
                goal_text=goal_text,
                category=str(data.get("category") or "technical"),
                season=str(data.get("season") or ""),
                target_metric=str(data.get("target_metric") or ""),
                baseline_value=(
                    float(data["baseline_value"])
                    if data.get("baseline_value") not in (None, "")
                    else None
                ),
                target_date=str(data.get("target_date") or ""),
            )
            return json.dumps({"success": True, "idp_id": gid})
        except Exception as e:
            logger.error(f"save_idp_goal failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ------------------------------------------------------------------
    # Nutrition and psychology daily check-ins
    # ------------------------------------------------------------------

    async def save_nutrition_log(self, player_id, payload: str | dict):
        """Record a nutrition log entry (meal type + hydration/fueling)."""
        try:
            self._check_rate_limit("training")
            pid = SecurityValidator.validate_int(player_id)
            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            log_date = str(data.get("log_date") or "").strip()
            meal_type = str(data.get("meal_type") or "").strip()
            if not log_date or not meal_type:
                return json.dumps({"error": "log_date and meal_type are required"})
            nid = await self._services["storage_service"].save_nutrition_log(
                player_id=pid,
                log_date=log_date,
                meal_type=meal_type,
                hydration_score=_clamp_scale(data.get("hydration_score"), "hydration", 3),
                fueling_score=_clamp_scale(data.get("fueling_score"), "fueling", 3),
                notes=str(data.get("notes") or ""),
            )
            return json.dumps({"success": True, "nutrition_id": nid})
        except Exception as e:
            logger.error(f"save_nutrition_log failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def save_psych_checkin(self, player_id, payload: str | dict):
        """Record a psychology check-in (1-5 scales, flaggable)."""
        try:
            self._check_rate_limit("training")
            pid = SecurityValidator.validate_int(player_id)
            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            checkin_date = str(data.get("checkin_date") or "").strip()
            if not checkin_date:
                return json.dumps({"error": "checkin_date is required"})
            mid = await self._services["storage_service"].save_psych_checkin(
                player_id=pid,
                checkin_date=checkin_date,
                confidence=_clamp_scale(data.get("confidence"), "confidence", 3),
                focus=_clamp_scale(data.get("focus"), "focus", 3),
                motivation=_clamp_scale(data.get("motivation"), "motivation", 3),
                anxiety=_clamp_scale(data.get("anxiety"), "anxiety", 3),
                notes=str(data.get("notes") or ""),
                flag_for_followup=bool(data.get("flag_for_followup")),
            )
            return json.dumps({"success": True, "checkin_id": mid})
        except Exception as e:
            logger.error(f"save_psych_checkin failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ------------------------------------------------------------------
    # Staff rituals (the operating rhythm)
    # ------------------------------------------------------------------

    async def save_ritual(self, ritual_type: str, scheduled_for: str, payload: str | dict = ""):
        """Upsert a staff ritual checklist (e.g. wellness_review, MD-1)."""
        try:
            self._check_rate_limit("training")
            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            rid = await self._services["storage_service"].save_ritual(
                ritual_type=str(ritual_type or "").strip(),
                scheduled_for=str(scheduled_for or "").strip(),
                checklist_state=data.get("checklist_state")
                if isinstance(data.get("checklist_state"), list)
                else [],
                notes=str(data.get("notes") or ""),
            )
            return json.dumps({"success": True, "ritual_id": rid})
        except Exception as e:
            logger.error(f"save_ritual failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def complete_ritual(self, ritual_type: str, scheduled_for: str, payload: str | dict = ""):
        """Mark a staff ritual complete (records completed_by)."""
        try:
            self._check_rate_limit("training")
            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            await self._services["storage_service"].complete_ritual(
                ritual_type=str(ritual_type or "").strip(),
                scheduled_for=str(scheduled_for or "").strip(),
                completed_by=str(data.get("completed_by") or ""),
                checklist_state=data.get("checklist_state")
                if isinstance(data.get("checklist_state"), list)
                else [],
                notes=str(data.get("notes") or ""),
            )
            return json.dumps({"success": True})
        except Exception as e:
            logger.error(f"complete_ritual failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_rituals(self, scheduled_for: str = "", ritual_type: str = ""):
        """Rituals for a date/type (the staff's operating-rhythm view)."""
        try:
            self._check_rate_limit("training")
            rituals = await self._services["storage_service"].get_rituals(
                scheduled_for=str(scheduled_for or "").strip() or None,
                ritual_type=str(ritual_type or "").strip() or None,
            )
            return json.dumps({"success": True, "rituals": rituals, "count": len(rituals)})
        except Exception as e:
            logger.error(f"get_rituals failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ------------------------------------------------------------------
    # Minutes management and medical clearance
    # ------------------------------------------------------------------

    async def save_minutes_entry(self, player_id, payload: str | dict):
        """Record a match minutes entry (minutes management).

        payload keys: match_id, minutes_played, started (bool),
        age_phase / bio_band (optional youth-load annotations).
        """
        try:
            self._check_rate_limit("training")
            pid = SecurityValidator.validate_int(player_id)
            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            match_id = SecurityValidator.validate_int(data.get("match_id"))
            minutes = int(data.get("minutes_played") or 0)
            if minutes < 0 or minutes > 130:
                return json.dumps({"error": "minutes_played must be within 0-130"})
            mid = await self._services["storage_service"].save_minutes_entry(
                player_id=pid,
                match_id=match_id,
                minutes_played=minutes,
                started=bool(data.get("started")),
                age_phase=str(data.get("age_phase") or ""),
                bio_band=str(data.get("bio_band") or ""),
            )
            return json.dumps({"success": True, "minutes_id": mid})
        except Exception as e:
            logger.error(f"save_minutes_entry failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def set_medical_clearance(self, player_id, payload: str | dict):
        """Set a player's medical clearance status (fit/limited/unavailable)."""
        try:
            self._check_rate_limit("training")
            pid = SecurityValidator.validate_int(player_id)
            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            status = str(data.get("status") or "").strip()
            if status not in ("fit", "limited", "unavailable"):
                return json.dumps({"error": "status must be fit | limited | unavailable"})
            mid = await self._services["storage_service"].set_medical_clearance(
                player_id=pid,
                status=status,
                cleared_by=str(data.get("cleared_by") or ""),
                reason=str(data.get("reason") or ""),
                source=str(data.get("source") or "manual"),
                effective_until=str(data.get("effective_until") or ""),
            )
            return json.dumps({"success": True, "clearance_id": mid})
        except Exception as e:
            logger.error(f"set_medical_clearance failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ------------------------------------------------------------------
    # Sports science (Phase C): load, testing, maturation, protocols
    # ------------------------------------------------------------------

    async def get_player_load_state(self, player_id):
        """Triangulated sRPE/GPS load state with descriptive flags."""
        try:
            self._check_rate_limit()
            from kawkab.services.load_monitoring_service import LoadMonitoringService

            pid = SecurityValidator.validate_int(player_id)
            state = await LoadMonitoringService(
                self._services["storage_service"]
            ).player_load_state(pid)
            return json.dumps({"success": True, **state})
        except Exception as e:
            logger.error(f"get_player_load_state failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def interpret_testing_result(self, player_id, payload: str | dict):
        """Interpret a testing result against the club's own distribution."""
        try:
            self._check_rate_limit()
            from kawkab.services.testing_battery_service import TestingBatteryService

            pid = SecurityValidator.validate_int(player_id)
            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            test_type = str(data.get("test_type") or "").strip()
            if not test_type:
                return json.dumps({"error": "test_type is required"})
            try:
                value = float(data["value"])
            except (TypeError, ValueError, KeyError):
                return json.dumps({"error": "value must be numeric"})
            svc = TestingBatteryService(self._services["storage_service"])
            out = await svc.interpret_result(
                pid, test_type, value, str(data.get("test_date") or "") or None
            )
            return json.dumps({"success": True, **out})
        except Exception as e:
            logger.error(f"interpret_testing_result failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_battery_summary(self, player_id):
        """Per-test-type trend summary for a player's testing battery."""
        try:
            self._check_rate_limit()
            from kawkab.services.testing_battery_service import TestingBatteryService

            pid = SecurityValidator.validate_int(player_id)
            out = await TestingBatteryService(self._services["storage_service"]).battery_summary(
                pid
            )
            return json.dumps({"success": True, **out})
        except Exception as e:
            logger.error(f"get_battery_summary failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def estimate_maturity_offset(self, payload: str | dict):
        """PHV/maturity-offset estimate (Mirwald), validity provenance included."""
        try:
            self._check_rate_limit()
            from kawkab.services.maturation_service import MaturationService

            data = json.loads(payload) if isinstance(payload, str) else dict(payload or {})
            try:
                age = float(data["age_years"])
                standing = float(data["standing_height_cm"])
                sitting = float(data["sitting_height_cm"])
            except (TypeError, ValueError, KeyError):
                return json.dumps(
                    {
                        "error": "age_years, standing_height_cm, sitting_height_cm are required numbers"
                    }
                )
            out = MaturationService().estimate_offset(
                age, standing, sitting, sex=str(data.get("sex") or "male")
            )
            return json.dumps({"success": out.get("maturity_offset") is not None, **out})
        except Exception as e:
            logger.error(f"estimate_maturity_offset failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_squad_readiness(self, record_date: str):
        """Morning-ritual readiness summary (conversation flags only)."""
        try:
            self._check_rate_limit()
            from kawkab.services.player_protocol_service import PlayerProtocolService

            out = await PlayerProtocolService(
                self._services["storage_service"]
            ).squad_readiness_summary(str(record_date or "").strip())
            return json.dumps({"success": True, **out})
        except Exception as e:
            logger.error(f"get_squad_readiness failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_player_protocol_flags(self, player_id):
        """Wellness/psych/nutrition conversation flags for one player."""
        try:
            self._check_rate_limit()
            from kawkab.services.player_protocol_service import PlayerProtocolService

            pid = SecurityValidator.validate_int(player_id)
            svc = PlayerProtocolService(self._services["storage_service"])
            psych = await svc.psych_flags(pid)
            nutrition = await svc.nutrition_flags(pid)
            return json.dumps(
                {"success": True, "player_id": pid, "psych": psych, "nutrition": nutrition}
            )
        except Exception as e:
            logger.error(f"get_player_protocol_flags failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_squad_overview(self, record_date: str, player_ids_json: str):
        """The staff's daily decision view: availability % + readiness flags.

        Merges SquadAvailabilityService (hard medical gates) with the
        readiness summary (conversation flags) — one honest picture of
        who can train today and who needs a conversation first.
        """
        try:
            self._check_rate_limit()
            from kawkab.services.availability_service import SquadAvailabilityService
            from kawkab.services.player_protocol_service import PlayerProtocolService

            raw_ids = (
                json.loads(player_ids_json)
                if isinstance(player_ids_json, str)
                else list(player_ids_json or [])
            )
            if not isinstance(raw_ids, list):
                return json.dumps({"error": "player_ids must be a JSON array"})
            player_ids = [SecurityValidator.validate_int(p) for p in raw_ids]

            availability = await SquadAvailabilityService(
                self._services["storage_service"]
            ).squad_availability(player_ids)
            readiness = await PlayerProtocolService(
                self._services["storage_service"]
            ).squad_readiness_summary(str(record_date or "").strip())
            total = len(player_ids)
            availability_pct = (
                round(100.0 * availability["available_count"] / total, 1) if total else 0.0
            )
            return json.dumps(
                {
                    "success": True,
                    "record_date": record_date,
                    "availability_pct": availability_pct,
                    "available_count": availability["available_count"],
                    "unavailable_count": availability["unavailable_count"],
                    "limited_count": availability["limited_count"],
                    "players": availability["players"],
                    "readiness": {
                        "submissions": readiness["submissions"],
                        "mean_wellness": readiness["mean_wellness"],
                        "flagged_for_conversation": readiness["flagged_for_conversation"],
                    },
                    "provenance": {
                        "availability": "hard gates: medical clearance + concussion RTP",
                        "readiness": "advisory conversation flags only",
                    },
                }
            )
        except Exception as e:
            logger.error(f"get_squad_overview failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ------------------------------------------------------------------
    # Academy (Phase D / D1)
    # ------------------------------------------------------------------

    async def get_academy_squad_phases(self, ref_date: str = ""):
        """EPPP-mapped development phases for the registered squad."""
        try:
            self._check_rate_limit("training")
            from kawkab.services.academy_service import AcademyService

            report = await AcademyService(self._services["storage_service"]).squad_phases(
                str(ref_date or "").strip() or None
            )
            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logger.error(f"get_academy_squad_phases failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_bio_banded_groups(self, measurements_json: str = ""):
        """Bio-banded training groups from somatic measurements (JSON list)."""
        try:
            self._check_rate_limit("training")
            from kawkab.services.academy_service import AcademyService

            measurements = json.loads(measurements_json or "[]")
            if not isinstance(measurements, list):
                return json.dumps({"error": "measurements_json must be a JSON array"})
            report = await AcademyService(self._services["storage_service"]).bio_banded_groups(
                measurements
            )
            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logger.error(f"get_bio_banded_groups failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_minutes_management(self, player_id):
        """Development-minutes management for one player (age-phase caps)."""
        try:
            self._check_rate_limit("training")
            pid = SecurityValidator.validate_int(player_id)
            from kawkab.services.academy_service import AcademyService

            report = await AcademyService(self._services["storage_service"]).minutes_management(pid)
            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logger.error(f"get_minutes_management failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_academy_selection_view(self, player_ids_json: str = ""):
        """Safeguarding-aware selection view for a player-id list."""
        try:
            self._check_rate_limit("training")
            ids_raw = json.loads(player_ids_json or "[]")
            if not isinstance(ids_raw, list):
                return json.dumps({"error": "player_ids_json must be a JSON array"})
            ids = [int(i) for i in ids_raw]
            from kawkab.services.academy_service import AcademyService

            report = await AcademyService(self._services["storage_service"]).academy_selection_view(
                ids
            )
            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logger.error(f"get_academy_selection_view failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ------------------------------------------------------------------
    # Operating program (Phase D / D3)
    # ------------------------------------------------------------------

    async def get_current_program(self):
        """The club's published weekly rhythm, RACI, and KPI tree."""
        try:
            self._check_rate_limit("training")
            from kawkab.services.operating_program_service import OperatingProgramService

            report = await OperatingProgramService(
                self._services["storage_service"]
            ).current_program()
            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logger.error(f"get_current_program failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def publish_program_document(self, doc_type: str, payload: str | dict = ""):
        """Publish one program document version.

        doc_type=weekly_rhythm takes a JSON *array* of day entries;
        raci and kpi_tree take JSON objects.
        """
        try:
            self._check_rate_limit("training")
            from kawkab.services.operating_program_service import OperatingProgramService

            svc = OperatingProgramService(self._services["storage_service"])
            if doc_type == "weekly_rhythm":
                week = json.loads(payload) if isinstance(payload, str) else payload
                if not isinstance(week, list):
                    return json.dumps(
                        {"error": "weekly_rhythm payload must be a JSON array of day entries"}
                    )
                result = await svc.publish_weekly_rhythm(week)
            elif doc_type == "raci":
                data = json.loads(payload) if isinstance(payload, str) else payload
                roles = data.get("roles") if isinstance(data, dict) else data
                if not isinstance(roles, list):
                    return json.dumps(
                        {
                            "error": 'raci payload must be a JSON array of roles (or {"roles": [...]})'
                        }
                    )
                result = await svc.publish_raci(roles)
            elif doc_type == "kpi_tree":
                data = json.loads(payload) if isinstance(payload, str) else payload
                kpis = data.get("kpis") if isinstance(data, dict) else data
                if not isinstance(kpis, list):
                    return json.dumps(
                        {
                            "error": 'kpi_tree payload must be a JSON array of KPIs (or {"kpis": [...]})'
                        }
                    )
                result = await svc.publish_kpi_tree(kpis)
            else:
                return json.dumps({"error": "doc_type must be weekly_rhythm | raci | kpi_tree"})
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            logger.error(f"publish_program_document failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def instantiate_week_rituals(self, week_start_date: str):
        """Schedule the published rhythm's rituals for the week starting Monday."""
        try:
            self._check_rate_limit("training")
            from kawkab.services.operating_program_service import OperatingProgramService

            report = await OperatingProgramService(
                self._services["storage_service"]
            ).instantiate_week_rituals(str(week_start_date or "").strip())
            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logger.error(f"instantiate_week_rituals failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_kpi_snapshot(self):
        """Live KPI snapshot over the published tree (measured or not_measured)."""
        try:
            self._check_rate_limit("training")
            from kawkab.services.operating_program_service import OperatingProgramService

            report = await OperatingProgramService(self._services["storage_service"]).kpi_snapshot()
            return json.dumps(report, ensure_ascii=False)
        except Exception as e:
            logger.error(f"get_kpi_snapshot failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ------------------------------------------------------------------
    # Match Week loop: diagnosis -> plan -> re-check
    # ------------------------------------------------------------------

    async def _match_week_service(self):
        """Lazily built; one instance per handler so the KB loads once."""
        service = getattr(self, "_match_week", None)
        if service is None:
            from kawkab.services.match_week_service import MatchWeekService

            service = MatchWeekService(
                self._services.get("storage_service"),
                await self._knowledge(),
                self._services.get("homography_service"),
            )
            self._match_week = service
        return service

    async def get_match_week_state(self, match_id):
        """The persisted loop state for one match (stage, plan, re-check)."""
        try:
            self._check_rate_limit("training")
            mid = SecurityValidator.validate_match_id(match_id)
            state = await (await self._match_week_service()).state(mid)
            return json.dumps({"success": True, **state.to_dict()}, ensure_ascii=False)
        except Exception as e:
            logger.error(f"get_match_week_state failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
