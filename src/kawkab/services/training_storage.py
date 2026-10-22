"""Training Operating System storage methods.

The closed-loop training program: game model → plans → sessions → drills →
execution records → wellness/sRPE → re-measurement.

Every method follows the storage honesty contract:
- Uninitialized backend raises :class:`StorageNotInitializedError` (never
  returns empty, never fabricates data).
- Write failures raise :class:`StorageWriteError` (never swallowed).

Postgres parity note: these methods are SQLite-first like the shortlist/
injury-tracker precedent — the Postgres adapter gains the same methods via
the parity suite before any cloud feature consumes them.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.services.storage_errors import (
    StorageNotInitializedError,
    StorageReadError,
    StorageWriteError,
)

logger = get_logger(__name__)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def _load_json_list(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _load_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


class TrainingStorageMixin:
    """Training-OS CRUD. Mixed into StorageService (see storage_service.py)."""

    # Shared backend state, declared here for type checking only: the
    # owning StorageService assigns these in its __init__ (mixin methods
    # must go through StorageNotInitializedError guards, never assume
    # they are set).
    _conn: sqlite3.Connection | None
    _db_path: Path | None
    _use_postgres: bool

    # ── Game model ──────────────────────────────────────────────────────

    async def save_game_model(
        self,
        team_id: int,
        in_possession: str,
        out_of_possession: str,
        transition_attack: str = "",
        transition_defence: str = "",
        set_pieces_off: str = "",
        set_pieces_def: str = "",
        player_roles: list[dict] | None = None,
        non_negotiables: list[str] | None = None,
        language: str = "en",
        created_by: str = "",
        notes: str = "",
    ) -> int:
        """Insert a new game-model version and deactivate previous versions."""
        if self._conn is None:
            raise StorageNotInitializedError("save_game_model")
        try:
            cur = self._conn.cursor()
            cur.execute("UPDATE game_model SET is_active = 0 WHERE team_id = ?", (team_id,))
            cur.execute(
                """INSERT INTO game_model
                   (team_id, version, is_active, in_possession, out_of_possession,
                    transition_attack, transition_defence, set_pieces_off, set_pieces_def,
                    player_roles, non_negotiables, language, created_by, notes)
                   VALUES (?, COALESCE((SELECT MAX(version) FROM game_model WHERE team_id = ?), 0) + 1,
                           1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    team_id,
                    team_id,
                    in_possession,
                    out_of_possession,
                    transition_attack,
                    transition_defence,
                    set_pieces_off,
                    set_pieces_def,
                    json.dumps(player_roles or []),
                    json.dumps(non_negotiables or []),
                    language,
                    created_by,
                    notes,
                ),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except Exception as e:
            logger.warning(f"save_game_model failed: {e}")
            raise StorageWriteError("save_game_model", e) from e

    async def get_active_game_model_for_match(self, match_id: int) -> dict[str, Any] | None:
        """Active game model via a match's home_team name -> team_id.

        Convenience for bridge handlers that only hold a match id:
        resolves the match's home team name to a club team id and loads
        that team's active game model. Returns None when the match or
        the team's game model does not exist.
        """
        if self._conn is None:
            raise StorageNotInitializedError("get_active_game_model_for_match")
        row = self._conn.execute(
            "SELECT home_team FROM matches WHERE id = ?",
            (match_id,),
        ).fetchone()
        if not row or not str(row[0] or "").strip():
            return None
        team_row = self._conn.execute(
            "SELECT id FROM teams WHERE name = ?", (str(row[0]),)
        ).fetchone()
        if not team_row:
            return None
        return await self.get_active_game_model(int(team_row[0]))

    async def get_active_game_model(self, team_id: int) -> dict[str, Any] | None:
        if self._conn is None:
            raise StorageNotInitializedError("get_active_game_model")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, team_id, version, is_active, in_possession, out_of_possession,
                      transition_attack, transition_defence, set_pieces_off, set_pieces_def,
                      player_roles, non_negotiables, language, created_by, created_at, notes
               FROM game_model WHERE team_id = ? AND is_active = 1
               ORDER BY version DESC LIMIT 1""",
            (team_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        out = _row_to_dict(row)
        out["player_roles"] = _load_json_list(out.get("player_roles"))
        out["non_negotiables"] = _load_json_list(out.get("non_negotiables"))
        return out

    async def get_game_model_history(self, team_id: int) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_game_model_history")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, team_id, version, is_active, created_at, created_by, notes
               FROM game_model WHERE team_id = ? ORDER BY version DESC""",
            (team_id,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    # ── Training plans ──────────────────────────────────────────────────

    async def save_training_plan(
        self,
        match_id: int,
        title: str,
        payload: dict[str, Any],
        duration_weeks: int = 4,
        priority_diagnoses: list[dict] | None = None,
        source: str = "reasoning_engine",
        created_by: str = "",
        status: str = "draft",
    ) -> int:
        if self._conn is None:
            raise StorageNotInitializedError("save_training_plan")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO training_plans
                   (match_id, title, status, duration_weeks, priority_diagnoses,
                    payload, source, created_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    match_id,
                    title,
                    status,
                    duration_weeks,
                    json.dumps(priority_diagnoses or []),
                    json.dumps(payload),
                    source,
                    created_by,
                ),
            )
            cur.execute(
                """INSERT INTO plan_versions (plan_id, version, change_note, payload)
                   VALUES (?, 1, 'initial', ?)""",
                (cur.lastrowid, json.dumps(payload)),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except Exception as e:
            logger.warning(f"save_training_plan failed: {e}")
            raise StorageWriteError("save_training_plan", e) from e

    async def update_training_plan_payload(
        self, plan_id: int, payload: dict[str, Any], change_note: str = ""
    ) -> None:
        """Coach edit: bump the plan payload and record a version row."""
        if self._conn is None:
            raise StorageNotInitializedError("update_training_plan_payload")
        try:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT COALESCE(MAX(version), 0) FROM plan_versions WHERE plan_id = ?",
                (plan_id,),
            )
            row = cur.fetchone()
            next_version = (row[0] if row else 0) + 1
            cur.execute(
                "UPDATE training_plans SET payload = ?, updated_at = datetime('now') WHERE id = ?",
                (json.dumps(payload), plan_id),
            )
            cur.execute(
                "INSERT INTO plan_versions (plan_id, version, change_note, payload) VALUES (?, ?, ?, ?)",
                (plan_id, next_version, change_note, json.dumps(payload)),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"update_training_plan_payload failed: {e}")
            raise StorageWriteError("update_training_plan_payload", e) from e

    async def set_training_plan_status(self, plan_id: int, status: str) -> None:
        if status not in ("draft", "active", "completed", "cancelled"):
            raise ValueError(f"invalid plan status: {status}")
        if self._conn is None:
            raise StorageNotInitializedError("set_training_plan_status")
        try:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE training_plans SET status = ?, updated_at = datetime('now') WHERE id = ?",
                (status, plan_id),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"set_training_plan_status failed: {e}")
            raise StorageWriteError("set_training_plan_status", e) from e

    async def get_training_plan(self, plan_id: int) -> dict[str, Any] | None:
        if self._conn is None:
            raise StorageNotInitializedError("get_training_plan")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, match_id, title, status, duration_weeks, priority_diagnoses,
                      payload, source, created_by, created_at, updated_at
               FROM training_plans WHERE id = ?""",
            (plan_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        out = _row_to_dict(row)
        out["priority_diagnoses"] = _load_json_list(out.get("priority_diagnoses"))
        out["payload"] = _load_json_dict(out.get("payload"))
        return out

    async def get_training_plans_for_match(self, match_id: int) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_training_plans_for_match")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, match_id, title, status, duration_weeks, source, created_at, updated_at
               FROM training_plans WHERE match_id = ? ORDER BY created_at DESC""",
            (match_id,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    async def list_training_plans(
        self, status: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("list_training_plans")
        cur = self._conn.cursor()
        if status:
            cur.execute(
                """SELECT id, match_id, title, status, duration_weeks, source, created_at, updated_at
                   FROM training_plans WHERE status = ? ORDER BY created_at DESC LIMIT ?""",
                (status, limit),
            )
        else:
            cur.execute(
                """SELECT id, match_id, title, status, duration_weeks, source, created_at, updated_at
                   FROM training_plans ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            )
        return [_row_to_dict(r) for r in cur.fetchall()]

    async def get_plan_versions(self, plan_id: int) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_plan_versions")
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, plan_id, version, change_note, payload, created_at FROM plan_versions WHERE plan_id = ? ORDER BY version DESC",
            (plan_id,),
        )
        rows = [_row_to_dict(r) for r in cur.fetchall()]
        for r in rows:
            r["payload"] = _load_json_dict(r.get("payload"))
        return rows

    # ── Sessions + drills ───────────────────────────────────────────────

    async def save_training_session(
        self,
        session_date: str,
        session_type: str,
        md_offset: str = "",
        plan_id: int | None = None,
        match_id: int | None = None,
        theme: str = "",
        duration_min: int = 90,
        intensity: str = "medium",
        notes: str = "",
    ) -> int:
        if self._conn is None:
            raise StorageNotInitializedError("save_training_session")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO training_sessions
                   (plan_id, match_id, session_date, md_offset, session_type, theme,
                    duration_min, intensity, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    plan_id,
                    match_id,
                    session_date,
                    md_offset,
                    session_type,
                    theme,
                    duration_min,
                    intensity,
                    notes,
                ),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except Exception as e:
            logger.warning(f"save_training_session failed: {e}")
            raise StorageWriteError("save_training_session", e) from e

    async def set_session_status(self, session_id: int, status: str) -> None:
        if status not in ("planned", "completed", "cancelled"):
            raise ValueError(f"invalid session status: {status}")
        if self._conn is None:
            raise StorageNotInitializedError("set_session_status")
        try:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE training_sessions SET status = ? WHERE id = ?", (status, session_id)
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"set_session_status failed: {e}")
            raise StorageWriteError("set_session_status", e) from e

    async def get_training_sessions(
        self,
        date_from: str | None = None,
        date_to: str | None = None,
        plan_id: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_training_sessions")
        cur = self._conn.cursor()
        clauses: list[str] = []
        params: list[Any] = []
        if date_from:
            clauses.append("session_date >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("session_date <= ?")
            params.append(date_to)
        if plan_id is not None:
            clauses.append("plan_id = ?")
            params.append(plan_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        cur.execute(
            f"""SELECT id, plan_id, match_id, session_date, md_offset, session_type, theme,
                       duration_min, intensity, status, notes, created_at
                FROM training_sessions {where} ORDER BY session_date DESC LIMIT ?""",
            params,
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    async def get_training_session(self, session_id: int) -> dict[str, Any] | None:
        if self._conn is None:
            raise StorageNotInitializedError("get_training_session")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, plan_id, match_id, session_date, md_offset, session_type, theme,
                      duration_min, intensity, status, notes, created_at
               FROM training_sessions WHERE id = ?""",
            (session_id,),
        )
        row = cur.fetchone()
        return _row_to_dict(row) if row else None

    async def add_session_drill(
        self,
        session_id: int,
        drill_id: str,
        order_index: int = 0,
        duration_min: int = 15,
    ) -> int:
        if self._conn is None:
            raise StorageNotInitializedError("add_session_drill")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO session_drills (session_id, drill_id, order_index, duration_min)
                   VALUES (?, ?, ?, ?)""",
                (session_id, drill_id, order_index, duration_min),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except Exception as e:
            logger.warning(f"add_session_drill failed: {e}")
            raise StorageWriteError("add_session_drill", e) from e

    async def set_drill_execution(
        self,
        session_drill_id: int,
        executed: int,
        executed_as: str = "",
        coach_note: str = "",
    ) -> None:
        """Record what actually happened to a planned drill.

        ``executed``: 0 planned, 1 executed, 2 modified, 3 skipped.
        """
        if executed not in (0, 1, 2, 3):
            raise ValueError(f"invalid executed flag: {executed}")
        if self._conn is None:
            raise StorageNotInitializedError("set_drill_execution")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """UPDATE session_drills SET executed = ?, executed_as = ?, coach_note = ?
                   WHERE id = ?""",
                (executed, executed_as, coach_note, session_drill_id),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"set_drill_execution failed: {e}")
            raise StorageWriteError("set_drill_execution", e) from e

    async def get_session_drills(self, session_id: int) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_session_drills")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, session_id, drill_id, order_index, duration_min, executed,
                      executed_as, coach_note
               FROM session_drills WHERE session_id = ? ORDER BY order_index""",
            (session_id,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    # ── Attendance ──────────────────────────────────────────────────────

    async def save_attendance(
        self,
        session_id: int,
        player_id: int,
        status: str,
        minutes_participated: int = 0,
        note: str = "",
    ) -> int:
        if status not in ("present", "absent", "injured", "ill", "excused", "late"):
            raise ValueError(f"invalid attendance status: {status}")
        if self._conn is None:
            raise StorageNotInitializedError("save_attendance")
        try:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT id FROM attendance WHERE session_id = ? AND player_id = ?",
                (session_id, player_id),
            )
            existing = cur.fetchone()
            if existing:
                cur.execute(
                    """UPDATE attendance SET status = ?, minutes_participated = ?, note = ?
                       WHERE id = ?""",
                    (status, minutes_participated, note, existing[0]),
                )
                self._conn.commit()
                return int(existing[0])
            cur.execute(
                """INSERT INTO attendance (session_id, player_id, status, minutes_participated, note)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, player_id, status, minutes_participated, note),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except Exception as e:
            logger.warning(f"save_attendance failed: {e}")
            raise StorageWriteError("save_attendance", e) from e

    async def get_session_attendance(self, session_id: int) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_session_attendance")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, session_id, player_id, status, minutes_participated, note
               FROM attendance WHERE session_id = ? ORDER BY player_id""",
            (session_id,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    # ── Session RPE (Foster sRPE) ───────────────────────────────────────

    async def save_session_rpe(
        self,
        session_id: int,
        player_id: int,
        rpe: float,
        minutes_played: int,
    ) -> int:
        if self._conn is None:
            raise StorageNotInitializedError("save_session_rpe")
        if not 0.0 <= rpe <= 10.0:
            raise ValueError("rpe must be within 0-10 (Borg CR10)")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO session_rpe (session_id, player_id, rpe, minutes_played, load)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, player_id, rpe, minutes_played, rpe * minutes_played),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except Exception as e:
            logger.warning(f"save_session_rpe failed: {e}")
            raise StorageWriteError("save_session_rpe", e) from e

    async def get_session_rpe(self, session_id: int) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_session_rpe")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, session_id, player_id, rpe, minutes_played, load, recorded_at
               FROM session_rpe WHERE session_id = ? ORDER BY player_id""",
            (session_id,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    async def get_player_rpe_history(self, player_id: int, limit: int = 30) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_player_rpe_history")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT r.session_id, r.rpe, r.minutes_played, r.load, r.recorded_at, s.session_date
               FROM session_rpe r JOIN training_sessions s ON s.id = r.session_id
               WHERE r.player_id = ? ORDER BY r.recorded_at DESC LIMIT ?""",
            (player_id, limit),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    # ── Wellness (Hooper-style 5-item) ──────────────────────────────────

    async def save_wellness(
        self,
        player_id: int,
        record_date: str,
        sleep_quality: int,
        fatigue: int,
        soreness: int,
        stress: int,
        mood: int,
        source: str = "manual",
    ) -> int:
        """Upsert today's wellness entry; stores the normalized mean score.

        Scale: 1-5 with 5 = best (sleep well, fresh, no soreness/stress,
        good mood) — the convention Hooper-index questionnaires use.
        """
        if self._conn is None:
            raise StorageNotInitializedError("save_wellness")
        for name, v in (
            ("sleep_quality", sleep_quality),
            ("fatigue", fatigue),
            ("soreness", soreness),
            ("stress", stress),
            ("mood", mood),
        ):
            if not 1 <= int(v) <= 5:
                raise ValueError(f"{name} must be within 1-5")
        score = (sleep_quality + fatigue + soreness + stress + mood) / 5.0
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO wellness
                   (player_id, record_date, sleep_quality, fatigue, soreness, stress, mood,
                    wellness_score, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(player_id, record_date)
                   DO UPDATE SET sleep_quality=excluded.sleep_quality,
                       fatigue=excluded.fatigue, soreness=excluded.soreness,
                       stress=excluded.stress, mood=excluded.mood,
                       wellness_score=excluded.wellness_score, source=excluded.source,
                       recorded_at=datetime('now')""",
                (
                    player_id,
                    record_date,
                    sleep_quality,
                    fatigue,
                    soreness,
                    stress,
                    mood,
                    score,
                    source,
                ),
            )
            self._conn.commit()
            cur.execute(
                "SELECT id FROM wellness WHERE player_id = ? AND record_date = ?",
                (player_id, record_date),
            )
            row = cur.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.warning(f"save_wellness failed: {e}")
            raise StorageWriteError("save_wellness", e) from e

    async def get_player_wellness(self, player_id: int, limit: int = 30) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_player_wellness")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, player_id, record_date, sleep_quality, fatigue, soreness, stress,
                      mood, wellness_score, source, recorded_at
               FROM wellness WHERE player_id = ? ORDER BY record_date DESC LIMIT ?""",
            (player_id, limit),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    async def get_squad_wellness_latest(self, record_date: str) -> list[dict[str, Any]]:
        """Every player's wellness for one date (the morning huddle view)."""
        if self._conn is None:
            raise StorageNotInitializedError("get_squad_wellness_latest")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT w.id, w.player_id, w.record_date, w.sleep_quality, w.fatigue,
                      w.soreness, w.stress, w.mood, w.wellness_score, w.source
               FROM wellness w
               WHERE w.record_date = ?
               ORDER BY w.wellness_score ASC""",
            (record_date,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    # ── Drill feedback ──────────────────────────────────────────────────

    async def save_drill_feedback(
        self,
        drill_id: str,
        effectiveness: int,
        session_id: int | None = None,
        player_scope: str = "squad",
        observations: str = "",
        targeted_metric: str = "",
    ) -> int:
        if not 1 <= effectiveness <= 5:
            raise ValueError("effectiveness must be within 1-5")
        if self._conn is None:
            raise StorageNotInitializedError("save_drill_feedback")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO drill_feedback
                   (drill_id, session_id, player_scope, effectiveness, observations, targeted_metric)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (drill_id, session_id, player_scope, effectiveness, observations, targeted_metric),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except Exception as e:
            logger.warning(f"save_drill_feedback failed: {e}")
            raise StorageWriteError("save_drill_feedback", e) from e

    async def get_drill_feedback(self, drill_id: str) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_drill_feedback")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, drill_id, session_id, player_scope, effectiveness, observations,
                      targeted_metric, created_at
               FROM drill_feedback WHERE drill_id = ? ORDER BY created_at DESC""",
            (drill_id,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    # ── Testing battery ─────────────────────────────────────────────────

    async def save_testing_result(
        self,
        player_id: int,
        test_date: str,
        test_type: str,
        value: float,
        unit: str = "",
        notes: str = "",
    ) -> int:
        if self._conn is None:
            raise StorageNotInitializedError("save_testing_result")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO testing_results (player_id, test_date, test_type, value, unit, notes)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (player_id, test_date, test_type, value, unit, notes),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except Exception as e:
            logger.warning(f"save_testing_result failed: {e}")
            raise StorageWriteError("save_testing_result", e) from e

    async def get_player_testing_history(
        self, player_id: int, test_type: str | None = None
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_player_testing_history")
        cur = self._conn.cursor()
        if test_type:
            cur.execute(
                """SELECT id, player_id, test_date, test_type, value, unit, percentile, notes
                   FROM testing_results WHERE player_id = ? AND test_type = ?
                   ORDER BY test_date DESC""",
                (player_id, test_type),
            )
        else:
            cur.execute(
                """SELECT id, player_id, test_date, test_type, value, unit, percentile, notes
                   FROM testing_results WHERE player_id = ? ORDER BY test_date DESC""",
                (player_id,),
            )
        return [_row_to_dict(r) for r in cur.fetchall()]

    async def get_testing_distribution(self, test_type: str) -> list[dict[str, Any]]:
        """All players' latest value per test — the raw material for percentiles."""
        if self._conn is None:
            raise StorageNotInitializedError("get_testing_distribution")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT player_id, value FROM testing_results
               WHERE test_type = ?
               AND id IN (SELECT MAX(id) FROM testing_results WHERE test_type = ? GROUP BY player_id)
               ORDER BY value DESC""",
            (test_type, test_type),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    # ── IDP goals ───────────────────────────────────────────────────────

    async def save_idp_goal(
        self,
        player_id: int,
        goal_text: str,
        category: str = "technical",
        season: str = "",
        target_metric: str = "",
        baseline_value: float | None = None,
        target_date: str = "",
    ) -> int:
        if self._conn is None:
            raise StorageNotInitializedError("save_idp_goal")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO idp_goals
                   (player_id, season, category, goal_text, target_metric, baseline_value, target_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    player_id,
                    season,
                    category,
                    goal_text,
                    target_metric,
                    baseline_value,
                    target_date,
                ),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except Exception as e:
            logger.warning(f"save_idp_goal failed: {e}")
            raise StorageWriteError("save_idp_goal", e) from e

    async def update_idp_goal(
        self,
        goal_id: int,
        current_value: float | None = None,
        status: str | None = None,
        review_note: dict | None = None,
    ) -> None:
        if status is not None and status not in ("active", "achieved", "revised", "dropped"):
            raise ValueError(f"invalid idp status: {status}")
        if self._conn is None:
            raise StorageNotInitializedError("update_idp_goal")
        try:
            cur = self._conn.cursor()
            if current_value is not None:
                cur.execute(
                    "UPDATE idp_goals SET current_value = ?, updated_at = datetime('now') WHERE id = ?",
                    (current_value, goal_id),
                )
            if status is not None:
                cur.execute(
                    "UPDATE idp_goals SET status = ?, updated_at = datetime('now') WHERE id = ?",
                    (status, goal_id),
                )
            if review_note is not None:
                cur.execute("SELECT review_notes FROM idp_goals WHERE id = ?", (goal_id,))
                row = cur.fetchone()
                reviews = _load_json_list(row[0] if row else None)
                reviews.append(review_note)
                cur.execute(
                    "UPDATE idp_goals SET review_notes = ?, updated_at = datetime('now') WHERE id = ?",
                    (json.dumps(reviews), goal_id),
                )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"update_idp_goal failed: {e}")
            raise StorageWriteError("update_idp_goal", e) from e

    async def get_player_idp(
        self, player_id: int, season: str | None = None
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_player_idp")
        cur = self._conn.cursor()
        if season:
            cur.execute(
                """SELECT id, player_id, season, category, goal_text, target_metric,
                          baseline_value, current_value, target_date, status, review_notes,
                          created_at, updated_at
                   FROM idp_goals WHERE player_id = ? AND season = ? ORDER BY id""",
                (player_id, season),
            )
        else:
            cur.execute(
                """SELECT id, player_id, season, category, goal_text, target_metric,
                          baseline_value, current_value, target_date, status, review_notes,
                          created_at, updated_at
                   FROM idp_goals WHERE player_id = ? ORDER BY id""",
                (player_id,),
            )
        rows = [_row_to_dict(r) for r in cur.fetchall()]
        for r in rows:
            r["review_notes"] = _load_json_list(r.get("review_notes"))
        return rows

    # ── Nutrition + psych check-ins ─────────────────────────────────────

    async def save_nutrition_log(
        self,
        player_id: int,
        log_date: str,
        meal_type: str,
        hydration_score: int = 3,
        fueling_score: int = 3,
        notes: str = "",
    ) -> int:
        if self._conn is None:
            raise StorageNotInitializedError("save_nutrition_log")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO nutrition_logs
                   (player_id, log_date, meal_type, hydration_score, fueling_score, notes)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(player_id, log_date, meal_type)
                   DO UPDATE SET hydration_score=excluded.hydration_score,
                       fueling_score=excluded.fueling_score, notes=excluded.notes""",
                (player_id, log_date, meal_type, hydration_score, fueling_score, notes),
            )
            self._conn.commit()
            cur.execute(
                "SELECT id FROM nutrition_logs WHERE player_id = ? AND log_date = ? AND meal_type = ?",
                (player_id, log_date, meal_type),
            )
            row = cur.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.warning(f"save_nutrition_log failed: {e}")
            raise StorageWriteError("save_nutrition_log", e) from e

    async def get_nutrition_logs(self, player_id: int, limit: int = 30) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_nutrition_logs")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, player_id, log_date, meal_type, hydration_score, fueling_score, notes, created_at
               FROM nutrition_logs WHERE player_id = ? ORDER BY log_date DESC LIMIT ?""",
            (player_id, limit),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    async def save_psych_checkin(
        self,
        player_id: int,
        checkin_date: str,
        confidence: int = 3,
        focus: int = 3,
        motivation: int = 3,
        anxiety: int = 3,
        notes: str = "",
        flag_for_followup: bool = False,
    ) -> int:
        if self._conn is None:
            raise StorageNotInitializedError("save_psych_checkin")
        for name, v in (
            ("confidence", confidence),
            ("focus", focus),
            ("motivation", motivation),
            ("anxiety", anxiety),
        ):
            if not 1 <= int(v) <= 5:
                raise ValueError(f"{name} must be within 1-5")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO psych_checkins
                   (player_id, checkin_date, confidence, focus, motivation, anxiety, notes, flag_for_followup)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(player_id, checkin_date)
                   DO UPDATE SET confidence=excluded.confidence, focus=excluded.focus,
                       motivation=excluded.motivation, anxiety=excluded.anxiety,
                       notes=excluded.notes, flag_for_followup=excluded.flag_for_followup""",
                (
                    player_id,
                    checkin_date,
                    confidence,
                    focus,
                    motivation,
                    anxiety,
                    notes,
                    int(flag_for_followup),
                ),
            )
            self._conn.commit()
            cur.execute(
                "SELECT id FROM psych_checkins WHERE player_id = ? AND checkin_date = ?",
                (player_id, checkin_date),
            )
            row = cur.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.warning(f"save_psych_checkin failed: {e}")
            raise StorageWriteError("save_psych_checkin", e) from e

    async def get_psych_checkins(self, player_id: int, limit: int = 30) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_psych_checkins")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, player_id, checkin_date, confidence, focus, motivation, anxiety,
                      notes, flag_for_followup, created_at
               FROM psych_checkins WHERE player_id = ? ORDER BY checkin_date DESC LIMIT ?""",
            (player_id, limit),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    async def get_flagged_psych_checkins(self, limit: int = 50) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_flagged_psych_checkins")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, player_id, checkin_date, confidence, focus, motivation, anxiety, notes
               FROM psych_checkins WHERE flag_for_followup = 1 ORDER BY checkin_date DESC LIMIT ?""",
            (limit,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    # ── Staff rituals ───────────────────────────────────────────────────

    async def save_ritual(
        self,
        ritual_type: str,
        scheduled_for: str,
        checklist_state: list[dict] | None = None,
        notes: str = "",
    ) -> int:
        if self._conn is None:
            raise StorageNotInitializedError("save_ritual")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO staff_rituals (ritual_type, scheduled_for, checklist_state, notes)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(ritual_type, scheduled_for)
                   DO UPDATE SET checklist_state=excluded.checklist_state, notes=excluded.notes""",
                (ritual_type, scheduled_for, json.dumps(checklist_state or []), notes),
            )
            self._conn.commit()
            cur.execute(
                "SELECT id FROM staff_rituals WHERE ritual_type = ? AND scheduled_for = ?",
                (ritual_type, scheduled_for),
            )
            row = cur.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.warning(f"save_ritual failed: {e}")
            raise StorageWriteError("save_ritual", e) from e

    async def complete_ritual(
        self,
        ritual_type: str,
        scheduled_for: str,
        completed_by: str,
        checklist_state: list[dict] | None = None,
        notes: str = "",
    ) -> None:
        if self._conn is None:
            raise StorageNotInitializedError("complete_ritual")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO staff_rituals
                   (ritual_type, scheduled_for, completed_at, completed_by, checklist_state, notes)
                   VALUES (?, ?, datetime('now'), ?, ?, ?)
                   ON CONFLICT(ritual_type, scheduled_for)
                   DO UPDATE SET completed_at=datetime('now'), completed_by=excluded.completed_by,
                       checklist_state=excluded.checklist_state, notes=excluded.notes""",
                (
                    ritual_type,
                    scheduled_for,
                    completed_by,
                    json.dumps(checklist_state or []),
                    notes,
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"complete_ritual failed: {e}")
            raise StorageWriteError("complete_ritual", e) from e

    async def get_rituals(
        self, scheduled_for: str | None = None, ritual_type: str | None = None
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_rituals")
        cur = self._conn.cursor()
        clauses: list[str] = []
        params: list[Any] = []
        if scheduled_for:
            clauses.append("scheduled_for = ?")
            params.append(scheduled_for)
        if ritual_type:
            clauses.append("ritual_type = ?")
            params.append(ritual_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur.execute(
            f"""SELECT id, ritual_type, scheduled_for, completed_at, completed_by, checklist_state, notes
                FROM staff_rituals {where} ORDER BY scheduled_for DESC""",
            params,
        )
        rows = [_row_to_dict(r) for r in cur.fetchall()]
        for r in rows:
            r["checklist_state"] = _load_json_list(r.get("checklist_state"))
        return rows

    # ── Medical clearance (gates selection) ─────────────────────────────

    async def set_medical_clearance(
        self,
        player_id: int,
        status: str,
        cleared_by: str = "",
        reason: str = "",
        source: str = "manual",
        effective_until: str = "",
    ) -> int:
        if status not in ("fit", "limited", "unavailable"):
            raise ValueError(f"invalid clearance status: {status}")
        if self._conn is None:
            raise StorageNotInitializedError("set_medical_clearance")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO medical_clearances
                   (player_id, status, reason, source, cleared_by, effective_until)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (player_id, status, reason, source, cleared_by, effective_until),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except Exception as e:
            logger.warning(f"set_medical_clearance failed: {e}")
            raise StorageWriteError("set_medical_clearance", e) from e

    async def get_latest_clearance(self, player_id: int) -> dict[str, Any] | None:
        if self._conn is None:
            raise StorageNotInitializedError("get_latest_clearance")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, player_id, status, reason, source, cleared_by, effective_until, created_at
               FROM medical_clearances WHERE player_id = ?
               ORDER BY id DESC LIMIT 1""",
            (player_id,),
        )
        row = cur.fetchone()
        return _row_to_dict(row) if row else None

    async def get_clearances_by_status(self, status: str) -> list[dict[str, Any]]:
        """Latest clearance per player filtered by status — the squad availability view."""
        if self._conn is None:
            raise StorageNotInitializedError("get_clearances_by_status")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT c.id, c.player_id, c.status, c.reason, c.source, c.cleared_by, c.created_at
               FROM medical_clearances c
               WHERE c.id IN (SELECT MAX(id) FROM medical_clearances GROUP BY player_id)
               AND c.status = ?
               ORDER BY c.player_id""",
            (status,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    # ── Minutes log ─────────────────────────────────────────────────────

    async def save_minutes_entry(
        self,
        player_id: int,
        match_id: int,
        minutes_played: int,
        started: bool,
        age_phase: str = "",
        bio_band: str = "",
    ) -> int:
        if self._conn is None:
            raise StorageNotInitializedError("save_minutes_entry")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO minutes_log
                   (player_id, match_id, minutes_played, started, age_phase, bio_band)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (player_id, match_id, minutes_played, int(started), age_phase, bio_band),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except Exception as e:
            logger.warning(f"save_minutes_entry failed: {e}")
            raise StorageWriteError("save_minutes_entry", e) from e

    async def get_player_minutes_history(
        self, player_id: int, limit: int = 30
    ) -> list[dict[str, Any]]:
        if self._conn is None:
            raise StorageNotInitializedError("get_player_minutes_history")
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, player_id, match_id, minutes_played, started, age_phase, bio_band
               FROM minutes_log WHERE player_id = ? ORDER BY id DESC LIMIT ?""",
            (player_id, limit),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    async def get_squad_minutes_summary(self, age_phase: str | None = None) -> list[dict[str, Any]]:
        """Per-player minutes totals — the game-time equity view."""
        if self._conn is None:
            raise StorageNotInitializedError("get_squad_minutes_summary")
        cur = self._conn.cursor()
        if age_phase:
            cur.execute(
                """SELECT player_id, COUNT(*) as matches, SUM(minutes_played) as total_minutes,
                          SUM(started) as starts
                   FROM minutes_log WHERE age_phase = ?
                   GROUP BY player_id ORDER BY total_minutes DESC""",
                (age_phase,),
            )
        else:
            cur.execute(
                """SELECT player_id, COUNT(*) as matches, SUM(minutes_played) as total_minutes,
                          SUM(started) as starts
                   FROM minutes_log GROUP BY player_id ORDER BY total_minutes DESC"""
            )
        return [_row_to_dict(r) for r in cur.fetchall()]

    # ── Phase D: persistent player profiles (academy phases need DOB) ───

    async def get_player_profile(self, player_id: int) -> dict[str, Any] | None:
        """A persistent player profile by id (date_of_birth lives here).

        Read is soft: an unknown id returns None; a lookup failure raises.
        """
        if self._conn is None:
            raise StorageNotInitializedError("get_player_profile")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """SELECT id, global_id, display_name, jersey_number, preferred_position,
                          height_cm, weight_kg, dominant_foot, date_of_birth, nationality,
                          team, is_active
                   FROM player_profiles WHERE id = ?""",
                (int(player_id),),
            )
            row = cur.fetchone()
            return _row_to_dict(row) if row else None
        except Exception as e:
            logger.warning(f"get_player_profile failed: {e}")
            raise StorageReadError("get_player_profile", e) from e

    # ── Phase D: versioned program documents (operating program) ────────

    async def save_program_document(
        self,
        doc_type: str,
        name: str,
        payload: dict[str, Any],
        created_by: str = "",
    ) -> int:
        """Publish a new version of a program document.

        Exactly one is_current row per (doc_type, name): the previous
        current version is superseded, not deleted — the audit trail is
        the point. Write fails loudly on any error.
        """
        if self._conn is None:
            raise StorageNotInitializedError("save_program_document")
        if not str(doc_type).strip():
            raise StorageWriteError("save_program_document", "doc_type must not be empty")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """SELECT id, version FROM program_documents
                   WHERE doc_type = ? AND name = ? AND is_current = 1""",
                (doc_type, name),
            )
            prev = cur.fetchone()
            prev_id = prev["id"] if prev else None
            next_version = (prev["version"] + 1) if prev else 1
            if prev_id is not None:
                cur.execute("UPDATE program_documents SET is_current = 0 WHERE id = ?", (prev_id,))
            cur.execute(
                """INSERT INTO program_documents
                   (doc_type, name, version, is_current, payload, created_by, supersedes_id)
                   VALUES (?, ?, ?, 1, ?, ?, ?)""",
                (
                    doc_type,
                    name,
                    next_version,
                    json.dumps(payload, ensure_ascii=False),
                    created_by,
                    prev_id,
                ),
            )
            self._conn.commit()
            new_id = cur.lastrowid
            return int(new_id) if new_id else 0
        except Exception as e:
            self._conn.commit()
            logger.warning(f"save_program_document failed: {e}")
            raise StorageWriteError("save_program_document", e) from e

    async def get_current_program_document(
        self, doc_type: str, name: str = ""
    ) -> dict[str, Any] | None:
        """The current version of a program document, or None (soft read)."""
        if self._conn is None:
            raise StorageNotInitializedError("get_current_program_document")
        try:
            cur = self._conn.cursor()
            if name:
                cur.execute(
                    """SELECT id, doc_type, name, version, payload, created_by,
                              supersedes_id, created_at
                       FROM program_documents
                       WHERE doc_type = ? AND name = ? AND is_current = 1""",
                    (doc_type, name),
                )
            else:
                cur.execute(
                    """SELECT id, doc_type, name, version, payload, created_by,
                              supersedes_id, created_at
                       FROM program_documents
                       WHERE doc_type = ? AND is_current = 1""",
                    (doc_type,),
                )
            row = cur.fetchone()
            if not row:
                return None
            out = _row_to_dict(row)
            out["payload"] = _load_json_dict(out.get("payload"))
            return out
        except Exception as e:
            logger.warning(f"get_current_program_document failed: {e}")
            raise StorageReadError("get_current_program_document", e) from e

    async def list_program_documents(self, doc_type: str) -> list[dict[str, Any]]:
        """All versions of a document kind, newest first (audit trail)."""
        if self._conn is None:
            raise StorageNotInitializedError("list_program_documents")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """SELECT id, doc_type, name, version, is_current, payload, created_by,
                          supersedes_id, created_at
                   FROM program_documents WHERE doc_type = ?
                   ORDER BY created_at DESC, version DESC""",
                (doc_type,),
            )
            rows = [_row_to_dict(r) for r in cur.fetchall()]
            for r in rows:
                r["payload"] = _load_json_dict(r.get("payload"))
            return rows
        except Exception as e:
            logger.warning(f"list_program_documents failed: {e}")
            raise StorageReadError("list_program_documents", e) from e

    # ── Phase D: evidence registry (trust layer groundedness gate) ─────

    async def save_evidence_record(
        self,
        kind: str,
        payload: dict[str, Any],
        match_id: int | None = None,
        label: str = "",
    ) -> int:
        """Persist one evidence record; returns its registry id.

        Fail-loud write: the groundedness gate is only as honest as
        this registry is durable.
        """
        if self._conn is None:
            raise StorageNotInitializedError("save_evidence_record")
        if not str(kind).strip():
            raise StorageWriteError("save_evidence_record", "kind must not be empty")
        try:
            cur = self._conn.cursor()
            cur.execute(
                """INSERT INTO evidence_records (match_id, kind, label, payload)
                   VALUES (?, ?, ?, ?)""",
                (match_id, kind, label, json.dumps(payload, ensure_ascii=False)),
            )
            self._conn.commit()
            new_id = cur.lastrowid
            return int(new_id) if new_id else 0
        except Exception as e:
            self._conn.commit()
            logger.warning(f"save_evidence_record failed: {e}")
            raise StorageWriteError("save_evidence_record", e) from e

    async def get_evidence_records(
        self, record_ids: list[int] | None = None, match_id: int | None = None
    ) -> list[dict[str, Any]]:
        """Fetch evidence records by ids and/or match (soft read)."""
        if self._conn is None:
            raise StorageNotInitializedError("get_evidence_records")
        try:
            cur = self._conn.cursor()
            clauses: list[str] = []
            params: list[Any] = []
            if record_ids:
                placeholders = ",".join("?" for _ in record_ids)
                clauses.append(f"id IN ({placeholders})")
                params.extend(int(i) for i in record_ids)
            if match_id is not None:
                clauses.append("match_id = ?")
                params.append(int(match_id))
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            cur.execute(
                f"""SELECT id, match_id, kind, label, payload, created_at
                    FROM evidence_records {where} ORDER BY id""",
                params,
            )
            rows = [_row_to_dict(r) for r in cur.fetchall()]
            for r in rows:
                r["payload"] = _load_json_dict(r.get("payload"))
            return rows
        except Exception as e:
            logger.warning(f"get_evidence_records failed: {e}")
            raise StorageReadError("get_evidence_records", e) from e
