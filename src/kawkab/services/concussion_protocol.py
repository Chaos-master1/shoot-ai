"""Concussion management — SCAT6-based graduated return-to-play protocol.

Implements the Amsterdam 2022 consensus / SCAT6 (2023) sport concussion
assessment tool: the six-stage graduated return-to-play (RTP) strategy
with 24-hour minimums per stage, immediate removal from play on suspicion
of concussion, and the SCAT6 red-flag list.

Upgraded from SCAT5: SCAT6 replaced SCAT5 as the consensus instrument in
2023. The stored ``assessment_type`` column records which instrument was
used; new assessments default to ``scat6``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from kawkab.core.encryption import decrypt_dict, encrypt_dict


class ConcussionClearance(StrEnum):
    """SCAT6 graduated return-to-play stages (Amsterdam 2022 consensus).

    Stage 1: symptom-limited activity (daily-life activity, light walking).
    Stage 2: aerobic exercise (2A light / 2B moderate, stationary cycling).
    Stage 3: individual sport-specific exercise (running drills, NO
             head-impact-risk activity).
    Stage 4: non-contact training drills (progressive resistance allowed).
    Stage 5: full-contact practice (after medical clearance).
    full_cleared: return to sport / competition (SCAT6 step 6).
    """

    NOT_CLEARED = "not_cleared"
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    STAGE_3 = "stage_3"
    STAGE_4 = "stage_4"
    STAGE_5 = "stage_5"
    FULL_CLEARED = "full_cleared"  # SCAT6 step 6: return to sport — normal game play


STAGE_DESCRIPTIONS: dict[str, str] = {
    "stage_1": "Symptom-limited activity — daily activities that do not exacerbate symptoms (light walking)",
    "stage_2": "Aerobic exercise — 2A light (stationary cycling, no resistance) then 2B moderate (jogging)",
    "stage_3": "Individual sport-specific exercise — running drills away from team, NO head-impact-risk activities",
    "stage_4": "Non-contact training drills — progressive resistance training and harder team drills, no head-impact risk",
    "stage_5": "Full-contact practice — normal training activities, after medical clearance",
    "full_cleared": "Return to sport — normal game play (SCAT6 step 6)",
}

# SCAT6 immediate red flags: any of these = emergency removal and urgent
# hospital assessment (neck pain, confusion, seizure, severe headache,
# double vision, weakness/tingling, vomiting, deteriorating consciousness,
# agitation, unusual behavior).
SCAT6_RED_FLAGS = [
    "neck_pain_or_tenderness",
    "double_vision",
    "weakness_or_tingling_burning",
    "severe_or_increasing_headache",
    "seizure_or_convulsion",
    "loss_of_consciousness",
    "deteriorating_conscious_state",
    "vomiting_repeatedly",
    "increasingly_restless_agitated_combative",
    "increasing_confusion_irritable",
]


def check_red_flags(flagged: list[str]) -> dict[str, Any]:
    """Evaluate SCAT6 immediate-removal red flags.

    Returns an emergency verdict when any red flag is present — the player
    must NOT return to play and requires urgent hospital assessment.
    """
    present = [f for f in flagged if f in SCAT6_RED_FLAGS]
    if present:
        return {
            "emergency": True,
            "red_flags_present": present,
            "action": (
                "Remove from play immediately. Do not return. Urgent hospital "
                "assessment required (SCAT6 red flags present)."
            ),
        }
    return {"emergency": False, "red_flags_present": [], "action": ""}


@dataclass
class SCAT6Assessment:
    """A concussion assessment recorded with the SCAT6 instrument.

    Kept the SCAT5Assessment field shape for backwards compatibility with
    existing storage rows and callers; new code should use this class and
    the instrument defaults to ``scat6``.
    """

    player_id: int
    symptoms_score: int = 0
    cognitive_score: int = 0
    balance_score: int = 0
    match_id: int | None = None
    clearance_status: str = "not_cleared"
    assessment_type: str = "scat6"
    red_flags: list[str] | None = None
    notes: str = ""
    id: int = 0

    def __post_init__(self) -> None:
        if self.red_flags is None:
            self.red_flags = []

    def total_score(self) -> int:
        return self.symptoms_score + self.cognitive_score + self.balance_score

    def is_symptomatic(self) -> bool:
        return self.symptoms_score > 0

    def has_red_flags(self) -> bool:
        return bool(check_red_flags(self.red_flags or [])["emergency"])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "player_id": self.player_id,
            "match_id": self.match_id,
            "assessment_type": self.assessment_type,
            "symptoms_score": self.symptoms_score,
            "cognitive_score": self.cognitive_score,
            "balance_score": self.balance_score,
            "total_score": self.total_score(),
            "is_symptomatic": self.is_symptomatic(),
            "red_flags": self.red_flags or [],
            "clearance_status": self.clearance_status,
            "notes": self.notes,
        }


# Backwards-compatible alias: SCAT5 data stays valid; the RTP stages are
# the same graduated strategy. New assessments record scat6.
SCAT5Assessment = SCAT6Assessment


class ConcussionProtocolService:
    """Implements the SCAT6 graduated return-to-play protocol for concussion management.

    Safety invariants:
    - minimum 24h rest before starting RTP stages,
    - minimum 24h per stage, no same-day advancement,
    - a player still symptomatic never advances,
    - red-flag assessments block advancement entirely,
    - full clearance requires medical sign-off (``cleared_by`` recorded).
    """

    MIN_REST_DAYS = 24  # hours minimum rest before starting RTP stages
    STAGE_MIN_DAYS = 1  # minimum days per stage (no same-day advancement)

    def __init__(self, db: Any) -> None:
        self._db = db
        # Column detection: pre-032 databases (and lightweight in-memory
        # fixtures) predate the assessment_type/red_flags columns. The
        # service adapts instead of crashing — legacy behavior for legacy
        # schema, full SCAT6 persistence where the columns exist.
        try:
            cols = {
                (r["name"] if isinstance(r, sqlite3.Row) else r[1])
                for r in self._db.execute("PRAGMA table_info(concussion_assessments)").fetchall()
            }
        except Exception:
            cols = set()
        self._has_type_col = "assessment_type" in cols
        self._has_flags_col = "red_flags" in cols

    def record_assessment(self, assessment: SCAT6Assessment) -> int:
        encrypted_notes = encrypt_dict({"notes": assessment.notes}, ["notes"], in_place=False)[
            "notes"
        ]
        import json as _json

        cols = [
            "player_id",
            "match_id",
            "symptoms_score",
            "cognitive_score",
            "balance_score",
            "clearance_status",
            "notes",
        ]
        vals: list[Any] = [
            assessment.player_id,
            assessment.match_id,
            assessment.symptoms_score,
            assessment.cognitive_score,
            assessment.balance_score,
            assessment.clearance_status,
            encrypted_notes,
        ]
        if self._has_type_col:
            cols.append("assessment_type")
            vals.append(getattr(assessment, "assessment_type", "scat6"))
        if self._has_flags_col:
            cols.append("red_flags")
            vals.append(_json.dumps(getattr(assessment, "red_flags", []) or []))
        placeholders = ", ".join("?" for _ in cols)
        cur = self._db.execute(
            f"INSERT INTO concussion_assessments ({', '.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        self._db.commit()
        return cur.lastrowid

    def get_assessments(self, player_id: int) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM concussion_assessments WHERE player_id = ? ORDER BY assessment_date DESC",
            (player_id,),
        ).fetchall()
        return [decrypt_dict(dict(r), ["notes"]) for r in rows]

    def advance_clearance(self, assessment_id: int, cleared_by: str = "") -> dict | None:
        row = self._db.execute(
            "SELECT * FROM concussion_assessments WHERE id = ?",
            (assessment_id,),
        ).fetchone()
        if row is None:
            return None
        current = decrypt_dict(dict(row), ["notes"])
        # SCAT6 red-flag gate: an assessment with red flags present never
        # advances — urgent hospital assessment is the required action.
        raw_flags = current.get("red_flags") or "[]"
        try:
            flagged = json.loads(raw_flags) if isinstance(raw_flags, str) else (raw_flags or [])
        except json.JSONDecodeError:
            flagged = []
        red = check_red_flags(list(flagged) if isinstance(flagged, list) else [])
        if red["emergency"]:
            current["red_flag_verdict"] = red
            current["advance_blocked"] = True
            current["advance_block_reason"] = (
                "SCAT6 red flags present — urgent hospital assessment required"
            )
            return current
        # Full clearance requires medical sign-off: an advancement into
        # full_cleared without a cleared_by identity is refused.
        stages = [s.value for s in ConcussionClearance]
        try:
            idx = stages.index(current["clearance_status"])
        except ValueError:
            return current
        if idx >= len(stages) - 1:
            return current
        next_status = stages[idx + 1]
        if next_status == ConcussionClearance.FULL_CLEARED and not (
            cleared_by or current.get("cleared_by")
        ):
            current["advance_blocked"] = True
            current["advance_block_reason"] = (
                "Full clearance requires medical sign-off (cleared_by)"
            )
            return current
        appended = current.get("notes", "") + f" | Advanced to {next_status}"
        encrypted = encrypt_dict({"notes": appended}, ["notes"], in_place=False)["notes"]
        self._db.execute(
            """UPDATE concussion_assessments
               SET clearance_status = ?, cleared_by = COALESCE(?, cleared_by), notes = ?
               WHERE id = ?""",
            (next_status, cleared_by, encrypted, assessment_id),
        )
        self._db.commit()
        current["clearance_status"] = next_status
        current["notes"] = appended
        return current

    def get_clearance_status(self, player_id: int) -> dict:
        row = self._db.execute(
            "SELECT * FROM concussion_assessments WHERE player_id = ? ORDER BY assessment_date DESC LIMIT 1",
            (player_id,),
        ).fetchone()
        if row is None:
            return {
                "player_id": player_id,
                "clearance_status": "not_cleared",
                "has_assessment": False,
            }
        r = decrypt_dict(dict(row), ["notes"])
        r["has_assessment"] = True
        r["stage_description"] = STAGE_DESCRIPTIONS.get(r["clearance_status"], "")
        return r

    def get_stage_protocol(self, stage: str) -> dict:
        return {
            "stage": stage,
            "description": STAGE_DESCRIPTIONS.get(stage, "Unknown stage"),
            "min_duration_days": self.STAGE_MIN_DAYS,
            "min_rest_hours": self.MIN_REST_DAYS,
        }

    def check_return_to_play_readiness(self, player_id: int) -> dict:
        status = self.get_clearance_status(player_id)
        if not status.get("has_assessment", False):
            return {
                "ready": False,
                "reason": "No concussion assessment recorded",
                "status": "not_assessed",
            }
        if status["clearance_status"] == "full_cleared":
            return {"ready": True, "status": "full_cleared", "reason": "Full clearance granted"}
        if status["is_symptomatic"]:
            return {
                "ready": False,
                "status": status["clearance_status"],
                "reason": "Player still symptomatic",
            }
        stage = status["clearance_status"]
        description = STAGE_DESCRIPTIONS.get(stage, "")
        return {
            "ready": stage == "full_cleared",
            "status": stage,
            "current_stage_description": description,
            "reason": "Progressing through RTP protocol"
            if stage != "full_cleared"
            else "Ready for full participation",
        }
