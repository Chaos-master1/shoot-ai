from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from kawkab.core.encryption import decrypt_dict, encrypt_dict


class RehabPhase(StrEnum):
    INITIAL = "initial"
    PROTECTIVE = "protective"
    RESTORATIVE = "restorative"
    FUNCTIONAL = "functional"
    RETURN_TO_PLAY = "return_to_play"
    MAINTENANCE = "maintenance"


REHAB_PHASE_DURATIONS: dict[str, int] = {
    "initial": 3,  # days
    "protective": 7,
    "restorative": 14,
    "functional": 14,
    "return_to_play": 7,
    "maintenance": 30,
}

REHAB_MILESTONES: dict[str, list[str]] = {
    "initial": [
        "Pain management initiated",
        "Range of motion assessed",
        "Ice/compression protocol started",
    ],
    "protective": ["Swelling reduced", "Protected weight-bearing achieved", "Pain at rest < 3/10"],
    "restorative": [
        "Full range of motion restored",
        "Strength > 70% of unaffected side",
        "Proprioception exercises started",
    ],
    "functional": [
        "Sport-specific drills initiated",
        "Strength > 90% of unaffected side",
        "Agility exercises passed",
    ],
    "return_to_play": [
        "Full training without limitation",
        "Medical clearance obtained",
        "RTP protocol completed",
    ],
    "maintenance": ["Maintenance program prescribed", "Follow-up scheduled"],
}


@dataclass
class RehabPlan:
    injury_id: int
    phase: str = "initial"
    start_date: str = ""
    milestones: list[str] = field(default_factory=list)
    status: str = "active"
    notes: str = ""
    id: int = 0

    def __post_init__(self):
        if self.milestones is None:
            self.milestones = []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "injury_id": self.injury_id,
            "phase": self.phase,
            "start_date": self.start_date,
            "status": self.status,
            "milestones": self.milestones,
            "notes": self.notes,
        }


class RehabService:
    def __init__(self, db: Any) -> None:
        self._db = db

    def create_plan(self, injury_id: int, start_date: str = "") -> RehabPlan:
        if not start_date:
            start_date = datetime.now().isoformat()
        milestones_text = json.dumps(REHAB_MILESTONES["initial"])
        encrypted_milestones = encrypt_dict(
            {"milestones": milestones_text}, ["milestones"], in_place=False
        )["milestones"]
        cur = self._db.execute(
            "INSERT INTO rehab_plans (injury_id, phase, start_date, milestones, status) VALUES (?, ?, ?, ?, ?)",
            (injury_id, "initial", start_date, encrypted_milestones, "active"),
        )
        self._db.commit()
        return RehabPlan(
            id=cur.lastrowid,
            injury_id=injury_id,
            phase="initial",
            start_date=start_date,
            milestones=list(REHAB_MILESTONES["initial"]),
        )

    def _decrypt_rehab(self, row: dict) -> dict:
        d = decrypt_dict(dict(row), ["milestones", "notes"])
        d["milestones"] = json.loads(d.get("milestones", "[]"))
        return d

    def get_plan(self, plan_id: int) -> dict | None:
        row = self._db.execute("SELECT * FROM rehab_plans WHERE id = ?", (plan_id,)).fetchone()
        if row is None:
            return None
        return self._decrypt_rehab(row)

    def get_plan_by_injury(self, injury_id: int) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM rehab_plans WHERE injury_id = ? ORDER BY created_at DESC LIMIT 1",
            (injury_id,),
        ).fetchone()
        if row is None:
            return None
        return self._decrypt_rehab(row)

    def advance_phase(self, plan_id: int) -> dict | None:
        plan = self.get_plan(plan_id)
        if plan is None:
            return None
        phases = list(RehabPhase.__members__.values())
        current_idx = next((i for i, p in enumerate(phases) if p.value == plan["phase"]), -1)
        if current_idx < 0 or current_idx >= len(phases) - 1:
            return plan
        next_phase = phases[current_idx + 1].value
        milestones = REHAB_MILESTONES.get(next_phase, [])
        encrypted = encrypt_dict(
            {"milestones": json.dumps(milestones)}, ["milestones"], in_place=False
        )["milestones"]
        self._db.execute(
            "UPDATE rehab_plans SET phase = ?, milestones = ?, updated_at = datetime('now') WHERE id = ?",
            (next_phase, encrypted, plan_id),
        )
        self._db.commit()
        plan["phase"] = next_phase
        plan["milestones"] = list(milestones)
        if next_phase == "return_to_play":
            self._db.execute(
                "UPDATE injuries SET status = 'recovered', date_recovered = datetime('now') WHERE id = ?",
                (plan["injury_id"],),
            )
            self._db.commit()
        return plan

    def complete_milestone(self, plan_id: int, milestone: str) -> dict | None:
        plan = self.get_plan(plan_id)
        if plan is None:
            return None
        ms = list(plan.get("milestones", []))
        if milestone in ms:
            ms.remove(milestone)
        encrypted = encrypt_dict({"milestones": json.dumps(ms)}, ["milestones"], in_place=False)[
            "milestones"
        ]
        self._db.execute(
            "UPDATE rehab_plans SET milestones = ?, updated_at = datetime('now') WHERE id = ?",
            (encrypted, plan_id),
        )
        self._db.commit()
        plan["milestones"] = ms
        return plan

    def close_plan(self, plan_id: int) -> None:
        self._db.execute(
            "UPDATE rehab_plans SET status = 'completed', actual_end_date = datetime('now'), updated_at = datetime('now') WHERE id = ?",
            (plan_id,),
        )
        self._db.commit()

    def get_active_plans(self, player_id: int | None = None) -> list[dict]:
        if player_id is not None:
            rows = self._db.execute(
                """SELECT rp.* FROM rehab_plans rp
                   JOIN injuries i ON rp.injury_id = i.id
                   WHERE i.player_id = ? AND rp.status = 'active'
                   ORDER BY rp.created_at DESC""",
                (player_id,),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM rehab_plans WHERE status = 'active' ORDER BY created_at DESC",
            ).fetchall()
        return [self._decrypt_rehab(r) for r in rows]
