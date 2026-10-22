from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from kawkab.core.encryption import decrypt_dict, encrypt_dict


class InjurySeverity(StrEnum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class InjuryStatus(StrEnum):
    ACTIVE = "active"
    RECOVERED = "recovered"
    CHRONIC = "chronic"


class BodyPart(StrEnum):
    HEAD = "head"
    NECK = "neck"
    SHOULDER = "shoulder"
    UPPER_ARM = "upper_arm"
    ELBOW = "elbow"
    FOREARM = "forearm"
    WRIST = "wrist"
    HAND = "hand"
    CHEST = "chest"
    RIBS = "ribs"
    UPPER_BACK = "upper_back"
    LOWER_BACK = "lower_back"
    ABDOMEN = "abdomen"
    HIP = "hip"
    GROIN = "groin"
    THIGH = "thigh"
    KNEE = "knee"
    SHIN = "shin"
    CALF = "calf"
    ANKLE = "ankle"
    FOOT = "foot"


# Evidence-based recovery estimates (min_days, expected_days, max_days)
# Sources: Ekstrand et al. (2011) UEFA Elite Club Injury Study,
#   NCAA Injury Surveillance Program, BMJ Open Sport & Exercise Medicine
RECOVERY_ESTIMATES: dict[str, tuple[int, int, int]] = {
    "minor": (3, 7, 14),
    "moderate": (14, 21, 42),
    "severe": (30, 60, 120),
    "critical": (60, 120, 365),
    "sprain_grade_1": (3, 10, 21),
    "sprain_grade_2": (14, 21, 42),
    "sprain_grade_3": (42, 84, 180),
    "strain_grade_1": (3, 10, 21),
    "strain_grade_2": (10, 21, 42),
    "strain_grade_3": (30, 56, 120),
    "fracture": (28, 42, 84),
    "stress_fracture": (28, 56, 120),
    "concussion": (7, 14, 28),
    "acl": (150, 180, 365),
    "hamstring_strain": (10, 28, 56),
    "hamstring_tear": (28, 56, 120),
    "groin_strain": (10, 21, 42),
    "ankle_sprain": (3, 14, 42),
    "calf_strain": (7, 21, 42),
    "meniscus": (42, 84, 180),
    "mcl_sprain": (14, 42, 90),
    "lcl_sprain": (14, 42, 90),
    "quad_strain": (7, 21, 42),
    "shoulder_dislocation": (21, 42, 84),
    "rib_fracture": (21, 42, 70),
    "illness": (3, 7, 14),
}


@dataclass
class InjuryRecord:
    player_id: int
    injury_type: str
    body_part: str
    severity: str = "minor"
    mechanism: str = ""
    date_injured: str = ""
    match_id: int | None = None
    status: str = "active"
    notes: str = ""
    id: int = 0
    date_recovered: str = ""
    created_at: str = ""

    def days_since_injury(self) -> int:
        if not self.date_injured:
            return 0
        try:
            d = datetime.strptime(self.date_injured[:10], "%Y-%m-%d").date()
            return (date.today() - d).days
        except (ValueError, TypeError):
            return 0

    def estimated_recovery_days(self) -> int:
        key = self.injury_type.lower()
        severity_key = self.severity.lower()
        if key in RECOVERY_ESTIMATES:
            _, expected, _ = RECOVERY_ESTIMATES[key]
            return expected
        if severity_key in RECOVERY_ESTIMATES:
            _, expected, _ = RECOVERY_ESTIMATES[severity_key]
            return expected
        return 14

    def recovery_range_days(self) -> tuple[int, int]:
        key = self.injury_type.lower()
        severity_key = self.severity.lower()
        if key in RECOVERY_ESTIMATES:
            mn, _, mx = RECOVERY_ESTIMATES[key]
            return (mn, mx)
        if severity_key in RECOVERY_ESTIMATES:
            mn, _, mx = RECOVERY_ESTIMATES[severity_key]
            return (mn, mx)
        return (3, 30)

    def days_until_expected_recovery(self) -> int:
        if not self.date_injured:
            return 0
        expected = self.estimated_recovery_days()
        return max(0, expected - self.days_since_injury())

    def injury_risk_score(self) -> float:
        days_since = self.days_since_injury()
        expected = self.estimated_recovery_days()
        if expected == 0:
            return 1.0
        completion = days_since / max(expected, 1)
        if completion < 1.0:
            return 0.0
        excess = completion - 1.0
        return min(1.0, 1.0 - math.exp(-excess * 3.0))

    def to_dict(self) -> dict:
        rec_min, rec_max = self.recovery_range_days()
        return {
            "id": self.id,
            "player_id": self.player_id,
            "injury_type": self.injury_type,
            "body_part": self.body_part,
            "severity": self.severity,
            "mechanism": self.mechanism,
            "date_injured": self.date_injured,
            "date_recovered": self.date_recovered,
            "match_id": self.match_id,
            "status": self.status,
            "notes": self.notes,
            "days_since_injury": self.days_since_injury(),
            "estimated_recovery_days": self.estimated_recovery_days(),
            "recovery_range_min": rec_min,
            "recovery_range_max": rec_max,
            "days_until_expected_recovery": self.days_until_expected_recovery(),
            "injury_risk_score": round(self.injury_risk_score(), 3),
            "created_at": self.created_at,
        }


class InjuryTrackerService:
    def __init__(self, db: Any) -> None:
        self._db = db

    def record_injury(self, record: InjuryRecord) -> int:
        encrypted_notes = encrypt_dict({"notes": record.notes}, ["notes"], in_place=False)["notes"]
        cur = self._db.execute(
            """INSERT INTO injuries (player_id, match_id, injury_type, body_part, severity, mechanism, date_injured, status, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.player_id,
                record.match_id,
                record.injury_type,
                record.body_part,
                record.severity,
                record.mechanism,
                record.date_injured or datetime.now().isoformat(),
                record.status,
                encrypted_notes,
            ),
        )
        self._db.commit()
        return cur.lastrowid

    _INJURY_COLUMNS = (
        "id, player_id, match_id, injury_type, body_part, severity, mechanism, "
        "date_injured, date_recovered, status, notes, created_at, updated_at"
    )

    def get_player_injuries(self, player_id: int) -> list[dict]:
        rows = self._db.execute(
            f"SELECT {self._INJURY_COLUMNS} FROM injuries WHERE player_id = ? ORDER BY date_injured DESC",
            (player_id,),
        ).fetchall()
        return [decrypt_dict(dict(r), ["notes"]) for r in rows]

    def get_active_injuries(self, team_player_ids: list[int] | None = None) -> list[dict]:
        if team_player_ids:
            placeholders = ",".join("?" * len(team_player_ids))
            rows = self._db.execute(
                f"SELECT {self._INJURY_COLUMNS} FROM injuries WHERE status IN ('active','chronic') "
                f"AND player_id IN ({placeholders}) ORDER BY date_injured DESC",
                team_player_ids,
            ).fetchall()
        else:
            rows = self._db.execute(
                f"SELECT {self._INJURY_COLUMNS} FROM injuries WHERE status IN ('active','chronic') "
                f"ORDER BY date_injured DESC",
            ).fetchall()
        return [decrypt_dict(dict(r), ["notes"]) for r in rows]

    def update_injury_status(self, injury_id: int, status: str, date_recovered: str = "") -> None:
        if status == "recovered" and not date_recovered:
            date_recovered = datetime.now().isoformat()
        self._db.execute(
            "UPDATE injuries SET status = ?, date_recovered = ?, updated_at = datetime('now') WHERE id = ?",
            (status, date_recovered, injury_id),
        )
        self._db.commit()

    def get_squad_injury_report(self, team_player_ids: list[int]) -> dict:
        active = self.get_active_injuries(team_player_ids)
        total = len(active)
        by_severity: dict[str, int] = {}
        by_body_part: dict[str, int] = {}
        high_risk: list[dict] = []
        for inj in active:
            s = inj.get("severity", "unknown")
            by_severity[s] = by_severity.get(s, 0) + 1
            bp = inj.get("body_part", "unknown")
            by_body_part[bp] = by_body_part.get(bp, 0) + 1
            risk = inj.get("injury_risk_score", 0)
            if risk and risk > 0.5:
                high_risk.append(inj)
        return {
            "total_active": total,
            "injuries": active,
            "by_severity": by_severity,
            "by_body_part": by_body_part,
            "high_risk_count": len(high_risk),
            "high_risk_injuries": high_risk,
            "report_date": datetime.now().isoformat(),
        }

    def get_injury_stats(self, player_id: int) -> dict:
        rows = self._db.execute(
            "SELECT injury_type, COUNT(*) as cnt FROM injuries WHERE player_id = ? GROUP BY injury_type ORDER BY cnt DESC",
            (player_id,),
        ).fetchall()
        total = self._db.execute(
            "SELECT COUNT(*) as cnt FROM injuries WHERE player_id = ?",
            (player_id,),
        ).fetchone()["cnt"]
        by_type = {r["injury_type"]: r["cnt"] for r in rows}
        recurrent = {k: v for k, v in by_type.items() if v >= 2}
        return {
            "total_injuries": total,
            "by_type": by_type,
            "recurrent_sites": recurrent,
            "has_recurrent_injuries": len(recurrent) > 0,
        }
