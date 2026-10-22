"""Contract Tracker — manage player contracts and expiry alerts."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class ContractTracker:
    _CONTRACT_COLUMNS = (
        "id, player_profile_id, player_name, contract_type, start_date, end_date, "
        "club_option_years, player_option_years, release_clause_millions, "
        "wage_weekly_pounds, agent_name, notes, last_updated"
    )

    def __init__(
        self, db_path: str | Path | None = None, conn: sqlite3.Connection | None = None
    ) -> None:
        self._db_path = Path(db_path) if db_path else None
        self._conn: sqlite3.Connection | None = conn

    @property
    def conn(self) -> sqlite3.Connection | None:
        if self._conn is None and self._db_path is not None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def set_connection(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def add_contract(
        self,
        player_profile_id: int,
        player_name: str,
        contract_type: str = "permanent",
        start_date: str = "",
        end_date: str = "",
        club_option_years: int = 0,
        player_option_years: int = 0,
        release_clause_millions: float | None = None,
        wage_weekly_pounds: float | None = None,
        agent_name: str = "",
        notes: str = "",
    ) -> int:
        if self.conn is None:
            return 0
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO player_contracts
                (player_profile_id, player_name, contract_type, start_date, end_date,
                 club_option_years, player_option_years, release_clause_millions,
                 wage_weekly_pounds, agent_name, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                player_profile_id,
                player_name,
                contract_type,
                start_date,
                end_date,
                club_option_years,
                player_option_years,
                release_clause_millions,
                wage_weekly_pounds,
                agent_name,
                notes,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid or 0

    def update_contract(self, contract_id: int, **updates: Any) -> bool:
        if self.conn is None:
            return False
        allowed = {
            "contract_type",
            "start_date",
            "end_date",
            "club_option_years",
            "player_option_years",
            "release_clause_millions",
            "wage_weekly_pounds",
            "agent_name",
            "notes",
        }
        sets = []
        vals: list[Any] = []
        for key, val in updates.items():
            if key in allowed:
                sets.append(f"{key} = ?")
                vals.append(val)
        if not sets:
            return False
        sets.append("last_updated = datetime('now')")
        vals.append(contract_id)
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE player_contracts SET {', '.join(sets)} WHERE id = ?", vals)
        self.conn.commit()
        return cursor.rowcount > 0

    def get_expiring_contracts(self, within_months: int = 6) -> list[dict]:
        if self.conn is None:
            return []
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT {self._CONTRACT_COLUMNS} FROM player_contracts
            WHERE end_date BETWEEN date('now') AND date('now', '+' || ? || ' months')
            ORDER BY end_date ASC
            """,
            (within_months,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_expired_contracts(self) -> list[dict]:
        if self.conn is None:
            return []
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT {self._CONTRACT_COLUMNS} FROM player_contracts WHERE end_date < date('now') ORDER BY end_date DESC"
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_contracts_ending_this_season(self) -> list[dict]:
        if self.conn is None:
            return []
        now = datetime.now()
        year = now.year
        season_end = f"{year + 1}-06-30" if now.month >= 7 else f"{year}-06-30"
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT {self._CONTRACT_COLUMNS} FROM player_contracts
            WHERE end_date BETWEEN date('now') AND date(?)
            ORDER BY end_date ASC
            """,
            (season_end,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_squad_contract_summary(self) -> dict:
        if self.conn is None:
            return {
                "total_contracts": 0,
                "expiring_this_season": 0,
                "by_type": {},
                "avg_wage": 0.0,
                "total_wage_bill": 0.0,
                "release_clause_total": 0.0,
            }
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as c FROM player_contracts")
        total = cursor.fetchone()["c"]
        cursor.execute(
            "SELECT contract_type, COUNT(*) as c FROM player_contracts GROUP BY contract_type"
        )
        by_type = {row["contract_type"]: row["c"] for row in cursor.fetchall()}
        cursor.execute(
            "SELECT AVG(wage_weekly_pounds) as a FROM player_contracts WHERE wage_weekly_pounds IS NOT NULL"
        )
        row = cursor.fetchone()
        avg_wage = round(float(row["a"]), 2) if row and row["a"] else 0.0
        cursor.execute(
            "SELECT COALESCE(SUM(wage_weekly_pounds), 0) as s FROM player_contracts WHERE wage_weekly_pounds IS NOT NULL"
        )
        total_wage = round(float(cursor.fetchone()["s"]), 2)
        cursor.execute(
            "SELECT COALESCE(SUM(release_clause_millions), 0) as s FROM player_contracts WHERE release_clause_millions IS NOT NULL"
        )
        release_total = round(float(cursor.fetchone()["s"]), 2)
        expiring = len(self.get_contracts_ending_this_season())
        return {
            "total_contracts": total,
            "expiring_this_season": expiring,
            "by_type": by_type,
            "avg_wage": avg_wage,
            "total_wage_bill": total_wage,
            "release_clause_total": release_total,
        }

    def get_contract_alerts(self) -> list[dict]:
        if self.conn is None:
            return []
        alerts: list[dict] = []
        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT {self._CONTRACT_COLUMNS} FROM player_contracts
            WHERE end_date BETWEEN date('now') AND date('now', '+3 months')
            ORDER BY end_date ASC
            """
        )
        for row in cursor.fetchall():
            alerts.append(
                {
                    "type": "expiring_soon",
                    "message": f"{row['player_name']} contract expires in less than 3 months ({row['end_date']})",
                    "contract_id": row["id"],
                    "player_name": row["player_name"],
                    "end_date": row["end_date"],
                }
            )
        cursor.execute(
            f"""
            SELECT {self._CONTRACT_COLUMNS} FROM player_contracts WHERE player_option_years > 0
            """
        )
        for row in cursor.fetchall():
            alerts.append(
                {
                    "type": "player_option",
                    "message": f"{row['player_name']} can trigger {row['player_option_years']}-year extension",
                    "contract_id": row["id"],
                    "player_name": row["player_name"],
                    "option_years": row["player_option_years"],
                }
            )
        return alerts
