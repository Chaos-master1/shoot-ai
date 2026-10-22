"""Shortlist Service — manage scouted/recruited player shortlist."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class ShortlistService:
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

    def add_to_shortlist(
        self,
        player_id: str,
        player_name: str,
        position: str = "",
        team: str = "",
        league: str = "",
        priority: str = "medium",
        notes: str = "",
        scout_rating: float = 0.0,
        age: int | None = None,
        nationality: str = "",
        estimated_value: float | None = None,
    ) -> int:
        if self.conn is None:
            return 0
        existing = self.get_player_on_shortlist(player_id)
        if existing is not None:
            return existing["id"]
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO player_shortlist
                (player_id, player_name, position, team, league, priority, notes,
                 scout_rating, estimated_value, age, nationality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                player_id,
                player_name,
                position,
                team,
                league,
                priority,
                notes,
                scout_rating,
                estimated_value,
                age,
                nationality,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid or 0

    def update_status(self, entry_id: int, status: str) -> bool:
        if self.conn is None:
            return False
        valid = {"scouted", "shortlisted", "contacted", "trial", "signed", "rejected", "archived"}
        if status not in valid:
            return False
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE player_shortlist SET status = ?, last_updated = datetime('now') WHERE id = ?",
            (status, entry_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def update_priority(self, entry_id: int, priority: str) -> bool:
        if self.conn is None:
            return False
        valid = {"low", "medium", "high", "urgent"}
        if priority not in valid:
            return False
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE player_shortlist SET priority = ?, last_updated = datetime('now') WHERE id = ?",
            (priority, entry_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def update_notes(self, entry_id: int, notes: str) -> bool:
        if self.conn is None:
            return False
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE player_shortlist SET notes = ?, last_updated = datetime('now') WHERE id = ?",
            (notes, entry_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def remove_player(self, entry_id: int) -> bool:
        return self.update_status(entry_id, "archived")

    def get_shortlist(
        self,
        status: str | None = None,
        priority: str | None = None,
        position: str | None = None,
        sort_by: str = "added_date",
        sort_dir: str = "DESC",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        if self.conn is None:
            return []
        allowed_sort = {
            "added_date",
            "last_updated",
            "player_name",
            "scout_rating",
            "estimated_value",
            "age",
            "priority",
        }
        if sort_by not in allowed_sort:
            sort_by = "added_date"
        sort_dir = "ASC" if sort_dir.upper() == "ASC" else "DESC"
        where = []
        params: list[Any] = []
        if status is not None:
            where.append("status = ?")
            params.append(status)
        if priority is not None:
            where.append("priority = ?")
            params.append(priority)
        if position is not None:
            where.append("position LIKE ?")
            params.append(f"%{position}%")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        cursor = self.conn.cursor()
        cursor.execute(
            f"SELECT id, player_id, player_name, position, team, league, added_date, "
            f"priority, status, notes, scout_rating, estimated_value, age, nationality, "
            f"last_updated FROM player_shortlist{clause} ORDER BY {sort_by} {sort_dir} LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_player_on_shortlist(self, player_id: str) -> dict | None:
        if self.conn is None:
            return None
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, player_id, player_name, position, team, league, added_date, "
            "priority, status, notes, scout_rating, estimated_value, age, nationality, "
            "last_updated FROM player_shortlist WHERE player_id = ? AND status != 'archived' "
            "ORDER BY added_date DESC LIMIT 1",
            (player_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_shortlist_stats(self) -> dict:
        if self.conn is None:
            return {"total": 0, "by_status": {}, "by_priority": {}, "by_position": {}}
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM player_shortlist WHERE status != 'archived'")
        total = cursor.fetchone()["total"]
        cursor.execute("SELECT status, COUNT(*) as cnt FROM player_shortlist GROUP BY status")
        by_status = {row["status"]: row["cnt"] for row in cursor.fetchall()}
        cursor.execute("SELECT priority, COUNT(*) as cnt FROM player_shortlist GROUP BY priority")
        by_priority = {row["priority"]: row["cnt"] for row in cursor.fetchall()}
        cursor.execute(
            "SELECT position, COUNT(*) as cnt FROM player_shortlist WHERE position != '' GROUP BY position ORDER BY cnt DESC LIMIT 10"
        )
        by_position = {row["position"]: row["cnt"] for row in cursor.fetchall()}
        return {
            "total": total,
            "by_status": by_status,
            "by_priority": by_priority,
            "by_position": by_position,
        }
