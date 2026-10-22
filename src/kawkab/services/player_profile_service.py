"""Player profile service - persistent player identity across matches.

Professional football analytics requires tracking players across multiple
matches. This service manages:
1. Player profile creation and management (name, photo, jersey, position, physical attributes)
2. Linking match-tracked players to persistent profiles
3. Career statistics across all matches
4. Performance trends and baselines

A player can be identified by:
- Manual assignment (coach confirms identity)
- Jersey number + team (semi-automatic)
- Track features + temporal consistency (automatic, lower confidence)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.core.paths import get_paths

logger = get_logger(__name__)


@dataclass
class PlayerProfile:
    """Persistent player profile that spans multiple matches."""

    id: int
    global_id: str
    display_name: str | None = None
    jersey_number: int | None = None
    preferred_position: str | None = None
    height_cm: int | None = None
    weight_kg: int | None = None
    dominant_foot: str | None = None
    date_of_birth: str | None = None
    nationality: str | None = None
    photo_path: str | None = None
    team: str = "home"
    is_active: bool = True
    created_at: str | None = None
    updated_at: str | None = None

    # Career stats (computed)
    matches_played: int = 0
    total_distance_km: float = 0.0
    avg_max_speed_kmh: float = 0.0
    total_goals: int = 0
    total_assists: int = 0
    pass_accuracy_avg: float = 0.0


@dataclass
class PlayerMatchAppearance:
    """A single match appearance by a player."""

    match_id: int
    match_name: str
    match_date: str | None
    opponent: str | None
    team: str
    jersey_number: int | None
    position: str | None
    minutes_played: float = 0.0
    distance_covered_m: float = 0.0
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    passes_attempted: int = 0
    passes_completed: int = 0
    shots: int = 0
    tackles: int = 0
    xg: float = 0.0
    xt: float = 0.0


class PlayerProfileService:
    """Manages persistent player profiles across matches."""

    def __init__(self) -> None:
        self._db_path = get_paths().database
        self._conn: sqlite3.Connection | None = None
        logger.info("PlayerProfileService initialized")

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    async def create_profile(
        self,
        global_id: str | None = None,
        display_name: str | None = None,
        jersey_number: int | None = None,
        preferred_position: str | None = None,
        height_cm: int | None = None,
        weight_kg: int | None = None,
        dominant_foot: str | None = None,
        date_of_birth: str | None = None,
        nationality: str | None = None,
        team: str = "home",
        football_data_person_id: int | None = None,
        football_data_team_id: int | None = None,
        bzzoiro_person_id: int | None = None,
        bzzoiro_team_id: int | None = None,
        apifb_person_id: int | None = None,
        apifb_team_id: int | None = None,
    ) -> PlayerProfile:
        """Create a new player profile."""
        conn = self._get_conn()
        cursor = conn.cursor()

        if global_id is None:
            global_id = f"player_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        cursor.execute(
            """
            INSERT INTO player_profiles (
                global_id, display_name, jersey_number, preferred_position,
                height_cm, weight_kg, dominant_foot, date_of_birth, nationality, team,
                football_data_person_id, football_data_team_id,
                bzzoiro_person_id, bzzoiro_team_id,
                apifb_person_id, apifb_team_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                global_id,
                display_name,
                jersey_number,
                preferred_position,
                height_cm,
                weight_kg,
                dominant_foot,
                date_of_birth,
                nationality,
                team,
                football_data_person_id,
                football_data_team_id,
                bzzoiro_person_id,
                bzzoiro_team_id,
                apifb_person_id,
                apifb_team_id,
            ),
        )
        conn.commit()
        profile_id = cursor.lastrowid or 0

        logger.info(f"Created player profile: {display_name} (ID: {profile_id})")
        profile = await self.get_profile(profile_id)
        if profile is None:
            # get_profile() is Optional (a lookup can legitimately miss);
            # this method's return type promises a real PlayerProfile since
            # the row was just inserted in this same call. Fail with a
            # clear message instead of letting None silently satisfy a
            # non-Optional return type -- callers here index straight into
            # profile.id without a None-check.
            raise RuntimeError(f"Failed to retrieve newly created player profile {profile_id}")
        return profile

    async def get_profile(self, profile_id: int) -> PlayerProfile | None:
        """Get a player profile by ID."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM player_profiles WHERE id = ?", (profile_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_profile(dict(row))

    async def get_profile_by_global_id(self, global_id: str) -> PlayerProfile | None:
        """Get a player profile by global_id."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM player_profiles WHERE global_id = ?", (global_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_profile(dict(row))

    async def get_all_profiles(self, team: str | None = None) -> list[PlayerProfile]:
        """Get all player profiles, optionally filtered by team."""
        conn = self._get_conn()
        cursor = conn.cursor()
        if team:
            cursor.execute(
                "SELECT * FROM player_profiles WHERE team = ? AND is_active = 1", (team,)
            )
        else:
            cursor.execute("SELECT * FROM player_profiles WHERE is_active = 1")
        rows = cursor.fetchall()
        return [self._row_to_profile(dict(row)) for row in rows]

    async def update_profile(self, profile_id: int, **kwargs) -> PlayerProfile | None:
        """Update a player profile."""
        conn = self._get_conn()
        cursor = conn.cursor()

        allowed_fields = {
            "display_name",
            "jersey_number",
            "preferred_position",
            "height_cm",
            "weight_kg",
            "dominant_foot",
            "date_of_birth",
            "nationality",
            "photo_path",
            "team",
            "is_active",
            "football_data_person_id",
            "football_data_team_id",
            "bzzoiro_person_id",
            "bzzoiro_team_id",
            "apifb_person_id",
            "apifb_team_id",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return await self.get_profile(profile_id)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [profile_id]

        cursor.execute(
            f"UPDATE player_profiles SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            values,
        )
        conn.commit()
        return await self.get_profile(profile_id)

    async def link_match_player(
        self,
        profile_id: int,
        match_id: int,
        track_id: int | None = None,
        confidence: float = 0.0,
    ) -> bool:
        """Link a match player to a persistent profile."""
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO player_match_links
                (player_id, match_id, track_id, confidence, created_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (profile_id, match_id, track_id, confidence),
            )
            conn.commit()
            logger.info(f"Linked profile {profile_id} to match {match_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to link player: {e}")
            return False

    async def get_profile_appearances(self, profile_id: int) -> list[PlayerMatchAppearance]:
        """Get all match appearances for a player."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                m.id as match_id,
                m.name as match_name,
                m.match_date,
                m.opponent,
                p.team,
                p.jersey_number,
                p.position,
                p.distance_covered_m,
                p.max_speed_kmh,
                p.avg_speed_kmh,
                p.passes_attempted,
                p.passes_completed,
                p.shots,
                p.tackles
            FROM player_match_links l
            JOIN matches m ON l.match_id = m.id
            LEFT JOIN players p ON p.match_id = m.id AND p.track_id = l.track_id
            WHERE l.player_id = ?
            ORDER BY m.match_date DESC
            """,
            (profile_id,),
        )
        rows = cursor.fetchall()
        return [self._row_to_appearance(dict(row)) for row in rows]

    async def compute_career_stats(self, profile_id: int) -> dict[str, Any]:
        """Compute aggregated career statistics for a player."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                COUNT(DISTINCT m.id) as matches_played,
                SUM(p.distance_covered_m) as total_distance,
                AVG(p.max_speed_kmh) as avg_max_speed,
                AVG(p.avg_speed_kmh) as avg_speed,
                SUM(p.shots) as total_shots,
                SUM(p.passes_attempted) as total_passes_attempted,
                SUM(p.passes_completed) as total_passes_completed,
                SUM(p.tackles) as total_tackles
            FROM player_match_links l
            JOIN matches m ON l.match_id = m.id
            LEFT JOIN players p ON p.match_id = m.id AND p.track_id = l.track_id
            WHERE l.player_id = ?
            """,
            (profile_id,),
        )
        row = cursor.fetchone()
        if not row:
            return {}

        data = dict(row)
        total_passes = data.get("total_passes_attempted", 0) or 0
        completed_passes = data.get("total_passes_completed", 0) or 0
        pass_accuracy = completed_passes / total_passes if total_passes > 0 else 0.0

        return {
            "matches_played": data.get("matches_played", 0),
            "total_distance_km": round((data.get("total_distance", 0) or 0) / 1000, 2),
            "avg_max_speed_kmh": round(data.get("avg_max_speed", 0) or 0, 2),
            "avg_speed_kmh": round(data.get("avg_speed", 0) or 0, 2),
            "total_shots": data.get("total_shots", 0) or 0,
            "total_passes_attempted": total_passes,
            "total_passes_completed": completed_passes,
            "pass_accuracy": round(pass_accuracy, 3),
            "total_tackles": data.get("total_tackles", 0) or 0,
        }

    async def auto_link_by_jersey(
        self, match_id: int, team: str | None = None
    ) -> list[dict[str, Any]]:
        """Auto-link match players to profiles by jersey number.

        Returns list of proposed links with confidence scores.
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        # Get match players
        team_filter = "AND team = ?" if team else ""
        params: list[Any] = [match_id]
        if team:
            params.append(team)

        cursor.execute(
            f"SELECT track_id, jersey_number, team FROM players WHERE match_id = ? {team_filter}",
            params,
        )
        match_players = cursor.fetchall()

        proposals = []
        for mp in match_players:
            track_id = mp["track_id"]
            jersey = mp["jersey_number"]
            player_team = mp["team"]

            if jersey is None:
                continue

            # Find matching profile by jersey + team
            cursor.execute(
                "SELECT id, display_name, jersey_number FROM player_profiles WHERE jersey_number = ? AND team = ?",
                (jersey, player_team),
            )
            profile = cursor.fetchone()

            if profile:
                proposals.append(
                    {
                        "track_id": track_id,
                        "profile_id": profile["id"],
                        "profile_name": profile["display_name"],
                        "jersey_number": jersey,
                        "team": player_team,
                        "confidence": 0.7,  # jersey match is reasonably confident
                        "method": "jersey_number",
                    }
                )

        return proposals

    async def get_team_roster(self, team: str = "home") -> list[PlayerProfile]:
        """Get the current roster for a team."""
        return await self.get_all_profiles(team=team)

    def _row_to_profile(self, row: dict[str, Any]) -> PlayerProfile:
        return PlayerProfile(
            id=row["id"],
            global_id=row["global_id"],
            display_name=row.get("display_name"),
            jersey_number=row.get("jersey_number"),
            preferred_position=row.get("preferred_position"),
            height_cm=row.get("height_cm"),
            weight_kg=row.get("weight_kg"),
            dominant_foot=row.get("dominant_foot"),
            date_of_birth=row.get("date_of_birth"),
            nationality=row.get("nationality"),
            photo_path=row.get("photo_path"),
            team=row.get("team", "home"),
            is_active=bool(row.get("is_active", 1)),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    def _row_to_appearance(self, row: dict[str, Any]) -> PlayerMatchAppearance:
        return PlayerMatchAppearance(
            match_id=row["match_id"],
            match_name=row.get("match_name", ""),
            match_date=row.get("match_date"),
            opponent=row.get("opponent"),
            team=row.get("team", "home"),
            jersey_number=row.get("jersey_number"),
            position=row.get("position"),
            distance_covered_m=row.get("distance_covered_m", 0.0) or 0.0,
            max_speed_kmh=row.get("max_speed_kmh", 0.0) or 0.0,
            avg_speed_kmh=row.get("avg_speed_kmh", 0.0) or 0.0,
            passes_attempted=row.get("passes_attempted", 0) or 0,
            passes_completed=row.get("passes_completed", 0) or 0,
            shots=row.get("shots", 0) or 0,
            tackles=row.get("tackles", 0) or 0,
        )

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
