"""Multi-match analysis service - season-level analytics and trends.

Aggregates data across multiple matches to provide:
1. Season statistics (team and player)
2. Performance trends over time
3. Match comparison
4. Team evolution (formation changes, tactical shifts)
5. Player development tracking
6. Opposition analysis

This is the core of professional football analytics - no coach cares about
one match in isolation. They want to see patterns, trends, and improvement
over time.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.core.paths import get_paths

logger = get_logger(__name__)


@dataclass
class SeasonSummary:
    """Summary statistics for a season."""

    season_id: int
    season_name: str
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    avg_possession: float
    avg_pass_accuracy: float
    avg_shots_per_match: float
    total_distance_km: float
    avg_max_speed_kmh: float
    formations_used: list[str]
    most_common_formation: str | None


@dataclass
class PlayerTrend:
    """Performance trend for a player over time."""

    player_id: int
    player_name: str
    metric_name: str
    values: list[tuple[str, float]]  # (date, value)
    trend_direction: str  # "improving", "declining", "stable"
    trend_slope: float
    avg_value: float
    best_value: float
    worst_value: float


@dataclass
class MatchComparison:
    """Side-by-side comparison of two matches."""

    match_1_id: int
    match_1_name: str
    match_2_id: int
    match_2_name: str
    possession_diff: dict[str, float]
    shots_diff: dict[str, int]
    passes_diff: dict[str, int]
    formation_diff: dict[str, str]
    line_height_diff: dict[str, float]
    ppda_diff: dict[str, float]
    xg_diff: dict[str, float]
    key_differences: list[str]
    tactical_evolution: str


@dataclass
class TeamEvolution:
    """How team tactics evolved over a period."""

    period: str
    matches_analyzed: int
    formation_trend: list[tuple[str, int]]  # (formation, count)
    possession_trend: list[tuple[str, float]]  # (date, possession%)
    ppda_trend: list[tuple[str, float]]  # (date, PPDA)
    line_height_trend: list[tuple[str, float]]  # (date, line_height_m)
    shot_volume_trend: list[tuple[str, int]]  # (date, shots)
    pass_accuracy_trend: list[tuple[str, float]]  # (date, accuracy%)
    overall_direction: str


class MultiMatchAnalysisService:
    """Aggregates and analyzes data across multiple matches."""

    def __init__(self) -> None:
        self._db_path = get_paths().database
        self._conn: sqlite3.Connection | None = None
        logger.info("MultiMatchAnalysisService initialized")

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    async def get_season_summary(self, season_id: int) -> SeasonSummary:
        """Get aggregated statistics for a season."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM seasons WHERE id = ?", (season_id,))
        season_row = cursor.fetchone()
        season_name = season_row["name"] if season_row else f"Season {season_id}"

        cursor.execute(
            """
            SELECT
                COUNT(*) as matches,
                SUM(score_home) as gf,
                SUM(score_away) as ga,
                AVG(possession_home) as avg_poss,
                AVG(passes_home) as avg_passes,
                AVG(shots_home) as avg_shots
            FROM matches
            WHERE season_id = ?
            """,
            (season_id,),
        )
        row = cursor.fetchone()

        # Compute wins/draws/losses from individual match scores
        wins = 0
        draws = 0
        losses = 0
        total_gf = row["gf"] or 0
        total_ga = row["ga"] or 0
        cursor.execute(
            "SELECT score_home, score_away FROM matches WHERE season_id = ?",
            (season_id,),
        )
        for mr in cursor.fetchall():
            sh = mr["score_home"] or 0
            sa = mr["score_away"] or 0
            if sh > sa:
                wins += 1
            elif sh < sa:
                losses += 1
            else:
                draws += 1

        # Get formations used
        cursor.execute(
            """
            SELECT full_data FROM analysis_results ar
            JOIN matches m ON ar.match_id = m.id
            WHERE m.season_id = ?
            """,
            (season_id,),
        )
        formations: defaultdict[str, int] = defaultdict(int)
        for r in cursor.fetchall():
            try:
                data = json.loads(r["full_data"] or "{}")
                for team in ["home", "away"]:
                    form = data.get("formations", {}).get(team, {}).get("formation")
                    if form:
                        formations[form] += 1
            except Exception:
                pass

        most_common = max(formations, key=formations.get) if formations else None  # type: ignore[arg-type,return-value]

        # Compute pass accuracy from available event data
        avg_pass_acc = 0.0
        cursor.execute(
            """
            SELECT AVG(accuracy) as avg_acc FROM (
                SELECT
                    CAST(SUM(CASE WHEN e.type = 'pass' AND e.outcome = 1 THEN 1 ELSE 0 END) AS REAL) /
                    NULLIF(CAST(COUNT(CASE WHEN e.type = 'pass' THEN 1 END) AS REAL), 0) as accuracy
                FROM events e
                JOIN matches m ON e.match_id = m.id
                WHERE m.season_id = ?
                GROUP BY e.match_id
            )
            """,
            (season_id,),
        )
        acc_row = cursor.fetchone()
        if acc_row and acc_row["avg_acc"] is not None:
            avg_pass_acc = round(acc_row["avg_acc"] * 100, 1)

        return SeasonSummary(
            season_id=season_id,
            season_name=season_name,
            matches_played=row["matches"] or 0,
            wins=wins,
            draws=draws,
            losses=losses,
            goals_for=total_gf,
            goals_against=total_ga,
            avg_possession=round(row["avg_poss"] or 0, 1),
            avg_pass_accuracy=avg_pass_acc,
            avg_shots_per_match=round(row["avg_shots"] or 0, 1),
            total_distance_km=0.0,
            avg_max_speed_kmh=0.0,
            formations_used=list(formations.keys()),
            most_common_formation=most_common,
        )

    async def get_player_trend(
        self,
        player_id: int,
        metric: str = "distance_covered_m",
        min_matches: int = 3,
    ) -> PlayerTrend | None:
        """Get performance trend for a player over time.

        Args:
            player_id: Profile ID from player_profiles
            metric: Metric to track (distance_covered_m, max_speed_kmh, shots, etc.)
            min_matches: Minimum matches for a meaningful trend
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                p.display_name,
                m.match_date,
                pl.distance_covered_m,
                pl.max_speed_kmh,
                pl.avg_speed_kmh,
                pl.passes_attempted,
                pl.passes_completed,
                pl.shots,
                pl.tackles
            FROM player_match_links l
            JOIN matches m ON l.match_id = m.id
            JOIN player_profiles p ON l.player_id = p.id
            LEFT JOIN players pl ON pl.match_id = m.id AND pl.track_id = l.track_id
            WHERE l.player_id = ?
            ORDER BY m.match_date ASC
            """,
            (player_id,),
        )
        rows = cursor.fetchall()

        if len(rows) < min_matches:
            return None

        player_name = rows[0]["display_name"] if rows[0] else "Unknown"
        values: list[tuple[str, float]] = []
        for row in rows:
            date = row["match_date"] or row[0]  # fallback
            if metric == "pass_accuracy":
                attempted = row["passes_attempted"] or 0
                completed = row["passes_completed"] or 0
                val = (completed / attempted * 100) if attempted > 0 else 0.0
            else:
                val = row[metric] or 0.0
            values.append((str(date), float(val)))

        # Simple linear regression for trend
        n = len(values)
        x = list(range(n))
        y = [v[1] for v in values]
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y, strict=False))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        slope = numerator / denominator if denominator > 0 else 0.0

        if slope > 0.05 * y_mean:
            direction = "improving"
        elif slope < -0.05 * y_mean:
            direction = "declining"
        else:
            direction = "stable"

        return PlayerTrend(
            player_id=player_id,
            player_name=player_name,
            metric_name=metric,
            values=values,
            trend_direction=direction,
            trend_slope=round(slope, 4),
            avg_value=round(y_mean, 2),
            best_value=round(max(y), 2),
            worst_value=round(min(y), 2),
        )

    async def compare_matches(self, match_id_1: int, match_id_2: int) -> MatchComparison:
        """Compare two matches side-by-side."""
        conn = self._get_conn()
        cursor = conn.cursor()

        def get_match_data(mid: int) -> dict[str, Any]:
            cursor.execute(
                """
                SELECT * FROM analysis_results WHERE match_id = ?
                """,
                (mid,),
            )
            row = cursor.fetchone()
            if not row:
                return {}
            data = dict(row)
            try:
                full = json.loads(data.get("full_data", "{}"))
                data.update(full)
            except Exception:
                pass
            return data

        def get_match_info(mid: int) -> dict[str, Any]:
            cursor.execute("SELECT * FROM matches WHERE id = ?", (mid,))
            row = cursor.fetchone()
            return dict(row) if row else {}

        m1 = get_match_data(match_id_1)
        m2 = get_match_data(match_id_2)
        info1 = get_match_info(match_id_1)
        info2 = get_match_info(match_id_2)

        # Compute differences
        key_diffs = []
        possession_diff = {}
        shots_diff = {}
        passes_diff = {}
        formation_diff = {}
        line_height_diff: dict[str, float] = {}
        ppda_diff: dict[str, float] = {}
        xg_diff: dict[str, float] = {}

        if m1 and m2:
            p1 = m1.get("possession_home", 0) or 0
            p2 = m2.get("possession_home", 0) or 0
            possession_diff = {"match_1": p1, "match_2": p2, "delta": round(p2 - p1, 1)}
            if abs(p2 - p1) > 10:
                key_diffs.append(f"Possession changed by {abs(p2 - p1):.1f}%")

            s1 = m1.get("shots_home", 0) or 0
            s2 = m2.get("shots_home", 0) or 0
            shots_diff = {"match_1": s1, "match_2": s2, "delta": s2 - s1}
            if abs(s2 - s1) >= 3:
                key_diffs.append(f"Shot volume changed by {abs(s2 - s1)}")

            pa1 = m1.get("passes_home", 0) or 0
            pa2 = m2.get("passes_home", 0) or 0
            passes_diff = {"match_1": pa1, "match_2": pa2, "delta": pa2 - pa1}

            formations = m1.get("formations", {})
            formations2 = m2.get("formations", {})
            f1 = (
                formations.get("home", {}).get("formation", "unknown")
                if isinstance(formations, dict)
                else "unknown"
            )
            f2 = (
                formations2.get("home", {}).get("formation", "unknown")
                if isinstance(formations2, dict)
                else "unknown"
            )
            formation_diff = {"match_1": f1, "match_2": f2}
            if f1 != f2:
                key_diffs.append(f"Formation changed from {f1} to {f2}")

        evolution = "No significant tactical evolution detected"
        if key_diffs:
            if len(key_diffs) >= 2:
                evolution = f"Multiple tactical changes: {', '.join(key_diffs[:2])}"
            else:
                evolution = key_diffs[0]

        return MatchComparison(
            match_1_id=match_id_1,
            match_1_name=info1.get("name", f"Match {match_id_1}"),
            match_2_id=match_id_2,
            match_2_name=info2.get("name", f"Match {match_id_2}"),
            possession_diff=possession_diff,
            shots_diff=shots_diff,
            passes_diff=passes_diff,
            formation_diff=formation_diff,
            line_height_diff=line_height_diff,
            ppda_diff=ppda_diff,
            xg_diff=xg_diff,
            key_differences=key_diffs,
            tactical_evolution=evolution,
        )

    async def get_team_evolution(
        self, season_id: int | None = None, match_ids: list[int] | None = None
    ) -> TeamEvolution:
        """Analyze how team tactics evolved over a period."""
        conn = self._get_conn()
        cursor = conn.cursor()

        if match_ids:
            placeholders = ",".join("?" * len(match_ids))
            cursor.execute(
                f"""
                SELECT ar.match_id, ar.full_data, m.match_date, m.name
                FROM analysis_results ar
                JOIN matches m ON ar.match_id = m.id
                WHERE ar.match_id IN ({placeholders})
                ORDER BY m.match_date ASC
                """,
                match_ids,
            )
        elif season_id:
            cursor.execute(
                """
                SELECT ar.match_id, ar.full_data, m.match_date, m.name
                FROM analysis_results ar
                JOIN matches m ON ar.match_id = m.id
                WHERE m.season_id = ?
                ORDER BY m.match_date ASC
                """,
                (season_id,),
            )
        else:
            cursor.execute(
                """
                SELECT ar.match_id, ar.full_data, m.match_date, m.name
                FROM analysis_results ar
                JOIN matches m ON ar.match_id = m.id
                ORDER BY m.match_date ASC
                """
            )

        rows = cursor.fetchall()

        formation_counts: defaultdict[str, int] = defaultdict(int)
        possession_trend = []
        ppda_trend = []
        line_height_trend = []
        shot_trend = []
        pass_acc_trend = []

        for row in rows:
            date = row["match_date"] or ""
            try:
                data = json.loads(row["full_data"] or "{}")
                # Formation
                form = data.get("formations", {}).get("home", {}).get("formation")
                if form:
                    formation_counts[form] += 1
                # Possession
                poss = data.get("possession_home", 0)
                possession_trend.append((str(date), float(poss)))
                # PPDA
                ppda = data.get("pressing_intensity", 0)
                ppda_trend.append((str(date), float(ppda)))
                # Line height
                lh = data.get("formations", {}).get("home", {}).get("line_height_m")
                if lh:
                    line_height_trend.append((str(date), float(lh)))
                # Shots
                shots = data.get("shots_home", 0)
                shot_trend.append((str(date), int(shots)))
                # Pass accuracy (from team stats if available)
                home = data.get("home_team", {})
                pa = home.get("pass_accuracy", 0) * 100 if isinstance(home, dict) else 0
                pass_acc_trend.append((str(date), float(pa)))
            except Exception:
                pass

        # Determine overall direction
        direction = "stable"
        if len(possession_trend) >= 2:
            first_poss = possession_trend[0][1]
            last_poss = possession_trend[-1][1]
            if last_poss - first_poss > 5:
                direction = "more possession-oriented"
            elif first_poss - last_poss > 5:
                direction = "more direct"

        if len(ppda_trend) >= 2:
            first_ppda = ppda_trend[0][1]
            last_ppda = ppda_trend[-1][1]
            if first_ppda > last_ppda + 2:
                direction += ", more pressing"
            elif last_ppda > first_ppda + 2:
                direction += ", less pressing"

        formation_list = sorted(formation_counts.items(), key=lambda x: -x[1])

        return TeamEvolution(
            period="season" if season_id else "custom",
            matches_analyzed=len(rows),
            formation_trend=formation_list,
            possession_trend=possession_trend,
            ppda_trend=ppda_trend,
            line_height_trend=line_height_trend,
            shot_volume_trend=shot_trend,
            pass_accuracy_trend=pass_acc_trend,
            overall_direction=direction.strip(", "),
        )

    async def get_leaderboard(
        self,
        season_id: int | None = None,
        metric: str = "distance_covered_m",
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Get player leaderboard for a metric across matches."""
        conn = self._get_conn()
        cursor = conn.cursor()

        if season_id:
            cursor.execute(
                """
                SELECT
                    p.id,
                    p.display_name,
                    p.jersey_number,
                    p.preferred_position,
                    AVG(pl.distance_covered_m) as avg_distance,
                    AVG(pl.max_speed_kmh) as avg_max_speed,
                    AVG(pl.avg_speed_kmh) as avg_speed,
                    SUM(pl.shots) as total_shots,
                    SUM(pl.passes_attempted) as total_passes,
                    SUM(pl.passes_completed) as total_passes_completed,
                    COUNT(DISTINCT pl.match_id) as matches
                FROM player_match_links l
                JOIN player_profiles p ON l.player_id = p.id
                LEFT JOIN players pl ON pl.match_id = l.match_id AND pl.track_id = l.track_id
                JOIN matches m ON l.match_id = m.id
                WHERE m.season_id = ?
                GROUP BY p.id
                ORDER BY avg_distance DESC
                LIMIT ?
                """,
                (season_id, top_n),
            )
        else:
            cursor.execute(
                """
                SELECT
                    p.id,
                    p.display_name,
                    p.jersey_number,
                    p.preferred_position,
                    AVG(pl.distance_covered_m) as avg_distance,
                    AVG(pl.max_speed_kmh) as avg_max_speed,
                    AVG(pl.avg_speed_kmh) as avg_speed,
                    SUM(pl.shots) as total_shots,
                    SUM(pl.passes_attempted) as total_passes,
                    SUM(pl.passes_completed) as total_passes_completed,
                    COUNT(DISTINCT pl.match_id) as matches
                FROM player_match_links l
                JOIN player_profiles p ON l.player_id = p.id
                LEFT JOIN players pl ON pl.match_id = l.match_id AND pl.track_id = l.track_id
                GROUP BY p.id
                ORDER BY avg_distance DESC
                LIMIT ?
                """,
                (top_n,),
            )

        rows = cursor.fetchall()
        result = []
        for row in rows:
            total_passes = row["total_passes"] or 0
            completed = row["total_passes_completed"] or 0
            pass_acc = completed / total_passes if total_passes > 0 else 0.0
            result.append(
                {
                    "player_id": row["id"],
                    "name": row["display_name"],
                    "jersey": row["jersey_number"],
                    "position": row["preferred_position"],
                    "matches": row["matches"],
                    "avg_distance_m": round(row["avg_distance"] or 0, 1),
                    "avg_max_speed_kmh": round(row["avg_max_speed"] or 0, 2),
                    "avg_speed_kmh": round(row["avg_speed"] or 0, 2),
                    "total_shots": row["total_shots"] or 0,
                    "pass_accuracy": round(pass_acc, 3),
                }
            )
        return result

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
