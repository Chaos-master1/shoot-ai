"""Season aggregation — tracks player and team stats across multiple matches.

Produces per-player season summaries, team trends, and head-to-head
comparisons similar to professional platforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlayerSeasonStats:
    """Aggregated player statistics across a season."""

    name: str = ""
    team: str = ""
    position: str = ""
    matches_played: int = 0
    total_minutes: float = 0.0
    goals: float = 0.0
    assists: int = 0
    shots: int = 0
    shots_on_target: int = 0
    passes_attempted: int = 0
    passes_completed: int = 0
    progressive_passes: int = 0
    key_passes: int = 0
    tackles: int = 0
    interceptions: int = 0
    carries: int = 0
    progressive_carries: int = 0
    total_distance_km: float = 0.0
    avg_rating: float = 0.0
    total_xg: float = 0.0

    def per90(self, value: float) -> float:
        return value / max(self.total_minutes / 90.0, 0.1)

    @property
    def pass_accuracy(self) -> float:
        if self.passes_attempted == 0:
            return 0.0
        return self.passes_completed / self.passes_attempted

    @property
    def goals_per90(self) -> float:
        return self.per90(self.goals)

    @property
    def assists_per90(self) -> float:
        return self.per90(float(self.assists))

    @property
    def shots_per90(self) -> float:
        return self.per90(float(self.shots))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "team": self.team,
            "position": self.position,
            "matches": self.matches_played,
            "minutes": round(self.total_minutes, 1),
            "goals": round(self.goals, 1),
            "assists": self.assists,
            "goals_per90": round(self.goals_per90, 2),
            "assists_per90": round(self.assists_per90, 2),
            "shots": self.shots,
            "shots_on_target": self.shots_on_target,
            "pass_accuracy": round(self.pass_accuracy, 3),
            "progressive_passes": self.progressive_passes,
            "key_passes": self.key_passes,
            "tackles": self.tackles,
            "interceptions": self.interceptions,
            "progressive_carries": self.progressive_carries,
            "distance_km": round(self.total_distance_km, 1),
            "avg_rating": round(self.avg_rating, 1),
            "total_xg": round(self.total_xg, 2),
        }


@dataclass
class SeasonReport:
    """Complete season report for a team."""

    team_name: str = ""
    matches: int = 0
    players: dict[int, PlayerSeasonStats] = field(default_factory=dict)
    total_goals: float = 0.0
    total_shots: int = 0
    total_passes: int = 0
    total_xg: float = 0.0
    avg_possession: float = 0.0
    avg_pass_accuracy: float = 0.0
    avg_rating: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_name": self.team_name,
            "matches": self.matches,
            "players": {str(k): v.to_dict() for k, v in self.players.items()},
            "total_goals": round(self.total_goals, 1),
            "total_shots": self.total_shots,
            "total_passes": self.total_passes,
            "total_xg": round(self.total_xg, 2),
            "avg_possession": round(self.avg_possession, 1),
            "avg_pass_accuracy": round(self.avg_pass_accuracy, 3),
            "avg_rating": round(self.avg_rating, 1),
        }


@dataclass
class HeadToHeadComparison:
    """Side-by-side comparison of two teams/players."""

    team_a_name: str = ""
    team_b_name: str = ""
    possession_a: float = 50.0
    possession_b: float = 50.0
    passes_a: int = 0
    passes_b: int = 0
    pass_accuracy_a: float = 0.0
    pass_accuracy_b: float = 0.0
    shots_a: int = 0
    shots_b: int = 0
    shots_on_target_a: int = 0
    shots_on_target_b: int = 0
    xg_a: float = 0.0
    xg_b: float = 0.0
    tackles_a: int = 0
    tackles_b: int = 0
    distance_a_km: float = 0.0
    distance_b_km: float = 0.0
    progressive_passes_a: int = 0
    progressive_passes_b: int = 0
    pitch_control_a: float = 50.0
    pitch_control_b: float = 50.0
    avg_rating_a: float = 0.0
    avg_rating_b: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_a": self.team_a_name,
            "team_b": self.team_b_name,
            "possession": {"a": round(self.possession_a, 1), "b": round(self.possession_b, 1)},
            "passes": {"a": self.passes_a, "b": self.passes_b},
            "pass_accuracy": {
                "a": round(self.pass_accuracy_a, 3),
                "b": round(self.pass_accuracy_b, 3),
            },
            "shots": {"a": self.shots_a, "b": self.shots_b},
            "shots_on_target": {"a": self.shots_on_target_a, "b": self.shots_on_target_b},
            "xg": {"a": round(self.xg_a, 2), "b": round(self.xg_b, 2)},
            "tackles": {"a": self.tackles_a, "b": self.tackles_b},
            "distance_km": {"a": round(self.distance_a_km, 1), "b": round(self.distance_b_km, 1)},
            "progressive_passes": {"a": self.progressive_passes_a, "b": self.progressive_passes_b},
            "pitch_control": {
                "a": round(self.pitch_control_a, 1),
                "b": round(self.pitch_control_b, 1),
            },
            "avg_rating": {"a": round(self.avg_rating_a, 1), "b": round(self.avg_rating_b, 1)},
        }


class SeasonAggregator:
    """Aggregates match data across a season."""

    def aggregate_team_season(
        self,
        match_data: list[dict[str, Any]],
        team_name: str = "",
    ) -> SeasonReport:
        """Aggregate match data for a team across a season.

        Args:
            match_data: List of per-match dicts with "home_team" stats,
                       "away_team" stats, "players" lists, "events".
            team_name: Team to aggregate for.

        Returns:
            SeasonReport with per-player and team aggregate stats.
        """
        if not match_data:
            return SeasonReport(team_name=team_name)

        player_agg: dict[int, PlayerSeasonStats] = {}
        total_goals = 0.0
        total_shots = 0
        total_passes = 0
        total_xg = 0.0
        possession_sum = 0.0
        pass_acc_sum = 0.0
        rating_sum = 0.0
        match_count = 0

        for match in match_data:
            match_count += 1
            home_team = match.get("home_team", {})
            away_team = match.get("away_team", {})
            is_home = home_team.get("team_name", "") == team_name or not team_name

            team_stats = home_team if is_home else away_team
            possession_sum += team_stats.get("possession", 50.0)
            pass_acc = team_stats.get("pass_accuracy", 0.5)
            pass_acc_sum += pass_acc
            total_shots += team_stats.get("shots", 0)

            events = match.get("events", [])
            for ev in events:
                if ev.get("team") == ("home" if is_home else "away"):
                    if ev.get("type") == "shot":
                        total_shots += 1
                        if ev.get("is_goal"):
                            total_goals += 1
                        total_xg += ev.get("xg", 0.0)
                    if ev.get("type") == "pass":
                        total_passes += 1

            players_data = match.get("players", {})
            if isinstance(players_data, dict):
                for tid_str, pdata in players_data.items():
                    tid = int(tid_str)
                    if tid not in player_agg:
                        player_agg[tid] = PlayerSeasonStats(
                            name=pdata.get("name", f"Player #{tid}"),
                            team=team_name or pdata.get("team", ""),
                            position=pdata.get("position", ""),
                        )
                    p = player_agg[tid]
                    p.matches_played += 1
                    p.total_minutes += match.get("duration", 5400) / 60.0
                    p.shots += pdata.get("shots", 0)
                    p.passes_attempted += pdata.get("passes_attempted", 0)
                    p.passes_completed += pdata.get("passes_completed", 0)
                    p.tackles += pdata.get("tackles", 0)
                    p.total_distance_km += pdata.get("distance_covered_m", 0) / 1000.0
                    p.total_xg += pdata.get("xg", 0) if "xg" in pdata else 0

                    if isinstance(pdata.get("rating"), dict):
                        rating = pdata["rating"].get("overall", 0)
                    else:
                        rating = pdata.get("avg_rating", 0)

                    if rating:
                        p.avg_rating = (
                            p.avg_rating * (p.matches_played - 1) + rating
                        ) / p.matches_played

        n = max(match_count, 1)
        avg_rating = rating_sum / n if rating_sum else 0.0

        return SeasonReport(
            team_name=team_name or "Team",
            matches=match_count,
            players=player_agg,
            total_goals=total_goals,
            total_shots=total_shots,
            total_passes=total_passes,
            total_xg=total_xg,
            avg_possession=possession_sum / n,
            avg_pass_accuracy=pass_acc_sum / n,
            avg_rating=avg_rating,
        )

    def compare_teams(
        self,
        match_data: list[dict[str, Any]],
        team_a_name: str,
        team_b_name: str,
    ) -> HeadToHeadComparison:
        """Compare two teams across shared matches."""
        if not match_data:
            return HeadToHeadComparison(team_a_name=team_a_name, team_b_name=team_b_name)

        stats_a = self.aggregate_team_season(
            [
                m
                for m in match_data
                if m.get("home_team", {}).get("team_name") == team_a_name
                or m.get("away_team", {}).get("team_name") == team_a_name
            ],
            team_a_name,
        )
        stats_b = self.aggregate_team_season(
            [
                m
                for m in match_data
                if m.get("home_team", {}).get("team_name") == team_b_name
                or m.get("away_team", {}).get("team_name") == team_b_name
            ],
            team_b_name,
        )

        def _team_stat(season: SeasonReport, attr: str) -> Any:
            return getattr(season, attr, 0)

        return HeadToHeadComparison(
            team_a_name=team_a_name,
            team_b_name=team_b_name,
            possession_a=stats_a.avg_possession,
            possession_b=stats_b.avg_possession,
            passes_a=stats_a.total_passes,
            passes_b=stats_b.total_passes,
            pass_accuracy_a=stats_a.avg_pass_accuracy,
            pass_accuracy_b=stats_b.avg_pass_accuracy,
            shots_a=stats_a.total_shots,
            shots_b=stats_b.total_shots,
            xg_a=stats_a.total_xg,
            xg_b=stats_b.total_xg,
        )
