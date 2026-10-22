"""Advanced goalkeeper analytics — PSxG integration, positioning, aerial command, distribution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from kawkab.core.game_constants import GAME
from kawkab.core.logging import get_logger
from kawkab.core.psxg_model import compute_psxg

logger = get_logger(__name__)


@dataclass
class GKPositioningMetrics:
    avg_position_x: float = 0.0
    avg_position_y: float = 0.0
    max_sweep_distance: float = 0.0
    sweep_actions: int = 0
    avg_set_piece_position_x: float = 0.0
    avg_set_piece_position_y: float = 0.0
    set_piece_positions: list[dict] = field(default_factory=list)
    position_samples: int = 0


@dataclass
class GKAerialCommand:
    crosses_claimed: int = 0
    crosses_punched: int = 0
    crosses_faced: int = 0
    aerial_success_rate: float = 0.0
    claimed_under_pressure: int = 0
    punches_effective: int = 0


@dataclass
class GKDistribution:
    short_attempts: int = 0
    short_successful: int = 0
    short_accuracy: float = 0.0
    long_attempts: int = 0
    long_successful: int = 0
    long_accuracy: float = 0.0
    medium_attempts: int = 0
    medium_successful: int = 0
    medium_accuracy: float = 0.0
    left_zone: int = 0
    center_zone: int = 0
    right_zone: int = 0
    progressive_passes: int = 0
    avg_distance: float = 0.0


@dataclass
class GKSaveQuality:
    total_shots_faced: int = 0
    saves: int = 0
    goals_conceded: int = 0
    save_rate: float = 0.0
    total_psxg: float = 0.0
    total_conceded_psxg: float = 0.0
    goals_prevented: float = 0.0
    avg_psxg_per_shot: float = 0.0
    high_quality_saves: int = 0
    one_on_one_saves: int = 0
    one_on_one_goals: int = 0
    one_on_one_save_rate: float = 0.0


@dataclass
class AdvancedGKReport:
    team: str
    player_name: str = ""
    save_quality: GKSaveQuality = field(default_factory=GKSaveQuality)
    positioning: GKPositioningMetrics = field(default_factory=GKPositioningMetrics)
    aerial: GKAerialCommand = field(default_factory=GKAerialCommand)
    distribution: GKDistribution = field(default_factory=GKDistribution)
    clean_sheet: bool = False
    rating: float = 0.0
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "player_name": self.player_name,
            "save_quality": {
                "total_shots_faced": self.save_quality.total_shots_faced,
                "saves": self.save_quality.saves,
                "goals_conceded": self.save_quality.goals_conceded,
                "save_rate": round(self.save_quality.save_rate, 3),
                "total_psxg": round(self.save_quality.total_psxg, 3),
                "goals_prevented": round(self.save_quality.goals_prevented, 3),
                "avg_psxg_per_shot": round(self.save_quality.avg_psxg_per_shot, 3),
                "high_quality_saves": self.save_quality.high_quality_saves,
                "one_on_one_saves": self.save_quality.one_on_one_saves,
                "one_on_one_goals": self.save_quality.one_on_one_goals,
                "one_on_one_save_rate": round(self.save_quality.one_on_one_save_rate, 3),
            },
            "positioning": {
                "avg_position_x": round(self.positioning.avg_position_x, 1),
                "avg_position_y": round(self.positioning.avg_position_y, 1),
                "max_sweep_distance": round(self.positioning.max_sweep_distance, 1),
                "sweep_actions": self.positioning.sweep_actions,
                "avg_set_piece_position_x": round(self.positioning.avg_set_piece_position_x, 1),
                "avg_set_piece_position_y": round(self.positioning.avg_set_piece_position_y, 1),
                "position_samples": self.positioning.position_samples,
            },
            "aerial": {
                "crosses_claimed": self.aerial.crosses_claimed,
                "crosses_punched": self.aerial.crosses_punched,
                "crosses_faced": self.aerial.crosses_faced,
                "aerial_success_rate": round(self.aerial.aerial_success_rate, 3),
                "claimed_under_pressure": self.aerial.claimed_under_pressure,
                "punches_effective": self.aerial.punches_effective,
            },
            "distribution": {
                "short_attempts": self.distribution.short_attempts,
                "short_successful": self.distribution.short_successful,
                "short_accuracy": round(self.distribution.short_accuracy, 3),
                "long_attempts": self.distribution.long_attempts,
                "long_successful": self.distribution.long_successful,
                "long_accuracy": round(self.distribution.long_accuracy, 3),
                "medium_attempts": self.distribution.medium_attempts,
                "medium_successful": self.distribution.medium_successful,
                "medium_accuracy": round(self.distribution.medium_accuracy, 3),
                "left_zone": self.distribution.left_zone,
                "center_zone": self.distribution.center_zone,
                "right_zone": self.distribution.right_zone,
                "progressive_passes": self.distribution.progressive_passes,
                "avg_distance": round(self.distribution.avg_distance, 1),
            },
            "clean_sheet": self.clean_sheet,
            "rating": round(self.rating, 1),
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
        }


class GoalkeeperAnalytics:
    """Advanced goalkeeper analysis integrating PSxG, positioning, aerial command, distribution."""

    PITCH_LENGTH = GAME.PITCH_LENGTH_M
    PITCH_WIDTH = GAME.PITCH_WIDTH_M

    def __init__(self):
        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def compute_save_quality(
        self,
        shots_faced: list[dict[str, Any]],
    ) -> GKSaveQuality:
        """Evaluate shot-stopping using PSxG model."""
        sq = GKSaveQuality()
        sq.total_shots_faced = len(shots_faced)
        if not shots_faced:
            return sq

        total_psxg = 0.0
        conceded_psxg = 0.0
        high_quality_count = 0
        o2o_saves = 0
        o2o_goals = 0
        goal_x = self.PITCH_LENGTH

        for s in shots_faced:
            placement_x = float(s.get("placement_x", 0.5))
            placement_y = float(s.get("placement_y", 0.5))
            shot_x = float(s.get("shot_x", 0.0))
            shot_y = float(s.get("shot_y", 34.0))
            speed = float(s.get("speed_mps", 25.0))
            is_header = bool(s.get("is_header", False))
            outcome = s.get("outcome", "save")
            is_o2o = bool(s.get("one_on_one", False))
            bp = "head" if is_header else "right_foot"

            distance_m = math.hypot(goal_x - shot_x, shot_y - self.PITCH_WIDTH / 2)
            angle_deg = 0.0
            if distance_m > 0:
                angle_rad = math.atan2(7.32, distance_m)
                angle_deg = math.degrees(angle_rad)

            result = compute_psxg(
                distance_m=distance_m,
                angle_deg=angle_deg,
                placement_x=placement_x,
                placement_y=placement_y,
                shot_speed=speed,
                body_part=bp,
            )
            psxg = result.psxg
            total_psxg += psxg

            if outcome == "goal":
                sq.goals_conceded += 1
                conceded_psxg += psxg
                if is_o2o:
                    o2o_goals += 1
            elif outcome in ("save", "goal_kick"):
                sq.saves += 1
                if psxg > 0.4:
                    high_quality_count += 1
                if is_o2o:
                    o2o_saves += 1

        sq.total_psxg = total_psxg
        sq.total_conceded_psxg = conceded_psxg
        sq.goals_prevented = total_psxg - conceded_psxg
        sq.avg_psxg_per_shot = total_psxg / max(1, sq.total_shots_faced)
        sq.high_quality_saves = high_quality_count
        sq.save_rate = sq.saves / max(1, sq.total_shots_faced)
        sq.one_on_one_saves = o2o_saves
        sq.one_on_one_goals = o2o_goals
        sq.one_on_one_save_rate = o2o_saves / max(1, o2o_saves + o2o_goals)

        return sq

    def compute_positioning(
        self,
        tracking_positions: list[dict[str, Any]] | None = None,
        sweeps: list[dict] | None = None,
        set_piece_positions: list[dict] | None = None,
    ) -> GKPositioningMetrics:
        """Analyze GK positioning from tracking data."""
        pm = GKPositioningMetrics()

        if tracking_positions:
            xs = [p.get("x", 0) for p in tracking_positions if p.get("x") is not None]
            ys = [p.get("y", 0) for p in tracking_positions if p.get("y") is not None]
            if xs:
                pm.avg_position_x = sum(xs) / len(xs)
                pm.position_samples = len(xs)
            if ys:
                pm.avg_position_y = sum(ys) / len(ys)

        if sweeps:
            pm.sweep_actions = len(sweeps)
            distances = []
            for sw in sweeps:
                sx = sw.get("x", 0) or 0
                sy = sw.get("y", 0) or 0
                gk_x = sw.get("gk_x", pm.avg_position_x or 0)
                gk_y = sw.get("gk_y", pm.avg_position_y or 0)
                distances.append(math.hypot(sx - gk_x, sy - gk_y))
            if distances:
                pm.max_sweep_distance = max(distances)

        if set_piece_positions:
            sp_xs = [p.get("x", 0) for p in set_piece_positions if p.get("x") is not None]
            sp_ys = [p.get("y", 0) for p in set_piece_positions if p.get("y") is not None]
            if sp_xs:
                pm.avg_set_piece_position_x = sum(sp_xs) / len(sp_xs)
                pm.avg_set_piece_position_y = sum(sp_ys) / len(sp_ys) if sp_ys else 0
            pm.set_piece_positions = set_piece_positions[:50]

        return pm

    def compute_aerial_command(
        self,
        cross_actions: list[dict[str, Any]],
    ) -> GKAerialCommand:
        """Analyze GK aerial command from cross-related actions."""
        ac = GKAerialCommand()
        if not cross_actions:
            return ac

        for ca in cross_actions:
            action = ca.get("action_type", "")
            _ = ca.get("outcome", "")
            pressure = bool(ca.get("under_pressure", False))
            effective = bool(ca.get("effective", True))
            ac.crosses_faced += 1
            if action == "claim":
                ac.crosses_claimed += 1
                if pressure:
                    ac.claimed_under_pressure += 1
            elif action == "punch":
                ac.crosses_punched += 1
                if effective:
                    ac.punches_effective += 1

        total_claimed_or_punched = ac.crosses_claimed + ac.crosses_punched
        ac.aerial_success_rate = total_claimed_or_punched / max(1, ac.crosses_faced)
        return ac

    def compute_distribution(
        self,
        distribution_actions: list[dict[str, Any]],
    ) -> GKDistribution:
        """Analyze GK distribution by type and zone."""
        gd = GKDistribution()
        if not distribution_actions:
            return gd

        distances: list[float] = []
        for da in distribution_actions:
            action = da.get("action_type", "")
            outcome = da.get("outcome", "")
            dist = float(da.get("distance_m", 0) or 0)
            dest_x = float(da.get("dest_x", 0) or 0)
            is_progressive = bool(da.get("progressive", False))

            if dist > 0:
                distances.append(dist)

            if dest_x <= 0.33 * self.PITCH_LENGTH:
                gd.left_zone += 1
            elif dest_x >= 0.67 * self.PITCH_LENGTH:
                gd.right_zone += 1
            else:
                gd.center_zone += 1

            is_success = outcome == "complete"

            if action == "short_dist":
                gd.short_attempts += 1
                if is_success:
                    gd.short_successful += 1
                if is_progressive:
                    gd.progressive_passes += 1
            elif action == "long_dist":
                gd.long_attempts += 1
                if is_success:
                    gd.long_successful += 1
                if is_progressive:
                    gd.progressive_passes += 1
            else:
                gd.medium_attempts += 1
                if is_success:
                    gd.medium_successful += 1

        gd.short_accuracy = gd.short_successful / max(1, gd.short_attempts)
        gd.long_accuracy = gd.long_successful / max(1, gd.long_attempts)
        gd.medium_accuracy = gd.medium_successful / max(1, gd.medium_attempts)
        if distances:
            gd.avg_distance = sum(distances) / len(distances)

        return gd

    def compute_match_report(
        self,
        team: str,
        shots_faced: list[dict[str, Any]] | None = None,
        cross_actions: list[dict[str, Any]] | None = None,
        distribution_actions: list[dict[str, Any]] | None = None,
        tracking_positions: list[dict[str, Any]] | None = None,
        sweeps: list[dict] | None = None,
        set_piece_positions: list[dict] | None = None,
        clean_sheet: bool = False,
        player_name: str = "",
    ) -> AdvancedGKReport:
        report = AdvancedGKReport(team=team, player_name=player_name, clean_sheet=clean_sheet)

        if shots_faced:
            report.save_quality = self.compute_save_quality(shots_faced)

        report.positioning = self.compute_positioning(
            tracking_positions=tracking_positions,
            sweeps=sweeps,
            set_piece_positions=set_piece_positions,
        )

        if cross_actions:
            report.aerial = self.compute_aerial_command(cross_actions)

        if distribution_actions:
            report.distribution = self.compute_distribution(distribution_actions)

        report.strengths, report.weaknesses = self._assess(report)
        report.rating = self._compute_rating(report)

        return report

    def _compute_rating(self, report: AdvancedGKReport) -> float:
        """Compute 0-100 goalkeeper rating."""
        rating = 50.0
        sq = report.save_quality
        if sq.save_rate > 0 and sq.total_shots_faced > 0:
            rating += (sq.save_rate - 0.5) * 40
            rating += min(20, sq.goals_prevented * 10)
            rating += min(10, sq.high_quality_saves * 3)
        if report.clean_sheet:
            rating += 10
        if report.aerial.aerial_success_rate > 0.7:
            rating += 5
        if report.distribution.short_accuracy > 0.8:
            rating += 5
        if report.positioning.sweep_actions > 5:
            rating += 5
        return max(0.0, min(100.0, rating))

    def _assess(
        self,
        report: AdvancedGKReport,
    ) -> tuple[list[str], list[str]]:
        strengths: list[str] = []
        weaknesses: list[str] = []
        sq = report.save_quality

        if sq.save_rate >= 0.75 and sq.total_shots_faced >= 3:
            strengths.append("Elite shot-stopper")
        elif sq.save_rate >= 0.6:
            strengths.append("Reliable shot-stopper")
        if sq.goals_prevented > 1.0:
            strengths.append(f"Saved {sq.goals_prevented:.1f} goals above expected")
        if sq.high_quality_saves >= 2:
            strengths.append("Makes difficult saves")
        if report.aerial.aerial_success_rate >= 0.8:
            strengths.append("Commanding in the air")
        if report.distribution.short_accuracy >= 0.85:
            strengths.append("Excellent short distribution")
        if report.distribution.long_accuracy >= 0.5:
            strengths.append("Effective long distribution")
        if report.positioning.sweep_actions > 3:
            strengths.append("Active sweeper-keeper")

        if sq.save_rate < 0.5 and sq.total_shots_faced >= 3:
            weaknesses.append("Below-average save rate")
        if sq.goals_prevented < -1.0:
            weaknesses.append(f"Conceded {abs(sq.goals_prevented):.1f} more goals than expected")
        if report.aerial.aerial_success_rate < 0.5 and report.aerial.crosses_faced > 0:
            weaknesses.append("Struggles with aerial command")
        if report.distribution.short_accuracy < 0.6 and report.distribution.short_attempts > 0:
            weaknesses.append("Poor short distribution accuracy")
        if report.positioning.sweep_actions == 0 and report.clean_sheet is False:
            weaknesses.append("Limited sweeping range")
        if sq.one_on_one_save_rate < 0.3 and (sq.one_on_one_saves + sq.one_on_one_goals) > 0:
            weaknesses.append("Poor one-on-one record")

        if not strengths:
            strengths.append("Insufficient data to assess strengths")
        if not weaknesses:
            weaknesses.append("No significant weaknesses identified")
        return strengths, weaknesses
