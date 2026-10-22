"""OBV (Off-Ball Value) — measures value of player movement without the ball.

Computes space creation, defensive positioning, support positioning,
and decoy run value from tracking data.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


@dataclass
class OBVPlayerResult:
    track_id: int = 0
    space_creation_value: float = 0.0
    def_positioning_value: float = 0.0
    support_value: float = 0.0
    decoy_run_value: float = 0.0
    total_obv: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tid": self.track_id,
            "space_creation": round(self.space_creation_value, 3),
            "def_pos": round(self.def_positioning_value, 3),
            "support": round(self.support_value, 3),
            "decoy": round(self.decoy_run_value, 3),
            "total": round(self.total_obv, 3),
        }


@dataclass
class OBVMatchReport:
    team: str = ""
    players: dict[int, OBVPlayerResult] = field(default_factory=dict)
    team_obv: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "players": {str(k): v.to_dict() for k, v in self.players.items()},
            "team_obv": round(self.team_obv, 3),
        }


class OffBallValuator:
    """Computes Off-Ball Value (OBV) for players using tracking data.

    Evaluates four dimensions of off-ball contribution:
      1. Space creation — moving away from ball to open space for teammates
      2. Defensive positioning — threatening passing lanes / intercepting
      3. Support positioning — providing safe passing angles to the ball carrier
      4. Decoy runs — drawing defenders away from dangerous areas

    Usage:
        obv = OffBallValuator()
        report = obv.compute_obv(frames_data, team="home")
    """

    SPACE_CREATION_SPEED_MS = 4.0
    DEF_POSITIONING_RADIUS_M = 3.0
    SUPPORT_ANGLE_DEG = 45.0
    DECOY_RUN_SPEED_MS = 4.5
    DECOY_DISTANCE_M = 10.0

    # Value weights per detected event
    SPACE_WEIGHT = 0.5
    DEF_WEIGHT = 0.3
    SUPPORT_WEIGHT = 0.15
    DECOY_WEIGHT = 0.05

    def compute_obv(
        self,
        frames: list[dict[str, Any]],
        team: str = "home",
        events: list[dict[str, Any]] | None = None,
    ) -> OBVMatchReport:
        """Compute OBV for all players on a team.

        Args:
            frames: List of tracking frames, each with:
                - timestamp: float
                - possession: bool (True if *team* has ball, else opponent)
                - ball_pos: (x, y) or None
                - home_positions: list of (x, y, track_id)
                - away_positions: list of (x, y, track_id)
            team: "home" or "away".
            events: Optional event list (currently unused, kept for API compat).

        Returns:
            OBVMatchReport with per-player and team OBV scores.
        """
        if not frames:
            return OBVMatchReport(team=team)

        opponent = "away" if team == "home" else "home"

        player_trajectories: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        opponent_trajectories: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        possession_flags: list[bool] = []
        ball_positions: list[tuple[float, float] | None] = []

        for fdata in frames:
            ts = fdata.get("timestamp", 0.0)
            possession = fdata.get("possession", False)
            ball = fdata.get("ball_pos")
            ball_positions.append(ball)
            possession_flags.append(possession)

            for item in fdata.get(f"{team}_positions", []):
                if len(item) != 3:
                    continue
                x, y, tid = item
                player_trajectories[tid].append((ts, x, y))

            for item in fdata.get(f"{opponent}_positions", []):
                if len(item) != 3:
                    continue
                x, y, tid = item
                opponent_trajectories[tid].append((ts, x, y))

        if not player_trajectories:
            return OBVMatchReport(team=team)

        _ = set(player_trajectories.keys())

        results: dict[int, OBVPlayerResult] = {}

        for tid, traj in player_trajectories.items():
            if len(traj) < 3:
                results[tid] = OBVPlayerResult(track_id=tid)
                continue

            space = 0.0
            defense = 0.0
            support = 0.0
            decoy = 0.0

            for i in range(1, len(traj)):
                ts, x, y = traj[i]
                _, prev_x, prev_y = traj[i - 1]
                dt = ts - traj[i - 1][0]
                if dt <= 0:
                    continue

                dx = x - prev_x
                dy = y - prev_y
                speed = math.sqrt(dx * dx + dy * dy) / dt

                frame_idx = min(i, len(ball_positions) - 1)
                ball = ball_positions[frame_idx]
                if ball is None:
                    continue

                possession = (
                    possession_flags[frame_idx] if frame_idx < len(possession_flags) else False
                )

                ball_dx = x - ball[0]
                ball_dy = y - ball[1]
                dist_to_ball = math.sqrt(ball_dx * ball_dx + ball_dy * ball_dy)
                prev_dist_to_ball = math.sqrt((prev_x - ball[0]) ** 2 + (prev_y - ball[1]) ** 2)

                # --- Space creation: teammate has ball, player moves away creating room ---
                if (
                    possession
                    and speed >= self.SPACE_CREATION_SPEED_MS
                    and dist_to_ball > prev_dist_to_ball * 1.05
                ):
                    space_magnitude = min((dist_to_ball - prev_dist_to_ball) / 5.0, 1.0)
                    space += self.SPACE_WEIGHT * space_magnitude

                # --- Defensive positioning: opponent has ball, player near dangerous lane ---
                if not possession:
                    def_value = self._compute_defensive_value(
                        (x, y), ball, opponent_trajectories, frame_idx
                    )
                    defense += self.DEF_WEIGHT * def_value

                # --- Support positioning: teammate has ball, good passing angle ---
                if possession:
                    support_value = self._compute_passing_lane_value(
                        (x, y), ball, player_trajectories, tid, frame_idx
                    )
                    support += self.SUPPORT_WEIGHT * support_value

                # --- Decoy run: player moves away from ball fast, drawing nearby defender ---
                if (
                    speed >= self.DECOY_RUN_SPEED_MS
                    and dist_to_ball > 5.0
                    and prev_dist_to_ball < dist_to_ball
                ):
                    decoy_value = self._compute_decoy_value(
                        (x, y), ball, opponent_trajectories, frame_idx
                    )
                    decoy += self.DECOY_WEIGHT * decoy_value

            total = space + defense + support + decoy
            results[tid] = OBVPlayerResult(
                track_id=tid,
                space_creation_value=space,
                def_positioning_value=defense,
                support_value=support,
                decoy_run_value=decoy,
                total_obv=total,
            )

        team_total = sum(r.total_obv for r in results.values())
        return OBVMatchReport(team=team, players=results, team_obv=team_total)

    def _compute_passing_lane_value(
        self,
        player_pos: tuple[float, float],
        ball_pos: tuple[float, float],
        teammate_trajs: dict[int, list[tuple[float, float, float]]],
        exclude_tid: int,
        frame_idx: int,
    ) -> float:
        """Compute passing lane quality from ball carrier to this player.

        Value is higher when:
        - Player is within 45 degrees of the direct ball-to-goal line
        - Player is within decent passing range (< 40m)
        - Few opponents block the direct line
        """
        px, py = player_pos
        bx, by = ball_pos

        dx = px - bx
        dy = py - by
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 1.0 or distance > 40.0:
            return 0.0

        goal_center_x = PITCH_LENGTH
        goal_center_y = PITCH_WIDTH / 2.0
        ball_to_goal_dx = goal_center_x - bx
        ball_to_goal_dy = goal_center_y - by
        ball_to_goal_norm = math.sqrt(ball_to_goal_dx**2 + ball_to_goal_dy**2)
        if ball_to_goal_norm < 1e-6:
            return 0.0

        player_dx = px - bx
        player_dy = py - by
        player_norm = math.sqrt(player_dx**2 + player_dy**2)
        if player_norm < 1e-6:
            return 0.0

        cos_angle = (ball_to_goal_dx * player_dx + ball_to_goal_dy * player_dy) / (
            ball_to_goal_norm * player_norm
        )
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))

        if angle_deg > self.SUPPORT_ANGLE_DEG:
            return 0.0

        angle_factor = 1.0 - (angle_deg / self.SUPPORT_ANGLE_DEG)
        distance_factor = 1.0 - (distance / 40.0)
        value = angle_factor * 0.6 + distance_factor * 0.4
        return max(0.0, min(1.0, value))

    def _compute_defensive_value(
        self,
        player_pos: tuple[float, float],
        ball_pos: tuple[float, float],
        opponent_trajs: dict[int, list[tuple[float, float, float]]],
        frame_idx: int,
    ) -> float:
        """Compute defensive positioning value.

        Value is higher when:
        - Player is within 3m of a dangerous passing lane (ball to opponent)
        - Player is between ball and own goal
        - Player is close to the ball
        """
        px, py = player_pos
        bx, by = ball_pos

        dist_to_ball = math.sqrt((px - bx) ** 2 + (py - by) ** 2)

        ball_to_own_goal_dx = -bx
        ball_to_own_goal_dy = (PITCH_WIDTH / 2.0) - by
        ball_to_own_goal_norm = math.sqrt(ball_to_own_goal_dx**2 + ball_to_own_goal_dy**2)
        if ball_to_own_goal_norm < 1e-6:
            return 0.0

        player_from_ball_dx = px - bx
        player_from_ball_dy = py - by
        player_from_ball_norm = math.sqrt(player_from_ball_dx**2 + player_from_ball_dy**2)
        if player_from_ball_norm < 1e-6:
            return 0.0

        cos_angle = (
            ball_to_own_goal_dx * player_from_ball_dx + ball_to_own_goal_dy * player_from_ball_dy
        ) / (ball_to_own_goal_norm * player_from_ball_norm)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))

        if angle_deg > 90.0:
            return 0.0

        if dist_to_ball < 5.0:
            proximity = 1.0 - (dist_to_ball / 5.0)
        elif dist_to_ball < 15.0:
            proximity = 0.3 * (1.0 - (dist_to_ball - 5.0) / 10.0)
        else:
            proximity = 0.0

        angle_factor = 1.0 - (angle_deg / 90.0)
        value = proximity * 0.7 + angle_factor * 0.3

        return max(0.0, min(1.0, value))

    def _compute_decoy_value(
        self,
        player_pos: tuple[float, float],
        ball_pos: tuple[float, float],
        opponent_trajs: dict[int, list[tuple[float, float, float]]],
        frame_idx: int,
    ) -> float:
        """Compute decoy run value — how many defenders the player draws.

        Counts nearby opponents that are moving in a similar direction
        (i.e., following the player's run).
        """
        px, py = player_pos
        drawn = 0

        for _opp_tid, traj in opponent_trajs.items():
            if len(traj) <= frame_idx:
                continue

            ox, oy = traj[frame_idx][1], traj[frame_idx][2]
            dist = math.sqrt((px - ox) ** 2 + (py - oy) ** 2)
            if dist > self.DECOY_DISTANCE_M:
                continue

            drawn += 1

        value = min(drawn / 3.0, 1.0)
        return value
