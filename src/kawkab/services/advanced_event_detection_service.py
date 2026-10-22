"""Advanced event detection service - extends basic pass/shot detection.

Derives 15+ event types from tracking data without requiring additional CV models:
- Dribbles: ball carrier moves with ball for N+ frames
- Tackles: defender wins ball from attacker with close proximity
- Interceptions: defender cuts off pass without attacker touching ball
- Clearances: ball kicked from defensive third to safety with high speed
- Crosses: pass from wide area into penalty area
- Ball recoveries: team wins possession back
- Blocks: defender stops shot/pass near own goal
- Duels: two players contest ball within 2m
- Carries: ball moves with player (no pass, no dribble)
- Progressive actions: ball advances >10m toward opponent goal
- Passes into final third / penalty area
- High turnovers: ball lost in final 40m
- Defensive actions: tackle + interception + clearance + block
- Set pieces: ball near corner flag, free kick positions

Uses the existing FrameDetections data from CVService.
"""

from __future__ import annotations

import math
from collections import defaultdict

from kawkab.core.logging import get_logger
from kawkab.services.cv_service import MatchTrackData

logger = get_logger(__name__)


class AdvancedEventDetectionService:
    """Detects advanced event types from tracking data."""

    def __init__(self) -> None:
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        logger.info("AdvancedEventDetectionService initialized")

    async def detect_all_advanced_events(
        self,
        track_data: MatchTrackData,
        base_events: list[dict],
        homography_matrix=None,
    ) -> list[dict]:
        """Detect all advanced event types and merge with base events.

        Args:
            track_data: Match tracking data
            base_events: Events already detected (passes, shots)
            homography_matrix: Optional pitch calibration

        Returns:
            Combined list of all events with enriched metadata
        """
        logger.info("Running advanced event detection...")

        all_events = list(base_events)  # copy

        # Detect each event type
        dribbles = self._detect_dribbles(track_data, homography_matrix)
        tackles = self._detect_tackles(track_data, base_events, homography_matrix)
        interceptions = self._detect_interceptions(track_data, base_events, homography_matrix)
        clearances = self._detect_clearances(track_data, homography_matrix)
        crosses = self._detect_crosses(track_data, base_events, homography_matrix)
        recoveries = self._detect_ball_recoveries(track_data, base_events, homography_matrix)
        blocks = self._detect_blocks(track_data, base_events, homography_matrix)
        duels = self._detect_duels(track_data, homography_matrix)
        carries = self._detect_carries(track_data, base_events, homography_matrix)
        progressive = self._detect_progressive_actions(track_data, base_events, homography_matrix)
        final_third_entries = self._detect_final_third_entries(
            track_data, base_events, homography_matrix
        )
        high_turnovers = self._detect_high_turnovers(track_data, base_events, homography_matrix)
        goals = self._detect_goals(track_data, base_events, homography_matrix)
        corners = self._detect_corners(track_data, homography_matrix)
        free_kicks = self._detect_free_kicks(track_data, homography_matrix)
        throw_ins = self._detect_throw_ins(track_data, homography_matrix)

        all_events.extend(dribbles)
        all_events.extend(tackles)
        all_events.extend(interceptions)
        all_events.extend(clearances)
        all_events.extend(crosses)
        all_events.extend(recoveries)
        all_events.extend(blocks)
        all_events.extend(duels)
        all_events.extend(carries)
        all_events.extend(progressive)
        all_events.extend(final_third_entries)
        all_events.extend(high_turnovers)
        all_events.extend(goals)
        all_events.extend(corners)
        all_events.extend(free_kicks)
        all_events.extend(throw_ins)

        # Sort by timestamp
        all_events.sort(key=lambda e: e.get("timestamp", 0))

        # Count summary
        counts: defaultdict[str, int] = defaultdict(int)
        for e in all_events:
            counts[e.get("type", "unknown")] += 1
        logger.info(f"Advanced events: {dict(counts)}")

        return all_events

    def _get_player_team(self, track_data: MatchTrackData, track_id: int) -> str:
        """Get team assignment for a track ID."""
        return (
            track_data.player_teams.get(track_id, "unknown")
            if track_data.player_teams
            else "unknown"
        )

    # ------------------------------------------------------------------
    # Normalized event accessors.
    #
    # Production feeds this service the typed-event to_dict() shape
    # (core/events.py), where a pass looks like:
    #   {"type": "pass", "timestamp": ..., "track_id": <passer>,
    #    "to_track_id": <receiver>, "completed": bool, "team": ...,
    #    "start_x": <pct 0-1 or meters>, ...}   # NO "from_track_id", NO "metadata"
    # The raw internal shape used in unit tests is:
    #   {"type": "pass", ..., "from_track_id": <passer>, "to_track_id": ...,
    #    "metadata": {"start_x_pct": ...}}
    # Six detectors silently produced zero events against the production
    # shape because they read only the raw keys. These accessors read
    # both shapes, so detectors work regardless of caller.
    # ------------------------------------------------------------------

    @staticmethod
    def _passer_id(event: dict):
        """Track id of the passer: from_track_id (raw shape) or track_id (typed shape)."""
        return event.get("from_track_id", event.get("track_id"))

    @staticmethod
    def _receiver_id(event: dict):
        return event.get("to_track_id")

    @staticmethod
    def _event_start_pos(event: dict) -> tuple[float | None, float | None]:
        """Start position (x, y) of a pass/dribble/carry event.

        Accepts both the typed to_dict() shape (top-level start_x/start_y,
        which for passes are pitch-fraction 0-1 values, or meters for
        dribbles/carries) and the raw shape (metadata.start_x/...).
        Returns (None, None) when absent so callers can skip honestly
        instead of treating 0.0 (the left touchline) as a real position.
        """
        meta = event.get("metadata")
        if isinstance(meta, dict) and meta.get("start_x") is not None:
            return meta.get("start_x"), meta.get("start_y")
        sx, sy = event.get("start_x"), event.get("start_y")
        if sx is None:
            return None, None
        # Typed PassEvent.start_x is a 0-1 pitch fraction; typed
        # dribble/carry dicts carry meters. A value <= 1.0 for a
        # fraction-shaped field is ambiguous, but passes are the only
        # typed producers here and their to_dict() always carries the
        # fraction; scale it to meters for the meter-based thresholds
        # used by the detectors.
        if event.get("type") == "pass" and 0.0 <= float(sx) <= 1.0:
            return float(sx) * 105.0, (float(sy) if sy is not None else 0.5) * 68.0
        return sx, sy

    @staticmethod
    def _event_end_pos(event: dict) -> tuple[float | None, float | None]:
        """End position (x, y), same convention as _event_start_pos."""
        meta = event.get("metadata")
        if isinstance(meta, dict) and meta.get("end_x") is not None:
            return meta.get("end_x"), meta.get("end_y")
        ex, ey = event.get("end_x"), event.get("end_y")
        if ex is None:
            return None, None
        if event.get("type") == "pass" and 0.0 <= float(ex) <= 1.0:
            return float(ex) * 105.0, (float(ey) if ey is not None else 0.5) * 68.0
        return ex, ey

    def _detect_dribbles(self, track_data: MatchTrackData, homography_matrix=None) -> list[dict]:
        """Detect dribbles: ball stays with same player for 3+ frames while moving."""
        events: list[dict] = []
        dribble_min_frames = 3
        dribble_min_distance = 1.0  # meters

        possession_chain: list[
            tuple[float, int, float, float]
        ] = []  # [(timestamp, track_id, ball_x, ball_y)]

        for frame in track_data.frames:
            ball_det = None
            closest_player = None
            closest_dist = float("inf")

            for det in frame.detections:
                if det.class_name == "sports ball":
                    ball_det = det
                elif det.class_name == "person" and ball_det is not None:
                    bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2
                    by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2
                    px = (det.bbox[0] + det.bbox[2]) / 2
                    py = (det.bbox[1] + det.bbox[3]) / 2
                    d = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
                    if d < closest_dist:
                        closest_dist = d
                        closest_player = det

            if closest_player and closest_player.track_id is not None and closest_dist < 60:
                bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2 if ball_det else 0
                by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2 if ball_det else 0

                # Spatial units: with homography, positions are real pitch
                # meters. Without it, raw pixels were previously compared
                # against the 1.0m threshold directly -- pixel distances
                # trivially passed it, so any ball movement produced a
                # "dribble" whose "distance_m" was actually pixels. Use the
                # same pixel->meter approximation the carry detector uses
                # (game_constants.CARRY_PIXEL_TO_METER_RATIO) so the threshold
                # keeps its meaning, and flag the event as approximate.
                calibrated = homography_matrix is not None
                if calibrated:
                    try:
                        bx, by = homography_matrix.pixel_to_pitch(bx, by)
                    except Exception as e:
                        # dribble distance below is reported as "distance_m" and
                        # compared against a meters threshold -- silently keeping
                        # bx/by in raw pixel space here would make that number
                        # wrong by orders of magnitude while still looking like a
                        # real measurement. Reset the chain instead of mixing
                        # pixel- and pitch-space points within it.
                        logger.debug(f"pixel_to_pitch failed, resetting possession chain: {e}")
                        possession_chain = []
                        continue
                else:
                    from kawkab.core.game_constants import GAME

                    bx, by = (
                        bx * GAME.CARRY_PIXEL_TO_METER_RATIO,
                        by * GAME.CARRY_PIXEL_TO_METER_RATIO,
                    )

                possession_chain.append((frame.timestamp, closest_player.track_id, bx, by))
            else:
                # Ball loose or no player near it: the possession chain ends.
                possession_chain = []

            # Check if we have a dribble sequence
            if len(possession_chain) >= dribble_min_frames:
                # Check if same player held ball for all frames
                tids = {p[1] for p in possession_chain}
                if len(tids) == 1:
                    tid = list(tids)[0]
                    # Check distance moved
                    start_x = possession_chain[0][2]
                    start_y = possession_chain[0][3]
                    end_x = possession_chain[-1][2]
                    end_y = possession_chain[-1][3]
                    dist = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

                    if dist >= dribble_min_distance:
                        team = self._get_player_team(track_data, tid)
                        events.append(
                            {
                                "type": "dribble",
                                "timestamp": possession_chain[-1][0],
                                "track_id": tid,
                                "team": team,
                                "distance_m": round(dist, 1),
                                "duration_s": round(
                                    possession_chain[-1][0] - possession_chain[0][0], 2
                                ),
                                "confidence": 0.6,
                                "metadata": {
                                    "start_x": round(start_x, 1),
                                    "start_y": round(start_y, 1),
                                    "end_x": round(end_x, 1),
                                    "end_y": round(end_y, 1),
                                    "spatial_units": "meters" if calibrated else "pixel_approx",
                                },
                            }
                        )
                        possession_chain = []  # reset after detection

        return events

    def _detect_tackles(
        self, track_data: MatchTrackData, base_events: list[dict], homography_matrix=None
    ) -> list[dict]:
        """Detect tackles: ball changes from attacker to defender with close proximity."""
        events: list[dict] = []
        tackle_events = set()

        for i, event in enumerate(base_events):
            if event.get("type") != "pass":
                continue
            from_tid = self._passer_id(event)
            to_tid = self._receiver_id(event)
            if from_tid is None or to_tid is None:
                continue

            from_team = self._get_player_team(track_data, from_tid)
            to_team = self._get_player_team(track_data, to_tid)

            if from_team != to_team and not event.get("completed", True):
                # Pass was intercepted = tackle/interception
                # But only if the defender was close to the attacker
                tackle_events.add(i)
                events.append(
                    {
                        "type": "tackle",
                        "timestamp": event["timestamp"],
                        "from_track_id": from_tid,
                        "to_track_id": to_tid,
                        "team": to_team,
                        "confidence": min(1.0, event.get("confidence", 0.5) + 0.3),
                        "metadata": {
                            "derived_from_pass": True,
                            "tackler_team": to_team,
                        },
                    }
                )

        return events

    def _detect_interceptions(
        self, track_data: MatchTrackData, base_events: list[dict], homography_matrix=None
    ) -> list[dict]:
        """Detect interceptions: ball changes team without a pass event (pass was cut off)."""
        events: list[dict] = []
        pass_timestamps = {e["timestamp"] for e in base_events if e.get("type") == "pass"}

        prev_possession = None
        prev_team = None
        for frame in track_data.frames:
            ball_det = None
            closest_player = None
            closest_dist = float("inf")

            for det in frame.detections:
                if det.class_name == "sports ball":
                    ball_det = det
                elif det.class_name == "person" and ball_det is not None:
                    bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2
                    by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2
                    px = (det.bbox[0] + det.bbox[2]) / 2
                    py = (det.bbox[1] + det.bbox[3]) / 2
                    d = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
                    if d < closest_dist:
                        closest_dist = d
                        closest_player = det

            if closest_player and closest_player.track_id is not None and closest_dist < 60:
                tid = closest_player.track_id
                team = self._get_player_team(track_data, tid)

                if (
                    prev_possession is not None
                    and tid != prev_possession
                    and team != prev_team
                    and team != "unknown"
                    and prev_team != "unknown"
                    and frame.timestamp not in pass_timestamps
                ):
                    # Possession changed without a pass event
                    events.append(
                        {
                            "type": "interception",
                            "timestamp": frame.timestamp,
                            "from_track_id": prev_possession,
                            "to_track_id": tid,
                            "team": team,
                            "confidence": 0.5,
                            "metadata": {"no_pass_detected": True},
                        }
                    )

                prev_possession = tid
                prev_team = team
            else:
                prev_possession = None
                prev_team = None

        return events

    def _detect_clearances(self, track_data: MatchTrackData, homography_matrix=None) -> list[dict]:
        """Detect clearances: ball kicked from defensive third to safety with high speed."""
        events: list[dict] = []
        ball_history = []

        for frame in track_data.frames:
            ball_det = None
            for det in frame.detections:
                if det.class_name == "sports ball":
                    ball_det = det
                    break

            if ball_det is None:
                continue

            bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2
            by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2

            pitch_x, pitch_y = bx, by
            if homography_matrix is not None:
                try:
                    pitch_x, pitch_y = homography_matrix.pixel_to_pitch(bx, by)
                except Exception as e:
                    # speed below is reported as "speed_mps" and compared
                    # against the clearance detection threshold -- silently
                    # keeping pitch_x/pitch_y in raw pixel space here would
                    # make both wrong by orders of magnitude while still
                    # looking like real measurements. Drop this frame from
                    # the history instead of mixing pixel- and pitch-space
                    # points within it.
                    logger.debug(f"pixel_to_pitch failed, skipping ball position: {e}")
                    continue
            else:
                # No homography: raw pixel positions were previously fed
                # straight into the meter thresholds below ("defensive
                # third" = x < 26.25 became "left quarter of the frame",
                # speed > 8 "m/s" became 8 px/s -- essentially any leftward
                # ball movement qualified). Approximate meters with the
                # same pixel->meter ratio the carry/dribble paths use so
                # the thresholds keep their meaning.
                from kawkab.core.game_constants import GAME

                pitch_x = bx * GAME.CARRY_PIXEL_TO_METER_RATIO
                pitch_y = by * GAME.CARRY_PIXEL_TO_METER_RATIO

            ball_history.append((frame.timestamp, pitch_x, pitch_y))
            if len(ball_history) > 5:
                ball_history.pop(0)

            if len(ball_history) >= 3:
                p0 = ball_history[-3]
                p1 = ball_history[-1]
                dt = p1[0] - p0[0]
                if dt > 0.01:
                    dx = p1[1] - p0[1]
                    dy = p1[2] - p0[2]
                    speed = math.sqrt(dx * dx + dy * dy) / dt

                    # In defensive third, ball moving away from goal fast
                    defensive_threshold = self.pitch_length * 0.25
                    if pitch_x < defensive_threshold and dx > 5 and speed > 8:
                        events.append(
                            {
                                "type": "clearance",
                                "timestamp": frame.timestamp,
                                "team": "unknown",  # can't determine from ball alone
                                "confidence": min(1.0, speed / 20.0),
                                "metadata": {
                                    "speed_mps": round(speed, 1),
                                    "defensive_zone": True,
                                    "direction": "forward",
                                },
                            }
                        )

        return events

    def _detect_goals(
        self, track_data: MatchTrackData, base_events: list[dict], homography_matrix=None
    ) -> list[dict]:
        """Detect goals: shot events where ball crosses goal line and slows."""
        events: list[dict] = []
        _ = 300

        ball_positions: list[tuple[float, float, float, float, int]] = []
        for frame in track_data.frames:
            for det in frame.detections:
                if det.class_name == "sports ball":
                    bx = (det.bbox[0] + det.bbox[2]) / 2
                    by = (det.bbox[1] + det.bbox[3]) / 2
                    if homography_matrix is not None:
                        try:
                            px, py = homography_matrix.pixel_to_pitch(bx, by)
                        except Exception:
                            px, py = None, None
                    else:
                        px, py = None, None
                    ball_positions.append((frame.timestamp, bx, by, px or -1, py or -1))

        for shot in base_events:
            if shot.get("type") != "shot":
                continue
            if not shot.get("on_target", False):
                continue

            ts = shot["timestamp"]
            near_idx = -1
            for i, (bt, _, _, _, _) in enumerate(ball_positions):
                if abs(bt - ts) < 0.1:
                    near_idx = i
                    break
            if near_idx < 0:
                continue

            follow_positions = ball_positions[
                near_idx : near_idx + min(60, len(ball_positions) - near_idx)
            ]
            if len(follow_positions) < 3:
                continue

            crossed_line = False
            post_speed = 999.0
            for j in range(1, len(follow_positions)):
                pt = follow_positions[j]
                pitch_x = pt[3]
                if pitch_x < 0:
                    continue
                near_goal_line = pitch_x <= 2.0 or pitch_x >= self.pitch_length - 2.0
                if near_goal_line:
                    crossed_line = True
                    if j + 1 < len(follow_positions):
                        dx = follow_positions[j + 1][3] - pt[3]
                        dy = follow_positions[j + 1][4] - pt[4]
                        dt = max(follow_positions[j + 1][0] - pt[0], 0.01)
                        post_speed = math.sqrt(dx * dx + dy * dy) / dt
                    break

            if crossed_line and post_speed < 5.0:
                team = shot.get("team", "unknown")
                events.append(
                    {
                        "type": "goal",
                        "timestamp": ts,
                        "team": team,
                        "shot_confidence": shot.get("confidence", 0.5),
                        "confidence": 0.7,
                        "metadata": {
                            "distance_to_goal_m": shot.get("metadata", {}).get(
                                "distance_to_goal_m", 0
                            ),
                            "angle_to_goal_deg": shot.get("metadata", {}).get(
                                "angle_to_goal_deg", 0
                            ),
                        },
                    }
                )

        return events

    def _detect_corners(self, track_data: MatchTrackData, homography_matrix=None) -> list[dict]:
        """Detect corners: ball near corner arc after out of play."""
        events: list[dict] = []
        corner_radius = 5.0
        _ = 200

        ball_trail: list[tuple[float, float, float, float, float]] = []
        ball_lost_frames = 0
        for frame in track_data.frames:
            ball_det = None
            for det in frame.detections:
                if det.class_name == "sports ball":
                    ball_det = det
                    break

            if ball_det is None:
                ball_lost_frames += 1
                if ball_lost_frames > 5 and len(ball_trail) >= 3:
                    last = ball_trail[-1]
                    bx, by = last[1], last[2]
                    if homography_matrix is not None and last[3] >= 0:
                        pitch_x = last[3]
                        pitch_y = last[4]
                        is_near_left_goal = pitch_x < corner_radius
                        is_near_right_goal = pitch_x > self.pitch_length - corner_radius
                        is_near_top = pitch_y < corner_radius
                        is_near_bottom = pitch_y > self.pitch_width - corner_radius

                        if (is_near_left_goal or is_near_right_goal) and (
                            is_near_top or is_near_bottom
                        ):
                            is_left_side = is_near_left_goal
                            is_top_side = is_near_top
                            corner_team = "unknown"

                            corner_events = [e for e in events if e.get("type") == "corner"]
                            if (
                                len(corner_events) == 0
                                or (frame.timestamp - corner_events[-1]["timestamp"]) > 5
                            ):
                                events.append(
                                    {
                                        "type": "corner",
                                        "timestamp": frame.timestamp,
                                        "team": corner_team,
                                        "confidence": 0.4,
                                        "metadata": {
                                            "pitch_x": round(pitch_x, 1),
                                            "pitch_y": round(pitch_y, 1),
                                            "side": "left" if is_left_side else "right",
                                            "height": "top" if is_top_side else "bottom",
                                        },
                                    }
                                )
                    ball_trail = []
                continue
            else:
                ball_lost_frames = 0

            bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2
            by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2
            pitch_x, pitch_y = -1, -1
            if homography_matrix is not None:
                try:
                    pitch_x, pitch_y = homography_matrix.pixel_to_pitch(bx, by)
                except Exception as e:
                    # pitch_x/pitch_y stay at the -1,-1 sentinel; already
                    # correctly guarded downstream (`>= 0` checks) before use.
                    logger.debug(f"pixel_to_pitch failed: {e}")
            ball_trail.append((frame.timestamp, bx, by, pitch_x, pitch_y))
            if len(ball_trail) > 30:
                ball_trail.pop(0)

        return events

    def _detect_free_kicks(self, track_data: MatchTrackData, homography_matrix=None) -> list[dict]:
        """Detect free kicks: ball stationary 2+ seconds then kicked hard."""
        events: list[dict] = []
        stationary_frames = 0
        stationary_start_time = 0.0
        stationary_pitch_pos: tuple[float, float] | None = None
        min_stationary_frames = int(60 / 3)
        max_stationary_dist = 5.0 if homography_matrix else 20.0
        kick_speed_threshold = 15.0 if homography_matrix else 400.0
        _ = 300

        prev_ball_center: tuple[float, float] | None = None
        for frame in track_data.frames:
            ball_det = None
            for det in frame.detections:
                if det.class_name == "sports ball":
                    ball_det = det
                    break

            if ball_det is None:
                stationary_frames = 0
                stationary_start_time = 0.0
                stationary_pitch_pos = None
                prev_ball_center = None
                continue

            bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2
            by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2
            pitch_x, pitch_y = -1, -1
            if homography_matrix is not None:
                try:
                    pitch_x, pitch_y = homography_matrix.pixel_to_pitch(bx, by)
                except Exception as e:
                    # pitch_x/pitch_y stay at the -1,-1 sentinel; already
                    # correctly guarded downstream (`>= 0` check) before use.
                    logger.debug(f"pixel_to_pitch failed: {e}")

            if prev_ball_center is not None:
                dist = math.sqrt((bx - prev_ball_center[0]) ** 2 + (by - prev_ball_center[1]) ** 2)
            else:
                dist = 999
            prev_ball_center = (bx, by)

            if dist < max_stationary_dist:
                stationary_frames += 1
                if stationary_start_time == 0.0:
                    stationary_start_time = frame.timestamp
                    stationary_pitch_pos = (pitch_x, pitch_y)
            else:
                if stationary_frames >= min_stationary_frames:
                    dt_prev = 1.0 / 30.0
                    speed = dist / max(dt_prev, 0.001)
                    if speed >= kick_speed_threshold:
                        team = "unknown"
                        closest_player = None
                        closest_dist = float("inf")
                        for det in frame.detections:
                            if det.class_name == "person":
                                px = (det.bbox[0] + det.bbox[2]) / 2
                                py = (det.bbox[1] + det.bbox[3]) / 2
                                d = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
                                if d < closest_dist:
                                    closest_dist = d
                                    closest_player = det
                        if closest_player and closest_player.track_id is not None:
                            team = self._get_player_team(track_data, closest_player.track_id)

                        pitch_loc = (
                            stationary_pitch_pos
                            if stationary_pitch_pos and stationary_pitch_pos[0] >= 0
                            else None
                        )
                        events.append(
                            {
                                "type": "free_kick",
                                "timestamp": stationary_start_time,
                                "team": team,
                                "confidence": 0.5,
                                "metadata": {
                                    "stationary_duration_s": round(
                                        frame.timestamp - stationary_start_time, 2
                                    ),
                                    "pitch_x": round(pitch_loc[0], 1) if pitch_loc else -1,
                                    "pitch_y": round(pitch_loc[1], 1) if pitch_loc else -1,
                                },
                            }
                        )

                stationary_frames = 0
                stationary_start_time = 0.0
                _ = None
                stationary_pitch_pos = None

        return events

    def _detect_throw_ins(self, track_data: MatchTrackData, homography_matrix=None) -> list[dict]:
        """Detect throw-ins: ball near sideline after going out of play."""
        events: list[dict] = []
        sideline_threshold = 8.0 if homography_matrix else 50
        ball_lost_frames = 0

        ball_trail: list[tuple[float, float, float, float, float]] = []
        for frame in track_data.frames:
            ball_det = None
            for det in frame.detections:
                if det.class_name == "sports ball":
                    ball_det = det
                    break

            if ball_det is None:
                ball_lost_frames += 1
                if ball_lost_frames > 5 and len(ball_trail) >= 3:
                    last = ball_trail[-1]
                    bx, by = last[1], last[2]
                    pitch_x, pitch_y = last[3], last[4]

                    near_sideline = False
                    side = "unknown"
                    if homography_matrix is not None and pitch_x >= 0:
                        near_sideline = (
                            pitch_y < sideline_threshold
                            or pitch_y > self.pitch_width - sideline_threshold
                        )
                        if pitch_y < sideline_threshold:
                            side = "top"
                        elif pitch_y > self.pitch_width - sideline_threshold:
                            side = "bottom"
                    else:
                        fh = track_data.frames[-1].image_height if track_data.frames else 480
                        near_sideline = by < sideline_threshold or by > fh - sideline_threshold
                        if by < sideline_threshold:
                            side = "top"
                        elif by > fh - sideline_threshold:
                            side = "bottom"

                    if near_sideline:
                        team = "unknown"
                        events.append(
                            {
                                "type": "throw_in",
                                "timestamp": frame.timestamp,
                                "team": team,
                                "confidence": 0.4,
                                "metadata": {
                                    "side": side,
                                    "pitch_x": round(pitch_x, 1)
                                    if homography_matrix and pitch_x >= 0
                                    else -1,
                                    "pitch_y": round(pitch_y, 1)
                                    if homography_matrix and pitch_y >= 0
                                    else -1,
                                },
                            }
                        )
                    ball_trail = []
                continue
            else:
                ball_lost_frames = 0

            bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2
            by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2
            pitch_x, pitch_y = -1, -1
            if homography_matrix is not None:
                try:
                    pitch_x, pitch_y = homography_matrix.pixel_to_pitch(bx, by)
                except Exception as e:
                    # pitch_x/pitch_y stay at the -1,-1 sentinel; already
                    # correctly guarded downstream (`>= 0` checks) before use.
                    logger.debug(f"pixel_to_pitch failed: {e}")
            ball_trail.append((frame.timestamp, bx, by, pitch_x, pitch_y))
            if len(ball_trail) > 30:
                ball_trail.pop(0)

        return events

    def _detect_crosses(
        self, track_data: MatchTrackData, base_events: list[dict], homography_matrix=None
    ) -> list[dict]:
        """Detect crosses: pass from wide area into penalty area."""
        events: list[dict] = []

        for event in base_events:
            if event.get("type") != "pass":
                continue

            start_x, start_y = self._event_start_pos(event)
            end_x, end_y = self._event_end_pos(event)
            if start_x is None or start_y is None or end_x is None or end_y is None:
                # No usable position information -- skip honestly instead of
                # treating a missing position as the top-left corner.
                continue

            # Wide area: y near edges (within 10m of sideline)
            is_wide = start_y < 10 or start_y > (self.pitch_width - 10)

            # Into penalty area: x near goal (within 16.5m), y near center
            goal_line = self.pitch_length
            in_penalty_area = end_x > (goal_line - 16.5) and 10 < end_y < (self.pitch_width - 10)

            if is_wide and in_penalty_area:
                events.append(
                    {
                        "type": "cross",
                        "timestamp": event["timestamp"],
                        "from_track_id": self._passer_id(event),
                        "to_track_id": self._receiver_id(event),
                        "team": event.get("team", "unknown"),
                        "completed": event.get("completed", False),
                        "confidence": event.get("confidence", 0.5) + 0.1,
                        "metadata": {
                            "from_wide": True,
                            "into_box": True,
                            "start_y": start_y,
                            "end_x": end_x,
                        },
                    }
                )

        return events

    def _detect_ball_recoveries(
        self, track_data: MatchTrackData, base_events: list[dict], homography_matrix=None
    ) -> list[dict]:
        """Detect ball recoveries: team wins possession back."""
        events: list[dict] = []
        prev_team = None
        recovery_cooldown = 0

        for event in sorted(base_events, key=lambda e: e.get("timestamp", 0)):
            team = event.get("team", "unknown")
            if team == "unknown":
                continue

            if prev_team is not None and team != prev_team and recovery_cooldown <= 0:
                events.append(
                    {
                        "type": "ball_recovery",
                        "timestamp": event["timestamp"],
                        "team": team,
                        "from_team": prev_team,
                        "confidence": 0.5,
                        "metadata": {
                            "recovery_after_loss": True,
                        },
                    }
                )
                recovery_cooldown = 3  # 3-second cooldown

            prev_team = team
            recovery_cooldown -= 1

        return events

    def _detect_blocks(
        self, track_data: MatchTrackData, base_events: list[dict], homography_matrix=None
    ) -> list[dict]:
        """Detect blocks: defender near ball when shot/pass is taken."""
        events: list[dict] = []

        for event in base_events:
            if event.get("type") not in ("shot", "pass"):
                continue

            from_tid = self._passer_id(event)
            from_team = self._get_player_team(track_data, from_tid) if from_tid else "unknown"

            # Check if the event was NOT completed (shot blocked or pass blocked)
            if not event.get("completed", True):
                # Look for a nearby defender in the frame
                events.append(
                    {
                        "type": "block",
                        "timestamp": event["timestamp"],
                        "team": from_team,  # team that attempted the action
                        "defending_team": "unknown",  # would need to check nearby players
                        "confidence": 0.5,
                        "metadata": {
                            "blocked_action": event.get("type"),
                            "original_event_id": event.get("timestamp"),
                        },
                    }
                )

        return events

    def _detect_duels(self, track_data: MatchTrackData, homography_matrix=None) -> list[dict]:
        """Detect duels: two players from opposite teams within 2m of each other near ball."""
        events: list[dict] = []
        duel_cooldown: dict[int, int] = defaultdict(int)

        for frame in track_data.frames:
            # Find ball position
            ball_det = None
            for det in frame.detections:
                if det.class_name == "sports ball":
                    ball_det = det
                    break

            if ball_det is None:
                continue

            bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2
            by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2

            # Find players near ball
            nearby_players = []
            for det in frame.detections:
                if det.class_name != "person" or det.track_id is None:
                    continue
                px = (det.bbox[0] + det.bbox[2]) / 2
                py = (det.bbox[1] + det.bbox[3]) / 2
                d = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
                if d < 80:  # within ~80 pixels of ball
                    team = self._get_player_team(track_data, det.track_id)
                    nearby_players.append((det.track_id, team, px, py))

            # Find pairs from different teams
            for i, (tid1, team1, x1, y1) in enumerate(nearby_players):
                if team1 == "unknown":
                    continue
                for j in range(i + 1, len(nearby_players)):
                    tid2, team2, x2, y2 = nearby_players[j]
                    if team2 == "unknown" or team1 == team2:
                        continue

                    d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    # within 50 pixels
                    if d < 50 and duel_cooldown[tid1] <= 0 and duel_cooldown[tid2] <= 0:
                        events.append(
                            {
                                "type": "duel",
                                "timestamp": frame.timestamp,
                                "track_id_1": tid1,
                                "team_1": team1,
                                "track_id_2": tid2,
                                "team_2": team2,
                                "confidence": min(1.0, 1.0 - d / 50),
                                "metadata": {"distance_px": round(d, 1), "near_ball": True},
                            }
                        )
                        duel_cooldown[tid1] = 15
                        duel_cooldown[tid2] = 15

            # Decrement cooldowns
            for tid in list(duel_cooldown.keys()):
                duel_cooldown[tid] = max(0, duel_cooldown[tid] - 1)

        return events

    def _detect_carries(
        self, track_data: MatchTrackData, base_events: list[dict], homography_matrix=None
    ) -> list[dict]:
        """Detect carries: ball moves with player without being a pass or dribble."""
        events: list[dict] = []
        # Carries are essentially short dribbles that don't meet dribble criteria
        # Or ball movement between pass and dribble
        # For now, this is a placeholder - carries are hard to distinguish from dribbles
        return events

    def _detect_progressive_actions(
        self, track_data: MatchTrackData, base_events: list[dict], homography_matrix=None
    ) -> list[dict]:
        """Tag progressive passes and carries (ball advances meaningfully
        toward the opponent's goal).

        The progressive/not-progressive decision is delegated to
        core.progressive_actions's individually-importable helpers --
        ratio-of-remaining-distance-to-goal + attacking-third check for
        passes, an absolute-metres threshold for carries, both driven by
        game_constants.py (the project's unified source of truth for these
        thresholds, per the Sprint 11 pitch-constants-unification work) --
        instead of this method's own flat 10m-absolute-progress rule,
        which had no zone awareness and used a magic number duplicated
        nowhere else. This keeps its own output shape (individual tagged
        events merged into the live detection pipeline's stream), since
        that's a different job from core.progressive_actions's
        analyze_progressive_passes(), which produces a team-level summary
        report from a differently-shaped input and isn't a fit here.
        """
        from kawkab.core.progressive_actions import _is_progressive_carry, _is_progressive_pass

        events: list[dict] = []

        for event in base_events:
            ev_type = event.get("type")
            if ev_type not in ("pass", "dribble", "carry"):
                continue

            start_x, start_y = self._event_start_pos(event)
            end_x, end_y = self._event_end_pos(event)
            if start_x is None or end_x is None:
                # No usable position -- cannot judge progress honestly.
                continue
            if start_y is None:
                start_y = self.pitch_width / 2
            if end_y is None:
                end_y = self.pitch_width / 2
            team = event.get("team", "home")
            # Toward opponent goal: if home, x increases; if away, x decreases
            attacking_direction = 1 if team == "home" else -1
            progress = (end_x - start_x) * attacking_direction

            if ev_type == "pass":
                is_progressive = _is_progressive_pass(
                    start_x, end_x, start_y, end_y, self.pitch_length, attacking_direction
                )
            else:  # dribble or carry
                is_progressive = _is_progressive_carry(
                    start_x, end_x, abs(progress), attacking_direction
                )

            if is_progressive:
                event["is_progressive"] = True
                event["progress_m"] = round(progress, 1)
                events.append(
                    {
                        "type": "progressive_action",
                        "timestamp": event["timestamp"],
                        "original_type": ev_type,
                        "team": team,
                        "progress_m": round(progress, 1),
                        "confidence": event.get("confidence", 0.5),
                        "metadata": {
                            "start_x": start_x,
                            "end_x": end_x,
                            "original_event": event,
                        },
                    }
                )

        return events

    def _detect_final_third_entries(
        self, track_data: MatchTrackData, base_events: list[dict], homography_matrix=None
    ) -> list[dict]:
        """Detect passes and carries that enter the final third (last 35m)."""
        events: list[dict] = []
        final_third_start = self.pitch_length * 0.67

        for event in base_events:
            if event.get("type") not in ("pass", "dribble"):
                continue

            start_x, _start_y = self._event_start_pos(event)
            end_x, _end_y = self._event_end_pos(event)
            if start_x is None or end_x is None:
                continue

            # Entry: started before final third, ended inside final third
            if start_x < final_third_start and end_x >= final_third_start:
                events.append(
                    {
                        "type": "final_third_entry",
                        "timestamp": event["timestamp"],
                        "original_type": event.get("type"),
                        "team": event.get("team", "unknown"),
                        "confidence": event.get("confidence", 0.5),
                        "metadata": {
                            "entry_point": round(end_x, 1),
                        },
                    }
                )

        return events

    def _detect_high_turnovers(
        self, track_data: MatchTrackData, base_events: list[dict], homography_matrix=None
    ) -> list[dict]:
        """Detect high turnovers: ball lost in final 40m of pitch."""
        events: list[dict] = []
        high_turnover_line = self.pitch_length * 0.6

        for event in base_events:
            if event.get("type") not in ("pass", "dribble"):
                continue

            if not event.get("completed", True):
                # Ball lost
                start_x, _start_y = self._event_start_pos(event)
                if start_x is None:
                    continue
                team = event.get("team", "unknown")

                # Check if lost in attacking area
                if (
                    team == "home"
                    and start_x > high_turnover_line
                    or team == "away"
                    and start_x < (self.pitch_length - high_turnover_line)
                ):
                    events.append(
                        {
                            "type": "high_turnover",
                            "timestamp": event["timestamp"],
                            "team": team,
                            "confidence": event.get("confidence", 0.5),
                            "metadata": {
                                "lost_in_final_third": True,
                                "position_x": start_x,
                            },
                        }
                    )

        return events
