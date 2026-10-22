"""Core analysis service — shared dataclasses, constants, and orchestrator."""

from __future__ import annotations

import contextlib
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kawkab.core.events import (
    BaseEvent,
    CarryEvent,
    PassEvent,
    PassType,
    PressureContext,
    ShotEvent,
)
from kawkab.core.game_constants import GAME
from kawkab.core.logging import get_logger
from kawkab.core.pitch_control import MatchPitchControl, VoronoiPitchControl
from kawkab.core.player_rating import (
    PlayerRating,
)
from kawkab.core.xg_model import active_xg_model, compute_xg_trained_from_shot_event
from kawkab.services.cv_service import MatchTrackData

logger = get_logger(__name__)

if TYPE_CHECKING:
    # AnalysisService composes the tracking/xg_xt/passing mixins onto this
    # class at runtime (services/analysis_service.py), but bare
    # AnalysisServiceCore is constructed directly in tests -- so declare the
    # mixin-supplied methods here for standalone type checking.
    class _MixinProtocol:
        pitch_width: float
        use_kalman: bool

        def _compute_player_stats(self, track_data, homography_matrix=None): ...
        def _compute_player_stats_kalman(
            self, track_data, homography_matrix, max_frame_delta_m
        ): ...
        def _compute_pass_network(self, events, player_teams=None): ...
        def compute_ppda(self, track_data, team="home", homography_matrix=None): ...
        def detect_formation(
            self, track_data, team="home", n_players=11, homography_matrix=None
        ): ...
        def compute_xt_simple(self, events): ...
        def _compute_player_ratings(
            self, players, typed_events, pitch_control, track_data, homography_matrix
        ): ...

else:
    _MixinProtocol = object

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


@dataclass
class PlayerStats:
    track_id: int
    jersey_number: int | None = None
    name: str | None = None
    team: str | None = None
    position: str | None = None
    distance_covered_m: float = 0.0
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    passes_attempted: int = 0
    passes_completed: int = 0
    shots: int = 0
    tackles: int = 0
    interceptions: int = 0
    possession_time_s: float = 0.0
    positions: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def pass_accuracy(self) -> float:
        if self.passes_attempted == 0:
            return 0.0
        return self.passes_completed / self.passes_attempted


@dataclass
class TeamStats:
    team_name: str
    possession_pct: float = 0.0
    passes_completed: int = 0
    passes_attempted: int = 0
    shots: int = 0
    shots_on_target: int = 0
    tackles: int = 0
    corners: int = 0
    fouls: int = 0
    distance_covered_km: float = 0.0

    @property
    def pass_accuracy(self) -> float:
        if self.passes_attempted == 0:
            return 0.0
        return self.passes_completed / self.passes_attempted


@dataclass
class MatchAnalysis:
    match_id: int
    duration_seconds: float
    home_team: TeamStats
    away_team: TeamStats
    players: dict[int, PlayerStats]
    events: list[dict] = field(default_factory=list)
    pass_network: dict = field(default_factory=dict)
    formations: dict = field(default_factory=dict)
    pressing_intensity: float = 0.0
    ppda_breakdown: dict = field(default_factory=dict)
    xg_total: dict = field(default_factory=dict)
    xt_total: dict = field(default_factory=dict)
    confidence_overall: float = 0.0
    confidence_breakdown: dict = field(default_factory=dict)
    typed_events: list[BaseEvent] = field(default_factory=list)
    pass_type_breakdown: dict = field(default_factory=dict)
    carry_events: list[dict] = field(default_factory=list)
    progressive_passes: list[dict] = field(default_factory=list)
    progressive_carries: list[dict] = field(default_factory=list)
    pitch_control: MatchPitchControl | None = None
    player_ratings: dict[int, PlayerRating] = field(default_factory=dict)


class AnalysisServiceCore(_MixinProtocol):
    def __init__(
        self,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        use_kalman: bool = True,
    ) -> None:
        self.pitch_length = pitch_length_m
        self.pitch_width = pitch_width_m
        self.use_kalman = use_kalman
        logger.info(
            f"AnalysisService: pitch={pitch_length_m}x{pitch_width_m}m, kalman={use_kalman}"
        )

    @staticmethod
    def _ensure_package_loaded(package_name: str) -> None:
        parts = package_name.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if prefix not in __import__("sys").modules:
                with contextlib.suppress(ImportError):
                    __import__("importlib").import_module(prefix)

    async def analyze_match(
        self, track_data: MatchTrackData, match_id: int = 0, homography_matrix=None
    ) -> MatchAnalysis:
        logger.info(f"Analyzing match: {track_data.total_frames} frames")

        # Auto-use homography from CV auto-calibration when none explicitly provided
        if homography_matrix is None:
            auto_hom = track_data.tracking_metrics.get("auto_homography")
            if auto_hom is not None:
                homography_matrix = auto_hom
                logger.info("Using auto-calibrated homography from PitchDetector")

        if homography_matrix is not None and track_data.player_teams:
            self._assign_teams_by_pitch_side(track_data, homography_matrix)

        players = self._compute_player_stats(track_data, homography_matrix)
        events = self._detect_events(track_data, homography_matrix)
        team_stats = self._compute_team_stats(players, events, track_data, homography_matrix)
        possession = self._compute_possession(track_data, homography_matrix)
        pass_network = self._compute_pass_network(events, track_data.player_teams)

        # Merge the event-driven team stats with the possession split.
        # (_compute_player_stats fills only physical fields; all event counts
        # live on these objects, so there is no double-counting.)
        home = team_stats["home"]
        away = team_stats["away"]
        home.possession_pct = possession["home"]
        away.possession_pct = possession["away"]

        typed_events: list[BaseEvent] = []
        for event in events:
            team = event.get("team", "home")
            target = home if team == "home" else away
            if event["type"] == "pass":
                target.passes_attempted += 1
                if event.get("completed"):
                    target.passes_completed += 1
                pe = self._build_typed_pass(event, homography_matrix)
                typed_events.append(pe)
            elif event["type"] == "shot":
                target.shots += 1
                if event.get("on_target"):
                    target.shots_on_target += 1
                se = self._build_typed_shot(event, homography_matrix)
                typed_events.append(se)

        for ev in typed_events:
            if isinstance(ev, PassEvent) and ev.track_id is not None:
                player = players.get(ev.track_id)
                if player:
                    player.passes_attempted += 1
                    if ev.completed:
                        player.passes_completed += 1

        carry_dicts = self._detect_carries(track_data, typed_events, homography_matrix)
        for cd in carry_dicts:
            typed_events.append(cd)

        self._infer_pressure_on_events(track_data, typed_events)
        self._classify_pass_types(typed_events, homography_matrix)
        progressive_passes_list = self._find_progressive_passes(typed_events, homography_matrix)
        pass_breakdown = self._compute_pass_type_breakdown(typed_events)

        carry_events_list = [c.to_dict() for c in typed_events if isinstance(c, CarryEvent)]
        progressive_carries_list = [
            c.to_dict() for c in typed_events if isinstance(c, CarryEvent) and c.is_progressive
        ]

        home_formation = self.detect_formation(
            track_data, team="home", homography_matrix=homography_matrix
        )
        away_formation = self.detect_formation(
            track_data, team="away", homography_matrix=homography_matrix
        )
        home_ppda = self.compute_ppda(track_data, team="home", homography_matrix=homography_matrix)
        away_ppda = self.compute_ppda(track_data, team="away", homography_matrix=homography_matrix)
        confidence = self._compute_confidence(track_data, events)

        shot_events = [e for e in typed_events if isinstance(e, ShotEvent)]
        _ = active_xg_model()
        for se in shot_events:
            gk_distance = None
            if se.gk_position_x is not None and se.x is not None and se.y is not None:
                assert se.gk_position_y is not None  # invariant: y set alongside x
                gk_distance = math.hypot(se.gk_position_x - se.x, se.gk_position_y - se.y)
            se.xg = compute_xg_trained_from_shot_event(se, gk_distance_m=gk_distance)
            se.xg = max(0.0, min(1.0, se.xg))
        home_xg = sum(e.xg for e in shot_events if e.team == "home")
        away_xg = sum(e.xg for e in shot_events if e.team == "away")
        xg_data = {
            "home": round(home_xg, 3),
            "away": round(away_xg, 3),
            "shot_details": [
                {
                    "timestamp": e.timestamp,
                    "team": e.team,
                    "distance_m": e.distance_m or 0,
                    "angle_deg": e.angle_deg or 0,
                    "xg": round(e.xg, 4),
                    "on_target": e.on_target,
                    "body_part": e.body_part.value if e.body_part else "unknown",
                }
                for e in shot_events
            ],
        }

        xt_data = self.compute_xt_simple(events)
        logger.info(f"xG: home={xg_data['home']} away={xg_data['away']} ({len(shot_events)} shots)")
        logger.info(f"xT: home={xt_data['home']} away={xt_data['away']}")

        pitch_control = self._compute_pitch_control(track_data, homography_matrix)
        player_ratings = self._compute_player_ratings(
            players, typed_events, pitch_control, track_data, homography_matrix
        )

        coords = "meters" if homography_matrix else "pixels"
        logger.info(
            f"Analysis complete: {len(players)} players, "
            f"{len(typed_events)} typed events, confidence={confidence:.2%}, "
            f"formations: {home_formation['formation']}/{away_formation['formation']}, "
            f"coords={coords}"
        )

        backward_compat_events = [e.to_dict() if hasattr(e, "to_dict") else e for e in typed_events]

        return MatchAnalysis(
            match_id=match_id,
            duration_seconds=track_data.duration_seconds,
            home_team=home,
            away_team=away,
            players=players,
            events=backward_compat_events,  # type: ignore[arg-type]
            pass_network=pass_network,
            confidence_overall=confidence,
            formations={
                "home": home_formation,
                "away": away_formation,
            },
            pressing_intensity=home_ppda.get("ppda") or 0.0,
            ppda_breakdown={
                "home": home_ppda,
                "away": away_ppda,
            },
            xg_total=xg_data,
            xt_total=xt_data,
            typed_events=typed_events,
            pass_type_breakdown=pass_breakdown,
            carry_events=carry_events_list,
            progressive_passes=progressive_passes_list,
            progressive_carries=progressive_carries_list,
            pitch_control=pitch_control,
            player_ratings=player_ratings,
        )

    def _build_typed_pass(self, event: dict, homography_matrix=None) -> PassEvent:
        meta = event.get("metadata", {})
        sx = meta.get("start_x_pct", 0.5)
        sy = meta.get("start_y_pct", 0.5)
        ex = meta.get("end_x_pct", 0.6)
        ey = meta.get("end_y_pct", 0.5)
        dx = (ex - sx) * self.pitch_length
        dy = (ey - sy) * self.pitch_width
        length_m = math.sqrt(dx * dx + dy * dy)
        return PassEvent(
            timestamp=event.get("timestamp", 0),
            team=event.get("team", "unknown"),
            track_id=event.get("from_track_id"),
            to_track_id=event.get("to_track_id"),
            start_x=sx,
            start_y=sy,
            end_x=ex,
            end_y=ey,
            completed=event.get("completed", True),
            length_m=length_m,
            confidence=event.get("confidence", 0.5),
            period=1,
        )

    def _build_typed_shot(self, event: dict, homography_matrix=None) -> ShotEvent:
        """Build a typed ShotEvent from a raw CV-pipeline shot event.

        Convention notes (see CLAUDE.md's angle-convention table):
        - ``angle_deg`` on ShotEvent is consumed by xG models in the
          DEVIATION-from-central convention (0° = straight at goal,
          larger = wider) — which is what the CV pipeline's
          ``angle_to_goal_deg`` metadata stores. StatsBomb-imported
          events store OPENING angle under the ``angle_deg`` metadata
          key instead and must not be fed through here unconverted.
        - Missing spatial metadata stays None. The old code silently
          fabricated distance=18.0 m / angle=30° and the xG model
          happily produced a plausible-looking number for a shot whose
          position was never known — a fabricated stat, not an estimate.
          The trained model treats gk_distance_m=0 as "feature absent",
          so an honestly-absent value degrades gracefully instead of
          lying.
        """
        meta = event.get("metadata", {})
        distance_raw = meta.get("distance_to_goal_m")
        angle_raw = meta.get("angle_to_goal_deg")
        return ShotEvent(
            timestamp=event.get("timestamp", 0),
            team=event.get("team", "unknown"),
            track_id=event.get("track_id"),
            on_target=event.get("on_target", False),
            distance_m=float(distance_raw) if distance_raw is not None else None,
            angle_deg=float(angle_raw) if angle_raw is not None else None,
            xg=meta.get("xg", 0.0),
            confidence=event.get("confidence", 0.5),
            gk_position_x=meta.get("gk_pitch_x"),
            gk_position_y=meta.get("gk_pitch_y"),
            period=1,
        )

    def _detect_carries(self, track_data, typed_events, homography_matrix=None) -> list[CarryEvent]:
        carries: list[CarryEvent] = []
        if not track_data.frames:
            return carries
        fps = max(1, track_data.fps)
        min_carry_frames = int(fps * 0.5)
        min_carry_distance = 2.0

        pass_frames: set[int] = set()
        for ev in typed_events:
            if isinstance(ev, PassEvent):
                idx = int(ev.timestamp * fps)
                for offset in range(-2, 3):
                    pass_frames.add(idx + offset)

        ball_tracker: dict[int, list[tuple[float, float, float, int]]] = {}
        carry_start: dict[int, tuple[int, float, float, float]] = {}

        for frame in track_data.frames:
            fno = frame.frame_number
            if fno in pass_frames:
                continue
            ts = frame.timestamp
            ball_det = None
            for det in frame.detections:
                if det.class_name == "sports ball":
                    ball_det = det
                    break
            if ball_det is None:
                continue
            bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2.0
            by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2.0
            closest = None
            closest_dist = float("inf")
            for det in frame.detections:
                if det.class_name != "person" or det.track_id is None:
                    continue
                px = (det.bbox[0] + det.bbox[2]) / 2.0
                py = (det.bbox[1] + det.bbox[3]) / 2.0
                d = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
                if d < closest_dist:
                    closest_dist = d
                    closest = det
            if closest is None or closest.track_id is None:
                continue
            tid = closest.track_id
            if tid not in ball_tracker:
                ball_tracker[tid] = [(ts, bx, by, fno)]
            else:
                ball_tracker[tid].append((ts, bx, by, fno))
            if tid not in carry_start:
                carry_start[tid] = (fno, ts, bx, by)
            else:
                cfno, cts, cbx, cby = carry_start[tid]
                elapsed_frames = fno - cfno
                if elapsed_frames >= min_carry_frames:
                    dx = bx - cbx
                    dy = by - cby
                    cd = math.sqrt(dx * dx + dy * dy)
                    if cd >= min_carry_distance:
                        cx, cy = bx, by
                        if homography_matrix is not None:
                            try:
                                cx, cy = homography_matrix.pixel_to_pitch(bx, by)
                            except Exception as e:
                                logger.warning(
                                    "Failed to convert ball pixel-to-pitch in carry detection: %s",
                                    e,
                                )
                        scx, scy = cbx, cby
                        if homography_matrix is not None:
                            try:
                                scx, scy = homography_matrix.pixel_to_pitch(cbx, cby)
                            except Exception as e:
                                logger.warning(
                                    "Failed to convert carry-start pixel-to-pitch: %s", e
                                )
                        team = "unknown"
                        if track_data.player_teams:
                            team = track_data.player_teams.get(tid, "unknown")
                        carries.append(
                            CarryEvent(
                                timestamp=cts,
                                team=team,
                                track_id=tid,
                                start_x=scx,
                                start_y=scy,
                                end_x=cx,
                                end_y=cy,
                                distance_m=cd if homography_matrix else cd * 0.015,
                                is_progressive=False,
                                confidence=0.5,
                            )
                        )
                    carry_start.pop(tid, None)
                    ball_tracker.pop(tid, None)
        return carries

    @staticmethod
    def _pixel_dist_to_meters(sx, sy, ox, oy, homography_matrix) -> float:
        if homography_matrix is not None:
            try:
                sx_m, sy_m = homography_matrix.pixel_to_pitch(sx, sy)
                ox_m, oy_m = homography_matrix.pixel_to_pitch(ox, oy)
                return math.sqrt((ox_m - sx_m) ** 2 + (oy_m - sy_m) ** 2)
            except Exception:
                pass
        # Uncalibrated: approximate with the project-wide pixel->meter
        # ratio rather than returning the raw pixel distance mislabeled
        # as meters (the old behavior fed pixel distances into meter
        # thresholds like the 2m "is_pressed" check unchanged).
        from kawkab.core.game_constants import GAME

        px_dist = math.sqrt((ox - sx) ** 2 + (oy - sy) ** 2)
        return px_dist * GAME.CARRY_PIXEL_TO_METER_RATIO

    def _infer_pressure_on_events(self, track_data, typed_events, homography_matrix=None) -> None:
        if not track_data.frames or not track_data.player_teams:
            return

        _ = {"home": "left", "away": "right"}

        for event in typed_events:
            ts = event.timestamp
            frame_idx = min(int(ts * track_data.fps), len(track_data.frames) - 1)
            if frame_idx < 0:
                continue
            frame = track_data.frames[frame_idx]

            home_players = []
            away_players = []
            ball_x, ball_y = 0, 0
            has_ball = False
            shooter_pos = None

            for det in frame.detections:
                if det.class_name == "sports ball":
                    bx = (det.bbox[0] + det.bbox[2]) / 2.0
                    by = (det.bbox[1] + det.bbox[3]) / 2.0
                    ball_x, ball_y = bx, by
                    has_ball = True
                elif det.class_name == "person" and det.track_id is not None:
                    px = (det.bbox[0] + det.bbox[2]) / 2.0
                    py = (det.bbox[1] + det.bbox[3]) / 2.0
                    team = track_data.player_teams.get(det.track_id)
                    if team == "home":
                        home_players.append((det.track_id, px, py))
                        if det.track_id == event.track_id:
                            shooter_pos = (px, py)
                    elif team == "away":
                        away_players.append((det.track_id, px, py))
                        if det.track_id == event.track_id:
                            shooter_pos = (px, py)

            if shooter_pos is None and has_ball:
                shooter_pos = (ball_x, ball_y)
            if shooter_pos is None:
                continue

            event_team = event.team
            opponents = away_players if event_team == "home" else home_players
            if not opponents:
                continue

            sx, sy = shooter_pos
            min_dist = float("inf")
            min_angle = 0.0
            count_within_5m = 0

            for _tid, ox, oy in opponents:
                dx = ox - sx
                dy = oy - sy
                dist_pitch = self._pixel_dist_to_meters(sx, sy, ox, oy, homography_matrix)
                if dx * dx + dy * dy < min_dist * min_dist:
                    min_dist = math.sqrt(dx * dx + dy * dy)
                    min_angle = math.degrees(math.atan2(abs(dy), abs(dx)))
                if dist_pitch < 5.0:
                    count_within_5m += 1

            min_dist_pitch = min(
                (
                    self._pixel_dist_to_meters(sx, sy, ox, oy, homography_matrix)
                    for _, ox, oy in opponents
                ),
                default=float("inf"),
            )

            event.pressure = PressureContext(
                nearest_defender_distance=min_dist_pitch,
                nearest_defender_angle=min_angle,
                defenders_within_5m=count_within_5m,
                is_pressed=min_dist_pitch < 2.0,
            )

            if isinstance(event, ShotEvent):
                event.was_pressed = min_dist_pitch < 2.0

    def _classify_pass_types(self, typed_events, homography_matrix=None) -> None:
        for event in typed_events:
            if not isinstance(event, PassEvent):
                continue
            sx, sy = (event.start_x or 0.5), (event.start_y or 0.5)
            ex, ey = (event.end_x or 0.6), (event.end_y or 0.5)

            x_gain = ex - sx
            y_gain = abs(ey - sy)

            if event.length_m > 30.0:
                event.pass_type = PassType.LONG_BALL
            if x_gain < 0:
                event.pass_type = PassType.BACK_PASS
                continue
            if y_gain > 0.5:
                event.pass_type = PassType.SWITCH
                continue
            if (ey < 0.2 or ey > 0.8) and ex > 0.7:
                event.pass_type = PassType.CROSS
                event.is_cross = True
                continue
            if x_gain > 0.15 and event.length_m > 15.0:
                event.pass_type = PassType.THROUGH_BALL
                event.is_through_ball = True
                continue
            if event.length_m < 5.0:
                event.pass_type = PassType.ONE_TOUCH
                continue
            event.pass_type = PassType.STANDARD

    def _find_progressive_passes(self, typed_events, homography_matrix=None) -> list[dict]:
        progressive = []
        for event in typed_events:
            if not isinstance(event, PassEvent):
                continue
            if not event.completed:
                continue
            sx, ex = (event.start_x or 0.5), (event.end_x or 0.6)
            x_gain = (ex - sx) * self.pitch_length
            is_progressive = x_gain > 5.0 and event.length_m > 10.0
            event.is_progressive = is_progressive
            if is_progressive:
                progressive.append(event.to_dict())
        return progressive

    def _compute_pass_type_breakdown(self, typed_events) -> dict:
        from collections import Counter

        counts: Counter[str] = Counter()
        progressive_count = 0
        key_pass_count = 0
        assist_count = 0
        total = 0

        for event in typed_events:
            if not isinstance(event, PassEvent) or not event.completed:
                continue
            total += 1
            counts[event.pass_type.value] += 1
            if event.is_progressive:
                progressive_count += 1
            if event.is_key_pass:
                key_pass_count += 1
            if event.is_assist:
                assist_count += 1

        return {
            "total": total,
            "by_type": dict(counts),
            "progressive": progressive_count,
            "key_passes": key_pass_count,
            "assists": assist_count,
        }

    def _compute_pitch_control(
        self, track_data, homography_matrix=None
    ) -> MatchPitchControl | None:
        if not track_data.frames:
            return None

        frame_data = []
        has_homography = homography_matrix is not None
        use_player_teams = bool(track_data.player_teams)

        for frame in track_data.frames:
            home_pos = []
            away_pos = []
            ball_pos = None

            for det in frame.detections:
                cx = (det.bbox[0] + det.bbox[2]) / 2.0
                cy = (det.bbox[1] + det.bbox[3]) / 2.0

                if det.class_name == "sports ball":
                    ball_pos = (cx, cy)
                    if has_homography:
                        try:
                            ball_pos = homography_matrix.pixel_to_pitch(cx, cy)
                        except Exception as e:
                            logger.warning(
                                "Failed to convert ball pixel-to-pitch in frame data: %s", e
                            )
                elif det.class_name == "person" and det.track_id is not None:
                    if has_homography:
                        try:
                            cx, cy = homography_matrix.pixel_to_pitch(cx, cy)
                        except Exception as e:
                            logger.warning(
                                "Failed to convert player pixel-to-pitch (track_id=%s): %s",
                                det.track_id,
                                e,
                            )
                    if use_player_teams:
                        team = track_data.player_teams.get(det.track_id)
                        if team == "home":
                            home_pos.append((cx, cy))
                        elif team == "away":
                            away_pos.append((cx, cy))
                    # No team assignment: the old tid%2 parity split assigned
                    # players to teams arbitrarily. Omit them -- Voronoi
                    # control with fewer, honest players beats control with
                    # half the players on the wrong teams.

            frame_data.append(
                {
                    "timestamp": frame.timestamp,
                    "home_positions": home_pos,
                    "away_positions": away_pos,
                    "ball_pos": ball_pos,
                }
            )

        pc = VoronoiPitchControl()
        return pc.compute_match_control(frame_data)

    def _assign_teams_by_pitch_side(self, track_data, homography_matrix) -> None:
        x_per_team: dict[str, list[float]] = {"home": [], "away": []}
        for tid, entry in track_data.track_registry.items():
            team = track_data.player_teams.get(tid)
            px = entry.get("first_pixel_x")
            if team not in ("home", "away") or px is None:
                continue
            try:
                pitch_x, _ = homography_matrix.pixel_to_pitch(px, 0)
                x_per_team[team].append(pitch_x)
            except Exception as e:
                logger.warning(
                    "Failed to convert pixel x for team assignment (track_id=%s): %s", tid, e
                )
                continue

        if len(x_per_team["home"]) < 3 or len(x_per_team["away"]) < 3:
            return

        home_med = statistics.median(x_per_team["home"])
        away_med = statistics.median(x_per_team["away"])

        if home_med > away_med:
            track_data.swap_teams()
            logger.info(
                f"Pitch-side heuristic: home players at x={home_med:.0f}m (right), "
                f"away at x={away_med:.0f}m (left) -> swapped teams"
            )
        else:
            logger.info(
                f"Pitch-side heuristic: home at x={home_med:.0f}m (left), "
                f"away at x={away_med:.0f}m (right) -> already correct"
            )

    def _compute_player_stats_kalman(
        self, track_data, homography_matrix, max_frame_delta_m: float
    ) -> dict[int, PlayerStats]:
        from kawkab.services.kalman_smoother import PlayerPositionSmoother

        players: dict[int, PlayerStats] = {}
        track_positions: dict[int, list[tuple[float, float, float]]] = defaultdict(list)

        for frame in track_data.frames:
            ts = frame.timestamp
            for det in frame.detections:
                if det.class_name != "person" or det.track_id is None:
                    continue
                tid = det.track_id
                if tid not in players:
                    players[tid] = PlayerStats(track_id=tid)
                    if track_data.player_teams:
                        players[tid].team = track_data.player_teams.get(tid, "unknown")

                x1, y1, x2, y2 = det.bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if homography_matrix is not None:
                    cx, cy = homography_matrix.pixel_to_pitch(cx, cy)

                track_positions[tid].append((ts, cx, cy))

        for tid, positions in track_positions.items():
            if len(positions) < 2:
                continue

            smoother = PlayerPositionSmoother(
                process_noise_std=0.3,
                measurement_noise_std=0.8,
            )

            smoothed: list[tuple[float, float, float]] = []
            for i, (ts, x, y) in enumerate(positions):
                if i == 0:
                    smoother.update(x, y, 0.0)
                else:
                    dt = positions[i][0] - positions[i - 1][0]
                    smoother.update(x, y, dt)
                sx, sy = smoother.get_position()
                smoothed.append((ts, sx, sy))

            total_distance = 0.0
            max_speed = 0.0
            for i in range(1, len(smoothed)):
                dt = smoothed[i][0] - smoothed[i - 1][0]
                if dt <= 0:
                    continue
                dx = smoothed[i][1] - smoothed[i - 1][1]
                dy = smoothed[i][2] - smoothed[i - 1][2]
                meters = math.sqrt(dx * dx + dy * dy)
                if meters > max_frame_delta_m:
                    meters = 0.0
                total_distance += meters
                speed_mps = meters / dt
                speed_kmh = speed_mps * 3.6
                if speed_kmh <= 36.0:
                    max_speed = max(max_speed, speed_kmh)

            players[tid].distance_covered_m = total_distance
            players[tid].max_speed_kmh = max_speed
            players[tid].positions = [(ts, x, y) for ts, x, y in smoothed]

        for _tid, player in players.items():
            if track_data.duration_seconds > 0:
                player.avg_speed_kmh = player.distance_covered_m / track_data.duration_seconds * 3.6

        return players

    def _nearest_goalkeeper_pitch_pos(
        self,
        track_data,
        frame,
        shot_team: str,
        homography_matrix,
        ball_pixel_pos: tuple[float, float] | None = None,
    ) -> tuple[float, float] | None:
        """Pitch-space position of the defending team's goalkeeper at shot time.

        The GK is the defending-side player whose center is closest to their
        own goal line — a cheap, camera-free heuristic that is correct in
        the situations that matter (GK on their line during a shot) and
        fails soft: None when homography or a defender-side player is
        missing, never a fabricated position.

        Returns (pitch_x, pitch_y) in meters, or None.
        """
        if homography_matrix is None or not track_data.player_teams:
            return None
        # Defending side = the team that did NOT shoot. "home" shoots →
        # the GK we want plays for "away" and defends the right-side goal
        # (x ≈ pitch_length); vice versa for "away" shooting. When the
        # shot's team is unknown/unassigned, fall back to picking the
        # goal the ball is actually closest to — the old else-branch
        # defaulted to "home" defending, which is only correct by
        # coincidence and picked the wrong goal half the time.
        if shot_team == "home":
            defending = "away"
        elif shot_team == "away":
            defending = "home"
        else:
            # Unknown shooter: the ball attacks the goal it is nearest
            # to, so the team defending that goal is the defending side.
            # The right-side goal (x ≈ pitch_length) is the one AWAY
            # defends — consistent with the shot-detection branch above
            # and with _assign_teams_by_pitch_side.
            ball_pitch = None
            if ball_pixel_pos is not None:
                try:
                    ball_pitch = homography_matrix.pixel_to_pitch(*ball_pixel_pos)
                except Exception:
                    ball_pitch = None
            near_right = ball_pitch is not None and ball_pitch[0] > self.pitch_length / 2
            defending = "away" if near_right else "home"
        goal_x = 0.0 if defending == "home" else self.pitch_length
        best: tuple[float, float] | None = None
        best_d = float("inf")
        for det in frame.detections:
            if det.class_name != "person" or det.track_id is None:
                continue
            if track_data.player_teams.get(det.track_id) != defending:
                continue
            cx = (det.bbox[0] + det.bbox[2]) / 2
            cy = (det.bbox[1] + det.bbox[3]) / 2
            try:
                px, py = homography_matrix.pixel_to_pitch(cx, cy)
            except Exception:
                continue
            d = abs(px - goal_x) + abs(py - self.pitch_width / 2)
            if d < best_d:
                best_d = d
                best = (px, py)
        return best

    def _detect_events(self, track_data, homography_matrix=None) -> list[dict]:
        events: list[dict] = []
        prev_possession: int | None = None
        pending_possession: int | None = None  # candidate awaiting confirmation
        pending_possession_frame: int | None = None
        ball_track_id: int | None = None
        frames_since_shot: int = 999

        player_proximity_threshold = 60
        shot_speed_threshold_pps = 600
        shot_speed_threshold_mps = 8.0
        goal_proximity_m = 20.0
        shot_cooldown_frames = 15
        # Possession-flip stability: a new possessor must hold the ball for
        # 2 consecutive real frames before the flip (and its pass) is
        # trusted. Enforced below via the pending_possession candidate's
        # frame gap (<= 3 frame numbers apart with frame_skip copies
        # excluded counts as consecutive).
        pass_completion_window = 10  # frames of look-ahead used to judge pass completion

        # Skipped frames are filled with verbatim copies of the last real
        # detection set by CVService. Re-evaluating ball possession on those
        # frozen copies double-counts whatever the previous real frame saw and
        # cannot add new information -- skip them entirely (same guard the
        # stats path in analysis/tracking.py already uses).
        _metrics = getattr(track_data, "tracking_metrics", None)
        frame_skip = 1
        if isinstance(_metrics, dict):
            try:
                frame_skip = max(1, int(_metrics.get("frame_skip", 1)))
            except (TypeError, ValueError):
                frame_skip = 1

        ball_history: list[tuple[float, float, float, float | None, float | None]] = []
        possession_ball_positions: dict[int, tuple[float, float]] = {}

        # Pending passes awaiting completion verdicts:
        # (timestamp, frame_number, passer, receiver, pass_index_in_events)
        pending_passes: list[tuple[float, int, int | None, int | None, int]] = []

        for frame in track_data.frames:
            # Frozen skip-frame copies carry no new information (see above).
            if frame_skip > 1 and frame.frame_number % frame_skip != 0:
                continue

            ball_det = None
            player_dets = []

            for det in frame.detections:
                if det.class_name == "sports ball":
                    ball_det = det
                    if det.track_id is not None:
                        ball_track_id = det.track_id
                elif det.class_name == "person":
                    player_dets.append(det)

            if ball_det is None or not player_dets:
                frames_since_shot += 1
                continue

            bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2
            by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2

            pitch_x: float | None = None
            pitch_y: float | None = None
            if homography_matrix is not None:
                try:
                    pitch_x, pitch_y = homography_matrix.pixel_to_pitch(bx, by)
                except Exception as e:
                    logger.warning("Failed to convert ball pixel-to-pitch in shot detection: %s", e)

            closest_player = None
            closest_dist = float("inf")
            for p in player_dets:
                px = (p.bbox[0] + p.bbox[2]) / 2
                py = (p.bbox[1] + p.bbox[3]) / 2
                d = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
                if d < closest_dist:
                    closest_dist = d
                    closest_player = p

            if closest_player is None or closest_player.track_id is None:
                frames_since_shot += 1
                continue

            possession_ball_positions[closest_player.track_id] = (bx, by)

            ball_history.append((frame.timestamp, bx, by, pitch_x, pitch_y))
            if len(ball_history) > 5:
                ball_history.pop(0)

            tid = closest_player.track_id
            frames_since_shot += 1

            if ball_track_id is not None and len(ball_history) >= 3:
                p0 = ball_history[-3]
                p1 = ball_history[-1]
                dt = p1[0] - p0[0]

                if dt > 0.01 and frames_since_shot >= shot_cooldown_frames:
                    dx = p1[1] - p0[1]
                    dy = p1[2] - p0[2]
                    speed_px = math.sqrt(dx * dx + dy * dy) / dt

                    is_shot = False
                    shot_conf = 0.0

                    p0_pitch_x: float | None = p0[3]
                    p0_pitch_y: float | None = p0[4]
                    p1_pitch_x: float | None = p1[3]
                    p1_pitch_y: float | None = p1[4]
                    if (
                        homography_matrix is not None
                        and p0_pitch_x is not None
                        and p1_pitch_x is not None
                        and p0_pitch_y is not None
                        and p1_pitch_y is not None
                    ):
                        dx_p = p1_pitch_x - p0_pitch_x
                        dy_p = p1_pitch_y - p0_pitch_y
                        speed_pitch = math.sqrt(dx_p * dx_p + dy_p * dy_p) / dt
                        if speed_pitch >= shot_speed_threshold_mps:
                            cx = p1_pitch_x
                            near_left = cx <= goal_proximity_m
                            near_right = cx >= (self.pitch_length - goal_proximity_m)
                            moving_left = dx_p < 0
                            moving_right = dx_p > 0
                            if (near_left and moving_left) or (near_right and moving_right):
                                is_shot = True
                                shot_conf = min(1.0, speed_pitch / 25.0)
                    elif speed_px >= shot_speed_threshold_pps:
                        img_h = frame.image_height
                        near_bottom = by > img_h * 0.7
                        near_top = by < img_h * 0.3
                        moving_down = dy > 0
                        moving_up = dy < 0
                        if (near_bottom and moving_down) or (near_top and moving_up):
                            is_shot = True
                            shot_conf = min(1.0, speed_px / 1200.0)

                    if is_shot:
                        frames_since_shot = 0
                        # Team attribution: only report a real team. The old
                        # tid % 2 parity fallback invented a 50/50 team split
                        # that had no relation to actual team membership and
                        # silently corrupted team-split shot/xG stats.
                        shot_team = "unknown"
                        if track_data.player_teams:
                            shot_team = track_data.player_teams.get(tid, "unknown")
                            if shot_team == "unknown" and prev_possession is not None:
                                shot_team = track_data.player_teams.get(prev_possession, "unknown")

                        shot_metadata: dict[str, Any] = {}
                        on_target = False
                        goal_width_m = 7.32
                        if (
                            homography_matrix is not None
                            and p1_pitch_x is not None
                            and p1_pitch_y is not None
                        ):
                            bx_pitch = p1_pitch_x
                            by_pitch = p1_pitch_y
                            pitch_len = self.pitch_length
                            pitch_wid = self.pitch_width
                            near_goal_x = 0 if bx_pitch <= pitch_len / 2 else pitch_len
                            goal_cx = near_goal_x
                            goal_cy = pitch_wid / 2
                            d_to_goal = math.sqrt(
                                (bx_pitch - goal_cx) ** 2 + (by_pitch - goal_cy) ** 2
                            )
                            angle_to_goal = math.degrees(
                                math.atan2(abs(by_pitch - goal_cy), abs(bx_pitch - goal_cx))
                            )
                            shot_metadata["distance_to_goal_m"] = round(d_to_goal, 1)
                            shot_metadata["angle_to_goal_deg"] = round(angle_to_goal, 1)
                            shot_metadata["pitch_x"] = round(bx_pitch, 1)
                            shot_metadata["pitch_y"] = round(by_pitch, 1)
                            # Goalkeeper position at shot time (pitch-space
                            # meters) — consumed by the trained xG model via
                            # ShotEvent.gk_position_x/y. Without this the
                            # live path runs with gk_distance=0 = feature
                            # absent, the strongest feature unused.
                            gk_pitch_pos = self._nearest_goalkeeper_pitch_pos(
                                track_data,
                                frame,
                                shot_team,
                                homography_matrix,
                                ball_pixel_pos=(bx, by),
                            )
                            if gk_pitch_pos is not None:
                                shot_metadata["gk_pitch_x"] = round(gk_pitch_pos[0], 1)
                                shot_metadata["gk_pitch_y"] = round(gk_pitch_pos[1], 1)
                            cross_line = abs(bx_pitch - near_goal_x) < 1.0
                            in_frame = abs(by_pitch - goal_cy) < goal_width_m / 2 + 1.0
                            on_target = cross_line and in_frame
                        else:
                            d_pix = math.sqrt(dx * dx + dy * dy)
                            shot_metadata["pixel_speed"] = round(d_pix / max(dt, 0.01), 1)
                            # Honest provenance: without homography there is
                            # no distance/angle metadata, so downstream xG is
                            # "feature absent", not a fabricated estimate.
                            shot_metadata["spatial_quality"] = "pixel_space"

                        logger.debug(
                            f"Shot by {shot_team}: d={shot_metadata.get('distance_to_goal_m', '?')}m, "
                            f"on_target={on_target}, conf={shot_conf:.2f}"
                        )
                        events.append(
                            {
                                "type": "shot",
                                "timestamp": frame.timestamp,
                                "team": shot_team,
                                "on_target": on_target,
                                "confidence": shot_conf,
                                "metadata": shot_metadata,
                            }
                        )

            closest_player = None
            closest_dist = float("inf")
            for p in player_dets:
                px = (p.bbox[0] + p.bbox[2]) / 2
                py = (p.bbox[1] + p.bbox[3]) / 2
                d = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
                if d < closest_dist:
                    closest_dist = d
                    closest_player = p

            if closest_player is None or closest_player.track_id is None:
                continue

            tid_now = closest_player.track_id

            if tid_now == prev_possession:
                # Same possessor retains the ball; any pending flip
                # candidate was single-frame jitter -- drop it.
                pending_possession = None
                pending_possession_frame = None
            elif prev_possession is not None and closest_dist < player_proximity_threshold:
                # Candidate new possessor actually near the ball.
                if (
                    pending_possession == tid_now
                    and pending_possession_frame is not None
                    and frame.frame_number - pending_possession_frame <= 3
                ):
                    # Confirmed: new possessor held the ball 2 frames in a row.
                    if track_data.player_teams:
                        team = track_data.player_teams.get(tid_now, "unknown")
                        from_team = track_data.player_teams.get(prev_possession, "unknown")
                    else:
                        # No team assignment exists at all -- an honest
                        # "unknown" beats the old tid%2 parity invention,
                        # which split teams arbitrarily and corrupted every
                        # downstream team-split stat.
                        team = "unknown"
                        from_team = "unknown"

                    start_ball = possession_ball_positions.get(prev_possession, (bx, by))
                    fw = frame.image_width or 1
                    fh = frame.image_height or 1
                    pass_metadata = {
                        "start_x_pct": round(start_ball[0] / fw, 4),
                        "start_y_pct": round(start_ball[1] / fh, 4),
                        "end_x_pct": round(bx / fw, 4),
                        "end_y_pct": round(by / fh, 4),
                    }

                    # Completion semantics: a possession flip straight to an
                    # OPPOSING player is not a completed pass -- the ball was
                    # won/intercepted. Mark it incomplete immediately so the
                    # tackle/interception/high-turnover detectors downstream
                    # can fire (they all key on completed=False).
                    flip_to_opponent = (
                        track_data.player_teams
                        and from_team != "unknown"
                        and team != "unknown"
                        and from_team != team
                    )
                    events.append(
                        {
                            "type": "pass",
                            "timestamp": frame.timestamp,
                            "from_track_id": prev_possession,
                            "to_track_id": tid_now,
                            "completed": not flip_to_opponent,
                            "team": team,
                            "confidence": min(1.0, 1.0 - closest_dist / 200),
                            "metadata": pass_metadata,
                        }
                    )
                    if flip_to_opponent:
                        events[-1]["metadata"]["outcome"] = "lost_to_opponent"
                    pending_passes.append(
                        (
                            frame.timestamp,
                            frame.frame_number,
                            prev_possession,
                            tid_now,
                            len(events) - 1,
                        )
                    )
                    prev_possession = tid_now
                    pending_possession = None
                    pending_possession_frame = None
                elif pending_possession != tid_now:
                    # New (or third-player) candidate: start/restart pending.
                    pending_possession = tid_now
                    pending_possession_frame = frame.frame_number
                # else: same candidate still waiting within the window.
            elif prev_possession is None:
                # First possessor of the match -- nothing to pass from.
                prev_possession = tid_now
            # else: ball far from every player (in flight / loose) -- keep the
            # last possessor until someone re-establishes proximity.

            # ---- Pass completion look-ahead ----
            # Decide completion of recently-emitted passes: watch the next
            # pass_completion_window frames of possession. Receiver keeps the
            # ball -> completed. Possession moves straight to the other team
            # -> intercepted (completed=False, feeds the tackle detector).
            if pending_passes:
                still_open: list[tuple[float, int, int | None, int | None, int]] = []
                for p in pending_passes:
                    pts, pfno, passer, receiver, eidx = p
                    if frame.frame_number - pfno > pass_completion_window:
                        # Window elapsed with the receiver (apparently)
                        # retaining possession -- leave completed as-is.
                        continue
                    if tid_now == receiver:
                        # Receiver confirmed in possession.
                        continue
                    cur_team = (
                        track_data.player_teams.get(tid_now, "unknown")
                        if track_data.player_teams
                        else "unknown"
                    )
                    rcv_team = (
                        track_data.player_teams.get(receiver, "unknown")
                        if track_data.player_teams
                        else "unknown"
                    )
                    if (
                        receiver is not None
                        and tid_now != receiver
                        and cur_team != rcv_team
                        and cur_team != "unknown"
                        and rcv_team != "unknown"
                    ):
                        # A different player from a DIFFERENT team took the
                        # ball before the receiver settled it: interception.
                        events[eidx]["completed"] = False
                        events[eidx]["metadata"]["outcome"] = "intercepted"
                    # else: keep waiting; ball may just be in flight.
                    still_open.append(p)
                pending_passes = still_open

        return events

    def _compute_team_stats(
        self, players, events, track_data, homography_matrix=None
    ) -> dict[str, TeamStats]:
        home = TeamStats(team_name="Home")
        away = TeamStats(team_name="Away")

        for player in players.values():
            target = home if player.team == "home" else away
            target.distance_covered_km += player.distance_covered_m / 1000.0
            target.passes_completed += player.passes_completed
            target.passes_attempted += player.passes_attempted
            target.shots += player.shots
            target.tackles += player.tackles

        return {"home": home, "away": away}

    def _compute_possession(self, track_data, homography_matrix=None) -> dict[str, float]:
        home_frames = 0
        away_frames = 0
        unknown_frames = 0
        use_player_teams = bool(track_data.player_teams)

        # Frozen skip-frame copies duplicate the previous real frame's
        # detections verbatim -- counting them would multiply whatever team
        # held the ball there by the frame-skip factor. Only real detection
        # frames should count toward possession.
        _metrics = getattr(track_data, "tracking_metrics", None)
        frame_skip = 1
        if isinstance(_metrics, dict):
            try:
                frame_skip = max(1, int(_metrics.get("frame_skip", 1)))
            except (TypeError, ValueError):
                frame_skip = 1

        for frame in track_data.frames:
            if frame_skip > 1 and frame.frame_number % frame_skip != 0:
                continue

            ball_det = None
            player_dets = []

            for det in frame.detections:
                if det.class_name == "sports ball":
                    ball_det = det
                elif det.class_name == "person":
                    player_dets.append(det)

            if ball_det is None or not player_dets:
                continue

            bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2
            by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2

            closest = None
            closest_dist = float("inf")
            for p in player_dets:
                px = (p.bbox[0] + p.bbox[2]) / 2
                py = (p.bbox[1] + p.bbox[3]) / 2
                d = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
                if d < closest_dist:
                    closest_dist = d
                    closest = p

            if closest and closest.track_id is not None:
                # Possession attribution cap: the old rule attributed the
                # ball to the nearest player no matter how far away --
                # a ball 500px from everyone still "belonged" to someone.
                # Beyond this radius the ball is contested/in flight;
                # count the frame as unknown rather than fabricating
                # possession. ~150px at 720p is roughly 2.5m of pitch.
                if closest_dist > 150.0:
                    unknown_frames += 1
                    continue
                if use_player_teams:
                    team = track_data.player_teams.get(closest.track_id)
                    if team == "home":
                        home_frames += 1
                    elif team == "away":
                        away_frames += 1
                    else:
                        unknown_frames += 1
                # No team assignment at all: the old tid%2 parity split
                # fabricated a 50/50 possession number from track IDs that
                # have no relation to team membership. Leave both counters
                # untouched and report honestly below instead.

        total = home_frames + away_frames
        if total == 0:
            # No team-attributable possession evidence exists. Returning the
            # old hardcoded 50/50 here presented a fabricated number as a
            # measurement; report zero-information honestly instead.
            return {"home": 0.0, "away": 0.0, "unknown": 100.0}

        return {
            "home": (home_frames / total) * 100,
            "away": (away_frames / total) * 100,
        }

    def _compute_confidence(self, track_data, events) -> float:
        if track_data.total_frames == 0:
            return 0.0

        frames_with_ball = sum(
            1 for f in track_data.frames if any(d.class_name == "sports ball" for d in f.detections)
        )
        frames_with_players = sum(
            1 for f in track_data.frames if any(d.class_name == "person" for d in f.detections)
        )

        ball_pct = frames_with_ball / track_data.total_frames
        player_pct = frames_with_players / track_data.total_frames

        return min(1.0, (ball_pct * 0.4 + player_pct * 0.6))
