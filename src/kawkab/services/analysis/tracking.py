"""Tracking analysis mixin — player stats, ratings, PPDA, formation detection."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

from kawkab.core.game_constants import GAME
from kawkab.core.logging import get_logger
from kawkab.core.player_rating import PlayerRating, compute_rating

logger = get_logger(__name__)


class TrackingMixin:
    # Provided by the composing AnalysisService (AnalysisServiceCore.__init__);
    # declared here so mypy sees the assembled surface per mixin.
    pitch_width: float
    use_kalman: bool

    if TYPE_CHECKING:
        # Kalman variant lives on the composing AnalysisServiceCore; declared
        # so the mixin's surface is visible to mypy (no runtime effect).
        def _compute_player_stats_kalman(
            self,
            track_data: Any,
            homography_matrix: Any,
            max_frame_delta_m: float,
        ) -> dict[int, dict[str, Any]]: ...

    def _compute_player_stats(
        self, track_data, homography_matrix=None, track_merge_map: dict[int, int] | None = None
    ):
        from .core import PlayerStats

        def _resolve(tid):
            """Apply merge map to resolve stitched track IDs."""
            if track_merge_map and tid in track_merge_map:
                return track_merge_map[tid]
            return tid

        players: dict[int, PlayerStats] = {}
        prev_positions: dict[int, tuple[float, float]] = {}
        max_speed_per_player: dict[int, float] = {}

        _ = track_data.fps
        # Uncalibrated-distance approximation. The old 720.0/self.pitch_width
        # constant assumed 720p frames with the pitch spanning the full frame
        # height -- wrong by ~1.5x on 1080p sources (and more on zoomed shots).
        # Derive the scale from the actual frame height when frames carry it;
        # otherwise fall back to the project-wide CARRY_PIXEL_TO_METER_RATIO.
        from kawkab.core.game_constants import GAME

        _frame_h = next(
            (f.image_height for f in track_data.frames if getattr(f, "image_height", 0)),
            0,
        )
        if _frame_h:
            pixels_per_meter = _frame_h / self.pitch_width
        else:
            pixels_per_meter = 1.0 / GAME.CARRY_PIXEL_TO_METER_RATIO

        # Read merge map from tracking_metrics if available and not explicitly provided
        if track_merge_map is None:
            metrics = getattr(track_data, "tracking_metrics", {}) or {}
            raw_map = metrics.get("stitch_merge_map", {})
            track_merge_map = {int(k): v for k, v in raw_map.items()} if raw_map else {}

        frame_skip = max(1, track_data.tracking_metrics.get("frame_skip", 1))

        max_frame_delta_m = 0.5 if homography_matrix is not None else 25.0

        use_kalman = (
            self.use_kalman
            and getattr(track_data, "match_type", None) == "full_match"
            and homography_matrix is not None
        )
        if use_kalman:
            logger.info("Using Kalman smoother for full-match distance/speed")
            return self._compute_player_stats_kalman(
                track_data, homography_matrix, max_frame_delta_m
            )

        prev_timestamps: dict[int, float] = {}

        def is_skip_frame(fno: int) -> bool:
            return fno % frame_skip != 0

        for frame in track_data.frames:
            current_positions: dict[int, tuple[float, float]] = {}
            ts = frame.timestamp
            frame_is_skipped = is_skip_frame(frame.frame_number)

            for det in frame.detections:
                if det.class_name != "person" or det.track_id is None:
                    continue

                tid = _resolve(det.track_id)
                if tid not in players:
                    players[tid] = PlayerStats(track_id=tid)
                    if track_data.player_teams:
                        players[tid].team = track_data.player_teams.get(tid, "unknown")

                x1, y1, x2, y2 = det.bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if homography_matrix is not None:
                    cx, cy = homography_matrix.pixel_to_pitch(cx, cy)

                current_positions[tid] = (cx, cy)
                players[tid].positions.append((ts, cx, cy))

                if frame_is_skipped:
                    continue

                if tid in prev_positions:
                    px, py = prev_positions[tid]
                    dx = cx - px
                    dy = cy - py

                    if homography_matrix is not None:
                        meters = math.sqrt(dx * dx + dy * dy)
                    else:
                        pixel_distance = math.sqrt(dx * dx + dy * dy)
                        meters = pixel_distance / pixels_per_meter

                    if meters > max_frame_delta_m:
                        meters = 0.0

                    players[tid].distance_covered_m += meters

                    if meters > 0 and tid in prev_timestamps:
                        dt = ts - prev_timestamps[tid]
                        if dt > 0:
                            speed_mps = meters / dt
                            speed_kmh = speed_mps * 3.6
                            if speed_kmh <= 36.0:
                                max_speed_per_player[tid] = max(
                                    max_speed_per_player.get(tid, 0), speed_kmh
                                )

                prev_timestamps[tid] = ts

            if not frame_is_skipped:
                prev_positions = current_positions

        for tid, player in players.items():
            if track_data.duration_seconds > 0:
                player.avg_speed_kmh = player.distance_covered_m / track_data.duration_seconds * 3.6
            player.max_speed_kmh = max_speed_per_player.get(tid, 0.0)

        return players

    def _compute_player_ratings(
        self, players, typed_events, pitch_control, track_data, homography_matrix=None
    ):

        ratings: dict[int, PlayerRating] = {}

        event_by_player: dict[int, dict] = {}
        for ev in typed_events:
            tid = ev.track_id
            if tid is None:
                continue
            if tid not in event_by_player:
                event_by_player[tid] = {
                    "passes": 0,
                    "completed_passes": 0,
                    "progressive": 0,
                    "key_passes": 0,
                    "assists": 0,
                    "shots": 0,
                    "sot": 0,
                    "tackles": 0,
                    "interceptions": 0,
                    "carries": 0,
                    "prog_carries": 0,
                    "goals": 0.0,
                    "xg": 0.0,
                }
            d = event_by_player[tid]
            from kawkab.core.events import CarryEvent, PassEvent, ShotEvent

            if isinstance(ev, PassEvent):
                d["passes"] += 1
                if ev.completed:
                    d["completed_passes"] += 1
                if ev.is_progressive:
                    d["progressive"] += 1
                if ev.is_key_pass:
                    d["key_passes"] += 1
                if ev.is_assist:
                    d["assists"] += 1
            elif isinstance(ev, ShotEvent):
                d["shots"] += 1
                if ev.on_target:
                    d["sot"] += 1
                d["xg"] += ev.xg
            elif isinstance(ev, CarryEvent):
                d["carries"] += 1
                if ev.is_progressive:
                    d["prog_carries"] += 1

        for tid, player in players.items():
            estats = event_by_player.get(tid, {})

            avg_x = None
            if player.positions:
                avg_x = sum(p[1] for p in player.positions) / len(player.positions)

            passes_from_events = estats.get("passes", 0)
            completed_passes = estats.get("completed_passes", 0)
            pass_acc = (
                completed_passes / max(passes_from_events, 1) if passes_from_events > 0 else 0.0
            )
            rating = compute_rating(
                pass_accuracy=pass_acc,
                passes_completed=completed_passes,
                passes_attempted=passes_from_events,
                progressive_passes=estats.get("progressive", 0),
                key_passes=estats.get("key_passes", 0),
                assists=estats.get("assists", 0),
                shots=estats.get("shots", 0),
                shots_on_target=estats.get("sot", 0),
                goals=estats.get("goals", 0.0),
                xg=estats.get("xg", 0.0),
                tackles=player.tackles,
                interceptions=player.interceptions,
                defensive_actions=player.tackles + player.interceptions,
                carries=estats.get("carries", 0),
                progressive_carries=estats.get("prog_carries", 0),
                distance_covered_m=player.distance_covered_m,
                max_speed_kmh=player.max_speed_kmh,
                minutes_played=track_data.duration_seconds / 60.0
                if track_data.duration_seconds > 0
                else 90.0,
                avg_x=avg_x,
                position=None,
                pitch_length=self.pitch_length,
            )

            ratings[tid] = rating

        return ratings

    def compute_ppda(self, track_data, team="home", homography_matrix=None):
        if not track_data.frames:
            return {
                "ppda": None,
                "intensity": "unknown",
                "passes": 0,
                "defensive_actions": 0,
            }

        press_threshold_m = GAME.PRESS_THRESHOLD_M
        possession_change_threshold_m = GAME.POSSESSION_CHANGE_DIST_M
        sample_every = 2

        n_passes = 0
        n_defensive_actions = 0
        prev_possessor_track_id = None
        prev_ball_pos = None
        team_possession_frames = 0
        opp_possession_frames = 0

        for i, frame in enumerate(track_data.frames):
            if i % sample_every != 0:
                continue

            ball_det = None
            team_players = []
            opp_players = []
            for det in frame.detections:
                if det.class_name == "sports ball" and ball_det is None:
                    ball_det = det
                elif (
                    det.class_name == "person"
                    and det.track_id is not None
                    and track_data.player_teams
                ):
                    assigned = track_data.player_teams.get(det.track_id)
                    if assigned == "home":
                        (team_players if team == "home" else opp_players).append(det)
                    elif assigned == "away":
                        (team_players if team == "away" else opp_players).append(det)
                    # No team assignment: skip this player rather than
                    # inventing one from track-ID parity (tid % 2 has no
                    # relation to team membership and silently corrupted
                    # PPDA/formation stats).

            if ball_det is None or not team_players or not opp_players:
                continue

            bx = (ball_det.bbox[0] + ball_det.bbox[2]) / 2
            by = (ball_det.bbox[1] + ball_det.bbox[3]) / 2

            if homography_matrix is not None:
                try:
                    bx_m, by_m = homography_matrix.pixel_to_pitch(bx, by)
                except Exception:
                    bx_m, by_m = bx, by
            else:
                bx_m, by_m = bx, by

            # Bind loop values as defaults: this closure is only used within
            # this iteration, but default-arg binding removes the late-binding
            # trap entirely (B023).
            def _opp_dist_pitch(p, bx_m=bx_m, by_m=by_m, bx=bx, by=by):
                ox = (p.bbox[0] + p.bbox[2]) / 2
                oy = (p.bbox[1] + p.bbox[3]) / 2
                if homography_matrix is not None:
                    try:
                        ox_m, oy_m = homography_matrix.pixel_to_pitch(ox, oy)
                        return math.sqrt((ox_m - bx_m) ** 2 + (oy_m - by_m) ** 2)
                    except Exception:
                        pass
                return math.sqrt((ox - bx) ** 2 + (oy - by) ** 2)

            closest_opp_dist = min(_opp_dist_pitch(p) for p in opp_players)

            closest_team_player = min(
                team_players,
                key=lambda p: _opp_dist_pitch(p),
            )
            current_possessor_track_id = closest_team_player.track_id

            if team == "home":
                if track_data.player_teams.get(current_possessor_track_id) == "home":
                    team_possession_frames += 1
                else:
                    opp_possession_frames += 1
            else:
                if track_data.player_teams.get(current_possessor_track_id) == "away":
                    team_possession_frames += 1
                else:
                    opp_possession_frames += 1

            if (
                prev_possessor_track_id is not None
                and current_possessor_track_id != prev_possessor_track_id
            ) and closest_opp_dist < press_threshold_m:
                n_defensive_actions += 1
                if prev_ball_pos is not None:
                    pbx, pby = prev_ball_pos
                    if homography_matrix is not None:
                        try:
                            bx_m, by_m = homography_matrix.pixel_to_pitch(bx, by)
                            pbx_m, pby_m = homography_matrix.pixel_to_pitch(pbx, pby)
                            ball_moved_m = math.sqrt((bx_m - pbx_m) ** 2 + (by_m - pby_m) ** 2)
                        except Exception:
                            ball_moved_m = math.sqrt((bx - pbx) ** 2 + (by - pby) ** 2)
                    else:
                        ball_moved_m = math.sqrt((bx - pbx) ** 2 + (by - pby) ** 2)
                    if ball_moved_m > possession_change_threshold_m:
                        n_passes += 1

            prev_possessor_track_id = current_possessor_track_id
            prev_ball_pos = (bx, by)

        if n_defensive_actions == 0:
            return {
                "ppda": None,
                "intensity": "unknown",
                "passes": n_passes,
                "defensive_actions": n_defensive_actions,
                "possession_pct": (
                    team_possession_frames / (team_possession_frames + opp_possession_frames) * 100
                    if (team_possession_frames + opp_possession_frames) > 0
                    else 50.0
                ),
            }

        ppda = n_passes / n_defensive_actions

        if ppda < 8:
            intensity = "high_press"
        elif ppda < 13:
            intensity = "medium_press"
        else:
            intensity = "low_press"

        return {
            "ppda": round(ppda, 2),
            "intensity": intensity,
            "passes": n_passes,
            "defensive_actions": n_defensive_actions,
            "possession_pct": round(
                team_possession_frames / (team_possession_frames + opp_possession_frames) * 100, 1
            )
            if (team_possession_frames + opp_possession_frames) > 0
            else 50.0,
        }

    def detect_formation(self, track_data, team="home", n_players=11, homography_matrix=None):

        team_player_positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
        track_first_seen: dict[int, float] = {}
        track_last_seen: dict[int, float] = {}
        use_player_teams = bool(track_data.player_teams)

        for frame in track_data.frames:
            ts = frame.timestamp
            for det in frame.detections:
                if det.class_name != "person" or det.track_id is None:
                    continue
                if use_player_teams:
                    assigned = track_data.player_teams.get(det.track_id)
                    if assigned is None:
                        continue
                    if team == "home" and assigned != "home":
                        continue
                    if team == "away" and assigned != "away":
                        continue
                else:
                    # No team assignment: this detection carries no reliable
                    # team evidence. The old tid%2 parity split fabricated
                    # half a formation out of arbitrary track IDs; skipping
                    # yields an honest "formation unknown" instead.
                    continue
                x1, y1, x2, y2 = det.bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if homography_matrix is not None:
                    cx, cy = homography_matrix.pixel_to_pitch(cx, cy)

                tid = det.track_id
                team_player_positions[tid].append((cx, cy))
                if tid not in track_first_seen:
                    track_first_seen[tid] = ts
                track_last_seen[tid] = ts

        min_span_seconds = max(3.0, track_data.duration_seconds * 0.05)
        team_player_positions = {
            tid: positions
            for tid, positions in team_player_positions.items()
            if (track_last_seen.get(tid, 0) - track_first_seen.get(tid, 0)) >= min_span_seconds
        }
        track_lifetimes = {
            tid: track_last_seen[tid] - track_first_seen[tid] for tid in team_player_positions
        }
        logger.debug(
            f"Formation team={team}: {len(team_player_positions)} tracks pass "
            f"span>={min_span_seconds:.1f}s filter"
        )

        if len(team_player_positions) < 5:
            return {
                "formation": "unknown",
                "confidence": 0.0,
                "line_height": None,
                "line_height_m": None,
                "player_count": len(team_player_positions),
                "coordinates": "meters" if homography_matrix else "pixels",
            }

        avg_positions = {
            tid: (
                sum(p[0] for p in positions) / len(positions),
                sum(p[1] for p in positions) / len(positions),
            )
            for tid, positions in team_player_positions.items()
        }

        outfield_n = n_players - 1
        max_keep = outfield_n + 3

        if len(avg_positions) > max_keep:
            sorted_by_lifetime = sorted(
                avg_positions.items(),
                key=lambda x: track_lifetimes[x[0]],
                reverse=True,
            )
            avg_positions = dict(sorted_by_lifetime[:max_keep])

        sorted_by_x = sorted(avg_positions.items(), key=lambda x: x[1][0])
        n = len(sorted_by_x)

        if n < 6:
            return {"formation": "unknown", "confidence": 0.0, "player_count": n}

        x_coords = [pos[1][0] for pos in sorted_by_x]
        centroids = sorted([x_coords[int(i * n / 3)] for i in range(3)])
        prev_labels = None
        for _ in range(20):
            labels = [min(range(3), key=lambda k: abs(x - centroids[k])) for x in x_coords]
            if labels == prev_labels:
                break
            prev_labels = labels
            for k in range(3):
                members = [x_coords[i] for i, lab in enumerate(labels) if lab == k]
                if members:
                    centroids[k] = sum(members) / len(members)
            centroids.sort()

        n_def = max(2, min(5, labels.count(0)))
        n_mid = max(2, min(6, labels.count(1)))
        n_att = max(1, min(4, labels.count(2)))

        defenders = [sorted_by_x[i][0] for i, lab in enumerate(labels) if lab == 0]
        midfielders = [sorted_by_x[i][0] for i, lab in enumerate(labels) if lab == 1]
        attackers = [sorted_by_x[i][0] for i, lab in enumerate(labels) if lab == 2]

        def_line_height = (
            sum(avg_positions[t][0] for t in defenders) / len(defenders) if defenders else 0
        )

        if homography_matrix is not None:
            def_line_height_m = round(def_line_height, 2)
            def_line_pct = def_line_height / homography_matrix.pitch_length_m
        elif def_line_height > 1.0:
            def_line_height_m = None
            def_line_pct = def_line_height / self.pitch_length
        else:
            def_line_height_m = None
            def_line_pct = def_line_height

        formation_str = f"{n_def}-{n_mid}-{n_att}"

        valid_formations = {
            "4-3-3",
            "4-4-2",
            "4-2-3-1",
            "3-5-2",
            "3-4-3",
            "5-3-2",
            "5-4-1",
            "4-1-4-1",
            "4-5-1",
            "3-4-1-2",
            "3-6-1",
            "4-3-1-2",
            "4-1-3-2",
            "4-4-1-1",
            "5-2-3",
            "3-5-1-1",
            "4-2-2-2",
            "4-3-2-1",
            "4-1-2-3",
        }
        confidence = (0.7 if n >= 8 else 0.4) if formation_str in valid_formations else 0.3

        return {
            "formation": formation_str,
            "confidence": confidence,
            "line_height": round(def_line_pct, 3),
            "line_height_m": def_line_height_m,
            "player_count": n,
            "defenders": defenders,
            "midfielders": midfielders,
            "attackers": attackers,
            "coordinates": "meters" if homography_matrix else "pixels",
        }

    def track_formations(self, track_data, window_minutes=5):
        if not hasattr(track_data, "frames") or not track_data.frames:
            return {"home_timeline": [], "away_timeline": [], "changes": 0}
        fps = max(1, getattr(track_data, "fps", 30))
        window_frames = int(window_minutes * 60 * fps)
        home_timeline = []
        away_timeline = []
        for start in range(0, len(track_data.frames), window_frames):
            end = min(start + window_frames, len(track_data.frames))
            window = track_data.frames[start:end]
            home_formation = self._classify_formation_in_window(window, "home")
            away_formation = self._classify_formation_in_window(window, "away")
            ts = getattr(window[0], "timestamp", 0.0) if window else 0.0
            home_timeline.append({"minute": round(ts / 60.0, 1), "formation": home_formation})
            away_timeline.append({"minute": round(ts / 60.0, 1), "formation": away_formation})
        changes = 0
        if home_timeline:
            prev = home_timeline[0]["formation"]
            for entry in home_timeline[1:]:
                if entry["formation"] != prev:
                    changes += 1
                    prev = entry["formation"]
        return {
            "home_timeline": home_timeline,
            "away_timeline": away_timeline,
            "changes": changes,
        }

    def _classify_formation_in_window(self, frames, team):
        if not frames:
            return "unknown"
        for frame in frames:
            detections = getattr(frame, "detections", []) or []
            team_dets = [
                d
                for d in detections
                if getattr(d, "team", None) == team and not getattr(d, "is_ball", False)
            ]
            if len(team_dets) >= 10:
                return self._detect_formation(team_dets)
        return "unknown"

    def _detect_formation(self, detections):

        thirds = Counter()
        for d in detections:
            x = getattr(d, "x", None) or 0
            if hasattr(d, "bbox") and d.bbox is not None:
                try:
                    x = d.bbox.cx
                except AttributeError:
                    x = 0
            third = "D" if x < 35 else ("M" if x < 70 else "A")
            thirds[third] += 1
        d_count = thirds["D"]
        m_count = thirds["M"]
        a_count = thirds["A"]
        if d_count == 0 and m_count == 0 and a_count == 0:
            return "unknown"
        return f"{d_count}-{m_count}-{a_count}"
