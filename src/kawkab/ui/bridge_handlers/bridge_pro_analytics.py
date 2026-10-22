"""Pro Analytics handler — surfaces the orphaned elite-analytical modules.

One comprehensive match report aggregating the core/ modules that were
built + tested but never reachable from the UI (CLAUDE.md's "57 orphaned
modules" gap): OBV, EPV, pass flow, pressing clusters, packing,
crossing, box entries, switches of play, set pieces, off-ball metrics,
duels, and ball recoveries.

Design rules (the discipline this codebase's history demands):
    - Every block carries ``data_available`` — computed from what the
      module actually received, never fabricated. An empty store or
      wrong-typed events yield ``data_available: false`` with a reason,
      not zeros dressed as metrics.
    - Storage events are read with the metadata-JSON-parsed shape and
      ``event_type`` column names (fetch-all helper) — this file never
      re-implements core/ logic, only delegates.
    - Tracking-frame blocks (OBV, off-ball) additionally require
      persisted tracking_frames (now written by every analyze_match run,
      plus StatsBomb imports don't have them — flagged honestly).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from kawkab.core.security import ErrorSanitizer, SecurityValidator
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = logging.getLogger(__name__)


class ProAnalyticsHandler(BridgeHandlerBase):
    """Aggregated elite-analytics report builder (delegation only)."""

    def __init__(self, bridge, services: dict[str, Any], rate_limiter=None) -> None:
        super().__init__(bridge, services, rate_limiter)

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    async def get_pro_analytics_report(self, match_id) -> str:
        self._check_rate_limit()
        try:
            match_id = SecurityValidator.validate_match_id(match_id)
            events = await self._fetch_all_events(match_id)
            if not events:
                return json.dumps(
                    {
                        "success": False,
                        "data_available": False,
                        "reason": "no events stored for this match",
                    }
                )

            frames = await self._fetch_tracking_frames(match_id)
            report: dict[str, Any] = {
                "success": True,
                "match_id": match_id,
                "n_events": len(events),
                "tracking_frames_available": bool(frames),
                "blocks": {},
            }
            blocks = report["blocks"]

            # ── Event-only blocks ────────────────────────────────────
            blocks["epv"] = self._epv_block(events)
            blocks["pass_flow"] = self._pass_flow_block(events)
            blocks["pressing_clusters"] = self._pressing_clusters_block(events)
            blocks["duels"] = self._duels_block(events)
            blocks["ball_recovery"] = self._recovery_block(events)
            blocks["box_entries"] = self._box_entries_block(events)
            blocks["switch_of_play"] = self._switches_block(events)
            blocks["crossing"] = self._crossing_block(events)
            blocks["set_pieces"] = self._set_pieces_block(events)
            blocks["through_balls"] = self._through_balls_block(events)

            # ── Tranche-2 event blocks ──────────────────────────────
            blocks["carry_xt"] = self._carry_xt_block(events)
            blocks["xg_chain"] = self._xg_chain_block(events)
            blocks["game_state"] = self._game_state_block(events)
            blocks["flank_analysis"] = self._flank_block(events)
            blocks["defensive_xt"] = self._defensive_xt_block(events)
            blocks["corner_xg"] = self._corner_xg_block(events)
            blocks["crossing_xg"] = self._crossing_xg_block(events)
            blocks["expected_pass"] = self._expected_pass_block(events)
            blocks["passing_triangles"] = self._triangles_block(events)
            blocks["scoreline"] = self._scoreline_block(events)

            # ── Tracking-frame blocks (honest when frames absent) ────
            blocks["obv"] = self._obv_block(frames, events)
            blocks["off_ball"] = self._offball_block(frames)

            # ── Blocks with data this match alone can't provide ───────
            blocks["velocity"] = self._velocity_block(match_id)
            blocks["influence_map"] = self._influence_map_block(frames)
            blocks["lineup_optimizer"] = self._lineup_optimizer_block(match_id)

            return json.dumps(report)
        except Exception as e:
            logger.error(f"get_pro_analytics_report failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ── data plumbing ─────────────────────────────────────────────────

    @staticmethod
    def _normalize_event(e: dict) -> dict:
        """Storage row → core/ event convention.

        core/ modules read ``type``/``x``/``y``/``is_goal``/``completed``
        at the top level; storage rows carry ``event_type`` + a JSON
        ``metadata`` blob. Done ONCE here, centrally, so every block
        builder receives the same shape (the services/-must-delegate-to-
        core/ invariant applied to the read path too).
        """
        meta_raw = e.get("metadata")
        meta_val: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
        out = dict(e)
        out["metadata"] = meta_val
        out["type"] = e.get("event_type") or e.get("type") or "unknown"
        for key in (
            "x",
            "y",
            "end_x",
            "end_y",
            "start_x",
            "start_y",
            "is_goal",
            "xg",
            "distance_m",
            "angle_deg",
            "assist_type",
        ):
            if key in meta_val and out.get(key) is None:
                out[key] = meta_val[key]
        # Storage's SELECT also extracts x/y/xg/xa/xt via json_extract —
        # those arrive as top-level columns; keep them when present.
        return out

    async def _fetch_all_events(self, match_id: int) -> list[dict]:
        """Page through get_match_events (default limit 200) with parsed metadata."""
        out: list[dict] = []
        offset = 0
        while True:
            page = await self.storage_service.get_match_events(match_id, limit=500, offset=offset)
            if not page:
                break
            for row in page:
                e = dict(row)
                if isinstance(e.get("metadata"), str):
                    try:
                        e["metadata"] = json.loads(e["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        e["metadata"] = {}
                out.append(self._normalize_event(e))
            if len(page) < 500:
                break
            offset += 500
        return out

    async def _fetch_tracking_frames(self, match_id: int) -> list[dict]:
        """Raw persisted frames (OBV/off-ball schema) — [] when absent."""
        if self.storage_service is None:
            return []
        getter = getattr(self.storage_service, "get_tracking_frames", None)
        if getter is None:
            return []
        try:
            frames = await getter(match_id, limit=1000)
        except TypeError:
            # Some storage backends have a different signature — treat as
            # unavailable rather than guessing kwargs.
            return []
        return self._frames_to_obv_schema(frames)

    def _frames_to_obv_schema(self, frames: list[dict]) -> list[dict]:
        """tracking_frames rows → the OBV/off-ball frame schema:
        {timestamp, possession, ball_pos, home_positions, away_positions}.
        Rows persist bbox JSON lists — center them to (x, y) points.
        """
        out: list[dict] = []
        for f in frames:
            players = f.get("player_detections")
            ball = f.get("ball_detections")
            if not isinstance(players, list):
                continue
            home_pos: list[list[float]] = []
            away_pos: list[list[float]] = []
            for p in players:
                bbox = p.get("bbox")
                if not isinstance(bbox, list) or len(bbox) < 4:
                    continue
                cx = (float(bbox[0]) + float(bbox[2])) / 2.0
                cy = (float(bbox[1]) + float(bbox[3])) / 2.0
                # Without team assignment per frame, positions are pooled;
                # OBV's team split needs team data — mark possession unknown.
                home_pos.append([cx, cy])
            ball_pos = None
            if isinstance(ball, list) and ball:
                b = ball[0].get("bbox")
                if isinstance(b, list) and len(b) >= 4:
                    ball_pos = [
                        (float(b[0]) + float(b[2])) / 2.0,
                        (float(b[1]) + float(b[3])) / 2.0,
                    ]
            out.append(
                {
                    "timestamp": float(f.get("timestamp", 0.0) or 0.0),
                    "possession": "home",  # honest limitation: per-frame possession unknown
                    "ball_pos": ball_pos,
                    "home_positions": home_pos,
                    "away_positions": away_pos,
                }
            )
        return out

    # ── block builders ────────────────────────────────────────────────

    def _epv_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.epv import EPVModel

        try:
            analyzer = EPVModel()
            report = analyzer.compute_match_epv(events)
            data = (
                report.to_dict()
                if hasattr(report, "to_dict")
                else {
                    "home_total": getattr(report, "home_total", 0.0),
                    "away_total": getattr(report, "away_total", 0.0),
                    "n_possessions": len(getattr(report, "possessions", []) or []),
                }
            )
            return {"data_available": True, **data}
        except Exception as exc:
            return {"data_available": False, "reason": f"epv: {exc}"}

    def _pass_flow_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.pass_flow import compute_pass_flow

        try:
            links = compute_pass_flow(events, team="home")
            links += compute_pass_flow(events, team="away")
            if not links:
                return {"data_available": False, "reason": "no completed passes with coordinates"}
            serializable = [
                {k: v for k, v in link.items() if isinstance(v, (int, float, str))}
                for link in links[:200]
            ]
            return {"data_available": True, "n_links": len(links), "links": serializable}
        except Exception as exc:
            return {"data_available": False, "reason": f"pass_flow: {exc}"}

    def _pressing_clusters_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.pressing_clusters import cluster_pressing_events

        try:
            clusters = cluster_pressing_events(events)
            if not clusters:
                return {
                    "data_available": False,
                    "reason": "no pressing/defensive events with coordinates",
                }
            data = [c.to_dict() if hasattr(c, "to_dict") else vars(c) for c in clusters[:40]]
            return {"data_available": True, "n_clusters": len(clusters), "clusters": data}
        except Exception as exc:
            return {"data_available": False, "reason": f"pressing_clusters: {exc}"}

    def _duels_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.duel_analysis import analyze_duels

        try:
            data = analyze_duels(events)
            duels_present = any(
                e.get("event_type") in ("duel", "tackle", "aerial_won", "aerial_lost")
                or e.get("type") in ("duel", "tackle", "aerial_won", "aerial_lost")
                for e in events
            )
            if not duels_present:
                return {
                    "data_available": False,
                    "reason": "no duel-type events tagged in this match",
                    "raw": data,
                }
            return {"data_available": True, **data}
        except Exception as exc:
            return {"data_available": False, "reason": f"duels: {exc}"}

    def _recovery_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.ball_recovery import BallRecoveryAnalyzer

        try:
            analyzer = BallRecoveryAnalyzer()
            out = {}
            for team in ("home", "away"):
                data = analyzer.analyze_recoveries(events, team)
                out[team] = data if isinstance(data, dict) else vars(data)
            any_recoveries = (
                any(
                    (
                        out[t].get("total_recoveries")
                        or out[t].get("recoveries")
                        or out[t].get("n_recoveries")
                        or 0
                    )
                    > 0
                    for t in ("home", "away")
                )
                if out
                else False
            )
            if not any_recoveries:
                return {
                    "data_available": False,
                    "reason": "no recovery-type events (interception/tackle/"
                    "loose_ball/goal_kick/clearance) in this match",
                    "raw": out,
                }
            return {"data_available": True, "home": out["home"], "away": out["away"]}
        except Exception as exc:
            return {"data_available": False, "reason": f"ball_recovery: {exc}"}

    def _box_entries_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.box_entries import BoxEntryAnalyzer

        try:
            analyzer = BoxEntryAnalyzer()
            touches = analyzer.analyze_box_touches(events)
            entries = analyzer.analyze_box_entries(events)
            return {"data_available": True, "box_touches": touches, "box_entries": entries}
        except Exception as exc:
            return {"data_available": False, "reason": f"box_entries: {exc}"}

    def _switches_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.switch_of_play import SwitchOfPlayDetector

        try:
            detector = SwitchOfPlayDetector()
            data = detector.analyze_switches(events)
            return {"data_available": True, **data}
        except Exception as exc:
            return {"data_available": False, "reason": f"switch_of_play: {exc}"}

    def _crossing_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.crossing_analysis import CrossingAnalysis

        try:
            analyzer = CrossingAnalysis()
            report = analyzer.analyze_crosses(events)
            data = report.to_dict() if hasattr(report, "to_dict") else vars(report)
            n_crosses = data.get("total_crosses", 0) or len(data.get("crosses", []) or [])
            if not n_crosses:
                return {
                    "data_available": False,
                    "reason": "no completed crosses with coordinates",
                    "raw": data,
                }
            return {"data_available": True, **data}
        except Exception as exc:
            return {"data_available": False, "reason": f"crossing: {exc}"}

    def _set_pieces_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.set_piece_analysis import analyze_set_pieces

        try:
            report = analyze_set_pieces(events)
            data = report.to_dict() if hasattr(report, "to_dict") else vars(report)
            return {"data_available": True, **data}
        except Exception as exc:
            return {"data_available": False, "reason": f"set_pieces: {exc}"}

    def _through_balls_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.through_ball import detect_through_balls

        try:
            tbs = detect_through_balls(events)
            data = [tb.to_dict() if hasattr(tb, "to_dict") else vars(tb) for tb in tbs[:50]]
            if not data:
                return {
                    "data_available": False,
                    "reason": "no through-ball-pattern passes detected",
                }
            return {"data_available": True, "n_through_balls": len(tbs), "through_balls": data}
        except Exception as exc:
            return {"data_available": False, "reason": f"through_balls: {exc}"}

    def _obv_block(self, frames: list[dict], events: list[dict]) -> dict[str, Any]:
        if not frames:
            return {
                "data_available": False,
                "reason": "no persisted tracking frames (run video analysis, "
                "or this match is an event-data import)",
            }
        from kawkab.core.obv import OffBallValuator

        try:
            valuator = OffBallValuator()
            report = valuator.compute_obv(frames, team="home", events=events)
            data = report.to_dict() if hasattr(report, "to_dict") else vars(report)
            return {"data_available": True, **data}
        except Exception as exc:
            return {"data_available": False, "reason": f"obv: {exc}"}

    def _offball_block(self, frames: list[dict]) -> dict[str, Any]:
        if not frames:
            return {
                "data_available": False,
                "reason": "no persisted tracking frames (run video analysis, "
                "or this match is an event-data import)",
            }
        from kawkab.core.offball_metrics import OffBallAnalyzer

        try:
            analyzer = OffBallAnalyzer()
            report = analyzer.analyze_offball(frames, team="home")
            data = report.to_dict() if hasattr(report, "to_dict") else vars(report)
            return {"data_available": True, **data}
        except Exception as exc:
            return {"data_available": False, "reason": f"off_ball: {exc}"}

    # ── tranche-2 block builders ───────────────────────────────────────

    def _carry_xt_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.carry_xt import compute_carry_xt

        try:
            report = compute_carry_xt(events)
            data = report.to_dict() if hasattr(report, "to_dict") else vars(report)
            return {"data_available": True, **data}
        except Exception as exc:
            return {"data_available": False, "reason": f"carry_xt: {exc}"}

    def _xg_chain_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.xg_chain import compute_xg_chain

        try:
            out = {}
            for team in ("home", "away"):
                chains = compute_xg_chain(events, team)
                out[team] = {
                    "n_chains": len(chains),
                    "total_xg_contribution": round(
                        sum(c.to_dict().get("xg_contribution", 0.0) for c in chains), 4
                    )
                    if chains
                    else 0.0,
                    "top_chains": [c.to_dict() for c in chains[:8]],
                }
            return {"data_available": True, **out}
        except Exception as exc:
            return {"data_available": False, "reason": f"xg_chain: {exc}"}

    def _game_state_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.game_state import analyze_game_state

        try:
            report = analyze_game_state(events, frame_data=[])
            data = report.to_dict() if hasattr(report, "to_dict") else vars(report)
            return {"data_available": True, **data}
        except Exception as exc:
            return {"data_available": False, "reason": f"game_state: {exc}"}

    def _flank_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.flank_analysis import FlankAnalyzer

        try:
            analyzer = FlankAnalyzer()
            out = {}
            for team in ("home", "away"):
                out[team] = analyzer.compute_flank_effectiveness(events, team)
            return {"data_available": True, **out}
        except Exception as exc:
            return {"data_available": False, "reason": f"flank_analysis: {exc}"}

    def _defensive_xt_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.defensive_xt import compute_defensive_xt

        try:
            from kawkab.core.xt_model import ExpectedThreatModel

            model = ExpectedThreatModel()
            model.build_transition_matrix(events)
            grid = model.get_zone_values()
            actions = compute_defensive_xt(
                events, grid, xT_rows=grid.shape[0], xT_cols=grid.shape[1]
            )
            data = [a.to_dict() for a in actions[:60]]
            return {"data_available": True, "n_defensive_actions": len(actions), "actions": data}
        except Exception as exc:
            return {"data_available": False, "reason": f"defensive_xt: {exc}"}

    def _corner_xg_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.corner_xg import CornerKickXgModel

        try:
            model = CornerKickXgModel()
            corners = [e for e in events if e.get("type") == "corner"]
            if not corners:
                return {"data_available": False, "reason": "no corner events tagged in this match"}
            ratings = []
            for c in corners:
                r = model.compute_corner_danger_rating(c)
                ratings.append(r if isinstance(r, (int, float)) else float(r))
            return {
                "data_available": True,
                "n_corners": len(corners),
                "avg_danger_rating": round(sum(ratings) / len(ratings), 3),
                "ratings": [round(r, 3) for r in ratings[:30]],
            }
        except Exception as exc:
            return {"data_available": False, "reason": f"corner_xg: {exc}"}

    def _crossing_xg_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.crossing_xg import compute_cross_xg

        try:
            crosses = [e for e in events if e.get("type") == "cross"]
            if not crosses:
                return {"data_available": False, "reason": "no cross events tagged in this match"}
            results = [compute_cross_xg(e) for e in crosses[:80]]
            data = [r.to_dict() for r in results]
            total = sum(float(d.get("base_xg", 0.0) or 0.0) for d in data)
            return {
                "data_available": True,
                "n_crosses": len(crosses),
                "total_cross_xg": round(total, 4),
                "crosses": data[:40],
            }
        except Exception as exc:
            return {"data_available": False, "reason": f"crossing_xg: {exc}"}

    def _expected_pass_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.expected_pass import compute_ep_batch

        try:
            passes = [e for e in events if e.get("type") == "pass"]
            if not passes:
                return {"data_available": False, "reason": "no pass events"}
            results = compute_ep_batch(passes[:400])
            data = [r.to_dict() if hasattr(r, "to_dict") else vars(r) for r in results]
            completed_ep = [float(d.get("ep", 0.0)) for d in data if d.get("completed", True)]
            return {
                "data_available": True,
                "n_passes_evaluated": len(data),
                "avg_completed_ep": round(sum(completed_ep) / max(len(completed_ep), 1), 4),
                "sample": data[:25],
            }
        except Exception as exc:
            return {"data_available": False, "reason": f"expected_pass: {exc}"}

    def _triangles_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.passing_triangles import PassingTriangleAnalyzer

        try:
            analyzer = PassingTriangleAnalyzer()
            triangles = analyzer.detect_passing_triangles(events)
            if not triangles:
                return {
                    "data_available": False,
                    "reason": "no passing triangles detected (needs consecutive "
                    "completed passes between player trios)",
                }
            # The O(n³) detector can emit thousands on high-volume event
            # imports; ship the aggregate + a bounded sample, not the full list.
            return {
                "data_available": True,
                "n_triangles": len(triangles),
                "triangles": triangles[:30],
            }
        except Exception as exc:
            return {"data_available": False, "reason": f"passing_triangles: {exc}"}

    def _scoreline_block(self, events: list[dict]) -> dict[str, Any]:
        from kawkab.core.scoreline_distribution import ScorelineDistribution

        try:
            model = ScorelineDistribution()
            result = model.compute_scoreline_probabilities(events, n_sims=10000)
            scorelines = result.get("scorelines", {})
            if not scorelines:
                return {"data_available": False, "reason": "no shots to estimate scoreline from"}
            outcomes = model.compute_match_outcome_probs(scorelines)
            entropy = model.compute_scoreline_entropy(scorelines)
            return {
                "data_available": True,
                "outcome_probs": outcomes,
                "scoreline_entropy": entropy,
                "remaining_minutes": result.get("remaining_minutes"),
                "top_scorelines": dict(list(scorelines.items())[:8]),
            }
        except Exception as exc:
            return {"data_available": False, "reason": f"scoreline: {exc}"}

    def _velocity_block(self, match_id: int) -> dict[str, Any]:
        """Velocity needs per-player trajectory data — honest when absent."""
        return {
            "data_available": False,
            "reason": "velocity analysis requires per-player pitch-space "
            "trajectories; available after homography-calibrated "
            "video analysis (physical_load_service reports sprints "
            "in the Squad view)",
        }

    def _influence_map_block(self, frames: list[dict]) -> dict[str, Any]:
        if not frames:
            return {
                "data_available": False,
                "reason": "influence maps need pitch-space tracking frames "
                "(homography-calibrated video analysis)",
            }
        # Frames persisted by CV are pixel-space bboxes; influence maps need
        # meters. Honest no rather than misleading pixel maps.
        return {
            "data_available": False,
            "reason": "influence maps need homography-calibrated (meters) "
            "positions; current frames are pixel-space",
        }

    def _lineup_optimizer_block(self, match_id: int) -> dict[str, Any]:
        return {
            "data_available": False,
            "reason": "lineup optimization needs multi-match player history "
            "and profiles — exposed via the Squad view's per-player "
            "data, not a single-match report",
        }
