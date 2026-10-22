"""Handlers extracted from bridge_analysis.py (Phase C1 split).

Thin Qt-bridge handlers: each owns one surface and forwards to its
service layer. Registered in bridge_handlers/__init__.py and wired in
ui/bridge.py. Validation lives in the service layer unless noted.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from kawkab.core.security import ErrorSanitizer, SecurityValidator
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = logging.getLogger(__name__)


class MatchIntelHandler(BridgeHandlerBase):
    """Pro-analytics leftovers: pitch control, pass sonar, roles, dominance,
    goals added, xG chain, xA/pressing reports."""

    def __init__(self, bridge, services: dict[str, Any], rate_limiter=None) -> None:
        super().__init__(bridge, services, rate_limiter)
        self.bridge = bridge  # public alias kept for backward compatibility

    # _check_rate_limit inherited from BridgeHandlerBase ("analysis" bucket).
    # History: the deleted override called a nonexistent RateLimiter.check(),
    # which raised AttributeError out of every rate-checked slot once the
    # real bridge limiter was wired in (caught by the GUI e2e audit).

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    @staticmethod
    def _normalize_event_types(events: list[dict]) -> list[dict]:
        """Map storage rows to the analytics models' event convention.

        StorageService.get_match_events returns rows keyed by event_type
        (the DB column), while ExpectedAssistModel and
        PressingEfficiencyAnalyzer filter on a plain "type" key. Without
        this mapping both reports silently computed zeros against every
        real match -- visible only with data shaped like the real rows,
        never with the earlier {"type": ...} test fixtures.
        """
        for ev in events:
            if "type" not in ev and ev.get("event_type") is not None:
                ev["type"] = ev["event_type"]
        return events

    async def get_pitch_control_overlay(self, match_id: str) -> str:
        """Compute pitch control grid overlay for a match.

        Returns JSON with home_grid, away_grid, ball_control_pct, hot_zones.
        """
        try:
            from kawkab.core.pitch_control import VoronoiPitchControl

            events = (
                await self.storage_service.get_match_events(match_id)
                if self.storage_service
                else []
            )
            events = self._normalize_event_types(events)
            if not events:
                return json.dumps(
                    {"home_grid": [], "away_grid": [], "ball_control_pct": 50.0, "hot_zones": []}
                )

            home_events = [
                e for e in events if e.get("team") == "home" and e.get("start_x") is not None
            ]
            away_events = [
                e for e in events if e.get("team") == "away" and e.get("start_x") is not None
            ]
            home_positions = [
                (e.get("start_x", 52.5), e.get("start_y", 34.0)) for e in home_events[:11]
            ]
            away_positions = [
                (e.get("start_x", 52.5), e.get("start_y", 34.0)) for e in away_events[:11]
            ]

            pc = VoronoiPitchControl()
            frame = pc.compute_frame_control(home_positions, away_positions)

            hot_zones: list = []
            import numpy as np

            hg = np.array(frame.home_grid)
            ag = np.array(frame.away_grid)
            total = hg.size
            home_cells = int(np.sum(hg > 0.5))
            _ = int(np.sum(ag > 0.5))
            ball_control_pct = round((home_cells / max(total, 1)) * 100.0, 1)

            return json.dumps(
                {
                    "home_grid": frame.home_grid,
                    "away_grid": frame.away_grid,
                    "ball_control_pct": ball_control_pct,
                    "hot_zones": hot_zones,
                }
            )
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_player_pass_sonar(self, match_id: str, track_id: str) -> str:
        """Compute pass direction sonar for a single player.

        Returns JSON with directions (8 compass points), pass_counts, accuracy_pct.
        """
        try:
            from kawkab.core.pass_sonars import compute_pass_sonars

            events = (
                await self.storage_service.get_match_events(match_id)
                if self.storage_service
                else []
            )
            events = self._normalize_event_types(events)
            sonars = compute_pass_sonars(events, sectors=8)
            target = [s for s in sonars if s.get("track_id") == str(track_id)]
            if not target:
                return json.dumps(
                    {
                        "directions": [],
                        "pass_counts": [],
                        "accuracy_pct": [],
                        "total_passes": 0,
                        "error": f"Player {track_id} not found",
                    }
                )
            player = target[0]
            directions = []
            pass_counts = []
            accuracy_pct = []
            for sec in player["sectors"]:
                directions.append(sec["angle_center"])
                pass_counts.append(sec["count"])
                accuracy_pct.append(round(sec["accuracy"] * 100.0, 1))
            return json.dumps(
                {
                    "directions": directions,
                    "pass_counts": pass_counts,
                    "accuracy_pct": accuracy_pct,
                    "total_passes": player["total_passes"],
                }
            )
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_space_control_heatmap(self, match_id: str) -> str:
        """Compute Voronoi-based space control heatmap for a match.

        Returns JSON with grid, team_control_pcts, hot_zones, space_gained.
        """
        try:
            from kawkab.core.space_control import compute_pitch_control_grid, identify_hot_zones

            events = (
                await self.storage_service.get_match_events(match_id)
                if self.storage_service
                else []
            )
            events = self._normalize_event_types(events)
            if not events:
                return json.dumps(
                    {"grid": [], "team_control_pcts": {}, "hot_zones": [], "space_gained": 0.0}
                )

            all_positions = []
            team_ids = []
            for e in events:
                sx = e.get("start_x")
                sy = e.get("start_y")
                if sx is not None and sy is not None and e.get("team") in ("home", "away"):
                    tid = 0 if e.get("team") == "home" else 1
                    all_positions.append((float(sx), float(sy), e.get("id", 0) or 0))
                    team_ids.append(tid)

            if not all_positions:
                return json.dumps(
                    {"grid": [], "team_control_pcts": {}, "hot_zones": [], "space_gained": 0.0}
                )

            grid, team_pcts = compute_pitch_control_grid(all_positions[:22], team_ids[:22])
            hot_zones = identify_hot_zones(grid, 0)

            space_gained_count = 0
            for e in events:
                if (
                    e.get("type") == "pass"
                    and e.get("start_x") is not None
                    and e.get("end_x") is not None
                ):
                    space_gained_count += 1
            space_gained = round(space_gained_count * 1.5, 1)

            grid_list = grid.tolist() if hasattr(grid, "tolist") else grid
            return json.dumps(
                {
                    "grid": grid_list,
                    "team_control_pcts": team_pcts,
                    "hot_zones": hot_zones,
                    "space_gained": space_gained,
                }
            )
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_player_role(self, match_id: str, track_id: str) -> str:
        """Classify a player's role from their event data.

        Returns JSON with primary_role, confidence, secondary_role, role_breakdown.
        """
        try:
            from kawkab.core.role_classifier import classify_player_role

            events = (
                await self.storage_service.get_match_events(match_id)
                if self.storage_service
                else []
            )
            events = self._normalize_event_types(events)
            player_events = [
                e
                for e in events
                if str(e.get("track_id")) == str(track_id)
                or str(e.get("player_id")) == str(track_id)
            ]
            if not player_events:
                return json.dumps(
                    {
                        "primary_role": "unknown",
                        "confidence": 0.0,
                        "secondary_role": "",
                        "role_breakdown": {},
                    }
                )

            role = classify_player_role(player_events)
            return json.dumps(
                {
                    "primary_role": role.primary_role,
                    "confidence": role.confidence,
                    "secondary_role": role.secondary_role,
                    "role_breakdown": role.role_scores,
                }
            )
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_dominance_index(self, match_id: str) -> str:
        """Compute composite dominance index (0-100) for a match.

        Returns JSON with index, sub_scores, per_phase.
        """
        try:
            from kawkab.core.dominance_index import compute_dominance_index

            events = (
                await self.storage_service.get_match_events(match_id)
                if self.storage_service
                else []
            )
            events = self._normalize_event_types(events)
            if not events:
                return json.dumps(
                    {
                        "index": 50.0,
                        "sub_scores": {},
                        "phases": {},
                        "team": "home",
                        "opponent": "away",
                    }
                )

            report = compute_dominance_index(events, "home")
            return json.dumps(
                {
                    "index": report.index,
                    "team": report.team,
                    "opponent": report.opponent,
                    "sub_scores": report.sub_scores,
                    "phases": report.phases,
                }
            )
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ── Advanced analysis modules (Sprint 12+) ────────────────────

    async def compute_goals_added(self, match_id: int) -> str:
        try:
            from kawkab.core.goals_added import compute_goals_added
            from kawkab.core.xg_model import compute_xg_trained_from_dict

            events = (
                await self.storage_service.get_match_events(match_id)
                if self.storage_service
                else []
            )
            events = self._normalize_event_types(events)
            players = (
                await self.storage_service.get_match_players(match_id)
                if self.storage_service
                else []
            )
            defensive_types = {"tackle", "interception", "clearance"}
            result = {}
            for p in players:
                tid = p.get("track_id")
                if tid is None:
                    continue
                p_events = [e for e in events if e.get("from_track_id") == tid]
                xg = 0.0
                for e in p_events:
                    if e.get("event_type") != "shot":
                        continue
                    try:
                        xg += compute_xg_trained_from_dict(e)
                    except Exception:
                        continue
                defensive_actions = sum(
                    1 for e in p_events if e.get("event_type") in defensive_types
                )
                match_stats = [
                    {
                        "match_id": match_id,
                        "xg": xg,
                        "defensive_actions": defensive_actions,
                        "minutes": 90,
                    }
                ]
                report = compute_goals_added(str(tid), match_stats, str(p.get("position", "MID")))
                result[str(tid)] = report.__dict__
            return json.dumps({"success": True, "result": result, "count": len(result)})
        except Exception as e:
            logger.error(f"compute_goals_added failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def analyze_finishing(self, match_id: int) -> str:
        try:
            _ = (match_id,)
            # analyze_finishing needs player-scoped shot events; this handler
            # only has match-level events and no player context, so the slot
            # is explicitly unwired instead of raising TypeError at runtime.
            result: dict = {"available": False, "reason": "needs player-scoped shot events"}
            return json.dumps({"success": True, "result": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def simulate_league(self, match_id: int, iterations: int = 10000) -> str:
        try:
            _ = (match_id, iterations)
            # simulate_league needs a full-season fixture list and the current
            # table; a single match's events provide neither. Dead since the
            # signature changed — return an explicit unavailable marker.
            result: dict = {"available": False, "reason": "needs season fixtures + current table"}
            return json.dumps({"success": True, "result": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def estimate_transfer_fee(self, match_id: int, track_id: int) -> str:
        try:
            _ = (match_id, track_id)
            # estimate_player_transfer_fee takes age/position/performance
            # stats, none of which this handler has. Dead since the valuation
            # API changed — return an explicit unavailable marker.
            result: dict = {
                "available": False,
                "reason": "needs player age/position/performance data",
            }
            return json.dumps({"success": True, "result": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def generate_match_report(self, match_id: int) -> str:
        try:
            _ = (match_id,)
            # generate_match_report needs match metadata (teams, date, score)
            # alongside events; metadata lookup is not wired in this handler.
            # Dead since the report API gained required params.
            result: dict = {"available": False, "reason": "match metadata lookup not wired"}
            return json.dumps({"success": True, "result": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def generate_game_plan(self, match_id: int, opponent_id: int) -> str:
        try:
            from kawkab.core.game_plan import generate_game_plan as _ggp

            events = (
                await self.storage_service.get_match_events(match_id)
                if self.storage_service
                else []
            )
            events = self._normalize_event_types(events)
            # generate_game_plan labels the plan with an opponent *name*; the
            # bridge only receives a numeric id (the UI passes 0), so omit it
            # and let the report fall back to "Unknown".
            _ = opponent_id
            result = _ggp(events)
            return json.dumps({"success": True, "result": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def compute_phase_xg(self, match_id: int) -> str:
        try:
            _ = (match_id,)
            # compute_phase_xg needs team_events/opponent_events/events with
            # team separation; the raw single-list fetch here predates the
            # current signature. Dead slot — explicit unavailable marker.
            result: dict = {"available": False, "reason": "team-separated event inputs not wired"}
            return json.dumps({"success": True, "result": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def analyze_build_up(self, match_id: int) -> str:
        try:
            _ = (match_id,)
            # analyze_build_up needs team_events plus a team_id; the raw
            # single-list fetch here predates the current signature. Dead slot.
            result: dict = {"available": False, "reason": "team-scoped inputs not wired"}
            return json.dumps({"success": True, "result": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    def compute_territory_value(self, match_id: int) -> str:
        try:
            _ = (match_id,)
            # compute_territory_value needs team_events/opponent_events and a
            # team_id; the raw single-list fetch predates the current
            # signature. Dead slot — explicit unavailable marker.
            result: dict = {"available": False, "reason": "team-scoped inputs not wired"}
            return json.dumps({"success": True, "result": result})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Sprint 5 — Data Quality Score
    # ================================================================

    async def get_match_quality_score(self, match_id: str) -> str:
        """Compute data quality score for a match using anomaly detection.

        Returns JSON with: score, anomaly_count, anomalies list, warnings.
        """
        try:
            mid = SecurityValidator.validate_match_id(match_id)
            events = (
                await self.storage_service.get_match_events(mid) if self.storage_service else []
            )
            events = self._normalize_event_types(events)

            from kawkab.core.match_anomaly_detection import (
                compute_data_quality_score,
                detect_anomalies,
            )

            report = detect_anomalies(events)
            quality_score = compute_data_quality_score(events)

            result = {
                "score": round(quality_score, 1),
                "anomaly_count": len(report.anomalies),
                "anomalies": report.anomalies,
                "warnings": [],
            }

            if quality_score >= 80:
                result["level"] = "good"
            elif quality_score >= 50:
                result["level"] = "fair"
            else:
                result["level"] = "poor"

            return json.dumps(result)
        except Exception as e:
            logger.error(f"get_match_quality_score failed: {e}")
            return json.dumps(
                {"error": ErrorSanitizer.sanitize_error(e), "score": 0.0, "level": "error"}
            )

    # ================================================================
    # Expected Assists (xA) report
    # ================================================================

    async def get_xa_report(self, match_id: str) -> str:
        """Compute match-level expected assists (xA) for both teams.

        Returns JSON with: home, away, total, home_sequence_xa,
        away_sequence_xa.
        """
        try:
            mid = SecurityValidator.validate_match_id(match_id)
            events = (
                await self.storage_service.get_match_events(mid) if self.storage_service else []
            )
            events = self._normalize_event_types(events)
            from kawkab.core.xa_model import ExpectedAssistModel

            report = ExpectedAssistModel().compute_match_xa(events)
            return json.dumps(report.to_dict())
        except Exception as e:
            logger.error(f"get_xa_report failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Pressing efficiency report
    # ================================================================

    async def get_pressing_report(self, match_id: str) -> str:
        """Compute pressing efficiency metrics for both teams.

        Returns JSON with per-team: trap_to_shot (traps, shots_from_traps,
        goals_from_traps, conversion_rate) and high_press_efficiency
        (traps-per-shot-conceded index; higher is better press discipline).

        Note: only trap-to-shot conversion and the high-press index are
        wired here -- PressingEfficiencyAnalyzer also has
        compute_trap_to_goal_rate and compute_press_recovery_attack (not
        surfaced yet), and core/pressing_clusters.py's spatial zone
        clustering needs a pitch-map visualization, not a stat card, so
        it's a separate follow-up (see CLAUDE.md).
        """
        try:
            mid = SecurityValidator.validate_match_id(match_id)
            events = (
                await self.storage_service.get_match_events(mid) if self.storage_service else []
            )
            events = self._normalize_event_types(events)
            from kawkab.core.pressing_efficiency import PressingEfficiencyAnalyzer

            analyzer = PressingEfficiencyAnalyzer()
            trap_to_shot = analyzer.compute_trap_to_shot_rate(events)
            high_press = analyzer.analyze_high_press_efficiency(events)

            return json.dumps(
                {
                    "home": {
                        **trap_to_shot.get("home", {}),
                        "high_press_index": high_press.get("home", 0.0),
                    },
                    "away": {
                        **trap_to_shot.get("away", {}),
                        "high_press_index": high_press.get("away", 0.0),
                    },
                }
            )
        except Exception as e:
            logger.error(f"get_pressing_report failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Sprint 16: GPS / Physical Data Pipeline
    # ================================================================
