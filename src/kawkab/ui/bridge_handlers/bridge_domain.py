"""Handlers extracted from bridge_analysis.py (Phase C1 split).

Thin Qt-bridge handlers: each owns one surface and forwards to its
service layer. Registered in bridge_handlers/__init__.py and wired in
ui/bridge.py. Validation lives in the service layer unless noted.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from kawkab.core.security import ErrorSanitizer
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = logging.getLogger(__name__)


class DomainHandler(BridgeHandlerBase):
    """Domain-analysis surfaces: set pieces, goalkeeper, substitutions,
    possession, psychology, football rules, tactical cards."""

    def __init__(self, bridge, services: dict[str, Any], rate_limiter=None) -> None:
        super().__init__(bridge, services, rate_limiter)
        self.bridge = bridge  # public alias kept for backward compatibility

    # _check_rate_limit inherited from BridgeHandlerBase ("analysis" bucket).
    # History: the deleted override called a nonexistent RateLimiter.check(),
    # so analyze_match_psychology crashed with the real bridge limiter wired
    # in (caught by the GUI e2e audit) -- the duplication that allowed that
    # drift now lives in exactly one place.

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    @property
    def cv_service(self):
        return self._services.get("cv_service")

    @property
    def enhancement_service(self):
        return self._services.get("enhancement_service")

    @property
    def analysis_service(self):
        return self._services.get("analysis_service")

    @property
    def llm_service(self):
        return self._services.get("llm_service")

    @property
    def knowledge_service(self):
        return self._services.get("knowledge_service")

    @property
    def homography_service(self):
        return self._services.get("homography_service")

    @property
    def lightglue_homography_service(self):
        return self._services.get("lightglue_homography_service")

    @property
    def benchmark_service(self):
        return self._services.get("benchmark_service")

    @property
    def player_profile_service(self):
        return self._services.get("player_profile_service")

    @property
    def multi_match_service(self):
        return self._services.get("multi_match_service")

    @property
    def visualization_service(self):
        return self._services.get("visualization_service")

    @property
    def quality_scoring_service(self):
        return self._services.get("quality_scoring_service")

    @property
    def advanced_event_detection_service(self):
        return self._services.get("advanced_event_detection_service")

    @property
    def physical_load_service(self):
        return self._services.get("physical_load_service")

    @property
    def prematch_briefing_service(self):
        if not hasattr(self, "_prematch_briefing_service"):
            from kawkab.analysis.prematch_briefing import PreMatchBriefingService

            self._prematch_briefing_service = PreMatchBriefingService()
        return self._prematch_briefing_service

    @property
    def injury_risk_predictor(self):
        from kawkab.core.injury_risk import InjuryRiskPredictor

        if not hasattr(self, "_injury_risk_predictor"):
            self._injury_risk_predictor = InjuryRiskPredictor()
        return self._injury_risk_predictor

    @property
    def training_plan_generator(self):
        if not hasattr(self, "_training_plan_generator"):
            from kawkab.services.training_plan_service import TrainingPlanGenerator

            kb = self.knowledge_service
            self._training_plan_generator = TrainingPlanGenerator(kb)
        return self._training_plan_generator

    @property
    def pressure_metrics_service(self):
        return self._services.get("pressure_metrics_service")

    @property
    def face_recognition_service(self):
        return self._services.get("face_recognition_service")

    @property
    def setpiece_service(self):
        return self._services.get("setpiece_service")

    @property
    def goalkeeper_service(self):
        return self._services.get("goalkeeper_service")

    @property
    def goalkeeper_analytics(self):
        # getattr (not bare attribute access): this lazily-built cache is
        # created here, and reading it before first construction raised
        # AttributeError out of every compute_goalkeeper_* slot.
        svc = getattr(self, "_goalkeeper_analytics", None)
        if svc is not None:
            return svc
        svc = self._services.get("goalkeeper_analytics")
        if svc is None:
            from kawkab.analysis.goalkeeper_analytics import GoalkeeperAnalytics

            svc = GoalkeeperAnalytics()
        self._goalkeeper_analytics = svc
        return svc

    @property
    def substitution_service(self):
        return self._services.get("substitution_service")

    @property
    def possession_service(self):
        return self._services.get("possession_service")

    @property
    def psychology_service(self):
        return self._services.get("psychology_service")

    @property
    def football_rules_service(self):
        return self._services.get("football_rules_service")

    @property
    def card_detection_service(self):
        return self._services.get("card_detection_service")

    @property
    def pose_analysis_service(self):
        return self._services.get("pose_analysis_service")

    @property
    def mujoco_ball_service(self):
        return self._services.get("mujoco_ball_service")

    @property
    def fluidx3d_service(self):
        return self._services.get("fluidx3d_service")

    @property
    def weather_service(self):
        return self._services.get("weather_service")

    @property
    def roboflow_sports_service(self):
        return self._services.get("roboflow_sports_service")

    @property
    def statsbomb_service(self):
        return self._services.get("statsbomb_service")

    @property
    def profiler(self):
        return self._services.get("profiler")

    @property
    def frame_skip(self):
        return self._services.get("frame_skip", 3)

    async def check_setpiece_status(self):
        if self.setpiece_service is None:
            return json.dumps({"available": False})
        return json.dumps({"available": self.setpiece_service.available})

    async def analyze_setpieces(self, events_json, home_team):
        if self.setpiece_service is None:
            return json.dumps({"error": "Set-piece service not initialized"})
        try:
            from kawkab.services.setpiece_service import SetPieceEvent

            events_data = json.loads(events_json) if events_json else []
            events = [
                SetPieceEvent(
                    set_piece_type=e.get("set_piece_type", "corner"),
                    minute=e.get("minute", 0),
                    second=e.get("second", 0),
                    team=e.get("team", ""),
                    delivery_x=e.get("delivery_x", 50),
                    delivery_y=e.get("delivery_y", 34),
                    delivery_style=e.get("delivery_style", "lofted"),
                    delivery_height=e.get("delivery_height", "medium"),
                    first_contact_x=e.get("first_contact_x"),
                    first_contact_y=e.get("first_contact_y"),
                    outcome=e.get("outcome", "unknown"),
                )
                for e in events_data
            ]
            away_team = "away" if home_team == "home" else "home"
            report = self.setpiece_service.analyze(events, home_team, away_team)
            return json.dumps(
                {
                    "home": {
                        "total_corners": report.home_stats.total_corners,
                        "total_free_kicks": report.home_stats.total_free_kicks,
                        "shots_per_corner": report.home_stats.shots_per_corner,
                        "goals_per_corner": report.home_stats.goals_per_corner,
                        "favorite_target_zone": report.home_stats.favorite_target_zone,
                        "common_routines": report.home_stats.common_routines,
                        "threat_per_set_piece": report.home_stats.threat_per_set_piece,
                    },
                    "away": {
                        "total_corners": report.away_stats.total_corners,
                        "total_free_kicks": report.away_stats.total_free_kicks,
                        "shots_per_corner": report.away_stats.shots_per_corner,
                        "goals_per_corner": report.away_stats.goals_per_corner,
                        "favorite_target_zone": report.away_stats.favorite_target_zone,
                        "common_routines": report.away_stats.common_routines,
                        "threat_per_set_piece": report.away_stats.threat_per_set_piece,
                    },
                    "home_threat_total": report.home_threat_total,
                    "away_threat_total": report.away_threat_total,
                    "set_piece_differential": report.set_piece_differential,
                    "notes": report.notes,
                }
            )
        except Exception as e:
            logger.error(f"analyze_setpieces failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Goalkeeper analysis
    # ================================================================

    async def check_goalkeeper_status(self):
        if self.goalkeeper_service is None:
            return json.dumps({"available": False})
        return json.dumps({"available": self.goalkeeper_service.available})

    async def analyze_goalkeeper(self, team, actions_json, shots_json, clean_sheet):
        if self.goalkeeper_service is None:
            return json.dumps({"error": "Goalkeeper service not initialized"})
        try:
            from kawkab.services.goalkeeper_service import GoalkeeperAction

            actions_data = json.loads(actions_json) if actions_json else []
            shots_data = json.loads(shots_json) if shots_json else []
            actions = [
                GoalkeeperAction(
                    action_type=a.get("action_type", "save"),
                    minute=a.get("minute", 0),
                    second=a.get("second", 0),
                    team=a.get("team", team),
                    player_track_id=a.get("player_track_id"),
                    outcome=a.get("outcome", "complete"),
                    quality=a.get("quality", 0.5),
                    x=a.get("x"),
                    y=a.get("y"),
                )
                for a in actions_data
            ]
            stats = self.goalkeeper_service.compute_stats(
                team, actions, shots_data, clean_sheet=clean_sheet
            )
            return json.dumps(
                {
                    "team": stats.team,
                    "saves": stats.saves,
                    "goals_conceded": stats.goals_conceded,
                    "shots_faced": stats.shots_faced,
                    "save_rate": stats.save_rate,
                    "goals_prevented_xgot": stats.goals_prevented_xgot,
                    "xgot_per_shot": stats.xgot_per_shot,
                    "crosses_claimed": stats.crosses_claimed,
                    "crosses_punched": stats.crosses_punched,
                    "crosses_missed": stats.crosses_missed,
                    "sweep_actions": stats.sweep_actions,
                    "short_distribution_attempts": stats.short_distribution_attempts,
                    "short_distribution_successful": stats.short_distribution_successful,
                    "long_distribution_attempts": stats.long_distribution_attempts,
                    "long_distribution_successful": stats.long_distribution_successful,
                    "clean_sheet": stats.clean_sheet,
                    "notes": stats.notes,
                }
            )
        except Exception as e:
            logger.error(f"analyze_goalkeeper failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def compute_xgot(self, shot_x, shot_y, body_part, one_on_one):
        if self.goalkeeper_service is None:
            return json.dumps({"error": "Goalkeeper service not initialized"})
        try:
            xgot = self.goalkeeper_service.compute_xgot_simple(
                float(shot_x), float(shot_y), body_part or "foot", bool(one_on_one)
            )
            return json.dumps({"xgot": round(xgot, 3)})
        except Exception as e:
            logger.error(f"compute_xgot failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def analyze_goalkeeper_advanced(
        self,
        team,
        shots_json,
        crosses_json,
        dist_json,
        positions_json,
        sweeps_json,
        sp_json,
        clean_sheet,
        player_name,
    ):
        gka = self.goalkeeper_analytics
        if gka is None:
            return json.dumps({"error": "GoalkeeperAnalytics not available"})
        try:
            shots = json.loads(shots_json) if shots_json else None
            crosses = json.loads(crosses_json) if crosses_json else None
            dist = json.loads(dist_json) if dist_json else None
            pos = json.loads(positions_json) if positions_json else None
            sweeps = json.loads(sweeps_json) if sweeps_json else None
            sp = json.loads(sp_json) if sp_json else None
            report = gka.compute_match_report(
                team=team,
                shots_faced=shots,
                cross_actions=crosses,
                distribution_actions=dist,
                tracking_positions=pos,
                sweeps=sweeps,
                set_piece_positions=sp,
                clean_sheet=bool(clean_sheet),
                player_name=player_name or "",
            )
            return json.dumps(report.to_dict())
        except Exception as e:
            logger.error(f"analyze_goalkeeper_advanced failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def compute_goalkeeper_save_quality(self, shots_json):
        gka = self.goalkeeper_analytics
        if gka is None:
            return json.dumps({"error": "GoalkeeperAnalytics not available"})
        try:
            shots = json.loads(shots_json) if shots_json else []
            sq = gka.compute_save_quality(shots)
            return json.dumps(
                {
                    "total_shots_faced": sq.total_shots_faced,
                    "saves": sq.saves,
                    "goals_conceded": sq.goals_conceded,
                    "save_rate": round(sq.save_rate, 3),
                    "total_psxg": round(sq.total_psxg, 3),
                    "goals_prevented": round(sq.goals_prevented, 3),
                    "avg_psxg_per_shot": round(sq.avg_psxg_per_shot, 3),
                    "high_quality_saves": sq.high_quality_saves,
                    "one_on_one_saves": sq.one_on_one_saves,
                    "one_on_one_goals": sq.one_on_one_goals,
                    "one_on_one_save_rate": round(sq.one_on_one_save_rate, 3),
                }
            )
        except Exception as e:
            logger.error(f"compute_goalkeeper_save_quality failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def compute_goalkeeper_aerial_command(self, crosses_json):
        gka = self.goalkeeper_analytics
        if gka is None:
            return json.dumps({"error": "GoalkeeperAnalytics not available"})
        try:
            crosses = json.loads(crosses_json) if crosses_json else []
            ac = gka.compute_aerial_command(crosses)
            return json.dumps(
                {
                    "crosses_claimed": ac.crosses_claimed,
                    "crosses_punched": ac.crosses_punched,
                    "crosses_faced": ac.crosses_faced,
                    "aerial_success_rate": round(ac.aerial_success_rate, 3),
                    "claimed_under_pressure": ac.claimed_under_pressure,
                }
            )
        except Exception as e:
            logger.error(f"compute_goalkeeper_aerial_command failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def compute_goalkeeper_distribution(self, dist_json):
        gka = self.goalkeeper_analytics
        if gka is None:
            return json.dumps({"error": "GoalkeeperAnalytics not available"})
        try:
            dist = json.loads(dist_json) if dist_json else []
            gd = gka.compute_distribution(dist)
            return json.dumps(
                {
                    "short_attempts": gd.short_attempts,
                    "short_successful": gd.short_successful,
                    "short_accuracy": round(gd.short_accuracy, 3),
                    "long_attempts": gd.long_attempts,
                    "long_successful": gd.long_successful,
                    "long_accuracy": round(gd.long_accuracy, 3),
                    "avg_distance": round(gd.avg_distance, 1),
                }
            )
        except Exception as e:
            logger.error(f"compute_goalkeeper_distribution failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Substitution analysis
    # ================================================================

    async def check_substitution_status(self):
        if self.substitution_service is None:
            return json.dumps({"available": False})
        return json.dumps({"available": self.substitution_service.available})

    async def analyze_substitutions(self, team, subs_json, events_json):
        if self.substitution_service is None:
            return json.dumps({"error": "Substitution service not initialized"})
        try:
            from kawkab.services.substitution_service import SubstitutionEvent

            subs_data = json.loads(subs_json) if subs_json else []
            events = json.loads(events_json) if events_json else []
            subs = [
                SubstitutionEvent(
                    minute=s.get("minute", 0),
                    second=s.get("second", 0),
                    team=s.get("team", team),
                    player_off_track_id=s.get("player_off_track_id"),
                    player_off_name=s.get("player_off_name"),
                    player_on_track_id=s.get("player_on_track_id"),
                    player_on_name=s.get("player_on_name"),
                    formation_before=s.get("formation_before"),
                    formation_after=s.get("formation_after"),
                    position_changed=s.get("position_changed", False),
                )
                for s in subs_data
            ]
            report = self.substitution_service.analyze(team, subs, events)
            return json.dumps(
                {
                    "team": report.team,
                    "total_impact": report.total_impact,
                    "avg_impact": report.avg_impact,
                    "tactical_changes": report.tactical_changes,
                    "formation_changes": report.formation_changes,
                    "best_sub": {
                        "minute": report.best_sub.substitution.minute,
                        "rating": report.best_sub.rating,
                        "verdict": report.best_sub.verdict,
                        "notes": report.best_sub.notes,
                    }
                    if report.best_sub
                    else None,
                    "worst_sub": {
                        "minute": report.worst_sub.substitution.minute,
                        "rating": report.worst_sub.rating,
                        "verdict": report.worst_sub.verdict,
                        "notes": report.worst_sub.notes,
                    }
                    if report.worst_sub
                    else None,
                    "impacts": [
                        {
                            "minute": i.substitution.minute,
                            "player_on": i.substitution.player_on_name,
                            "player_off": i.substitution.player_off_name,
                            "rating": i.rating,
                            "verdict": i.verdict,
                            "xg_delta": i.xg_delta,
                            "possession_delta": i.possession_delta,
                            "goals_for": i.goals_for,
                            "goals_against": i.goals_against,
                            "notes": i.notes,
                        }
                        for i in report.impacts
                    ],
                }
            )
        except Exception as e:
            logger.error(f"analyze_substitutions failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Possession analysis
    # ================================================================

    async def check_possession_status(self):
        if self.possession_service is None:
            return json.dumps({"available": False})
        return json.dumps({"available": self.possession_service.available})

    async def analyze_possession(self, home_team, away_team, events_json):
        if self.possession_service is None:
            return json.dumps({"error": "Possession service not initialized"})
        try:
            events = json.loads(events_json) if events_json else []
            report = self.possession_service.analyze(home_team, away_team, events)
            return json.dumps(
                {
                    "home_possession_pct": report.home_possession_pct,
                    "away_possession_pct": report.away_possession_pct,
                    "counter_presses": report.counter_presses,
                    "avg_chain_duration_s": report.avg_chain_duration_s,
                    "longest_chain_s": report.longest_chain_s,
                    "home_chains_count": len(report.home_chains),
                    "away_chains_count": len(report.away_chains),
                    "home_player_stats": {
                        str(tid): {
                            "touches": s.touches,
                            "possession_time_s": round(s.total_possession_time_s, 1),
                            "successful_passes": s.successful_passes,
                            "failed_passes": s.failed_passes,
                        }
                        for tid, s in report.home_player_stats.items()
                    },
                    "away_player_stats": {
                        str(tid): {
                            "touches": s.touches,
                            "possession_time_s": round(s.total_possession_time_s, 1),
                            "successful_passes": s.successful_passes,
                            "failed_passes": s.failed_passes,
                        }
                        for tid, s in report.away_player_stats.items()
                    },
                    "notes": report.notes,
                }
            )
        except Exception as e:
            logger.error(f"analyze_possession failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Psychology analysis
    # ================================================================

    async def check_psychology_status(self):
        if self.psychology_service is None:
            return json.dumps({"available": False})
        return json.dumps({"available": self.psychology_service.available})

    async def analyze_match_psychology(self, match_id, home_team, away_team, events_json):
        self._check_rate_limit()
        if self.psychology_service is None:
            return json.dumps({"error": "Psychology service not initialized"})
        try:
            events = json.loads(events_json) if events_json else []
            report = self.psychology_service.analyze(match_id, home_team, away_team, events)
            return json.dumps(
                {
                    "match_id": report.match_id,
                    "home_team": report.home_team,
                    "away_team": report.away_team,
                    "score_state_transitions": [
                        {
                            "minute": t.minute,
                            "second": t.second,
                            "team": t.team,
                            "from_state": t.from_state.value,
                            "to_state": t.to_state.value,
                            "trigger": t.trigger_event,
                        }
                        for t in report.score_state_transitions
                    ],
                    "momentum_timeline": [
                        {"minute": m.minute, "home": m.home_momentum, "away": m.away_momentum}
                        for m in report.momentum_timeline[::3]
                    ],
                    "psychology_events": [
                        {
                            "type": e.event_type.value,
                            "minute": e.minute,
                            "second": e.second,
                            "team": e.team,
                            "description": e.description,
                            "severity": e.severity,
                        }
                        for e in report.psychology_events
                    ],
                    "post_goal_lull_count": report.post_goal_lull_count,
                    "comeback_count": report.comeback_count,
                    "capitulation_count": report.capitulation_count,
                    "avg_late_game_passing_drop": report.avg_late_game_passing_drop,
                    "notes": report.notes,
                }
            )
        except Exception as e:
            logger.error(f"analyze_match_psychology failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Football Rules
    # ================================================================

    async def check_rules_status(self):
        if self.football_rules_service is None:
            return json.dumps({"available": False})
        return json.dumps(
            {
                "available": self.football_rules_service.available,
                "laws_count": len(self.football_rules_service._laws),
            }
        )

    async def get_law_summary(self, law_number):
        if self.football_rules_service is None:
            return json.dumps({"error": "Rules service not initialized"})
        law = self.football_rules_service.get_law_summary(law_number)
        if not law:
            return json.dumps({"error": f"Law {law_number} not found"})
        return json.dumps(law)

    async def get_all_laws(self):
        if self.football_rules_service is None:
            return json.dumps({"laws": []})
        return json.dumps({"laws": self.football_rules_service.get_all_laws()})

    async def classify_event_rule(self, event_type, x, y, side):
        if self.football_rules_service is None:
            return json.dumps({"error": "Rules service not initialized"})
        try:
            ref = self.football_rules_service.classify_event(event_type, x, y, side)
            return json.dumps(
                {
                    "law": ref.law,
                    "law_name": ref.law_name,
                    "restart": ref.restart.value if ref.restart else None,
                    "description": ref.description,
                    "card_likely": ref.card_likely,
                }
            )
        except Exception as e:
            logger.error(f"classify_event_rule failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def check_offside(self, attacker_x, defender_x, ball_x, attacking_direction):
        if self.football_rules_service is None:
            return json.dumps({"error": "Rules service not initialized"})
        try:
            result = self.football_rules_service.is_offside(
                attacker_x, defender_x, ball_x, attacking_direction
            )
            return json.dumps(
                {
                    "is_offside": result.is_offside,
                    "attacker_x": result.attacker_x,
                    "second_last_defender_x": result.second_last_defender_x,
                    "ball_x": result.ball_x,
                    "explanation": result.explanation,
                }
            )
        except Exception as e:
            logger.error(f"check_offside failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Card Detection
    # ================================================================

    async def check_cards_status(self):
        if self.card_detection_service is None:
            return json.dumps({"available": False})
        return json.dumps({"available": self.card_detection_service.available})

    async def infer_cards_tactically(self, events_json):
        if self.card_detection_service is None:
            return json.dumps({"error": "Card detection service not initialized"})
        try:
            events = json.loads(events_json) if events_json else []
            cards = self.card_detection_service.infer_cards_tactically(events)
            return json.dumps(
                {
                    "cards": [
                        {
                            "card_type": c.card_type.value,
                            "minute": c.minute,
                            "second": c.second,
                            "player_track_id": c.player_track_id,
                            "player_name": c.player_name,
                            "team": c.team,
                            "source": c.source.value,
                            "confidence": c.confidence,
                            "description": c.description,
                        }
                        for c in cards
                    ]
                }
            )
        except Exception as e:
            logger.error(f"infer_cards_tactically failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def fetch_external_cards(self, match_id):
        if self.card_detection_service is None:
            return json.dumps({"error": "Card detection service not initialized"})
        try:
            cards = await self.card_detection_service.fetch_external_cards(
                match_id,
                statsbomb_service=self.statsbomb_service,
            )
            return json.dumps(
                {
                    "cards": [
                        {
                            "card_type": c.card_type.value,
                            "minute": c.minute,
                            "second": c.second,
                            "player_name": c.player_name,
                            "team": c.team,
                            "source": c.source.value,
                            "confidence": c.confidence,
                            "description": c.description,
                        }
                        for c in cards
                    ]
                }
            )
        except Exception as e:
            logger.error(f"fetch_external_cards failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Pose analysis
    # ================================================================
