"""Tactical reasoning service - diagnoses match issues using the knowledge base.

This is the "Detective" layer of Kawkab AI. It:
1. Takes match analysis data (stats, formations, PPDA, events)
2. Queries the knowledge base for relevant tactical rules
3. Tests each rule's hypotheses against the data
4. Returns ranked diagnoses with confidence scores
5. Recommends training drills for each diagnosed issue
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.services.analysis_service import MatchAnalysis, TeamStats
from kawkab.services.knowledge_service import KnowledgeService, TacticalRule
from kawkab.services.reasoning import checkers as _checkers
from kawkab.services.reasoning.metrics import evaluate_hypothesis, precompute_event_stats

logger = get_logger(__name__)


@dataclass
class Diagnosis:
    """A single diagnosis of a tactical issue."""

    rule_id: str
    rule_name: str
    rule_name_ar: str
    category: str
    severity: str
    confidence: float
    evidence: dict[str, Any]
    explanation: str
    explanation_ar: str
    recommended_drills: list[str] = field(default_factory=list)
    video_timestamps: list[dict] = field(default_factory=list)
    # Hypothesis-level provenance (Phase B): each entry records what the
    # rule's hypotheses required, what was observed, whether criteria
    # were met, and whether that earned a confidence boost. See
    # reasoning.metrics.evaluate_hypothesis.
    hypothesis_evaluation: list[dict[str, Any]] = field(default_factory=list)
    # Pattern type that produced this diagnosis (provenance).
    pattern_type: str = ""
    # How many distinct matches contributed the events behind this
    # diagnosis (multi-match mode).
    match_count: int = 1
    # Confidence tier: "confirmed" only when the rule's min_matches
    # threshold is met by real distinct-match evidence.
    confirmation: str = "provisional"


@dataclass
class DiagnosisReport:
    """Complete tactical diagnosis report for a match."""

    match_id: int
    diagnoses: list[Diagnosis]
    overall_assessment: str
    overall_assessment_ar: str
    priority_actions: list[str]
    priority_actions_ar: list[str]
    confidence: float
    # Multi-match provenance: distinct match ids pooled, and whether the
    # report was produced in multi-match mode.
    match_ids: list[int] = field(default_factory=list)
    multi_match: bool = False


class ReasoningService:
    """Diagnoses tactical issues from match analysis data.

    Uses the knowledge base of tactical rules and the match analysis
    output to identify what's going wrong and what to fix.
    """

    def __init__(self, knowledge_service: KnowledgeService) -> None:
        self.kb = knowledge_service
        self._initialized = False
        logger.info("ReasoningService created")

    async def initialize(self) -> None:
        """Initialize by loading the knowledge base."""
        if not self._initialized:
            await self.kb.initialize()
            self._initialized = True
            logger.info(f"ReasoningService ready with {self.kb.stats['rules']} rules")

    async def diagnose_match(
        self,
        analysis: MatchAnalysis,
        events: list[dict] | None = None,
        language: str = "en",
        match_ids: list[int] | None = None,
    ) -> DiagnosisReport:
        """Run full tactical diagnosis on match analysis.

        Args:
            analysis: Output from AnalysisService.analyze_match
            events: Optional list of events (uses analysis.events if None)
            language: "en" or "ar" for explanations
            match_ids: Distinct match ids contributing events. More than
                one id switches to multi-match mode (see diagnose_matches).

        Returns:
            DiagnosisReport with ranked diagnoses and recommendations
        """
        ids = match_ids if match_ids is not None else [analysis.match_id]
        return await self._diagnose(
            [analysis],
            events if events is not None else analysis.events,
            language,
            ids,
        )

    async def diagnose_matches(
        self,
        analyses: list[MatchAnalysis],
        language: str = "en",
        match_ids: list[int] | None = None,
    ) -> DiagnosisReport:
        """Multi-match diagnosis: pool events across distinct matches.

        The knowledge-base rules declare ``min_matches`` thresholds
        (all 40 rule files set 3). Pooling several matches raises the
        evidence base and flips diagnoses from "provisional" to
        "confirmed" once a rule's threshold is met. Diagnoses are
        always labeled with how many matches contributed.
        """
        if not analyses:
            raise ValueError("diagnose_matches requires at least one analysis")
        ids = match_ids if match_ids is not None else [a.match_id for a in analyses]
        pooled_events: list[dict] = []
        for a in analyses:
            pooled_events.extend(a.events or [])
        return await self._diagnose(
            [self._aggregate_analyses(analyses)], pooled_events, language, ids
        )

    @staticmethod
    def _aggregate_analyses(analyses: list[MatchAnalysis]) -> MatchAnalysis:
        """Merge several match analyses into one pooled analysis.

        Numeric team stats are summed; possession is averaged; events
        are concatenated by the caller. First non-empty formation wins
        (formation snapshots are per-match observations, not additive).
        """
        if len(analyses) == 1:
            return analyses[0]
        base = analyses[0]
        home = TeamStats(team_name=base.home_team.team_name)
        away = TeamStats(team_name=base.away_team.team_name)
        for a in analyses:
            for tgt, src in ((home, a.home_team), (away, a.away_team)):
                tgt.possession_pct += src.possession_pct
                tgt.passes_attempted += src.passes_attempted
                tgt.passes_completed += src.passes_completed
                tgt.shots += src.shots
                tgt.shots_on_target += src.shots_on_target
                tgt.tackles += src.tackles
                tgt.corners += src.corners
                tgt.fouls += src.fouls
        n = len(analyses)
        home.possession_pct /= n
        away.possession_pct /= n
        formations: dict[Any, Any] = next((a.formations for a in analyses if a.formations), {})
        players: dict[Any, Any] = {}
        for a in analyses:
            players.update(a.players or {})
        return MatchAnalysis(
            match_id=base.match_id,
            duration_seconds=max(a.duration_seconds or 0.0 for a in analyses),
            home_team=home,
            away_team=away,
            players=players,
            events=[],
            formations=formations,
        )

    async def _diagnose(
        self,
        analyses: list[MatchAnalysis],
        events: list[dict],
        language: str,
        match_ids: list[int],
    ) -> DiagnosisReport:
        """Shared diagnosis pipeline for single- and multi-match mode."""
        await self.initialize()

        analysis = analyses[0]
        match_count = max(1, len(set(match_ids)))
        multi_match = match_count > 1

        logger.info(
            f"Diagnosing match {analysis.match_id} ({match_count} match(es)): "
            f"{len(events)} events, "
            f"possession {analysis.home_team.possession_pct:.1f}%/{analysis.away_team.possession_pct:.1f}%"
        )

        event_stats = precompute_event_stats(events)

        diagnoses = []
        unmatched_patterns: list[str] = []
        all_rules = self.kb.get_all_rules()

        for rule in all_rules:
            diagnosis = await self._test_rule(rule, analysis, event_stats, match_count)
            if diagnosis is None:
                sig_type = str(rule.pattern_signature.get("type", ""))
                if sig_type and sig_type not in self._dispatch_map():
                    unmatched_patterns.append(sig_type)
                continue
            if diagnosis.confidence > 0.3:
                diagnoses.append(diagnosis)
            elif diagnosis.pattern_type not in self._dispatch_map():
                unmatched_patterns.append(diagnosis.pattern_type)

        if unmatched_patterns:
            # Honesty: a rule whose pattern has no checker can never
            # fire. Surface that fact in the log (and via the report's
            # metadata) instead of silently ignoring the rule file.
            logger.warning(
                f"{len(set(unmatched_patterns))} KB pattern type(s) have no "
                f"checker implemented: {sorted(set(unmatched_patterns))[:10]}"
            )

        diagnoses.sort(key=lambda d: d.confidence, reverse=True)

        priority_actions = self._build_priority_actions(diagnoses, language)
        overall = self._build_overall_assessment(diagnoses, analysis, language)
        overall_conf = (
            sum(d.confidence for d in diagnoses[:5]) / min(5, len(diagnoses)) if diagnoses else 0.0
        )

        logger.info(
            f"Diagnosis complete: {len(diagnoses)} issues found, "
            f"top: {diagnoses[0].rule_name if diagnoses else 'none'}"
        )

        return DiagnosisReport(
            match_id=analysis.match_id,
            diagnoses=diagnoses,
            overall_assessment=overall["en"],
            overall_assessment_ar=overall["ar"],
            priority_actions=priority_actions["en"],
            priority_actions_ar=priority_actions["ar"],
            confidence=overall_conf,
            match_ids=list(dict.fromkeys(match_ids)),
            multi_match=multi_match,
        )

    def _precompute_event_stats(self, events: list[dict]) -> dict:
        """Single-pass event analysis for all check methods."""
        stats: dict[str, Any] = {
            "goals": [],
            "turnovers": [],
            "shots": [],
            "passes": [],
            "crosses": [],
            "set_piece_goals": [],
            "counter_attack_shots": [],
            "counter_attack_goals": [],
            "final_third_events": [],
            "behind_def_line_events": [],
            "1v1_situations": [],
            "striker_passes": [],
            "own_half_turnovers": [],
            "late_events": [],
            "first_events": [],
            "total_events": 0,
        }

        if not events:
            return stats

        total_duration = 0.0
        timestamps = [e.get("timestamp", 0) for e in events if isinstance(e, dict)]
        if timestamps:
            total_duration = max(timestamps)

        for ev in events:
            if not isinstance(ev, dict):
                continue
            stats["total_events"] += 1
            etype = ev.get("type", "")
            zone = ev.get("zone", "")
            timestamp = ev.get("timestamp", 0)
            _ = ev.get("team", "")
            situation = ev.get("situation", "")
            outcome = ev.get("outcome", "")

            if etype == "goal":
                stats["goals"].append(ev)
                if situation in ("corner", "free_kick", "throw_in"):
                    stats["set_piece_goals"].append(ev)
            elif etype == "turnover":
                stats["turnovers"].append(ev)
                if zone in ("defensive_third", "middle_third"):
                    stats["own_half_turnovers"].append(ev)
            elif etype == "shot":
                stats["shots"].append(ev)
                if situation == "counter_attack":
                    stats["counter_attack_shots"].append(ev)
                    if outcome == "goal":
                        stats["counter_attack_goals"].append(ev)
            elif etype == "pass":
                stats["passes"].append(ev)
                if ev.get("to_position") == "striker":
                    stats["striker_passes"].append(ev)
            elif etype == "cross":
                stats["crosses"].append(ev)
            elif etype == "1v1_situation":
                stats["1v1_situations"].append(ev)

            if zone == "final_third":
                stats["final_third_events"].append(ev)
            if zone == "behind_defensive_line":
                stats["behind_def_line_events"].append(ev)

            if total_duration > 0:
                q1 = total_duration * 0.25
                q3 = total_duration * 0.75
                if timestamp > q3:
                    stats["late_events"].append(ev)
                elif timestamp < q1:
                    stats["first_events"].append(ev)

        return stats

    def _run_checker(
        self,
        rule: TacticalRule,
        analysis: MatchAnalysis,
        event_stats: dict,
        match_count: int = 1,
    ) -> tuple[float, dict] | None:
        """Run a rule's pattern checker and return its raw observation.

        This is the *observation* half of ``_test_rule``: no confidence
        gate, no volume gate, no hypothesis boosts. Measurement code that
        must compare the same quantity before and after an intervention
        uses ``observe_rule`` (which wraps this) rather than re-deriving
        the metric from a different bucket, so the number it compares is
        the number the diagnosis actually observed.

        Returns ``(confidence, evidence)`` or ``None`` when the rule's
        pattern type has no checker.
        """
        pattern_type = rule.pattern_signature.get("type", "")
        checker = self._dispatch_map().get(pattern_type)
        if checker is None:
            return None
        confidence, evidence = checker(self, rule, analysis, event_stats, match_count)
        return float(confidence), evidence

    async def observe_rule(
        self,
        rule_id: str,
        analysis: MatchAnalysis,
        events: list[dict] | None = None,
        match_count: int = 1,
    ) -> dict[str, Any] | None:
        """Evaluate one rule's checker on one match, without the gates.

        Used by intervention efficacy: the delta it reports must be the
        rule's own observed quantity, computed by the same checker the
        diagnosis used. Deliberately bypasses the confidence and volume
        gates -- those decide whether a *diagnosis* is claimable, not
        whether a quantity can be measured.

        Returns None when the rule or its pattern checker is unknown
        (that is a contract gap, not a zero). Otherwise:

            {
              "rule_id", "pattern_type", "confidence",
              "evidence": {...},
              "hypotheses": [evaluate_hypothesis(...), ...],
              "event_stats": {...},
            }
        """
        await self.initialize()
        rule = self.kb.get_rule(rule_id)
        if rule is None:
            return None

        event_stats = precompute_event_stats(events or [])
        observed = self._run_checker(rule, analysis, event_stats, match_count)
        if observed is None:
            return None
        confidence, evidence = observed
        hypotheses = [
            evaluate_hypothesis(h, evidence, event_stats)
            for h in (rule.hypotheses or [])
            if isinstance(h, dict)
        ]
        return {
            "rule_id": rule.rule_id,
            "pattern_type": str(rule.pattern_signature.get("type", "")),
            "confidence": round(confidence, 3),
            "evidence": evidence,
            "hypotheses": hypotheses,
            "event_stats": event_stats,
        }

    async def _test_rule(
        self,
        rule: TacticalRule,
        analysis: MatchAnalysis,
        event_stats: dict,
        match_count: int = 1,
    ) -> Diagnosis | None:
        """Test if a rule's pattern matches the (possibly pooled) data.

        Returns None if rule doesn't apply, else a Diagnosis whose
        confidence = base pattern strength + hypothesis boosts earned
        from observable evidence, with full provenance attached.
        """
        sig = rule.pattern_signature
        pattern_type = sig.get("type", "")

        observed = self._run_checker(rule, analysis, event_stats, match_count)
        if observed is None:
            return None
        confidence, evidence = observed

        if confidence < 0.3:
            return None

        # Volume gates from the rule's own pattern signature. Rule
        # files use several spellings (min_events, min_goals): each
        # demands that the largest single concrete quantity in the
        # evidence reach the declared minimum before the pattern is
        # claimable at all. Ratios (min_percentage) are the checkers'
        # own concern.
        thresholds = sig.get("thresholds", {})
        for volume_key in ("min_events", "min_goals"):
            minimum = thresholds.get(volume_key)
            if minimum:
                quantities = [v for v in evidence.values() if isinstance(v, (int, float))]
                if not quantities or max(quantities) < minimum:
                    return None

        # min_matches gating (multi-match mode): the rule declares how
        # many distinct matches a pattern must span before the
        # diagnosis counts as confirmed. Provisional diagnoses are
        # still reported — honestly labeled — never suppressed.
        min_matches = sig.get("thresholds", {}).get("min_matches", 1)
        confirmation = "confirmed" if match_count >= min_matches else "provisional"

        # Hypothesis evaluation with provenance: every declared
        # hypothesis gets a criteria record (required metric, raw
        # threshold, observed value, met/known). Boosts apply only to
        # hypotheses whose criteria were all observable AND met —
        # unobservable evidence never earns confidence.
        hyp_results = [
            evaluate_hypothesis(h, evidence, event_stats)
            for h in rule.hypotheses
            if isinstance(h, dict)
        ]
        boost = sum(r["boost_applied"] for r in hyp_results)
        confidence = min(1.0, confidence + boost)
        evidence["_hypotheses_evaluated"] = len(hyp_results)
        evidence["_confidence_boost_applied"] = round(boost, 3)
        evidence["_pattern_type"] = pattern_type
        evidence["_match_count"] = match_count
        evidence["_confirmation"] = confirmation

        primary_hyp = rule.hypotheses[0] if rule.hypotheses else None
        if isinstance(primary_hyp, dict):
            explanation_en = primary_hyp.get("coaching_notes", {}).get("en", "")
            explanation_ar = primary_hyp.get("coaching_notes", {}).get("ar", "")
        else:
            explanation_en = primary_hyp.coaching_notes.get("en", "") if primary_hyp else ""
            explanation_ar = primary_hyp.coaching_notes.get("ar", "") if primary_hyp else ""

        recommended_drill_ids = []
        if primary_hyp:
            if isinstance(primary_hyp, dict):
                recommended_drill_ids = primary_hyp.get("recommended_drills", [])
            elif primary_hyp.recommended_drills:
                recommended_drill_ids = primary_hyp.recommended_drills
        if (
            not recommended_drill_ids
            and rule.recommended_drills
            and isinstance(rule.recommended_drills, list)
        ) and isinstance(rule.recommended_drills[0], dict):
            recommended_drill_ids = [
                d.get("drill_id")
                for d in rule.recommended_drills
                if isinstance(d, dict) and d.get("drill_id")
            ]

        return Diagnosis(
            rule_id=rule.rule_id,
            rule_name=rule.names.get("en", rule.rule_id),
            rule_name_ar=rule.names.get("ar", rule.rule_id),
            category=rule.category,
            severity=rule.severity,
            confidence=round(confidence, 3),
            evidence=evidence,
            explanation=explanation_en,
            explanation_ar=explanation_ar,
            recommended_drills=recommended_drill_ids,
            hypothesis_evaluation=hyp_results,
            pattern_type=pattern_type,
            match_count=match_count,
            confirmation=confirmation,
        )

    _DISPATCH: dict[str, Any] | None = None

    @classmethod
    def _dispatch_map(cls) -> dict[str, Any]:
        """pattern_signature.type -> checker, covering all KB patterns.

        Built once, lazily. Legacy pattern types route through adapter
        wrappers around the original methods (their pre-existing call
        contracts are preserved for tests); every other KB pattern type
        routes to a function in ``reasoning.checkers``. A rule file
        whose pattern has no checker can never fire — the diagnose
        pipeline warns loudly about any such pattern instead of
        silently skipping it.
        """
        if cls._DISPATCH is None:

            def _legacy(method_name):
                """Adapt (rule, analysis, stats) methods to the uniform protocol."""

                def _run(svc, rule, analysis, stats, match_count=1):
                    return getattr(svc, method_name)(rule, analysis, stats)

                _run.__name__ = method_name
                _run.__qualname__ = f"ReasoningService._legacy.{method_name}"
                return _run

            def _check_adapter(func):
                """Adapt (analysis, stats, match_count) checkers to the protocol."""

                def _run(svc, rule, analysis, stats, match_count=1):
                    return func(analysis, stats, match_count)

                _run.__name__ = getattr(func, "__name__", "checker")
                _run.__qualname__ = f"checkers.{_run.__name__}"
                return _run

            cls._DISPATCH = {
                # legacy 11 — adapters over the original methods
                "zone_based_goal_concession": _legacy("_check_zone_concession"),
                "zone_based_possession_loss": _legacy("_check_possession_loss"),
                "through_balls_behind_defense": _legacy("_check_high_line"),
                "counter_attack_conceded": _legacy("_check_counter_attack"),
                "set_piece_goals_conceded": _legacy("_check_set_piece"),
                "low_final_third_entries": _legacy("_check_final_third"),
                "fullback_isolated_1v1": _legacy("_check_fullback_iso"),
                "striker_isolated": _legacy("_check_striker_iso"),
                "high_turnover_rate": _legacy("_check_turnovers"),
                "late_game_decline": _legacy("_check_late_game"),
                "poor_wide_play": _legacy("_check_wide_play"),
                # Phase B — full-checker functions in reasoning.checkers,
                # adapted to the uniform checker protocol
                "low_passing_accuracy": _check_adapter(_checkers.check_low_passing_accuracy),
                "low_shot_conversion": _check_adapter(_checkers.check_low_shot_conversion),
                "poor_corner_kicks": _check_adapter(_checkers.check_poor_corner_kicks),
                "low_dribble_success": _check_adapter(_checkers.check_low_dribble_success),
                "low_vertical_progression": _check_adapter(
                    _checkers.check_low_vertical_progression
                ),
                "high_possession_low_progression": _check_adapter(
                    _checkers.check_high_possession_low_progression
                ),
                "lopsided_attack": _check_adapter(_checkers.check_lopsided_attack),
                "low_creative_actions": _check_adapter(_checkers.check_low_creative_actions),
                "reception_in_zone_gap": _check_adapter(_checkers.check_reception_in_zone_gap),
                "aerial_duel_loss_defensive": _check_adapter(
                    _checkers.check_aerial_duel_loss_defensive
                ),
                "poor_defensive_transition": _check_adapter(
                    _checkers.check_poor_defensive_transition
                ),
                "press_broken_by_simple_pass": _check_adapter(
                    _checkers.check_press_broken_by_simple_pass
                ),
                "high_foul_rate": _check_adapter(_checkers.check_high_foul_rate),
                "turnover_in_defensive_third": _check_adapter(
                    _checkers.check_turnover_in_defensive_third
                ),
                "low_compactness": _check_adapter(_checkers.check_low_compactness),
                "low_offside_pressure": _check_adapter(_checkers.check_low_offside_pressure),
                "offside_trap_ineffective": _check_adapter(
                    _checkers.check_offside_trap_ineffective
                ),
                "offside_failed": _check_adapter(_checkers.check_offside_failed),
                "no_pressure_after_loss": _check_adapter(_checkers.check_no_pressure_after_loss),
                "slow_recovery_after_loss": _check_adapter(
                    _checkers.check_slow_recovery_after_loss
                ),
                "late_goals_conceded": _check_adapter(_checkers.check_late_goals_conceded),
                "defensive_disorganization": _check_adapter(
                    _checkers.check_defensive_disorganization
                ),
                "winger_not_tracking_back": _check_adapter(
                    _checkers.check_winger_not_tracking_back
                ),
                "striker_passive_press": _check_adapter(_checkers.check_striker_passive_press),
                "midfield_disconnected": _check_adapter(_checkers.check_midfield_disconnected),
                "gk_distribution_loss": _check_adapter(_checkers.check_gk_distribution_loss),
                "fullback_not_attacking": _check_adapter(_checkers.check_fullback_not_attacking),
                "cb_positioning_error": _check_adapter(_checkers.check_cb_positioning_error),
                "low_aerial_win_rate": _check_adapter(_checkers.check_low_aerial_win_rate),
            }
        return cls._DISPATCH

    def _ensure_event_stats(self, event_stats: dict | list) -> dict:
        if isinstance(event_stats, list):
            return self._precompute_event_stats(event_stats)
        return event_stats

    def _check_zone_concession(
        self, rule: TacticalRule, analysis: MatchAnalysis, event_stats: dict | list
    ) -> tuple[float, dict]:
        """Check for goals conceded from a specific zone (e.g., left channel)."""
        event_stats = self._ensure_event_stats(event_stats)
        zone_events = [
            e for e in event_stats["goals"] if e.get("zone") == rule.pattern_signature.get("zone")
        ]
        all_goals = event_stats["goals"]
        if not all_goals:
            return 0.0, {}
        pct = len(zone_events) / len(all_goals)
        if pct >= 0.5 and len(zone_events) >= 2:
            confidence = 0.6 + min(0.3, pct - 0.5)
            return confidence, {
                "zone_goals": len(zone_events),
                "total_goals_conceded": len(all_goals),
                "zone_pct": round(pct, 2),
            }
        return 0.0, {}

    def _check_possession_loss(
        self, rule: TacticalRule, analysis: MatchAnalysis, event_stats: dict | list
    ) -> tuple[float, dict]:
        """Check for high possession loss in defensive third."""
        event_stats = self._ensure_event_stats(event_stats)
        def_turnovers = [e for e in event_stats["turnovers"] if e.get("zone") == "defensive_third"]
        all_turnovers = event_stats["turnovers"]
        if not all_turnovers:
            return 0.0, {}
        pct = len(def_turnovers) / len(all_turnovers)
        if pct >= 0.3:
            confidence = 0.5 + min(0.3, (pct - 0.3) * 0.5)
            return confidence, {
                "def_third_turnovers": len(def_turnovers),
                "total_turnovers": len(all_turnovers),
                "pct": round(pct, 2),
            }
        return 0.0, {}

    def _check_high_line(
        self, rule: TacticalRule, analysis: MatchAnalysis, event_stats: dict | list
    ) -> tuple[float, dict]:
        """Check if high defensive line is being exposed."""
        event_stats = self._ensure_event_stats(event_stats)
        formations = analysis.formations
        if not formations:
            return 0.0, {}
        home = formations.get("home", {})
        line_h = home.get("line_height")
        if line_h is None:
            return 0.0, {}
        behind_events = event_stats["behind_def_line_events"]
        if line_h > 0.7 and len(behind_events) >= 2:
            return 0.7, {
                "line_height": round(line_h, 2),
                "through_balls_behind": len(behind_events),
            }
        return 0.0, {}

    def _check_counter_attack(
        self, rule: TacticalRule, analysis: MatchAnalysis, event_stats: dict | list
    ) -> tuple[float, dict]:
        """Check for counter-attack vulnerability."""
        event_stats = self._ensure_event_stats(event_stats)
        ca_shots = event_stats["counter_attack_shots"]
        ca_goals = event_stats["counter_attack_goals"]
        if len(ca_shots) >= 3:
            return 0.6, {
                "counter_attack_shots": len(ca_shots),
                "counter_attack_goals": len(ca_goals),
            }
        return 0.0, {}

    def _check_set_piece(
        self, rule: TacticalRule, analysis: MatchAnalysis, event_stats: dict | list
    ) -> tuple[float, dict]:
        """Check for set piece weakness."""
        event_stats = self._ensure_event_stats(event_stats)
        sp_goals = event_stats["set_piece_goals"]
        all_goals = event_stats["goals"]
        if not all_goals:
            return 0.0, {}
        pct = len(sp_goals) / len(all_goals)
        if pct >= 0.3 and len(sp_goals) >= 2:
            return 0.65, {
                "set_piece_goals": len(sp_goals),
                "pct": round(pct, 2),
            }
        return 0.0, {}

    def _check_final_third(
        self, rule: TacticalRule, analysis: MatchAnalysis, event_stats: dict | list
    ) -> tuple[float, dict]:
        """Check for low final third entries."""
        event_stats = self._ensure_event_stats(event_stats)
        ft_entries = event_stats["final_third_events"]
        shots = event_stats["shots"]
        if not shots:
            return 0.0, {}
        if len(ft_entries) < 15 and len(shots) < 8:
            return 0.55, {
                "ft_entries": len(ft_entries),
                "shots": len(shots),
            }
        return 0.0, {}

    def _check_fullback_iso(
        self, rule: TacticalRule, analysis: MatchAnalysis, event_stats: dict | list
    ) -> tuple[float, dict]:
        """Check for isolated fullback situations."""
        event_stats = self._ensure_event_stats(event_stats)
        iso_events = [e for e in event_stats["1v1_situations"] if e.get("position") == "fullback"]
        if len(iso_events) >= 5:
            opp_success = sum(1 for e in iso_events if e.get("outcome") == "beaten") / len(
                iso_events
            )
            if opp_success > 0.55:
                return 0.6, {
                    "1v1_count": len(iso_events),
                    "opponent_success_rate": round(opp_success, 2),
                }
        return 0.0, {}

    def _check_striker_iso(
        self, rule: TacticalRule, analysis: MatchAnalysis, event_stats: dict | list
    ) -> tuple[float, dict]:
        """Check for isolated striker."""
        event_stats = self._ensure_event_stats(event_stats)
        striker_passes = event_stats["striker_passes"]
        if 0 < len(striker_passes) < 15:
            return 0.5, {
                "passes_to_striker": len(striker_passes),
            }
        return 0.0, {}

    def _check_turnovers(
        self, rule: TacticalRule, analysis: MatchAnalysis, event_stats: dict | list
    ) -> tuple[float, dict]:
        """Check for high turnover rate."""
        event_stats = self._ensure_event_stats(event_stats)
        turnovers = event_stats["turnovers"]
        own_half = event_stats["own_half_turnovers"]
        total_events = event_stats["total_events"]
        if not total_events:
            return 0.0, {}
        turnover_rate = len(turnovers) / total_events
        if len(turnovers) > 40 and len(own_half) / max(1, len(turnovers)) > 0.25:
            return 0.55, {
                "total_turnovers": len(turnovers),
                "own_half_turnovers": len(own_half),
                "turnover_rate": round(turnover_rate, 3),
            }
        return 0.0, {}

    def _check_late_game(
        self, rule: TacticalRule, analysis: MatchAnalysis, event_stats: dict | list
    ) -> tuple[float, dict]:
        """Check for late-game decline."""
        event_stats = self._ensure_event_stats(event_stats)
        last_events = event_stats["late_events"]
        first_events = event_stats["first_events"]

        late_goals = sum(1 for e in last_events if e.get("type") == "goal")
        late_shots = sum(1 for e in last_events if e.get("type") == "shot")
        first_goals = sum(1 for e in first_events if e.get("type") == "goal")
        first_shots = sum(1 for e in first_events if e.get("type") == "shot")

        if late_goals > first_goals and first_shots > 0:
            return 0.6, {
                "late_goals": late_goals,
                "first_quarter_goals": first_goals,
                "late_shots": late_shots,
                "first_quarter_shots": first_shots,
            }
        return 0.0, {}

    def _check_wide_play(
        self, rule: TacticalRule, analysis: MatchAnalysis, event_stats: dict | list
    ) -> tuple[float, dict]:
        """Check for poor wide play."""
        event_stats = self._ensure_event_stats(event_stats)
        crosses = event_stats["crosses"]
        if not crosses:
            return 0.0, {}
        accurate = sum(1 for c in crosses if c.get("outcome") in {"completed", "shot_created"})
        accuracy = accurate / len(crosses) if crosses else 0
        if len(crosses) < 8 and accuracy < 0.25:
            return 0.5, {
                "crosses": len(crosses),
                "accuracy": round(accuracy, 2),
            }
        return 0.0, {}

    def _build_priority_actions(
        self, diagnoses: list[Diagnosis], language: str
    ) -> dict[str, list[str]]:
        """Build prioritized list of actions for the coach."""
        actions_en = []
        actions_ar = []
        for i, diag in enumerate(diagnoses[:5]):
            if diag.recommended_drills:
                drills = [self.kb.get_drill(d_id) for d_id in diag.recommended_drills[:2]]
                typed_drills = [d for d in drills if d is not None]
                if typed_drills:
                    drill_names_en = ", ".join(str(d.name) for d in typed_drills)
                    drill_names_ar = ", ".join(
                        str(getattr(d, "name_ar", None) or d.name) for d in typed_drills
                    )
                    actions_en.append(
                        f"Priority {i + 1}: {drill_names_en} "
                        f"(addresses {diag.rule_name}, confidence {diag.confidence:.0%})"
                    )
                    actions_ar.append(
                        f"الأولوية {i + 1}: {drill_names_ar} "
                        f"(يعالج: {diag.rule_name_ar}, الثقة {diag.confidence:.0%})"
                    )
        if not actions_en:
            actions_en = ["No critical issues detected. Continue current training plan."]
            actions_ar = ["لم يتم اكتشاف مشاكل حرجة. استمر في خطة التدريب الحالية."]
        return {"en": actions_en, "ar": actions_ar}

    def _build_overall_assessment(
        self, diagnoses: list[Diagnosis], analysis: MatchAnalysis, language: str
    ) -> dict[str, str]:
        """Build overall assessment summary."""
        if not diagnoses:
            return {
                "en": (
                    f"Match analysis complete. Possession: "
                    f"{analysis.home_team.possession_pct:.1f}% vs "
                    f"{analysis.away_team.possession_pct:.1f}%. "
                    f"No critical tactical issues detected. "
                    f"Continue monitoring for patterns across multiple matches."
                ),
                "ar": (
                    f"اكتمل تحليل المباراة. الاستحواذ: "
                    f"{analysis.home_team.possession_pct:.1f}% مقابل "
                    f"{analysis.away_team.possession_pct:.1f}%. "
                    f"لم يتم اكتشاف مشاكل تكتيكية حرجة. "
                    f"استمر في المراقبة لاكتشاف الأنماط عبر مباريات متعددة."
                ),
            }

        top = diagnoses[0]
        return {
            "en": (
                f"Match analysis complete. {len(diagnoses)} tactical issues identified. "
                f"Top concern: {top.rule_name} (confidence {top.confidence:.0%}). "
                f"{top.explanation or 'See recommended drills below.'}"
            ),
            "ar": (
                f"اكتمل تحليل المباراة. تم تحديد {len(diagnoses)} مشكلة تكتيكية. "
                f"أهم مشكلة: {top.rule_name_ar} (الثقة {top.confidence:.0%}). "
                f"{top.explanation_ar or 'راجع التدريبات الموصى بها أدناه.'}"
            ),
        }
