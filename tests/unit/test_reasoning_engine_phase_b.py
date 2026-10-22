"""Phase B reasoning-engine tests: dispatch completeness, hypothesis
evidence evaluation, volume gates, multi-match mode, and persistence.

The structural guarantee lives in TestDispatchCompleteness: every
pattern type declared in a knowledge-base YAML rule file must have a
checker in the dispatch map. A rule file without a checker can never
fire — that was the audit's core finding, so it is now a CI failure.
"""

from __future__ import annotations

import asyncio
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Real knowledge base, immune to test-order effects: other test files
# call install_kawkab_stubs(), which replaces kawkab.core.paths with a
# per-process stub whose knowledge_base dir is empty. KnowledgeService
# resolves its KB root through kawkab.services.knowledge_service's own
# get_paths import, so pinning that symbol to the real repo knowledge
# dir makes these tests independent of which stub ran first.
_REAL_KB_ROOT = Path(__file__).resolve().parents[2] / "src" / "kawkab" / "knowledge"

from kawkab.services.analysis.core import MatchAnalysis, TeamStats
from kawkab.services.knowledge_service import KnowledgeService
from kawkab.services.reasoning import checkers as checker_mod
from kawkab.services.reasoning.metrics import (
    evaluate_hypothesis,
    extract_metric,
    precompute_event_stats,
)
from kawkab.services.reasoning_service import ReasoningService

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _mk_analysis(match_id=1, events=None, possession=50.0):
    home = TeamStats(team_name="Home")
    away = TeamStats(team_name="Away")
    home.possession_pct = possession
    away.possession_pct = 100.0 - possession
    return MatchAnalysis(
        match_id=match_id,
        duration_seconds=5400.0,
        home_team=home,
        away_team=away,
        players={},
        events=events or [],
    )


def _rule(pattern_type, zone=None, thresholds=None, hypotheses=None):
    """Minimal TacticalRule compatible with the dispatch path."""
    from kawkab.services.knowledge_service import TacticalRule

    sig = {"type": pattern_type}
    if zone:
        sig["zone"] = zone
    if thresholds:
        sig["thresholds"] = thresholds
    return TacticalRule(
        rule_id="test_rule",
        category="defensive",
        subcategory="",
        severity="medium",
        names={"en": "Test", "ar": "اختبار"},
        description={"en": "", "ar": ""},
        pattern_signature=sig,
        hypotheses=hypotheses or [],
        recommended_drills=[],
    )


def _evidence_conf(rule, events, match_count=1, analysis=None):
    svc = ReasoningService(mock.MagicMock())
    stats = precompute_event_stats(events)
    return asyncio.run(
        svc._test_rule(rule, analysis or _mk_analysis(events=events), stats, match_count)
    )


# ---------------------------------------------------------------------
# Dispatch completeness — the structural guarantee
# ---------------------------------------------------------------------


class TestDispatchCompleteness:
    def test_every_kb_pattern_type_has_a_checker(self):
        """Every pattern_signature.type in the KB rule files dispatches.

        Greps the rule YAMLs for their type declarations and asserts
        each one has an entry in the engine's dispatch map. A new rule
        file without a checker fails here — the inert-KB regression
        can never silently return.
        """
        kb_root = Path(__file__).resolve().parents[2] / "src" / "kawkab" / "knowledge" / "tactics"
        declared: set[str] = set()
        for yml in kb_root.rglob("*.yaml"):
            for line in yml.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("type: "):
                    declared.add(stripped.split("type: ", 1)[1].strip().strip("\"'"))
        assert declared, "pattern-type scan found nothing — scan is broken"
        dispatch = ReasoningService._dispatch_map()
        missing = declared - set(dispatch)
        assert not missing, f"KB pattern types with no checker: {sorted(missing)}"

    def test_dispatch_has_40_entries(self):
        dispatch = ReasoningService._dispatch_map()
        assert len(dispatch) == 40

    def test_unmatched_pattern_warns(self, monkeypatch):
        """A rule whose pattern has no checker is surfaced, not silent."""
        records: list[str] = []

        class _Capture:
            def warning(self, msg):
                records.append(str(msg))

            def info(self, msg):
                pass

        monkeypatch.setattr("kawkab.services.reasoning_service.logger", _Capture())
        svc = ReasoningService(mock.MagicMock())
        rule = _rule("totally_unknown_pattern")
        svc.kb.get_all_rules.return_value = [rule]
        svc._initialized = True
        asyncio.run(svc.diagnose_match(_mk_analysis()))
        assert records and "no checker" in records[-1]


# ---------------------------------------------------------------------
# Hypothesis evidence evaluation
# ---------------------------------------------------------------------


class TestHypothesisEvidence:
    def test_threshold_parsing_and_met(self):
        hyp = {
            "hypothesis_id": "h1",
            "weight": 0.4,
            "confidence_boost": 0.35,
            "evidence_required": [{"metric": "zone_goals", "threshold": ">2"}],
        }
        result = evaluate_hypothesis(hyp, {"zone_goals": 3}, {})
        assert result["evaluated"] is True
        assert result["met_all"] is True
        assert result["boost_applied"] == 0.35
        assert result["criteria"][0]["observed"] == 3.0
        assert result["criteria"][0]["met"] is True

    def test_threshold_not_met_no_boost(self):
        hyp = {
            "hypothesis_id": "h1",
            "confidence_boost": 0.3,
            "evidence_required": [{"metric": "offsides_won", "threshold": ">5"}],
        }
        result = evaluate_hypothesis(hyp, {}, {"offsides_won": [1, 2, 3]})
        assert result["criteria"][0]["observed"] == 3.0
        assert result["met_all"] is False
        assert result["boost_applied"] == 0.0

    def test_unknown_metric_is_not_a_zero(self):
        """Unobservable evidence must be 'unknown', never 0."""
        result = evaluate_hypothesis(
            {"confidence_boost": 0.3, "evidence_required": [{"metric": "nope", "threshold": ">1"}]},
            {},
            {},
        )
        c = result["criteria"][0]
        assert c["known"] is False
        assert c["observed"] is None
        assert result["evaluated"] is False
        assert result["boost_applied"] == 0.0

    def test_unit_suffix_thresholds_parse(self):
        assert extract_metric.__module__  # sanity
        from kawkab.services.reasoning.metrics import _parse_threshold

        assert _parse_threshold(">5m") == (">", 5.0)
        assert _parse_threshold("<-15m") == ("<", -15.0)
        assert _parse_threshold(">=2.5s") == (">=", 2.5)
        assert _parse_threshold("3") == (">=", 3.0)
        assert _parse_threshold("not-a-number") is None

    def test_extract_metric_prefers_checker_evidence(self):
        assert extract_metric("zone_goals", {"zone_goals": 4}, {}) == 4.0

    def test_extract_metric_reads_event_buckets(self):
        stats = precompute_event_stats(
            [{"type": "goal", "timestamp": 1.0, "team": "away", "zone": "left_third"}]
        )
        assert extract_metric("goals_from_through_balls", {}, stats) == 1.0

    def test_boost_applies_into_diagnosis(self):
        """A met hypothesis raises the diagnosis confidence above base."""
        rule = _rule(
            "zone_based_goal_concession",
            zone="left_third",
            hypotheses=[
                {
                    "id": "h_met",
                    "confidence_boost": 0.2,
                    "evidence_required": [{"metric": "zone_goals", "threshold": ">=2"}],
                    "coaching_notes": {"en": "notes", "ar": "ملاحظات"},
                }
            ],
        )
        events = [
            {"type": "goal", "timestamp": 10.0 * i, "team": "away", "zone": "left_third"}
            for i in range(3)
        ]
        diag = _evidence_conf(rule, events)
        assert diag is not None
        base = 0.6 + min(0.3, 1.0 - 0.5)
        assert diag.confidence == pytest.approx(min(1.0, base + 0.2), abs=0.01)
        assert diag.hypothesis_evaluation[0]["boost_applied"] == 0.2
        assert diag.hypothesis_evaluation[0]["hypothesis_id"] == "h_met"


# ---------------------------------------------------------------------
# Volume gates and confirmation tiers
# ---------------------------------------------------------------------


class TestGatesAndConfirmation:
    def test_min_goals_gate_blocks(self):
        rule = _rule(
            "zone_based_goal_concession",
            zone="left_third",
            thresholds={"min_goals": 5, "min_matches": 3},
        )
        events = [
            {"type": "goal", "timestamp": 10.0 * i, "team": "away", "zone": "left_third"}
            for i in range(3)
        ]
        assert _evidence_conf(rule, events) is None

    def test_min_matches_single_match_is_provisional(self):
        rule = _rule(
            "zone_based_goal_concession",
            zone="left_third",
            thresholds={"min_goals": 3, "min_matches": 3},
        )
        events = [
            {"type": "goal", "timestamp": 10.0 * i, "team": "away", "zone": "left_third"}
            for i in range(3)
        ]
        diag = _evidence_conf(rule, events, match_count=1)
        assert diag is not None
        assert diag.confirmation == "provisional"

    def test_min_matches_met_is_confirmed(self):
        rule = _rule(
            "zone_based_goal_concession",
            zone="left_third",
            thresholds={"min_goals": 3, "min_matches": 3},
        )
        events = [
            {"type": "goal", "timestamp": 10.0 * i, "team": "away", "zone": "left_third"}
            for i in range(3)
        ]
        diag = _evidence_conf(rule, events, match_count=3)
        assert diag is not None
        assert diag.confirmation == "confirmed"
        assert diag.match_count == 3

    def test_min_events_gate_blocks(self):
        rule = _rule(
            "through_balls_behind_defense",
            thresholds={"min_events": 100, "min_matches": 3},
        )
        svc = ReasoningService(mock.MagicMock())
        analysis = _mk_analysis()
        analysis.formations = {"home": {"line_height": 0.9}}
        stats = precompute_event_stats(
            [{"type": "pass", "timestamp": 1.0, "zone": "behind_defensive_line"} for _ in range(2)]
        )
        result = asyncio.run(svc._test_rule(rule, analysis, stats, 1))
        assert result is None


# ---------------------------------------------------------------------
# Multi-match mode
# ---------------------------------------------------------------------


class TestMultiMatchMode:
    @pytest.fixture(autouse=True)
    def _real_kb(self, monkeypatch):
        import kawkab.services.knowledge_service as ks_mod

        class _RealPaths:
            knowledge_base = _REAL_KB_ROOT

        monkeypatch.setattr(ks_mod, "get_paths", lambda: _RealPaths())

    def test_diagnose_matches_pools_and_confirms(self):
        """Three pooled matches with 3 left-third goals confirm the rule."""
        kb = KnowledgeService()
        asyncio.run(kb.initialize())
        svc = ReasoningService(kb)
        analyses = []
        for mid in (1, 2, 3):
            events = [
                {"type": "goal", "timestamp": 10.0 * i, "team": "away", "zone": "left_third"}
                for i in range(3)
            ]
            analyses.append(_mk_analysis(match_id=mid, events=events))
        report = asyncio.run(svc.diagnose_matches(analyses))
        assert report.multi_match is True
        assert report.match_ids == [1, 2, 3]
        zone = [d for d in report.diagnoses if d.pattern_type == "zone_based_goal_concession"]
        assert zone, "zone rule should fire on pooled evidence"
        assert zone[0].confirmation == "confirmed"

    def test_single_match_never_confirms_at_min_matches_3(self):
        kb = KnowledgeService()
        asyncio.run(kb.initialize())
        svc = ReasoningService(kb)
        events = [
            {"type": "goal", "timestamp": 10.0 * i, "team": "away", "zone": "left_third"}
            for i in range(3)
        ]
        report = asyncio.run(svc.diagnose_match(_mk_analysis(events=events)))
        zone = [d for d in report.diagnoses if d.pattern_type == "zone_based_goal_concession"]
        assert zone and zone[0].confirmation == "provisional"

    def test_diagnose_matches_empty_raises(self):
        svc = ReasoningService(mock.MagicMock())
        with pytest.raises(ValueError):
            asyncio.run(svc.diagnose_matches([]))

    def test_aggregation_sums_and_averages(self):
        a1 = _mk_analysis(1, possession=60.0)
        a2 = _mk_analysis(2, possession=50.0)
        a1.home_team.shots = 4
        a2.home_team.shots = 6
        merged = ReasoningService._aggregate_analyses([a1, a2])
        assert merged.home_team.shots == 10
        assert merged.home_team.possession_pct == pytest.approx(55.0)

    def test_diagnosis_metadata_travels_through_report(self):
        report = asyncio.run(ReasoningService(KnowledgeService()).diagnose_match(_mk_analysis()))
        for d in report.diagnoses:
            assert d.pattern_type
            assert d.match_count == 1
            assert d.confirmation in {"provisional", "confirmed"}
            assert "_hypotheses_evaluated" in d.evidence
            assert "_confidence_boost_applied" in d.evidence


# ---------------------------------------------------------------------
# New checkers behave on their declared metrics
# ---------------------------------------------------------------------


class TestNewCheckers:
    def test_low_passing_accuracy_fires(self):
        events = [{"type": "pass", "timestamp": float(i), "completed": False} for i in range(25)]
        conf, ev = checker_mod.check_low_passing_accuracy(None, precompute_event_stats(events), 1)
        assert conf >= 0.5 and ev["pass_accuracy"] < 0.1

    def test_low_passing_accuracy_unknown_completion_is_honest(self):
        events = [{"type": "pass", "timestamp": float(i)} for i in range(25)]
        conf, ev = checker_mod.check_low_passing_accuracy(None, precompute_event_stats(events), 1)
        assert conf == 0.0 and ev == {}

    def test_high_foul_rate_scales_with_matches(self):
        """25 fouls: one match is indiscipline, two is normal."""
        events = [{"type": "foul", "timestamp": float(i), "team": "home"} for i in range(25)]
        conf_one, _ = checker_mod.check_high_foul_rate(None, precompute_event_stats(events), 1)
        conf_two, _ = checker_mod.check_high_foul_rate(None, precompute_event_stats(events), 2)
        assert conf_one > 0  # 25 fouls in one match >= 15/match
        assert conf_two == 0.0  # 12.5/match below threshold

    def test_offside_pressure_scales_with_matches(self):
        """Zero traps won, 4 balls behind: fires for one match (no
        offsides won at all), vanishes when pooled over two (4 balls
        behind is below the 3/match pooled bar of 6)."""
        events = [
            {"type": "pass", "timestamp": 1.0, "zone": "behind_defensive_line"},
            {"type": "pass", "timestamp": 2.0, "zone": "behind_defensive_line"},
            {"type": "pass", "timestamp": 4.0, "zone": "behind_defensive_line"},
            {"type": "pass", "timestamp": 5.0, "zone": "behind_defensive_line"},
        ]
        stats = precompute_event_stats(events)
        conf_one, ev_one = checker_mod.check_low_offside_pressure(None, stats, 1)
        conf_two, _ = checker_mod.check_low_offside_pressure(None, stats, 2)
        assert conf_one > 0 and ev_one["offsides_won"] == 0
        assert conf_two == 0.0

    def test_every_checker_signature_is_uniform(self):
        for name in dir(checker_mod):
            if name.startswith("check_"):
                fn = getattr(checker_mod, name)
                params = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])
                assert params == ["analysis", "event_stats", "match_count"], name
