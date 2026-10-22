"""Tests for trap → transition linkage."""

from kawkab.core.pressing_traps import PressingTrap
from kawkab.core.transitions import PhaseTransition
from kawkab.core.trap_transition_linkage import (
    TrapTransitionAnalysis,
    TrapTransitionLink,
    analyze_trap_transitions,
    summarize_trap_transition,
)


def _make_ev(etype, team, ts, x, y, **kw):
    ev = {"type": etype, "team": team, "timestamp": ts, "x": x, "y": y}
    ev.update(kw)
    return ev


def _make_trap(zone_name, x_range, y_range, regains=0):
    return PressingTrap(
        zone_name=zone_name,
        zone_x_range=x_range,
        zone_y_range=y_range,
        trigger_event_type="backward_pass",
        defensive_actions_in_zone=5,
        opponent_passes_into_zone=3,
        regain_possession_count=regains,
        success_rate=regains / 5.0,
        intensity=2.0,
        trap_rating=0.5,
    )


def _make_transition(team, ts, start_x, start_y):
    return PhaseTransition(
        timestamp=ts,
        team=team,
        transition_type="counter_attack",
        start_x=start_x,
        start_y=start_y,
        duration_s=3.0,
        speed_mps=8.0,
        outcome="shot",
        ended_in_final_third=True,
    )


# Central-mid zone midpoint: ((34.65+70.35)/2, (17+51)/2) = (52.5, 34.0)
CENTRAL_MID_XR = (34.65, 70.35)
CENTRAL_MID_YR = (17.0, 51.0)
CENTRAL_MID_XM = 52.5
CENTRAL_MID_YM = 34.0


class TestAnalyzeTrapTransitions:
    def test_empty_traps(self):
        analysis = analyze_trap_transitions([], [], [])
        assert analysis.total_traps == 0
        assert analysis.successful_traps == 0
        assert analysis.transitions_from_traps == []
        assert analysis.conversion_rate == 0.0

    def test_empty_transitions(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34),
            _make_ev("pass", "away", 2.0, 50, 34),
            _make_ev("tackle", "home", 3.0, 52, 34),
            _make_ev("pass", "home", 3.5, 55, 34),
            _make_ev("tackle", "home", 5.0, 52, 34),
            _make_ev("pass", "home", 5.5, 55, 34),
        ]
        traps = [_make_trap("central_mid", CENTRAL_MID_XR, CENTRAL_MID_YR, regains=2)]
        analysis = analyze_trap_transitions(traps, [], events)
        assert analysis.total_traps == 1
        assert analysis.successful_traps == 1
        assert analysis.transitions_from_traps == []
        assert analysis.conversion_rate == 0.0

    def test_traps_no_regains_zero_conversion(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34),
            _make_ev("tackle", "home", 3.0, 52, 34),
        ]
        traps = [_make_trap("central_mid", CENTRAL_MID_XR, CENTRAL_MID_YR, regains=0)]
        transitions = [
            _make_transition("home", 5.0, CENTRAL_MID_XM, CENTRAL_MID_YM),
        ]
        analysis = analyze_trap_transitions(traps, transitions, events)
        assert analysis.total_traps == 1
        assert analysis.successful_traps == 0
        assert analysis.transitions_from_traps == []
        assert analysis.conversion_rate == 0.0

    def test_full_trap_to_transition_to_shot(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34),
            _make_ev("pass", "away", 2.0, 50, 34),
            _make_ev("tackle", "home", 3.0, 52, 34),
            _make_ev("pass", "home", 3.5, 55, 34),
            _make_ev("shot", "home", 12.0, 80, 34, is_goal=False),
        ]
        traps = [_make_trap("central_mid", CENTRAL_MID_XR, CENTRAL_MID_YR, regains=2)]
        transitions = [
            _make_transition("home", 4.0, CENTRAL_MID_XM + 5, CENTRAL_MID_YM),
        ]
        analysis = analyze_trap_transitions(traps, transitions, events)
        assert analysis.total_traps == 1
        assert analysis.successful_traps == 1
        assert len(analysis.transitions_from_traps) == 1
        link = analysis.transitions_from_traps[0]
        assert link.shot_created
        assert not link.goal_scored
        assert link.time_delta <= 3.0
        assert link.spatial_distance <= 20.0
        assert analysis.conversion_rate == 1.0
        assert analysis.goal_conversion_rate == 0.0

    def test_full_trap_to_transition_to_goal(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34),
            _make_ev("pass", "away", 2.0, 50, 34),
            _make_ev("tackle", "home", 3.0, 52, 34),
            _make_ev("pass", "home", 3.5, 55, 34),
            _make_ev("shot", "home", 12.0, 80, 34, is_goal=True),
        ]
        traps = [_make_trap("central_mid", CENTRAL_MID_XR, CENTRAL_MID_YR, regains=2)]
        transitions = [
            _make_transition("home", 4.0, CENTRAL_MID_XM + 5, CENTRAL_MID_YM),
        ]
        analysis = analyze_trap_transitions(traps, transitions, events)
        assert analysis.total_traps == 1
        assert len(analysis.transitions_from_traps) == 1
        link = analysis.transitions_from_traps[0]
        assert link.shot_created
        assert link.goal_scored
        assert analysis.goal_conversion_rate == 1.0

    def test_temporal_threshold_exceeded(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34),
            _make_ev("pass", "away", 2.0, 50, 34),
            _make_ev("tackle", "home", 3.0, 52, 34),
            _make_ev("pass", "home", 3.5, 55, 34),
        ]
        traps = [_make_trap("central_mid", CENTRAL_MID_XR, CENTRAL_MID_YR, regains=2)]
        transitions = [
            _make_transition("home", 7.0, CENTRAL_MID_XM + 5, CENTRAL_MID_YM),
        ]
        analysis = analyze_trap_transitions(traps, transitions, events)
        assert len(analysis.transitions_from_traps) == 0

    def test_spatial_threshold_exceeded(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34),
            _make_ev("pass", "away", 2.0, 50, 34),
            _make_ev("tackle", "home", 3.0, 52, 34),
            _make_ev("pass", "home", 3.5, 55, 34),
        ]
        traps = [_make_trap("central_mid", CENTRAL_MID_XR, CENTRAL_MID_YR, regains=2)]
        transitions = [
            _make_transition("home", 4.0, CENTRAL_MID_XM + 50, CENTRAL_MID_YM + 30),
        ]
        analysis = analyze_trap_transitions(traps, transitions, events)
        assert len(analysis.transitions_from_traps) == 0

    def test_conversion_rates_computed_correctly(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34),
            _make_ev("pass", "away", 2.0, 50, 34),
            _make_ev("tackle", "home", 3.0, 52, 34),
            _make_ev("pass", "home", 3.5, 55, 34),
            _make_ev("shot", "home", 12.0, 80, 34, is_goal=True),
        ]
        traps = [_make_trap("central_mid", CENTRAL_MID_XR, CENTRAL_MID_YR, regains=1)]
        transitions = [_make_transition("home", 4.0, CENTRAL_MID_XM + 5, CENTRAL_MID_YM)]
        analysis = analyze_trap_transitions(traps, transitions, events)
        assert analysis.conversion_rate == 1.0
        assert analysis.goal_conversion_rate == 1.0
        assert analysis.avg_transition_time == 0.5

    def test_partial_conversion_rates(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34),
            _make_ev("pass", "away", 2.0, 50, 34),
            _make_ev("tackle", "home", 3.0, 52, 34),
            _make_ev("pass", "home", 3.5, 55, 34),
            _make_ev("shot", "home", 12.0, 80, 34, is_goal=False),
            _make_ev("pass", "away", 50.0, 50, 34),
            _make_ev("pass", "away", 51.0, 50, 34),
            _make_ev("tackle", "home", 52.0, 52, 34),
            _make_ev("pass", "home", 52.5, 55, 34),
            _make_ev("shot", "home", 60.0, 80, 34, is_goal=True),
        ]
        traps = [_make_trap("central_mid", CENTRAL_MID_XR, CENTRAL_MID_YR, regains=2)]
        transitions = [
            _make_transition("home", 4.0, CENTRAL_MID_XM + 5, CENTRAL_MID_YM),
            _make_transition("home", 53.5, CENTRAL_MID_XM + 5, CENTRAL_MID_YM),
        ]
        analysis = analyze_trap_transitions(traps, transitions, events)
        assert len(analysis.transitions_from_traps) == 2
        assert analysis.conversion_rate == 1.0
        assert analysis.goal_conversion_rate == 1.0

    def test_wrong_team_transition_not_linked(self):
        events = [
            _make_ev("pass", "home", 1.0, 50, 34),
            _make_ev("pass", "home", 2.0, 50, 34),
            _make_ev("tackle", "away", 3.0, 52, 34),
            _make_ev("pass", "away", 3.5, 55, 34),
        ]
        traps = [_make_trap("central_mid", CENTRAL_MID_XR, CENTRAL_MID_YR, regains=2)]
        transitions = [
            _make_transition("home", 4.0, CENTRAL_MID_XM + 5, CENTRAL_MID_YM),
        ]
        analysis = analyze_trap_transitions(traps, transitions, events)
        assert len(analysis.transitions_from_traps) == 0

    def test_avg_transition_time(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34),
            _make_ev("pass", "away", 2.0, 50, 34),
            _make_ev("tackle", "home", 3.0, 52, 34),
            _make_ev("pass", "home", 3.5, 55, 34),
        ]
        traps = [_make_trap("central_mid", CENTRAL_MID_XR, CENTRAL_MID_YR, regains=2)]
        transitions = [
            _make_transition("home", 4.0, CENTRAL_MID_XM + 5, CENTRAL_MID_YM),
        ]
        analysis = analyze_trap_transitions(traps, transitions, events)
        if analysis.transitions_from_traps:
            assert analysis.avg_transition_time == 0.5


class TestSummarizeTrapTransition:
    def test_empty_analysis(self):
        analysis = TrapTransitionAnalysis(total_traps=0, successful_traps=0)
        summary = summarize_trap_transition(analysis)
        assert "No pressing traps detected" in summary["trap_frequency"]
        assert "0% of pressing traps" in summary["chance_conversion"]
        assert "No trap-to-transition links" in summary["avg_transition_time"]

    def test_with_links(self):
        analysis = TrapTransitionAnalysis(
            total_traps=5,
            successful_traps=3,
            transitions_from_traps=[
                TrapTransitionLink(
                    trap_index=0,
                    transition_index=0,
                    time_delta=1.5,
                    spatial_distance=8.0,
                    goal_scored=False,
                    shot_created=True,
                ),
            ],
            conversion_rate=0.333,
            goal_conversion_rate=0.0,
            avg_transition_time=1.5,
        )
        summary = summarize_trap_transition(analysis)
        assert "18.0 minutes" in summary["trap_frequency"]
        assert "33%" in summary["chance_conversion"]
        assert "1.5 seconds" in summary["avg_transition_time"]

    def test_no_links(self):
        analysis = TrapTransitionAnalysis(
            total_traps=3,
            successful_traps=2,
            conversion_rate=0.0,
            goal_conversion_rate=0.0,
            avg_transition_time=0.0,
        )
        summary = summarize_trap_transition(analysis)
        assert "30.0 minutes" in summary["trap_frequency"]
        assert "0%" in summary["chance_conversion"]
        assert "No trap-to-transition links" in summary["avg_transition_time"]
