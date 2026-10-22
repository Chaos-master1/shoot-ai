"""Tests for Phase xG Breakdown module."""

from kawkab.core.phase_xg import (
    PhaseXgBreakdown,
    PhaseXgReport,
    PossessionPhase,
    _detect_possession_chains,
    _find_chain_for_shot,
    classify_possession_phase,
    compute_phase_xg,
)


def _make_event(
    idx: int,
    etype: str = "pass",
    team: str = "home",
    timestamp: float = 0.0,
    start_x: float = 0.0,
    start_y: float = 34.0,
    end_x: float = 0.0,
    end_y: float = 34.0,
    completed: bool = True,
    is_goal: bool = False,
    on_target: bool = False,
    xg: float = 0.0,
) -> dict:
    ev = {
        "type": etype,
        "team": team,
        "timestamp": timestamp,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "completed": completed,
    }
    if etype == "shot":
        ev["is_goal"] = is_goal
        ev["on_target"] = on_target
        ev["xg"] = xg
    return ev


class TestClassifyPossessionPhase:
    def test_settled_possession_long_chain(self):
        """Many passes and long duration → SETTLED_POSSESSION."""
        events = [_make_event(i, timestamp=float(i)) for i in range(8)]
        chain = events[:7]
        phase = classify_possession_phase(events, 7, chain, set())
        assert phase == PossessionPhase.SETTLED_POSSESSION

    def test_settled_possession_long_duration(self):
        """Few passes but long duration → SETTLED_POSSESSION."""
        events = [
            _make_event(0, timestamp=0.0),
            _make_event(1, timestamp=12.0, etype="shot"),
        ]
        phase = classify_possession_phase(events, 1, events, set())
        assert phase == PossessionPhase.SETTLED_POSSESSION

    def test_transition_attack_few_passes(self):
        """Few passes, short duration, own half start → TRANSITION_ATTACK."""
        events = [
            _make_event(0, timestamp=0.0, start_x=40.0),
            _make_event(1, timestamp=4.0),
            _make_event(2, timestamp=8.0, etype="shot"),
        ]
        chain = events[:2]
        phase = classify_possession_phase(events, 2, chain, set())
        assert phase == PossessionPhase.TRANSITION_ATTACK

    def test_counter_attack_from_deep(self):
        """Start in defensive third (<35m), few passes, short → COUNTER_ATTACK."""
        events = [
            _make_event(0, timestamp=0.0, start_x=20.0),
            _make_event(1, timestamp=3.0),
            _make_event(2, timestamp=6.0, etype="shot"),
        ]
        phase = classify_possession_phase(events, 2, events[:2], set())
        assert phase == PossessionPhase.COUNTER_ATTACK

    def test_set_piece_within_three_events(self):
        """Shot within 3 events of a corner → SET_PIECE."""
        events = [
            _make_event(0, etype="corner", team="home", timestamp=0.0),
            _make_event(1, etype="pass", team="home", timestamp=1.0),
            _make_event(2, etype="shot", team="home", timestamp=2.0),
        ]
        chain = [events[1]]
        phase = classify_possession_phase(events, 2, chain, {0})
        assert phase == PossessionPhase.SET_PIECE

    def test_set_piece_free_kick(self):
        """Free kick directly leading to shot."""
        events = [
            _make_event(0, etype="free_kick", team="home", timestamp=0.0),
            _make_event(1, etype="shot", team="home", timestamp=1.0),
        ]
        phase = classify_possession_phase(events, 1, [], {0})
        assert phase == PossessionPhase.SET_PIECE

    def test_direct_play_long_ball(self):
        """First pass >30m aimed at final third → DIRECT_PLAY."""
        events = [
            _make_event(0, timestamp=0.0, start_x=30.0, end_x=75.0),
            _make_event(1, timestamp=3.0, etype="shot", start_x=75.0),
        ]
        phase = classify_possession_phase(events, 1, events[:1], set())
        assert phase == PossessionPhase.DIRECT_PLAY

    def test_empty_chain_returns_unknown(self):
        """Empty possession chain → UNKNOWN."""
        phase = classify_possession_phase([], 0, [], set())
        assert phase == PossessionPhase.UNKNOWN

    def test_transition_attack_from_middle(self):
        """Own half start (not defensive third), few passes → TRANSITION."""
        events = [
            _make_event(0, timestamp=0.0, start_x=45.0),
            _make_event(1, timestamp=2.0),
            _make_event(2, timestamp=5.0, etype="shot"),
        ]
        phase = classify_possession_phase(events, 2, events[:2], set())
        assert phase == PossessionPhase.TRANSITION_ATTACK


class TestDetectPossessionChains:
    def test_single_team_chain(self):
        events = [
            _make_event(0, team="home", timestamp=0.0),
            _make_event(1, team="home", timestamp=1.0),
        ]
        chains = _detect_possession_chains(events)
        assert len(chains) == 1
        assert len(chains[0]) == 2

    def test_two_teams_two_chains(self):
        events = [
            _make_event(0, team="home", timestamp=0.0),
            _make_event(1, team="home", timestamp=1.0),
            _make_event(2, team="away", timestamp=2.0),
            _make_event(3, team="away", timestamp=3.0),
        ]
        chains = _detect_possession_chains(events)
        assert len(chains) == 2
        assert len(chains[0]) == 2
        assert len(chains[1]) == 2

    def test_events_without_team_skip(self):
        events = [
            _make_event(0, team="home", timestamp=0.0),
            {"type": "ball_out", "timestamp": 0.5},
            _make_event(1, team="home", timestamp=1.0),
        ]
        chains = _detect_possession_chains(events)
        assert len(chains) == 1

    def test_empty_events(self):
        assert _detect_possession_chains([]) == []

    def test_single_event(self):
        events = [_make_event(0, team="home", timestamp=0.0)]
        chains = _detect_possession_chains(events)
        assert len(chains) == 1
        assert chains[0] == [0]


class TestFindChainForShot:
    def test_finds_correct_chain(self):
        chains = [[0, 1], [2, 3, 4]]
        assert _find_chain_for_shot(3, chains) == [2, 3, 4]

    def test_none_for_absent_shot(self):
        assert _find_chain_for_shot(99, [[0], [1]]) is None


class TestComputePhaseXg:
    def test_empty_events_returns_empty_report(self):
        report = compute_phase_xg([], [], [])
        assert report.team == "home"
        assert report.team_breakdown.totals.get("shots", 0) == 0
        assert report.opponent_breakdown.totals.get("shots", 0) == 0
        assert report.match_id == ""

    def test_shots_classified_in_phases(self):
        events = [
            _make_event(0, team="home", timestamp=0.0, start_x=40.0, end_x=50.0),
            _make_event(1, team="home", timestamp=1.0, start_x=50.0, end_x=60.0),
            _make_event(2, team="home", timestamp=2.0, start_x=60.0, end_x=70.0),
            _make_event(3, team="home", timestamp=3.0, start_x=70.0, end_x=80.0),
            _make_event(4, team="home", timestamp=4.0, start_x=80.0, end_x=85.0),
            _make_event(5, team="home", timestamp=5.0, etype="shot", xg=0.5, on_target=True),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        report = compute_phase_xg(home_events, [], events)
        tb = report.team_breakdown
        assert tb.totals["shots"] == 1
        assert tb.phases["settled_possession"]["shots"] >= 1

    def test_opponent_breakdown_populated(self):
        events = [
            _make_event(0, team="home", timestamp=0.0),
            _make_event(1, team="away", timestamp=1.0),
            _make_event(2, team="away", timestamp=2.0, etype="shot", xg=0.3),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        away_events = [e for e in events if e.get("team") == "away"]
        report = compute_phase_xg(home_events, away_events, events)
        assert report.opponent_breakdown.totals["shots"] >= 1

    def test_phase_distribution_sums_to_100(self):
        events = [
            _make_event(0, team="home", timestamp=0.0, start_x=40.0, end_x=50.0),
            _make_event(1, team="home", timestamp=1.0, start_x=50.0, end_x=60.0),
            _make_event(2, team="home", timestamp=2.0, start_x=60.0, end_x=70.0),
            _make_event(3, team="home", timestamp=3.0, start_x=70.0, end_x=80.0),
            _make_event(4, team="home", timestamp=4.0, start_x=80.0, end_x=85.0),
            _make_event(5, team="home", timestamp=5.0, etype="shot", xg=0.5),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        report = compute_phase_xg(home_events, [], events)
        total_pct = sum(report.team_breakdown.phase_distribution_pct.values())
        assert abs(total_pct - 100.0) < 0.1 or total_pct == 0.0

    def test_set_piece_phase_detected(self):
        events = [
            _make_event(0, etype="corner", team="home", timestamp=0.0),
            _make_event(1, team="home", timestamp=1.0, start_x=80.0, end_x=85.0),
            _make_event(2, team="home", timestamp=2.0, etype="shot", xg=0.4),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        report = compute_phase_xg(home_events, [], events)
        assert report.team_breakdown.phases["set_piece"]["shots"] >= 1
        assert report.team_breakdown.phases["set_piece"]["xg"] >= 0.3

    def test_transition_phase_populated(self):
        events = [
            _make_event(0, team="home", timestamp=0.0, start_x=45.0, end_x=55.0),
            _make_event(1, team="home", timestamp=2.0, start_x=55.0, end_x=65.0),
            _make_event(2, team="home", timestamp=4.0, etype="shot", xg=0.2),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        report = compute_phase_xg(home_events, [], events)
        assert report.team_breakdown.phases["transition_attack"]["shots"] >= 1


class TestPhaseXgReport:
    def test_summary_text_non_empty(self):
        bd = PhaseXgBreakdown(
            team="home",
            phases={
                "settled_possession": {
                    "shots": 3,
                    "goals": 1,
                    "xg": 1.2,
                    "shots_on_target": 2,
                    "avg_xg_per_shot": 0.4,
                },
            },
            totals={"shots": 3, "goals": 1, "xg": 1.2},
            phase_distribution_pct={"settled_possession": 100.0},
        )
        obd = PhaseXgBreakdown(
            team="away",
            phases={},
            totals={"shots": 0, "goals": 0, "xg": 0.0},
            phase_distribution_pct={},
        )
        report = PhaseXgReport(
            team="home", match_id="match_1", team_breakdown=bd, opponent_breakdown=obd
        )
        text = report.summary_text()
        assert isinstance(text, str)
        assert len(text) > 20
        assert "Phase xG Breakdown" in text

    def test_to_dict(self):
        bd = PhaseXgBreakdown(
            team="home",
            phases={},
            totals={"shots": 0, "goals": 0, "xg": 0.0},
            phase_distribution_pct={},
        )
        obd = PhaseXgBreakdown(
            team="away",
            phases={},
            totals={"shots": 0, "goals": 0, "xg": 0.0},
            phase_distribution_pct={},
        )
        report = PhaseXgReport(
            team="home", match_id="m1", team_breakdown=bd, opponent_breakdown=obd
        )
        d = report.to_dict()
        assert d["team"] == "home"
        assert d["match_id"] == "m1"

    def test_goal_counter_accumulated(self):
        events = [
            _make_event(0, team="home", timestamp=0.0, start_x=40.0, end_x=50.0),
            _make_event(1, team="home", timestamp=1.0, start_x=50.0, end_x=60.0),
            _make_event(2, team="home", timestamp=2.0, start_x=60.0, end_x=70.0),
            _make_event(3, team="home", timestamp=3.0, start_x=70.0, end_x=80.0),
            _make_event(4, team="home", timestamp=4.0, start_x=80.0, end_x=85.0),
            _make_event(
                5, team="home", timestamp=5.0, etype="shot", xg=0.6, is_goal=True, on_target=True
            ),
        ]
        home_events = [e for e in events if e.get("team") == "home"]
        report = compute_phase_xg(home_events, [], events)
        assert report.team_breakdown.totals["goals"] == 1
