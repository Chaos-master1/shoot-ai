"""Tests for pressing trap detection module — zone classification,
trigger detection, and full pipeline."""

from kawkab.core.pressing_traps import (
    PressingTrap,
    PressingTrapReport,
    _classify_trap_zone,
    _compute_trap_rating,
    _find_trigger_events,
    detect_pressing_traps,
)

# ── Constants ──────────────────────────────────────────────────────────────

PL = 105.0
PW = 68.0

# Convenient boundaries
DEF_MAX = PL * 0.33  # 34.65
MID_MIN = PL * 0.33
MID_MAX = PL * 0.67  # 70.35
ATT_MIN = PL * 0.67

LEFT_MAX = PW * 0.25  # 17.0
RIGHT_MIN = PW * 0.75  # 51.0


def _make_ev(etype, team, ts, x, y, **kw):
    ev = {"type": etype, "team": team, "timestamp": ts, "x": x, "y": y}
    ev.update(kw)
    return ev


class TestClassifyTrapZone:
    """_classify_trap_zone — all 9 zones plus boundaries."""

    def test_left_defensive(self):
        z = _classify_trap_zone(10, 5)
        assert z == "left_defensive"

    def test_left_mid(self):
        z = _classify_trap_zone(50, 5)
        assert z == "left_mid"

    def test_left_attacking(self):
        z = _classify_trap_zone(90, 5)
        assert z == "left_attacking"

    def test_right_defensive(self):
        z = _classify_trap_zone(10, 60)
        assert z == "right_defensive"

    def test_right_mid(self):
        z = _classify_trap_zone(50, 60)
        assert z == "right_mid"

    def test_right_attacking(self):
        z = _classify_trap_zone(90, 60)
        assert z == "right_attacking"

    def test_central_defensive(self):
        z = _classify_trap_zone(10, 34)
        assert z == "central_defensive"

    def test_central_mid(self):
        z = _classify_trap_zone(50, 34)
        assert z == "central_mid"

    def test_central_attacking(self):
        z = _classify_trap_zone(90, 34)
        assert z == "central_attacking"

    # ── Edge-of-boundary cases ──

    def test_just_below_x_defensive_boundary(self):
        z = _classify_trap_zone(DEF_MAX - 0.01, 5)
        assert z == "left_defensive"

    def test_just_below_x_mid_boundary(self):
        z = _classify_trap_zone(MID_MAX - 0.01, 5)
        assert z == "left_mid"

    def test_just_inside_x_attacking(self):
        z = _classify_trap_zone(ATT_MIN + 0.01, 5)
        assert z == "left_attacking"

    def test_just_below_y_left_boundary(self):
        z = _classify_trap_zone(50, LEFT_MAX - 0.01)
        assert z == "left_mid"

    def test_just_above_y_right_boundary(self):
        z = _classify_trap_zone(50, RIGHT_MIN + 0.01)
        assert z == "right_mid"

    def test_just_above_central_lower(self):
        z = _classify_trap_zone(50, LEFT_MAX + 0.01)
        assert z == "central_mid"

    def test_just_below_central_upper(self):
        z = _classify_trap_zone(50, RIGHT_MIN - 0.01)
        assert z == "central_mid"

    def test_custom_pitch_dimensions(self):
        z = _classify_trap_zone(25, 10, pitch_length=90, pitch_width=45)
        assert z == "left_defensive"

    def test_origin_zero_zero(self):
        z = _classify_trap_zone(0, 0)
        assert z == "left_defensive"

    def test_max_corner(self):
        z = _classify_trap_zone(PL, PW)
        # y > 51 → right, x > 70.35 → attacking
        assert z == "right_attacking"


class TestComputeTrapRating:
    """_compute_trap_rating — sanity and edge values."""

    def test_zero_actions(self):
        assert _compute_trap_rating(0, 10, 5) == 0.0

    def test_all_max(self):
        r = _compute_trap_rating(20, 20, 20)
        action_score = min(1.0, 20 / 20.0)
        funnel_score = min(1.0, 20 / 20 * 2.0)
        regain_rate = 20 / 20
        regain_score = regain_rate**0.6
        expected = round(0.25 * action_score + 0.30 * funnel_score + 0.45 * regain_score, 4)
        assert r == expected

    def test_no_regains(self):
        r = _compute_trap_rating(10, 5, 0)
        assert r > 0.0
        assert r < 1.0

    def test_low_actions_many_passes(self):
        r = _compute_trap_rating(1, 10, 0)
        assert r > 0.0

    def test_partial_success(self):
        r = _compute_trap_rating(10, 5, 3)
        action_score = min(1.0, 10 / 20.0)
        funnel_score = min(1.0, 5 / 10 * 2.0)
        regain_rate = 3 / 10
        regain_score = regain_rate**0.6
        expected = round(0.25 * action_score + 0.30 * funnel_score + 0.45 * regain_score, 4)
        assert r == expected

    def test_high_regains_less_actions(self):
        r = _compute_trap_rating(5, 2, 4)
        assert r > 0.5

    def test_rounding_four_decimals(self):
        r = _compute_trap_rating(7, 3, 2)
        assert isinstance(r, float)
        # Check that it's rounded to 4 decimal places
        s = str(r)
        if "." in s:
            decimals = len(s.split(".")[1])
            assert decimals <= 4


class TestFindTriggerEvents:
    """_find_trigger_events — various preceding-event types."""

    def test_backward_pass_trigger(self):
        events = [
            _make_ev("pass", "away", 1.0, 60, 34, start_x=60, end_x=58, end_y=34),
            _make_ev("tackle", "home", 5.0, 60, 34),
        ]
        trigger, cnt = _find_trigger_events(events, "central_mid", "home")
        assert trigger == "backward_pass"
        assert cnt == 1

    def test_pass_to_wide_trigger(self):
        events = [
            _make_ev("pass", "away", 1.0, 60, 34, start_x=60, end_x=70, end_y=60),
            _make_ev("tackle", "home", 5.0, 60, 60),
        ]
        trigger, cnt = _find_trigger_events(events, "right_mid", "home")
        assert trigger == "pass_to_wide"
        assert cnt == 1

    def test_pass_forward_trigger(self):
        events = [
            _make_ev("pass", "away", 1.0, 30, 34, start_x=30, end_x=50, end_y=34),
            _make_ev("tackle", "home", 5.0, 50, 34),
        ]
        trigger, cnt = _find_trigger_events(events, "central_mid", "home")
        assert trigger == "pass_forward"
        assert cnt == 1

    def test_dribble_inside_trigger(self):
        events = [
            _make_ev("carry", "away", 1.0, 50, 34),
            _make_ev("interception", "home", 5.0, 50, 34),
        ]
        trigger, cnt = _find_trigger_events(events, "central_mid", "home")
        assert trigger == "dribble_inside"
        assert cnt == 1

    def test_shot_blocked_trigger(self):
        events = [
            _make_ev("shot", "away", 1.0, 50, 34),
            _make_ev("foul", "home", 5.0, 50, 34),
        ]
        trigger, cnt = _find_trigger_events(events, "central_mid", "home")
        assert trigger == "shot_blocked"
        assert cnt == 1

    def test_other_event_trigger(self):
        events = [
            _make_ev("clearance", "away", 1.0, 50, 34),
            _make_ev("tackle", "home", 5.0, 50, 34),
        ]
        trigger, cnt = _find_trigger_events(events, "central_mid", "home")
        assert trigger == "other_clearance"
        assert cnt == 1

    def test_no_defensive_actions(self):
        events = [_make_ev("pass", "away", 1.0, 50, 34)]
        trigger, cnt = _find_trigger_events(events, "central_mid", "home")
        assert trigger == "unknown"
        assert cnt == 0

    def test_outside_time_window(self):
        events = [
            _make_ev("pass", "away", 0.0, 50, 34, start_x=50, end_x=52, end_y=34),
            _make_ev("tackle", "home", 10.0, 50, 34),
        ]
        trigger, cnt = _find_trigger_events(events, "central_mid", "home")
        assert trigger == "unknown"
        assert cnt == 0

    def test_same_team_ignored_as_trigger(self):
        events = [
            _make_ev("pass", "home", 1.0, 50, 34, start_x=50, end_x=52, end_y=34),
            _make_ev("tackle", "home", 5.0, 50, 34),
        ]
        trigger, cnt = _find_trigger_events(events, "central_mid", "home")
        assert trigger == "unknown"
        assert cnt == 0

    def test_missing_coordinates(self):
        ev = {"type": "tackle", "team": "home", "timestamp": 5.0}
        trigger, cnt = _find_trigger_events([ev], "central_mid", "home")
        assert trigger == "unknown"
        assert cnt == 0

    def test_wrong_zone_ignored(self):
        events = [
            _make_ev("pass", "away", 1.0, 10, 5, start_x=10, end_x=12, end_y=5),
            _make_ev("tackle", "home", 5.0, 10, 5),
        ]
        trigger, cnt = _find_trigger_events(events, "central_mid", "home")
        assert trigger == "unknown"
        assert cnt == 0

    def test_multiple_triggers_picks_most_common(self):
        events = [
            _make_ev("carry", "away", 1.0, 50, 34),
            _make_ev("pass", "away", 3.5, 50, 34, start_x=50, end_x=48, end_y=34),
            _make_ev("pass", "away", 4.5, 50, 34, start_x=50, end_x=48, end_y=34),
            _make_ev("tackle", "home", 5.0, 50, 34),
            _make_ev("tackle", "home", 6.0, 50, 34),
        ]
        trigger, cnt = _find_trigger_events(events, "central_mid", "home")
        assert trigger == "backward_pass"
        assert cnt == 2


class TestDetectPressingTraps:
    """detect_pressing_traps — integration scenarios."""

    def test_empty_events(self):
        report = detect_pressing_traps([], "home")
        assert isinstance(report, PressingTrapReport)
        assert report.team == "home"
        assert report.total_traps == 0
        assert report.traps == []

    def test_no_defensive_actions(self):
        events = [
            _make_ev("pass", "home", 1.0, 50, 34),
            _make_ev("pass", "away", 2.0, 50, 34),
        ]
        report = detect_pressing_traps(events, "home")
        assert report.total_traps == 0

    def test_not_enough_actions_in_zone(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34, end_x=50, end_y=34),
            _make_ev("tackle", "home", 5.0, 50, 34),
        ]
        report = detect_pressing_traps(events, "home")
        assert report.total_traps == 0

    def test_single_zone_trap(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34, end_x=50, end_y=34),
            _make_ev("pass", "away", 2.0, 50, 34, end_x=50, end_y=34),
            _make_ev("tackle", "home", 3.0, 50, 34),
            _make_ev("tackle", "home", 4.0, 50, 34),
            _make_ev("pass", "home", 4.5, 50, 34),  # regain
        ]
        report = detect_pressing_traps(events, "home")
        assert report.total_traps >= 1
        trap = report.traps[0]
        assert trap.zone_name == "central_mid"
        assert trap.defensive_actions_in_zone == 2
        assert trap.regain_possession_count >= 1

    def test_multiple_zones(self):
        events = [
            _make_ev("pass", "away", 1.0, 10, 5, end_x=10, end_y=5),
            _make_ev("pass", "away", 2.0, 10, 5, end_x=10, end_y=5),
            _make_ev("tackle", "home", 3.0, 10, 5),
            _make_ev("tackle", "home", 4.0, 10, 5),
            _make_ev("pass", "home", 4.5, 10, 5),
            _make_ev("pass", "away", 5.0, 60, 60, end_x=60, end_y=60),
            _make_ev("pass", "away", 6.0, 60, 60, end_x=60, end_y=60),
            _make_ev("tackle", "home", 7.0, 60, 60),
            _make_ev("tackle", "home", 8.0, 60, 60),
        ]
        report = detect_pressing_traps(events, "home")
        assert report.total_traps >= 2

    def test_away_team_detection(self):
        events = [
            _make_ev("pass", "home", 1.0, 50, 34, end_x=50, end_y=34),
            _make_ev("pass", "home", 2.0, 50, 34, end_x=50, end_y=34),
            _make_ev("tackle", "away", 3.0, 50, 34),
            _make_ev("tackle", "away", 4.0, 50, 34),
            _make_ev("pass", "away", 4.5, 50, 34),
        ]
        report = detect_pressing_traps(events, "away")
        assert report.total_traps >= 1
        assert report.team == "away"

    def test_report_to_dict(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34, end_x=50, end_y=34),
            _make_ev("pass", "away", 2.0, 50, 34, end_x=50, end_y=34),
            _make_ev("tackle", "home", 3.0, 50, 34),
            _make_ev("tackle", "home", 4.0, 50, 34),
        ]
        report = detect_pressing_traps(events, "home")
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["team"] == "home"
        assert "traps" in d
        assert "total_traps" in d
        assert "overall_success_rate" in d
        assert "most_common_trigger" in d
        assert "dangerous_zones" in d
        for t in d["traps"]:
            assert "zone_name" in t
            assert "trap_rating" in t

    def test_pressing_trap_to_dict(self):
        trap = PressingTrap(
            zone_name="central_mid",
            zone_x_range=(34.65, 70.35),
            zone_y_range=(17.0, 51.0),
            trigger_event_type="backward_pass",
            defensive_actions_in_zone=5,
            opponent_passes_into_zone=3,
            regain_possession_count=2,
            success_rate=0.4,
            intensity=3.0,
            trap_rating=0.625,
        )
        d = trap.to_dict()
        assert d["zone_name"] == "central_mid"
        assert d["defensive_actions_in_zone"] == 5
        assert isinstance(d["zone_x_range"], list)
        assert d["success_rate"] == 0.4  # already rounded
        assert d["intensity"] == 3.0
        assert d["trap_rating"] == 0.625

    def test_regain_within_3_events(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34, end_x=50, end_y=34),
            _make_ev("pass", "away", 2.0, 50, 34, end_x=50, end_y=34),
            _make_ev("tackle", "home", 3.0, 50, 34),
            _make_ev("pass", "home", 3.5, 50, 34),  # regain — within ~3 events
            _make_ev("pass", "away", 4.0, 50, 34, end_x=50, end_y=34),
            _make_ev("tackle", "home", 5.0, 50, 34),
        ]
        report = detect_pressing_traps(events, "home")
        assert report.total_traps >= 1
        trap = report.traps[0]
        assert trap.regain_possession_count >= 1

    def test_all_events_same_zone(self):
        events = []
        for i in range(10):
            events.append(
                _make_ev(
                    "pass" if i % 2 == 0 else "tackle",
                    "home" if i % 2 == 0 else "home",
                    float(i),
                    50,
                    34,
                    **({"end_x": 50, "end_y": 34} if i % 2 == 0 else {}),
                )
            )
        events.append(_make_ev("pass", "home", 10.0, 50, 34))
        report = detect_pressing_traps(events, "home")
        assert report.total_traps >= 1

    def test_intensity_calculation_with_time_span(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34, end_x=50, end_y=34),
            _make_ev("pass", "away", 2.0, 50, 34, end_x=50, end_y=34),
            _make_ev("tackle", "home", 3.0, 50, 34),
            _make_ev("tackle", "home", 60.0, 50, 34),
        ]
        report = detect_pressing_traps(events, "home")
        if report.traps:
            trap = report.traps[0]
            assert trap.intensity > 0

    def test_overall_success_rate_in_report(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34, end_x=50, end_y=34),
            _make_ev("pass", "away", 2.0, 50, 34, end_x=50, end_y=34),
            _make_ev("tackle", "home", 3.0, 50, 34),
            _make_ev("tackle", "home", 4.0, 50, 34),
            _make_ev("pass", "home", 4.5, 50, 34),
            _make_ev("pass", "away", 5.0, 10, 5, end_x=10, end_y=5),
            _make_ev("pass", "away", 6.0, 10, 5, end_x=10, end_y=5),
            _make_ev("tackle", "home", 7.0, 10, 5),
            _make_ev("tackle", "home", 8.0, 10, 5),
        ]
        report = detect_pressing_traps(events, "home")
        assert 0.0 <= report.overall_success_rate <= 1.0
        assert isinstance(report.overall_success_rate, float)

    def test_dangerous_zones_low_success(self):
        events = [
            _make_ev("pass", "away", 1.0, 50, 34, end_x=50, end_y=34),
            _make_ev("pass", "away", 2.0, 50, 34, end_x=50, end_y=34),
            _make_ev("tackle", "home", 3.0, 50, 34),
            _make_ev("tackle", "home", 4.0, 50, 34),
        ]
        report = detect_pressing_traps(events, "home")
        if report.traps:
            trap = report.traps[0]
            if trap.success_rate < 0.30:
                assert trap.zone_name in report.dangerous_zones

    def test_unsorted_input_still_works(self):
        events = [
            _make_ev("tackle", "home", 4.0, 50, 34),
            _make_ev("pass", "away", 1.0, 50, 34, end_x=50, end_y=34),
            _make_ev("pass", "away", 2.0, 50, 34, end_x=50, end_y=34),
            _make_ev("tackle", "home", 3.0, 50, 34),
            _make_ev("pass", "home", 4.5, 50, 34),
        ]
        report = detect_pressing_traps(events, "home")
        assert report.total_traps >= 1

    def test_no_opponent_passes_in_zone(self):
        events = [
            # Opponent passes go to a different zone
            _make_ev("pass", "away", 1.0, 10, 5, end_x=10, end_y=5),
            _make_ev("tackle", "home", 3.0, 50, 34),
            _make_ev("tackle", "home", 4.0, 50, 34),
        ]
        report = detect_pressing_traps(events, "home")
        if report.traps:
            trap = report.traps[0]
            assert trap.opponent_passes_into_zone == 0
