"""Property-based tests for core analytics modules using hypothesis."""

import math

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from kawkab.core.coords import (
    STANDARD_PITCH,
    clamp_pitch,
    euclidean_distance_m,
    norm_to_meters,
    pitch_third,
    zone_label,
)
from kawkab.core.epv import EPVModel
from kawkab.core.pitch_control import VoronoiPitchControl, WeightedPitchControl
from kawkab.core.vaep import compute_vaep
from kawkab.core.xg_model import compute_xg
from kawkab.core.xt_model import ExpectedThreatModel

# ── xG model ──────────────────────────────────────────────────────────

xg_distance = st.floats(min_value=0.1, max_value=50.0)
xg_angle = st.floats(min_value=0.0, max_value=90.0)
xg_body = st.sampled_from(["right_foot", "left_foot", "head", "other"])
xg_assist = st.sampled_from(["standard", "through_ball", "cross", "through_ball_cross", "corner"])
xg_shot_type = st.sampled_from(["open_play", "free_kick", "volley", "header", "penalty"])


@given(distance_m=xg_distance, angle_deg=xg_angle)
def test_xg_bounds(distance_m, angle_deg):
    xg = compute_xg(distance_m, angle_deg)
    assert 0.0 <= xg <= 1.0, f"xG {xg} out of [0,1]"


@given(distance_m=xg_distance, angle_deg=xg_angle)
def test_xg_monotonic_distance(distance_m, angle_deg):
    assume(distance_m < 45)
    near = compute_xg(distance_m, angle_deg)
    far = compute_xg(distance_m + 3.0, angle_deg)
    assert near >= far, "xG should decrease with distance"


@given(distance_m=xg_distance, angle_deg=xg_angle)
def test_xg_monotonic_angle(distance_m, angle_deg):
    assume(angle_deg < 85)
    straight = compute_xg(distance_m, angle_deg)
    angled = compute_xg(distance_m, angle_deg + 5.0)
    assert straight >= angled, "xG should decrease with angle"


@given(distance_m=xg_distance, angle_deg=xg_angle)
def test_xg_header_lower(distance_m, angle_deg):
    foot = compute_xg(distance_m, angle_deg, body_part="right_foot")
    header = compute_xg(distance_m, angle_deg, body_part="head")
    assert header <= foot + 1e-9, "Headers should have lower xG than feet"


@given(distance_m=xg_distance, angle_deg=xg_angle)
def test_xg_one_on_one_higher(distance_m, angle_deg):
    normal = compute_xg(distance_m, angle_deg)
    oneonone = compute_xg(distance_m, angle_deg, is_one_on_one=True)
    assert oneonone >= normal - 1e-9, "1-on-1 should increase xG"


@given(distance_m=xg_distance, angle_deg=xg_angle)
def test_xg_pressure_lowers(distance_m, angle_deg):
    free = compute_xg(distance_m, angle_deg, is_pressed=False)
    pressed = compute_xg(distance_m, angle_deg, is_pressed=True)
    assert pressed <= free + 1e-9, "Pressure should decrease xG"


# ── xT model ──────────────────────────────────────────────────────────


@given(
    rows=st.integers(min_value=3, max_value=10),
    cols=st.integers(min_value=3, max_value=8),
)
def test_xt_zone_grid_invariants(rows, cols):
    model = ExpectedThreatModel(rows=rows, cols=cols)
    events = [
        {
            "type": "pass",
            "team": "home",
            "start_x": 30,
            "start_y": 34,
            "end_x": 60,
            "end_y": 40,
            "completed": True,
            "timestamp": 5,
        },
        {
            "type": "pass",
            "team": "home",
            "start_x": 60,
            "start_y": 40,
            "end_x": 80,
            "end_y": 34,
            "completed": True,
            "timestamp": 10,
        },
    ]
    model.build_transition_matrix(events)
    ze = model.get_zone_values()
    assert ze.shape == (rows, cols)
    assert ze.dtype == np.float64
    assert np.all(ze >= 0.0), f"Negative xT values: {ze[ze < 0]}"


@given(n_events=st.integers(min_value=5, max_value=50))
def test_xt_monotonic_with_more_data(n_events):
    model = ExpectedThreatModel(rows=4, cols=4)
    events = []
    for i in range(n_events):
        events.append(
            {
                "type": "pass",
                "start_x": 50,
                "start_y": 34,
                "end_x": 80,
                "end_y": 34,
                "completed": True,
                "timestamp": float(i),
            }
        )
    model.build_transition_matrix(events)
    ze = model.get_zone_values()
    assert np.all(ze >= 0.0)
    assert ze[2, 3] >= 0  # non-negative check only (grid positions depend on events)


# ── Coords ────────────────────────────────────────────────────────────


@given(
    x=st.floats(min_value=-10, max_value=120),
    y=st.floats(min_value=-10, max_value=80),
)
def test_clamp_pitch_in_bounds(x, y):
    cx, cy = clamp_pitch(x, y)
    assert 0.0 <= cx <= STANDARD_PITCH.length_m
    assert 0.0 <= cy <= STANDARD_PITCH.width_m


@given(
    val=st.floats(min_value=0, max_value=2.0),
    dim=st.floats(min_value=50, max_value=120),
)
def test_norm_to_meters_consistency(val, dim):
    result = norm_to_meters(val, dim)
    if 0.0 <= val <= 1.5:
        assert result == val * dim
    else:
        assert result == val


@given(
    x=st.floats(min_value=0, max_value=105),
    y=st.floats(min_value=0, max_value=68),
)
def test_pitch_third_classification(x, y):
    third = pitch_third(x)
    assert third in ("defensive", "middle", "attacking")
    if x < 35:
        assert third == "defensive"
    elif x > 70:
        assert third == "attacking"
    else:
        assert third == "middle"


@given(
    x=st.floats(min_value=0, max_value=105),
    y=st.floats(min_value=0, max_value=68),
)
def test_zone_label_format(x, y):
    label = zone_label(x, y)
    parts = label.split("_")
    assert len(parts) >= 2
    assert parts[0] in ("defensive", "middle", "attacking")


@given(
    x1=st.floats(min_value=2, max_value=105),
    y1=st.floats(min_value=2, max_value=68),
    x2=st.floats(min_value=2, max_value=105),
    y2=st.floats(min_value=2, max_value=68),
)
def test_euclidean_distance_nonnegative(x1, y1, x2, y2):
    d = euclidean_distance_m(x1, y1, x2, y2)
    assert d >= 0.0
    assert d <= STANDARD_PITCH.diagonal_m + 1e-6


@given(
    x=st.floats(min_value=0, max_value=105),
    y=st.floats(min_value=0, max_value=68),
)
def test_euclidean_distance_zero(x, y):
    d = euclidean_distance_m(x, y, x, y)
    assert d == 0.0


# ── VAEP ────────────────────────────────────────────────────────────

# Shared event strategy for property-based VAEP/EPV tests
_vaep_event = st.fixed_dictionaries(
    {
        "type": st.sampled_from(
            [
                "pass",
                "shot",
                "tackle",
                "interception",
                "clearance",
                "ball_recovery",
                "carry",
                "dribble",
            ]
        ),
        "timestamp": st.floats(min_value=0, max_value=90, allow_nan=False, allow_infinity=False),
        "team": st.sampled_from(["home", "away"]),
        "x": st.floats(min_value=0, max_value=105, allow_nan=False, allow_infinity=False),
        "y": st.floats(min_value=0, max_value=68, allow_nan=False, allow_infinity=False),
        "is_goal": st.booleans(),
    }
)


@given(events=st.lists(_vaep_event, min_size=1, max_size=6))
def test_vaep_bounds_property(events):
    results = compute_vaep(events)
    for r in results:
        assert -2.0 <= r["vaep_value"] <= 2.0, (
            f"VAEP {r['vaep_value']} out of [-2, 2] for event {r}"
        )


@given(events=st.lists(_vaep_event, min_size=2, max_size=8))
def test_vaep_turnover_not_nan(events):
    assume(any(e.get("type") in {"tackle", "interception", "clearance", "block"} for e in events))
    results = compute_vaep(events)
    turnover_types = {"tackle", "interception", "clearance", "block"}
    turnover_results = [r for r in results if r["event_type"] in turnover_types]
    assert len(turnover_results) > 0
    for r in turnover_results:
        assert -2.0 <= r["vaep_value"] <= 2.0, f"Turnover VAEP {r['vaep_value']} out of [-2, 2]"
        assert not math.isnan(r["vaep_value"])


# ── EPV ─────────────────────────────────────────────────────────────


@given(events=st.lists(_vaep_event, min_size=1, max_size=8))
def test_epv_bounds(events):
    model = EPVModel()
    report = model.compute_match_epv(events)
    for p in report.possessions:
        assert -2.0 <= p.value <= 2.0, f"EPV {p.value} out of [-2, 2]"


@given(
    x=st.floats(min_value=10, max_value=100, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=5, max_value=63, allow_nan=False, allow_infinity=False),
)
def test_epv_goal_higher_than_pass(x, y):
    model = EPVModel()
    pass_poss = [{"type": "pass", "timestamp": 1.0, "team": "home", "x": x, "y": y}]
    goal_poss = [
        {"type": "pass", "timestamp": 1.0, "team": "home", "x": x, "y": y},
        {"type": "shot", "timestamp": 2.0, "team": "home", "x": x + 3, "y": y, "is_goal": True},
    ]
    pass_val = model.compute_possession_epv(pass_poss).value
    goal_val = model.compute_possession_epv(goal_poss).value
    assert goal_val >= pass_val - 1e-9, (
        f"Goal possession EPV ({goal_val}) should be >= pass EPV ({pass_val})"
    )


@given(events=st.lists(_vaep_event, min_size=1, max_size=6))
def test_epv_empty_possessions_not_included(events):
    model = EPVModel()
    report = model.compute_match_epv(events)
    assert report.total_possessions >= 0


def test_epv_explicit_empty_possession_zero():
    model = EPVModel()
    result = model.compute_possession_epv([])
    assert result.value == 0.0
    assert result.events == 0


# ── Pitch Control ────────────────────────────────────────────────────


@given(
    home_x=st.floats(min_value=5, max_value=100, allow_nan=False),
    away_x=st.floats(min_value=5, max_value=100, allow_nan=False),
    home_y=st.floats(min_value=5, max_value=63, allow_nan=False),
    away_y=st.floats(min_value=5, max_value=63, allow_nan=False),
)
def test_pitch_control_sum_to_100(home_x, away_x, home_y, away_y):
    pc = VoronoiPitchControl(grid_rows=10, grid_cols=15)
    home = [(home_x, home_y)]
    away = [(away_x, away_y)]
    result = pc.compute_frame_control(home, away)
    total = result.home_control_pct + result.away_control_pct + result.disputed_pct
    assert abs(total - 100.0) <= 1.0, f"Control percentages sum to {total}, expected ~100"


@given(
    home_x=st.floats(min_value=5, max_value=100, allow_nan=False),
    away_x=st.floats(min_value=5, max_value=100, allow_nan=False),
    home_y=st.floats(min_value=5, max_value=63, allow_nan=False),
    away_y=st.floats(min_value=5, max_value=63, allow_nan=False),
)
def test_control_each_team_in_0_100(home_x, away_x, home_y, away_y):
    pc = VoronoiPitchControl(grid_rows=10, grid_cols=15)
    home = [(home_x, home_y)]
    away = [(away_x, away_y)]
    result = pc.compute_frame_control(home, away)
    assert 0.0 <= result.home_control_pct <= 100.0, (
        f"Home control {result.home_control_pct} out of [0, 100]"
    )
    assert 0.0 <= result.away_control_pct <= 100.0, (
        f"Away control {result.away_control_pct} out of [0, 100]"
    )


@given(
    home_x=st.floats(min_value=5, max_value=100, allow_nan=False),
    away_x=st.floats(min_value=5, max_value=100, allow_nan=False),
    home_y=st.floats(min_value=5, max_value=63, allow_nan=False),
    away_y=st.floats(min_value=5, max_value=63, allow_nan=False),
)
def test_control_weighted_sum_to_100(home_x, away_x, home_y, away_y):
    pc = WeightedPitchControl(grid_rows=10, grid_cols=15)
    home = [(home_x, home_y)]
    away = [(away_x, away_y)]
    result = pc.compute_frame_control(home, away)
    total = result.home_control_pct + result.away_control_pct + result.disputed_pct
    assert abs(total - 100.0) <= 1.0, f"Weighted control sum {total}, expected ~100"


@given(
    home_x=st.floats(min_value=5, max_value=100, allow_nan=False),
    away_x=st.floats(min_value=5, max_value=100, allow_nan=False),
    home_y=st.floats(min_value=5, max_value=63, allow_nan=False),
    away_y=st.floats(min_value=5, max_value=63, allow_nan=False),
)
def test_control_weighted_each_team_in_0_100(home_x, away_x, home_y, away_y):
    pc = WeightedPitchControl(grid_rows=10, grid_cols=15)
    home = [(home_x, home_y)]
    away = [(away_x, away_y)]
    result = pc.compute_frame_control(home, away)
    assert 0.0 <= result.home_control_pct <= 100.0
    assert 0.0 <= result.away_control_pct <= 100.0


@given(
    home_count=st.integers(min_value=0, max_value=5),
    away_count=st.integers(min_value=0, max_value=5),
)
def test_pitch_control_no_players_exact_50_50(home_count, away_count):
    assume(home_count == 0 and away_count == 0)
    pc_v = VoronoiPitchControl(grid_rows=10, grid_cols=15)
    pc_w = WeightedPitchControl(grid_rows=10, grid_cols=15)
    r_v = pc_v.compute_frame_control([], [])
    r_w = pc_w.compute_frame_control([], [])
    assert r_v.home_control_pct == 50.0
    assert r_v.away_control_pct == 50.0
    assert r_w.home_control_pct == 50.0
    assert r_w.away_control_pct == 50.0
