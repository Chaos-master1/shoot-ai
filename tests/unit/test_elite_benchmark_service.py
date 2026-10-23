"""Tests for the elite benchmark service (services/elite_benchmark_service.py).

The service turns a stored match into a comparison against the committed
elite cohort. The rules it must not break:

- a metric that cannot be measured for this match is reported as
  unavailable with a reason — never as zero;
- xG is only computed when EVERY shot carries the position features the
  model needs (a partial sum would understate the team and tilt its
  percentile);
- the population travels with the answer, because a percentile without
  its population is not a benchmark.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kawkab.core.metric_contract import OUTPUT_KEYS, STYLE_KEYS  # noqa: E402
from kawkab.services.elite_benchmark_service import EliteBenchmarkService  # noqa: E402


class StubStorage:
    """Minimal storage surface the service uses."""

    def __init__(self, match=None, events=None, players=None):
        self._match = match
        self._events = events or []
        self._players = players

    async def get_match(self, match_id):
        return self._match

    async def get_match_events(self, match_id, limit=200, offset=0):
        return self._events[offset : offset + limit]

    async def get_match_players(self, match_id, limit=100):
        return (self._players or [])[:limit]


def _shot(team, distance=12.0, angle=15.0, **extra):
    meta = {"distance_m": distance, "angle_to_goal_deg": angle}
    meta.update(extra)
    return {
        "match_id": 1,
        "event_type": "shot",
        "timestamp": 10.0,
        "team": team,
        "metadata": json.dumps(meta),
    }


def _pass(team, start, end, completed=True, ts=1.0, event_type="pass"):
    return {
        "match_id": 1,
        "event_type": event_type,
        "timestamp": ts,
        "team": team,
        "completed": completed,
        "metadata": json.dumps(
            {"start_x": start[0], "start_y": start[1], "end_x": end[0], "end_y": end[1]}
        ),
    }


def _action(team, event_type, x, y=34.0, won=None, ts=1.0):
    event = {
        "match_id": 1,
        "event_type": event_type,
        "timestamp": ts,
        "team": team,
        "metadata": json.dumps({"start_x": x, "start_y": y}),
    }
    if won is not None:
        event["completed"] = won
    return event


def _player_event(team, track_id, event_type, start, end=None, completed=True):
    meta = {"start_x": start[0], "start_y": start[1]}
    if end is not None:
        meta.update({"end_x": end[0], "end_y": end[1]})
    return {
        "match_id": 1,
        "event_type": event_type,
        "timestamp": 1.0,
        "team": team,
        "from_track_id": track_id,
        "completed": completed,
        "metadata": json.dumps(meta),
    }


def _player(track_id, name, position, minutes, team="home", external_id="1"):
    return {
        "track_id": track_id,
        "name": name,
        "position": position,
        "minutes_played": minutes,
        "team": team,
        "external_id": external_id,
    }


def _goal(team):
    return {
        "match_id": 1,
        "event_type": "goal",
        "timestamp": 20.0,
        "team": team,
        "metadata": json.dumps({}),
    }


def _match(**extra):
    base = {
        "id": 1,
        "name": "Test Utd vs AFC Test",
        "home_team": "Test Utd",
        "away_team": "AFC Test",
        "duration_seconds": 5400,
        "score_home": None,
        "score_away": None,
    }
    base.update(extra)
    return base


def _run(coro):
    return asyncio.run(coro)


class TestCompareMatch:
    def test_unknown_match_is_refused(self):
        service = EliteBenchmarkService(StubStorage(match=None))
        out = _run(service.compare_match(1))
        assert out["success"] is False
        assert out["data_available"] is False
        assert out["reason"]

    def test_match_without_shots_says_so(self):
        storage = StubStorage(match=_match(), events=[])
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        assert out["success"] is True
        assert out["data_available"] is False
        assert "no shots" in out["reason"]

    def test_both_teams_are_measured_against_the_population(self):
        storage = StubStorage(
            match=_match(score_home=1, score_away=0),
            events=[_shot("home"), _shot("home"), _shot("away")],
        )
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        assert out["data_available"] is True
        teams = {t["side"]: t for t in out["teams"]}
        assert set(teams) == {"home", "away"}
        assert teams["home"]["shots"] == 2
        assert teams["away"]["shots"] == 1
        for team in teams.values():
            for key, metric in team["metrics"].items():
                if key in OUTPUT_KEYS:
                    assert metric["comparable"] is True, key
                    assert 0.0 <= metric["percentile"] <= 100.0
                    # the population travels with the answer
                    assert metric["population"]["n"] == 594
                    # so does how precisely that population pins the rank down
                    assert metric["interval"] and len(metric["interval"]) == 2
                else:
                    # a shot-only feed cannot measure style: reason, not zero
                    assert key in STYLE_KEYS
                    assert metric["comparable"] is False, key
                    assert metric["reason"], key

    def test_style_metrics_compute_from_stored_events(self):
        """PPDA, tilt, progression and completion on the shared definitions.

        home: 90 completed passes from 60 m to 72 m, 6 final-third
        passes; away: 30 build-up passes plus 2 final-third passes;
        home also makes 5 interceptions in its pressing zone.
        """
        events = [_shot("home", distance=12.0, angle=15.0)]
        events += [_pass("home", (60.0, 34.0), (72.0, 34.0), ts=10.0 + i) for i in range(84)]
        events += [_pass("home", (75.0, 34.0), (78.0, 34.0), ts=200.0 + i) for i in range(16)]
        events += [_pass("away", (40.0, 34.0), (48.0, 34.0), ts=300.0 + i) for i in range(30)]
        events += [_pass("away", (75.0, 34.0), (78.0, 34.0), ts=400.0 + i) for i in range(4)]
        events += [_action("home", "interception", 50.0, ts=500.0 + i) for i in range(5)]

        storage = StubStorage(match=_match(score_home=1, score_away=0), events=events)
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        home = next(t for t in out["teams"] if t["side"] == "home")
        metrics = home["metrics"]

        # 30 opponent build-up passes / 5 home defensive actions
        assert metrics["ppda"]["value"] == pytest.approx(6.0)
        assert metrics["ppda"]["comparable"] is True
        # 16 of 20 completed final-third passes
        assert metrics["field_tilt"]["value"] == pytest.approx(0.8)
        assert metrics["pass_completion_pct"]["value"] == pytest.approx(100.0)
        assert metrics["progressive_passes_per_90"]["value"] == pytest.approx(84.0)
        # every style metric carries the cohort population
        for key in STYLE_KEYS:
            assert metrics[key]["population"]["n"] == 594, key

    def test_a_style_metric_below_its_coverage_floor_says_so(self):
        events = [_shot("home"), _action("home", "interception", 50.0)]
        storage = StubStorage(match=_match(score_home=0, score_away=0), events=events)
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        home = next(t for t in out["teams"] if t["side"] == "home")
        ppda = home["metrics"]["ppda"]
        assert ppda["comparable"] is False
        assert "needs at least" in ppda["reason"]

    def test_a_big_performance_lands_in_the_top_decile(self):
        storage = StubStorage(
            match=_match(),
            events=[_shot("home", distance=8.0, angle=5.0) for _ in range(40)],
        )
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        home = next(t for t in out["teams"] if t["side"] == "home")
        shots_metric = home["metrics"]["shots"]
        assert shots_metric["percentile"] >= 90.0
        assert shots_metric["band"] in ("top_decile", "top_quartile")

    def test_a_one_shot_match_lands_below_the_median(self):
        storage = StubStorage(match=_match(), events=[_shot("home")])
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        home = next(t for t in out["teams"] if t["side"] == "home")
        assert home["metrics"]["shots"]["percentile"] < 50.0

    def test_xg_is_unavailable_when_a_shot_lacks_position(self):
        """A partial xG sum would understate the team and tilt its percentile."""
        storage = StubStorage(
            match=_match(),
            events=[
                _shot("home"),
                {
                    "match_id": 1,
                    "event_type": "shot",
                    "timestamp": 12.0,
                    "team": "home",
                    "metadata": json.dumps({"xg": 0.2}),
                },
            ],
        )
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        home = next(t for t in out["teams"] if t["side"] == "home")
        assert home["xg"] is None
        xg_metric = home["metrics"]["xg"]
        assert xg_metric["comparable"] is False
        assert xg_metric["percentile"] is None
        assert "lack the position features" in xg_metric["reason"]
        # shots are still comparable — the shot count was measured
        assert home["metrics"]["shots"]["comparable"] is True

    def test_goals_come_from_goal_events(self):
        storage = StubStorage(
            match=_match(),
            events=[_shot("home"), _goal("home")],
        )
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        home = next(t for t in out["teams"] if t["side"] == "home")
        assert home["goals"] == 1
        assert home["metrics"]["goals"]["comparable"] is True

    def test_goals_come_from_the_shot_outcome_flag(self):
        """Vendor imports record the outcome on the shot (StatsBomb: is_goal)."""
        storage = StubStorage(
            match=_match(),
            events=[
                _shot("home", is_goal=True),
                _shot("home", is_goal=False),
                _shot("away", is_goal=False),
            ],
        )
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        home = next(t for t in out["teams"] if t["side"] == "home")
        away = next(t for t in out["teams"] if t["side"] == "away")
        assert home["goals"] == 1
        assert home["metrics"]["goals"]["comparable"] is True
        # a complete feed with outcomes knows a genuine zero
        assert away["goals"] == 0
        assert away["metrics"]["goals"]["comparable"] is True

    def test_goal_events_win_over_shot_flags(self):
        """Hybrid feeds must not double-count a goal."""
        storage = StubStorage(
            match=_match(),
            events=[_shot("home", is_goal=True), _goal("home")],
        )
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        home = next(t for t in out["teams"] if t["side"] == "home")
        assert home["goals"] == 1

    def test_goals_fall_back_to_the_stored_score(self):
        storage = StubStorage(
            match=_match(score_home=2, score_away=0),
            events=[_shot("home")],
        )
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        home = next(t for t in out["teams"] if t["side"] == "home")
        assert home["goals"] == 2

    def test_goals_are_unavailable_when_the_match_records_none(self):
        storage = StubStorage(match=_match(), events=[_shot("home")])
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        home = next(t for t in out["teams"] if t["side"] == "home")
        assert home["goals"] is None
        goals_metric = home["metrics"]["goals"]
        assert goals_metric["comparable"] is False
        assert goals_metric["reason"]

    def test_match_length_scales_the_per_90_metrics(self):
        storage = StubStorage(
            match=_match(duration_seconds=2700),
            events=[_shot("home")],
        )
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        home = next(t for t in out["teams"] if t["side"] == "home")
        assert home["minutes"] == pytest.approx(45.0)
        assert home["metrics"]["shots_per_90"]["value"] == pytest.approx(2.0)

    def test_population_and_limitations_are_always_reported(self):
        storage = StubStorage(match=_match(), events=[_shot("home")])
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        population = out["population"]
        assert population["n_team_matches"] == 594
        assert population["n_matches"] == 297
        assert population["source"]
        assert population["model"]
        assert out["limitations"]
        for limitation in out["limitations"]:
            assert limitation["key"]
            assert limitation["text"]

    def test_imported_matches_with_real_team_names_are_labelled(self):
        storage = StubStorage(
            match=_match(home_team="Real Madrid", away_team="Liverpool"),
            events=[_shot("Real Madrid"), _shot("Liverpool")],
        )
        out = _run(EliteBenchmarkService(storage).compare_match(1))
        labels = {t["team"] for t in out["teams"]}
        assert labels == {"Real Madrid", "Liverpool"}


class TestPlayerBenchmark:
    """A player is ranked against same-position peers, or refused."""

    def _service(self, players, events=None):
        return EliteBenchmarkService(
            StubStorage(
                match=_match(score_home=1, score_away=0),
                events=events or [_shot("home")],
                players=players,
            )
        )

    def test_player_identity_reaches_the_benchmark_and_is_percentiled(self):
        events = [_shot("home")] + [
            _player_event("home", 7, "pass", (30.0, 34.0), (60.0, 34.0)) for _ in range(15)
        ]
        service = self._service([_player(7, "Ada Striker", "CF", 90.0)], events=events)
        out = _run(service.compare_match(1))
        players = out["players"]
        assert players["available"] is True
        assert players["n_benchmarked"] == 1
        row = players["rows"][0]
        assert row["name"] == "Ada Striker"
        assert row["position_group"] == "FWD"
        assert row["benchmarked"] is True
        assert row["population"]["position_group"] == "FWD"
        assert row["population"]["n"] > 0
        assert "forwards" in row["population"]["label"]
        # measured metrics carry a percentile inside 0-100; refused ones a reason
        for key, metric in row["metrics"].items():
            if metric["percentile"] is not None:
                assert 0.0 <= metric["percentile"] <= 100.0, key
            else:
                assert metric["reason"], key
        assert players["population"]["positions"]["FWD"] == row["population"]["n"]
        assert players["definitions"]

    def test_the_position_group_selects_the_population(self):
        events = [_shot("home")]
        service = self._service(
            [
                _player(7, "Ada Striker", "CF", 90.0),
                _player(4, "Bea Back", "CB", 90.0),
            ],
            events=events,
        )
        players = _run(service.compare_match(1))["players"]
        by_name = {row["name"]: row for row in players["rows"]}
        assert by_name["Ada Striker"]["population"]["position_group"] == "FWD"
        assert by_name["Bea Back"]["population"]["position_group"] == "DEF"
        assert by_name["Ada Striker"]["population"]["n"] != by_name["Bea Back"]["population"]["n"]

    def test_a_short_appearance_is_refused_with_its_minutes(self):
        service = self._service([_player(7, "Ada Striker", "CF", 29.0)])
        players = _run(service.compare_match(1))["players"]
        assert players["available"] is True
        assert players["n_benchmarked"] == 0
        row = players["rows"][0]
        assert row["benchmarked"] is False
        assert row["metrics"] == {}
        assert "29" in row["reason"] and "30" in row["reason"]

    def test_an_unknown_position_is_refused_not_guessed(self):
        service = self._service([_player(7, "Ada Mystery", "XX", 90.0)])
        players = _run(service.compare_match(1))["players"]
        row = players["rows"][0]
        assert row["benchmarked"] is False
        assert "cohort position group" in row["reason"]

    def test_a_stored_row_without_identity_is_refused(self):
        service = self._service([_player(7, "", "CF", 90.0)])
        players = _run(service.compare_match(1))["players"]
        assert players["rows"][0]["benchmarked"] is False
        assert "no player identity" in players["rows"][0]["reason"]

    def test_a_match_without_players_reports_why(self):
        service = self._service([])
        players = _run(service.compare_match(1))["players"]
        assert players["available"] is False
        assert "no stored player identities" in players["reason"]

    def test_player_limitations_travel_with_the_rows(self):
        service = self._service([_player(7, "Ada Striker", "CF", 90.0)])
        players = _run(service.compare_match(1))["players"]
        assert players["limitations"]
        texts = " ".join(item["text"] for item in players["limitations"])
        assert "30 minutes" in texts and "in-sample" in texts


class SeasonStorage:
    """Stored-match surface the season block uses."""

    def __init__(self, matches, events_by_match):
        self._matches = matches
        self._events = events_by_match

    async def get_all_matches(self):
        return self._matches

    async def get_match_events(self, match_id, limit=500, offset=0):
        return self._events.get(match_id, [])[offset : offset + limit]


def _season_match(match_id, date, home="Test Utd", away="Opp FC", score=2):
    return {
        "id": match_id,
        "name": f"{home} vs {away}",
        "home_team": home,
        "away_team": away,
        "duration_seconds": 5400,
        "score_home": score,
        "score_away": 0,
        "match_date": date,
    }


def _season_fixture(shots=(10, 12, 8, 10), missing_xg=False, no_shot_match=None):
    matches, events = [], {}
    for index, count in enumerate(shots):
        match_id = index + 1
        matches.append(_season_match(match_id, f"2026-0{index + 1}-01"))
        if match_id == no_shot_match:
            events[match_id] = []
            continue
        evs = [_shot("home") for _ in range(count)]
        if missing_xg and index == 0:
            evs.append(
                {
                    "match_id": match_id,
                    "event_type": "shot",
                    "timestamp": 90.0,
                    "team": "home",
                    "metadata": json.dumps({}),
                }
            )
        events[match_id] = evs
    return matches, events


class TestCompareSeason:
    def test_a_block_needs_at_least_three_matches(self):
        matches, events = _season_fixture(shots=(10, 12))
        out = _run(EliteBenchmarkService(SeasonStorage(matches, events)).compare_season())
        assert out["data_available"] is False
        assert "at least 3" in out["reason"]

    def test_the_modal_home_team_is_the_analysed_team(self):
        matches = [
            _season_match(1, "2026-01-01", home="A"),
            _season_match(2, "2026-01-02", home="A"),
            _season_match(3, "2026-01-03", home="A"),
            _season_match(4, "2026-01-04", home="B"),
        ]
        events = {i: [_shot("home")] for i in (1, 2, 3, 4)}
        out = _run(EliteBenchmarkService(SeasonStorage(matches, events)).compare_season())
        assert out["team"] == "A"
        assert out["n_matches"] == 3
        assert out["n_matches_stored"] == 3

    def test_block_metrics_are_pooled_and_ranked_against_same_size_blocks(self):
        matches, events = _season_fixture()
        storage = SeasonStorage(matches, events)
        out = _run(EliteBenchmarkService(storage).compare_season(draws=200))
        assert out["data_available"] is True
        assert out["team"] == "Test Utd"
        assert out["n_matches"] == 4
        assert out["block"]["size"] == 4
        assert out["block"]["draws"] == 200
        assert out["block"]["method"]

        shots_metric = out["metrics"]["shots_per_90"]
        assert shots_metric["value"] == pytest.approx(10.0)
        assert 0.0 <= shots_metric["percentile"] <= 100.0
        assert shots_metric["population"]["n"] == 594
        # the comparison distribution is blocks of four, not single matches
        assert shots_metric["block_distribution"]["block_size"] == 4
        assert shots_metric["block_distribution"]["n_draws"] == 200
        assert out["metrics"]["goals_per_90"]["value"] == pytest.approx(2.0)
        assert shots_metric["interval"] and len(shots_metric["interval"]) == 2

    def test_style_metrics_are_unavailable_without_events(self):
        matches, events = _season_fixture()
        out = _run(EliteBenchmarkService(SeasonStorage(matches, events)).compare_season(draws=100))
        ppda = out["metrics"]["ppda"]
        assert ppda["comparable"] is False
        assert "not every match" in ppda["reason"]

    def test_the_trend_splits_the_block_in_half(self):
        matches, events = _season_fixture()
        out = _run(EliteBenchmarkService(SeasonStorage(matches, events)).compare_season(draws=100))
        trend = out["trend"]
        assert trend["available"] is True
        assert trend["split"] == "first 2 vs last 2 matches"
        assert trend["delta"] == pytest.approx(trend["after"] - trend["before"])
        assert "not a causal verdict" in trend["note"]

    def test_a_three_match_block_reports_no_trend(self):
        matches, events = _season_fixture(shots=(10, 12, 8))
        out = _run(EliteBenchmarkService(SeasonStorage(matches, events)).compare_season(draws=100))
        assert out["trend"]["available"] is False
        assert "at least 4" in out["trend"]["reason"]

    def test_matches_without_shot_events_are_excluded_and_counted(self):
        matches, events = _season_fixture(no_shot_match=2)
        out = _run(EliteBenchmarkService(SeasonStorage(matches, events)).compare_season(draws=100))
        assert out["n_matches"] == 3
        assert out["excluded_matches"] == [2]

    def test_xg_is_refused_when_one_match_lacks_positions(self):
        matches, events = _season_fixture(missing_xg=True)
        out = _run(EliteBenchmarkService(SeasonStorage(matches, events)).compare_season(draws=100))
        xg = out["metrics"]["xg_per_90"]
        assert xg["comparable"] is False
        assert "position" in xg["reason"]
        assert out["metrics"]["shots_per_90"]["comparable"] is True

    def test_season_limitations_replace_the_single_match_caveat(self):
        matches, events = _season_fixture()
        out = _run(EliteBenchmarkService(SeasonStorage(matches, events)).compare_season(draws=100))
        keys = [limitation["key"] for limitation in out["limitations"]]
        assert "single_match" not in keys
        assert "block_not_season" in keys
        assert "matched_block_size" in keys
        assert out["population"]["n_team_matches"] == 594


class TestPopulationSummary:
    def test_summary_describes_the_population(self):
        summary = EliteBenchmarkService(StubStorage()).population_summary()
        assert summary["n_team_matches"] == 594
        assert summary["competitions"]
        assert summary["model_features"]
        assert summary["model_excluded_features"]

    def test_summary_publishes_the_models_own_error(self):
        card = EliteBenchmarkService(StubStorage()).population_summary()["model_card"]
        assert card["available"] is True
        assert card["n_shots"] > 1000
        assert card["brier"] > 0
        assert card["calibration"]
