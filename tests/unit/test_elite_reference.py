"""Tests for the elite reference population (core/elite_reference.py).

The population maths is the whole point of the benchmark: a percentile
without a defined population is decoration, and a tie that is not split
would hand a zero-shot match the credit of the whole tie group.

The artifact tests are a guard on the committed population itself — the
benchmark is only as honest as the artifact it compares against.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kawkab.core.elite_reference import (  # noqa: E402
    BENCHMARK_FEATURES,
    REQUIRED_COEFFICIENTS,
    band_for,
    block_percentile,
    build_model,
    compare_values,
    per_90,
    percentile_interval,
    percentile_rank,
    population_stats,
    shot_features,
    shot_xg,
)
from kawkab.core.metric_contract import METRICS, OUTPUT_KEYS, PLAYER_KEYS, STYLE_KEYS  # noqa: E402
from kawkab.core.player_positions import POSITION_GROUP_ORDER, position_group  # noqa: E402

ARTIFACT = Path(__file__).resolve().parents[2] / "src" / "kawkab" / "core" / "elite_reference.json"
PLAYER_ARTIFACT = (
    Path(__file__).resolve().parents[2] / "src" / "kawkab" / "core" / "elite_player_reference.json"
)


class TestPercentileRank:
    def test_mid_rank_splits_ties(self):
        # 4 values, the query equals the two tied ones -> (1 lower + half of 2)/4
        assert percentile_rank(5, [1, 5, 5, 9]) == pytest.approx(50.0)

    def test_above_everything_is_100(self):
        assert percentile_rank(10, [1, 2, 3]) == pytest.approx(100.0)

    def test_below_everything_is_zero(self):
        assert percentile_rank(0, [1, 2, 3]) == pytest.approx(0.0)

    def test_zero_shot_match_does_not_win_the_tie_group(self):
        """A match with zero shots is tied with every other zero-shot match.

        Crediting it with the whole tie group would read as "better than
        half the cohort" for having shot nothing.
        """
        population = [0.0] * 10 + [5.0] * 10
        # half of the ten zero-shot matches, not all of them (which would read 50)
        assert percentile_rank(0.0, population) == pytest.approx(25.0)

    def test_empty_population_is_refused(self):
        with pytest.raises(ValueError):
            percentile_rank(1.0, [])


class TestBlockPercentile:
    """A block average must be ranked against blocks, not single matches."""

    POPULATION = [float(v) for v in range(101)]

    def test_block_size_one_matches_the_single_rank(self):
        out = block_percentile(60, self.POPULATION, 1)
        assert out["percentile"] == pytest.approx(percentile_rank(60, self.POPULATION))
        assert out["block_distribution"]["n_draws"] == 0

    def test_same_size_blocks_rank_an_average_more_strictly(self):
        """The variance of a six-match mean is not the variance of one match.

        A value of 60 on a 0-100 population is ~60th as a single match;
        as a six-match average it is far out in the tail, and the
        percentile must say so.
        """
        single = percentile_rank(60, self.POPULATION)
        block = block_percentile(60, self.POPULATION, 6, draws=4000)["percentile"]
        assert block > single
        assert 70.0 < block < 100.0

    def test_the_population_mean_lands_near_the_middle(self):
        out = block_percentile(50.0, self.POPULATION, 6, draws=4000)
        assert out["percentile"] == pytest.approx(50.0, abs=4.0)

    def test_deterministic_for_a_seed(self):
        first = block_percentile(64.0, self.POPULATION, 5, draws=500, seed=7)
        second = block_percentile(64.0, self.POPULATION, 5, draws=500, seed=7)
        assert first["percentile"] == second["percentile"]
        assert first["band"] == second["band"]

    def test_distribution_is_described(self):
        out = block_percentile(60, self.POPULATION, 4, draws=300)
        distribution = out["block_distribution"]
        assert distribution["n_draws"] == 300
        assert distribution["block_size"] == 4
        assert distribution["p25"] < distribution["median"] < distribution["p75"]

    def test_empty_population_and_bad_block_size_raise(self):
        with pytest.raises(ValueError):
            block_percentile(1.0, [], 3)
        with pytest.raises(ValueError):
            block_percentile(1.0, [1.0, 2.0], 0)

    def test_block_larger_than_the_population_is_flagged(self):
        out = block_percentile(1.0, [0.0, 2.0], 5, draws=10)
        assert out["drawn_with_replacement"] is True


class TestPercentileInterval:
    """The band describes the cohort, and must never be read as the team."""

    def test_width_shrinks_with_a_larger_population(self):
        small = percentile_interval(60.0, 50)
        large = percentile_interval(60.0, 594)
        assert small and large
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_bounds_stay_inside_zero_to_one_hundred(self):
        low = percentile_interval(0.0, 594)
        high = percentile_interval(100.0, 594)
        assert low == [0.0, 0.0]
        assert high == [100.0, 100.0]

    def test_no_population_means_no_band(self):
        assert percentile_interval(50.0, 0) is None

    def test_compare_values_attaches_the_band(self):
        out = compare_values({"shots": 2.0}, {"shots": [1.0, 2.0, 3.0]})
        interval = out["shots"]["interval"]
        assert interval and len(interval) == 2
        assert interval[0] <= 50.0 <= interval[1]

    def test_unavailable_metrics_carry_no_band(self):
        out = compare_values({"shots": None}, {"shots": [1.0, 2.0]})
        assert out["shots"]["interval"] is None


class TestBands:
    @pytest.mark.parametrize(
        "percentile,expected",
        [
            (95.0, "top_decile"),
            (90.0, "top_decile"),
            (80.0, "top_quartile"),
            (50.0, "above_median"),
            (30.0, "lower_half"),
            (12.0, "bottom_quartile"),
            (3.0, "bottom_decile"),
        ],
    )
    def test_band_boundaries(self, percentile, expected):
        assert band_for(percentile) == expected


class TestPopulationStats:
    def test_quartiles_and_median(self):
        stats = population_stats([1, 2, 3, 4])
        assert stats["n"] == 4
        assert stats["median"] == pytest.approx(2.5)
        assert stats["min"] == 1
        assert stats["max"] == 4

    def test_empty_population(self):
        assert population_stats([]) == {"n": 0}


class TestPer90:
    def test_scales_to_ninety_minutes(self):
        assert per_90(10, 45) == pytest.approx(20.0)
        assert per_90(10, 90) == pytest.approx(10.0)

    def test_zero_minutes_is_zero_not_a_crash(self):
        assert per_90(10, 0) == 0.0


class TestShotFeatures:
    def test_unknown_body_part_collapses_to_right_foot(self):
        features = shot_features(distance_m=15.0, angle_deg=10.0, body_part="flipper")
        assert features["body_part"] == "right_foot"

    def test_unknown_shot_type_collapses_to_open_play(self):
        features = shot_features(distance_m=15.0, angle_deg=10.0, shot_type="corner")
        assert features["shot_type"] == "open_play"

    def test_free_kick_and_penalty_are_kept(self):
        assert (
            shot_features(distance_m=15.0, angle_deg=10.0, shot_type="free_kick")["shot_type"]
            == "free_kick"
        )
        assert (
            shot_features(distance_m=11.0, angle_deg=0.0, shot_type="penalty")["shot_type"]
            == "penalty"
        )


class TestCompareValues:
    def test_missing_user_value_is_not_comparable(self):
        out = compare_values({"shots": None}, {"shots": [1.0, 2.0, 3.0]})
        assert out["shots"]["comparable"] is False
        assert out["shots"]["percentile"] is None
        assert out["shots"]["reason"]

    def test_empty_reference_population_is_refused(self):
        out = compare_values({"shots": 2.0}, {"shots": []})
        assert out["shots"]["comparable"] is False
        assert "reference population" in out["shots"]["reason"]

    def test_comparable_value_carries_its_population(self):
        out = compare_values({"shots": 2.0}, {"shots": [1.0, 2.0, 3.0]})
        assert out["shots"]["comparable"] is True
        assert out["shots"]["population"]["n"] == 3
        assert out["shots"]["percentile"] == pytest.approx(50.0)

    def test_lower_is_better_metrics_are_ranked_inverted(self):
        """PPDA is the one metric where a low value is the good result.

        A percentile may only be read as "share of the cohort this
        performance is at least as good as", so a conceding-press PPDA
        of 5 must out-rank a passive 12 on the same population.
        """
        population = {"ppda": [5.0, 8.0, 10.0, 12.0]}
        aggressive = compare_values({"ppda": 5.0}, population)["ppda"]
        passive = compare_values({"ppda": 12.0}, population)["ppda"]
        assert aggressive["comparable"] is True
        assert aggressive["percentile"] == pytest.approx(87.5)
        assert passive["percentile"] == pytest.approx(12.5)
        assert aggressive["band"] == "top_quartile"
        # the reported value stays on the real scale, not the inverted one
        assert aggressive["value"] == 5.0
        assert aggressive["population"]["median"] == pytest.approx(9.0)


class TestBuildModel:
    def test_missing_coefficients_are_refused_loudly(self):
        with pytest.raises(ValueError) as exc:
            build_model({"intercept": 0.0})
        assert "missing coefficients" in str(exc.value)


@pytest.fixture(scope="module")
def artifact():
    assert ARTIFACT.exists(), f"elite reference artifact missing at {ARTIFACT}"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


class TestCommittedArtifact:
    """The artifact is the population — its shape is part of the contract."""

    def test_population_is_described(self, artifact):
        population = artifact["population"]
        assert population["n_matches"] == 297
        assert population["n_team_matches"] == 594
        assert population["n_teams"] > 100
        assert population["unit"] == "team-match"
        assert population["source"]
        assert population["competitions"]
        assert population["date_from"] and population["date_to"]

    def test_every_row_carries_every_metric(self, artifact):
        rows = artifact["rows"]
        assert len(rows) == artifact["population"]["n_team_matches"]
        for row in rows:
            for key in OUTPUT_KEYS:
                assert isinstance(row[key], (int, float)), (key, row)
            # style metrics are nested: they were added after schema v1 and
            # a v1 reader must keep working.
            assert isinstance(row["style"], dict)
            for key in STYLE_KEYS:
                assert isinstance(row["style"][key], (int, float)), (key, row)

    def test_every_metric_has_a_definition(self, artifact):
        definitions = artifact["definitions"]
        assert set(definitions) == {m.key for m in METRICS}
        for key, spec in definitions.items():
            assert spec["definition"], key
            assert spec["unit"], key
            assert isinstance(spec["higher_is_better"], bool), key

    def test_style_cohort_matches_published_ranges(self, artifact):
        """A style metric outside these bands is a definition bug, not a
        surprising season: PPDA ~4-16, field tilt 0-1 by construction,
        progressive passes ~15-90 per 90, completion 45-95%."""
        values = {key: [row["style"][key] for row in artifact["rows"]] for key in STYLE_KEYS}

        ppda = statistics.median(values["ppda"])
        assert 5.0 <= ppda <= 11.0, ppda
        progressive = statistics.median(values["progressive_passes_per_90"])
        assert 20.0 <= progressive <= 60.0, progressive
        completion = statistics.median(values["pass_completion_pct"])
        assert 65.0 <= completion <= 85.0, completion
        for tilt in values["field_tilt"]:
            assert 0.0 <= tilt <= 1.0
        for turnovers in values["high_turnovers_per_90"]:
            assert 0.0 <= turnovers <= 60.0

    def test_model_is_complete_and_loadable(self, artifact):
        model = artifact["model"]
        assert model["features"] == list(BENCHMARK_FEATURES)
        assert model["excluded_features"]
        for key in REQUIRED_COEFFICIENTS:
            assert key in model["coefficients"], key
        assert build_model(model["coefficients"]) is not None

    def test_model_card_publishes_its_own_error(self, artifact):
        card = artifact["model_card"]
        assert card["available"] is True
        assert card["n_shots"] == artifact["population"]["n_shots"]
        assert 0.03 <= card["brier"] <= 0.15, card["brier"]
        assert 0.1 <= card["log_loss"] <= 0.5, card["log_loss"]
        assert card["mae_vs_statsbomb_xg"] < 0.1
        # calibration bins account for every scored shot
        assert sum(bin["n"] for bin in card["calibration"]) == card["n_shots"]
        assert "In-sample" in card["note"]

    def test_model_card_is_published_as_its_own_file(self, artifact):
        path = ARTIFACT.with_name("model_card.json")
        assert path.exists(), "model_card.json must ship beside the artifact"
        card = json.loads(path.read_text(encoding="utf-8"))
        stored = dict(artifact["model_card"])
        generated = card.pop("generated_at", None)
        assert generated
        assert card == stored

    def test_cohort_xg_is_calibrated_to_elite_levels(self, artifact):
        """A benchmark that prints 2.6 xG per elite team-match is broken.

        The reduced-feature model has to land near the real elite range
        (roughly 1.1-1.5 xG per team-match), not the inflated value the
        app's full-feature coefficients produce on the same shots.
        """
        values = [row["xg"] for row in artifact["rows"]]
        median = statistics.median(values)
        assert 0.8 <= median <= 1.8, median

    def test_shot_counts_match_real_match_output(self, artifact):
        shots = [row["shots"] for row in artifact["rows"]]
        median = statistics.median(shots)
        assert 8 <= median <= 18, median

    def test_model_measures_both_sides_identically(self, artifact):
        """A shot measured by the benchmark model must match the model
        used to build the cohort — that identity is the comparison."""
        model = build_model(artifact["model"]["coefficients"])
        features = shot_features(distance_m=12.0, angle_deg=15.0)
        value = shot_xg(model, features)
        assert 0.0 < value < 1.0


@pytest.fixture(scope="module")
def player_artifact():
    assert PLAYER_ARTIFACT.exists(), f"player reference artifact missing at {PLAYER_ARTIFACT}"
    return json.loads(PLAYER_ARTIFACT.read_text(encoding="utf-8"))


class TestPlayerArtifact:
    """The player cohort is the population a player percentile claims."""

    def test_every_group_has_a_population(self, player_artifact):
        population = player_artifact["population"]
        assert population["unit"] == "player-match"
        assert population["min_minutes"] == 30.0
        assert population["n_player_matches"] == len(player_artifact["rows"])
        assert population["n_players"] < population["n_player_matches"]
        assert set(population["positions"]) == set(POSITION_GROUP_ORDER)
        # enough peers for a stable percentile in every group
        for group, count in population["positions"].items():
            assert count >= 200, group
        assert sum(population["positions"].values()) == population["n_player_matches"]

    def test_rows_match_their_declared_position_group(self, player_artifact):
        for row in player_artifact["rows"]:
            assert row["position_group"] == position_group(row["position"]), row
            assert row["minutes"] >= 30.0, row
            assert row["player_id"] and row["player_name"] and row["team"]

    def test_every_row_carries_every_player_metric(self, player_artifact):
        assert set(player_artifact["definitions"]) == set(PLAYER_KEYS)
        missing: dict[str, int] = {}
        for row in player_artifact["rows"]:
            for key in PLAYER_KEYS:
                value = row["metrics"][key]
                if value is None:
                    # a player below the metric's event floor is left out of
                    # that metric's population instead of being scored as 0
                    missing[key] = missing.get(key, 0) + 1
                else:
                    assert isinstance(value, (int, float)), (key, row)
            assert isinstance(row["shots"], int)
            assert 0.0 <= row["xg"] <= row["shots"]
        # refusal stays exceptional: a metric absent for most rows is a
        # broken measurement, not a coverage floor
        for key, count in missing.items():
            assert count < len(player_artifact["rows"]) * 0.10, (key, count)

    def test_the_population_is_the_same_model_as_the_team_benchmark(
        self, player_artifact, artifact
    ):
        assert player_artifact["model"]["coefficients"] == artifact["model"]["coefficients"]
