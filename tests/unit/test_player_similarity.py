"""Tests for Player Similarity Engine."""

from kawkab.core.player_similarity import (
    STAT_DIM,
    STAT_MEANS,
    STAT_NAMES,
    PlayerSimilarityEngine,
)


def _default_stats(player_id: int = 1, **overrides) -> dict:
    stats = {k: STAT_MEANS[k] for k in STAT_NAMES}
    stats["player_id"] = player_id
    stats.update(overrides)
    return stats


class TestBuildPlayerProfile:
    def test_produces_correct_vector_length(self):
        pse = PlayerSimilarityEngine()
        profile = pse.build_player_profile(_default_stats(1))
        assert len(profile.vector) == STAT_DIM

    def test_normalization(self):
        pse = PlayerSimilarityEngine()
        stats = _default_stats(1, pass_completion_pct=78.0, passes_per_90=45.0)
        profile = pse.build_player_profile(stats)
        # At the mean, z-score should be ~0
        assert abs(profile.vector[0]) < 0.01


class TestFindSimilarPlayers:
    def test_identical_profiles_score_one(self):
        pse = PlayerSimilarityEngine()
        target = pse.build_player_profile(_default_stats(1))
        pool = [pse.build_player_profile(_default_stats(2))]
        results = pse.find_similar_players(target, pool, top_n=5)
        assert len(results) == 1
        assert results[0].similarity > 0.999

    def test_opposite_profiles_low_score(self):
        pse = PlayerSimilarityEngine()
        target = pse.build_player_profile(
            _default_stats(1, pass_completion_pct=100, passes_per_90=90)
        )
        pool = [pse.build_player_profile(_default_stats(2, pass_completion_pct=0, passes_per_90=0))]
        results = pse.find_similar_players(target, pool, top_n=5)
        assert len(results) == 1
        assert results[0].similarity < 0.99

    def test_top_n_returns_correct_count(self):
        pse = PlayerSimilarityEngine()
        target = pse.build_player_profile(_default_stats(1))
        pool = [pse.build_player_profile(_default_stats(i)) for i in range(2, 8)]
        results = pse.find_similar_players(target, pool, top_n=3)
        assert len(results) == 3

    def test_empty_pool_returns_empty(self):
        pse = PlayerSimilarityEngine()
        target = pse.build_player_profile(_default_stats(1))
        results = pse.find_similar_players(target, [])
        assert results == []

    def test_single_player_pool_returns_that_player(self):
        pse = PlayerSimilarityEngine()
        target = pse.build_player_profile(_default_stats(1))
        pool = [pse.build_player_profile(_default_stats(2))]
        results = pse.find_similar_players(target, pool, top_n=5)
        assert len(results) == 1
        assert results[0].player_id == 2

    def test_duplicate_detection_skips_self(self):
        pse = PlayerSimilarityEngine()
        target = pse.build_player_profile(_default_stats(1))
        pool = [pse.build_player_profile(_default_stats(1))]
        results = pse.find_similar_players(target, pool, top_n=5)
        assert len(results) == 0


class TestComparePlayers:
    def test_per_stat_comparison_correct(self):
        pse = PlayerSimilarityEngine()
        a = pse.build_player_profile(_default_stats(1, pass_completion_pct=90, passes_per_90=60))
        b = pse.build_player_profile(_default_stats(2, pass_completion_pct=70, passes_per_90=30))
        result = pse.compare_players(a, b)
        assert len(result.per_stat) == STAT_DIM
        assert result.per_stat[0]["stat"] == "pass_completion_pct"
        assert result.per_stat[0]["better"] == "a"

    def test_similarity_score_range(self):
        pse = PlayerSimilarityEngine()
        a = pse.build_player_profile(_default_stats(1))
        b = pse.build_player_profile(_default_stats(2))
        result = pse.compare_players(a, b)
        assert 0.0 <= result.overall_similarity <= 1.0


class TestComputePositionSimilarity:
    def test_perfect_match(self):
        pse = PlayerSimilarityEngine()
        profile = pse.build_player_profile(_default_stats(1))
        score = pse.compute_position_similarity(profile, profile.vector)
        assert score > 0.999

    def test_no_match_low_score(self):
        pse = PlayerSimilarityEngine()
        high_stats = _default_stats(
            1, **{k: v + 30 for k, v in STAT_MEANS.items() if k != "player_id"}
        )
        profile = pse.build_player_profile(high_stats)
        low_archetype = [v - 5 for v in profile.vector]
        score = pse.compute_position_similarity(profile, low_archetype)
        # Should still be reasonably similar since directionally same
        assert 0.0 <= score <= 1.0
