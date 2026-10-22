"""Tests for player rating / performance index."""

from kawkab.core.player_rating import (
    PlayerPosition,
    PlayerRating,
    _infer_position_from_x,
    compute_rating,
)


class TestPositionInference:
    def test_gk_at_low_x(self):
        pos = _infer_position_from_x(2.0, 105.0)
        assert pos == PlayerPosition.GK

    def test_cb_at_low_x(self):
        pos = _infer_position_from_x(20.0, 105.0)
        assert pos == PlayerPosition.CB

    def test_cdm_at_mid_low_x(self):
        pos = _infer_position_from_x(30.0, 105.0)
        assert pos == PlayerPosition.CDM

    def test_cm_at_mid_x(self):
        pos = _infer_position_from_x(45.0, 105.0)
        assert pos == PlayerPosition.CM

    def test_cam_at_mid_high_x(self):
        pos = _infer_position_from_x(65.0, 105.0)
        assert pos == PlayerPosition.CAM

    def test_winger_at_high_x(self):
        pos = _infer_position_from_x(80.0, 105.0)
        assert pos == PlayerPosition.WING

    def test_striker_at_highest_x(self):
        pos = _infer_position_from_x(95.0, 105.0)
        assert pos == PlayerPosition.ST


class TestComputeRating:
    def test_centre_back_high_defending(self):
        r = compute_rating(
            pass_accuracy=0.80,
            passes_completed=30,
            passes_attempted=38,
            tackles=5,
            interceptions=3,
            defensive_actions=8,
            distance_covered_m=9000,
            max_speed_kmh=30,
            avg_x=20.0,
        )
        assert 0.0 <= r.overall <= 10.0
        assert r.defending > r.shooting, "CB should defend better than shoot"
        assert r.position == PlayerPosition.CB

    def test_striker_high_shooting(self):
        r = compute_rating(
            pass_accuracy=0.75,
            passes_completed=15,
            passes_attempted=20,
            shots=4,
            shots_on_target=3,
            goals=1.0,
            xg=0.8,
            tackles=0,
            interceptions=0,
            distance_covered_m=8000,
            max_speed_kmh=32,
            avg_x=95.0,
        )
        assert 0.0 <= r.overall <= 10.0
        assert r.shooting > r.defending, "ST should shoot better than defend"
        assert r.position == PlayerPosition.ST

    def test_cm_balanced_rating(self):
        r = compute_rating(
            pass_accuracy=0.85,
            passes_completed=50,
            passes_attempted=59,
            tackles=4,
            interceptions=2,
            defensive_actions=6,
            distance_covered_m=11000,
            max_speed_kmh=31,
            avg_x=50.0,
        )
        assert 0.0 <= r.overall <= 10.0
        assert r.passing > 0 and r.defending > 0

    def test_zero_minutes_returns_default(self):
        r = compute_rating(
            avg_x=50.0,
            minutes_played=0,
        )
        assert 0.0 <= r.overall <= 10.0

    def test_explicit_position_override(self):
        r = compute_rating(
            pass_accuracy=0.80,
            passes_completed=30,
            passes_attempted=38,
            tackles=2,
            interceptions=1,
            avg_x=50.0,
            position=PlayerPosition.GK,
        )
        assert r.position == PlayerPosition.GK

    def test_defaults_with_no_stats(self):
        r = compute_rating(avg_x=50.0)
        assert 0.0 <= r.overall <= 10.0
        assert all(
            0.0 <= getattr(r, attr) <= 10.0
            for attr in ["passing", "shooting", "defending", "physical", "positioning", "dribbling"]
        )

    def test_elite_stats_high_rating(self):
        r = compute_rating(
            pass_accuracy=0.95,
            passes_completed=80,
            passes_attempted=84,
            progressive_passes=12,
            key_passes=4,
            assists=1,
            shots=5,
            shots_on_target=4,
            goals=2.0,
            xg=1.5,
            tackles=6,
            interceptions=4,
            defensive_actions=10,
            carries=30,
            progressive_carries=8,
            distance_covered_m=12000,
            max_speed_kmh=34,
            sprints=20,
            possession_time_s=90,
            minutes_played=90,
            avg_x=50.0,
            position=PlayerPosition.CM,
        )
        assert r.overall >= 6.0, f"Elite CM should be high: {r.overall}"
        assert r.passing >= 6.0

    def test_poor_stats_low_rating(self):
        r = compute_rating(
            pass_accuracy=0.40,
            passes_completed=5,
            passes_attempted=12,
            shots=0,
            tackles=0,
            interceptions=0,
            distance_covered_m=3000,
            max_speed_kmh=18,
            minutes_played=90,
            avg_x=50.0,
        )
        assert r.overall <= 5.0, f"Poor stats should give low rating: {r.overall}"

    def test_to_dict(self):
        r = PlayerRating(
            overall=7.5,
            passing=8.0,
            shooting=6.5,
            defending=7.0,
            physical=7.5,
            positioning=7.0,
            dribbling=6.0,
            position=PlayerPosition.CM,
        )
        d = r.to_dict()
        assert d["overall"] == 7.5
        assert d["position"] == "Central Midfield"
