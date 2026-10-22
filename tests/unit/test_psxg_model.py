"""Tests for PSxG model."""

from kawkab.core.psxg_model import compute_match_psxg, compute_psxg


class TestPSxG:
    def test_off_target_returns_zero(self):
        result = compute_psxg(distance_m=15, angle_deg=30, on_target=False)
        assert result.psxg == 0.0

    def test_on_target_returns_positive(self):
        result = compute_psxg(distance_m=10, angle_deg=15, on_target=True)
        assert result.psxg > 0.0

    def test_closer_higher_psxg(self):
        close = compute_psxg(5, 10, on_target=True)
        far = compute_psxg(35, 10, on_target=True)
        assert close.psxg > far.psxg

    def test_corner_placement_higher_than_center(self):
        corner = compute_psxg(12, 20, placement_x=0.95, placement_y=0.95, on_target=True)
        center = compute_psxg(12, 20, placement_x=0.5, placement_y=0.5, on_target=True)
        assert corner.psxg > center.psxg

    def test_to_dict(self):
        result = compute_psxg(10, 20, on_target=True)
        d = result.to_dict()
        assert "psxg" in d
        assert "save_probability" in d
        assert d["psxg"] + d["save_probability"] == 1.0

    def test_compute_match_psxg(self):
        events = [
            {
                "type": "shot",
                "team": "home",
                "distance_m": 10,
                "angle_deg": 20,
                "on_target": True,
                "is_goal": False,
                "timestamp": 600,
            },
            {
                "type": "shot",
                "team": "away",
                "distance_m": 5,
                "angle_deg": 10,
                "on_target": True,
                "is_goal": True,
                "timestamp": 1200,
            },
        ]
        report = compute_match_psxg(events)
        assert report.home_psxg > 0
        assert report.away_psxg > 0
        assert report.home_goals_conceded == 1

    def test_empty_events(self):
        report = compute_match_psxg([])
        assert report.home_psxg == 0
        assert report.away_psxg == 0


class TestTrainedModelBehavior:
    """Behavioral contracts ported from the superseded psxg_improved /
    psxg_model_trained heuristics (deleted 2026-09-07 — both were
    hand-tuned; the canonical psxg_model now loads fitted coefficients
    trained on 2,768 StatsBomb on-target shots)."""

    def test_header_lower_than_foot(self):
        foot = compute_psxg(
            12, 20, placement_x=0.2, placement_y=0.4, on_target=True, body_part="right_foot"
        )
        header = compute_psxg(
            12, 20, placement_x=0.2, placement_y=0.4, on_target=True, body_part="head"
        )
        assert foot.psxg > header.psxg

    def test_bounds_fuzz(self):
        import random

        for _ in range(20):
            r = compute_psxg(
                random.uniform(2, 40),
                random.uniform(2, 90),
                placement_x=random.uniform(0, 1),
                placement_y=random.uniform(0, 1),
                on_target=True,
                body_part=random.choice(["right_foot", "left_foot", "head"]),
            )
            assert 0.01 <= r.psxg <= 0.98

    def test_off_target_is_zero(self):
        r = compute_psxg(12, 20, placement_x=0.9, placement_y=0.9, on_target=False)
        assert r.psxg == 0.0
