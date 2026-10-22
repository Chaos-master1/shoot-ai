"""Tests for tactical period detection."""

from kawkab.core.tactical_periods import TacticalPeriodReport, detect_tactical_periods


class TestTacticalPeriods:
    def test_empty_data(self):
        result = detect_tactical_periods([])
        assert len(result.phases) == 0

    def test_single_phase_settled(self):
        frames = []
        for i in range(100):
            frames.append(
                {
                    "timestamp": i * 0.5,
                    "possession": True,
                    "home_positions": [(30 + i * 0.1, 34)],
                    "away_positions": [(70, 34)],
                    "ball_pos": (50, 34),
                }
            )
        result = detect_tactical_periods(frames)
        assert len(result.phases) >= 1
        assert any(p.label == "settled_possession" for p in result.phases)

    def test_high_press_detected(self):
        frames = []
        for i in range(100):
            # possession=False means home is defending, away has ball
            # High press: home (defending) pushes up > 55m
            frames.append(
                {
                    "timestamp": i * 0.5,
                    "possession": False,
                    "home_positions": [(65, 34)],
                    "away_positions": [(50, 34)],
                    "ball_pos": (55, 34),
                }
            )
        result = detect_tactical_periods(frames, pitch_length=105.0)
        assert any(p.label == "high_press" for p in result.phases)

    def test_low_block_detected(self):
        frames = []
        for i in range(100):
            # Low block: home (defending) drops deep < 30m
            frames.append(
                {
                    "timestamp": i * 0.5,
                    "possession": False,
                    "home_positions": [(20, 34)],
                    "away_positions": [(55, 34)],
                    "ball_pos": (60, 34),
                }
            )
        result = detect_tactical_periods(frames, pitch_length=105.0)
        assert any(p.label == "low_block" for p in result.phases)

    def test_report_aggregates(self):
        frames = []
        for i in range(100):
            frames.append(
                {
                    "timestamp": i * 0.5,
                    "possession": True,
                    "home_positions": [(50, 34)],
                    "away_positions": [(70, 34)],
                    "ball_pos": (55, 34),
                }
            )
        result = detect_tactical_periods(frames)
        assert isinstance(result, TacticalPeriodReport)
        assert result.settled_possession_pct >= 0.0

    def test_to_dict(self):
        frames = [
            {
                "timestamp": 0.0,
                "possession": True,
                "home_positions": [(50, 34)],
                "away_positions": [(70, 34)],
                "ball_pos": (55, 34),
            }
        ]
        frames.append(
            {
                "timestamp": 10.0,
                "possession": True,
                "home_positions": [(50, 34)],
                "away_positions": [(70, 34)],
                "ball_pos": (55, 34),
            }
        )
        result = detect_tactical_periods(frames)
        d = result.to_dict()
        assert "phases" in d
        assert "press_pct" in d
