"""Tests for CardDetectionService."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_svc = load_service_module("cards_test", "card_detection_service.py")
CardDetectionService = _svc.CardDetectionService
CardEvent = _svc.CardEvent
CardType = _svc.CardType
CardSource = _svc.CardSource
AudioCardSignal = _svc.AudioCardSignal

import numpy as np
import pytest


@pytest.fixture
def cards() -> CardDetectionService:
    return CardDetectionService()


class TestTacticalInference:
    def test_no_events(self, cards: CardDetectionService) -> None:
        result = cards.infer_cards_tactically([])
        assert result == []

    def test_foul_in_box_yellow(self, cards: CardDetectionService) -> None:
        # Foul in penalty area but NOT in last-man zone (need x in 88.5-100 range for away, 0-16.5 for home)
        # Home team defends away goal (x=100). Last man is x>80.
        # Foul at x=85 is last man territory, not penalty box.
        # To get penalty box (yellow), need x=90+ but that's also last-man zone for home.
        # Use team="away" attacking the home goal (x=0). Last-man zone is x<20. Penalty box is x<16.5.
        # So foul at x=10 by away, severity 0.5 — should be in penalty area but not last-man
        events = [
            {
                "type": "foul",
                "team": "away",
                "minute": 25,
                "second": 0,
                "x": 10,
                "severity": 0.5,
                "player_track_id": 5,
                "player_name": "ST",
            },
        ]
        result = cards.infer_cards_tactically(events)
        # Foul at x=10 is in home penalty area (x<16.5), so yellow card
        assert len(result) >= 1
        assert result[0].card_type == CardType.YELLOW

    def test_last_man_foul_red(self, cards: CardDetectionService) -> None:
        events = [
            {
                "type": "foul",
                "team": "home",
                "minute": 30,
                "second": 0,
                "x": 90,
                "severity": 0.85,
                "player_track_id": 4,
                "player_name": "CB",
            },
        ]
        result = cards.infer_cards_tactically(events)
        assert len(result) >= 1
        assert result[0].card_type == CardType.RED

    def test_high_severity_foul_yellow(self, cards: CardDetectionService) -> None:
        events = [
            {
                "type": "foul",
                "team": "home",
                "minute": 40,
                "second": 0,
                "x": 60,
                "severity": 0.8,
                "player_track_id": 7,
                "player_name": "CM",
            },
        ]
        result = cards.infer_cards_tactically(events)
        assert len(result) >= 1
        assert result[0].card_type == CardType.YELLOW

    def test_second_yellow(self, cards: CardDetectionService) -> None:
        events = [
            {
                "type": "foul",
                "team": "home",
                "minute": 20,
                "second": 0,
                "x": 92,
                "severity": 0.6,
                "player_track_id": 5,
                "player_name": "CB",
            },
            {
                "type": "foul",
                "team": "home",
                "minute": 60,
                "second": 0,
                "x": 92,
                "severity": 0.6,
                "player_track_id": 5,
                "player_name": "CB",
            },
        ]
        result = cards.infer_cards_tactically(events)
        assert any(c.card_type == CardType.SECOND_YELLOW for c in result)

    def test_team_tag_correct(self, cards: CardDetectionService) -> None:
        events = [
            {
                "type": "foul",
                "team": "away",
                "minute": 30,
                "second": 0,
                "x": 90,
                "severity": 0.7,
                "player_track_id": 4,
                "player_name": "CB",
            },
        ]
        result = cards.infer_cards_tactically(events)
        assert result[0].team == "away"

    def test_minute_preserved(self, cards: CardDetectionService) -> None:
        events = [
            {
                "type": "foul",
                "team": "home",
                "minute": 42,
                "second": 30,
                "x": 92,
                "severity": 0.6,
                "player_track_id": 5,
                "player_name": "CB",
            },
        ]
        result = cards.infer_cards_tactically(events)
        assert result[0].minute == 42
        assert result[0].second == 30


class TestCardFusion:
    def test_external_cards_authoritative(self, cards: CardDetectionService) -> None:
        external = [
            CardEvent(
                card_type=CardType.RED,
                minute=30,
                second=0,
                team="home",
                source=CardSource.EXTERNAL,
                confidence=0.95,
            ),
        ]
        fused = cards.fuse_cards(external, [], [], external)
        assert len(fused) == 1
        assert fused[0].card_type == CardType.RED

    def test_merge_duplicates(self, cards: CardDetectionService) -> None:
        c1 = CardEvent(
            card_type=CardType.YELLOW,
            minute=30,
            second=0,
            team="home",
            source=CardSource.TACTICAL,
            confidence=0.5,
        )
        c2 = CardEvent(
            card_type=CardType.YELLOW,
            minute=30,
            second=0,
            team="home",
            source=CardSource.VISUAL,
            confidence=0.6,
        )
        fused = cards.fuse_cards([c1], [c2], [], [])
        assert len(fused) == 1
        assert fused[0].confidence > 0.5

    def test_different_minutes_kept(self, cards: CardDetectionService) -> None:
        c1 = CardEvent(
            card_type=CardType.YELLOW, minute=20, second=0, team="home", source=CardSource.TACTICAL
        )
        c2 = CardEvent(
            card_type=CardType.YELLOW, minute=40, second=0, team="home", source=CardSource.TACTICAL
        )
        fused = cards.fuse_cards([c1], [c2], [], [])
        assert len(fused) == 2

    def test_different_teams_kept(self, cards: CardDetectionService) -> None:
        c1 = CardEvent(
            card_type=CardType.YELLOW, minute=30, second=0, team="home", source=CardSource.TACTICAL
        )
        c2 = CardEvent(
            card_type=CardType.YELLOW, minute=30, second=0, team="away", source=CardSource.TACTICAL
        )
        fused = cards.fuse_cards([c1], [c2], [], [])
        assert len(fused) == 2


class TestCardEvent:
    def test_card_event_to_dict_fields(self) -> None:
        ev = CardEvent(
            card_type=CardType.YELLOW,
            minute=30,
            second=15,
            team="home",
            player_name="Test",
            source=CardSource.TACTICAL,
            confidence=0.7,
        )
        assert ev.card_type.value == "yellow"
        assert ev.minute == 30
        assert ev.confidence == 0.7


class TestDetectCardsAudio:
    """Regression test: detect_cards_audio indexed Sxx[whistle_mask_mask]
    (a typo'd, always-undefined name) instead of Sxx[whistle_mask]. It only
    threw NameError when whistle_mask.any() was True, i.e. whenever the
    spectrogram actually has energy in the 1-3.5kHz whistle band -- which a
    silent/empty chunk (the only case previously exercised) never triggers.
    """

    def test_whistle_tone_does_not_raise(self, cards: CardDetectionService) -> None:
        sample_rate = 22050
        t = np.linspace(0, 1.0, sample_rate, endpoint=False)
        whistle = 5.0 * np.sin(2 * np.pi * 2000 * t)  # 2kHz tone, inside the whistle band
        result = cards.detect_cards_audio(whistle.astype(np.float32), sample_rate=sample_rate)
        assert isinstance(result, AudioCardSignal)
        assert result.has_whistle is True

    def test_silence_returns_no_whistle(self, cards: CardDetectionService) -> None:
        silence = np.zeros(22050, dtype=np.float32)
        result = cards.detect_cards_audio(silence, sample_rate=22050)
        assert isinstance(result, AudioCardSignal)
        assert result.has_whistle is False

    def test_empty_chunk_returns_default_signal(self, cards: CardDetectionService) -> None:
        result = cards.detect_cards_audio(np.array([]), sample_rate=22050)
        assert result.has_whistle is False
        assert result.has_crowd_reaction is False
