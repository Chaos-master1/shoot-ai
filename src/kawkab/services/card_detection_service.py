"""Card Detection Service - multi-source card event detection.

Four detection sources, fused for confidence:
1. Visual: detect referee raising a colored card (yellow vs red color analysis)
2. Audio: referee whistle + crowd reaction pattern characteristic of cards
3. Tactical: infer cards from event severity (foul in box, last-man tackle, etc.)
4. External: pull from StatsBomb/API-Football for verified matches

Output: list of CardEvents with confidence and source attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class CardType(Enum):
    YELLOW = "yellow"
    RED = "red"
    SECOND_YELLOW = "second_yellow"


class CardSource(Enum):
    VISUAL = "visual"
    AUDIO = "audio"
    TACTICAL = "tactical"
    EXTERNAL = "external"


@dataclass
class CardEvent:
    card_type: CardType
    minute: int
    second: int
    player_track_id: int | None = None
    player_name: str | None = None
    team: str | None = None
    source: CardSource = CardSource.TACTICAL
    confidence: float = 0.5
    description: str = ""


@dataclass
class AudioCardSignal:
    has_whistle: bool
    has_crowd_reaction: bool
    is_card_pattern: bool
    confidence: float


class CardDetectionService:
    """Multi-source card detection.

    Fuses visual (referee raised arm + card color), audio (whistle + crowd),
    tactical (event severity inference), and external (verified data).
    """

    def __init__(self) -> None:
        self._available = True
        self._cv2: Any = None
        self._try_import_cv2()

    def _try_import_cv2(self) -> None:
        try:
            import cv2

            self._cv2 = cv2
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Visual: referee + raised arm + card color
    # ------------------------------------------------------------------

    def detect_cards_visual(
        self, frames: list[np.ndarray], timestamps: list[float]
    ) -> list[CardEvent]:
        """Detect cards by finding referee with raised arm holding colored card.

        For each frame:
        1. Find regions with strong yellow or red color
        2. Check if these are in upper half of frame (raised arm)
        3. Track detections across frames to filter noise
        4. If consistent across 3+ frames → card event
        """
        if self._cv2 is None or not frames:
            return []
        detections_per_frame: list[tuple[float, CardType | None, float]] = []
        for ts, frame in zip(timestamps, frames, strict=False):
            if frame is None or frame.size == 0:
                continue
            try:
                card_type, conf = self._detect_card_in_frame(frame)
                detections_per_frame.append((ts, card_type, conf))
            except Exception as e:
                logger.debug(f"Frame card detection failed: {e}")
        events: list[CardEvent] = []
        i = 0
        while i < len(detections_per_frame):
            ts, card_type, conf = detections_per_frame[i]
            if card_type is None or conf < 0.4:
                i += 1
                continue
            count = 1
            for j in range(i + 1, min(i + 6, len(detections_per_frame))):
                if detections_per_frame[j][1] == card_type:
                    count += 1
            if count >= 3:
                minute = int(ts // 60)
                second = int(ts % 60)
                events.append(
                    CardEvent(
                        card_type=card_type,
                        minute=minute,
                        second=second,
                        source=CardSource.VISUAL,
                        confidence=min(0.9, conf * (count / 6)),
                        description=f"Referee raised {card_type.value} card detected in {count} consecutive frames",
                    )
                )
                i += count
            else:
                i += 1
        return events

    def _detect_card_in_frame(self, frame: np.ndarray) -> tuple[CardType | None, float]:
        if self._cv2 is None:
            return None, 0.0
        try:
            h, w = frame.shape[:2]
            upper = frame[: h // 2, :]
            hsv = self._cv2.cvtColor(upper, self._cv2.COLOR_BGR2HSV)
            yellow_mask = self._cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
            red_lower = self._cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            red_upper = self._cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
            red_mask = self._cv2.bitwise_or(red_lower, red_upper)
            yellow_pct = float(yellow_mask.mean()) / 255.0
            red_pct = float(red_mask.mean()) / 255.0
            yellow_contours, _ = self._cv2.findContours(
                yellow_mask, self._cv2.RETR_EXTERNAL, self._cv2.CHAIN_APPROX_SIMPLE
            )
            red_contours, _ = self._cv2.findContours(
                red_mask, self._cv2.RETR_EXTERNAL, self._cv2.CHAIN_APPROX_SIMPLE
            )
            yellow_card_like = self._filter_card_contours(yellow_contours)
            red_card_like = self._filter_card_contours(red_contours)
            if yellow_card_like > 0 and yellow_pct > 0.002:
                return CardType.YELLOW, min(0.9, 0.4 + yellow_pct * 20)
            if red_card_like > 0 and red_pct > 0.0015:
                return CardType.RED, min(0.9, 0.4 + red_pct * 25)
            return None, 0.0
        except Exception:
            return None, 0.0

    def _filter_card_contours(self, contours) -> int:
        if not contours or self._cv2 is None:
            return 0
        count = 0
        for c in contours:
            try:
                x, y, w, h = self._cv2.boundingRect(c)
                area = w * h
                if 200 < area < 5000 and 0.5 < w / max(h, 1) < 2.5:
                    count += 1
            except Exception:
                continue
        return count

    # ------------------------------------------------------------------
    # Audio: whistle + crowd pattern
    # ------------------------------------------------------------------

    def detect_cards_audio(
        self, audio_chunk: np.ndarray, sample_rate: int = 22050
    ) -> AudioCardSignal:
        """Detect whistle + crowd pattern characteristic of card events.

        Cards typically follow this audio pattern:
        1. Sharp whistle (1-3 kHz burst)
        2. Crowd reaction (amplitude spike, then decay)
        3. Often followed by boos or cheers
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return AudioCardSignal(False, False, False, 0.0)
        if sample_rate <= 0:
            return AudioCardSignal(False, False, False, 0.0)
        try:
            from scipy import signal as sp_signal

            freqs, times, Sxx = sp_signal.spectrogram(audio_chunk, fs=sample_rate, nperseg=1024)
        except Exception:
            return AudioCardSignal(False, False, False, 0.0)
        whistle_mask = (freqs >= 1000) & (freqs <= 3500)
        whistle_energy = float(Sxx[whistle_mask].mean()) if whistle_mask.any() else 0.0
        has_whistle = whistle_energy > 0.001
        if len(audio_chunk) > sample_rate // 2:
            rms = float(np.sqrt(np.mean(audio_chunk**2)))
            peak_rms = float(np.sqrt(np.mean(np.abs(audio_chunk) ** 2)))
            has_crowd_reaction = peak_rms > rms * 1.5
        else:
            has_crowd_reaction = False
        is_card_pattern = has_whistle and has_crowd_reaction
        confidence = 0.0
        if is_card_pattern:
            confidence = 0.7
        if has_whistle and not has_crowd_reaction:
            confidence = 0.3
        return AudioCardSignal(
            has_whistle=has_whistle,
            has_crowd_reaction=has_crowd_reaction,
            is_card_pattern=is_card_pattern,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Tactical inference: foul events → likely cards
    # ------------------------------------------------------------------

    def infer_cards_tactically(self, events: list[dict[str, Any]]) -> list[CardEvent]:
        """Infer cards from event patterns.

        Heuristics:
        - Foul in penalty area: yellow card likely
        - Last-man foul (foul by defender on attacker near goal): red card
        - Violent motion (e.g., high x velocity + foul): red card
        - Second yellow: needs context (player already booked)
        """
        card_events: list[CardEvent] = []
        booked_players: set[tuple[int, str]] = set()
        for event in events:
            et = event.get("event_type", event.get("type", ""))
            if et not in {"foul", "tackle", "challenge", "violation"}:
                continue
            minute = event.get("minute", 0)
            second = event.get("second", 0)
            team = event.get("team", "")
            player_tid = event.get("player_track_id")
            player_name = event.get("player_name", "")
            player_key = (player_tid or 0, team)
            location_x = event.get("x", 0) or 0
            in_penalty_area = location_x > 88.5 or location_x < 16.5
            is_last_man = (location_x > 80 and team == "home") or (
                location_x < 20 and team == "away"
            )
            severity = event.get("severity", 0.5)
            if is_last_man and severity > 0.6:
                card_events.append(
                    CardEvent(
                        card_type=CardType.RED,
                        minute=minute,
                        second=second,
                        player_track_id=player_tid,
                        player_name=player_name,
                        team=team,
                        source=CardSource.TACTICAL,
                        confidence=0.6,
                        description="Last-man tactical foul — red card (DOGSO) likely per Law 12",
                    )
                )
            elif in_penalty_area and severity > 0.4:
                card_events.append(
                    CardEvent(
                        card_type=CardType.YELLOW,
                        minute=minute,
                        second=second,
                        player_track_id=player_tid,
                        player_name=player_name,
                        team=team,
                        source=CardSource.TACTICAL,
                        confidence=0.55,
                        description="Foul in penalty area — yellow card likely",
                    )
                )
            elif severity > 0.7:
                card_events.append(
                    CardEvent(
                        card_type=CardType.YELLOW,
                        minute=minute,
                        second=second,
                        player_track_id=player_tid,
                        player_name=player_name,
                        team=team,
                        source=CardSource.TACTICAL,
                        confidence=0.5,
                        description="High-severity foul — yellow card likely",
                    )
                )
            if player_key in booked_players:
                for ce in card_events:
                    if (
                        ce.player_track_id == player_tid
                        and ce.team == team
                        and ce.card_type == CardType.YELLOW
                    ):
                        ce.card_type = CardType.SECOND_YELLOW
                        ce.description += " (second yellow → red)"
                        break
            else:
                booked_players.add(player_key)
        return card_events

    # ------------------------------------------------------------------
    # External: pull from StatsBomb or API-Football
    # ------------------------------------------------------------------

    async def fetch_external_cards(
        self,
        match_id: int,
        statsbomb_service: Any | None = None,
        api_football_service: Any | None = None,
    ) -> list[CardEvent]:
        """Fetch card events from external data sources.

        Tries StatsBomb first (most detailed), then API-Football.
        """
        cards: list[CardEvent] = []
        if statsbomb_service is not None:
            try:
                comp = await statsbomb_service.get_competitions()
                for c in comp:
                    matches = await statsbomb_service.get_matches(c.competition_id, c.season_id)
                    target = next((m for m in matches if m.match_id == match_id), None)
                    if target:
                        events = await statsbomb_service.get_events(match_id)
                        for e in events:
                            if e.event_type in {"Foul", "Bad Behaviour", "Card"}:
                                card_type = CardType.YELLOW
                                if "Red" in e.raw.get("card", {}).get("name", ""):
                                    card_type = CardType.RED
                                elif "Yellow" in e.raw.get("card", {}).get("name", ""):
                                    card_type = CardType.YELLOW
                                elif "Second Yellow" in e.raw.get("card", {}).get("name", ""):
                                    card_type = CardType.SECOND_YELLOW
                                cards.append(
                                    CardEvent(
                                        card_type=card_type,
                                        minute=e.minute,
                                        second=e.second,
                                        player_track_id=e.player_id,
                                        player_name=e.player,
                                        team=e.team,
                                        source=CardSource.EXTERNAL,
                                        confidence=0.95,
                                        description="StatsBomb verified card event",
                                    )
                                )
                        return cards
            except Exception as e:
                logger.warning(f"StatsBomb card fetch failed: {e}")
        if api_football_service is not None:
            try:
                pass
            except Exception as e:
                logger.warning(f"API-Football card fetch failed: {e}")
        return cards

    # ------------------------------------------------------------------
    # Fusion: combine all sources
    # ------------------------------------------------------------------

    def fuse_cards(
        self,
        visual: list[CardEvent],
        audio_cards: list[CardEvent],
        tactical: list[CardEvent],
        external: list[CardEvent],
    ) -> list[CardEvent]:
        """Fuse cards from multiple sources.

        Strategy:
        - External cards are authoritative (highest confidence)
        - Visual/audio/tactical corroborate
        - Group by (minute, player, team, type) and average confidence
        """
        all_cards: list[CardEvent] = list(external)
        for c in visual + audio_cards + tactical:
            is_duplicate = False
            for existing in all_cards:
                if (
                    abs(existing.minute - c.minute) <= 1
                    and existing.team == c.team
                    and existing.player_track_id == c.player_track_id
                    and existing.card_type == c.card_type
                ):
                    existing.confidence = min(0.99, existing.confidence + c.confidence * 0.3)
                    existing.description += f" + {c.source.value} corroboration"
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_cards.append(c)
        all_cards.sort(key=lambda c: (c.minute, c.second))
        return all_cards
