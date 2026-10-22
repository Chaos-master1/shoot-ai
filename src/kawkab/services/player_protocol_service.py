"""Daily wellness / psychology / nutrition protocol aggregation (Phase C).

Rolls the raw daily check-ins into staff-facing summaries. The design
constraint is strict: this layer produces **flags and conversation
starters**, never diagnoses. Psychology especially — a low confidence
score is a signal to talk to the player, not a clinical finding.
Anything that looks more than a conversation-starter is escalated to
"refer to a qualified professional" with no further interpretation.
"""

from __future__ import annotations

from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

# Wellness score (1-5, 5 = best) below which a player is flagged for a
# staff conversation. Deliberately conservative: it starts talks.
_LOW_WELLNESS_THRESHOLD = 2.5
# Fueling/hydration (1-5) sustained below this over the window.
_LOW_FUELING_THRESHOLD = 2.5


def _r2(v: float) -> float:
    return round(v, 2)


class PlayerProtocolService:
    """Aggregates wellness, psychology, and nutrition check-ins."""

    def __init__(self, storage_service: Any) -> None:
        self._storage = storage_service

    # ── wellness ────────────────────────────────────────────────────────

    async def squad_readiness_summary(self, record_date: str) -> dict[str, Any]:
        """Morning-ritual view: squad wellness distribution + who to check on."""
        try:
            entries = await self._storage.get_squad_wellness_latest(str(record_date))
        except Exception as e:
            logger.warning(f"squad wellness lookup failed for {record_date}: {e}")
            entries = []
        scores = [
            float(e["wellness_score"]) for e in entries if e.get("wellness_score") is not None
        ]
        flagged = [
            {
                "player_id": e.get("player_id"),
                "wellness_score": e.get("wellness_score"),
                "lowest_components": self._lowest_components(e),
            }
            for e in entries
            if e.get("wellness_score") is not None
            and float(e["wellness_score"]) < _LOW_WELLNESS_THRESHOLD
        ]
        return {
            "record_date": record_date,
            "submissions": len(entries),
            "mean_wellness": _r2(sum(scores) / len(scores)) if scores else None,
            "spread": _r2(max(scores) - min(scores)) if len(scores) >= 2 else None,
            "flagged_for_conversation": flagged,
            "provenance": {
                "threshold": _LOW_WELLNESS_THRESHOLD,
                "note": "self-reported 1-5 scales (5 = best); a flag starts a "
                "conversation, it is not a diagnosis or a selection verdict",
            },
        }

    @staticmethod
    def _lowest_components(entry: dict[str, Any]) -> list[str]:
        """Which of the five Hooper components scored lowest (≤2)."""
        comps = ("sleep_quality", "fatigue", "soreness", "stress", "mood")
        lows = [c for c in comps if entry.get(c) is not None and int(entry[c]) <= 2]
        return lows

    # ── psychology ──────────────────────────────────────────────────────

    async def psych_flags(self, player_id: int, days: int = 14) -> dict[str, Any]:
        """Conversation flags from psychology check-ins (never diagnoses)."""
        flags: list[dict[str, Any]] = []
        try:
            checkins = await self._storage.get_psych_checkins(player_id, limit=max(7, days))
        except Exception as e:
            logger.warning(f"psych check-ins unavailable for player {player_id}: {e}")
            checkins = []
        explicit = [c for c in checkins if c.get("flag_for_followup")]
        if explicit:
            flags.append(
                {
                    "type": "explicit_followup_request",
                    "count": len(explicit),
                    "escalate": True,
                    "note": "player or staff explicitly requested follow-up — honor it",
                }
            )
        confidence = [float(c["confidence"]) for c in checkins if c.get("confidence") is not None]
        anxiety = [float(c["anxiety"]) for c in checkins if c.get("anxiety") is not None]
        # check-ins are returned newest-first; compare recent 3 vs older
        if len(confidence) >= 6:
            recent, older = confidence[:3], confidence[3:]
            if sum(recent) / 3 < sum(older) / len(older) - 1.0:
                flags.append(
                    {
                        "type": "confidence_declining",
                        "note": "self-reported confidence has dropped notably over the "
                        "window — a conversation is recommended",
                    }
                )
        if len(anxiety) >= 6:
            recent, older = anxiety[:3], anxiety[3:]
            if sum(recent) / 3 > sum(older) / len(older) + 1.0:
                flags.append(
                    {
                        "type": "anxiety_rising",
                        "note": "self-reported anxiety is trending up over the window — "
                        "a conversation is recommended",
                    }
                )
        # An explicit follow-up request escalates on its own: it is the
        # player asking for support, not a pattern to be corroborated.
        escalated = any(f.get("escalate") for f in flags)
        return {
            "player_id": player_id,
            "checkins_reviewed": len(checkins),
            "flags": flags,
            "escalate": (
                "Multiple converging flags — involve a qualified sports "
                "psychologist rather than continuing coach conversations alone"
                if escalated
                else ""
            ),
            "provenance": {
                "note": "these are self-report conversation starters, not clinical "
                "indicators; anything beyond a conversation belongs with a "
                "qualified professional",
            },
        }

    # ── nutrition ───────────────────────────────────────────────────────

    async def nutrition_flags(self, player_id: int, days: int = 14) -> dict[str, Any]:
        """Fueling/hydration conversation flags from nutrition logs."""
        flags: list[dict[str, Any]] = []
        try:
            logs = await self._storage.get_nutrition_logs(player_id, limit=max(7, days * 4))
        except Exception as e:
            logger.warning(f"nutrition logs unavailable for player {player_id}: {e}")
            logs = []
        hydration = [
            float(entry["hydration_score"])
            for entry in logs
            if entry.get("hydration_score") is not None
        ]
        fueling = [
            float(entry["fueling_score"])
            for entry in logs
            if entry.get("fueling_score") is not None
        ]
        if len(fueling) >= 6 and sum(fueling) / len(fueling) < _LOW_FUELING_THRESHOLD:
            flags.append(
                {
                    "type": "sustained_low_fueling",
                    "mean": _r2(sum(fueling) / len(fueling)),
                    "note": "self-reported fueling consistently below target — review "
                    "fueling strategy with the player (and a qualified sports "
                    "nutritionist where available)",
                }
            )
        if len(hydration) >= 6 and sum(hydration) / len(hydration) < _LOW_FUELING_THRESHOLD:
            flags.append(
                {
                    "type": "sustained_low_hydration",
                    "mean": _r2(sum(hydration) / len(hydration)),
                    "note": "self-reported hydration consistently below target",
                }
            )
        return {
            "player_id": player_id,
            "logs_reviewed": len(logs),
            "flags": flags,
            "provenance": {
                "note": "self-reported 1-5 scales; flags start conversations, they "
                "are not dietary prescriptions",
            },
        }
