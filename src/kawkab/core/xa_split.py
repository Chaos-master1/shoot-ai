"""Split xA (Expected Assists) by event type.

Decomposes a player's total xA into open-play, corner, free kick,
and throw-in contributions, and compares expected vs actual assists.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from kawkab.core.game_constants import GAME

PITCH_LENGTH = GAME.PITCH_LENGTH_M
PITCH_WIDTH = GAME.PITCH_WIDTH_M


@dataclass
class XaSplit:
    open_play: float = 0.0
    corner: float = 0.0
    free_kick: float = 0.0
    throw_in: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_play": round(self.open_play, 4),
            "corner": round(self.corner, 4),
            "free_kick": round(self.free_kick, 4),
            "throw_in": round(self.throw_in, 4),
            "total": round(self.total, 4),
        }


@dataclass
class XaExpectedVsActual:
    player_id: int = 0
    xa: float = 0.0
    actual_assists: int = 0
    difference: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "xa": round(self.xa, 4),
            "actual_assists": self.actual_assists,
            "diff": round(self.difference, 4),
        }


# Mapping from set piece event types to xA split category
_SET_PIECE_MAP: dict[str, str] = {
    "corner": "corner",
    "corner_kick": "corner",
    "free_kick": "free_kick",
    "throw_in": "throw_in",
}


def compute_xa_by_type(
    events: list[dict[str, Any]],
) -> XaSplit:
    """Split xA into open-play and set-piece contributions.

    Analyzes all pass events that have an xA value, categorizing
    them by the preceding set piece type or open play.

    Args:
        events: List of event dicts with type, xA, and optional
            set_piece_type.

    Returns:
        XaSplit with xA totals per category.
    """
    split = XaSplit()

    for _i, ev in enumerate(events):
        xa_val = ev.get("xA", ev.get("xa", 0.0))
        if not isinstance(xa_val, (int, float)) or xa_val <= 0:
            continue

        etype = ev.get("type", "")
        sp_type = ev.get("set_piece_type", "")

        if etype == "pass" and sp_type:
            category = _SET_PIECE_MAP.get(sp_type, "open_play")
        elif etype == "pass":
            category = "open_play"
        elif etype in _SET_PIECE_MAP:
            category = _SET_PIECE_MAP[etype]
        else:
            category = "open_play"

        xa_val = float(xa_val)
        if category == "corner":
            split.corner += xa_val
        elif category == "free_kick":
            split.free_kick += xa_val
        elif category == "throw_in":
            split.throw_in += xa_val
        else:
            split.open_play += xa_val

    split.total = split.open_play + split.corner + split.free_kick + split.throw_in
    split.open_play = round(split.open_play, 4)
    split.corner = round(split.corner, 4)
    split.free_kick = round(split.free_kick, 4)
    split.throw_in = round(split.throw_in, 4)
    split.total = round(split.total, 4)

    return split


def compute_xa_expected_vs_actual(
    events: list[dict[str, Any]],
) -> list[XaExpectedVsActual]:
    """Compute xA vs actual assists for all players.

    Args:
        events: List of event dicts with type, from_track_id, xA,
            and assist flags.

    Returns:
        List of XaExpectedVsActual per player, sorted by difference.
    """
    player_xa: dict[int, float] = defaultdict(float)
    player_assists: dict[int, int] = defaultdict(int)

    for ev in events:
        if ev.get("type") == "pass":
            player_id = ev.get("from_track_id", 0)
            xa_val = ev.get("xA", ev.get("xa", 0.0))
            if isinstance(xa_val, (int, float)) and xa_val > 0:
                player_xa[player_id] += float(xa_val)

        if ev.get("type") == "goal":
            assist_id = ev.get("assist_track_id", ev.get("from_track_id", 0))
            if assist_id:
                player_assists[assist_id] += 1

    results: list[XaExpectedVsActual] = []
    all_ids = set(player_xa.keys()) | set(player_assists.keys())
    for pid in all_ids:
        xa = player_xa.get(pid, 0.0)
        ast = player_assists.get(pid, 0)
        results.append(
            XaExpectedVsActual(
                player_id=pid,
                xa=round(xa, 4),
                actual_assists=ast,
                difference=round(ast - xa, 4),
            )
        )

    results.sort(key=lambda r: abs(r.difference), reverse=True)
    return results
