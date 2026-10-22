"""Squad availability — the selection gate (transformation WS4.2).

One service answers "who can play?": medical clearances, SCAT6 concussion
RTP status, and (as advisory flags, never blocks) load-state flags.

Design decisions (from the transformation plan):
- Medical clearance is a **hard gate**: ``unavailable`` excludes a player
  from selection. ``limited`` is surfaced as restricted.
- Concussion RTP: any status other than ``full_cleared`` on an active
  assessment excludes the player (return-to-play gating per the SCAT6
  protocol), because premature return after concussion is the highest-
  severity availability failure in football.
- Wellness/load flags are **advisory**: ACWR is a conversation-starting
  flag (its predictive validity is contested in the literature), never a
  selection block. The coach sees it, the staff decides.
"""

from __future__ import annotations

import json
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


class SquadAvailabilityService:
    """Compute selection availability for a squad from medical + load data."""

    def __init__(self, storage_service: Any) -> None:
        self._storage = storage_service

    # ── single player ───────────────────────────────────────────────────

    async def player_availability(self, player_id: int) -> dict[str, Any]:
        verdict: dict[str, Any] = {
            "player_id": player_id,
            "available": True,
            "status": "fit",
            "blockers": [],
            "advisories": [],
            "provenance": {
                "clearance": "none",
                "concussion": "none",
                "wellness": "none",
            },
        }

        # 1) Medical clearance (hard gate)
        latest_clearance: dict | None = None
        try:
            latest_clearance = await self._storage.get_latest_clearance(player_id)
        except Exception as e:  # storage errors must not silently clear a player
            logger.warning(f"clearance lookup failed for player {player_id}: {e}")
            verdict["provenance"]["clearance"] = f"error: {e}"
            verdict["advisories"].append(
                "Medical-clearance lookup failed — confirm status manually before selection"
            )
        if latest_clearance:
            status = latest_clearance.get("status", "fit")
            verdict["provenance"]["clearance"] = status
            if status == "unavailable":
                verdict["available"] = False
                verdict["status"] = "unavailable"
                verdict["blockers"].append(
                    {
                        "source": "medical_clearance",
                        "reason": latest_clearance.get("reason", ""),
                        "cleared_by": latest_clearance.get("cleared_by", ""),
                    }
                )
            elif status == "limited":
                verdict["status"] = "limited"
                verdict["advisories"].append(
                    f"Medical limited status: {latest_clearance.get('reason', '')}"
                )

        # 2) Concussion RTP (hard gate unless full_cleared)
        try:
            from kawkab.services.concussion_protocol import ConcussionProtocolService

            db = getattr(self._storage, "_conn", None)
            if db is not None:
                conc = ConcussionProtocolService(db)
                status = conc.get_clearance_status(player_id)
                if status.get("has_assessment"):
                    conc_status = status.get("clearance_status", "not_cleared")
                    verdict["provenance"]["concussion"] = conc_status
                    if conc_status != "full_cleared":
                        verdict["available"] = False
                        verdict["status"] = "concussion_protocol"
                        verdict["blockers"].append(
                            {
                                "source": "concussion_protocol",
                                "reason": (
                                    "Return-to-play protocol not complete "
                                    f"({conc_status}) — SCAT6 graduated RTP"
                                ),
                                "stage": conc_status,
                            }
                        )
        except Exception as e:
            logger.warning(f"concussion lookup failed for player {player_id}: {e}")
            verdict["provenance"]["concussion"] = f"error: {e}"

        # 3) Load state (advisory only, via LoadMonitoringService) —
        # single source of load-flag logic for the whole app.
        try:
            from kawkab.services.load_monitoring_service import LoadMonitoringService

            load_state = await LoadMonitoringService(self._storage).player_load_state(player_id)
            verdict["provenance"]["load_band"] = load_state.get("srpe", {}).get("band", "no_data")
            for flag in load_state.get("flags", []):
                verdict["advisories"].append(
                    f"Load advisory ({flag.get('source', 'sRPE')}): {flag.get('note', '')}"
                )
            if load_state.get("wellness_trend") == "declining":
                verdict["advisories"].append(
                    "Wellness trend declining over the last two weeks — check with player"
                )
        except Exception as e:  # load data must never silently block or clear
            logger.warning(f"load-state lookup failed for player {player_id}: {e}")
            verdict["provenance"]["load_band"] = f"error: {e}"

        return verdict

    # ── squad view ──────────────────────────────────────────────────────

    async def squad_availability(self, player_ids: list[int]) -> dict[str, Any]:
        """Availability for a list of players, with squad-level counts."""
        entries: list[dict[str, Any]] = []
        for pid in player_ids:
            entries.append(await self.player_availability(pid))
        return {
            "players": entries,
            "available_count": sum(1 for e in entries if e["available"]),
            "unavailable_count": sum(1 for e in entries if not e["available"]),
            "limited_count": sum(1 for e in entries if e["available"] and e["status"] == "limited"),
        }

    # ── selection gate ──────────────────────────────────────────────────

    async def validate_selection(self, player_ids: list[int]) -> dict[str, Any]:
        """Hard gate for match selection: returns ok=False with blockers if
        any selected player is medically unavailable."""
        squad = await self.squad_availability(player_ids)
        blocked = [
            {"player_id": e["player_id"], "blockers": e["blockers"], "status": e["status"]}
            for e in squad["players"]
            if not e["available"]
        ]
        return {
            "ok": not blocked,
            "blocked": blocked,
            "summary": {
                "selected": len(player_ids),
                "available": squad["available_count"],
                "unavailable": squad["unavailable_count"],
            },
        }

    # ── JSON payload for bridge handlers ────────────────────────────────

    async def squad_availability_json(self, player_ids: list[int]) -> str:
        return json.dumps(await self.squad_availability(player_ids))
