"""Load monitoring — triangulated, honesty-first (transformation Phase C).

Answers: how big is the recent load, how does it compare with what the
player has been adapted to, and how many players are actually available?

Design decisions (documented, not hidden):

- **Flags, not predictions.** Load ratios are descriptive flags that
  start conversations. They do not predict injury (the ACWR literature
  is contested: Impellizzeri et al. argue the ratio adds little beyond
  raw load; Gabbett's work suggests danger zones). Every flag carries
  provenance so the staff can judge the evidence themselves.
- **Data-source honesty.** sRPE loads come from what players *reported*;
  GPS loads from what devices *measured*. They are computed separately
  and labeled — never silently blended into one number.
- **Triangulation.** A flag is only meaningful when corroborated by a
  second signal (self-reported wellness trend). Uncorroborated flags
  are labeled as such rather than suppressed or inflated.
- **Availability is operational truth.** The % of the squad that is
  selectable is the number a head coach actually plans around.
"""

from __future__ import annotations

from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

# Conventional ACWR bands (Gabbett). Presented as descriptive flags.
ACWR_BANDS = {
    "undertrained": (0.0, 0.8),
    "normal": (0.8, 1.3),
    "high": (1.3, 1.5),
    "very_high": (1.5, float("inf")),
}


def acwr_band(acwr: float | None) -> str:
    if acwr is None:
        return "no_data"
    for band, (lo, hi) in ACWR_BANDS.items():
        if lo <= acwr < hi:
            return band
    return "no_data"


def _r2(v: float) -> float:
    return round(v, 2)


class LoadMonitoringService:
    """sRPE + wellness triangulation and squad availability metrics."""

    def __init__(self, storage_service: Any) -> None:
        self._storage = storage_service

    # ── per player ──────────────────────────────────────────────────────

    async def player_load_state(self, player_id: int) -> dict[str, Any]:
        """Triangulated load state for one player, with provenance."""
        state: dict[str, Any] = {
            "player_id": player_id,
            "srpe": {"acwr": None, "band": "no_data", "source": "player_reported_sRPE"},
            "gps": {"acwr": None, "band": "no_data", "source": "device_measured_GPS"},
            "wellness_trend": "no_data",
            "flags": [],
            "provenance": {
                "method": "acute=7d sum, chronic=28d mean of weekly sums (rolling)",
                "triangulated": False,
            },
        }

        # 1) sRPE-based ratio (primary: the daily-capture data source)
        try:
            rpe_rows = await self._storage.get_player_rpe_history(player_id, limit=400)
        except Exception as e:
            logger.warning(f"sRPE history unavailable for player {player_id}: {e}")
            rpe_rows = []
            state["provenance"]["srpe_error"] = str(e)
        srpe_acwr = self._acwr_from_sessions(
            [(r.get("session_date"), r.get("load")) for r in rpe_rows if r.get("load") is not None]
        )
        if srpe_acwr is not None:
            state["srpe"]["acwr"] = _r2(srpe_acwr)
            state["srpe"]["band"] = acwr_band(srpe_acwr)

        # 2) GPS-based ratio (secondary: imported device sessions)
        try:
            acwr_rows = await self._storage.get_player_acwr(player_id, limit=90)
        except Exception as e:
            logger.warning(f"GPS ACWR unavailable for player {player_id}: {e}")
            acwr_rows = []
        gps_values = [r.get("acwr") for r in acwr_rows if r.get("acwr") is not None]
        if gps_values:
            state["gps"]["acwr"] = _r2(gps_values[0])
            state["gps"]["band"] = acwr_band(gps_values[0])

        # 3) Wellness trend (corroborating signal, 1-5 where 5 = best)
        try:
            wellness_rows = await self._storage.get_player_wellness(player_id, limit=14)
        except Exception as e:
            logger.warning(f"wellness trend unavailable for player {player_id}: {e}")
            wellness_rows = []
        scores = [
            r.get("wellness_score") for r in wellness_rows if r.get("wellness_score") is not None
        ]
        if len(scores) >= 3:
            recent = sum(scores[:3]) / 3.0
            baseline = sum(scores[3:]) / len(scores[3:])
            if recent < baseline - 0.5:
                state["wellness_trend"] = "declining"
            elif recent > baseline + 0.5:
                state["wellness_trend"] = "improving"
            else:
                state["wellness_trend"] = "stable"
        elif scores:
            state["wellness_trend"] = "insufficient_history"

        # 4) Triangulated flags — descriptive, corroborated where possible
        flags: list[dict[str, Any]] = []
        concerning_bands = {"high", "very_high"}
        srpe_flagged = state["srpe"]["band"] in concerning_bands
        gps_flagged = state["gps"]["band"] in concerning_bands
        if srpe_flagged or gps_flagged:
            corroborated = state["wellness_trend"] == "declining"
            flags.append(
                {
                    "type": "load_spike",
                    "band": state["srpe"]["band"] if srpe_flagged else state["gps"]["band"],
                    "source": "sRPE" if srpe_flagged else "GPS",
                    "corroborated_by_wellness": corroborated,
                    "note": (
                        "Load ratio above the conventional high band AND wellness declining"
                        if corroborated
                        else "Load ratio above the conventional high band; wellness does not "
                        "corroborate — treat as a conversation starter, not a verdict"
                    ),
                }
            )
        if srpe_acwr is not None and srpe_acwr < 0.8:
            flags.append(
                {
                    "type": "undertrained",
                    "band": "undertrained",
                    "source": "sRPE",
                    "note": "Chronic load is well above recent load — reconditioning risk "
                    "if acute load suddenly resumes at previous levels",
                }
            )
        state["flags"] = flags
        state["provenance"]["triangulated"] = srpe_flagged or gps_flagged
        return state

    @staticmethod
    def _acwr_from_sessions(session_loads: list[tuple[str | None, float]]) -> float | None:
        """ACWR from (ISO date, session load) pairs.

        Acute = sum of the last 7 days; chronic = mean of the last four
        7-day sums (rolling 28d). Needs at least 4 distinct active days
        to say anything at all.
        """
        if len(session_loads) < 4:
            return None
        try:
            from datetime import date

            dated = sorted(
                ((date.fromisoformat(str(d)), float(v)) for d, v in session_loads if d),
                key=lambda t: t[0],
            )
        except (ValueError, TypeError):
            return None
        if not dated:
            return None
        today = dated[-1][0]
        sums: list[float] = []
        for week_start_offset in range(4):
            window_end = today.toordinal() - week_start_offset * 7
            window_start = window_end - 6
            total = sum(v for d, v in dated if window_start <= d.toordinal() <= window_end)
            sums.append(total)
        acute = sums[0]
        chronic = sum(sums) / len(sums)
        if chronic <= 0:
            return None
        return acute / chronic
