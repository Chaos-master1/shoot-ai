"""Youth academy service — development phases, bio-banding, minutes
management, and safeguarding-aware selection gates (Phase D).

Honesty contract:

- Development phases are derived from stored ``player_profiles``
  ``date_of_birth`` values. A profile without a DOB is ``phase:
  unknown`` — never guessed from a jersey number or a hunch.
- Bio-banded groups come from the real ``MaturationService``
  maturity-offset estimate per player. Players whose estimate is not
  computable (female coefficients unverified, implausible
  measurements) are listed in ``unbanded`` with the reason — they are
  never silently dropped into a band.
- Minutes guidance uses published professional-yield norms as
  *guidance*, not quotas; small samples are labeled
  ``small_sample`` with the count shown.
- A safeguarding hold (clearance row with ``source='safeguarding'``)
  is a hard selection block, same tier as a failed medical
  clearance. It is never advisory.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from kawkab.core.logging import get_logger
from kawkab.services.maturation_service import MaturationService

logger = get_logger(__name__)

# EPPP-mapped development phases (Elite Player Performance Plan, England).
# The strings match the minutes_log.age_phase vocabulary from migration 032.
EPPP_PHASES: dict[str, dict[str, Any]] = {
    "foundation": {
        "label": "Foundation phase",
        "age_range": (8, 11),
        "focus": "technical mastery, love of the game, multi-sport sampling",
    },
    "youth_development": {
        "label": "Youth development phase",
        "age_range": (12, 16),
        "focus": "position-specific technique, tactical concepts, growth-window care",
    },
    "professional_development": {
        "label": "Professional development phase",
        "age_range": (17, 21),
        "focus": "professional habits, senior minutes, loan/withdrawal decisions",
    },
    "senior": {
        "label": "Senior squad",
        "age_range": (22, 40),
        "focus": "performance output and availability",
    },
}

# Bio-banding bands (somatic maturity offset relative to PHV).
BIO_BANDS = ("pre_phv", "circum_phv", "post_phv")

# Development-minutes guidance (share of available minutes), by phase.
# These are published academy-practice norms used as *guidance* — the
# output text says so explicitly; they are not league rules or quotas.
MINUTES_GUIDANCE: dict[str, dict[str, Any]] = {
    "foundation": {"min_share": 0.0, "note": "game time is universal at this age"},
    "youth_development": {"min_share": 0.0, "note": "every squad player should start regularly"},
    "professional_development": {"min_share": 0.4, "note": "guide: ~40%+ of available minutes"},
    "senior": {"min_share": 0.0, "note": "selected on merit and availability"},
}

SMALL_SAMPLE_MATCHES = 5


def _age_on(dob: str, ref_date: date) -> int | None:
    """Age in whole years from an ISO date-of-birth, or None if invalid."""
    try:
        born = datetime.strptime(str(dob)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    if born > ref_date:
        return None
    return ref_date.year - born.year - ((ref_date.month, ref_date.day) < (born.month, born.day))


def _phase_for_age(age: int | None) -> str:
    if age is None:
        return "unknown"
    for phase, meta in EPPP_PHASES.items():
        lo, hi = meta["age_range"]
        if lo <= age <= hi:
            return phase
    if age < 8:
        return "foundation"
    return "senior"


class AcademyService:
    """EPPP-mapped development view over stored profiles and minutes."""

    def __init__(self, storage_service: Any, maturation: MaturationService | None = None) -> None:
        self._storage = storage_service
        self._maturation = maturation or MaturationService()

    # -- Development phases ------------------------------------------------

    async def squad_phases(self, ref_date: str | None = None) -> dict[str, Any]:
        """Classify every persistent profile into its EPPP phase."""
        ref = datetime.strptime(str(ref_date)[:10], "%Y-%m-%d").date() if ref_date else date.today()
        profiles = await self._storage.get_all_player_profiles(limit=1000)
        by_phase: dict[str, list[dict[str, Any]]] = {p: [] for p in EPPP_PHASES}
        by_phase["unknown"] = []
        unknown_dob = 0
        for prof in profiles:
            age = _age_on(prof.get("date_of_birth") or "", ref)
            phase = _phase_for_age(age)
            by_phase[phase].append(
                {
                    "player_id": prof.get("id"),
                    "name": prof.get("display_name") or prof.get("global_id"),
                    "age_years": age,
                }
            )
            if age is None:
                unknown_dob += 1
        counts = {p: len(v) for p, v in by_phase.items()}
        return {
            "phases": {
                p: {"label": m["label"], "focus": m["focus"]} for p, m in EPPP_PHASES.items()
            },
            "squad": by_phase,
            "counts": counts,
            "provenance": {
                "basis": "player_profiles.date_of_birth",
                "profiles_without_dob": unknown_dob,
                "eipp_reference": "Elite Player Performance Plan (Premier League, 2011; reviews 2019/2024)",
            },
        }

    # -- Bio-banded grouping -------------------------------------------------

    async def bio_banded_groups(
        self,
        measurements: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Group players by somatic maturation band, not age group.

        ``measurements`` entries: {player_id, name?, age_years,
        standing_height_cm, sitting_height_cm, sex}. Each is passed
        through the real Mirwald estimator; players whose estimate is
        not computable are returned in ``unbanded`` with the reason.
        """
        groups: dict[str, list[dict[str, Any]]] = {b: [] for b in BIO_BANDS}
        unbanded: list[dict[str, Any]] = []
        for m in measurements:
            est = self._maturation.estimate_offset(
                age_years=float(m.get("age_years") or 0),
                standing_height_cm=float(m.get("standing_height_cm") or 0),
                sitting_height_cm=float(m.get("sitting_height_cm") or 0),
                sex=str(m.get("sex") or "male"),
            )
            entry = {
                "player_id": m.get("player_id"),
                "name": m.get("name"),
                "maturity_offset": est.get("maturity_offset"),
                "peak_growth_window": est.get("peak_growth_window", False),
            }
            if est.get("band") in BIO_BANDS:
                groups[est["band"]].append(entry)
            else:
                entry["reason"] = est.get("provenance", {}).get("error") or est.get(
                    "classification"
                )
                unbanded.append(entry)
        return {
            "groups": groups,
            "unbanded": unbanded,
            "provenance": {
                "basis": "Mirwald et al. (2002) maturity offset via MaturationService",
                "banding_rationale": (
                    "bio-banding matches training stress to biological rather than "
                    "chronological maturity (Cumming et al., 2017)"
                ),
            },
        }

    # -- Minutes management ---------------------------------------------------

    async def minutes_management(self, player_id: int) -> dict[str, Any]:
        """Development-minutes view for one player, with honest samples."""
        history = await self._storage.get_player_minutes_history(player_id)
        profile = await self._storage.get_player_profile(player_id)
        total = sum(int(r.get("minutes_played") or 0) for r in history)
        starts = sum(1 for r in history if r.get("started"))
        by_phase: dict[str, dict[str, Any]] = {}
        for r in history:
            phase = r.get("age_phase") or "unknown"
            slot = by_phase.setdefault(phase, {"matches": 0, "minutes": 0, "starts": 0})
            slot["matches"] += 1
            slot["minutes"] += int(r.get("minutes_played") or 0)
            slot["starts"] += 1 if r.get("started") else 0
        ref_phase = _phase_for_age(
            _age_on((profile or {}).get("date_of_birth") or "", date.today())
        )
        guidance = MINUTES_GUIDANCE.get(ref_phase, MINUTES_GUIDANCE["senior"])
        small_sample = len(history) < SMALL_SAMPLE_MATCHES
        return {
            "player_id": player_id,
            "phase": ref_phase,
            "matches_logged": len(history),
            "total_minutes": total,
            "starts": starts,
            "minutes_by_phase": by_phase,
            "guidance": {
                "min_share": guidance["min_share"],
                "note": guidance["note"],
                "applies_to_phase": ref_phase,
            },
            "sample_size": "small_sample" if small_sample else "adequate",
            "provenance": {
                "basis": "minutes_log",
                "sample_matches": len(history),
                "honesty_note": (
                    "guidance, not a quota; with fewer than "
                    f"{SMALL_SAMPLE_MATCHES} logged matches the share math is "
                    "directional only"
                )
                if small_sample
                else "guidance, not a quota",
            },
        }

    # -- Safeguarding ----------------------------------------------------------

    async def safeguarding_holds(self, player_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Safeguarding holds for the given players.

        A hold is a clearance row with ``source='safeguarding'`` and
        status ``unavailable``. Selection code must treat it exactly
        like a failed medical gate: a hard block, not an advisory.
        """
        holds: dict[int, dict[str, Any]] = {}
        for pid in player_ids:
            clearance = await self._storage.get_latest_clearance(int(pid))
            if (
                clearance
                and str(clearance.get("source")) == "safeguarding"
                and str(clearance.get("status")) == "unavailable"
            ):
                holds[int(pid)] = {
                    "hold": True,
                    "reason": clearance.get("reason") or "",
                    "cleared_by": clearance.get("cleared_by") or "",
                    "tier": "hard_block",
                }
        return holds

    async def academy_selection_view(self, player_ids: list[int]) -> dict[str, Any]:
        """One view a staff member can act on: gate + phase + hold per player."""
        out: list[dict[str, Any]] = []
        holds = await self.safeguarding_holds(player_ids)
        for pid in player_ids:
            clearance = await self._storage.get_latest_clearance(int(pid))
            profile = await self._storage.get_player_profile(int(pid))
            age = _age_on((profile or {}).get("date_of_birth") or "", date.today())
            hold = holds.get(int(pid))
            out.append(
                {
                    "player_id": int(pid),
                    "name": (profile or {}).get("display_name"),
                    "phase": _phase_for_age(age),
                    "medical_status": (clearance or {}).get("status") or "no_clearance",
                    "safeguarding_hold": bool(hold),
                    "selection": "blocked" if hold else "per_clearance",
                }
            )
        return {
            "players": out,
            "provenance": {
                "basis": "medical_clearances + player_profiles",
                "hard_blocks": sum(1 for p in out if p["safeguarding_hold"]),
            },
        }
