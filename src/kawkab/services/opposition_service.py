"""Opposition intelligence service (Phase D).

Wires the previously dead ``OpponentDossier`` analysis module through
real stored data, adds the set-play library, and provides vendor-data
import probing with an explicit honest no-provider state.

Honesty contract:

- The dossier is computed over the club's own stored opponent matches
  and events. Fewer than the dossier's meaningful sample is labeled
  ``small_sample`` with the count; an opponent with no stored matches
  returns ``no_data``, never a template report.
- What the raw event store cannot support is reported as unknown, not
  approximated: pressing intensity (PPDA) and build-up style need
  tracking/breakpoint data this repo does not produce, so the dossier
  shows the engine's ``unknown``; the predicted lineup needs linked
  player minutes, so it stays at the engine's fallback.
- Vendor imports (kloppy-supported providers: StatsBomb, Opta,
  SkillCorner, Second Spectrum...) are probed at runtime. kloppy is
  declared in pyproject but optional: if the import fails, the state is
  ``provider_available: false`` with a reason — same honesty as the
  Transfermarkt precedent. No synthetic vendor data is ever produced.
- The scoreline ``prediction`` inside the dossier is a naive historical
  average, surfaced with a provenance note — it is not a model forecast.
"""

from __future__ import annotations

from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

SMALL_SAMPLE_MATCHES = 3

# Known kloppy-supported vendor formats (kloppy >= 3.12). Used only for
# labeling the probe result; the availability check is the real import.
KLOPPY_FORMATS = (
    "statsbomb",
    "opta",
    "skillcorner",
    "second_spectrum",
    "tracab",
    "metrica",
    "sportec",
)


def _probe_kloppy() -> dict[str, Any]:
    """Return the honest provider state for kloppy-backed imports.

    When kloppy is importable the probe goes beyond availability: it
    reports the per-provider capability report from the import service
    (which providers this installation can actually convert), so the UI
    shows "available with these loaders", not just "importable".
    """
    try:
        import kloppy  # noqa: F401
    except ImportError:
        return {
            "provider_available": False,
            "provider": "kloppy",
            "reason": (
                "kloppy is not installed in this environment; install it "
                "(pip install kloppy) to enable vendor data imports"
            ),
            "providers": [
                {
                    "provider": p,
                    "kind": kind,
                    "available": False,
                    "status": "provider_unavailable",
                    "reason": "kloppy is not installed",
                }
                for p, kind in (("statsbomb", "events"), ("skillcorner", "tracking"))
            ],
            "supported_formats": list(KLOPPY_FORMATS),
        }
    except Exception as e:  # pragma: no cover - defensive
        return {
            "provider_available": False,
            "provider": "kloppy",
            "reason": f"kloppy import failed: {e}",
            "providers": [],
            "supported_formats": list(KLOPPY_FORMATS),
        }
    import importlib.metadata as _md

    try:
        version = _md.version("kloppy")
    except Exception:  # pragma: no cover - defensive
        version = "unknown"

    from kawkab.services.kloppy_import_service import KloppyImportService

    return {
        "provider_available": True,
        "provider": "kloppy",
        "version": version,
        "providers": KloppyImportService.provider_capabilities(),
        "supported_formats": list(KLOPPY_FORMATS),
        "honesty_note": (
            "availability probe only; no vendor account, credentials, or "
            "data feed is bundled with this application"
        ),
    }


class OppositionService:
    """Pre-match scouting over stored matches/events plus set plays."""

    def __init__(self, storage_service: Any, knowledge_service: Any = None) -> None:
        self._storage = storage_service
        self._kb = knowledge_service

    # -- Dossier -------------------------------------------------------------

    async def build_dossier(self, opponent_team: str) -> dict[str, Any]:
        """Real dossier for an opponent from this club's stored matches.

        Uses the existing ``analysis.opponent_dossier`` engine over
        matches where the opponent appears as home or away team.
        Goals are attributed to named players when the scorer's track
        id is resolvable in that match's player rows.
        """
        opponent = str(opponent_team or "").strip()
        if not opponent:
            return {"error": "opponent_team is required"}

        all_matches = await self._storage.get_all_matches()
        opp_matches = [
            m
            for m in all_matches
            if opponent.lower() in str(m.get("home_team") or "").lower()
            or opponent.lower() in str(m.get("away_team") or "").lower()
        ]
        if not opp_matches:
            return {
                "provider_available": False,
                "state": "no_data",
                "opponent": opponent,
                "honesty_note": (
                    "no stored matches feature this opponent; a dossier "
                    "would be a template, not scouting"
                ),
            }

        rows: list[dict[str, Any]] = []
        for m in opp_matches:
            mid = int(m["id"])
            # limit=5000: the default 200-event page would truncate a real
            # match and silently bias the tallies.
            events = await self._storage.get_match_events(mid, limit=5000)
            try:
                player_rows = await self._storage.get_match_players(mid, limit=200)
            except Exception:
                player_rows = []
            name_by_track: dict[int, str] = {}
            for p in player_rows:
                tid = p.get("track_id")
                nm = p.get("name")
                if tid is not None and nm:
                    name_by_track[int(tid)] = str(nm)

            opp_is_home = opponent.lower() in str(m.get("home_team") or "").lower()
            own_side = "home" if opp_is_home else "away"
            their_side = "away" if opp_is_home else "home"

            goals_for = 0
            goals_against = 0
            scorers: list[dict[str, Any]] = []
            for e in events:
                if e.get("event_type") != "goal":
                    continue
                if e.get("team") == own_side:
                    goals_for += 1
                elif e.get("team") == their_side:
                    goals_against += 1
                tid = e.get("from_track_id")
                scorer = name_by_track.get(int(tid)) if tid is not None else None
                if e.get("team") == own_side and scorer:
                    scorers.append({"player": scorer, "goals": 1})

            # Possession as completed-pass share (labeled in provenance).
            completed = [e for e in events if e.get("event_type") == "pass" and e.get("completed")]
            own_passes = sum(1 for e in completed if e.get("team") == own_side)
            total_passes = sum(1 for e in completed if e.get("team") in (own_side, their_side))
            poss = (100.0 * own_passes / total_passes) if total_passes else 50.0

            corners_for = sum(
                1
                for e in events
                if e.get("team") == own_side and "corner" in str(e.get("event_type") or "")
            )

            rows.append(
                {
                    "possession_pct": poss,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "corners_for": corners_for,
                    "scorers": scorers,
                    "players": [
                        {"name": s["player"], "position": "", "minutes_played": 0, "goals": 1}
                        for s in scorers
                    ],
                }
            )

        from kawkab.analysis.opponent_dossier import generate_dossier

        dossier = generate_dossier(opponent, rows, our_team_name="")
        out = dossier.to_dict()
        small = len(rows) < SMALL_SAMPLE_MATCHES
        out["state"] = "ok"
        out["provider_available"] = True
        out["provenance"] = {
            "matches_analyzed": len(rows),
            "match_ids": [int(m["id"]) for m in opp_matches],
            "sample_size": "small_sample" if small else "adequate",
            "honesty_note": (
                "computed from the club's own stored footage of this "
                "opponent; possession is completed-pass share, not "
                "tracking-true possession; pressing style and build-up "
                "pattern need tracking data this event store does not "
                "produce, so they read unknown; scorers are attributed "
                "only where track ids resolve to named players"
            ),
            "scoreline_note": (
                "scoreline prediction is a naive average of past results, not a model forecast"
            ),
        }
        return out

    # -- Vendor import probing --------------------------------------------------

    async def vendor_import_status(self) -> dict[str, Any]:
        """Honest availability of kloppy-backed vendor imports."""
        return _probe_kloppy()

    # -- Set-play library ----------------------------------------------------------

    async def set_play_library(self) -> dict[str, Any]:
        """Surface the club's set-play knowledge: drills, rules, game model."""
        drills = self._kb.get_all_drills() if self._kb is not None else []
        set_piece_drills = [
            {
                "drill_id": d.drill_id,
                "name": d.name,
                "category": d.category,
                "targets": list(d.targets),
                "duration_min": d.duration_min,
                "players_required": d.players_required,
            }
            for d in drills
            if "set_piece" in str(d.category).lower()
            or "set piece" in str(d.category).lower()
            or "aerial" in str(d.category).lower()
        ]
        rules = self._kb.get_all_rules() if self._kb is not None else []
        set_piece_rules = [
            {
                "rule_id": r.rule_id,
                "name": r.names.get("en", r.rule_id),
                "category": r.category,
                "severity": r.severity,
            }
            for r in rules
            if "set_piece" in str(r.category).lower()
            or "set piece" in str(r.category).lower()
            or "aerial" in str(r.category).lower()
        ]
        game_model = None
        try:
            game_model = await self._storage.get_active_game_model(team_id=None)
        except Exception:
            game_model = None
        return {
            "drills": set_piece_drills,
            "rules": set_piece_rules,
            "game_model_set_pieces": (
                {
                    "offence": (game_model or {}).get("set_pieces_off") or "",
                    "defence": (game_model or {}).get("set_pieces_def") or "",
                }
                if game_model
                else None
            ),
            "provenance": {
                "basis": "knowledge base drills/rules + game_model",
                "drills_total": len(set_piece_drills),
                "rules_total": len(set_piece_rules),
            },
        }
