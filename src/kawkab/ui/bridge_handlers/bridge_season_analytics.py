"""Season-level Pro Analytics — multi-match aggregations.

Surfaces the season/admin modules (formation_effectiveness,
suspension_tracker, fixture_difficulty) with the same honest
data_available discipline as the match-level Pro Analytics handler.
These need cross-match context, so they live on their own handler fed
by the full match list, not single-match reports.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from kawkab.core.security import ErrorSanitizer
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = logging.getLogger(__name__)


class SeasonAnalyticsHandler(BridgeHandlerBase):
    """Cross-match analytics: formation trends, discipline risk, fixture
    difficulty over the stored match list."""

    def __init__(self, bridge, services: dict[str, Any], rate_limiter=None) -> None:
        super().__init__(bridge, services, rate_limiter)

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    async def get_season_pro_report(self) -> str:
        self._check_rate_limit()
        try:
            matches = await self.storage_service.get_all_matches()
            if not matches or len(matches) < 2:
                return json.dumps(
                    {
                        "success": False,
                        "data_available": False,
                        "reason": "season analytics needs at least 2 stored matches",
                    }
                )

            report: dict[str, Any] = {
                "success": True,
                "n_matches": len(matches),
                "blocks": {},
            }
            blocks = report["blocks"]

            all_events: dict[int, list[dict]] = {}
            for m in matches:
                mid = int(m.get("id", 0))
                if mid:
                    all_events[mid] = await self._fetch_all_events(mid)

            blocks["formation_trends"] = self._formation_trends_block(matches, all_events)
            blocks["discipline"] = self._discipline_block(matches, all_events)
            blocks["fixture_difficulty"] = self._fixture_difficulty_block(matches, all_events)

            return json.dumps(report)
        except Exception as e:
            logger.error(f"get_season_pro_report failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ── plumbing ───────────────────────────────────────────────────────

    async def _fetch_all_events(self, match_id: int) -> list[dict]:
        out: list[dict] = []
        offset = 0
        while True:
            page = await self.storage_service.get_match_events(match_id, limit=500, offset=offset)
            if not page:
                break
            for row in page:
                e = dict(row)
                if isinstance(e.get("metadata"), str):
                    try:
                        e["metadata"] = json.loads(e["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        e["metadata"] = {}
                e["type"] = e.get("event_type") or e.get("type") or "unknown"
                meta_raw = e.get("metadata")
                meta_val: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
                e["metadata"] = meta_val
                for key in ("x", "y", "is_goal", "card_type"):
                    if key in meta_val and e.get(key) is None:
                        e[key] = meta_val[key]
                out.append(e)
            if len(page) < 500:
                break
            offset += 500
        return out

    # ── blocks ──────────────────────────────────────────────────────────

    def _formation_trends_block(
        self, matches: list[dict], all_events: dict[int, list[dict]]
    ) -> dict[str, Any]:
        """Formation effectiveness across stored matches.

        No lineup/formation column exists in the schema and the real
        formation detector lives in the tracking pipeline (needs track
        data, not events). Approximation, honestly labeled: bucket each
        match's home-team event locations into positional lines by
        x-quartiles (GK-excluded) and read line counts as D-M-F. Good
        enough for trend comparison, not a pitch-perfect formation read.
        """
        from kawkab.core.formation_effectiveness import FormationEffectivenessAnalyzer

        try:
            history: list[dict[str, Any]] = []
            for m in matches:
                mid = int(m.get("id", 0))
                events = all_events.get(mid, [])
                if not events:
                    continue

                def _formation_from_events(team_events: list[dict]) -> str:
                    xs = [
                        e["x"]
                        for e in team_events
                        if e.get("x") is not None and 0.0 <= e.get("x", -1) <= 105.0
                    ]
                    if len(xs) < 20:
                        return "unknown"
                    xs_sorted = sorted(xs)
                    n = len(xs_sorted)
                    q = [xs_sorted[int(n * f)] for f in (0.0, 0.25, 0.5, 0.75)]
                    depth = sum(1 for i in range(1, len(q)) if q[i] - q[i - 1] > 18.0)
                    flat = max(q) - min(q) < 45.0
                    if flat:
                        return "flat-4"
                    return ["deep-block", "mid-block", "high-line", "very-high-line"][min(depth, 3)]

                goals_h = sum(
                    1 for e in events if e.get("type") == "goal" and e.get("team") == "home"
                )
                goals_a = sum(
                    1 for e in events if e.get("type") == "goal" and e.get("team") == "away"
                )
                history.append(
                    {
                        "match_id": mid,
                        "formation": _formation_from_events(
                            [e for e in events if e.get("team") == "home"]
                        ),
                        "opponent_formation": _formation_from_events(
                            [e for e in events if e.get("team") == "away"]
                        ),
                        "result": "W" if goals_h > goals_a else ("L" if goals_h < goals_a else "D"),
                        "goals_for": goals_h,
                        "goals_against": goals_a,
                    }
                )
            usable = [h for h in history if h["formation"] != "unknown"]
            if len(usable) < 2:
                return {
                    "data_available": False,
                    "reason": "fewer than 2 matches with enough located events "
                    "to approximate a shape (approximation from event "
                    "positions; the true formation detector needs "
                    "tracking data)",
                }

            analyzer = FormationEffectivenessAnalyzer()
            comparisons = analyzer.compare_formation_performances(usable)
            flexibility = analyzer.compute_formation_flexibility_score(usable)
            return {
                "data_available": True,
                "method": "approximated from event x-distributions (not tracking)",
                "n_matches_with_shapes": len(usable),
                "formation_history": usable[:40],
                "comparisons": comparisons,
                "flexibility_score": flexibility,
            }
        except Exception as exc:
            return {"data_available": False, "reason": f"formation_trends: {exc}"}

    def _discipline_block(
        self, matches: list[dict], all_events: dict[int, list[dict]]
    ) -> dict[str, Any]:
        """Suspension risk from accumulated card events across the season."""
        from kawkab.core.suspension_tracker import analyze_suspensions

        try:
            card_events: list[dict] = []
            for events in all_events.values():
                for e in events:
                    if e.get("type") in ("card", "yellow_card", "red_card", "foul"):
                        card_events.append(e)
            if not card_events:
                return {
                    "data_available": False,
                    "reason": "no card events stored in any match — "
                    "live tagging or event import needed",
                }
            report = analyze_suspensions(card_events, competition="default", team_id="home")
            data = report.to_dict() if hasattr(report, "to_dict") else vars(report)
            return {"data_available": True, **data}
        except Exception as exc:
            return {"data_available": False, "reason": f"discipline: {exc}"}

    def _fixture_difficulty_block(
        self, matches: list[dict], all_events: dict[int, list[dict]]
    ) -> dict[str, Any]:
        """Difficulty of the stored fixture list, strengths inferred from
        stored results (honest about the inference)."""
        from kawkab.core.fixture_difficulty import analyze_fixture_difficulty

        try:
            # Infer opponent strength from stored results: points-per-game
            # against us (0-100 scale). Matches without events are skipped.
            opponent_strength: dict[str, float] = {}
            fixtures: list[dict] = []
            results_by_opp: dict[str, list[float]] = {}
            for m in matches:
                mid = int(m.get("id", 0))
                events = all_events.get(mid, [])
                if not events:
                    continue
                away = m.get("away_team") or "Away"
                goals_h = sum(
                    1 for e in events if e.get("type") == "goal" and e.get("team") == "home"
                )
                goals_a = sum(
                    1 for e in events if e.get("type") == "goal" and e.get("team") == "away"
                )
                pts = 3.0 if goals_h > goals_a else (1.0 if goals_h == goals_a else 0.0)
                results_by_opp.setdefault(away, []).append(pts)
                fixtures.append({"opponent_id": away, "venue": "home", "match_id": mid})
            for opp, pts_list in results_by_opp.items():
                opponent_strength[opp] = round(33.3 * (sum(pts_list) / max(len(pts_list), 1)), 1)
            if len(fixtures) < 2:
                return {
                    "data_available": False,
                    "reason": "fewer than 2 played fixtures with events to "
                    "infer opponent strength from",
                }

            report = analyze_fixture_difficulty(
                team_id="home", fixtures=fixtures, opponent_strength=opponent_strength
            )
            data = report.to_dict() if hasattr(report, "to_dict") else vars(report)
            return {"data_available": True, "n_fixtures": len(fixtures), **data}
        except Exception as exc:
            return {"data_available": False, "reason": f"fixture_difficulty: {exc}"}
