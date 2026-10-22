"""Handler for live-tagging bridge methods — sessions, tagging, KPIs,
pitch map and xG chart."""

from __future__ import annotations

import json

from kawkab.core.logging import get_logger
from kawkab.core.security import ErrorSanitizer
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


def _compute_hot_zones(events, grid_cols=6, grid_rows=4):
    if not events:
        return []
    # Coerce to float: coordinate values can arrive as strings depending on
    # the producer (JS bridge numbers-vs-strings), and the arithmetic below
    # raises TypeError on str - str otherwise.
    x_vals = [float(e["x"]) for e in events if e.get("x") is not None]
    y_vals = [float(e["y"]) for e in events if e.get("y") is not None]
    if not x_vals or not y_vals:
        return []
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)
    x_range = max(max_x - min_x, 1)
    y_range = max(max_y - min_y, 1)
    cells = {}
    for e in events:
        if e.get("x") is None or e.get("y") is None:
            continue
        cx = int((float(e["x"]) - min_x) / x_range * grid_cols)
        cy = int((float(e["y"]) - min_y) / y_range * grid_rows)
        key = f"{cx},{cy}"
        cells[key] = cells.get(key, 0) + 1
    max_count = max(cells.values()) if cells else 1
    return [
        {
            "x": int(k.split(",")[0]),
            "y": int(k.split(",")[1]),
            "count": v,
            "intensity": round(v / max_count, 2),
        }
        for k, v in cells.items()
    ]


class LiveHandler(BridgeHandlerBase):
    """Live-tagging bridge surface."""

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    # ================================================================
    # Sprint 4 — Live Tagging Service
    # ================================================================

    async def live_start_session(self, home_team="Home", away_team="Away"):
        try:
            svc = self._services.get("live_tagging_service")
            if svc is None:
                from kawkab.services.live_tagging_service import LiveTaggingService

                svc = LiveTaggingService()
                self._services["live_tagging_service"] = svc
            return svc.start_session(home_team, away_team)
        except Exception as e:
            logger.error(f"live_start_session failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def live_stop_session(self):
        try:
            svc = self._services.get("live_tagging_service")
            if svc is None:
                return json.dumps({"error": "No live tagging service"})
            return svc.stop_session()
        except Exception as e:
            logger.error(f"live_stop_session failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def live_tag_event(self, event_type, team="", player_id=0, notes="", x=None, y=None):
        try:
            svc = self._services.get("live_tagging_service")
            if svc is None:
                return json.dumps({"error": "No live tagging service"})
            return svc.tag_event(event_type, team, player_id, notes, x, y)
        except Exception as e:
            logger.error(f"live_tag_event failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def live_set_period(self, period):
        try:
            svc = self._services.get("live_tagging_service")
            if svc is None:
                return json.dumps({"error": "No live tagging service"})
            return svc.set_period(period)
        except Exception as e:
            logger.error(f"live_set_period failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def live_get_stats(self):
        try:
            svc = self._services.get("live_tagging_service")
            if svc is None:
                return json.dumps({"stats": {"tags_count": 0}})
            return svc.get_stats()
        except Exception as e:
            logger.error(f"live_get_stats failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def live_get_tags(self):
        try:
            svc = self._services.get("live_tagging_service")
            if svc is None:
                return json.dumps({"tags": [], "total": 0})
            return svc.get_all_tags()
        except Exception as e:
            logger.error(f"live_get_tags failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def live_clear_tags(self):
        try:
            svc = self._services.get("live_tagging_service")
            if svc is None:
                return json.dumps({"error": "No live tagging service"})
            return svc.clear_tags()
        except Exception as e:
            logger.error(f"live_clear_tags failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def live_get_hotkeys(self):
        try:
            svc = self._services.get("live_tagging_service")
            if svc is None:
                return json.dumps({"hotkeys": {}})
            return svc.get_hotkeys()
        except Exception as e:
            logger.error(f"live_get_hotkeys failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def live_export(self):
        try:
            svc = self._services.get("live_tagging_service")
            if svc is None:
                return json.dumps({"error": "No live tagging service"})
            return svc.export_tags()
        except Exception as e:
            logger.error(f"live_export failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ================================================================
    # Phase 6 Sprint 2 — Live Tagging Dashboard
    # ================================================================

    async def get_live_kpis(self, session_id):
        try:
            svc = self._services.get("live_tagging_service")
            if svc is None:
                return json.dumps({"error": "No live tagging service"})
            raw = json.loads(svc.get_stats())
            if "error" in raw:
                return json.dumps(raw)
            s = raw.get("stats", {})
            ev = s.get("events_by_type", {})
            # The tagging service counts shots match-wide (no per-team keys
            # exist); the old code assigned the same total to both teams and
            # summed it -- double-counting. Report it once, honestly.
            total_shots = ev.get("shot", 0)
            home_shots = total_shots
            away_shots = 0
            home_goals = s.get("home_goals", 0)
            away_goals = s.get("away_goals", 0)
            total_shots_on = ev.get("shot_ontarget", 0) or ev.get("shot_on_target", 0) or 0
            home_shots_on = total_shots_on
            away_shots_on = 0
            # Rough league-average placeholder: a shot is worth ~0.11 xG.
            # Labeled as approximate -- the real per-shot model lives in
            # kawkab.core.xg_model (needs distance/angle, which tagging
            # does not capture).
            xg_approx = round(total_shots * 0.11, 2)
            xg_diff = round(xg_approx - (home_goals + away_goals) * 0.5, 2)
            period = svc._current_period if hasattr(svc, "_current_period") else 1
            return json.dumps(
                {
                    "possession_pct": s.get("home_possession_pct", 50.0),
                    "shots": total_shots,
                    "shots_ontarget": total_shots_on,
                    "goals": home_goals + away_goals,
                    "xg": xg_approx,
                    "xg_is_approx": True,
                    "xg_diff": xg_diff,
                    "period": period,
                    "team_stats": {
                        "home": {
                            "goals": home_goals,
                            "shots": home_shots,
                            "shots_ontarget": home_shots_on,
                        },
                        "away": {
                            "goals": away_goals,
                            "shots": away_shots,
                            "shots_ontarget": away_shots_on,
                        },
                    },
                }
            )
        except Exception as e:
            logger.error(f"get_live_kpis failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_live_pitch_map(self, session_id):
        try:
            svc = self._services.get("live_tagging_service")
            if svc is None:
                return json.dumps({"error": "No live tagging service"})
            raw = json.loads(svc.get_all_tags())
            if "error" in raw:
                return json.dumps(raw)
            tags = raw.get("tags", [])
            home_events = []
            away_events = []
            for t in tags:
                entry = {"type": t.get("type"), "x": t.get("x"), "y": t.get("y"), "t": t.get("t")}
                if t.get("team") == svc._home_team:
                    home_events.append(entry)
                elif t.get("team") == svc._away_team or t.get("type") in (
                    "goal",
                    "shot",
                    "pass",
                    "tackle",
                ):
                    away_events.append(entry)
            home_hot = _compute_hot_zones([e for e in home_events if e["x"] is not None])
            away_hot = _compute_hot_zones([e for e in away_events if e["x"] is not None])
            return json.dumps(
                {
                    "home_events": home_events,
                    "away_events": away_events,
                    "home_hot_zones": home_hot,
                    "away_hot_zones": away_hot,
                }
            )
        except Exception as e:
            logger.error(f"get_live_pitch_map failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_live_xg_chart(self, session_id):
        try:
            svc = self._services.get("live_tagging_service")
            if svc is None:
                return json.dumps({"error": "No live tagging service"})
            raw = json.loads(svc.get_all_tags())
            if "error" in raw:
                return json.dumps(raw)
            tags = raw.get("tags", [])
            shot_tags = [t for t in tags if t.get("type") == "shot" or t.get("type") == "goal"]
            timeline = []
            home_cum = 0.0
            away_cum = 0.0
            for t in shot_tags:
                minute = int(t.get("t", 0)) // 60
                xg_val = 0.11
                team = t.get("team", "")
                if team == svc._home_team if hasattr(svc, "_home_team") else "":
                    home_cum += xg_val
                else:
                    away_cum += xg_val
                timeline.append(
                    {"minute": minute, "home_xg": round(home_cum, 2), "away_xg": round(away_cum, 2)}
                )
            return json.dumps(
                {
                    "timeline": timeline,
                    "cumulative_home": round(home_cum, 2),
                    "cumulative_away": round(away_cum, 2),
                }
            )
        except Exception as e:
            logger.error(f"get_live_xg_chart failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
