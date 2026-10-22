"""Handlers extracted from bridge_analysis.py (Phase C1 split).

Thin Qt-bridge handlers: each owns one surface and forwards to its
service layer. Registered in bridge_handlers/__init__.py and wired in
ui/bridge.py. Validation lives in the service layer unless noted.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from kawkab.core.security import ErrorSanitizer, SecurityValidator
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = logging.getLogger(__name__)


class PhysicalHandler(BridgeHandlerBase):
    """Physical surfaces: GPS file import/sessions/summaries, ACWR, and the
    pre-match briefing generator."""

    def __init__(self, bridge, services: dict[str, Any], rate_limiter=None) -> None:
        super().__init__(bridge, services, rate_limiter)
        self.bridge = bridge  # public alias kept for backward compatibility

    # _check_rate_limit inherited from BridgeHandlerBase ("analysis" bucket).
    # History: the deleted override called a nonexistent RateLimiter.check(),
    # so generate_briefing crashed with the real bridge limiter wired in.

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    async def import_gps_file(
        self,
        match_id: str,
        player_id: str,
        file_path: str,
        session_type: str = "match",
        vendor: str = "catapult",
    ) -> str:
        """Import a GPS data file and associate with a match and player."""
        try:
            mid = SecurityValidator.validate_match_id(match_id)
            pid = SecurityValidator.validate_int(player_id)
            # GPS vendor exports are CSV/JSON data files, not videos:
            # validate_video_path (video-only extensions) rejected every
            # legitimate Catapult/Kinexon import. The wearable validator
            # carries exactly the right allowlist (.gpx/.fit/.tcx/...)
            # plus the documents-directory confinement.
            path = SecurityValidator.validate_wearable_path(file_path)

            from kawkab.services.gps_import import compute_session_summary
            from kawkab.services.gps_import import import_gps_file as _import

            samples = _import(str(path))
            if not samples:
                return json.dumps({"error": "No samples found in GPS file"})

            svc = self.storage_service
            # Storage methods are async (sqlite via aiosqlite-style loop);
            # every call below MUST be awaited or the coroutine object is
            # returned unexecuted and the slot replies with a JSON
            # serialization error instead of touching the database.
            session_id = await svc.save_gps_session(mid, pid, session_type, vendor)
            if not session_id:
                return json.dumps({"error": "Failed to create GPS session"})

            count = await svc.save_gps_samples_bulk(session_id, samples)
            summary = compute_session_summary(samples)
            await svc.update_gps_session_stats(session_id, summary)

            return json.dumps(
                {
                    "success": True,
                    "session_id": session_id,
                    "sample_count": count,
                    "summary": summary,
                }
            )
        except Exception as e:
            logger.error(f"import_gps_file failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_gps_sessions(self, match_id: str) -> str:
        """Get all GPS sessions for a match."""
        try:
            mid = SecurityValidator.validate_match_id(match_id)
            sessions = (
                await self.storage_service.get_gps_sessions(mid) if self.storage_service else []
            )
            return json.dumps({"success": True, "sessions": sessions})
        except Exception as e:
            logger.error(f"get_gps_sessions failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_gps_samples(self, session_id: str) -> str:
        """Get GPS samples for a session."""
        try:
            sid = SecurityValidator.validate_int(session_id)
            samples = (
                await self.storage_service.get_gps_samples(sid) if self.storage_service else []
            )
            return json.dumps({"success": True, "samples": samples})
        except Exception as e:
            logger.error(f"get_gps_samples failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_player_gps_summary(self, player_id: str) -> str:
        """Get GPS summary for a player across sessions."""
        try:
            pid = SecurityValidator.validate_int(player_id)
            data = (
                await self.storage_service.get_player_gps_summary(pid)
                if self.storage_service
                else []
            )
            return json.dumps({"success": True, "sessions": data})
        except Exception as e:
            logger.error(f"get_player_gps_summary failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_player_acwr(self, player_id: str) -> str:
        """Get ACWR data for a player."""
        try:
            pid = SecurityValidator.validate_int(player_id)
            data = await self.storage_service.get_player_acwr(pid) if self.storage_service else []
            return json.dumps({"success": True, "acwr": data})
        except Exception as e:
            logger.error(f"get_player_acwr failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    @property
    def prematch_briefing_service(self):
        if not hasattr(self, "_prematch_briefing_service"):
            from kawkab.analysis.prematch_briefing import PreMatchBriefingService

            self._prematch_briefing_service = PreMatchBriefingService()
        return self._prematch_briefing_service

    def generate_briefing(self, match_id: str) -> str:
        """Generate a pre-match briefing for a given match."""
        try:
            self._check_rate_limit()
            match_id_val = SecurityValidator.validate_match_id(match_id)
            if self.storage_service is None:
                return json.dumps({"error": "Storage service not available"})

            match_data = self.storage_service.get_match(match_id_val) or {}
            events = self.storage_service.get_match_events(match_id_val) or []
            _ = self.storage_service.get_match_players(match_id_val) or []

            home_team = match_data.get("home_team", "Home")
            away_team = match_data.get("away_team", "Away")
            competition = match_data.get("competition", "")
            venue = match_data.get("venue", "")
            match_date = match_data.get("match_date", "")
            kickoff = match_data.get("kickoff", "")

            our_form_data: list = []
            our_injuries_data: list = []
            our_suspensions_data: list = []
            our_top_scorer = ""
            our_formation = ""
            our_predicted_lineup_data: list = []
            our_avg_possession = 50.0

            opponent_form_data: list = []
            opponent_key_players_data: list = []
            opponent_vulnerabilities: list = []
            opponent_strengths: list = []
            opponent_preferred_formation = ""
            opponent_pressing = ""
            opponent_build_up = ""

            h2h_data: list = []

            goal_counts: dict[str, int] = {}
            for ev in events:
                event_type = ev.get("event_type", "")
                _ = ev.get("team", "")
                player_name = ev.get("player", "")
                if event_type == "goal" and player_name:
                    goal_counts[player_name] = goal_counts.get(player_name, 0) + 1

            if goal_counts:
                our_top_scorer = max(goal_counts, key=goal_counts.get)  # type: ignore[arg-type]

            service = self.prematch_briefing_service

            our_side = "home"

            briefing = service.generate(
                home_team=home_team,
                away_team=away_team,
                our_side=our_side,
                competition=competition,
                venue=venue,
                match_date=match_date,
                kickoff=kickoff,
                our_form_data=our_form_data,
                our_injuries_data=our_injuries_data,
                our_suspensions_data=our_suspensions_data,
                our_top_scorer=our_top_scorer,
                our_formation=our_formation,
                our_predicted_lineup_data=our_predicted_lineup_data,
                our_avg_possession=our_avg_possession,
                opponent_form_data=opponent_form_data,
                opponent_preferred_formation=opponent_preferred_formation,
                opponent_pressing=opponent_pressing,
                opponent_build_up=opponent_build_up,
                opponent_key_players_data=opponent_key_players_data,
                opponent_vulnerabilities=opponent_vulnerabilities,
                opponent_strengths=opponent_strengths,
                h2h_data=h2h_data,
            )

            return json.dumps(
                {
                    "success": True,
                    "briefing": briefing.to_dict(),
                    "markdown": briefing.to_markdown(),
                    "html": briefing.to_html(),
                }
            )
        except Exception as e:
            logger.exception(f"generate_briefing failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
