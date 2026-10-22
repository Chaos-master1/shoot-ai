"""Handler for video coding workspace bridge methods — Sportscode/Nacsport-style manual tagging."""

from __future__ import annotations

import json
from pathlib import Path

from kawkab.core.logging import get_logger
from kawkab.core.security import ErrorSanitizer
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class CodingHandler(BridgeHandlerBase):
    """Handles manual video tagging operations for the coding workspace."""

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    @property
    def clip_service(self):
        return self._services.get("clip_service")

    # Keeps the narrower "coding" bucket default; the acquire() contract
    # itself lives in BridgeHandlerBase.
    def _check_rate_limit(self, category: str = "coding") -> None:
        super()._check_rate_limit(category)

    # ── Tag CRUD ─────────────────────────────────────────────────

    async def save_tag(self, match_id: int, tag_json: str) -> str:
        """Save a manual coding tag from the workspace."""
        try:
            self._check_rate_limit("coding")
            tag = json.loads(tag_json)
            tag_id = await self.storage_service.save_coding_tag(match_id, tag)
            if not tag_id:
                return json.dumps({"error": "Tag rejected: event_type/tag_type required"})
            return json.dumps({"success": True, "tag_id": tag_id})
        except Exception as e:
            logger.error(f"save_tag failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_tags(self, match_id: int) -> str:
        """Get all coding tags for a match."""
        try:
            self._check_rate_limit("coding")
            tags = await self.storage_service.get_coding_tags(match_id)
            return json.dumps({"success": True, "tags": tags})
        except Exception as e:
            logger.error(f"get_tags failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def update_tag(self, tag_id: int, updates_json: str) -> str:
        """Update a coding tag's fields."""
        try:
            self._check_rate_limit("coding")
            updates = json.loads(updates_json)
            ok = await self.storage_service.update_coding_tag(tag_id, updates)
            return json.dumps({"success": ok})
        except Exception as e:
            logger.error(f"update_tag failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def delete_tag(self, tag_id: int) -> str:
        """Delete a coding tag."""
        try:
            self._check_rate_limit("coding")
            ok = await self.storage_service.delete_coding_tag(tag_id)
            return json.dumps({"success": ok})
        except Exception as e:
            logger.error(f"delete_tag failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_tag_stats(self, match_id: int) -> str:
        """Get aggregate coding tag stats for a match."""
        try:
            self._check_rate_limit("coding")
            stats = await self.storage_service.get_coding_tag_stats(match_id)
            return json.dumps({"success": True, "stats": stats})
        except Exception as e:
            logger.error(f"get_tag_stats failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_tags_by_type(self, match_id: int, event_type: str) -> str:
        """Get coding tags filtered by event type."""
        try:
            self._check_rate_limit("coding")
            tags = await self.storage_service.get_coding_tags_by_type(match_id, event_type)
            return json.dumps({"success": True, "tags": tags})
        except Exception as e:
            logger.error(f"get_tags_by_type failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_tags_by_player(self, match_id: int, player_track_id: int) -> str:
        """Get coding tags filtered by player track ID."""
        try:
            self._check_rate_limit("coding")
            tags = await self.storage_service.get_coding_tags_by_player(match_id, player_track_id)
            return json.dumps({"success": True, "tags": tags})
        except Exception as e:
            logger.error(f"get_tags_by_player failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_match_players_simple(self, match_id: int) -> str:
        """Get simplified player list for coding workspace."""
        try:
            self._check_rate_limit("coding")
            players = await self.storage_service.get_match_players(match_id)
            simple = [
                {
                    "track_id": p.get("track_id", 0),
                    "name": p.get("name", f"Player {p.get('track_id', 0)}"),
                    "jersey": p.get("jersey_number", "?"),
                    "team": p.get("team", "unknown"),
                }
                for p in players
            ]
            return json.dumps({"success": True, "players": simple})
        except Exception as e:
            logger.error(f"get_match_players_simple failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ── Clip extraction from tags ─────────────────────────────────

    async def extract_tag_clip(self, match_id: int, tag_id: int) -> str:
        """Generate a video clip from a coding tag using lead/lag times."""
        try:
            self._check_rate_limit("coding")
            tags = await self.storage_service.get_coding_tags(match_id)
            tag = next((t for t in tags if t.get("id") == tag_id), None)
            if not tag:
                return json.dumps({"error": "Tag not found"})

            match = await self.storage_service.get_match(match_id)
            if not match or not match.get("video_path"):
                return json.dumps({"error": "Match video not found"})

            video_path = Path(match["video_path"])
            if not video_path.exists():
                return json.dumps({"error": f"Video file not found: {video_path}"})

            lead_s = tag.get("lead_ms", 2000) / 1000.0
            lag_s = tag.get("lag_ms", 3000) / 1000.0
            start = max(0, tag["video_time"] - lead_s)
            end = tag["video_time"] + lag_s

            clip_svc = self.clip_service
            if clip_svc is None:
                from kawkab.services.clip_service import ClipExtractionService

                clip_svc = ClipExtractionService()

            output_name = (
                f"tag_{tag_id}_{tag.get('event_type', 'event')}_{tag.get('video_time', 0):.0f}s.mp4"
            )
            clip_path = await clip_svc.extract_clip(
                video_path=video_path,
                start_time=start,
                end_time=end,
                output_name=output_name,
                quality="medium",
            )
            if clip_path:
                return json.dumps({"success": True, "clip_path": str(clip_path)})
            return json.dumps({"error": "Clip extraction failed"})
        except Exception as e:
            logger.error(f"extract_tag_clip failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def extract_tag_clips_batch(self, match_id: int, tag_ids_json: str) -> str:
        """Generate video clips for multiple coding tags."""
        try:
            self._check_rate_limit("coding")
            tag_ids = json.loads(tag_ids_json)
            if not tag_ids:
                return json.dumps({"error": "No tag IDs provided"})

            results = []
            for tid in tag_ids:
                result = await self.extract_tag_clip(match_id, tid)
                results.append(json.loads(result))

            return json.dumps({"success": True, "results": results})
        except Exception as e:
            logger.error(f"extract_tag_clips_batch failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ── Template management ───────────────────────────────────────

    async def get_default_tag_templates(self) -> str:
        self._check_rate_limit("coding")
        """Get the default set of matrix tagging button templates."""
        templates = {
            "categories": [
                {
                    "id": "attack",
                    "label": "Attack",
                    "color": "#16a34a",
                    "buttons": [
                        {"id": "pass", "label": "Pass", "shortcut": "1", "color": "#22c55e"},
                        {
                            "id": "through_ball",
                            "label": "Through",
                            "shortcut": "q",
                            "color": "#4ade80",
                        },
                        {"id": "shot", "label": "Shot", "shortcut": "2", "color": "#ef4444"},
                        {"id": "goal", "label": "Goal", "shortcut": "3", "color": "#dc2626"},
                        {"id": "dribble", "label": "Dribble", "shortcut": "4", "color": "#3b82f6"},
                        {"id": "cross", "label": "Cross", "shortcut": "5", "color": "#60a5fa"},
                        {"id": "carry", "label": "Carry", "shortcut": "6", "color": "#818cf8"},
                        {
                            "id": "key_pass",
                            "label": "Key Pass",
                            "shortcut": "w",
                            "color": "#a3e635",
                        },
                    ],
                },
                {
                    "id": "defense",
                    "label": "Defense",
                    "color": "#ea580c",
                    "buttons": [
                        {"id": "tackle", "label": "Tackle", "shortcut": "7", "color": "#f97316"},
                        {
                            "id": "interception",
                            "label": "Intercept",
                            "shortcut": "8",
                            "color": "#fb923c",
                        },
                        {"id": "press", "label": "Press", "shortcut": "9", "color": "#a855f7"},
                        {"id": "clearance", "label": "Clear", "shortcut": "e", "color": "#c084fc"},
                        {"id": "block", "label": "Block", "shortcut": "r", "color": "#e879f9"},
                        {"id": "foul", "label": "Foul", "shortcut": "t", "color": "#f43f5e"},
                    ],
                },
                {
                    "id": "mistake",
                    "label": "Mistake",
                    "color": "#dc2626",
                    "buttons": [
                        {
                            "id": "error_positional",
                            "label": "Pos Error",
                            "shortcut": "z",
                            "color": "#92400e",
                        },
                        {
                            "id": "error_technical",
                            "label": "Tech Error",
                            "shortcut": "x",
                            "color": "#b45309",
                        },
                        {
                            "id": "error_decision",
                            "label": "Dec Error",
                            "shortcut": "c",
                            "color": "#d97706",
                        },
                        {
                            "id": "error_physical",
                            "label": "Phy Error",
                            "shortcut": "v",
                            "color": "#f59e0b",
                        },
                        {
                            "id": "missed_tackle",
                            "label": "Miss Tackle",
                            "shortcut": "b",
                            "color": "#ef4444",
                        },
                        {
                            "id": "bad_pass",
                            "label": "Bad Pass",
                            "shortcut": "n",
                            "color": "#fca5a5",
                        },
                    ],
                },
                {
                    "id": "setpiece",
                    "label": "Set Piece",
                    "color": "#0891b2",
                    "buttons": [
                        {"id": "corner", "label": "Corner", "shortcut": "m", "color": "#06b6d4"},
                        {
                            "id": "free_kick",
                            "label": "Free Kick",
                            "shortcut": ",",
                            "color": "#22d3ee",
                        },
                        {
                            "id": "throw_in",
                            "label": "Throw In",
                            "shortcut": ".",
                            "color": "#67e8f9",
                        },
                        {
                            "id": "goal_kick",
                            "label": "Goal Kick",
                            "shortcut": "/",
                            "color": "#a5f3fc",
                        },
                        {"id": "penalty", "label": "Penalty", "shortcut": "p", "color": "#2dd4bf"},
                    ],
                },
            ],
            "lead_ms_default": 2000,
            "lag_ms_default": 3000,
        }
        return json.dumps({"success": True, "templates": templates})
