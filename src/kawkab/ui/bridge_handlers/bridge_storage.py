"""Handler for storage bridge methods (saveEvent, deleteEvent, getMatches, getFeedback, etc.)."""

from __future__ import annotations

import json

from kawkab.core.logging import get_logger
from kawkab.core.security import ErrorSanitizer
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


class StorageHandler(BridgeHandlerBase):
    """Handles CRUD storage operations for Bridge."""

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    @property
    def feedback_service(self):
        return self._services.get("feedback_service")

    # ── Event CRUD ───────────────────────────────────────────────

    async def update_event(self, event_id, updates_json):
        try:
            self._check_rate_limit()
            updates = json.loads(updates_json)
            ok = await self.storage_service.update_event(event_id, updates)
            return json.dumps({"success": ok})
        except Exception as e:
            logger.error(f"update_event failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def delete_event(self, event_id):
        try:
            self._check_rate_limit()
            ok = await self.storage_service.delete_event(event_id)
            return json.dumps({"success": ok})
        except Exception as e:
            logger.error(f"delete_event failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    # ── Feedback ─────────────────────────────────────────────────

    async def submit_feedback(self, feedback_json):
        self._check_rate_limit()
        from kawkab.services.feedback_service import CoachFeedback

        if self.feedback_service is None:
            return json.dumps({"error": "Feedback service not available"})

        try:
            data = json.loads(feedback_json)
            feedback = CoachFeedback(
                coach_id=data.get("coach_id", "anonymous"),
                match_id=data.get("match_id", 0),
                overall_rating=data.get("overall_rating", 3),
                tracking_rating=data.get("tracking_rating"),
                events_rating=data.get("events_rating"),
                report_rating=data.get("report_rating"),
                ui_rating=data.get("ui_rating"),
                comments=data.get("comments", ""),
                issues=data.get("issues"),
            )
            fid = await self.feedback_service.submit_feedback(feedback)
            return json.dumps({"feedback_id": fid, "status": "saved"})
        except Exception as e:
            logger.error(f"submit_feedback failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def submit_issue(self, issue_json):
        self._check_rate_limit()
        from kawkab.services.feedback_service import IssueReport

        if self.feedback_service is None:
            return json.dumps({"error": "Feedback service not available"})

        try:
            data = json.loads(issue_json)
            issue = IssueReport(
                category=data.get("category", "other"),
                severity=data.get("severity", "medium"),
                description=data.get("description", ""),
                match_id=data.get("match_id"),
            )
            iid = await self.feedback_service.submit_issue(issue)
            return json.dumps({"issue_id": iid, "status": "saved"})
        except Exception as e:
            logger.error(f"submit_issue failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_feedback_stats(self):
        self._check_rate_limit()
        if self.feedback_service is None:
            return json.dumps({"error": "Feedback service not available"})

        try:
            stats = await self.feedback_service.get_summary_stats()
            return json.dumps(stats)
        except Exception as e:
            logger.error(f"get_feedback_stats failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
