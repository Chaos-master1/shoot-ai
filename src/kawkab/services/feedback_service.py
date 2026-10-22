"""Feedback and telemetry service for coach validation (v0.8.0).

Collects structured feedback from coaches:
- Overall rating (1-5 stars)
- Per-feature ratings (tracking, events, reports, UI)
- Free-text comments
- Issue reports with severity
- Usage analytics (anonymized)

All data is stored locally in SQLite for privacy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CoachFeedback:
    """Structured feedback from a coach."""

    coach_id: str
    match_id: int
    overall_rating: int  # 1-5
    tracking_rating: int | None = None  # 1-5
    events_rating: int | None = None  # 1-5
    report_rating: int | None = None  # 1-5
    ui_rating: int | None = None  # 1-5
    comments: str = ""
    issues: list[dict] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IssueReport:
    """A reported issue with context."""

    category: str  # "tracking", "events", "performance", "ui", "crash", "other"
    severity: str  # "low", "medium", "high", "critical"
    description: str
    match_id: int | None = None
    screenshot_path: str | None = None
    logs: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UsageSession:
    """Anonymized usage session data."""

    session_id: str
    features_used: list[str]  # e.g., ["analyze", "export_csv", "benchmark"]
    duration_seconds: float
    match_count: int
    gpu_tier: str
    model_size: str
    error_count: int = 0
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackService:
    """Collect and manage coach feedback, issues, and usage data.

    All data is stored locally. No external servers.
    """

    def __init__(self, storage_service=None) -> None:
        self.storage = storage_service
        self._pending_feedback: list[CoachFeedback] = []
        self._pending_issues: list[IssueReport] = []
        self._pending_sessions: list[UsageSession] = []

    async def submit_feedback(self, feedback: CoachFeedback) -> int:
        """Submit coach feedback. Returns feedback ID."""
        if self.storage is not None:
            feedback_id = await self.storage.save_feedback(feedback.to_dict())
            logger.info(f"Feedback saved: ID={feedback_id}, coach={feedback.coach_id}")
            return feedback_id
        else:
            self._pending_feedback.append(feedback)
            logger.info(f"Feedback queued (no storage): coach={feedback.coach_id}")
            return len(self._pending_feedback)

    async def submit_issue(self, issue: IssueReport) -> int:
        """Submit an issue report. Returns issue ID."""
        if self.storage is not None:
            issue_id = await self.storage.save_issue(issue.to_dict())
            logger.warning(
                f"Issue reported: ID={issue_id}, category={issue.category}, severity={issue.severity}"
            )
            return issue_id
        else:
            self._pending_issues.append(issue)
            logger.warning(f"Issue queued (no storage): category={issue.category}")
            return len(self._pending_issues)

    async def record_session(self, session: UsageSession) -> int:
        """Record an anonymized usage session. Returns session ID."""
        if self.storage is not None:
            session_id = await self.storage.save_usage_session(session.to_dict())
            logger.info(f"Session recorded: ID={session_id}, features={session.features_used}")
            return session_id
        else:
            self._pending_sessions.append(session)
            logger.info(f"Session queued (no storage): features={session.features_used}")
            return len(self._pending_sessions)

    def get_pending_counts(self) -> dict[str, int]:
        """Get counts of pending items (when storage is offline)."""
        return {
            "feedback": len(self._pending_feedback),
            "issues": len(self._pending_issues),
            "sessions": len(self._pending_sessions),
        }

    async def get_all_feedback(self) -> list[dict]:
        """Get all feedback entries."""
        if self.storage is not None:
            return await self.storage.get_all_feedback()
        return [f.to_dict() for f in self._pending_feedback]

    async def get_all_issues(self) -> list[dict]:
        """Get all issue reports."""
        if self.storage is not None:
            return await self.storage.get_all_issues()
        return [i.to_dict() for i in self._pending_issues]

    async def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for dashboard display."""
        feedback = await self.get_all_feedback()
        issues = await self.get_all_issues()

        if not feedback:
            return {
                "total_feedback": 0,
                "average_rating": 0.0,
                "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                "total_issues": len(issues),
                "issue_by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            }

        ratings = [f["overall_rating"] for f in feedback]
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in ratings:
            if r in distribution:
                distribution[r] += 1

        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for i in issues:
            s = i.get("severity", "low")
            if s in severity_counts:
                severity_counts[s] += 1

        return {
            "total_feedback": len(feedback),
            "average_rating": sum(ratings) / len(ratings) if ratings else 0.0,
            "rating_distribution": distribution,
            "total_issues": len(issues),
            "issue_by_severity": severity_counts,
        }
