"""Coach validation program — tracks feedback and usage from real coaches.

Manages the beta tester program, collects structured feedback, and tracks
which features are actually used vs ignored.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CoachProfile:
    coach_id: str
    name: str
    email: str
    club: str = ""
    level: str = "amateur"  # amateur, semi-pro, pro, youth
    joined_at: str = ""
    last_active: str = ""
    feedback_count: int = 0
    rating: float = 0.0


@dataclass
class CoachFeedback:
    feedback_id: str = ""
    coach_id: str = ""
    match_id: int = 0
    rating: int = 0  # 1-5
    comments: str = ""
    category: str = "general"  # tracking, analysis, ui, reports, calibration, other
    timestamp: str = ""
    is_critical: bool = False


@dataclass
class FeatureUsage:
    feature_name: str = ""
    coach_id: str = ""
    usage_count: int = 0
    last_used: str = ""
    first_used: str = ""


class CoachValidationService:
    """Manages the coach validation program — beta tester onboarding, feedback collection, usage tracking."""

    def __init__(self):
        self._coaches: dict[str, CoachProfile] = {}
        self._feedback: list[CoachFeedback] = []
        self._usage: dict[str, dict[str, FeatureUsage]] = {}  # coach_id -> feature_name -> usage

    def register_coach(
        self, name: str, email: str, club: str = "", level: str = "amateur"
    ) -> CoachProfile:
        coach_id = f"coach_{int(time.time())}_{hash(email) % 10000}"
        now = datetime.now(UTC).isoformat()
        profile = CoachProfile(
            coach_id=coach_id,
            name=name,
            email=email,
            club=club,
            level=level,
            joined_at=now,
            last_active=now,
        )
        self._coaches[coach_id] = profile
        logger.info(f"Registered coach: {name} ({coach_id})")
        return profile

    def get_coach(self, coach_id: str) -> CoachProfile | None:
        return self._coaches.get(coach_id)

    def list_coaches(self) -> list[CoachProfile]:
        return list(self._coaches.values())

    def remove_coach(self, coach_id: str) -> bool:
        if coach_id in self._coaches:
            del self._coaches[coach_id]
            self._usage.pop(coach_id, None)
            self._feedback = [f for f in self._feedback if f.coach_id != coach_id]
            return True
        return False

    def submit_feedback(
        self, coach_id: str, match_id: int, rating: int, comments: str, category: str = "general"
    ) -> CoachFeedback:
        now = datetime.now(UTC).isoformat()
        fb = CoachFeedback(
            feedback_id=f"fb_{int(time.time())}_{len(self._feedback)}",
            coach_id=coach_id,
            match_id=match_id,
            rating=rating,
            comments=comments,
            category=category,
            timestamp=now,
            is_critical=rating <= 2,
        )
        self._feedback.append(fb)
        if coach_id in self._coaches:
            coach = self._coaches[coach_id]
            coach.feedback_count += 1
            coach.last_active = now
            total = coach.feedback_count
            coach.rating = ((coach.rating * (total - 1)) + rating) / total
        logger.info(f"Feedback from {coach_id}: rating={rating}, category={category}")
        return fb

    def get_feedback(
        self, coach_id: str = "", category: str = "", min_rating: int = 0
    ) -> list[CoachFeedback]:
        results = self._feedback[:]
        if coach_id:
            results = [f for f in results if f.coach_id == coach_id]
        if category:
            results = [f for f in results if f.category == category]
        if min_rating:
            results = [f for f in results if f.rating >= min_rating]
        return results

    def get_critical_feedback(self) -> list[CoachFeedback]:
        return [f for f in self._feedback if f.is_critical]

    def track_usage(self, coach_id: str, feature_name: str):
        now = datetime.now(UTC).isoformat()
        if coach_id not in self._usage:
            self._usage[coach_id] = {}
        if feature_name not in self._usage[coach_id]:
            self._usage[coach_id][feature_name] = FeatureUsage(
                feature_name=feature_name,
                coach_id=coach_id,
                usage_count=0,
                first_used=now,
                last_used=now,
            )
        usage = self._usage[coach_id][feature_name]
        usage.usage_count += 1
        usage.last_used = now
        if coach_id in self._coaches:
            self._coaches[coach_id].last_active = now

    def get_usage_stats(self, coach_id: str = "") -> dict:
        if coach_id:
            usages = self._usage.get(coach_id, {}).values()
            return {
                "coach_id": coach_id,
                "total_features_used": len(usages),
                "total_actions": sum(u.usage_count for u in usages),
                "features": sorted(
                    [
                        {"name": u.feature_name, "count": u.usage_count, "last_used": u.last_used}
                        for u in usages
                    ],
                    key=lambda x: x["count"],
                    reverse=True,
                ),
            }
        all_usages = []
        for _cid, feats in self._usage.items():
            for feat in feats.values():
                all_usages.append(feat)
        return {
            "total_coaches": len(self._coaches),
            "total_feedback": len(self._feedback),
            "total_actions": sum(u.usage_count for u in all_usages),
        }

    def get_top_issues(self, limit: int = 10) -> list[dict]:
        critical = self.get_critical_feedback()
        from collections import Counter

        category_counts = Counter(f.category for f in critical)
        return [
            {
                "category": cat,
                "count": count,
                "pct": round(count / len(critical) * 100, 1) if critical else 0,
            }
            for cat, count in category_counts.most_common(limit)
        ]

    def get_validation_report(self) -> dict:
        coaches = self.list_coaches()
        feedback = self.get_feedback()
        return {
            "total_coaches": len(coaches),
            "active_coaches": sum(1 for c in coaches if c.feedback_count > 0),
            "total_feedback": len(feedback),
            "critical_issues": len(self.get_critical_feedback()),
            "average_rating": round(sum(f.rating for f in feedback) / len(feedback), 2)
            if feedback
            else 0.0,
            "top_issues": self.get_top_issues(),
            "coach_levels": {
                level: sum(1 for c in coaches if c.level == level)
                for level in {c.level for c in coaches}
            },
            "feature_adoption": {
                "tracked": sum(
                    1
                    for c in coaches
                    if any(u.usage_count > 0 for u in self._usage.get(c.coach_id, {}).values())
                ),
                "total": len(coaches),
            },
        }

    def clear(self):
        self._coaches.clear()
        self._feedback.clear()
        self._usage.clear()
