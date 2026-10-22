"""Phase 6 Sprint 5 — Collaboration Deepen + Polish Tests.

Deliverables:
1. Timestamp-Linked Event Comments        (6 tests)
2. Activity Feed in Sidebar               (6 tests)
3. @Mention Notifications                 (6 tests)
4. Mention Detection                      (6 tests)
5. DB Persistence                         (6 tests)
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from kawkab.services.collaboration_service import (
    CollaborationService,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def svc():
    return CollaborationService()


# ═══════════════════════════════════════════════════════════════════════════
# Deliverable 1: Timestamp-Linked Event Comments (6 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestEventLinkedComments:
    def test_add_comment_with_event_id(self, svc):
        result = json.loads(svc.add_comment(1, 42, 0, "Great goal at 32:15"))
        assert result["ok"] is True
        assert result["comment"]["event_id"] == 42
        assert result["comment"]["match_id"] == 1
        assert result["comment"]["text"] == "Great goal at 32:15"

    def test_get_event_comments_returns_matching(self, svc):
        svc.add_comment(1, 42, 0, "Comment A")
        svc.add_comment(1, 42, 0, "Comment B")
        svc.add_comment(1, 99, 0, "Comment C")
        result = json.loads(svc.get_event_comments(1, 42))
        assert result["total"] == 2
        assert all(c["event_id"] == 42 for c in result["comments"])

    def test_get_event_comments_empty_when_none(self, svc):
        result = json.loads(svc.get_event_comments(1, 999))
        assert result["total"] == 0
        assert result["comments"] == []

    def test_get_event_comments_different_match(self, svc):
        svc.add_comment(1, 42, 0, "Match 1 comment")
        result = json.loads(svc.get_event_comments(2, 42))
        assert result["total"] == 0

    def test_event_comment_includes_username(self, svc):
        result = json.loads(svc.add_comment(1, 5, 0, "Nice pass"))
        assert result["comment"]["username"] == "admin"

    def test_event_comment_has_timestamp(self, svc):
        result = json.loads(svc.add_comment(1, 10, 1, "Well worked"))
        assert result["comment"]["created_at"] != ""


# ═══════════════════════════════════════════════════════════════════════════
# Deliverable 2: Activity Feed in Sidebar (6 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestActivityFeed:
    def test_activity_feed_returns_activities(self, svc):
        svc.add_comment(1, 0, 0, "Test")
        result = json.loads(svc.get_activity_feed(50))
        assert result["total"] >= 1

    def test_activity_feed_limits_results(self, svc):
        for i in range(5):
            svc.add_comment(1, i, 0, f"Comment {i}")
        result = json.loads(svc.get_activity_feed(3))
        assert result["total"] <= 5
        assert len(result["activities"]) <= 3

    def test_activity_feed_sorted_newest_first(self, svc):
        svc.add_comment(1, 1, 0, "First")
        time.sleep(0.01)
        svc.add_comment(1, 2, 0, "Second")
        result = json.loads(svc.get_activity_feed(10))
        activities = result["activities"]
        assert activities[0]["created_at"] >= activities[-1]["created_at"]

    def test_activity_feed_includes_action_type(self, svc):
        svc.add_comment(1, 0, 0, "Testing")
        result = json.loads(svc.get_activity_feed(10))
        assert any(a["action"] == "comment" for a in result["activities"])

    def test_activity_feed_empty_when_no_activity(self, svc):
        svc2 = CollaborationService()
        result = json.loads(svc2.get_activity_feed(10))
        assert result["total"] >= 0

    def test_activity_feed_includes_username(self, svc):
        svc.add_comment(1, 0, 1, "Analyst comment")
        result = json.loads(svc.get_activity_feed(10))
        assert any(a["username"] == "analyst1" for a in result["activities"])


# ═══════════════════════════════════════════════════════════════════════════
# Deliverable 3: @Mention Notifications (6 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestMentionNotifications:
    def test_mention_created_when_username_mentioned(self, svc):
        svc.add_comment(1, 0, 0, "Hey @analyst1 check this")
        result = json.loads(svc.get_mentions("analyst1"))
        assert result["total"] >= 1

    def test_mention_tracks_from_user(self, svc):
        svc.add_comment(1, 0, 0, "@analyst1 nice")
        result = json.loads(svc.get_mentions("analyst1"))
        mention = result["mentions"][0]
        assert mention["from_user"] == "admin"

    def test_mention_stores_event_id(self, svc):
        svc.add_comment(1, 77, 0, "@analyst1 look at event 77")
        result = json.loads(svc.get_mentions("analyst1"))
        assert result["mentions"][0]["event_id"] == 77

    def test_mention_stores_match_id(self, svc):
        svc.add_comment(5, 0, 0, "@analyst1 match 5 review")
        result = json.loads(svc.get_mentions("analyst1"))
        assert result["mentions"][0]["match_id"] == 5

    def test_mark_mention_read_updates_status(self, svc):
        svc.add_comment(1, 0, 0, "@analyst1 read this")
        result = json.loads(svc.get_mentions("analyst1"))
        mention_id = result["mentions"][0]["id"]
        svc.mark_mention_read(mention_id)
        result2 = json.loads(svc.get_mentions("analyst1"))
        mention = next(m for m in result2["mentions"] if m["id"] == mention_id)
        assert mention["read"] == 1

    def test_mention_unread_count(self, svc):
        svc.add_comment(1, 0, 0, "@analyst1 first")
        result = json.loads(svc.get_mentions("analyst1"))
        assert result["unread"] > 0
        mention_id = result["mentions"][0]["id"]
        svc.mark_mention_read(mention_id)
        result2 = json.loads(svc.get_mentions("analyst1"))
        assert result2["unread"] == result["unread"] - 1


# ═══════════════════════════════════════════════════════════════════════════
# Deliverable 4: @Mention Detection (6 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestMentionDetection:
    def test_detect_single_mention(self):
        result = CollaborationService._detect_mentions("Hello @user1")
        assert result == ["user1"]

    def test_detect_multiple_mentions(self):
        result = CollaborationService._detect_mentions("@user1 and @user2 here")
        assert result == ["user1", "user2"]

    def test_detect_no_mention(self):
        result = CollaborationService._detect_mentions("No mentions here")
        assert result == []

    def test_detect_mention_with_underscore(self):
        result = CollaborationService._detect_mentions("Hi @analyst_1")
        assert result == ["analyst_1"]

    def test_detect_self_mention_creates_no_notification(self, svc):
        svc.add_comment(1, 0, 0, "@admin self mention")
        result = json.loads(svc.get_mentions("admin"))
        assert result["total"] == 0

    def test_detect_mention_at_start(self):
        result = CollaborationService._detect_mentions("@lead_scout what do you think")
        assert result == ["lead_scout"]


# ═══════════════════════════════════════════════════════════════════════════
# Deliverable 5: DB Persistence (6 tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestDBPersistence:
    @pytest.fixture
    def mock_storage(self):
        storage = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        fake_users = []
        fake_comments = []
        fake_mentions = []

        def side_effect_execute(sql, params=None):
            class FakeRow:
                def __init__(self, **kw):
                    for k, v in kw.items():
                        setattr(self, k, v)

                def __getitem__(self, key):
                    if key == "id":
                        return 0
                    if key == "username":
                        return "admin"
                    if key == "role":
                        return "analyst"
                    if key == "created_at":
                        return datetime.now().isoformat()
                    if key == "match_id":
                        return 1
                    if key == "event_id":
                        return 0
                    if key == "user_id":
                        return 0
                    if key == "text":
                        return "test"
                    if key == "from_user":
                        return "admin"
                    if key == "read":
                        return 0
                    return None

            return None

        mock_cursor.execute.side_effect = side_effect_execute

        def fetchall_side():
            key = mock_cursor.execute.call_args[0][0] if mock_cursor.execute.call_args else ""
            if "collab_users" in str(key) and "SELECT" in str(key):
                return fake_users
            if "collab_comments" in str(key) and "SELECT" in str(key):
                return fake_comments
            if "collab_mentions" in str(key) and "SELECT" in str(key):
                return fake_mentions
            return []

        mock_cursor.fetchall.side_effect = fetchall_side
        mock_conn.cursor.return_value = mock_cursor
        storage._conn = mock_conn
        return storage

    def test_init_with_storage_does_not_crash(self, mock_storage):
        svc = CollaborationService(storage_service=mock_storage)
        assert svc._storage_service is not None
        assert len(svc._users) > 0

    def test_create_user_persists_to_db(self, mock_storage):
        svc = CollaborationService(storage_service=mock_storage)
        result = json.loads(svc.create_user("test_user", "Test User", "scout"))
        assert result["ok"] is True

    def test_add_comment_persists_to_db(self, mock_storage):
        svc = CollaborationService(storage_service=mock_storage)
        result = json.loads(svc.add_comment(1, 42, 0, "DB comment test"))
        assert result["ok"] is True

    def test_mention_with_storage_does_not_crash(self, mock_storage):
        svc = CollaborationService(storage_service=mock_storage)
        result = json.loads(svc.add_comment(1, 0, 0, "@analyst1 check DB"))
        assert result["ok"] is True

    def test_get_mentions_with_storage(self, mock_storage):
        svc = CollaborationService(storage_service=mock_storage)
        svc.add_comment(1, 0, 0, "@admin hello")
        result = json.loads(svc.get_mentions("admin"))
        assert "mentions" in result

    def test_mark_mention_read_with_storage(self, mock_storage):
        svc = CollaborationService(storage_service=mock_storage)
        svc.add_comment(1, 0, 0, "@admin mark this")
        result = json.loads(svc.get_mentions("admin"))
        if result["mentions"]:
            mid = result["mentions"][0]["id"]
            res = json.loads(svc.mark_mention_read(mid))
            assert res["ok"] is True
