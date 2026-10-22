from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CollabUser:
    id: int = 0
    username: str = ""
    display_name: str = ""
    role: str = "analyst"  # admin, coach, analyst, scout
    created_at: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
        }


@dataclass
class Comment:
    id: int = 0
    match_id: int = 0
    event_id: int = 0
    user_id: int = 0
    username: str = ""
    text: str = ""
    created_at: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "match_id": self.match_id,
            "event_id": self.event_id,
            "user_id": self.user_id,
            "username": self.username,
            "text": self.text,
            "created_at": self.created_at,
        }


@dataclass
class ActivityEntry:
    id: int = 0
    user_id: int = 0
    username: str = ""
    action: str = ""  # "analyze", "comment", "export", "import", "tag", "review"
    description: str = ""
    match_id: int = 0
    created_at: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "username": self.username,
            "action": self.action,
            "description": self.description,
            "match_id": self.match_id,
            "created_at": self.created_at,
        }


@dataclass
class Mention:
    id: int = 0
    username: str = ""
    from_user: str = ""
    text: str = ""
    match_id: int = 0
    event_id: int = 0
    read: int = 0
    created_at: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "from_user": self.from_user,
            "text": self.text,
            "match_id": self.match_id,
            "event_id": self.event_id,
            "read": self.read,
            "created_at": self.created_at,
        }


class CollaborationService:
    _MENTION_RE = re.compile(r"@(\w+)")

    def __init__(self, storage_service=None):
        self._users: dict[int, CollabUser] = {}
        self._comments: list[Comment] = []
        self._activities: list[ActivityEntry] = []
        self._mentions: list[Mention] = []
        self._next_user_id = 1
        self._next_comment_id = 1
        self._next_activity_id = 1
        self._next_mention_id = 1
        self._storage_service = storage_service
        if storage_service is None:
            self._add_default_users()
        else:
            self._load_from_db()

    def _load_from_db(self):
        try:
            conn = self._storage_service._conn
            if conn is None:
                self._add_default_users()
                return
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, role, created_at FROM collab_users ORDER BY id")
            rows = cursor.fetchall()
            for row in rows:
                user = CollabUser(
                    id=row["id"],
                    username=row["username"],
                    display_name=row["username"],
                    role=row["role"],
                    created_at=row["created_at"],
                )
                self._users[user.id] = user
            if not self._users:
                self._add_default_users()
            else:
                self._next_user_id = max(u.id for u in self._users.values()) + 1
            cursor.execute(
                "SELECT id, match_id, event_id, user_id, username, text, created_at FROM collab_comments ORDER BY id"
            )
            rows = cursor.fetchall()
            for row in rows:
                comment = Comment(
                    id=row["id"],
                    match_id=row["match_id"],
                    event_id=row["event_id"],
                    user_id=row["user_id"],
                    username=row["username"],
                    text=row["text"],
                    created_at=row["created_at"],
                )
                self._comments.append(comment)
            if self._comments:
                self._next_comment_id = max(c.id for c in self._comments) + 1
            cursor.execute(
                "SELECT id, username, from_user, text, match_id, event_id, read, created_at FROM collab_mentions ORDER BY id"
            )
            rows = cursor.fetchall()
            for row in rows:
                mention = Mention(
                    id=row["id"],
                    username=row["username"],
                    from_user=row["from_user"],
                    text=row["text"],
                    match_id=row["match_id"],
                    event_id=row["event_id"],
                    read=row["read"],
                    created_at=row["created_at"],
                )
                self._mentions.append(mention)
            if self._mentions:
                self._next_mention_id = max(m.id for m in self._mentions) + 1
        except Exception as e:
            logger.warning(f"Could not load collab from DB: {e}")
            self._add_default_users()

    def _save_user_to_db(self, user: CollabUser):
        svc = self._storage_service
        if svc is None or svc._conn is None:
            return
        try:
            cursor = svc._conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO collab_users (id, username, role, created_at) VALUES (?, ?, ?, ?)",
                (user.id, user.username, user.role, user.created_at),
            )
            svc._conn.commit()
        except Exception as e:
            logger.warning(f"Could not save user to DB: {e}")

    def _save_comment_to_db(self, comment: Comment):
        svc = self._storage_service
        if svc is None or svc._conn is None:
            return
        try:
            cursor = svc._conn.cursor()
            cursor.execute(
                "INSERT INTO collab_comments (id, match_id, event_id, user_id, username, text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    comment.id,
                    comment.match_id,
                    comment.event_id,
                    comment.user_id,
                    comment.username,
                    comment.text,
                    comment.created_at,
                ),
            )
            svc._conn.commit()
        except Exception as e:
            logger.warning(f"Could not save comment to DB: {e}")

    def _save_mention_to_db(self, mention: Mention):
        svc = self._storage_service
        if svc is None or svc._conn is None:
            return
        try:
            cursor = svc._conn.cursor()
            cursor.execute(
                "INSERT INTO collab_mentions (id, username, from_user, text, match_id, event_id, read, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    mention.id,
                    mention.username,
                    mention.from_user,
                    mention.text,
                    mention.match_id,
                    mention.event_id,
                    mention.read,
                    mention.created_at,
                ),
            )
            svc._conn.commit()
        except Exception as e:
            logger.warning(f"Could not save mention to DB: {e}")

    def _add_default_users(self):
        self._users[0] = CollabUser(
            id=0,
            username="admin",
            display_name="Head Coach",
            role="admin",
            created_at=datetime.now().isoformat(),
        )
        self._users[1] = CollabUser(
            id=1,
            username="analyst1",
            display_name="Video Analyst",
            role="analyst",
            created_at=datetime.now().isoformat(),
        )
        self._next_user_id = 2

    @staticmethod
    def _detect_mentions(text: str) -> list[str]:
        return CollaborationService._MENTION_RE.findall(text)

    def create_user(self, username: str, display_name: str, role: str = "analyst") -> str:
        try:
            for u in self._users.values():
                if u.username == username:
                    return json.dumps({"error": "Username already exists"})
            uid = self._next_user_id
            self._next_user_id += 1
            user = CollabUser(
                id=uid,
                username=username,
                display_name=display_name,
                role=role,
                created_at=datetime.now().isoformat(),
            )
            self._users[uid] = user
            self._add_activity(uid, username, "user_created", f"User {username} created")
            self._save_user_to_db(user)
            return json.dumps({"user": user.to_dict(), "ok": True})
        except Exception as e:
            logger.error(f"create_user failed: {e}")
            return json.dumps({"error": str(e)})

    def get_users(self) -> str:
        try:
            return json.dumps(
                {"users": [u.to_dict() for u in self._users.values()], "total": len(self._users)}
            )
        except Exception as e:
            logger.error(f"get_users failed: {e}")
            return json.dumps({"error": str(e)})

    def delete_user(self, user_id: int) -> str:
        try:
            if user_id in self._users and user_id > 1:
                del self._users[user_id]
                return json.dumps({"ok": True})
            return json.dumps({"error": "User not found or cannot delete default users"})
        except Exception as e:
            logger.error(f"delete_user failed: {e}")
            return json.dumps({"error": str(e)})

    def add_comment(self, match_id: int, event_id: int, user_id: int, text: str) -> str:
        try:
            cid = self._next_comment_id
            self._next_comment_id += 1
            username = self._users.get(user_id, CollabUser(username="unknown")).username
            comment = Comment(
                id=cid,
                match_id=match_id,
                event_id=event_id,
                user_id=user_id,
                username=username,
                text=text,
                created_at=datetime.now().isoformat(),
            )
            self._comments.append(comment)
            self._save_comment_to_db(comment)
            self._add_activity(
                user_id, username, "comment", f"Commented on match {match_id}", match_id
            )
            mentioned = self._detect_mentions(text)
            for m_username in mentioned:
                if m_username != username:
                    mid = self._next_mention_id
                    self._next_mention_id += 1
                    mention = Mention(
                        id=mid,
                        username=m_username,
                        from_user=username,
                        text=text,
                        match_id=match_id,
                        event_id=event_id,
                        read=0,
                        created_at=datetime.now().isoformat(),
                    )
                    self._mentions.append(mention)
                    self._save_mention_to_db(mention)
            return json.dumps({"comment": comment.to_dict(), "ok": True})
        except Exception as e:
            logger.error(f"add_comment failed: {e}")
            return json.dumps({"error": str(e)})

    def get_comments(self, match_id: int, event_id: int = 0) -> str:
        try:
            results = [
                c
                for c in self._comments
                if c.match_id == match_id and (event_id == 0 or c.event_id == event_id)
            ]
            return json.dumps({"comments": [c.to_dict() for c in results], "total": len(results)})
        except Exception as e:
            logger.error(f"get_comments failed: {e}")
            return json.dumps({"error": str(e)})

    def get_event_comments(self, match_id: int, event_id: int) -> str:
        try:
            results = [
                c for c in self._comments if c.match_id == match_id and c.event_id == event_id
            ]
            return json.dumps({"comments": [c.to_dict() for c in results], "total": len(results)})
        except Exception as e:
            logger.error(f"get_event_comments failed: {e}")
            return json.dumps({"error": str(e)})

    def get_mentions(self, username: str) -> str:
        try:
            results = [m for m in self._mentions if m.username == username]
            return json.dumps(
                {
                    "mentions": [m.to_dict() for m in results],
                    "total": len(results),
                    "unread": sum(1 for m in results if m.read == 0),
                }
            )
        except Exception as e:
            logger.error(f"get_mentions failed: {e}")
            return json.dumps({"error": str(e)})

    def mark_mention_read(self, mention_id: int) -> str:
        try:
            for m in self._mentions:
                if m.id == mention_id:
                    m.read = 1
                    svc = self._storage_service
                    if svc is not None and svc._conn is not None:
                        try:
                            cursor = svc._conn.cursor()
                            cursor.execute(
                                "UPDATE collab_mentions SET read = 1 WHERE id = ?", (mention_id,)
                            )
                            svc._conn.commit()
                        except Exception as e:
                            logger.warning(f"Could not update mention in DB: {e}")
                    return json.dumps({"ok": True})
            return json.dumps({"error": "Mention not found"})
        except Exception as e:
            logger.error(f"mark_mention_read failed: {e}")
            return json.dumps({"error": str(e)})

    def delete_comment(self, comment_id: int) -> str:
        try:
            self._comments = [c for c in self._comments if c.id != comment_id]
            return json.dumps({"ok": True})
        except Exception as e:
            logger.error(f"delete_comment failed: {e}")
            return json.dumps({"error": str(e)})

    def export_project(self, match_data: dict) -> str:
        try:
            project = {
                "version": "0.13.0",
                "type": "kawkab_project",
                "exported_at": datetime.now().isoformat(),
                "match": match_data,
                "comments": [
                    c.to_dict() for c in self._comments if c.match_id == match_data.get("id")
                ],
            }
            return json.dumps({"project": project, "ok": True})
        except Exception as e:
            logger.error(f"export_project failed: {e}")
            return json.dumps({"error": str(e)})

    def import_project(self, project_json: str) -> str:
        try:
            data = json.loads(project_json) if isinstance(project_json, str) else project_json
            if data.get("type") != "kawkab_project":
                return json.dumps({"error": "Invalid project format"})
            match = data.get("match", {})
            for c_data in data.get("comments", []):
                comment = Comment(
                    id=self._next_comment_id,
                    match_id=match.get("id", 0),
                    event_id=c_data.get("event_id", 0),
                    user_id=0,
                    username=c_data.get("username", "imported"),
                    text=c_data.get("text", ""),
                    created_at=c_data.get("created_at", datetime.now().isoformat()),
                )
                self._next_comment_id += 1
                self._comments.append(comment)
            self._add_activity(
                0,
                "system",
                "import",
                f"Imported project for match {match.get('id', '?')}",
                match.get("id", 0),
            )
            return json.dumps(
                {"match": match, "comments_imported": len(data.get("comments", [])), "ok": True}
            )
        except Exception as e:
            logger.error(f"import_project failed: {e}")
            return json.dumps({"error": str(e)})

    def get_activity_feed(self, limit: int = 50) -> str:
        try:
            feed = sorted(self._activities, key=lambda a: a.created_at, reverse=True)[:limit]
            return json.dumps({"activities": [a.to_dict() for a in feed], "total": len(feed)})
        except Exception as e:
            logger.error(f"get_activity_feed failed: {e}")
            return json.dumps({"error": str(e)})

    def _add_activity(
        self, user_id: int, username: str, action: str, description: str, match_id: int = 0
    ):
        aid = self._next_activity_id
        self._next_activity_id += 1
        entry = ActivityEntry(
            id=aid,
            user_id=user_id,
            username=username,
            action=action,
            description=description,
            match_id=match_id,
            created_at=datetime.now().isoformat(),
        )
        self._activities.append(entry)
