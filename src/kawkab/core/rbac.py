"""Role-based access control — permissions, roles, and authorization middleware."""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    ADMIN = "admin"
    COACH = "coach"
    ANALYST = "analyst"
    SCOUT = "scout"
    VIEWER = "viewer"


ROLE_HIERARCHY: dict[Role, int] = {
    Role.ADMIN: 100,
    Role.COACH: 80,
    Role.ANALYST: 60,
    Role.SCOUT: 50,
    Role.VIEWER: 30,
}

PERMISSION_ROLES: dict[str, Role] = {
    "match:read": Role.VIEWER,
    "match:write": Role.ANALYST,
    "match:delete": Role.COACH,
    "event:read": Role.VIEWER,
    "event:write": Role.ANALYST,
    "event:delete": Role.COACH,
    "player:read": Role.VIEWER,
    "player:write": Role.ANALYST,
    "analysis:read": Role.VIEWER,
    "analysis:run": Role.ANALYST,
    "tag:read": Role.VIEWER,
    "tag:write": Role.ANALYST,
    "tag:delete": Role.COACH,
    "admin:users": Role.ADMIN,
    "admin:settings": Role.ADMIN,
    "recruitment:read": Role.SCOUT,
    "recruitment:write": Role.SCOUT,
    "medical:read": Role.COACH,
    "medical:write": Role.COACH,
    "export:data": Role.ANALYST,
    "export:video": Role.COACH,
}


class RBACMiddleware:
    def __init__(self):
        self._users: dict[int, dict] = {}

    def register_user(self, user_id: int, username: str, role: Role, team: str = "") -> dict:
        user = {
            "id": user_id,
            "username": username,
            "role": role,
            "role_level": ROLE_HIERARCHY.get(role, 0),
            "team": team,
        }
        self._users[user_id] = user
        return user

    def get_user(self, user_id: int) -> dict | None:
        return self._users.get(user_id)

    def has_permission(self, user_id: int, permission: str, resource_team: str = "") -> bool:
        user = self._users.get(user_id)
        if not user:
            return False
        required_role = PERMISSION_ROLES.get(permission)
        if required_role is None:
            return False
        if user["role_level"] >= ROLE_HIERARCHY.get(required_role, 0):
            return not (
                resource_team
                and user["team"]
                and user["team"] != resource_team
                and user["role_level"] < ROLE_HIERARCHY[Role.ADMIN]
            )
        return False

    def require_permission(self, user_id: int, permission: str, resource_team: str = ""):
        if not self.has_permission(user_id, permission, resource_team):
            from fastapi import HTTPException

            raise HTTPException(status_code=403, detail=f"Missing permission: {permission}")

    def get_role(self, user_id: int) -> Role | None:
        user = self._users.get(user_id)
        return user["role"] if user else None


rbac = RBACMiddleware()


def get_rbac() -> RBACMiddleware:
    return rbac


def require_permission(permission: str, resource_team: str = "", allow_anonymous: bool = False):
    """FastAPI dependency factory — checks current user has required permission.

    By default (allow_anonymous=False), missing or invalid credentials return 401.
    Set allow_anonymous=True for endpoints that should work without auth
    (e.g. health checks, local desktop mode)."""
    from fastapi import Depends, HTTPException
    from fastapi import status as http_status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    from kawkab.cloud.auth import decode_token

    _bearer = HTTPBearer(auto_error=False)

    async def _checker(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> dict:
        current_user = None
        if credentials:
            payload = decode_token(credentials.credentials)
            if payload:
                db = None
                try:
                    from kawkab.cloud.database import get_cloud_db

                    db = get_cloud_db()
                    row = db.execute(
                        "SELECT id, username, email, display_name, role, is_active, token_version, created_at FROM users WHERE id = ?",
                        (int(payload["sub"]),),
                    ).fetchone()
                    # This dependency has its own, independent auth check
                    # rather than calling cloud.auth.get_current_user --
                    # it must therefore also independently check "tv"
                    # (token_version), or every /api/v1 route (all 39 of
                    # them use require_permission()) would keep accepting
                    # a token after it was supposed to be revoked (e.g. by
                    # a password change), even though cloud.auth's own
                    # get_current_user correctly rejects it.
                    if row and row["is_active"] and payload.get("tv", 0) == row["token_version"]:
                        current_user = dict(row)
                except Exception:
                    pass

        if current_user is None:
            if not allow_anonymous:
                raise HTTPException(
                    status_code=http_status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            current_user = {
                "id": 0,
                "username": "anonymous",
                "role": "viewer",
                "role_level": 30,
                "email": "",
                "team": "",
            }

        user_role = current_user.get("role", "viewer")
        try:
            role_enum = Role(user_role)
        except ValueError:
            role_enum = Role.VIEWER
        user_level = ROLE_HIERARCHY.get(role_enum, 0)
        required_role = PERMISSION_ROLES.get(permission)
        if required_role is None:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail=f"Unknown permission: {permission}",
            )
        required_level = ROLE_HIERARCHY.get(required_role, 0)
        if user_level < required_level:
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail=f"Missing permission: {permission} (need {required_role.value}, have {user_role})",
            )
        if resource_team:
            user_team = current_user.get("team", "")
            if user_team and user_team != resource_team and user_level < ROLE_HIERARCHY[Role.ADMIN]:
                raise HTTPException(
                    status_code=http_status.HTTP_403_FORBIDDEN,
                    detail=f"Team mismatch: cannot access resource of team '{resource_team}'",
                )
        return current_user

    return _checker
