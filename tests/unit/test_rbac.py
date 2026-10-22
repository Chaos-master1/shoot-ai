"""Tests for RBAC middleware."""

from __future__ import annotations

import contextlib

import pytest
from fastapi import HTTPException

from kawkab.core.rbac import (
    PERMISSION_ROLES,
    ROLE_HIERARCHY,
    RBACMiddleware,
    Role,
    get_rbac,
    rbac,
)


class TestRBACMiddleware:
    @pytest.fixture
    def middleware(self):
        m = RBACMiddleware()
        m.register_user(1, "admin", Role.ADMIN)
        m.register_user(2, "coach", Role.COACH)
        m.register_user(3, "analyst", Role.ANALYST)
        m.register_user(4, "scout", Role.SCOUT, team="academy")
        m.register_user(5, "viewer", Role.VIEWER)
        return m

    def test_admin_has_all_permissions(self, middleware):
        for perm in PERMISSION_ROLES:
            assert middleware.has_permission(1, perm), f"Admin should have {perm}"

    def test_viewer_only_read(self, middleware):
        assert middleware.has_permission(5, "match:read") is True
        assert middleware.has_permission(5, "match:write") is False
        assert middleware.has_permission(5, "event:read") is True
        assert middleware.has_permission(5, "event:write") is False

    def test_analyst_can_write_events(self, middleware):
        assert middleware.has_permission(3, "event:write") is True
        assert middleware.has_permission(3, "event:read") is True
        assert middleware.has_permission(3, "match:delete") is False

    def test_coach_can_delete(self, middleware):
        assert middleware.has_permission(2, "match:delete") is True
        assert middleware.has_permission(2, "event:delete") is True

    def test_scout_has_recruitment(self, middleware):
        assert middleware.has_permission(4, "recruitment:read") is True
        assert middleware.has_permission(4, "recruitment:write") is True
        assert middleware.has_permission(4, "analysis:run") is False

    def test_unknown_user_has_no_permissions(self, middleware):
        assert middleware.has_permission(99, "match:read") is False

    def test_permission_not_found(self, middleware):
        assert middleware.has_permission(1, "nonexistent:perm") is False

    def test_resource_team_restriction(self, middleware):
        # Analyst (no team) should access any team's data
        assert middleware.has_permission(3, "match:read", "first_team") is True
        # Scout (team=academy) should not access first_team data
        assert middleware.has_permission(4, "match:read", "first_team") is False

    def test_admin_bypasses_team_restriction(self, middleware):
        assert middleware.has_permission(1, "match:read", "any_team") is True

    def test_require_permission_raises(self, middleware):
        with pytest.raises(HTTPException):
            middleware.require_permission(5, "match:write")

    def test_require_permission_passes(self, middleware):
        middleware.require_permission(1, "match:write")

    def test_get_role(self, middleware):
        assert middleware.get_role(1) == Role.ADMIN
        assert middleware.get_role(5) == Role.VIEWER
        assert middleware.get_role(99) is None

    def test_register_user(self, middleware):
        user = middleware.register_user(10, "new", Role.ANALYST)
        assert user["role"] == Role.ANALYST
        assert user["role_level"] == ROLE_HIERARCHY[Role.ANALYST]

    def test_get_user(self, middleware):
        user = middleware.get_user(1)
        assert user is not None
        assert user["username"] == "admin"

    def test_get_user_nonexistent(self, middleware):
        assert middleware.get_user(999) is None

    def test_role_hierarchy(self):
        assert ROLE_HIERARCHY[Role.ADMIN] > ROLE_HIERARCHY[Role.COACH]
        assert ROLE_HIERARCHY[Role.COACH] > ROLE_HIERARCHY[Role.ANALYST]
        assert ROLE_HIERARCHY[Role.ANALYST] > ROLE_HIERARCHY[Role.SCOUT]
        assert ROLE_HIERARCHY[Role.SCOUT] > ROLE_HIERARCHY[Role.VIEWER]

    def test_permission_role_mapping(self):
        assert PERMISSION_ROLES["match:read"] == Role.VIEWER
        assert PERMISSION_ROLES["admin:users"] == Role.ADMIN


class TestRbacSingleton:
    def test_get_rbac_returns_instance(self):
        assert get_rbac() is rbac
        assert isinstance(rbac, RBACMiddleware)


class TestRequirePermissionDependency:
    """Tests for the FastAPI require_permission dependency factory."""

    @pytest.fixture(autouse=True)
    def setup_env_and_db(self):
        import os
        import tempfile

        self._old_secret = os.environ.get("KAWKAB_JWT_SECRET")
        self._old_db_url = os.environ.get("KAWKAB_DB_URL")
        self._old_cloud_db = os.environ.get("KAWKAB_CLOUD_DB")
        os.environ["KAWKAB_JWT_SECRET"] = "test-secret-for-testing-purposes-only-32chars"
        os.environ.pop("KAWKAB_DB_URL", None)
        db_path = os.path.join(tempfile.gettempdir(), f"kawkab_test_rbac_{id(self)}.db")
        os.environ["KAWKAB_CLOUD_DB"] = db_path
        yield
        os.environ.pop("KAWKAB_DB_URL", None)
        if self._old_secret:
            os.environ["KAWKAB_JWT_SECRET"] = self._old_secret
        if self._old_cloud_db:
            os.environ["KAWKAB_CLOUD_DB"] = self._old_cloud_db
        with contextlib.suppress(OSError):
            os.remove(db_path)

    @pytest.fixture
    def client(self):
        import threading

        from fastapi.testclient import TestClient

        from kawkab.cloud import database
        from kawkab.cloud.server import app

        database._local = threading.local()
        # Reset cached DB reference
        if hasattr(database._local, "conn"):
            with contextlib.suppress(Exception):
                database._local.conn.close()
            del database._local.conn
        return TestClient(app)

    def test_no_credentials_returns_401(self, client):
        resp = client.get("/api/v1/matches")
        assert resp.status_code == 401
        assert "Authentication required" in resp.text

    def test_expired_token_returns_401(self, client):
        # JWT with "exp" in the past
        import time

        import jwt

        token = jwt.encode(
            {"sub": 1, "exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200},
            "test-secret-for-testing-purposes-only-32chars",
            algorithm="HS256",
        )
        resp = client.get("/api/v1/matches", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401

    def test_garbage_token_returns_401(self, client):
        resp = client.get(
            "/api/v1/matches", headers={"Authorization": "Bearer garbage.invalid.token"}
        )
        assert resp.status_code == 401

    def test_anonymous_endpoint_works_without_auth(self, client):
        # Health endpoint has no require_permission — always public
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.fixture
    def registered_user(self, client):
        """Register a user and return a valid access token."""
        import uuid

        suffix = uuid.uuid4().hex[:8]
        resp = client.post(
            "/auth/register",
            json={
                "username": f"rbac_user_{suffix}",
                "email": f"rbac_{suffix}@test.com",
                "password": "TestPass123!",
                "display_name": "RBAC User",
            },
        )
        assert resp.status_code == 200, f"Register failed: {resp.status_code} {resp.text[:200]}"
        data = resp.json()
        yield data["access_token"]
        # Cleanup
        from kawkab.cloud.database import get_cloud_db

        db = get_cloud_db()
        db.execute("DELETE FROM users WHERE email = ?", (f"rbac_{suffix}@test.com",))
        db.commit()

    def test_valid_token_returns_200(self, client, registered_user):
        resp = client.get("/api/v1/matches", headers={"Authorization": f"Bearer {registered_user}"})
        assert resp.status_code == 200

    def test_viewer_cannot_access_admin_endpoint(self, client, registered_user):
        resp = client.get(
            "/api/v1/webhooks",
            headers={"Authorization": f"Bearer {registered_user}"},
        )
        assert resp.status_code == 403

    def test_analyst_cannot_access_medical(self, client, registered_user):
        # Registered user is analyst (level 60), medical requires coach (level 80)
        resp = client.get(
            "/api/v1/players/1/injury-risk",
            headers={"Authorization": f"Bearer {registered_user}"},
        )
        assert resp.status_code == 403
