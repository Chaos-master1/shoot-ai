"""Tests for POST /auth/login on the cloud server.

Regression coverage for a real P0 bug: on the default SQLite backend,
`login()` called `row.get("role", "analyst")` directly on a `sqlite3.Row`,
which has no `.get()` method (only `row["role"]` subscript access) --
so every login attempt raised AttributeError -> HTTP 500. No test
anywhere exercised this route before, so CI could never have caught it.
See cloud/server.py::login and CLAUDE.md's Tier 0 launch-blocker notes.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import uuid

import pytest


class TestAuthLogin:
    """Exercises /auth/login against the default (SQLite) cloud backend."""

    @pytest.fixture(autouse=True)
    def setup_env_and_db(self):
        self._old_secret = os.environ.get("KAWKAB_JWT_SECRET")
        self._old_db_url = os.environ.get("KAWKAB_DB_URL")
        self._old_cloud_db = os.environ.get("KAWKAB_CLOUD_DB")
        os.environ["KAWKAB_JWT_SECRET"] = "test-secret-for-testing-purposes-only-32chars"
        os.environ.pop("KAWKAB_DB_URL", None)
        db_path = os.path.join(tempfile.gettempdir(), f"kawkab_test_login_{id(self)}.db")
        os.environ["KAWKAB_CLOUD_DB"] = db_path
        yield
        os.environ.pop("KAWKAB_DB_URL", None)
        if self._old_secret:
            os.environ["KAWKAB_JWT_SECRET"] = self._old_secret
        else:
            os.environ.pop("KAWKAB_JWT_SECRET", None)
        if self._old_cloud_db:
            os.environ["KAWKAB_CLOUD_DB"] = self._old_cloud_db
        else:
            os.environ.pop("KAWKAB_CLOUD_DB", None)
        if self._old_db_url:
            os.environ["KAWKAB_DB_URL"] = self._old_db_url
        with contextlib.suppress(OSError):
            os.remove(db_path)

    @pytest.fixture
    def client(self):
        import threading

        from fastapi.testclient import TestClient

        from kawkab.cloud import database
        from kawkab.cloud.server import app

        database._local = threading.local()
        return TestClient(app)

    @pytest.fixture
    def registered_user(self, client):
        """Register a fresh user and return (email, password)."""
        suffix = uuid.uuid4().hex[:8]
        email = f"login_{suffix}@test.com"
        password = "TestPass123!"
        resp = client.post(
            "/auth/register",
            json={
                "username": f"login_user_{suffix}",
                "email": email,
                "password": password,
                "display_name": "Login Test User",
            },
        )
        assert resp.status_code == 200, f"Register failed: {resp.status_code} {resp.text[:200]}"
        yield email, password
        from kawkab.cloud.database import get_cloud_db

        db = get_cloud_db()
        db.execute("DELETE FROM users WHERE email = ?", (email,))
        db.commit()

    def test_login_with_correct_credentials_succeeds(self, client, registered_user):
        """The bug this guards: this used to be a 500 on every call on SQLite."""
        email, password = registered_user
        resp = client.post("/auth/login", json={"email": email, "password": password})
        assert resp.status_code == 200, f"Login failed: {resp.status_code} {resp.text[:300]}"
        data = resp.json()
        assert data["access_token"]
        assert data["user"]["email"] == email
        assert data["user"]["role"] == "analyst"
        assert "password_hash" not in data["user"]

    def test_login_token_is_usable(self, client, registered_user):
        """The token issued by login() must actually authenticate later requests --
        catches a regression where role/id were read from the wrong place."""
        email, password = registered_user
        login_resp = client.post("/auth/login", json={"email": email, "password": password})
        assert login_resp.status_code == 200
        token = login_resp.json()["access_token"]

        me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp.status_code == 200
        assert me_resp.json()["email"] == email

    def test_login_with_wrong_password_returns_401(self, client, registered_user):
        email, _password = registered_user
        resp = client.post("/auth/login", json={"email": email, "password": "wrong-password"})
        assert resp.status_code == 401

    def test_login_with_unknown_email_returns_401(self, client):
        resp = client.post(
            "/auth/login", json={"email": "nobody-here@test.com", "password": "whatever123"}
        )
        assert resp.status_code == 401

    def test_login_never_returns_500(self, client, registered_user):
        """Directly names the regression: no login path -- success or failure --
        should ever surface an unhandled server error."""
        email, password = registered_user
        for creds in (
            {"email": email, "password": password},
            {"email": email, "password": "wrong"},
            {"email": "nobody@test.com", "password": "whatever123"},
        ):
            resp = client.post("/auth/login", json=creds)
            assert resp.status_code != 500, (
                f"/auth/login returned 500 for {creds['email']!r}: {resp.text[:300]}"
            )
