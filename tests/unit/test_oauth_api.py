from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

os.environ.setdefault("KAWKAB_JWT_SECRET", "test-secret-for-testing-purposes-only")
os.environ["KAWKAB_CLOUD_DB"] = os.path.join(tempfile.gettempdir(), "kawkab_test_oauth_api.db")
os.environ["KAWKAB_RATE_LIMIT_DISABLE"] = "1"

from fastapi.testclient import TestClient

from kawkab.cloud.oauth import PROVIDERS, OAuthProvider, OAuthProviderConfig
from kawkab.cloud.server import app


@pytest.fixture(autouse=True)
def reset_jwt_secret():
    global _jwt_secret
    _jwt_secret = None
    yield


@pytest.fixture(autouse=True)
def setup_db():
    from kawkab.cloud.database import get_cloud_db

    db = get_cloud_db()
    db.execute("DELETE FROM oauth_accounts")
    db.execute("DELETE FROM users")
    db.execute("DELETE FROM refresh_tokens")
    db.commit()
    yield


@pytest.fixture(autouse=True)
def setup_providers():
    PROVIDERS.clear()
    cfg = OAuthProviderConfig(
        client_id="test_id",
        client_secret="test_secret",
        authorize_url="https://auth.example.com/auth",
        token_url="https://auth.example.com/token",
        userinfo_url="https://auth.example.com/userinfo",
        scopes=["openid", "email"],
    )
    PROVIDERS["test_prov"] = OAuthProvider(cfg)
    yield
    PROVIDERS.clear()


client = TestClient(app)


class TestOAuthAPI:
    def test_list_providers(self):
        resp = client.get("/auth/oauth/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert "test_prov" in data["providers"]

    def test_authorize_unknown_provider(self):
        resp = client.get("/auth/oauth/unknown/authorize")
        assert resp.status_code == 400

    def test_authorize_known_provider(self):
        resp = client.get("/auth/oauth/test_prov/authorize?redirect_uri=http://localhost/cb")
        assert resp.status_code == 200
        data = resp.json()
        assert "authorize_url" in data
        assert "state" in data
        assert data["provider"] == "test_prov"
        assert "auth.example.com" in data["authorize_url"]

    def test_callback_invalid_state(self):
        resp = client.post(
            "/auth/oauth/test_prov/callback",
            json={"code": "abc", "state": "bad_state", "provider": "test_prov"},
        )
        assert resp.status_code == 400

    def test_callback_valid_state_exchange_fails(self):
        resp = client.get("/auth/oauth/test_prov/authorize")
        state = resp.json()["state"]
        with patch.object(OAuthProvider, "exchange_code", return_value=None):
            resp2 = client.post(
                "/auth/oauth/test_prov/callback",
                json={"code": "abc", "state": state, "provider": "test_prov"},
            )
        assert resp2.status_code == 400

    def test_callback_valid_state_userinfo_fails(self):
        resp = client.get("/auth/oauth/test_prov/authorize")
        state = resp.json()["state"]
        with (
            patch.object(OAuthProvider, "exchange_code", return_value={"access_token": "tok1"}),
            patch.object(OAuthProvider, "get_userinfo", return_value=None),
        ):
            resp2 = client.post(
                "/auth/oauth/test_prov/callback",
                json={"code": "abc", "state": state, "provider": "test_prov"},
            )
        assert resp2.status_code == 400

    def test_callback_creates_new_user(self):
        resp = client.get("/auth/oauth/test_prov/authorize")
        state = resp.json()["state"]
        with (
            patch.object(
                OAuthProvider,
                "exchange_code",
                return_value={"access_token": "tok1", "refresh_token": "rt1"},
            ),
            patch.object(
                OAuthProvider,
                "get_userinfo",
                return_value={"id": "ext123", "email": "ext@test.com", "name": "External User"},
            ),
        ):
            resp2 = client.post(
                "/auth/oauth/test_prov/callback",
                json={"code": "abc", "state": state, "provider": "test_prov"},
            )
        assert resp2.status_code == 200
        data = resp2.json()
        assert "access_token" in data
        assert data["user"]["email"] == "ext@test.com"
        assert data["user"]["display_name"] == "External User"

    def test_callback_links_existing_user_by_email(self):
        from kawkab.cloud.database import get_cloud_db

        db = get_cloud_db()
        from kawkab.cloud.auth import hash_password

        db.execute(
            "INSERT INTO users (username, email, password_hash, display_name) VALUES (?, ?, ?, ?)",
            ("existing", "existing@test.com", hash_password("testpass123"), "Existing"),
        )
        db.commit()
        resp = client.get("/auth/oauth/test_prov/authorize")
        state = resp.json()["state"]
        with (
            patch.object(OAuthProvider, "exchange_code", return_value={"access_token": "tok1"}),
            patch.object(
                OAuthProvider,
                "get_userinfo",
                return_value={"id": "ext456", "email": "existing@test.com", "name": "Existing"},
            ),
        ):
            resp2 = client.post(
                "/auth/oauth/test_prov/callback",
                json={"code": "abc", "state": state, "provider": "test_prov"},
            )
        assert resp2.status_code == 200
        assert resp2.json()["user"]["email"] == "existing@test.com"

    def test_callback_returns_existing_oauth_user(self):
        from kawkab.cloud.database import get_cloud_db

        db = get_cloud_db()
        from kawkab.cloud.auth import hash_password

        db.execute(
            "INSERT INTO users (username, email, password_hash, display_name) VALUES (?, ?, ?, ?)",
            ("oauthuser", "oauth@test.com", hash_password("testpass123"), "OAuth User"),
        )
        user_id = db.execute(
            "SELECT id FROM users WHERE email = ?", ("oauth@test.com",)
        ).fetchone()["id"]
        db.execute(
            "INSERT INTO oauth_accounts (user_id, provider, provider_user_id, access_token) VALUES (?, ?, ?, ?)",
            (user_id, "test_prov", "ext789", "old_tok"),
        )
        db.commit()
        resp = client.get("/auth/oauth/test_prov/authorize")
        state = resp.json()["state"]
        with (
            patch.object(OAuthProvider, "exchange_code", return_value={"access_token": "new_tok"}),
            patch.object(
                OAuthProvider,
                "get_userinfo",
                return_value={"id": "ext789", "email": "oauth@test.com", "name": "OAuth User"},
            ),
        ):
            resp2 = client.post(
                "/auth/oauth/test_prov/callback",
                json={"code": "abc", "state": state, "provider": "test_prov"},
            )
        assert resp2.status_code == 200
        assert resp2.json()["user"]["email"] == "oauth@test.com"

    def test_link_oauth_endpoint_removed(self):
        """Removed: it trusted a client-asserted provider_user_id (identity
        injection into any account) and had no consumer. The route must
        stay gone — 404, not 401, so a reintroduction fails loudly here."""
        resp = client.post("/auth/link-oauth?provider=test_prov&provider_user_id=ext111")
        assert resp.status_code == 404

    def test_list_oauth_accounts_requires_auth(self):
        resp = client.get("/auth/oauth/accounts")
        assert resp.status_code == 401

    def test_unlink_oauth_requires_auth(self):
        resp = client.post("/auth/oauth/unlink?provider=test_prov")
        assert resp.status_code == 401
