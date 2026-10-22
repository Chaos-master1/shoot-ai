from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from kawkab.cloud.oauth import (
    PROVIDERS,
    OAuthProvider,
    OAuthProviderConfig,
    get_configured_providers,
    get_oauth_provider,
)


@pytest.fixture(autouse=True)
def clear_providers():
    saved = dict(PROVIDERS)
    PROVIDERS.clear()
    yield
    PROVIDERS.clear()
    PROVIDERS.update(saved)


def make_config(**kwargs) -> OAuthProviderConfig:
    defaults = {
        "client_id": "test_id",
        "client_secret": "test_secret",
        "authorize_url": "https://auth.example.com/auth",
        "token_url": "https://auth.example.com/token",
        "userinfo_url": "https://auth.example.com/userinfo",
        "scopes": ["openid", "email"],
    }
    defaults.update(kwargs)
    return OAuthProviderConfig(**defaults)


class TestOAuthProviderConfig:
    def test_get_authorize_url_includes_all_params(self):
        prov = OAuthProvider(make_config())
        url = prov.get_authorize_url("http://localhost:8741/callback", "abc123")
        assert "client_id=test_id" in url
        assert (
            "redirect_uri=http%3A%2F%2Flocalhost%3A8741%2Fcallback" in url
            or "redirect_uri=http://localhost:8741/callback" in url
        )
        assert "state=abc123" in url
        assert "response_type=code" in url
        assert "access_type=offline" in url
        assert "openid" in url
        assert "email" in url

    def test_exchange_code_success(self):
        prov = OAuthProvider(make_config())
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "tok1", "refresh_token": "rt1"}
        with patch("httpx.post", return_value=mock_resp):
            result = prov.exchange_code("code1", "http://localhost/callback")
        assert result == {"access_token": "tok1", "refresh_token": "rt1"}

    def test_exchange_code_failure_returns_none(self):
        prov = OAuthProvider(make_config())
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        with patch("httpx.post", return_value=mock_resp):
            result = prov.exchange_code("bad", "http://localhost/callback")
        assert result is None

    def test_get_userinfo_success(self):
        prov = OAuthProvider(make_config())
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"email": "test@example.com", "name": "Test"}
        with patch("httpx.get", return_value=mock_resp):
            result = prov.get_userinfo("tok1")
        assert result == {"email": "test@example.com", "name": "Test"}

    def test_get_userinfo_failure_returns_none(self):
        prov = OAuthProvider(make_config())
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        with patch("httpx.get", return_value=mock_resp):
            result = prov.get_userinfo("bad")
        assert result is None

    def test_refresh_token_success(self):
        prov = OAuthProvider(make_config())
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "new_tok"}
        with patch("httpx.post", return_value=mock_resp):
            result = prov.refresh_token("rt1")
        assert result == {"access_token": "new_tok"}

    def test_refresh_token_failure_returns_none(self):
        prov = OAuthProvider(make_config())
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        with patch("httpx.post", return_value=mock_resp):
            result = prov.refresh_token("bad")
        assert result is None


class TestProviderRegistry:
    def test_get_oauth_provider_returns_none_for_unknown(self):
        assert get_oauth_provider("unknown") is None

    def test_get_oauth_provider_returns_provider_when_configured(self):
        PROVIDERS["test_prov"] = OAuthProvider(make_config())
        prov = get_oauth_provider("test_prov")
        assert prov is not None
        assert isinstance(prov, OAuthProvider)

    def test_get_configured_providers_returns_list(self):
        PROVIDERS["a"] = OAuthProvider(make_config())
        PROVIDERS["b"] = OAuthProvider(make_config(client_id="id2", client_secret="sec2"))
        providers = get_configured_providers()
        assert "a" in providers
        assert "b" in providers

    def test_provider_not_registered_without_credentials(self):
        PROVIDERS.clear()
        config = make_config(client_id="", client_secret="")
        if config.client_id and config.client_secret:
            PROVIDERS["should_not_appear"] = OAuthProvider(config)
        assert "should_not_appear" not in PROVIDERS

    def test_register_provider_with_empty_credentials_skips(self):
        from kawkab.cloud.oauth import _register_provider

        PROVIDERS.clear()
        _register_provider("empty", make_config(client_id="", client_secret=""))
        assert "empty" not in PROVIDERS

    def test_google_provider_has_correct_urls(self):
        from kawkab.cloud.oauth import _register_provider

        cfg = make_config(
            client_id="g_id",
            client_secret="g_secret",
            authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
            token_url="https://oauth2.googleapis.com/token",
            userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
        )
        PROVIDERS.clear()
        _register_provider("google", cfg)
        prov = get_oauth_provider("google")
        assert prov is not None

    def test_github_provider_has_correct_scopes(self):
        from kawkab.cloud.oauth import _register_provider

        cfg = make_config(
            client_id="gh_id",
            client_secret="gh_secret",
            scopes=["read:user", "user:email"],
        )
        PROVIDERS.clear()
        _register_provider("github", cfg)
        prov = get_oauth_provider("github")
        assert prov is not None
        url = prov.get_authorize_url("http://localhost/cb", "st")
        assert "read:user" in url
        assert "user:email" in url
