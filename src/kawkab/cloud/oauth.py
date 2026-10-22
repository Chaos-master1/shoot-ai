from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass
class OAuthProviderConfig:
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scopes: list[str]


class OAuthProvider:
    def __init__(self, config: OAuthProviderConfig) -> None:
        self.config = config

    def get_authorize_url(self, redirect_uri: str, state: str) -> str:
        scope = " ".join(self.config.scopes)
        return (
            f"{self.config.authorize_url}?"
            f"client_id={self.config.client_id}&"
            f"redirect_uri={redirect_uri}&"
            f"scope={scope}&"
            f"state={state}&"
            f"response_type=code&"
            f"access_type=offline"
        )

    def exchange_code(self, code: str, redirect_uri: str) -> dict | None:
        resp = httpx.post(
            self.config.token_url,
            data={
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        return resp.json()

    def get_userinfo(self, access_token: str) -> dict | None:
        resp = httpx.get(
            self.config.userinfo_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        return resp.json()

    def refresh_token(self, refresh_token: str) -> dict | None:
        resp = httpx.post(
            self.config.token_url,
            data={
                "client_id": self.config.client_id,
                "client_secret": self.config.client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            headers={"Accept": "application/json"},
            timeout=10.0,
        )
        if resp.status_code != 200:
            return None
        return resp.json()


def _env(name: str) -> str:
    val = os.environ.get(name)
    if val is None:
        return ""
    return val


PROVIDERS: dict[str, OAuthProvider] = {}


def _register_provider(name: str, config: OAuthProviderConfig) -> None:
    if config.client_id and config.client_secret:
        PROVIDERS[name] = OAuthProvider(config)


_register_provider(
    "google",
    OAuthProviderConfig(
        client_id=_env("KAWKAB_GOOGLE_CLIENT_ID"),
        client_secret=_env("KAWKAB_GOOGLE_CLIENT_SECRET"),
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo",
        scopes=["openid", "email", "profile"],
    ),
)

_register_provider(
    "github",
    OAuthProviderConfig(
        client_id=_env("KAWKAB_GITHUB_CLIENT_ID"),
        client_secret=_env("KAWKAB_GITHUB_CLIENT_SECRET"),
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        userinfo_url="https://api.github.com/user",
        scopes=["read:user", "user:email"],
    ),
)

_register_provider(
    "apple",
    OAuthProviderConfig(
        client_id=_env("KAWKAB_APPLE_CLIENT_ID"),
        client_secret=_env("KAWKAB_APPLE_CLIENT_SECRET"),
        authorize_url="https://appleid.apple.com/auth/authorize",
        token_url="https://appleid.apple.com/auth/token",
        userinfo_url="https://appleid.apple.com/auth/userinfo",
        scopes=["name", "email"],
    ),
)


def get_oauth_provider(name: str) -> OAuthProvider | None:
    return PROVIDERS.get(name)


def get_configured_providers() -> list[str]:
    return list(PROVIDERS.keys())
