from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import httpx

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

CLOUD_URL = os.environ.get("KAWKAB_CLOUD_URL", "http://localhost:8741")
TOKEN_FILE = Path.home() / ".kawkab" / "cloud_token"


class CloudSyncService:
    def __init__(self, cloud_url: str = CLOUD_URL):
        self.cloud_url = cloud_url
        self._token: str | None = None
        self._user: dict | None = None
        self._load_token()

    def _load_token(self):
        try:
            if TOKEN_FILE.exists():
                self._token = TOKEN_FILE.read_text().strip()
        except Exception:
            pass

    def _save_token(self, token: str):
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(token)

    def _headers(self) -> dict:
        if self._token:
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    # ── Auth ──

    def register(self, username: str, email: str, password: str, display_name: str = "") -> str:
        try:
            resp = httpx.post(
                f"{self.cloud_url}/auth/register",
                json={
                    "username": username,
                    "email": email,
                    "password": password,
                    "display_name": display_name,
                },
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._token = data["access_token"]
                self._user = data["user"]
                self._save_token(self._token)
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    def login(self, email: str, password: str) -> str:
        try:
            resp = httpx.post(
                f"{self.cloud_url}/auth/login",
                json={"email": email, "password": password},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._token = data["access_token"]
                self._user = data["user"]
                self._save_token(self._token)
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    def logout(self) -> str:
        self._token = None
        self._user = None
        with contextlib.suppress(Exception):
            TOKEN_FILE.unlink(missing_ok=True)
        return json.dumps({"ok": True})

    def get_me(self) -> str:
        if not self._token:
            return json.dumps({"error": "Not logged in"})
        try:
            resp = httpx.get(f"{self.cloud_url}/auth/me", headers=self._headers(), timeout=10.0)
            if resp.status_code == 200:
                self._user = resp.json()
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    def is_logged_in(self) -> str:
        return json.dumps({"logged_in": self._token is not None, "user": self._user})

    # ── OAuth ──

    def oauth_providers(self) -> str:
        try:
            resp = httpx.get(f"{self.cloud_url}/auth/oauth/providers", timeout=5.0)
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e), "providers": []})

    def oauth_authorize_url(self, provider: str, redirect_uri: str = "") -> str:
        try:
            params = f"?redirect_uri={redirect_uri}" if redirect_uri else ""
            resp = httpx.get(
                f"{self.cloud_url}/auth/oauth/{provider}/authorize{params}", timeout=10.0
            )
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    def oauth_exchange(self, provider: str, code: str, state: str) -> str:
        try:
            resp = httpx.post(
                f"{self.cloud_url}/auth/oauth/{provider}/callback",
                json={"code": code, "state": state, "provider": provider},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._token = data["access_token"]
                self._user = data["user"]
                self._save_token(self._token)
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Sync ──

    def sync_push(self, device_id: str, operations: list) -> str:
        if not self._token:
            return json.dumps({"error": "Not logged in"})
        try:
            resp = httpx.post(
                f"{self.cloud_url}/sync/push",
                headers=self._headers(),
                json={"device_id": device_id, "operations": operations},
                timeout=30.0,
            )
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    def sync_pull(self, device_id: str) -> str:
        if not self._token:
            return json.dumps({"error": "Not logged in"})
        try:
            resp = httpx.post(
                f"{self.cloud_url}/sync/pull",
                headers=self._headers(),
                json={"device_id": device_id, "operations": []},
                timeout=30.0,
            )
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Teams ──

    def create_team(self, name: str, description: str = "") -> str:
        if not self._token:
            return json.dumps({"error": "Not logged in"})
        try:
            resp = httpx.post(
                f"{self.cloud_url}/teams",
                headers=self._headers(),
                json={"name": name, "description": description},
                timeout=10.0,
            )
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    def list_teams(self) -> str:
        if not self._token:
            return json.dumps({"error": "Not logged in"})
        try:
            resp = httpx.get(f"{self.cloud_url}/teams", headers=self._headers(), timeout=10.0)
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    def invite_member(self, team_id: int, email: str) -> str:
        if not self._token:
            return json.dumps({"error": "Not logged in"})
        try:
            resp = httpx.post(
                f"{self.cloud_url}/teams/{team_id}/invite",
                headers=self._headers(),
                json={"email": email, "role": "member"},
                timeout=10.0,
            )
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    def accept_invite(self, token: str) -> str:
        if not self._token:
            return json.dumps({"error": "Not logged in"})
        try:
            resp = httpx.post(
                f"{self.cloud_url}/teams/join/{token}", headers=self._headers(), timeout=10.0
            )
            return resp.text
        except Exception as e:
            return json.dumps({"error": str(e)})

    def check_health(self) -> str:
        try:
            resp = httpx.get(f"{self.cloud_url}/health", timeout=5.0)
            return resp.text
        except Exception as e:
            return json.dumps({"status": "offline", "error": str(e)})
