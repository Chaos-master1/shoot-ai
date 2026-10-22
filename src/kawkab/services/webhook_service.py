"""Webhook service — register/deliver webhooks for match and analysis events."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

WEBHOOKS_DB_PATH = None


def _get_db():
    global WEBHOOKS_DB_PATH
    if WEBHOOKS_DB_PATH is None:
        WEBHOOKS_DB_PATH = Path.home() / ".kawkab" / "webhooks.json"
    WEBHOOKS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not WEBHOOKS_DB_PATH.exists():
        WEBHOOKS_DB_PATH.write_text("[]", encoding="utf-8")
    return WEBHOOKS_DB_PATH


class WebhookService:
    def __init__(self, db_path: str | None = None):
        global WEBHOOKS_DB_PATH
        if db_path:
            WEBHOOKS_DB_PATH = Path(db_path)
        self._db_path = _get_db()

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self._db_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _save(self, webhooks: list[dict]):
        self._db_path.write_text(json.dumps(webhooks, indent=2), encoding="utf-8")

    def register(self, url: str, secret: str = "", events: list[str] | None = None) -> dict:
        webhooks = self._load()
        wh = {
            "id": int(time.time() * 1000) % (2**31),
            "url": url,
            "secret": secret,
            "events": events or ["*"],
            "is_active": True,
            "created_at": datetime.now(UTC).isoformat(),
        }
        webhooks.append(wh)
        self._save(webhooks)
        return wh

    def unregister(self, webhook_id: int) -> bool:
        webhooks = self._load()
        new_list = [wh for wh in webhooks if wh["id"] != webhook_id]
        if len(new_list) == len(webhooks):
            return False
        self._save(new_list)
        return True

    def list_all(self) -> list[dict]:
        return self._load()

    def deliver(self, event_type: str, payload: dict):
        webhooks = self._load()
        for wh in webhooks:
            if not wh.get("is_active", True):
                continue
            if "*" not in wh.get("events", []) and event_type not in wh.get("events", []):
                continue
            with contextlib.suppress(Exception):
                self._send(wh, event_type, payload)

    def _send(self, webhook: dict, event_type: str, payload: dict):
        body = json.dumps(
            {
                "event": event_type,
                "timestamp": datetime.now(UTC).isoformat(),
                "payload": payload,
            },
            default=str,
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Event": event_type,
        }
        secret = webhook.get("secret", "")
        if secret:
            sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            headers["X-Webhook-Signature"] = sig

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(webhook["url"], content=body, headers=headers)
                resp.raise_for_status()
        except Exception:
            pass
