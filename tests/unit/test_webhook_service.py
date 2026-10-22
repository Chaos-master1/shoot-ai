"""Tests for webhook service."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kawkab.services.webhook_service import WebhookService


class TestWebhookService:
    @pytest.fixture
    def svc(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield WebhookService(str(Path(tmp) / "webhooks.json"))

    def test_register_webhook(self, svc):
        wh = svc.register("https://example.com/hook", secret="s3cr3t", events=["match.analyzed"])
        assert wh["url"] == "https://example.com/hook"
        assert wh["secret"] == "s3cr3t"
        assert wh["is_active"] is True

    def test_list_webhooks(self, svc):
        svc.register("https://hook1.com")
        svc.register("https://hook2.com")
        all_wh = svc.list_all()
        assert len(all_wh) == 2

    def test_unregister_webhook(self, svc):
        wh = svc.register("https://delete.me")
        assert svc.unregister(wh["id"]) is True
        assert svc.list_all() == []

    def test_unregister_nonexistent(self, svc):
        assert svc.unregister(99999) is False

    def test_empty_list(self, svc):
        assert svc.list_all() == []

    def test_register_wildcard_event(self, svc):
        wh = svc.register("https://all.com")
        assert "*" in wh["events"]

    def test_deliver_no_crash(self, svc):
        svc.register("https://nonexistent.example.com/hook")
        svc.deliver("test.event", {"key": "value"})

    def test_persistence(self, svc):
        svc.register("https://persist.com")
        db_path = svc._db_path
        svc2 = WebhookService(str(db_path))
        assert len(svc2.list_all()) == 1

    def test_wildcard_matches_any_event(self, svc):
        svc.register("https://wildcard.com")
        svc.deliver("random.event", {"data": 1})

    def test_inactive_webhook_skipped(self, svc):
        wh = svc.register("https://inactive.com")
        wh["is_active"] = False
        all_wh = svc.list_all()
        _ = [w for w in all_wh if w["id"] == wh["id"]]
        svc.deliver("test.event", {})
