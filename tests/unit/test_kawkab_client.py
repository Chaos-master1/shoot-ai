"""Tests for the KawkabClient SDK."""

from __future__ import annotations

import pytest

from kawkab.api.client import KawkabClient


class TestKawkabClient:
    @pytest.fixture
    def client(self):
        return KawkabClient(base_url="http://localhost:18741")

    def test_init(self, client):
        assert client.base_url == "http://localhost:18741"

    def test_close(self, client):
        import asyncio

        asyncio.run(client.close())

    def test_get_match_shots_no_server(self, client):
        import asyncio

        import httpx

        with pytest.raises(httpx.HTTPError):
            asyncio.run(client.get_match_shots(1))

    def test_list_matches_no_server(self, client):
        import asyncio

        import httpx

        with pytest.raises(httpx.HTTPError):
            asyncio.run(client.list_matches())
