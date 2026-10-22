"""Kawkab AI Python Client SDK — REST API client for v1 analytics endpoints.

Usage:
    from kawkab.api.client import KawkabClient
    client = KawkabClient(base_url="http://localhost:8741")
    matches = await client.list_matches()
    shots = await client.get_match_shots(match_id=1)
"""

from __future__ import annotations

from typing import Any

import httpx


class KawkabClient:
    """Async HTTP client for the Kawkab AI REST API."""

    def __init__(self, base_url: str = "http://localhost:8741", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30.0)
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    async def close(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = await self._client.get(
            f"{self.base_url}{path}",
            headers=self._headers,
            params=params,
        )
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, body: dict[str, Any] | list[Any] | None = None) -> Any:
        resp = await self._client.post(
            f"{self.base_url}{path}",
            headers=self._headers,
            json=body or {},
        )
        resp.raise_for_status()
        return resp.json()

    # ── Matches ──

    async def list_matches(self) -> list[dict]:
        # The server wraps list endpoints in _paginate(): {"items": [...], ...}.
        result: dict[str, Any] = await self._get("/api/v1/matches")
        items: list[dict] = result.get("items", [])
        return items

    async def get_match(self, match_id: int) -> dict:
        result: dict[str, Any] = await self._get(f"/api/v1/matches/{match_id}")
        return result

    async def get_match_events(
        self, match_id: int, event_type: str | None = None, limit: int = 1000
    ) -> list[dict]:
        params: dict[str, Any] = {"limit": limit}
        if event_type:
            params["event_type"] = event_type
        result: dict[str, Any] = await self._get(
            f"/api/v1/matches/{match_id}/events", params=params
        )
        items: list[dict] = result.get("items", [])
        return items

    async def get_match_players(self, match_id: int) -> list[dict]:
        result: dict[str, Any] = await self._get(f"/api/v1/matches/{match_id}/players")
        items: list[dict] = result.get("items", [])
        return items

    # ── Analysis ──

    async def get_match_shots(self, match_id: int) -> dict:
        result: dict[str, Any] = await self._get(f"/api/v1/matches/{match_id}/analysis/shots")
        return result

    async def get_tactical_shapes(self, match_id: int) -> dict:
        result: dict[str, Any] = await self._get(
            f"/api/v1/matches/{match_id}/analysis/tactical-shapes"
        )
        return result

    async def get_pressing(self, match_id: int) -> dict:
        result: dict[str, Any] = await self._get(f"/api/v1/matches/{match_id}/analysis/pressing")
        return result

    async def get_match_report(self, match_id: int) -> dict:
        result: dict[str, Any] = await self._get(f"/api/v1/matches/{match_id}/analysis/report")
        return result

    # ── AI ──

    async def ask_llm(self, match_id: int, question: str) -> str:
        result: dict[str, Any] = await self._post(
            f"/api/v1/matches/{match_id}/ai/ask", {"match_id": match_id, "question": question}
        )
        answer: str = result.get("answer", "")
        return answer

    # ── Player Ratings ──

    async def get_player_ratings(self, match_id: int) -> list[dict]:
        result: list[dict] = await self._get(f"/api/v1/matches/{match_id}/ratings")
        return result

    # ── Calibration ──

    async def get_calibration(self, match_id: int) -> dict:
        result: dict[str, Any] = await self._get(f"/api/v1/matches/{match_id}/calibration")
        return result

    # ── Model Comparison ──

    async def compare_models(self, shots: list[dict], n_folds: int = 5) -> dict:
        result: dict[str, Any] = await self._post(
            f"/api/v1/model-comparison?n_folds={n_folds}", shots
        )
        return result

    # ── Fitness ──

    async def get_player_fitness(self, track_id: int, match_id: int) -> dict:
        result: dict[str, Any] = await self._get(
            f"/api/v1/players/{track_id}/fitness", params={"match_id": match_id}
        )
        return result

    # ── Recruitment ──

    async def search_players(
        self, position: str = "", min_age: int = 16, max_age: int = 40, **kwargs
    ) -> list[dict]:
        body = {"position": position, "min_age": min_age, "max_age": max_age, **kwargs}
        result: dict[str, Any] = await self._post("/api/v1/recruitment/search", body)
        items: list[dict] = result.get("results", [])
        return items

    async def estimate_transfer_fee(self, player_name: str) -> dict:
        result: dict[str, Any] = await self._get(f"/api/v1/recruitment/transfer-fee/{player_name}")
        return result

    async def get_shortlist(self) -> list[dict]:
        result: list[dict] = await self._get("/api/v1/recruitment/shortlist")
        return result

    # ── Game Plan ──

    async def get_game_plan(self, match_id: int, opponent: str) -> dict:
        result: dict[str, Any] = await self._get(f"/api/v1/game-plan/{match_id}/vs/{opponent}")
        return result

    # ── Monitoring ──

    async def get_monitoring_dashboard(self) -> dict:
        result: dict[str, Any] = await self._get("/api/v1/monitoring/dashboard")
        return result

    async def get_drift_alerts(self) -> list[dict]:
        result: dict[str, Any] = await self._get("/api/v1/monitoring/drift")
        items: list[dict] = result.get("alerts", [])
        return items

    # ── Webhooks ──

    async def create_webhook(
        self, url: str, events: list[str] | None = None, secret: str = ""
    ) -> dict:
        result: dict[str, Any] = await self._post(
            "/api/v1/webhooks", {"url": url, "events": events or ["*"], "secret": secret}
        )
        return result

    async def list_webhooks(self) -> list[dict]:
        result: list[dict] = await self._get("/api/v1/webhooks")
        return result

    async def delete_webhook(self, webhook_id: int) -> None:
        await self._client.delete(
            f"{self.base_url}/api/v1/webhooks/{webhook_id}", headers=self._headers
        )
