"""Regression pins for the v0.13.2 slot-audit fixes.

Every test here drives the production-shaped surface (async storage, real
row shapes, real payload keys) that the earlier sync-stub tests could not
see -- the exact blindness that shipped five dead features in v0.13.1.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.ui.bridge_handlers.bridge_analysis import AnalysisHandler  # noqa: E402
from kawkab.ui.bridge_handlers.bridge_coding import CodingHandler  # noqa: E402
from kawkab.ui.bridge_handlers.bridge_domain import DomainHandler  # noqa: E402
from kawkab.ui.bridge_handlers.bridge_match_intel import MatchIntelHandler  # noqa: E402


def _real_row_shaped_events() -> list[dict]:
    """Rows exactly as StorageService.get_match_events returns them:
    event_type (not type), from_track_id, metadata-extracted x/y (None
    when absent), confidence floats, user_corrected ints."""
    return [
        {
            "id": 1,
            "event_type": "pass",
            "timestamp": 1.0,
            "team": "home",
            "from_track_id": 5,
            "to_track_id": 6,
            "completed": 1,
            "confidence": 0.9,
            "x": 30.0,
            "y": 34.0,
            "metadata": "{}",
        },
        {
            "id": 2,
            "event_type": "pass",
            "timestamp": 2.0,
            "team": "home",
            "from_track_id": 5,
            "to_track_id": 7,
            "completed": 0,
            "confidence": 0.4,
            "x": 50.0,
            "y": 35.0,
            "metadata": "{}",
        },
        {
            "id": 3,
            "event_type": "shot",
            "timestamp": 5.0,
            "team": "home",
            "from_track_id": 5,
            "completed": 1,
            "confidence": 0.8,
            "x": 95.0,
            "y": 34.0,
            "metadata": '{"xg": 0.25}',
        },
        {
            "id": 4,
            "event_type": "tackle",
            "timestamp": 6.0,
            "team": "away",
            "from_track_id": 8,
            "completed": 1,
            "confidence": 0.7,
            "x": None,
            "y": None,
            "metadata": "{}",
        },
    ]


def _handler():
    storage = MagicMock(
        get_match_events=AsyncMock(return_value=_real_row_shaped_events()),
        get_match_players=AsyncMock(
            return_value=[
                {"id": 1, "track_id": 5, "name": "P5", "team": "Reds", "jersey_number": 5},
            ]
        ),
    )
    return MatchIntelHandler(None, {"storage_service": storage}, rate_limiter=None), storage


MATCH_INTEL_ASYNC_SLOTS = [
    "analyze_build_up",
    "analyze_finishing",
    "compute_goals_added",
    "compute_phase_xg",
    "compute_territory_value",
    "estimate_transfer_fee",
    "generate_game_plan",
    "generate_match_report",
    "get_dominance_index",
    "get_match_quality_score",
    "get_pitch_control_overlay",
    "get_player_pass_sonar",
    "get_player_role",
    "get_space_control_heatmap",
    "simulate_league",
    "get_xa_report",
    "get_pressing_report",
]


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.mark.parametrize("slot", MATCH_INTEL_ASYNC_SLOTS)
def test_match_intel_slot_awaits_and_answers(slot):
    h, storage = _handler()
    args = {
        "estimate_transfer_fee": ("1", "5"),
        "get_player_pass_sonar": ("1", "5"),
        "get_player_role": ("1", "5"),
        "generate_game_plan": ("1", "1"),
        "simulate_league": ("1", "3"),
    }.get(slot, ("1",))

    async def main():
        out = getattr(h, slot)(*args)
        if inspect.isawaitable(out):
            out = await out
        return json.loads(out)

    d = _run(main())
    assert "error" not in d, f"{slot} failed on real-shaped rows: {d}"


def test_pass_sonar_attributes_passes_via_from_track_id():
    """v0.13.2 pin: pass_sonars read player_id/track_id keys that real
    storage rows never carry (they use from_track_id) and start_x/y keys
    real rows do not have (they use metadata-extracted x/y) -- the sonar
    was empty for every player through the app."""
    h, _ = _handler()

    async def main():
        return json.loads(await h.get_player_pass_sonar("1", "5"))

    d = _run(main())
    assert "error" not in d, d
    assert d["total_passes"] == 2, f"real passes not attributed: {d}"
    assert any(c > 0 for c in d["pass_counts"]), d


def test_save_coding_tag_accepts_timestamp_alias_and_persists(tmp_path, monkeypatch):
    """v0.13.2 pin: sqlite save_coding_tag demanded video_time and silently
    dropped tags using the timestamp key (which the Postgres adapter
    accepts) while the handler reported success. Round trip must work and
    persist."""
    import kawkab.core.paths as paths_mod

    monkeypatch.setenv("XDG_DOCUMENTS_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("HOME", str(tmp_path))
    if hasattr(paths_mod, "_paths"):
        monkeypatch.setattr(paths_mod, "_paths", None)

    from kawkab.services.storage_service import StorageService

    async def main():
        storage = StorageService()
        await storage.initialize()
        try:
            # coding_tags.match_id carries an FK to matches(id) (migration
            # 018), and save_coding_tag swallows FK violations to 0 — so the
            # tag MUST reference a match created here, never leftover state
            # from a shared test database.
            match_id = await storage.save_match("tag-persistence-test", "test.mp4")
            assert match_id > 0, "match row for FK not created"
            h = CodingHandler(None, {"storage_service": storage}, rate_limiter=None)
            out = json.loads(
                await h.save_tag(
                    str(match_id),
                    json.dumps(
                        {
                            "event_type": "shot",
                            "timestamp": 5.0,
                            "team": "home",
                            "player_track_id": 5,
                        }
                    ),
                )
            )
            assert out.get("success") is True and out.get("tag_id", 0) > 0, out
            tags = json.loads(await h.get_tags(str(match_id)))
            assert tags["success"] is True, tags
            saved = [t for t in tags["tags"] if t["id"] == out["tag_id"]]
            assert saved, f"tag {out['tag_id']} not persisted: {tags}"
            assert saved[0]["video_time"] == 5.0, saved
        finally:
            await storage.close()

    _run(main())


def test_save_tag_reports_rejection_honestly():
    """A tag missing any type key must come back as an error payload, not
    {"success": true, "tag_id": 0}."""
    storage = MagicMock(save_coding_tag=AsyncMock(return_value=0))
    h = CodingHandler(None, {"storage_service": storage}, rate_limiter=None)

    async def main():
        return json.loads(await h.save_tag("1", json.dumps({"notes": "no type"})))

    d = _run(main())
    assert "error" in d, d


def test_goalkeeper_analytics_slots_never_attributeerror():
    """v0.13.2 pin: the goalkeeper_analytics property read its own
    not-yet-created cache attribute, so all three compute_goalkeeper_*
    slots crashed with AttributeError on every call."""
    dm = DomainHandler(None, {}, rate_limiter=None)

    async def main():
        results = {}
        for name, payload in (
            ("compute_goalkeeper_aerial_command", [{"minute": 10, "outcome": "caught"}]),
            ("compute_goalkeeper_distribution", [{"minute": 15, "distribution_type": "throw"}]),
            ("compute_goalkeeper_save_quality", [{"minute": 20, "xg": 0.3, "saved": 1}]),
        ):
            out = getattr(dm, name)(json.dumps(payload))
            if inspect.isawaitable(out):
                out = await out
            results[name] = json.loads(out)
        return results

    results = _run(main())
    for name, d in results.items():
        assert "error" not in d, f"{name}: {d}"


def test_knowledge_base_stats_returns_error_payload_not_exception():
    """v0.13.2 pin: get_knowledge_base_stats was the only analysis slot
    that let exceptions escape un-JSON'd (undefined reaches the UI)."""
    kb = MagicMock()
    kb.initialize = AsyncMock(side_effect=RuntimeError("kb down"))
    h = AnalysisHandler(None, {"knowledge_service": kb}, rate_limiter=None)

    async def main():
        return json.loads(await h.get_knowledge_base_stats())

    d = _run(main())
    assert "error" in d and "kb down" in d["error"], d


def test_goalkeeper_analytics_property_caches_after_first_build():
    dm = DomainHandler(None, {}, rate_limiter=None)
    first = dm.goalkeeper_analytics
    assert first is dm.goalkeeper_analytics, "lazy cache must be stable"


def test_rate_limiter_users_call_acquire_not_check():
    """v0.13.2 pin: three handlers (domain, match_intel, physical) called
    the nonexistent RateLimiter.check(), so any slot that rate-checked
    crashed with AttributeError once the real bridge limiter was wired in
    (unit tests passed because they passed rate_limiter=None)."""
    from kawkab.core.security import RateLimiter
    from kawkab.ui.bridge_handlers.bridge_match_intel import MatchIntelHandler
    from kawkab.ui.bridge_handlers.bridge_physical import PhysicalHandler

    for handler_cls in (DomainHandler, MatchIntelHandler, PhysicalHandler):
        # Fresh limiter per class: in the real bridge all handlers share one
        # limiter, so repeated calls on a shared instance exhaust the bucket
        # (which is the correct behavior, just not what this pin tests).
        h = handler_cls(None, {}, rate_limiter=RateLimiter())
        # First call acquires a token; a second call must not raise
        # AttributeError (the .check() regression).
        h._check_rate_limit()
        h._check_rate_limit()

    # And rate limiting itself still bites when the bucket is empty.
    strict = RateLimiter()
    strict.configure("analysis", 0)  # zero-capacity bucket
    h = MatchIntelHandler(None, {}, rate_limiter=strict)
    with pytest.raises(RuntimeError, match="Rate limit"):
        h._check_rate_limit()


def test_dominance_index_tolerates_none_coordinates():
    """v0.13.2 pin: json_extract emits None (not a missing key) for rows
    without metadata coordinates, so end_x/start_x default comparisons
    raised TypeError through get_dominance_index on real storage rows."""
    from kawkab.core.dominance_index import compute_dominance_index

    events = [
        {"type": "pass", "team": "home", "end_x": 90.0, "start_x": 10.0},
        {"type": "shot", "team": "home", "start_x": 95.0, "end_x": None},
        {"type": "pass", "team": "away", "end_x": 90.0, "start_x": None},
        {"type": "tackle", "team": "away", "end_x": None, "start_x": None},
    ]
    report = compute_dominance_index(events, "home")
    assert 0.0 <= report.index <= 100.0
    # 2 of 3 final-third-attacking events belong to home (the away tackle is
    # not a final-third-relevant type and its None coords must not raise).
    assert report.sub_scores["territory"] == 66.7
