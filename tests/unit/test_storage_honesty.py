"""Fault-injection tests for the storage honesty contract (A1 of the elite plan).

These tests FORCE database failures and assert the failure surfaces as a
typed exception — never a silent 0/[]/False. They exist because storage
swallowed errors for months; the proven damage:

- save_coding_tag's FK violation surfaced to the UI as "Tag rejected:
  event_type/tag_type required" (CI, 2026-09-19) — the user was blamed
  for a database problem.
- The Postgres adapter's no-pool fallback fed [] to every read, so a
  cloud deployment once returned empty data for all calls.

Contract under test (coding-tags cluster is the first convert):
- Not-initialized  -> StorageNotInitialized
- DB failure       -> StorageWriteError / StorageReadError
- Rejected input   -> legacy falsy return (distinguishable from failure)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from conftest import install_kawkab_stubs

install_kawkab_stubs()

from kawkab.core.migration_manager import MigrationManager  # noqa: E402
from kawkab.core.paths import get_paths  # noqa: E402
from kawkab.services.postgres_storage import PostgresStorageAdapter  # noqa: E402
from kawkab.services.storage_errors import (  # noqa: E402
    StorageDuplicateError,
    StorageNotInitializedError,
    StorageReadError,
    StorageWriteError,
    is_duplicate_violation,
)
from kawkab.services.storage_service import StorageService  # noqa: E402

# ── helpers ──────────────────────────────────────────────────────────────


def _migrated_sqlite_storage() -> StorageService:
    """A real sqlite StorageService with the full schema applied."""
    svc = StorageService()
    svc._use_postgres = False
    svc._pg = None
    svc._db_path = Path(tempfile.mkdtemp(prefix="kawkab_honesty_")) / "honesty.db"
    svc._conn = sqlite3.connect(str(svc._db_path))
    svc._conn.row_factory = sqlite3.Row
    # Mirror production initialize(): FK enforcement is what makes the
    # coding_tags->matches constraint fire instead of silently passing.
    svc._conn.execute("PRAGMA foreign_keys=ON")
    MigrationManager(svc._db_path, get_paths().migrations).migrate()
    return svc


# ── sqlite: closed connection (a real user flow) ─────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op,args",
    [
        ("save_coding_tag", (1, {"event_type": "shot"})),
        ("get_coding_tags", (1,)),
        ("get_coding_tags_by_type", (1, "shot")),
        ("get_coding_tags_by_player", (1, 5)),
        ("update_coding_tag", (1, {"notes": "x"})),
        ("delete_coding_tag", (1,)),
        ("hard_delete_coding_tag", (1,)),
        ("restore_coding_tag", (1,)),
        ("get_coding_tag_stats", (1,)),
    ],
)
async def test_closed_connection_raises_not_initialized(op, args):
    svc = _migrated_sqlite_storage()
    svc._conn.close()
    svc._conn = None
    with pytest.raises(StorageNotInitializedError):
        await getattr(svc, op)(*args)


# ── sqlite: live connection, forced DB failure ───────────────────────────


@pytest.mark.asyncio
async def test_fk_violation_raises_write_error_not_rejection():
    """THE regression pin: a missing match row (FK violation) must raise
    StorageWriteError, not return 0 that the UI renders as 'Tag rejected:
    event_type/tag_type required'."""
    svc = _migrated_sqlite_storage()
    try:
        with pytest.raises(StorageWriteError) as excinfo:
            await svc.save_coding_tag(424242, {"event_type": "shot", "video_time": 1.0})
        assert "save_coding_tag" in str(excinfo.value)
        # The typed error chain keeps the real cause for logs.
        assert excinfo.value.__cause__ is not None
    finally:
        svc._conn.close()


@pytest.mark.asyncio
async def test_read_failure_raises_read_error():
    svc = _migrated_sqlite_storage()
    try:
        svc._conn.execute("DROP TABLE coding_tags")
        svc._conn.commit()
        with pytest.raises(StorageReadError):
            await svc.get_coding_tags(1)
    finally:
        svc._conn.close()


@pytest.mark.asyncio
async def test_write_failure_raises_write_error_on_update():
    svc = _migrated_sqlite_storage()
    try:
        svc._conn.execute("DROP TABLE coding_tags")
        svc._conn.commit()
        with pytest.raises(StorageWriteError):
            await svc.update_coding_tag(1, {"notes": "x"})
    finally:
        svc._conn.close()


# ── rejected input stays a falsy return, distinguishable from failure ────


@pytest.mark.asyncio
async def test_rejected_input_still_returns_falsy():
    svc = _migrated_sqlite_storage()
    try:
        assert await svc.save_coding_tag(1, {"notes": "no type key"}) == 0
        assert await svc.update_coding_tag(1, {"not_a_field": 1}) is False
    finally:
        svc._conn.close()


# ── Postgres adapter: pool down ──────────────────────────────────────────


def _pg_without_pool() -> PostgresStorageAdapter:
    a = PostgresStorageAdapter(dsn="postgresql://user:pass@localhost:5432/kawkab_test")
    a._pool = None
    a._available = False
    return a


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op,args",
    [
        ("save_coding_tag", (1, {"event_type": "shot"})),
        ("get_coding_tags", (1,)),
        ("get_coding_tags_by_type", (1, "shot")),
        ("get_coding_tags_by_player", (1, 5)),
        ("update_coding_tag", (1, {"notes": "x"})),
        ("delete_coding_tag", (1,)),
        ("hard_delete_coding_tag", (1,)),
        ("restore_coding_tag", (1,)),
        ("get_coding_tag_stats", (1,)),
    ],
)
async def test_pg_pool_down_raises_not_initialized(op, args):
    """Pin: pool-down must RAISE for the converted cluster. The old silent
    [] fallback is the exact mechanism that made a Postgres deployment
    return empty data for every call."""
    adapter = _pg_without_pool()
    with pytest.raises(StorageNotInitializedError):
        await getattr(adapter, op)(*args)


# ── event-mutation + bulk-save cluster (batch 3) ─────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op,args",
    [
        ("update_event", (1, {"team": "away"})),
        ("delete_event", (1,)),
        ("hard_delete_event", (1,)),
        ("restore_event", (1,)),
        ("save_validation_result", (None,)),
        ("save_players_bulk", (1, [])),
        ("save_advanced_metrics_bulk", (1, [])),
    ],
)
async def test_batch3_closed_connection_raises_not_initialized(op, args):
    """Batch-3 converts must raise — never a silent False/[]/0. The old
    silent False on delete/update made a closed-connection failure
    indistinguishable from 'event not found', and the empty-input 0 return
    masked 'storage was never opened' as 'nothing to save'."""
    svc = _migrated_sqlite_storage()
    svc._conn.close()
    svc._conn = None
    with pytest.raises(StorageNotInitializedError):
        if op == "save_validation_result":
            from kawkab.services.validation_service import ValidationReport

            await svc.save_validation_result(
                ValidationReport(match_id=1, ground_truth_source="auto")
            )
        else:
            await getattr(svc, op)(*args)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op,args",
    [
        ("delete_event", (1,)),
        ("hard_delete_event", (1,)),
        ("restore_event", (1,)),
        ("save_players_bulk", (1, [])),
        ("save_advanced_metrics_bulk", (1, [])),
        ("hard_delete_match", (1,)),
        ("restore_match", (1,)),
        ("update_event", (1, {"team": "away"})),
        ("hard_delete_player", (1,)),
        ("save_benchmark", (None,)),
    ],
)
async def test_batch3_pg_pool_down_raises_not_initialized(op, args):
    """PG-side of the same contract — pool down raises, with the *correct*
    operation name (the first pass of this batch copy-pasted
    save_events_bulk/save_advanced_metrics_bulk into delete_event and
    save_validation_result, and batch 2 had left get_match/get_all_matches
    inside hard_delete_match/restore_match and get_match_events/
    get_match_players/get_reports inside update_event/hard_delete_player/
    save_benchmark — every failure mislabeled)."""
    adapter = _pg_without_pool()
    with pytest.raises(StorageNotInitializedError) as excinfo:
        await getattr(adapter, op)(*args)
    assert excinfo.value.operation == op


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op,args",
    [
        ("save_players_bulk", (424242, [{"track_id": 1, "name": "P"}])),
        ("save_advanced_metrics_bulk", (424242, [{"metric_name": "xG", "metric_value": 0.1}])),
    ],
)
async def test_batch3_fk_violation_on_bulk_save_raises_write_error(op, args):
    """A missing match row (FK violation) during a bulk save must RAISE,
    not return the row count as if all rows persisted — the exact lie the
    old 0-return told importers on partial FK failures."""
    svc = _migrated_sqlite_storage()
    try:
        with pytest.raises(StorageWriteError) as excinfo:
            await getattr(svc, op)(*args)
        assert excinfo.value.operation == op
        assert excinfo.value.__cause__ is not None
    finally:
        svc._conn.close()


@pytest.mark.asyncio
async def test_batch3_empty_inputs_keep_zero_contract():
    """Empty list is a CONTRACT outcome (nothing to save), not a failure —
    the batch kept the legacy 0 return while making pool-down raise."""
    svc = _migrated_sqlite_storage()
    try:
        assert await svc.save_players_bulk(1, []) == 0
        assert await svc.save_advanced_metrics_bulk(1, []) == 0
    finally:
        svc._conn.close()


# ── handler surface: failure vs rejection vs success ─────────────────────────


@pytest.mark.asyncio
async def test_handler_distinguishes_failure_from_rejection():
    """Through the real handler: DB failure -> sanitized error payload;
    rejected input -> the same 'Tag rejected' payload as before (input
    feedback, not error masking)."""
    from kawkab.ui.bridge_handlers.bridge_coding import CodingHandler

    failing = MagicMock(
        save_coding_tag=AsyncMock(side_effect=StorageWriteError("save_coding_tag", "boom"))
    )
    h = CodingHandler(None, {"storage_service": failing})
    out = json.loads(await h.save_tag("1", json.dumps({"event_type": "shot"})))
    assert "error" in out and "save_coding_tag" in out["error"]

    rejecting = MagicMock(save_coding_tag=AsyncMock(return_value=0))
    h2 = CodingHandler(None, {"storage_service": rejecting})
    out2 = json.loads(await h2.save_tag("1", json.dumps({"notes": "no type"})))
    assert "error" in out2 and "Tag rejected" in out2["error"]


@pytest.mark.asyncio
async def test_handler_reports_read_failure_honestly():
    from kawkab.ui.bridge_handlers.bridge_coding import CodingHandler

    failing = MagicMock(
        get_coding_tags=AsyncMock(side_effect=StorageReadError("get_coding_tags", "disk full"))
    )
    h = CodingHandler(None, {"storage_service": failing})
    out = json.loads(await h.get_tags("1"))
    assert "error" in out


# ── batch 4b: every remaining write raises — media, GPS, recruitment, auth ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op,args",
    [
        ("save_benchmark", (None,)),
        ("save_feedback", ({"match_id": 1, "overall_rating": 4},)),
        ("save_issue", ({"description": "x"},)),
        ("save_usage_session", ({"session_id": "s"},)),
        ("save_clip", ({"match_id": 1},)),
        ("save_playlist", ({"name": "p"},)),
        ("save_player_profile", ({"display_name": "P"},)),
        ("update_player_profile_face", (1, "[0.1]", 0.9)),
        ("save_gps_session", (1, 1, "match", "catapult")),
        ("update_gps_session_stats", (1, {})),
        ("save_gps_samples_bulk", (1, [{"timestamp": 0.0}])),
        ("save_acwr", (1, "2026-01-01", 5000, 4500, 1.11)),
        ("save_shortlist_entry", ({"player_id": 1, "player_name": "P"},)),
        ("update_shortlist_entry", (1, {"status": "contacted"})),
        ("delete_shortlist_entry", (1,)),
        (
            "save_contract",
            ({"player_profile_id": 1, "player_name": "P", "start_date": "d", "end_date": "d"},),
        ),
        ("create_user", ("u", "h")),
        ("update_user_login", (1,)),
        ("record_failed_login", ("u",)),
        ("save_session", (1, "h", "e")),
        ("delete_session", ("h",)),
        ("audit_log", (1, "u", "login")),
        ("change_password", (1, "h")),
        ("clear_expired_lock", (1,)),
    ],
)
async def test_batch4b_sqlite_closed_connection_raises(op, args):
    """The whole remaining write surface raises: a silent 0 on
    save_session or audit_log previously made authentication and the
    security trail silently disappear when the DB handle was gone."""
    svc = _migrated_sqlite_storage()
    svc._conn.close()
    svc._conn = None
    with pytest.raises(StorageNotInitializedError):
        if op == "save_benchmark":
            from kawkab.services.benchmark_service import BenchmarkResult

            await svc.save_benchmark(BenchmarkResult())
        else:
            await getattr(svc, op)(*args)


@pytest.mark.asyncio
async def test_batch4b_contract_outcomes_keep_falsy_returns():
    """Rejected input and unknown-user stay falsy CONTRACT outcomes, not
    failures — distinguishable from a DB error which raises."""
    svc = _migrated_sqlite_storage()
    try:
        assert await svc.save_feedback({}) == 0
        assert await svc.save_issue({}) == 0
        assert await svc.save_clip({}) == 0
        assert await svc.save_playlist({}) == 0
        assert await svc.save_usage_session({}) == 0
        assert await svc.save_gps_samples_bulk(1, []) == 0
        # Unknown username in record_failed_login: 0 = nothing to count.
        assert await svc.record_failed_login("no_such_user") == 0
    finally:
        svc._conn.close()


@pytest.mark.asyncio
async def test_batch4b_auth_lockout_flow_survives_raises():
    """The designed lockout flow still works through the honest contract:
    5 failed logins lock, 0 remaining, and the raise never masks it."""
    svc = _migrated_sqlite_storage()
    try:
        uid = await svc.create_user("lockme", "hash", must_reset_password=False)
        assert uid > 0
        remaining = 5
        for _ in range(5):
            remaining = await svc.record_failed_login("lockme")
        assert remaining == 0
        user = await svc.get_user_by_id(uid)
        assert user["is_locked"] in (1, True)
        # Audit trail records the lock event honestly too.
        assert await svc.audit_log(uid, "lockme", "account_locked") > 0
    finally:
        svc._conn.close()


# ── duplicates: a contract outcome, not a failure ───────────────────────


@pytest.mark.asyncio
async def test_duplicate_event_raises_typed_duplicate_not_write_error():
    """The events dedup index (migration 015) makes same-second-same-type
    events a real, EXPECTED outcome (consecutive ball receipts). It must
    raise StorageDuplicateError — catchable as failure by generic callers,
    catchable as skip by importers — never a swallowed 0 like before, and
    never an opaque failure that breaks importer dedup (the v0.13.2-batch
    regression)."""
    from kawkab.services.storage_errors import StorageDuplicateError

    svc = _migrated_sqlite_storage()
    try:
        match_id = await svc.save_match("dup-test", "")
        ev = {"type": "pass", "timestamp": 12.5, "team": "home", "from_track_id": 77}
        assert await svc.save_event(match_id, ev) > 0
        with pytest.raises(StorageDuplicateError) as excinfo:
            await svc.save_event(match_id, ev)
        # Subclass contract: generic failure handlers still catch it.
        assert isinstance(excinfo.value, StorageWriteError)
    finally:
        svc._conn.close()


@pytest.mark.asyncio
async def test_fk_violation_is_not_classified_as_duplicate():
    """Classification precision: a missing referenced row is a FAILURE,
    not a duplicate — misclassifying it would make importers silently
    skip data that should have failed loudly (the v0.13.2 bug class)."""
    svc = _migrated_sqlite_storage()
    try:
        with pytest.raises(StorageWriteError) as excinfo:
            await svc.save_event(424242, {"type": "pass", "timestamp": 1.0, "from_track_id": 5})
        assert not isinstance(excinfo.value, StorageDuplicateError)
    finally:
        svc._conn.close()


def test_duplicate_classification_by_driver_code_and_message():
    """Classifier: structured codes where the driver offers them, message
    fallback where it doesn't; FK violations never classify as duplicate."""

    # sqlite extended result codes: 1555 PK, 2067 unique index.
    pk = sqlite3.IntegrityError("UNIQUE constraint failed: events.id")
    pk.sqlite_errorcode = 1555
    idx = sqlite3.IntegrityError("UNIQUE constraint failed: x.y")
    idx.sqlite_errorcode = 2067
    assert is_duplicate_violation(pk)
    assert is_duplicate_violation(idx)

    # Postgres SQLSTATE 23505 (asyncpg carries .sqlstate).
    pg_like = RuntimeError("x")
    pg_like.sqlstate = "23505"
    assert is_duplicate_violation(pg_like)

    # Message fallback for drivers with neither attribute.
    assert is_duplicate_violation(RuntimeError("duplicate key value violates"))
    assert not is_duplicate_violation(RuntimeError("FOREIGN KEY constraint failed"))
    assert not is_duplicate_violation(RuntimeError("disk I/O error"))


@pytest.mark.asyncio
async def test_importer_dedup_contract_end_to_end():
    """The full designed flow through the real importer: the corpus sample
    contains same-second same-type events, which hit the migration-015
    dedup index, raise the typed error, and are SKIPPED + counted — not
    swallowed, not fatal (the exact path broken by the opaque
    StorageWriteError wrap)."""
    from kawkab.services.statsbomb_import_service import StatsBombImportService

    svc = _migrated_sqlite_storage()
    try:
        corpus = Path(__file__).resolve().parents[2] / "data" / "statsbomb_corpus"
        sample = next(corpus.glob("*.json"), None)
        if sample is None:
            pytest.skip("no committed statsbomb corpus file")
        importer = StatsBombImportService(svc)
        summary = await importer.import_match(sample)
        assert summary["events_imported"] > 0
        assert summary["events_skipped"] > 0
    finally:
        svc._conn.close()


# ── batch 4a: sealing the holes in previously-converted clusters ───────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op,args",
    [
        ("hard_delete_match", (1,)),
        ("restore_match", (1,)),
        ("hard_delete_player", (1,)),
        ("restore_player", (1,)),
        ("ensure_team", ("T",)),
        ("save_team", ("T",)),
        ("save_tracking_frame", (1, 1, 0.0, [], [])),
        ("save_tracking_frames_bulk", (1, [{"frame_number": 1}])),
        ("save_tracking_import", (1, "skillcorner")),
        ("delete_tracking_import", (1,)),
        ("save_event_frame_links_bulk", (1, [{"event_id": 1}])),
        ("register_match_external_id", (1, "statsbomb", "999")),
        ("delete_tracking_frames", (1,)),
    ],
)
async def test_batch4a_sqlite_closed_connection_raises(op, args):
    """The delete/restore twins, team helpers, and the whole tracking/
    provenance cluster raise like the rest of their clusters — a silent
    False here made a lost import-provenance row look like a no-op."""
    svc = _migrated_sqlite_storage()
    svc._conn.close()
    svc._conn = None
    with pytest.raises(StorageNotInitializedError):
        await getattr(svc, op)(*args)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op,args",
    [
        ("update_match_apifootball", (1,)),
        ("update_match_bzzoiro", (1,)),
        ("restore_player", (1,)),
        ("ensure_team", ("T",)),
        ("save_team", ("T",)),
        ("save_tracking_frame", (1, 1, 0.0, [], [])),
        ("save_tracking_frames_bulk", (1, [{"frame_number": 1}])),
        ("save_tracking_import", (1, "skillcorner")),
        ("delete_tracking_import", (1,)),
        ("save_event_frame_links_bulk", (1, [{"event_id": 1}])),
        ("register_match_external_id", (1, "statsbomb", "999")),
        ("delete_tracking_frames", (1,)),
    ],
)
async def test_batch4a_pg_pool_down_raises(op, args):
    """PG side of batch 4a — including the two match-cluster stragglers
    (update_match_apifootball/bzzoiro) that batch 2 left fail-soft."""
    adapter = _pg_without_pool()
    with pytest.raises(StorageNotInitializedError) as excinfo:
        await getattr(adapter, op)(*args)
    assert excinfo.value.operation == op


@pytest.mark.asyncio
async def test_batch4a_tracking_import_fk_failure_raises():
    """Import-provenance rows FK to matches: a bulk provenance save against
    a missing match must RAISE, not return a row id that doesn't exist."""
    svc = _migrated_sqlite_storage()
    try:
        with pytest.raises(StorageWriteError) as excinfo:
            await svc.save_tracking_import(424242, "skillcorner")
        assert excinfo.value.operation == "save_tracking_import"
    finally:
        svc._conn.close()


# ── op-name integrity: every typed error names ITS OWN method ───────────


def _assert_raise_operation_names_match_methods(file_path: Path) -> list[str]:
    """AST-walk a backend file; return every `Storage*Error("name")` whose
    literal doesn't equal its enclosing function/method name."""
    import ast

    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []

    def _walk(node, enclosing: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _walk(child, child.name)
                continue
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                cls = getattr(child.func, "id", "")
                if (
                    cls
                    in (
                        "StorageNotInitializedError",
                        "StorageWriteError",
                        "StorageReadError",
                        "StorageDuplicateError",
                    )
                    and child.args
                    and isinstance(child.args[0], ast.Constant)
                ):
                    op = child.args[0].value
                    if enclosing is not None and op != enclosing:
                        mismatches.append(
                            f"{file_path.name}:{child.lineno}: {enclosing} raises {cls}({op!r})"
                        )
            _walk(child, enclosing)

    _walk(tree, None)
    return mismatches


def test_typed_error_operation_names_match_their_methods():
    """THE copy-paste bug class, dead permanently. Seven real instances
    shipped across batches 2-3 (delete_event raising 'save_events_bulk',
    save_validation_result raising 'save_advanced_metrics_bulk',
    hard_delete_match raising 'get_match', restore_match raising
    'get_all_matches', update_event raising 'get_match_events',
    hard_delete_player raising 'get_match_players', save_benchmark raising
    'get_reports') — every DB failure in logs/handler payloads mislabeled
    with an unrelated operation. This meta-test fails the build the moment
    any typed error names a different method."""
    backends_dir = Path(__file__).resolve().parents[2] / "src" / "kawkab" / "services"
    mismatches: list[str] = []
    for name in ("postgres_storage.py", "storage_service.py"):
        mismatches.extend(_assert_raise_operation_names_match_methods(backends_dir / name))
    assert not mismatches, "typed errors with wrong operation name:\n" + "\n".join(mismatches)


# ── batch 5b: reads raise too — silent []/None hid broken backends ──────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op,args",
    [
        ("get_all_matches", ()),
        ("get_match", (1,)),
        ("get_match_events", (1,)),
        ("get_match_players", (1,)),
        ("get_all_feedback", ()),
        ("get_all_issues", ()),
        ("get_clips_for_match", (1,)),
        ("get_playlists", ()),
        ("get_tracking_imports", (1,)),
        ("get_tracking_import_by_id", (1,)),
        ("get_event_frame_links", (1,)),
        ("get_match_by_external_id", ("skillcorner", "x")),
        ("get_gps_sessions", (1,)),
        ("get_gps_samples", (1,)),
        ("get_player_acwr", (1,)),
        ("get_player_gps_summary", (1,)),
        ("get_squad_injury_report", (1,)),
        ("get_user_by_username", ("admin",)),
        ("get_user_by_id", (1,)),
        ("get_audit_log", ()),
        ("validate_session", ("tok",)),
        ("rotate_encryption_key", ()),
    ],
)
async def test_read_closed_connection_raises_not_initialized(op, args):
    svc = StorageService()
    svc._pg = None
    svc._use_postgres = False
    svc._conn = None
    with pytest.raises(StorageNotInitializedError) as ei:
        await getattr(svc, op)(*args)
    assert ei.value.operation == op


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "op,args",
    [
        ("get_tracking_imports", (1,)),
        ("get_tracking_import_by_id", (1,)),
        ("get_match_by_external_id", ("skillcorner", "x")),
        ("get_event_frame_links", (1,)),
        ("get_user_by_username", ("admin",)),
        ("get_user_by_id", (1,)),
        ("get_all_users", ()),
        ("get_audit_log", ()),
        ("get_gps_sessions", (1,)),
        ("get_gps_samples", (1,)),
        ("get_player_acwr", (1,)),
        ("get_player_gps_summary", (1,)),
        ("get_squad_injury_report", (1,)),
        ("get_tracking_frame_count", (1,)),
        ("get_encryption_key", ()),
        ("get_cache", ("k",)),
        ("get_contracts_expiring_soon", (30,)),
        ("get_setting", ("k",)),
        ("get_schema_version", ()),
        ("validate_session", ("tok",)),
    ],
)
async def test_pg_read_pool_down_raises_with_operation_name(op, args):
    adapter = PostgresStorageAdapter("postgresql://x:x@localhost:1/db")
    adapter._pool = None
    with pytest.raises(StorageNotInitializedError) as ei:
        await getattr(adapter, op)(*args)
    assert ei.value.operation == op


# ── Phase 5: PG-only methods must exist on SQLite too ───────────────────


@pytest.mark.asyncio
async def test_save_wearable_session_persists_on_sqlite():
    """wearables/service.save_session called save_wearable_session, which
    existed only on the PG adapter — desktop (SQLite) imports crashed with
    a bare AttributeError. The method now exists on StorageService and
    honestly persists via migration 020's wearable_sessions table."""
    svc = _migrated_sqlite_storage()

    with pytest.raises(StorageNotInitializedError):
        # Uninitialized service refuses honestly instead of AttributeError.
        await StorageService().save_wearable_session({"device_type": "catapult"})

    row = {
        "match_id": None,
        "athlete_id": "a-1",
        "athlete_name": "Test Athlete",
        "device_type": "catapult",
        "duration_s": 3000.0,
        "total_distance_m": 8500.0,
        "point_count": 1200,
        "metadata_json": "{}",
    }
    session_id = await svc.save_wearable_session(row)
    assert session_id > 0  # type: ignore[unreachable]

    # Roundtrip through the wearables facade — the production path.
    from kawkab.services.wearables.models import WearableDataPoint, WearableSession
    from kawkab.services.wearables.service import WearableImportService

    parsed = WearableSession(
        device_type="polar",
        athlete_id="a-2",
        athlete_name="Facade Athlete",
        duration_s=60.0,
        data=[
            WearableDataPoint(timestamp_s=0.0, heart_rate_bpm=120.0, speed_ms=3.0),
            WearableDataPoint(timestamp_s=1.0, heart_rate_bpm=150.0, speed_ms=5.0),
        ],
    )
    result = await WearableImportService().save_session(parsed, storage_service=svc, match_id=None)
    assert result.get("ok") is True, result
    assert result.get("session_id", 0) > 0
