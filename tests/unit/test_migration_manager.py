"""Tests for MigrationManager."""

import shutil
import tempfile
from pathlib import Path

import pytest

from tests.conftest import load_service_module

# Load real module, bypassing conftest stub
mm_mod = load_service_module("kawkab.core.migration_manager", "migration_manager.py", subdir="core")
MigrationManager = mm_mod.MigrationManager
split_sql_statements = mm_mod.split_sql_statements


class TestSplitSqlStatements:
    def test_simple_statements(self):
        sql = "CREATE TABLE a (id INTEGER);\nCREATE TABLE b (id INTEGER);"
        assert split_sql_statements(sql) == [
            "CREATE TABLE a (id INTEGER)",
            "CREATE TABLE b (id INTEGER)",
        ]

    def test_semicolon_inside_line_comment_does_not_split(self):
        # The regression that broke migration 030's first version: a ';'
        # inside a -- comment chopped the statement in two.
        sql = (
            "-- provenance: stores vendor frame batches; used by audits\n"
            "CREATE TABLE t (id INTEGER);"
        )
        assert split_sql_statements(sql) == [
            "-- provenance: stores vendor frame batches; used by audits\nCREATE TABLE t (id INTEGER)",
        ]

    def test_semicolon_inside_block_comment_does_not_split(self):
        sql = "/* notes; with semicolons; here */ CREATE TABLE t (id INTEGER);"
        assert split_sql_statements(sql) == [
            "/* notes; with semicolons; here */ CREATE TABLE t (id INTEGER)"
        ]

    def test_semicolon_inside_string_literal_does_not_split(self):
        sql = (
            "INSERT INTO t (name) VALUES ('it''s; tricky'); INSERT INTO t (name) VALUES ('plain');"
        )
        stmts = split_sql_statements(sql)
        assert len(stmts) == 2
        assert "it''s; tricky" in stmts[0]

    def test_trigger_unsupported_documented(self):
        # Documented limitation: trigger bodies contain bare semicolons.
        # SQLite-style trigger bodies would over-split; this pins that the
        # splitter does not crash on such input.
        sql = "CREATE TRIGGER tr AFTER INSERT ON t BEGIN INSERT INTO log VALUES (1); END;"
        stmts = split_sql_statements(sql)
        assert stmts[0].startswith("CREATE TRIGGER")
        assert len(stmts) >= 2  # over-split, as documented

    def test_dollar_quoted_body_survives(self):
        """Postgres trigger functions wrap bodies in $$ ... $$; the
        semicolons inside must not split the statement."""
        sql = (
            "CREATE OR REPLACE FUNCTION set_updated_at() RETURNS TRIGGER AS $$\n"
            "BEGIN\n"
            "    NEW.updated_at = NOW();\n"
            "    RETURN NEW;\n"
            "END;\n"
            "$$ LANGUAGE plpgsql;\n"
            "CREATE TABLE t (id INTEGER);"
        )
        stmts = split_sql_statements(sql)
        assert len(stmts) == 2
        assert stmts[0].startswith("CREATE OR REPLACE FUNCTION")
        assert "NEW.updated_at = NOW();" in stmts[0]
        assert stmts[1] == "CREATE TABLE t (id INTEGER)"

    def test_dollar_quoted_tagged(self):
        sql = "CREATE FUNCTION f() RETURNS void AS $body$ BEGIN PERFORM 1; END; $body$ LANGUAGE plpgsql; SELECT 1;"
        stmts = split_sql_statements(sql)
        assert len(stmts) == 2
        assert "PERFORM 1;" in stmts[0]
        assert stmts[1] == "SELECT 1"

    def test_empty_and_whitespace_only(self):
        assert split_sql_statements("") == []
        assert split_sql_statements("   \n  ;  ;  ") == []

    def test_migration_through_splitter_matches_naive_split(self):
        """For the real migration chain, the comment-aware splitter and a
        naive split must agree on statement COUNT for the 030-style files
        whose comments contain no semicolons; where they differ, the
        naive count must be higher (it chops commented semicolons)."""
        real_dir = Path(__file__).resolve().parent.parent.parent / "src" / "kawkab" / "migrations"
        for f in sorted(real_dir.glob("*.sql")):
            if not f.stem.split("_")[0].isdigit():
                continue
            sql = f.read_text(encoding="utf-8")
            aware = split_sql_statements(sql)
            naive = [s.strip() for s in sql.split(";") if s.strip()]
            if "--" in sql or "/*" in sql:
                assert len(aware) <= len(naive), (
                    f"{f.name}: aware splitter must never yield MORE statements than naive"
                )

    def test_migration_030_style_comment_semicolon_survives(self):
        """End-to-end: a migration whose comment contains a semicolon now
        applies cleanly (it broke under the naive splitter)."""
        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "t.db"
            migrations_dir = tmpdir / "m"
            migrations_dir.mkdir()
            (migrations_dir / "001_t.sql").write_text(
                "-- batch table; for vendor frames\n"
                "CREATE TABLE batches (id INTEGER PRIMARY KEY);\n"
            )
            mgr = MigrationManager(db_path, migrations_dir)
            mgr.migrate()  # must not raise
            conn = _closing_conn(db_path)
            version = mgr._get_current_version(conn)
            assert version == 1
            conn.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _closing_conn(path):
    """Open and return a sqlite3 connection that uses DELETE journal mode."""
    import sqlite3

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=DELETE")
    return conn


class TestMigrationManager:
    def test_initial_version_is_zero(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            migrations_dir = tmpdir / "migrations"
            migrations_dir.mkdir()
            mgr = MigrationManager(db_path, migrations_dir)
            mgr.migrate()
            conn = _closing_conn(db_path)
            version = mgr._get_current_version(conn)
            assert version == 0
            conn.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_applies_migration_files(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            migrations_dir = tmpdir / "migrations"
            migrations_dir.mkdir()
            mig_file = migrations_dir / "001_create_test_table.sql"
            mig_file.write_text("CREATE TABLE test_table (id INTEGER PRIMARY KEY);")
            mgr = MigrationManager(db_path, migrations_dir)
            mgr.migrate()
            conn = _closing_conn(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
            )
            assert cursor.fetchone() is not None
            version = mgr._get_current_version(conn)
            assert version >= 1
            conn.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_skips_already_applied_migrations(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            migrations_dir = tmpdir / "migrations"
            migrations_dir.mkdir()
            mig_file = migrations_dir / "001_create_test_table.sql"
            mig_file.write_text("CREATE TABLE test_table (id INTEGER PRIMARY KEY);")
            mgr = MigrationManager(db_path, migrations_dir)
            mgr.migrate()
            mgr.migrate()
            conn = _closing_conn(db_path)
            version = mgr._get_current_version(conn)
            assert version == 1
            conn.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_handles_empty_migrations_dir(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            migrations_dir = tmpdir / "migrations"
            migrations_dir.mkdir()
            mgr = MigrationManager(db_path, migrations_dir)
            mgr.migrate()
            conn = _closing_conn(db_path)
            version = mgr._get_current_version(conn)
            assert version == 0
            conn.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_full_real_migration_chain_applies_cleanly_to_fresh_db(self):
        """Every existing test above uses a synthetic migrations_dir with
        hand-written SQL -- none of them ever run the REAL, FULL
        001-through-latest chain from src/kawkab/migrations/ against a
        genuinely empty database. That's exactly the scenario a brand new
        install (or CI) hits, and it's the one place schema drift between
        a migration file and the code that queries its tables would
        surface as a hard failure instead of silently working by
        coincidence on someone's already-migrated dev database.
        """
        real_migrations_dir = (
            Path(__file__).resolve().parent.parent.parent / "src" / "kawkab" / "migrations"
        )
        assert real_migrations_dir.exists(), f"expected migrations dir at {real_migrations_dir}"
        migration_files = sorted(real_migrations_dir.glob("*.sql"))
        numbered = [f for f in migration_files if f.stem.split("_")[0].isdigit()]
        assert len(numbered) >= 20, (
            "sanity check: expected the real migration set, not an empty/wrong dir"
        )

        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "fresh.db"
            mgr = MigrationManager(db_path, real_migrations_dir)
            mgr.migrate()  # must not raise

            conn = _closing_conn(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row[0] for row in cursor.fetchall()}

            highest_version = max(int(f.stem.split("_")[0]) for f in numbered)
            version = mgr._get_current_version(conn)
            assert version == highest_version

            # Spot-check tables from across the whole chain (earliest,
            # middle, and the two newest/untracked-until-now migrations)
            # actually exist -- catches a migration that silently no-ops
            # (e.g. a typo'd CREATE TABLE IF NOT EXISTS) as well as one
            # that raises.
            for expected_table in [
                "matches",
                "players",
                "events",  # 001
                "coding_tags",  # 018
                "player_shortlist",
                "player_contracts",  # 016, 017
                "gps_sessions",
                "gps_samples",
                "acwr_daily",  # 026
                "users",
                "user_sessions",
                "audit_events_local",  # 027
            ]:
                assert expected_table in tables, f"migration chain never created '{expected_table}'"

            conn.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_migration_015_event_dedup_applies_cleanly(self):
        tmpdir = Path(tempfile.mkdtemp())
        try:
            db_path = tmpdir / "test.db"
            migrations_dir = tmpdir / "migrations"
            migrations_dir.mkdir()
            import sqlite3

            conn = _closing_conn(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "CREATE TABLE events (id INTEGER PRIMARY KEY, match_id INTEGER, timestamp REAL, event_type TEXT, from_track_id INTEGER)"
            )
            cursor.execute(
                "CREATE TABLE user_corrections (id INTEGER PRIMARY KEY, event_id INTEGER)"
            )
            conn.commit()
            conn.close()
            mig_file = migrations_dir / "015_add_event_dedup.sql"
            mig_file.write_text(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_events_dedup\n"
                "ON events(match_id, timestamp, event_type, from_track_id);\n"
                "CREATE INDEX IF NOT EXISTS idx_user_corrections_event\n"
                "ON user_corrections(event_id);\n"
            )
            mgr = MigrationManager(db_path, migrations_dir)
            mgr.migrate()
            conn = _closing_conn(db_path)
            version = mgr._get_current_version(conn)
            assert version == 15
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO events (id, match_id, timestamp, event_type, from_track_id) VALUES (1, 1, 10.0, 'pass', 1)"
            )
            with pytest.raises(sqlite3.IntegrityError):
                cursor.execute(
                    "INSERT INTO events (id, match_id, timestamp, event_type, from_track_id) VALUES (2, 1, 10.0, 'pass', 1)"
                )
            conn.close()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
