"""Database migration system for Kawkab AI.

Simple numbered migration system. Migrations are SQL scripts that upgrade
the schema from one version to the next. On startup, StorageService checks
the current schema_version and applies any pending migrations.

Migration files: src/kawkab/migrations/001_initial.sql, 002_add_seasons.sql, etc.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from kawkab.core.logging import get_logger

logger = get_logger(__name__)


def split_sql_statements(sql: str) -> list[str]:
    """Split a SQL script into individual statements, comment- and
    string-literal-aware.

    A naive ``sql.split(";")`` corrupts any script containing a semicolon
    inside a ``--`` line comment, a ``/* ... */`` block comment, or a
    string literal ('it''s; here' with SQLite's doubled-quote escaping):
    the semicolon chops one real statement into two broken ones. This
    scanner treats those regions as opaque, so a ``;`` there never splits.
    Postgres dollar-quoted bodies (``$$ ... $$``, e.g. trigger functions)
    are also treated as opaque, which is what lets a function body full of
    semicolons survive as ONE statement.

    Scope note: this is NOT a full SQLite parser. ``CREATE TRIGGER ...
    BEGIN ... END;`` bodies still contain bare semicolons and are not
    supported (no migration file in this project uses one -- if that
    changes, this function needs to grow trigger-body awareness).
    """
    statements: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if ch == "-" and nxt == "-":  # line comment: opaque until newline
            j = sql.find("\n", i)
            j = n if j == -1 else j
            buf.append(sql[i:j])
            i = j
        elif ch == "/" and nxt == "*":  # block comment: opaque until */
            j = sql.find("*/", i + 2)
            j = n if j == -1 else j + 2
            buf.append(sql[i:j])
            i = j
        elif ch == "$":  # Postgres dollar-quoted body ($$...$$ or $tag$...$tag$)
            m = re.match(r"\$[A-Za-z_]*\$", sql[i:])
            if m:
                tag = m.group(0)
                j = sql.find(tag, i + len(tag))
                j = n if j == -1 else j + len(tag) - 1
                buf.append(sql[i : j + 1])
                i = j + 1
            else:
                buf.append(ch)
                i += 1
        elif ch == "'" or ch == '"':  # string literal / quoted identifier
            quote = ch
            j = i + 1
            while j < n:
                if sql[j] == quote:
                    if j + 1 < n and sql[j + 1] == quote:  # doubled = escaped
                        j += 2
                        continue
                    break
                j += 1
            buf.append(sql[i : j + 1])
            i = j + 1
        elif ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
        else:
            buf.append(ch)
            i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


class MigrationManager:
    """Manages database schema migrations."""

    def __init__(self, db_path: Path, migrations_dir: Path) -> None:
        self.db_path = db_path
        self.migrations_dir = migrations_dir

    def _get_current_version(self, conn: sqlite3.Connection) -> int:
        """Get current schema version from the database."""
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else 0

    def _set_version(self, conn: sqlite3.Connection, version: int) -> None:
        """Record that a migration has been applied.

        Does NOT commit -- the caller (migrate()) controls the transaction
        boundary so this write lands in the same transaction as the
        migration's own schema changes. Committing here independently
        would defeat that: a failure between this call and the schema
        changes committing would record a version bump for a migration
        that never actually finished applying.
        """
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, CURRENT_TIMESTAMP)",
            (version,),
        )

    def _get_migration_files(self) -> list[Path]:
        """Get all migration files sorted by version number."""
        if not self.migrations_dir.exists():
            return []
        all_files = list(self.migrations_dir.glob("*.sql"))
        numeric_files = []
        for f in all_files:
            stem = f.stem
            parts = stem.split("_")
            if parts and parts[0].isdigit():
                numeric_files.append((int(parts[0]), f))
        numeric_files.sort(key=lambda x: x[0])
        return [f for _, f in numeric_files]

    def migrate(self) -> None:
        """Apply all pending migrations.

        Each migration file runs inside one explicit transaction, so a
        failure partway through a file rolls back every statement from
        that file instead of leaving the schema partially changed with no
        way forward. Previously this used conn.executescript(sql) with no
        transaction at all: sqlite3.Connection.executescript() issues an
        implicit COMMIT before running and then autocommits each
        statement in the script as it executes, so it can't be made
        atomic just by wrapping it in BEGIN/COMMIT -- the implicit COMMIT
        would silently swallow that. If migration N failed at, say,
        statement 5 of 9: statements 1-4 were permanently applied,
        _set_version() never ran (it came after the failure), so the next
        startup saw version N-1 and re-applied migration N from statement
        1 -- which then failed immediately re-running a CREATE TABLE/ALTER
        TABLE that had already succeeded. The app was permanently bricked
        with no rollback path.

        Statements are split by split_sql_statements() (comment- and
        string-literal-aware -- a ";" inside a comment or literal no
        longer splits a statement in two) and executed individually
        inside a manual transaction (conn.isolation_level = None -- see
        below). Trigger bodies (CREATE TRIGGER ... BEGIN ... END;) remain
        unsupported; no migration file in this project uses one.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # Manual transaction control: without this, sqlite3's default
        # "autocommit unless a DML statement started one" mode does its
        # own implicit BEGIN handling that fights with the explicit
        # BEGIN/COMMIT/ROLLBACK below.
        conn.isolation_level = None
        try:
            current = self._get_current_version(conn)
            files = self._get_migration_files()

            applied = 0
            for file in files:
                version = int(file.stem.split("_")[0])
                if version > current:
                    sql = file.read_text(encoding="utf-8")
                    logger.info(f"Applying migration {version}: {file.name}")
                    statements = split_sql_statements(sql)
                    conn.execute("BEGIN")
                    try:
                        for stmt in statements:
                            conn.execute(stmt)
                        self._set_version(conn, version)
                        conn.execute("COMMIT")
                        applied += 1
                    except Exception as e:
                        conn.execute("ROLLBACK")
                        logger.error(f"Migration {version} failed and was rolled back: {e}")
                        raise
        finally:
            conn.close()

        if applied > 0:
            logger.info(
                f"Applied {applied} migration(s). Schema now at version {current + applied}"
            )
        else:
            logger.debug(f"Schema up to date at version {current}")
