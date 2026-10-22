from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
import threading
from pathlib import Path

CLOUD_DB_PATH = os.environ.get("KAWKAB_CLOUD_DB", str(Path.home() / ".kawkab" / "cloud.db"))


class _ResultRow(dict):
    """A row that supports both dict access and attribute-style access like sqlite3.Row."""

    __slots__ = ()

    def __getitem__(self, key):
        if isinstance(key, int):
            keys = list(self.keys())
            return super().__getitem__(keys[key])
        return super().__getitem__(key)


class _PostgresCursor:
    """Sync cursor wrapper around asyncpg result, mimicking sqlite3.Cursor."""

    def __init__(self, rows: list[dict] | None = None, lastrowid: int = 0):
        self._rows = rows or []
        self._idx = 0
        self._lastrowid = lastrowid

    @property
    def lastrowid(self) -> int:
        return self._lastrowid

    def fetchone(self) -> _ResultRow | None:
        if self._idx >= len(self._rows):
            return None
        row = _ResultRow(self._rows[self._idx])
        self._idx += 1
        return row

    def fetchall(self) -> list[_ResultRow]:
        result = [_ResultRow(r) for r in self._rows[self._idx :]]
        self._idx = len(self._rows)
        return result

    def __iter__(self):
        return iter(self._rows)

    @property
    def rowcount(self) -> int:
        return len(self._rows)


class _PostgresConnection:
    """Sync PostgreSQL connection that mimics sqlite3.Connection interface.

    Uses asyncpg with asyncio.run() internally, safe for sync FastAPI routes.
    """

    def __init__(self, dsn: str):
        self._dsn = dsn
        self.row_factory = None

    def _run(self, coro):
        try:
            _ = asyncio.get_running_loop()
            # Already inside an event loop (e.g. TestClient) — schedule & wait
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, coro)
                return fut.result()
        except RuntimeError:
            return asyncio.run(coro)

    def _with_conn(self, callback):
        """Create a fresh connection, run callback, close it."""
        import asyncpg

        async def _impl():
            conn = await asyncpg.connect(self._dsn)
            try:
                return await callback(conn)
            finally:
                await conn.close()

        return self._run(_impl())

    @staticmethod
    def _convert_placeholders(query: str) -> str:
        parts = list(query)
        result: list[str] = []
        idx = 1
        i = 0
        while i < len(parts):
            c = parts[i]
            if c == "'":
                j = i + 1
                while j < len(parts):
                    if parts[j] == "'" and (j + 1 >= len(parts) or parts[j + 1] != "'"):
                        break
                    j += 1
                result.append(query[i : j + 1])
                i = j + 1
            elif c == "?":
                result.append(f"${idx}")
                idx += 1
                i += 1
            else:
                result.append(c)
                i += 1
        return "".join(result)

    def execute(self, query: str, parameters: tuple = ()) -> _PostgresCursor:
        pg_query = self._convert_placeholders(query) if "?" in query else query

        def _do_exec(conn):
            async def _exec():
                q = pg_query.strip()
                if q.upper().startswith("INSERT"):
                    await conn.execute(pg_query, *parameters)
                    lastval = await conn.fetchval("SELECT lastval()")
                    return [], lastval or 0
                elif q.upper().startswith(("SELECT", "WITH")):
                    rows = await conn.fetch(pg_query, *parameters)
                    return [dict(r) for r in rows], 0
                else:
                    await conn.execute(pg_query, *parameters)
                    return [], 0

            return _exec()

        rows, lastid = self._with_conn(_do_exec)
        return _PostgresCursor(rows, lastrowid=lastid)

    def executescript(self, script: str) -> None:
        statements = [s.strip() for s in script.split(";") if s.strip()]
        if not statements:
            return

        def _do_script(conn):
            async def _exec():
                for stmt in statements:
                    if stmt:
                        await conn.execute(stmt)

            return _exec()

        self._with_conn(_do_script)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


class _SqliteConnection:
    """Thin wrapper exposing a sqlite3.Connection through the same interface as _PostgresConnection."""

    def __init__(self, path: str):
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.row_factory = sqlite3.Row

    def execute(self, query: str, parameters: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(query, parameters)

    def executescript(self, script: str) -> None:
        self._conn.executescript(script)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


def _get_cloud_db_impl() -> _SqliteConnection | _PostgresConnection:
    dsn = os.environ.get("KAWKAB_DB_URL")
    if dsn:
        conn = _PostgresConnection(dsn)
        _pg_migrate(conn)
        return conn
    path = CLOUD_DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn_sqlite = _SqliteConnection(path)
    _sqlite_migrate(conn_sqlite)
    return conn_sqlite


_local = threading.local()


def get_cloud_db() -> _SqliteConnection | _PostgresConnection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _get_cloud_db_impl()
        _local.conn = conn
    return conn


def _sqlite_migrate(db: _SqliteConnection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT DEFAULT '',
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            owner_id INTEGER NOT NULL REFERENCES users(id),
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS team_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL REFERENCES teams(id),
            user_id INTEGER NOT NULL REFERENCES users(id),
            role TEXT DEFAULT 'member',
            UNIQUE(team_id, user_id)
        );

        CREATE TABLE IF NOT EXISTS team_invites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id INTEGER NOT NULL REFERENCES teams(id),
            email TEXT NOT NULL,
            role TEXT DEFAULT 'member',
            token TEXT UNIQUE NOT NULL,
            accepted INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            owner_id INTEGER NOT NULL REFERENCES users(id),
            team_id INTEGER REFERENCES teams(id),
            data TEXT DEFAULT '{}',
            version INTEGER DEFAULT 1,
            is_shared INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sync_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            device_id TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            synced_at TEXT DEFAULT (datetime('now'))
        );

        INSERT OR IGNORE INTO schema_version VALUES (1);

        CREATE TABLE IF NOT EXISTS oauth_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            provider TEXT NOT NULL,
            provider_user_id TEXT NOT NULL,
            access_token TEXT,
            refresh_token TEXT,
            expires_at REAL,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(provider, provider_user_id),
            UNIQUE(user_id, provider)
        );

        INSERT OR IGNORE INTO schema_version VALUES (2);

        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            token_hash TEXT UNIQUE NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

    """)
    db.commit()

    # Migration 3: add role column (idempotent via try/except)
    try:
        db.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'analyst'")
        db.execute("INSERT OR IGNORE INTO schema_version VALUES (3)")
        db.commit()
    except Exception:
        db.rollback()

    # Migration 4: api_keys table
    try:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                key_hash TEXT UNIQUE NOT NULL,
                prefix TEXT DEFAULT '',
                permission TEXT NOT NULL DEFAULT 'read',
                is_active INTEGER DEFAULT 1,
                last_used_at TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                expires_at TEXT
            );
            INSERT OR IGNORE INTO schema_version VALUES (4);
        """)
        db.commit()
    except Exception:
        db.rollback()

    # Migration 5: token_version column -- JWTs carry a "tv" claim checked
    # against this on every request (see cloud/auth.py get_current_user).
    # Bumping it (e.g. on password change) instantly invalidates every
    # previously-issued token for that user without needing a shared
    # denylist -- 30-day JWTs previously had no revocation path at all.
    try:
        db.execute("ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0")
        db.execute("INSERT OR IGNORE INTO schema_version VALUES (5)")
        db.commit()
    except Exception:
        db.rollback()


PG_CLOUD_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT DEFAULT '',
    is_active BOOLEAN DEFAULT TRUE,
    role TEXT DEFAULT 'analyst',
    token_version INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS teams (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    owner_id INTEGER NOT NULL REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS team_members (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT DEFAULT 'member',
    UNIQUE(team_id, user_id)
);
CREATE TABLE IF NOT EXISTS team_invites (
    id SERIAL PRIMARY KEY,
    team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
    email TEXT NOT NULL,
    role TEXT DEFAULT 'member',
    token TEXT UNIQUE NOT NULL,
    accepted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    owner_id INTEGER NOT NULL REFERENCES users(id),
    team_id INTEGER REFERENCES teams(id),
    data JSONB DEFAULT '{}',
    version INTEGER DEFAULT 1,
    is_shared BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS sync_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    device_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    synced_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS oauth_accounts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    provider_user_id TEXT NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    expires_at DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(provider, provider_user_id),
    UNIQUE(user_id, provider)
);
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT UNIQUE NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT UNIQUE NOT NULL,
    prefix TEXT DEFAULT '',
    permission TEXT NOT NULL DEFAULT 'read',
    is_active BOOLEAN DEFAULT TRUE,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);
"""


def _pg_migrate(db: _PostgresConnection) -> None:
    """Create auth tables in PostgreSQL with idempotent migrations."""
    db.executescript(PG_CLOUD_SCHEMA)
    # PG_CLOUD_SCHEMA's users table is CREATE TABLE IF NOT EXISTS, which is
    # a no-op against a database that already has a users table from
    # before token_version existed -- ADD COLUMN IF NOT EXISTS (Postgres-
    # only syntax, unlike SQLite) covers that upgrade path too.
    with contextlib.suppress(Exception):
        db.execute(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS token_version INTEGER NOT NULL DEFAULT 0"
        )
