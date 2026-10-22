"""Tests for local authentication (AuthHandler) and user storage."""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import install_kawkab_stubs, load_service_module

install_kawkab_stubs()
_storage_mod = load_service_module("storage_test", "storage_service.py")
StorageService = _storage_mod.StorageService
from kawkab.ui.bridge_handlers.bridge_auth import AuthHandler, _hash_password, _verify_password

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    display_name TEXT DEFAULT '',
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'analyst',
    team TEXT DEFAULT '',
    is_active INTEGER DEFAULT 1,
    is_locked INTEGER DEFAULT 0,
    failed_attempts INTEGER DEFAULT 0,
    locked_until TEXT,
    must_reset_password INTEGER DEFAULT 0,
    last_login TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token_hash TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS audit_events_local (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    username TEXT,
    action TEXT NOT NULL,
    resource_type TEXT,
    resource_id TEXT,
    details TEXT DEFAULT '{}',
    ip_address TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


@pytest.fixture
def storage():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        svc = StorageService()
        svc._pg = None
        svc._use_postgres = False
        svc._db_path = db_path
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.executescript(CREATE_SQL)
        conn.commit()
        svc._conn = conn
        try:
            yield svc
        finally:
            if svc._conn:
                svc._conn.close()
                svc._conn = None


def _make_auth(svc):
    handler = AuthHandler(None, {"storage_service": svc}, None)
    handler._default_admin_created = True
    return handler


@pytest.mark.asyncio
async def test_create_user(storage):
    uid = await storage.create_user("testuser", "hash123", "analyst", "test@test.com", "Test User")
    assert uid > 0
    user = await storage.get_user_by_username("testuser")
    assert user is not None
    assert user["role"] == "analyst"
    assert user["display_name"] == "Test User"


@pytest.mark.asyncio
async def test_get_user_by_id(storage):
    uid = await storage.create_user("user1", "hash1", "coach")
    user = await storage.get_user_by_id(uid)
    assert user is not None
    assert user["username"] == "user1"


@pytest.mark.asyncio
async def test_get_user_not_found(storage):
    assert await storage.get_user_by_id(999) is None
    assert await storage.get_user_by_username("nonexistent") is None


@pytest.mark.asyncio
async def test_update_user_login(storage):
    uid = await storage.create_user("loginuser", "hash")
    await storage.update_user_login(uid)
    user = await storage.get_user_by_id(uid)
    assert user["last_login"] is not None


@pytest.mark.asyncio
async def test_failed_login_locks_account(storage):
    await storage.create_user("lockuser", "hash")
    for _i in range(4):
        r = await storage.record_failed_login("lockuser")
        assert r > 0
    r = await storage.record_failed_login("lockuser")
    assert r == 0
    user = await storage.get_user_by_username("lockuser")
    assert user["is_locked"] == 1


@pytest.mark.asyncio
async def test_save_and_validate_session(storage):
    uid = await storage.create_user("sessionuser", "hash")
    sid = await storage.save_session(uid, "tokhash123", "2099-01-01T00:00:00")
    assert sid > 0
    user = await storage.validate_session("tokhash123")
    assert user is not None
    assert user["username"] == "sessionuser"


@pytest.mark.asyncio
async def test_validate_expired_session(storage):
    uid = await storage.create_user("expireduser", "hash")
    await storage.save_session(uid, "expiredtok", "2020-01-01T00:00:00")
    user = await storage.validate_session("expiredtok")
    assert user is None


@pytest.mark.asyncio
async def test_delete_session(storage):
    uid = await storage.create_user("deluser", "hash")
    await storage.save_session(uid, "deltok", "2099-01-01T00:00:00")
    assert await storage.delete_session("deltok") is True
    assert await storage.validate_session("deltok") is None


@pytest.mark.asyncio
async def test_audit_log(storage):
    aid = await storage.audit_log(1, "admin", "login", "auth", "session")
    assert aid > 0
    logs = await storage.get_audit_log()
    assert len(logs) >= 1
    assert logs[0]["action"] == "login"


@pytest.mark.asyncio
async def test_audit_log_limit(storage):
    await storage.audit_log(1, "admin", "login")
    await storage.audit_log(1, "admin", "delete", "match", "42")
    logs = await storage.get_audit_log(limit=1)
    assert len(logs) == 1


@pytest.mark.asyncio
async def test_change_password(storage):
    uid = await storage.create_user("changepwd", "oldhash")
    assert await storage.change_password(uid, "newhash") is True
    user = await storage.get_user_by_id(uid)
    assert user is not None


@pytest.mark.asyncio
async def test_get_all_users(storage):
    await storage.create_user("a", "h1", "analyst")
    await storage.create_user("b", "h2", "coach")
    users = await storage.get_all_users()
    assert len(users) >= 2


def test_password_hashing():
    pwd = "SecureP@ss123"
    h = _hash_password(pwd)
    assert h != pwd
    assert "$" in h
    assert _verify_password(pwd, h) is True
    assert _verify_password("WrongPass", h) is False


# Handler-surface tests drive the real async convention (the v0.13.2
# sync implementation used asyncio.run() per storage call, which raises
# under the app's running qasync loop -- dead login in the real app
# while these tests stayed green; the e2e slot audit caught it).


@pytest.mark.asyncio
async def test_auth_login_success(storage):
    pwd_hash = _hash_password("testpass")
    uid = await storage.create_user("logintest", pwd_hash, "analyst")
    assert uid > 0
    auth = _make_auth(storage)
    result = json.loads(await auth.login("logintest", "testpass"))
    assert result.get("success") is True
    assert "token" in result
    assert result["user"]["username"] == "logintest"
    assert result["user"]["role"] == "analyst"


@pytest.mark.asyncio
async def test_auth_login_wrong_password(storage):
    pwd_hash = _hash_password("okpass")
    await storage.create_user("failuser", pwd_hash, "viewer")
    auth = _make_auth(storage)
    result = json.loads(await auth.login("failuser", "wrongpass"))
    assert result.get("success") is not True


@pytest.mark.asyncio
async def test_auth_login_unknown_user(storage):
    auth = _make_auth(storage)
    result = json.loads(await auth.login("ghost", "pass"))
    assert result.get("success") is not True


@pytest.mark.asyncio
async def test_auth_logout(storage):
    pwd_hash = _hash_password("p")
    await storage.create_user("logoutuser", pwd_hash)
    auth = _make_auth(storage)
    login_resp = json.loads(await auth.login("logoutuser", "p"))
    assert login_resp.get("success") is True
    token = login_resp["token"]
    logout_resp = json.loads(await auth.logout(token))
    assert logout_resp.get("success") is True
    get_resp = json.loads(await auth.get_current_user(token))
    assert get_resp.get("success") is not True


@pytest.mark.asyncio
async def test_get_current_user(storage):
    pwd_hash = _hash_password("p")
    await storage.create_user("getuser", pwd_hash)
    auth = _make_auth(storage)
    login_resp = json.loads(await auth.login("getuser", "p"))
    token = login_resp["token"]
    get_resp = json.loads(await auth.get_current_user(token))
    assert get_resp.get("success") is True
    assert get_resp["user"]["username"] == "getuser"


@pytest.mark.asyncio
async def test_get_current_user_invalid_token(storage):
    auth = _make_auth(storage)
    result = json.loads(await auth.get_current_user("invalid_token"))
    assert result.get("success") is not True


@pytest.mark.asyncio
async def test_change_password_via_auth(storage):
    pwd_hash = _hash_password("oldpwd")
    await storage.create_user("changepwduser", pwd_hash)
    auth = _make_auth(storage)
    login_resp = json.loads(await auth.login("changepwduser", "oldpwd"))
    token = login_resp["token"]
    result = json.loads(await auth.change_password(token, "oldpwd", "newpassword6"))
    assert result.get("success") is True


@pytest.mark.asyncio
async def test_change_password_wrong_old(storage):
    pwd_hash = _hash_password("okpwd")
    await storage.create_user("wrongold", pwd_hash)
    auth = _make_auth(storage)
    login_resp = json.loads(await auth.login("wrongold", "okpwd"))
    token = login_resp["token"]
    result = json.loads(await auth.change_password(token, "wrong", "newpwd6"))
    assert result.get("success") is not True


@pytest.mark.asyncio
async def test_list_users_admin(storage):
    await storage.create_user("adminuser", _hash_password("adminpass"), "admin")
    await storage.create_user("normal", _hash_password("p"), "analyst")
    auth = _make_auth(storage)
    login_resp = json.loads(await auth.login("adminuser", "adminpass"))
    token = login_resp["token"]
    result = json.loads(await auth.list_users(token))
    assert result.get("success") is True
    assert len(result["users"]) >= 2


@pytest.mark.asyncio
async def test_list_users_non_admin(storage):
    await storage.create_user("nonadmin", _hash_password("p"), "viewer")
    auth = _make_auth(storage)
    login_resp = json.loads(await auth.login("nonadmin", "p"))
    token = login_resp["token"]
    result = json.loads(await auth.list_users(token))
    assert result.get("success") is not True
