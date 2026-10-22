from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from kawkab.cloud.database import get_cloud_db

_jwt_secret: str | None = None
MIN_JWT_SECRET_LENGTH = 32  # bytes; matches HS256's recommended minimum HMAC key length


def _get_jwt_secret() -> str:
    global _jwt_secret
    if _jwt_secret is not None:
        return _jwt_secret
    val = os.environ.get("KAWKAB_JWT_SECRET")
    if not val:
        raise RuntimeError(
            "KAWKAB_JWT_SECRET environment variable is not set. "
            'Generate a strong secret (e.g., `python -c "import secrets; print(secrets.token_hex(32))"`) '
            "and export KAWKAB_JWT_SECRET before starting the cloud server."
        )
    if len(val) < MIN_JWT_SECRET_LENGTH:
        raise RuntimeError(
            f"KAWKAB_JWT_SECRET is only {len(val)} characters -- HS256 needs at least "
            f"{MIN_JWT_SECRET_LENGTH} to resist brute-force forgery of session tokens. "
            'Generate a strong secret (e.g., `python -c "import secrets; print(secrets.token_hex(32))"`) '
            "and export it as KAWKAB_JWT_SECRET before starting the cloud server."
        )
    _jwt_secret = val
    return val


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    salt = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600_000)
    return f"{salt}${pwd_hash.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, pwd_hash = stored.split("$", 1)
        computed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600_000)
        return hmac.compare_digest(computed.hex(), pwd_hash)
    except (ValueError, AttributeError):
        return False


def create_access_token(user_id: int, role: str = "analyst", token_version: int = 0) -> str:
    expire = datetime.now(UTC) + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    # "tv" (token_version) lets a 30-day token be revoked before it
    # naturally expires: get_current_user() below rejects any token whose
    # tv doesn't match the user's *current* token_version in the DB.
    # Bumping that column (e.g. on password change) instantly invalidates
    # every token issued before the bump, with no denylist to maintain.
    return jwt.encode(
        {"sub": str(user_id), "role": role, "tv": token_version, "exp": expire},
        _get_jwt_secret(),
        algorithm=ALGORITHM,
    )


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, _get_jwt_secret(), algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    db = get_cloud_db()
    user = db.execute(
        "SELECT id, username, email, display_name, role, is_active, token_version, created_at FROM users WHERE id = ?",
        (int(payload["sub"]),),
    ).fetchone()
    if user is None or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive"
        )
    # A token minted before create_access_token() carried "tv" (e.g. one
    # issued before this fix shipped) has no "tv" claim at all -- treat
    # that as version 0 rather than erroring, so already-issued tokens
    # for users who've never had their token_version bumped keep working.
    if payload.get("tv", 0) != user["token_version"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked, please log in again",
        )
    return dict(user)


class APIKeyManager:
    """Manage API keys stored in the cloud DB."""

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def create_api_key(user_id: int, name: str, permission: str = "read") -> dict:
        db = get_cloud_db()
        prefix = name[:4].upper() + "_"
        raw_key = prefix + secrets.token_hex(24)
        key_hash = APIKeyManager._hash_key(raw_key)
        db.execute(
            "INSERT INTO api_keys (user_id, name, key_hash, prefix, permission) VALUES (?, ?, ?, ?, ?)",
            (user_id, name, key_hash, prefix, permission),
        )
        db.commit()
        return {"api_key": raw_key, "name": name, "permission": permission}

    @staticmethod
    def validate_api_key(key: str) -> dict | None:
        db = get_cloud_db()
        key_hash = APIKeyManager._hash_key(key)
        row = db.execute(
            "SELECT api_keys.*, users.role FROM api_keys JOIN users ON api_keys.user_id = users.id WHERE key_hash = ? AND is_active = 1",
            (key_hash,),
        ).fetchone()
        if row is None:
            return None
        db.execute("UPDATE api_keys SET last_used_at = datetime('now') WHERE id = ?", (row["id"],))
        db.commit()
        return dict(row)

    @staticmethod
    def revoke_api_key(key: str) -> bool:
        db = get_cloud_db()
        key_hash = APIKeyManager._hash_key(key)
        cur = db.execute("UPDATE api_keys SET is_active = 0 WHERE key_hash = ?", (key_hash,))
        db.commit()
        return cur.rowcount > 0

    @staticmethod
    def list_api_keys(user_id: int) -> list[dict]:
        db = get_cloud_db()
        rows = db.execute(
            "SELECT id, name, prefix, permission, is_active, last_used_at, created_at, expires_at FROM api_keys WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user_or_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    api_key: str | None = Depends(_api_key_header),
) -> dict:
    if credentials:
        payload = decode_token(credentials.credentials)
        if payload:
            db = get_cloud_db()
            user = db.execute(
                "SELECT id, username, email, display_name, role, is_active, token_version, created_at FROM users WHERE id = ?",
                (int(payload["sub"]),),
            ).fetchone()
            if user and user["is_active"] and payload.get("tv", 0) == user["token_version"]:
                return dict(user)
    if api_key:
        key_data = APIKeyManager.validate_api_key(api_key)
        if key_data:
            db = get_cloud_db()
            user = db.execute(
                "SELECT id, username, email, display_name, role, is_active, created_at FROM users WHERE id = ?",
                (key_data["user_id"],),
            ).fetchone()
            if user and user["is_active"]:
                result = dict(user)
                result["_api_key_permission"] = key_data["permission"]
                return result
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
