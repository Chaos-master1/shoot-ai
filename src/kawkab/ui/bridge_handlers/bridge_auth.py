"""Auth bridge handler — login, logout, user management, audit log.

All storage-touching methods are async and awaited by their bridge
delegators (the codebase-wide handler convention). The original sync
implementation called ``asyncio.run()`` per storage call, which raises
"asyncio.run() cannot be called from a running event loop" the moment
the real qasync loop is up -- every login/logout/session check died in
the running app while the unit tests (which called the handler directly
with no loop) stayed green. Caught by the GUI slot-dispatch audit.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
from datetime import UTC, datetime, timedelta

from kawkab.core.logging import get_logger
from kawkab.core.paths import get_paths
from kawkab.core.security import ErrorSanitizer
from kawkab.ui.bridge_handlers.base import BridgeHandlerBase

logger = get_logger(__name__)


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _hash_password(password: str) -> str:
    salt = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600_000)
    return f"{salt}${pwd_hash.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, pwd_hash = stored.split("$", 1)
        computed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 600_000)
        return secrets.compare_digest(computed.hex(), pwd_hash)
    except (ValueError, AttributeError):
        return False


class AuthHandler(BridgeHandlerBase):
    """Handles authentication and authorization for Bridge."""

    def __init__(self, bridge, services, rate_limiter=None):
        super().__init__(bridge, services, rate_limiter)
        self._default_admin_created = False

    @property
    def storage_service(self):
        return self._services.get("storage_service")

    async def _ensure_admin(self):
        """Create default admin on first run if no users exist.

        Previously created a fixed, publicly-documented "admin"/"admin123"
        account and logged that literal credential to the app's own log
        file -- anyone who read the source (or the log) had a standing
        admin login for every install. Now generates a random password,
        forces a reset on first use (must_reset_password=True -- login()
        surfaces this so the caller can't silently keep using it), and
        writes the one-time password only to a dedicated file the coach
        has to go look up, never to the rotating log.
        """
        if self._default_admin_created:
            return
        svc = self.storage_service
        if svc is None:
            return
        try:
            svc = self.storage_service
            if svc is None:
                return
            user = await svc.get_user_by_username("admin")
            if user is None:
                initial_password = secrets.token_urlsafe(16)
                pwd_hash = _hash_password(initial_password)
                await svc.create_user(
                    "admin",
                    pwd_hash,
                    "admin",
                    "admin@kawkab.ai",
                    "Admin",
                    must_reset_password=True,
                )
                creds_file = get_paths().appdata / "FIRST_RUN_ADMIN_PASSWORD.txt"
                creds_file.write_text(
                    "Kawkab AI -- one-time admin credential\n"
                    "========================================\n"
                    "username: admin\n"
                    f"password: {initial_password}\n\n"
                    "You will be required to set a new password on first login.\n"
                    "Delete this file once you've logged in.\n",
                    encoding="utf-8",
                )
                # best-effort on platforms without POSIX permissions (Windows)
                with contextlib.suppress(OSError):
                    os.chmod(creds_file, 0o600)
                logger.info(
                    "Created default admin user; one-time password written to "
                    f"{creds_file} (not logged; password reset required on first login)"
                )
            self._default_admin_created = True
        except Exception as e:
            logger.warning(f"Could not create default admin: {e}")

    async def login(self, username: str, password: str) -> str:
        """Authenticate user and return session token."""
        try:
            svc = self.storage_service
            if svc is None:
                return json.dumps({"error": "Storage unavailable"})
            await self._ensure_admin()

            user = await svc.get_user_by_username(username)
            if user is None:
                return json.dumps({"error": "Invalid username or password"})

            if user.get("is_locked"):
                # record_failed_login() promises "try again in 1 hour" via
                # locked_until, but nothing previously read it back -- the
                # only unlock path was update_user_login(), reachable only
                # via a successful login, which is_locked blocked in the
                # first place. Auto-clear once locked_until has passed
                # instead of leaving the account locked forever.
                locked_until = user.get("locked_until")
                still_locked = True
                if locked_until:
                    try:
                        until_dt = datetime.fromisoformat(locked_until.replace(" ", "T"))
                        still_locked = datetime.now(UTC).replace(tzinfo=None) < until_dt
                    except ValueError:
                        pass
                if still_locked:
                    return json.dumps({"error": "Account locked. Try again later."})
                await svc.clear_expired_lock(user["id"])
                user["is_locked"] = 0

            if not user.get("is_active"):
                return json.dumps({"error": "Account deactivated."})

            if not _verify_password(password, user.get("password_hash", "")):
                remaining = await svc.record_failed_login(username)
                msg = "Invalid password"
                if remaining > 0:
                    msg += f" ({remaining} attempts remaining)"
                return json.dumps({"error": msg})

            await svc.update_user_login(user["id"])

            token_raw = secrets.token_hex(32)
            token_hash = _hash_str(token_raw)
            expires = (datetime.now(UTC) + timedelta(days=7)).isoformat()
            await svc.save_session(user["id"], token_hash, expires)
            await svc.audit_log(user["id"], username, "login", "auth", "session")

            return json.dumps(
                {
                    "success": True,
                    "must_reset_password": bool(user.get("must_reset_password")),
                    "token": token_raw,
                    "user": {
                        "id": user["id"],
                        "username": user["username"],
                        "display_name": user.get("display_name", ""),
                        "role": user["role"],
                        "team": user.get("team", ""),
                        "email": user.get("email", ""),
                    },
                }
            )
        except Exception as e:
            logger.error(f"login failed: {e}")
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def logout(self, token: str) -> str:
        """Invalidate user session."""
        try:
            svc = self.storage_service
            if svc is None:
                return json.dumps({"error": "Storage unavailable"})
            token_hash = _hash_str(token)
            await svc.delete_session(token_hash)
            return json.dumps({"success": True})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_current_user(self, token: str) -> str:
        """Get current user from session token."""
        try:
            if not token:
                return json.dumps({"error": "Not authenticated"})
            svc = self.storage_service
            if svc is None:
                return json.dumps({"error": "Storage unavailable"})
            token_hash = _hash_str(token)
            user = await svc.validate_session(token_hash)
            if user is None:
                return json.dumps({"error": "Session expired or invalid"})
            return json.dumps({"success": True, "user": dict(user)})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def change_password(self, token: str, old_password: str, new_password: str) -> str:
        """Change password for authenticated user."""
        try:
            svc = self.storage_service
            if svc is None:
                return json.dumps({"error": "Storage unavailable"})
            token_hash = _hash_str(token)
            user = await svc.validate_session(token_hash)
            if user is None:
                return json.dumps({"error": "Not authenticated"})
            full = await svc.get_user_by_id(user["id"])
            if full is None:
                return json.dumps({"error": "User not found"})
            if not _verify_password(old_password, full.get("password_hash", "")):
                return json.dumps({"error": "Current password is incorrect"})
            if len(new_password) < 8:
                # Matches the cloud backend's minimum (cloud/models.py
                # UserRegister.password) -- local auth used to allow 6.
                return json.dumps({"error": "New password must be at least 8 characters"})
            new_hash = _hash_password(new_password)
            await svc.change_password(user["id"], new_hash)
            await svc.audit_log(user["id"], user["username"], "change_password", "auth", "password")
            return json.dumps({"success": True})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def get_audit_log(self, token: str, limit: str = "50") -> str:
        """Get audit log (admin only)."""
        try:
            svc = self.storage_service
            if svc is None:
                return json.dumps({"error": "Storage unavailable"})
            token_hash = _hash_str(token)
            user = await svc.validate_session(token_hash)
            if user is None:
                return json.dumps({"error": "Not authenticated"})
            if user["role"] != "admin":
                return json.dumps({"error": "Admin only"})
            lim = max(1, min(200, int(limit)))
            events = await svc.get_audit_log(limit=lim)
            return json.dumps({"success": True, "events": events})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})

    async def list_users(self, token: str) -> str:
        """List all users (admin only)."""
        try:
            svc = self.storage_service
            if svc is None:
                return json.dumps({"error": "Storage unavailable"})
            token_hash = _hash_str(token)
            user = await svc.validate_session(token_hash)
            if user is None:
                return json.dumps({"error": "Not authenticated"})
            if user["role"] != "admin":
                return json.dumps({"error": "Admin only"})
            users_list = await svc.get_all_users()
            return json.dumps({"success": True, "users": users_list})
        except Exception as e:
            return json.dumps({"error": ErrorSanitizer.sanitize_error(e)})
