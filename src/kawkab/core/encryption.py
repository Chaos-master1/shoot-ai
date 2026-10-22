"""Data-at-rest encryption for sensitive fields (medical, PII).

Uses Fernet symmetric encryption with a key stored in the OS keychain
(via keyring), falling back to a file-based key for environments
without keychain support.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import keyring
import keyring.errors
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from kawkab.core.logging import get_logger

logger = get_logger(__name__)

_fernet: Fernet | None = None
_FALLBACK_KEY_FILE = Path.home() / ".kawkab" / ".medical_key"
_KEYRING_SERVICE = "kawkab-medical"
_KEYRING_USER = "encryption-key"


def _derive_key(raw: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=b"kawkab-medical-v1", iterations=600_000
    )
    return base64.urlsafe_b64encode(kdf.derive(raw))


def _load_or_create_key() -> bytes:
    """Try OS keychain first, fall back to file-based key."""
    # Attempt OS keychain
    try:
        stored = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER)
        if stored:
            logger.info("Using key from OS keychain")
            return _derive_key(bytes.fromhex(stored))
        raw = os.urandom(32)
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, raw.hex())
        logger.info("Generated and stored new encryption key in OS keychain")
        return _derive_key(raw)
    except Exception as exc:
        logger.warning(f"OS keychain unavailable ({exc}), falling back to file-based key")

    # Fallback to file-based key
    if _FALLBACK_KEY_FILE.exists():
        raw = _FALLBACK_KEY_FILE.read_bytes()
        logger.info("Using key from fallback file")
    else:
        raw = os.urandom(32)
        _FALLBACK_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _FALLBACK_KEY_FILE.write_bytes(raw)
        if os.name != "nt":
            _FALLBACK_KEY_FILE.chmod(0o600)
        logger.info("Generated and stored new encryption key in fallback file")
    return _derive_key(raw)


def init_fernet(key_value: str | None = None) -> None:
    global _fernet
    if key_value:
        raw = bytes.fromhex(key_value)
        key = _derive_key(raw)
    else:
        key = _load_or_create_key()
    _fernet = Fernet(key)


def get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        init_fernet()
    assert _fernet is not None
    return _fernet


def encrypt(plaintext: str) -> str:
    return get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    return get_fernet().decrypt(ciphertext.encode()).decode()


def encrypt_dict(data: dict, fields: list[str], in_place: bool = True) -> dict:
    result = data if in_place else dict(data)
    for field in fields:
        val = result.get(field)
        if val and isinstance(val, str):
            result[field] = encrypt(val)
    return result


def decrypt_dict(data: dict, fields: list[str], in_place: bool = True) -> dict:
    result = data if in_place else dict(data)
    for field in fields:
        val = result.get(field)
        if val and isinstance(val, str):
            try:
                result[field] = decrypt(val)
            except Exception as exc:
                # A failed decrypt on a medical/PII field must never be
                # silent: with a wrong or rotated key, "ciphertext" would
                # otherwise be indistinguishable from "no data" downstream.
                logger.warning(
                    f"decrypt failed for field '{field}' "
                    f"(len={len(val)}): {type(exc).__name__}: {exc}"
                )
    return result
