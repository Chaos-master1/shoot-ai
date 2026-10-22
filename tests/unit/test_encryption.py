"""Tests for the encryption module — medical/PII data-at-rest encryption."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from kawkab.core.encryption import (
    _derive_key,
    decrypt,
    decrypt_dict,
    encrypt,
    encrypt_dict,
    get_fernet,
    init_fernet,
)


@pytest.fixture(autouse=True)
def _reset_fernet():
    """Reset global Fernet instance and override key file path before each test."""
    import kawkab.core.encryption as enc_mod

    enc_mod._fernet = None
    # Redirect fallback key to temp dir
    tmp = tempfile.mkdtemp()
    enc_mod._FALLBACK_KEY_FILE = Path(tmp) / ".medical_key"
    yield
    enc_mod._fernet = None
    # Cleanup
    for f in Path(tmp).iterdir():
        f.unlink()
    os.rmdir(tmp)


def test_init_fernet_with_key():
    """init_fernet with explicit hex key produces a working Fernet instance."""
    init_fernet("aabb" * 16)  # 32 bytes hex
    f = get_fernet()
    assert isinstance(f, Fernet)


def test_encrypt_decrypt_roundtrip():
    """encrypt and decrypt round-trips correctly."""
    init_fernet("c0ffee" * 16)
    plain = "sensitive medical note"
    cipher = encrypt(plain)
    assert cipher != plain
    assert decrypt(cipher) == plain


def test_decrypt_invalid_ciphertext_raises():
    """decrypt raises on garbage ciphertext."""
    init_fernet("deadbeef" * 16)
    from cryptography.fernet import InvalidToken

    with pytest.raises(InvalidToken):
        decrypt("not-valid-ciphertext!!")


def test_encrypt_empty_string():
    """encrypt of empty string works."""
    init_fernet("ba5eba11" * 16)
    cipher = encrypt("")
    assert isinstance(cipher, str)
    assert len(cipher) > 0
    assert decrypt(cipher) == ""


def test_encrypt_dict_in_place():
    """encrypt_dict encrypts specified fields in place."""
    init_fernet("ca1fca1f" * 16)
    data = {"name": "John", "notes": "ACL recovery", "age": 25}
    result = encrypt_dict(data, ["notes"])
    assert result is data
    assert result["notes"] != "ACL recovery"
    assert result["name"] == "John"
    assert result["age"] == 25


def test_encrypt_dict_not_in_place():
    """encrypt_dict with in_place=False returns a copy."""
    init_fernet("ca1fca1f" * 16)
    data = {"notes": "concussion follow-up"}
    result = encrypt_dict(data, ["notes"], in_place=False)
    assert result is not data
    assert data["notes"] == "concussion follow-up"
    assert result["notes"] != data["notes"]


def test_encrypt_dict_skips_missing_field():
    """encrypt_dict skips fields not present in data."""
    init_fernet("ca1fca1f" * 16)
    data = {"name": "Test"}
    result = encrypt_dict(data, ["notes"])
    assert result == data


def test_encrypt_dict_skips_non_string():
    """encrypt_dict skips non-string values."""
    init_fernet("ca1fca1f" * 16)
    data = {"value": 42}
    result = encrypt_dict(data, ["value"])
    assert result["value"] == 42


def test_decrypt_dict_roundtrip():
    """encrypt_dict then decrypt_dict recovers original."""
    init_fernet("ca1fca1f" * 16)
    original = {"name": "Ali", "diagnosis": "concussion", "age": 28}
    enc = encrypt_dict(dict(original), ["diagnosis"])
    dec = decrypt_dict(enc, ["diagnosis"])
    assert dec["diagnosis"] == original["diagnosis"]
    assert dec["name"] == original["name"]
    assert dec["age"] == original["age"]


def test_decrypt_dict_ignores_unencrypted():
    """decrypt_dict passes through unencrypted fields unchanged."""
    init_fernet("ca1fca1f" * 16)
    data = {"name": "plain text"}
    result = decrypt_dict(data, ["name"])
    assert result["name"] == "plain text"


def test_key_derivation_deterministic():
    """_derive_key produces same output for same input."""
    k1 = _derive_key(b"same-secret-32-bytes-long!!!!!")
    k2 = _derive_key(b"same-secret-32-bytes-long!!!!!")
    assert k1 == k2


def test_key_derivation_different():
    """_derive_key produces different output for different input."""
    k1 = _derive_key(b"secret-one-here-00000000000000")
    k2 = _derive_key(b"secret-two-here-00000000000000")
    assert k1 != k2


def test_key_derivation_length():
    """_derive_key returns valid base64 Fernet key (44 chars)."""
    key = _derive_key(os.urandom(32))
    assert len(key) == 44
    assert key.endswith(b"=")


def test_encrypt_decrypt_different_keys():
    """decrypt with wrong key raises exception."""
    init_fernet("aaaa" * 16)
    cipher = encrypt("secret data")
    init_fernet("bbbb" * 16)
    from cryptography.fernet import InvalidToken

    with pytest.raises(InvalidToken):
        decrypt(cipher)


def test_encrypt_long_text():
    """encrypt handles long text (10KB)."""
    init_fernet("c0ffee" * 16)
    plain = "A" * 10_000
    cipher = encrypt(plain)
    assert decrypt(cipher) == plain
