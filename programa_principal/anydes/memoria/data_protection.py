from __future__ import annotations

import base64
import ctypes
import os
from ctypes import wintypes
from typing import Protocol


class DataProtectionError(RuntimeError):
    """Raised when sensitive local data cannot be protected or recovered."""


class DataProtector(Protocol):
    scheme: str

    def protect(self, plaintext: bytes) -> str: ...

    def unprotect(self, token: str) -> bytes: ...


class WindowsDpapiProtector:
    """Protect bytes for the current Windows user with the operating system DPAPI."""

    scheme = "windows-dpapi-current-user-v1"
    _entropy = b"Anydes conversation archive v2"
    _ui_forbidden = 0x1

    def __init__(self) -> None:
        if os.name != "nt":
            raise DataProtectionError("Windows DPAPI is unavailable on this platform.")
        self._crypt32 = ctypes.WinDLL("Crypt32.dll", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)

    def protect(self, plaintext: bytes) -> str:
        protected = self._transform("CryptProtectData", plaintext)
        return base64.b64encode(protected).decode("ascii")

    def unprotect(self, token: str) -> bytes:
        try:
            protected = base64.b64decode(token, validate=True)
        except (ValueError, TypeError) as exc:
            raise DataProtectionError("Protected data has an invalid encoding.") from exc
        return self._transform("CryptUnprotectData", protected)

    def _transform(self, function_name: str, payload: bytes) -> bytes:
        payload_buffer, payload_blob = _blob_from_bytes(payload)
        entropy_buffer, entropy_blob = _blob_from_bytes(self._entropy)
        output_blob = _DataBlob()
        function = getattr(self._crypt32, function_name)
        if function_name == "CryptProtectData":
            ok = function(
                ctypes.byref(payload_blob),
                None,
                ctypes.byref(entropy_blob),
                None,
                None,
                self._ui_forbidden,
                ctypes.byref(output_blob),
            )
        else:
            description = wintypes.LPWSTR()
            ok = function(
                ctypes.byref(payload_blob),
                ctypes.byref(description),
                ctypes.byref(entropy_blob),
                None,
                None,
                self._ui_forbidden,
                ctypes.byref(output_blob),
            )
            if description:
                self._kernel32.LocalFree(description)
        _ = (payload_buffer, entropy_buffer)
        if not ok:
            error = ctypes.get_last_error()
            raise DataProtectionError(f"Windows DPAPI failed with system error {error}.")
        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData)
        finally:
            if output_blob.pbData:
                self._kernel32.LocalFree(output_blob.pbData)


class PassphraseAesGcmProtector:
    """Authenticated, process-independent protection derived from a passphrase."""

    scheme = "aes-256-gcm-pbkdf2-sha256-v1"
    _salt_bytes = 16
    _nonce_bytes = 12
    _iterations = 600_000
    _associated_data = b"Anydes protected local data v1"

    def __init__(self, passphrase: str) -> None:
        if not passphrase:
            raise DataProtectionError("AES-GCM protection requires a passphrase.")
        self._passphrase = passphrase.encode("utf-8")
        self._write_salt = os.urandom(self._salt_bytes)
        self._derived_keys: dict[bytes, bytes] = {}

    def protect(self, plaintext: bytes) -> str:
        AESGCM, hashes, PBKDF2HMAC = _cryptography_primitives()
        salt = self._write_salt
        nonce = os.urandom(self._nonce_bytes)
        key = self._key_for(salt, hashes, PBKDF2HMAC)
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, self._associated_data)
        return base64.b64encode(salt + nonce + ciphertext).decode("ascii")

    def unprotect(self, token: str) -> bytes:
        try:
            payload = base64.b64decode(token, validate=True)
        except (ValueError, TypeError) as exc:
            raise DataProtectionError("Protected data has an invalid encoding.") from exc
        minimum = self._salt_bytes + self._nonce_bytes + 16
        if len(payload) < minimum:
            raise DataProtectionError("Protected AES-GCM data is truncated.")
        salt = payload[: self._salt_bytes]
        nonce = payload[self._salt_bytes : self._salt_bytes + self._nonce_bytes]
        ciphertext = payload[self._salt_bytes + self._nonce_bytes :]
        AESGCM, hashes, PBKDF2HMAC = _cryptography_primitives()
        key = self._key_for(salt, hashes, PBKDF2HMAC)
        try:
            return AESGCM(key).decrypt(nonce, ciphertext, self._associated_data)
        except Exception as exc:
            raise DataProtectionError(
                "AES-GCM authentication failed for protected local data."
            ) from exc

    def _key_for(self, salt: bytes, hashes, PBKDF2HMAC) -> bytes:
        cached = self._derived_keys.get(salt)
        if cached is not None:
            return cached
        key = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self._iterations,
        ).derive(self._passphrase)
        self._derived_keys[salt] = key
        return key


def default_data_protector(*, passphrase: str | None = None) -> DataProtector:
    """Use portable authenticated encryption when explicitly configured."""

    if passphrase:
        return PassphraseAesGcmProtector(passphrase)

    return WindowsDpapiProtector()


def _cryptography_primitives():
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError as exc:
        raise DataProtectionError(
            "The cryptography package is required for passphrase protection."
        ) from exc
    return AESGCM, hashes, PBKDF2HMAC


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _blob_from_bytes(data: bytes) -> tuple[ctypes.Array[ctypes.c_char], _DataBlob]:
    buffer = ctypes.create_string_buffer(data, len(data))
    blob = _DataBlob(
        len(data),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)),
    )
    return buffer, blob
