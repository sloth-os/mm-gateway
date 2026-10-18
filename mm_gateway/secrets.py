"""Authenticated encryption for credentials in YAML and management overlays."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4

from cryptography.fernet import Fernet, InvalidToken

ENCRYPTED_PREFIX = "enc:v1:"
ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")
_SECRET = re.compile(
    r"(^key$|auth|api.?key|token|secret|password|passwd|passphrase|cookie|credential|"
    r"private.?key|access.?key|subscription.?key|^user$|username|user.?name|login|client.?id|"
    r"service.?account|signature|^sig$|^jwt$|client.?certificate)",
    re.I,
)


class SecretError(ValueError):
    """A credential could not be protected or authenticated; never includes its value."""


def is_secret(value: Any, name: str = "", *, header: bool = False) -> bool:
    """Share the same credential boundaries between persistence and redaction."""
    if header or (_SECRET.search(name) and name not in {"keys", "credentials"}):
        return True
    if name == "credentials" and isinstance(value, str):  # YAML account shorthand
        return True
    if isinstance(value, str) and "://" in value:
        try:
            parsed = urlsplit(value)
            return bool(parsed.username or parsed.password or
                        any(_SECRET.search(key) for key, _ in parse_qsl(parsed.query, keep_blank_values=True)))
        except ValueError:
            return True
    return False


def is_header_map(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return normalized == "headers" or normalized.endswith("_headers")


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: str | Path, text: str) -> None:
    """Replace a private file only after its complete contents are durable."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class SecretStore:
    def __init__(self, config_path: str | Path):
        self.key_path = Path(os.environ.get("MM_GATEWAY_SECRET_KEY_FILE") or
                             Path(config_path).resolve().parent / ".mm-gateway.key")
        self._cipher: Fernet | None = None

    def _fernet(self, *, create: bool) -> Fernet:
        if self._cipher is not None:
            return self._cipher
        if create and not self.key_path.exists():
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".mm-gateway-key-", dir=self.key_path.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(Fernet.generate_key() + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    # Publish a complete key without replacing a concurrent winner.
                    os.link(temporary, self.key_path)
                    _sync_directory(self.key_path.parent)
                except FileExistsError:
                    pass
            finally:
                os.unlink(temporary)
        try:
            key = self.key_path.read_bytes().strip()
        except FileNotFoundError:
            raise SecretError("Encrypted configuration requires its existing key file. "
                              "Restore .mm-gateway.key or set MM_GATEWAY_SECRET_KEY_FILE.") from None
        try:
            self._cipher = Fernet(key)
        except (ValueError, TypeError):
            raise SecretError("The configuration encryption key file is invalid.") from None
        return self._cipher

    def _seal(self, value: Any) -> str:
        payload = json.dumps(value, allow_nan=False).encode("utf-8")
        return ENCRYPTED_PREFIX + self._fernet(create=True).encrypt(payload).decode("ascii")

    def _open(self, value: str) -> Any:
        if not value.startswith(ENCRYPTED_PREFIX):
            raise SecretError("Unsupported encrypted configuration version.")
        cipher = self._fernet(create=False)
        try:
            return json.loads(cipher.decrypt(value[len(ENCRYPTED_PREFIX):].encode("ascii")))
        except (InvalidToken, ValueError, UnicodeError):
            raise SecretError("Could not decrypt configuration credentials. "
                              "Restore the matching key file and unmodified configuration.") from None

    def encrypt(self, value: Any, name: str = "", *, header: bool = False,
                preserve_env: bool = False) -> Any:
        if value is None or value == "":
            return value
        if isinstance(value, str) and value.startswith("enc:"):
            self._open(value)  # Never double-encrypt, or silently accept corrupt ciphertext.
            return value
        if is_secret(value, name, header=header):
            reference = ENV_PATTERN.fullmatch(value) if isinstance(value, str) else None
            if preserve_env and reference and reference.group(2) in (None, ""):
                return value  # Keep references, without persisting resolved environment secrets.
            return self._seal(self.decrypt(value))
        if isinstance(value, dict):
            return {k: self.encrypt(v, str(k), header=is_header_map(name), preserve_env=preserve_env)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [self.encrypt(v, name, preserve_env=preserve_env) for v in value]
        return value

    def decrypt(self, value: Any, *, interpolate: Callable[[str], str] | None = None) -> Any:
        def resolve(plain: Any) -> Any:
            if isinstance(plain, str) and interpolate:
                # Only encrypted templates need interpolation here. Ordinary YAML
                # values have already gone through the existing text interpolator.
                resolved = interpolate(plain)
                return None if ENV_PATTERN.fullmatch(plain) and resolved == "" else resolved
            if isinstance(plain, dict):
                return {k: resolve(v) for k, v in plain.items()}
            if isinstance(plain, list):
                return [resolve(v) for v in plain]
            return plain

        if isinstance(value, str) and value.startswith("enc:"):
            return resolve(self._open(value))
        if isinstance(value, dict):
            return {k: self.decrypt(v, interpolate=interpolate) for k, v in value.items()}
        if isinstance(value, list):
            return [self.decrypt(v, interpolate=interpolate) for v in value]
        return value

    def protect_yaml(self, path: str | Path, text: str) -> None:
        import yaml

        # Mask templates before parsing: an unquoted ${ENV} is legal in the
        # legacy text interpolator, including inside YAML flow collections.
        references: dict[str, str] = {}
        def mask(match: re.Match) -> str:
            # Equal-length masks keep YAML source offsets usable for migration.
            size = len(match.group(0))
            marker = "_" + (uuid4().hex * (size // 32 + 1))[:size - 1]
            while marker in text or marker in references:
                marker = "_" + (uuid4().hex * (size // 32 + 1))[:size - 1]
            references[marker] = match.group(0)
            return marker

        masked = ENV_PATTERN.sub(mask, text)
        raw = yaml.safe_load(masked)
        def restore(value: Any) -> Any:
            if isinstance(value, str):
                for marker, reference in references.items():
                    value = value.replace(marker, reference)
                return value
            if isinstance(value, dict):
                return {restore(k): restore(v) for k, v in value.items()}
            if isinstance(value, list):
                return [restore(v) for v in value]
            return value

        raw = restore(raw)
        encrypted = self.encrypt(raw, preserve_env=True)
        if encrypted != raw:
            tokens = list(yaml.scan(masked))
            if any(isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken, yaml.tokens.TagToken))
                   for token in tokens):
                # Aliases can share a credential with an unrelated field. A full
                # serialization safely updates every occurrence and its YAML type.
                secret_objects: set[int] = set()
                secret_strings: set[str] = set()
                def remember(value):
                    secret_objects.add(id(value))
                    if isinstance(value, str):
                        secret_strings.add(value)
                    elif isinstance(value, dict):
                        for child in value.values():
                            remember(child)
                    elif isinstance(value, list):
                        for child in value:
                            remember(child)

                def find(before, after):
                    if before != after and isinstance(after, str) and after.startswith(ENCRYPTED_PREFIX):
                        remember(before)
                    elif isinstance(before, dict) and isinstance(after, dict):
                        for key, child in before.items():
                            find(child, after[key])
                    elif isinstance(before, list) and isinstance(after, list):
                        for old, new in zip(before, after):
                            find(old, new)

                def protect_aliases(before, after):
                    shared_secret = id(before) in secret_objects or (isinstance(before, str) and before in secret_strings)
                    if shared_secret and before is not None and before != "":
                        return self._seal(self.decrypt(before))
                    if isinstance(before, dict) and isinstance(after, dict):
                        return {key: protect_aliases(child, after[key]) for key, child in before.items()}
                    if isinstance(before, list) and isinstance(after, list):
                        return [protect_aliases(old, new) for old, new in zip(before, after)]
                    return after

                find(raw, encrypted)
                encrypted = protect_aliases(raw, encrypted)
                protected = yaml.safe_dump(encrypted, sort_keys=False, allow_unicode=True)
            else:
                replacements: list[tuple[int, int, str]] = []
                constructor = yaml.SafeLoader("")
                def replace(node, before, after):
                    if before == after:
                        return
                    if isinstance(after, str) and after.startswith(ENCRYPTED_PREFIX):
                        start, end = node.start_mark.index, node.end_mark.index
                        trailing = re.search(r"\s*$", text[start:end]).group(0)
                        replacements.append((start, end, json.dumps(after) + trailing))
                    elif isinstance(node, yaml.MappingNode):
                        for key_node, child in node.value:
                            key = restore(constructor.construct_object(key_node, deep=True))
                            replace(child, before[key], after[key])
                    elif isinstance(node, yaml.SequenceNode):
                        for child, old, new in zip(node.value, before, after):
                            replace(child, old, new)

                replace(yaml.compose(masked), raw, encrypted)
                constructor.dispose()
                protected = text
                for start, end, replacement in sorted(replacements, reverse=True):
                    protected = protected[:start] + replacement + protected[end:]
                # Validate the edited document before replacing the original.
                if restore(yaml.safe_load(ENV_PATTERN.sub(mask, protected))) != encrypted:
                    raise SecretError("Could not safely rewrite YAML credentials.")
            atomic_write(path, protected)
