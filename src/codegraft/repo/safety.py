"""Safety filtering: never let secrets, binaries, or noise into model context.

Git-native ignore rules are about *version control*, not *what is safe to send
to an LLM*. So on top of discovery we apply a hard denylist. The posture is
local-first, least-context: when unsure, exclude.

Patterns use gitignore semantics, so a slash-free pattern like ``*.pem``
matches at any depth (just like ``.gitignore``).
"""

from __future__ import annotations

from pathlib import Path

from pathspec import PathSpec

# Credentials, keys, certs, and environment files. These must never be sent.
_SECRET_PATTERNS = [
    ".env",
    ".env.*",
    # The suffix form is just as common as the dotfile form and was slipping
    # through: "production.env", "local.env" matched neither pattern above.
    "*.env",
    "*.pem",
    "*.key",
    "*.crt",
    "*.cer",
    "*.der",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "*.jks",
    "*.ppk",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "*.pkcs12",
    "*.p8",           # Apple / APNs auth keys
    "*.asc",          # ASCII-armoured PGP key material
    "*.gpg",
    "credentials",
    "credentials.*",  # e.g. GCP service-account "credentials.json"
    ".netrc",
    "_netrc",         # the Windows spelling
    ".git-credentials",
    "*.tfvars",       # Terraform variable files routinely hold secrets
    "*.tfvars.json",
    ".npmrc",
    ".pypirc",
    "*.secret",
    "secrets.*",
]

# Generated / minified / noise that adds tokens but little planning signal.
_GENERATED_PATTERNS = [
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",  # lockfiles are noisy; manifests carry the real signal
    "*.snap",
    "*.bundle.js",
    "*.bundle.css",
]

_SECRET_SPEC = PathSpec.from_lines("gitignore", _SECRET_PATTERNS)
_GENERATED_SPEC = PathSpec.from_lines("gitignore", _GENERATED_PATTERNS)

_BINARY_SNIFF_BYTES = 8192


def is_binary(path: Path) -> bool:
    """Heuristic: a NUL byte in the first chunk means binary (git's own test)."""

    try:
        with path.open("rb") as fh:
            chunk = fh.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return False
    return b"\0" in chunk


def classify(rel_path: str, abs_path: Path, size_bytes: int, max_file_bytes: int) -> str | None:
    """Return a skip reason for a candidate file, or ``None`` if it is safe.

    Checks are ordered cheapest-first: path-pattern matches (no I/O) before the
    binary sniff (a small read).
    """

    if _SECRET_SPEC.match_file(rel_path):
        return "secret"
    if size_bytes > max_file_bytes:
        return "oversized"
    if _GENERATED_SPEC.match_file(rel_path):
        return "generated"
    if is_binary(abs_path):
        return "binary"
    return None
