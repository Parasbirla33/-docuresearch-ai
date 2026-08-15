"""ID generation and content hashing helpers."""

from __future__ import annotations

import hashlib
import uuid
from urllib.parse import urlsplit, urlunsplit


def new_id(prefix: str) -> str:
    """Generate a short, prefixed unique id, e.g. `new_id("S")` -> `S-3f9a1c2b`."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def content_hash(text: str) -> str:
    """Stable SHA-256 hash of text content, used for dedup and cache keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_url(url: str) -> str:
    """Normalize a URL for deduplication: lowercase scheme/host, strip fragment/trailing slash."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))
