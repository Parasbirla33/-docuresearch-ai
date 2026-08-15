"""File-based research cache: avoid re-downloading URLs already collected.

Phase 2 keeps this filesystem-backed and dependency-free. A DB-backed cache
(Phase 7, SQLAlchemy) can replace the storage backend later without changing
this interface - callers only use `get`/`set`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from docuresearch.utils.hashing import content_hash, normalize_url
from docuresearch.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TTL = timedelta(days=7)


@dataclass
class CacheEntry:
    url: str
    content_hash: str
    retrieved_at: str
    title: str | None
    content: str
    metadata: dict[str, Any]


class ResearchCache:
    """Content-hash-keyed cache of previously fetched URLs, with TTL-based expiry."""

    def __init__(self, cache_dir: Path, *, ttl: timedelta = DEFAULT_TTL) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl

    def _path_for(self, url: str) -> Path:
        key = content_hash(normalize_url(url))
        return self._dir / f"{key}.json"

    def get(self, url: str) -> CacheEntry | None:
        """Return the cached entry for `url`, or None if absent/expired/corrupt."""
        path = self._path_for(url)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry = CacheEntry(**data)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("cache_entry_corrupted", url=url, error=str(exc))
            return None

        retrieved_at = datetime.fromisoformat(entry.retrieved_at)
        if datetime.now(UTC) - retrieved_at > self._ttl:
            logger.info("cache_entry_expired", url=url)
            return None
        return entry

    def set(
        self, url: str, *, title: str | None, content: str, metadata: dict[str, Any] | None = None
    ) -> CacheEntry:
        """Store/overwrite the cache entry for `url`."""
        entry = CacheEntry(
            url=url,
            content_hash=content_hash(content),
            retrieved_at=datetime.now(UTC).isoformat(),
            title=title,
            content=content,
            metadata=metadata or {},
        )
        self._path_for(url).write_text(json.dumps(asdict(entry)), encoding="utf-8")
        return entry

    def invalidate(self, url: str) -> None:
        """Remove a cached entry, forcing the next `get` to miss."""
        path = self._path_for(url)
        if path.exists():
            path.unlink()
