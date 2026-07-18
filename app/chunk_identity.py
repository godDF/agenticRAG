from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping


def normalize_chunk_text(value: Any) -> str:
    """Return a stable representation used by evaluation and indexing."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def build_chunk_uid(
    *,
    category: Any,
    title: Any,
    source_url: Any,
    content: Any,
) -> str:
    canonical = "|".join(
        normalize_chunk_text(item)
        for item in (category, title, source_url, content)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def chunk_uid_from_payload(payload: Mapping[str, Any]) -> str:
    existing = normalize_chunk_text(payload.get("chunk_uid"))
    if existing:
        return existing
    return build_chunk_uid(
        category=payload.get("category"),
        title=payload.get("title"),
        source_url=payload.get("source_url"),
        content=payload.get("content"),
    )
