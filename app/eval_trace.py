from __future__ import annotations

import copy
import inspect
from typing import Any, Awaitable, Callable

from app.chunk_identity import chunk_uid_from_payload


EvaluationEventSink = Callable[[dict[str, Any]], Awaitable[None] | None]


def ranked_hit(hit: dict[str, Any], rank: int, *, accepted: bool = False) -> dict[str, Any]:
    return {
        "point_id": str(hit.get("point_id", "")),
        "chunk_uid": chunk_uid_from_payload(hit),
        "document_id": str(hit.get("document_id", "")),
        "category": str(hit.get("category", "")),
        "title": str(hit.get("title", "")),
        "source_url": str(hit.get("source_url", "")),
        "file_path": str(hit.get("file_path", "")),
        "chunk_index": int(hit.get("chunk_index", 0) or 0),
        "rank": rank,
        "vector_score": round(float(hit.get("score", 0) or 0), 8),
        "rrf_score": round(float(hit.get("rrf_score", 0) or 0), 8),
        "accepted": accepted,
        "content": str(hit.get("content", "")),
    }


async def emit_evaluation_event(
    sink: EvaluationEventSink | None,
    event: dict[str, Any],
) -> None:
    if sink is None:
        return
    try:
        result = sink(copy.deepcopy(event))
        if inspect.isawaitable(result):
            await result
    except Exception:
        # Evaluation observability must not change the RAG answer path.
        return
