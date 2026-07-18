from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient, models

from app.clients import EmbeddingClient
from app.chunk_identity import chunk_uid_from_payload
from app.config import Settings
from app.observability import trace_tool

if TYPE_CHECKING:
    from app.ingestion import IngestionReporter


class VectorStore:
    def __init__(self, settings: Settings, embedding: EmbeddingClient):
        self.settings = settings
        self.embedding = embedding
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=10,
            check_compatibility=False,
            trust_env=False,
        )

    def _ensure_collection_sync(self) -> None:
        name = self.settings.qdrant_collection
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.settings.embedding_vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        collection = self.client.get_collection(name)
        existing = set((collection.payload_schema or {}).keys())
        for field_name in ("category", "document_id"):
            if field_name not in existing:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

    async def ensure_collection(self) -> None:
        await asyncio.to_thread(self._ensure_collection_sync)

    @trace_tool(name="qdrant_retrieval", tool_type="vector_database")
    async def search(self, query: str, category: str) -> list[dict[str, Any]]:
        vector = (await self.embedding.embed([query]))[0]

        def run():
            return self.client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="category", match=models.MatchValue(value=category)
                        )
                    ]
                ),
                limit=self.settings.retrieve_top_k,
                with_payload=True,
                score_threshold=self.settings.retrieve_threshold,
            )

        response = await asyncio.to_thread(run)
        hits = []
        for point in response.points:
            hit = {
                "point_id": str(point.id),
                "score": float(point.score),
                **(point.payload or {}),
            }
            hit["chunk_uid"] = chunk_uid_from_payload(hit)
            hits.append(hit)
        return hits

    @trace_tool(name="qdrant_index", tool_type="vector_database")
    async def upsert_document(
        self,
        document_id: str,
        chunks: list[dict[str, Any]],
        reporter: "IngestionReporter | None" = None,
    ) -> int:
        event_id = None
        if reporter:
            event_id = await reporter.start(
                kind="tool",
                name="qdrant_prepare_collection",
                label="检查向量库与索引",
                progress=50,
                details={"collection": self.settings.qdrant_collection},
            )
        await self.ensure_collection()
        if reporter and event_id:
            await reporter.finish(event_id, progress=55)

        event_id = None
        if reporter:
            event_id = await reporter.start(
                kind="tool",
                name="bge_m3_embedding",
                label="调用 BGE-M3 生成向量",
                progress=56,
                details={
                    "model": self.settings.embedding_model,
                    "chunk_count": len(chunks),
                    "local_model": False,
                },
            )
        vectors, usage = await self.embedding.embed_with_usage(
            [chunk["content"] for chunk in chunks]
        )
        embedding_tokens = int(usage.get("total_tokens", 0))
        cost = (
            embedding_tokens
            * self.settings.embedding_cost_per_million_cny
            / 1_000_000
        )
        if reporter and event_id:
            await reporter.finish(
                event_id,
                progress=82,
                details={
                    "vector_count": len(vectors),
                    "vector_size": len(vectors[0]) if vectors else 0,
                    "provider_reported_tokens": embedding_tokens,
                    "price_per_million_tokens_cny": self.settings.embedding_cost_per_million_cny,
                    "pricing_source": self.settings.embedding_pricing_source,
                    "pricing_note": (
                        "当前模型官方免费"
                        if self.settings.embedding_cost_per_million_cny == 0
                        else "按输入 Token 计费"
                    ),
                },
                metrics={
                    "embedding_tokens": embedding_tokens,
                    "estimated_cost_cny": cost,
                    "cost_configured": self.settings.embedding_pricing_configured,
                    "pricing_free": (
                        self.settings.embedding_pricing_configured
                        and self.settings.embedding_cost_per_million_cny == 0
                    ),
                    "pricing_source": self.settings.embedding_pricing_source,
                    "pricing_model": self.settings.embedding_model,
                },
            )
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{index}"))
            points.append(models.PointStruct(id=point_id, vector=vector, payload=chunk))
        event_id = None
        if reporter:
            event_id = await reporter.start(
                kind="tool",
                name="qdrant_upsert",
                label="写入 Qdrant 向量库",
                progress=84,
                details={"point_count": len(points)},
            )
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.settings.qdrant_collection,
            points=points,
            wait=True,
        )
        if reporter and event_id:
            await reporter.finish(event_id, progress=93)
        return len(points)

    async def delete_document(self, document_id: str) -> None:
        if not await asyncio.to_thread(
            self.client.collection_exists, self.settings.qdrant_collection
        ):
            return
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
        )
        await asyncio.to_thread(
            self.client.delete,
            collection_name=self.settings.qdrant_collection,
            points_selector=selector,
            wait=True,
        )

    async def ready(self) -> tuple[bool, str]:
        try:
            await asyncio.to_thread(self.client.get_collections)
            return True, "ok"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"


def reciprocal_rank_fusion(result_sets: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, dict[str, Any]] = {}
    for results in result_sets:
        for rank, hit in enumerate(results, start=1):
            key = str(hit.get("point_id") or f"{hit.get('file_path')}:{hit.get('chunk_index')}")
            scores[key] += 1.0 / (k + rank)
            if key not in best or float(hit.get("score", 0)) > float(best[key].get("score", 0)):
                best[key] = dict(hit)
    fused = []
    for key, hit in best.items():
        hit["rrf_score"] = scores[key]
        fused.append(hit)
    return sorted(fused, key=lambda item: (item["rrf_score"], item.get("score", 0)), reverse=True)
