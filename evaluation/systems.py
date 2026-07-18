from __future__ import annotations

import time
import uuid
from dataclasses import replace
from typing import Any, Protocol

from app.agents import RagAgents
from app.clients import EmbeddingClient, LlmClient
from app.config import Settings, settings
from app.eval_trace import ranked_hit
from app.local_trace import TraceRecorder
from app.retrieval import VectorStore
from app.service import AgenticRagService, NOT_FOUND_MESSAGE, calculate_deepseek_cost_cny
from evaluation.corpus import DEFAULT_EVAL_COLLECTION


class EvaluationSystem(Protocol):
    name: str

    async def answer(self, case: dict[str, Any]) -> dict[str, Any]: ...


def _runtime(collection: str, base: Settings = settings):
    config = replace(base, qdrant_collection=collection)
    config.validate_query_runtime()
    embedding = EmbeddingClient(config)
    llm = LlmClient(config)
    vectors = VectorStore(config, embedding)
    agents = RagAgents(config, llm)
    service = AgenticRagService(config, agents, vectors)
    return config, vectors, agents, service


class TraditionalRagSystem:
    name = "baseline"

    def __init__(self, collection: str = DEFAULT_EVAL_COLLECTION):
        self.settings, self.vectors, self.agents, _service = _runtime(collection)
        self.collection = collection

    async def answer(self, case: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        trace = TraceRecorder(f"eval-baseline-{uuid.uuid4().hex}")
        hits = await self.vectors.search(case["question"], case["category"])
        contexts = hits[: self.settings.context_top_k]
        event = {
            "stage": "retrieval",
            "round": 1,
            "queries": [case["question"]],
            "query_results": [
                {
                    "query": case["question"],
                    "hits": [ranked_hit(hit, rank) for rank, hit in enumerate(hits, 1)],
                }
            ],
            "fused_hits": [ranked_hit(hit, rank) for rank, hit in enumerate(hits, 1)],
        }
        if contexts:
            answer = await self.agents.generate(case["question"], contexts, trace)
            found = True
        else:
            answer = NOT_FOUND_MESSAGE
            found = False
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "system": self.name,
            "collection": self.collection,
            "found": found,
            "answer": answer,
            "sources": AgenticRagService._sources(contexts),
            "trace_id": trace.trace_id,
            "trace": [],
            "evaluation_events": [
                event,
                {
                    "stage": "evidence_grading",
                    "round": 1,
                    "sufficient": found,
                    "missing_aspects": [],
                    "accepted_hits": [
                        ranked_hit(hit, rank, accepted=True)
                        for rank, hit in enumerate(contexts, 1)
                    ],
                    "fused_hits": event["fused_hits"],
                },
            ],
            "meta": {
                "retrieval_rounds": 1,
                "max_retrieval_rounds": 1,
                "retrieved_chunks": len(hits),
                "accepted_chunks": len(contexts),
                "rewritten": False,
                "verified": False,
                "latency_ms": latency_ms,
                "input_tokens": trace.input_tokens,
                "output_tokens": trace.output_tokens,
                "cache_hit_input_tokens": trace.cache_hit_input_tokens,
                "cache_miss_input_tokens": trace.cache_miss_input_tokens,
                "cache_usage_reported": trace.cache_usage_reported,
                "estimated_cost_cny": calculate_deepseek_cost_cny(
                    self.settings,
                    cache_hit_input_tokens=trace.cache_hit_input_tokens,
                    cache_miss_input_tokens=trace.cache_miss_input_tokens,
                    output_tokens=trace.output_tokens,
                ),
                "pricing_model": self.settings.llm_model,
                "pricing_currency": "CNY",
            },
        }


class AgenticEvaluationSystem:
    def __init__(
        self,
        *,
        name: str = "agentic",
        collection: str = DEFAULT_EVAL_COLLECTION,
    ):
        self.name = name
        self.settings, _vectors, _agents, self.service = _runtime(collection)
        self.collection = collection

    async def answer(self, case: dict[str, Any]) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        result = await self.service.answer(
            query=case["question"],
            category=case["category"],
            session_id=f"eval-{case['case_id']}",
            request_id=f"eval-{self.name}-{uuid.uuid4().hex}",
            evaluation_sink=events.append,
        )
        return {
            "system": self.name,
            "collection": self.collection,
            **result,
            "evaluation_events": events,
        }


def create_systems(
    names: list[str],
    *,
    eval_collection: str = DEFAULT_EVAL_COLLECTION,
) -> dict[str, EvaluationSystem]:
    result: dict[str, EvaluationSystem] = {}
    for name in names:
        if name == "baseline":
            result[name] = TraditionalRagSystem(collection=eval_collection)
        elif name == "agentic":
            result[name] = AgenticEvaluationSystem(collection=eval_collection)
        elif name == "production":
            result[name] = AgenticEvaluationSystem(
                name="production",
                collection=settings.qdrant_collection,
            )
        else:
            raise ValueError(f"未知评测系统: {name}")
    return result
