from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from app.agents import RagAgents
from app.config import Settings
from app.eval_trace import EvaluationEventSink, emit_evaluation_event, ranked_hit
from app.local_trace import TraceEventSink, TraceRecorder, trace_store
from app.observability import trace_agent
from app.retrieval import VectorStore, reciprocal_rank_fusion


NOT_FOUND_MESSAGE = "知识库中暂未查询到可靠信息，请换一种描述后重试。"


def calculate_deepseek_cost_cny(
    settings: Settings,
    *,
    cache_hit_input_tokens: int,
    cache_miss_input_tokens: int,
    output_tokens: int,
) -> float:
    return round(
        (
            cache_hit_input_tokens * settings.llm_cache_hit_cost_per_million_cny
            + cache_miss_input_tokens * settings.llm_cache_miss_cost_per_million_cny
            + output_tokens * settings.llm_output_cost_per_million_cny
        ) / 1_000_000,
        8,
    )


class AgenticRagService:
    def __init__(self, settings: Settings, agents: RagAgents, vectors: VectorStore):
        self.settings = settings
        self.agents = agents
        self.vectors = vectors

    @trace_agent(name="agentic_rag_coordinator", agent_type="agentic_rag")
    async def answer(
        self,
        *,
        query: str,
        category: str,
        session_id: str,
        request_id: str | None = None,
        event_sink: TraceEventSink | None = None,
        evaluation_sink: EvaluationEventSink | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        trace_id = request_id or uuid.uuid4().hex
        trace = TraceRecorder(trace_id, event_sink=event_sink)
        retrieval_rounds = 0
        retrieved_count = 0
        rewritten = False
        accepted: list[dict[str, Any]] = []

        try:
            async with trace.stage(
                "query_planning",
                label="查询规划 Agent",
                event_type="agent",
                progress=8,
            ) as details:
                try:
                    subqueries = await self.agents.plan(query, category, trace)
                except Exception:
                    subqueries = [query]
                    details["status"] = "fallback"
                    details["reason_code"] = "planner_failed_use_original_query"
                details["subquery_count"] = len(subqueries)
                details["subqueries"] = subqueries
                details["decision_summary"] = (
                    "问题较集中，使用一个检索问题" if len(subqueries) == 1
                    else "问题包含多个规则点，拆分为多个检索问题"
                )
                details["selected_tool"] = "BGE-M3 Embedding + Qdrant 向量检索"
                details["next_action"] = "生成查询向量并检索当前知识分类"

            for round_index in range(self.settings.max_rounds):
                retrieval_rounds = round_index + 1
                round_label = "第一轮" if retrieval_rounds == 1 else "第二轮"
                async with trace.stage(
                    "retrieval",
                    label=f"{round_label}向量检索",
                    event_type="tool",
                    progress=20 if retrieval_rounds == 1 else 62,
                    round=retrieval_rounds,
                ) as details:
                    details["queries"] = subqueries
                    details["category"] = category
                    details["selected_tool"] = "BGE-M3 Embedding + Qdrant"
                    result_sets = await asyncio.gather(
                        *(self.vectors.search(item, category) for item in subqueries)
                    )
                    fused = reciprocal_rank_fusion(result_sets)
                    fused = fused[: self.settings.retrieve_top_k]
                    retrieved_count = max(retrieved_count, len(fused))
                    details["query_count"] = len(subqueries)
                    details["retrieved_chunks"] = len(fused)
                    details["top_score"] = round(float(fused[0].get("score", 0)), 4) if fused else 0
                    details["decision_summary"] = (
                        f"检索到 {len(fused)} 个候选片段，交给证据评估 Agent 判断"
                        if fused else "没有达到检索阈值的候选片段"
                    )
                    details["next_action"] = (
                        "评估候选证据是否足以回答" if fused
                        else ("恢复原问题后再检索一次" if round_index + 1 < self.settings.max_rounds else "停止检索并返回未找到可靠信息")
                    )
                    await emit_evaluation_event(
                        evaluation_sink,
                        {
                            "stage": "retrieval",
                            "round": retrieval_rounds,
                            "queries": list(subqueries),
                            "query_results": [
                                {
                                    "query": subquery,
                                    "hits": [
                                        ranked_hit(hit, rank)
                                        for rank, hit in enumerate(hits, start=1)
                                    ],
                                }
                                for subquery, hits in zip(subqueries, result_sets)
                            ],
                            "fused_hits": [
                                ranked_hit(hit, rank)
                                for rank, hit in enumerate(fused, start=1)
                            ],
                        },
                    )

                if not fused:
                    if round_index + 1 >= self.settings.max_rounds:
                        break
                    subqueries = [query]
                    continue

                async with trace.stage(
                    "evidence_grading",
                    label="证据评估 Agent",
                    event_type="agent",
                    progress=38 if retrieval_rounds == 1 else 72,
                    round=retrieval_rounds,
                ) as details:
                    try:
                        grade = await self.agents.grade(query, fused, trace)
                    except Exception:
                        deterministic = [
                            index + 1
                            for index, hit in enumerate(fused)
                            if float(hit.get("score", 0)) >= self.settings.accept_threshold
                        ][: self.settings.context_top_k]
                        grade = {
                            "sufficient": bool(deterministic),
                            "accepted_indices": deterministic,
                            "missing_aspects": [],
                            "decision_summary": "LLM 评估失败，改用相似度阈值进行保守判断",
                        }
                        details["status"] = "fallback"
                        details["reason_code"] = "grader_failed_use_score_threshold"
                    round_accepted = [
                        fused[index - 1]
                        for index in grade["accepted_indices"][: self.settings.context_top_k]
                    ]
                    details["accepted_chunks"] = len(round_accepted)
                    details["sufficient"] = grade["sufficient"]
                    details["accepted_titles"] = [
                        str(item.get("title", "未命名片段")) for item in round_accepted
                    ]
                    details["missing_aspects"] = grade["missing_aspects"]
                    details["decision_summary"] = grade.get("decision_summary") or (
                        "证据达到发布要求" if grade["sufficient"]
                        else "候选证据不能直接、完整支持回答"
                    )
                    details["next_action"] = (
                        "使用已接受证据生成回答"
                        if grade["sufficient"] and round_accepted
                        else (
                            "调用查询改写 Agent，再次检索 BGE-M3 与 Qdrant"
                            if round_index + 1 < self.settings.max_rounds
                            else "达到最大检索轮次，停止生成并返回未找到可靠信息"
                        )
                    )
                    accepted_uids = {
                        ranked_hit(hit, 0)["chunk_uid"] for hit in round_accepted
                    }
                    await emit_evaluation_event(
                        evaluation_sink,
                        {
                            "stage": "evidence_grading",
                            "round": retrieval_rounds,
                            "sufficient": bool(grade["sufficient"]),
                            "missing_aspects": list(grade["missing_aspects"]),
                            "accepted_hits": [
                                ranked_hit(hit, rank, accepted=True)
                                for rank, hit in enumerate(round_accepted, start=1)
                            ],
                            "fused_hits": [
                                ranked_hit(
                                    hit,
                                    rank,
                                    accepted=(ranked_hit(hit, rank)["chunk_uid"] in accepted_uids),
                                )
                                for rank, hit in enumerate(fused, start=1)
                            ],
                        },
                    )

                if grade["sufficient"] and round_accepted:
                    accepted = round_accepted
                    break
                if round_index + 1 >= self.settings.max_rounds:
                    break
                async with trace.stage(
                    "query_rewrite",
                    label="查询改写 Agent",
                    event_type="agent",
                    progress=52,
                ) as details:
                    rewritten = True
                    original_subqueries = list(subqueries)
                    try:
                        subqueries = await self.agents.rewrite(
                            query, category, grade["missing_aspects"], trace
                        )
                    except Exception:
                        subqueries = [query]
                        details["status"] = "fallback"
                        details["reason_code"] = "rewriter_failed_use_original_query"
                    details["subquery_count"] = len(subqueries)
                    details["original_queries"] = original_subqueries
                    details["missing_aspects"] = grade["missing_aspects"]
                    details["rewritten_queries"] = subqueries
                    details["decision_summary"] = "围绕上一轮缺失信息生成更具体的检索表达"
                    details["selected_tool"] = "BGE-M3 Embedding + Qdrant 向量检索"
                    details["next_action"] = "使用改写后的问题执行第二轮检索"

            if not accepted:
                return await self._finalize(
                    trace=trace,
                    started=started,
                    found=False,
                    answer=NOT_FOUND_MESSAGE,
                    sources=[],
                    retrieval_rounds=retrieval_rounds,
                    retrieved_count=retrieved_count,
                    accepted_count=0,
                    rewritten=rewritten,
                    verified=False,
                    session_id=session_id,
                )

            async with trace.stage(
                "answer_generation",
                label="回答生成 LLM",
                event_type="llm",
                progress=82,
            ) as details:
                details["evidence_count"] = len(accepted)
                details["evidence_titles"] = [str(item.get("title", "")) for item in accepted]
                details["decision_summary"] = "仅使用通过证据评估的片段生成带引用回答"
                details["next_action"] = "交给回答校验 Agent 检查事实支持与引用"
                answer = await self.agents.generate(query, accepted, trace)

            verified = False
            async with trace.stage(
                "answer_verification",
                label="回答校验 Agent",
                event_type="agent",
                progress=92,
            ) as details:
                verification = await self.agents.verify(query, answer, accepted, trace)
                details["verdict"] = verification["verdict"]
                details["decision_summary"] = (
                    "回答通过证据支持与引用检查" if verification["verdict"] == "pass"
                    else verification["revision_instruction"] or "回答未通过发布检查"
                )
                details["next_action"] = {
                    "pass": "发布回答",
                    "revise": "按照校验意见修正一次",
                    "insufficient": "停止发布并返回未找到可靠信息",
                }[verification["verdict"]]
            if verification["verdict"] == "pass":
                verified = True
            elif verification["verdict"] == "revise":
                async with trace.stage(
                    "answer_revision",
                    label="回答修正 LLM",
                    event_type="llm",
                    progress=95,
                ) as details:
                    details["revision_instruction"] = verification["revision_instruction"]
                    details["decision_summary"] = "根据发布前校验意见进行一次受限修正"
                    details["next_action"] = "重新执行回答校验"
                    answer = await self.agents.generate(
                        query,
                        accepted,
                        trace,
                        revision_instruction=verification["revision_instruction"],
                        previous_answer=answer,
                    )
                async with trace.stage(
                    "answer_reverification",
                    label="回答复核 Agent",
                    event_type="agent",
                    progress=98,
                ) as details:
                    second = await self.agents.verify(query, answer, accepted, trace)
                    details["verdict"] = second["verdict"]
                    details["decision_summary"] = (
                        "修正后的回答通过复核" if second["verdict"] == "pass"
                        else second["revision_instruction"] or "修正后仍未通过复核"
                    )
                    details["next_action"] = (
                        "发布回答" if second["verdict"] == "pass"
                        else "停止发布并返回未找到可靠信息"
                    )
                verified = second["verdict"] == "pass"

            if not verified:
                answer = NOT_FOUND_MESSAGE
                accepted = []

            sources = self._sources(accepted)
            return await self._finalize(
                trace=trace,
                started=started,
                found=verified,
                answer=answer,
                sources=sources,
                retrieval_rounds=retrieval_rounds,
                retrieved_count=retrieved_count,
                accepted_count=len(accepted),
                rewritten=rewritten,
                verified=verified,
                session_id=session_id,
            )
        except Exception:
            await trace_store.put(trace_id, {
                "trace_id": trace_id,
                "session_id": session_id,
                "events": trace.events,
                "input_tokens": trace.input_tokens,
                "output_tokens": trace.output_tokens,
                "cache_hit_input_tokens": trace.cache_hit_input_tokens,
                "cache_miss_input_tokens": trace.cache_miss_input_tokens,
                "status": "failed",
            })
            raise

    async def _finalize(
        self,
        *,
        trace: TraceRecorder,
        started: float,
        found: bool,
        answer: str,
        sources: list[dict[str, Any]],
        retrieval_rounds: int,
        retrieved_count: int,
        accepted_count: int,
        rewritten: bool,
        verified: bool,
        session_id: str,
    ) -> dict[str, Any]:
        latency_ms = int((time.perf_counter() - started) * 1000)
        meta = {
            "retrieval_rounds": retrieval_rounds,
            "max_retrieval_rounds": self.settings.max_rounds,
            "retrieved_chunks": retrieved_count,
            "accepted_chunks": accepted_count,
            "rewritten": rewritten,
            "verified": verified,
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
            "cost_configured": True,
            "pricing_model": self.settings.llm_model,
            "pricing_currency": "CNY",
        }
        payload = {
            "found": found,
            "answer": answer,
            "sources": sources,
            "trace_id": trace.trace_id,
            "meta": meta,
            "trace": trace.events,
        }
        await trace_store.put(trace.trace_id, {
            **payload,
            "session_id": session_id,
            "status": "completed",
        })
        return payload

    @staticmethod
    def _sources(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sources = []
        seen = set()
        for hit in evidence:
            key = (hit.get("document_id"), hit.get("title"), hit.get("source_url"))
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "document_id": hit.get("document_id"),
                "title": hit.get("title"),
                "source_name": hit.get("source_name"),
                "source_url": hit.get("source_url"),
                "updated_at": str(hit.get("updated_at", "")),
                "score": round(float(hit.get("score", 0)), 4),
            })
        return sources
