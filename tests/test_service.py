import asyncio
from dataclasses import replace

from app.config import settings
from app.service import AgenticRagService, NOT_FOUND_MESSAGE


HIT = {
    "point_id": "p1",
    "score": 0.86,
    "content": "学生优惠票仅在规定区间和时间内使用。",
    "document_id": "d1",
    "title": "学生票规则",
    "source_name": "官方来源",
    "source_url": "https://example.com/rule",
    "updated_at": "2026-07-16",
}


class FakeVectors:
    async def search(self, query, category):
        return [dict(HIT)]


class PassingAgents:
    async def plan(self, query, category, trace):
        return [query]

    async def grade(self, query, hits, trace):
        return {"sufficient": True, "accepted_indices": [1], "missing_aspects": []}

    async def rewrite(self, query, category, missing_aspects, trace):
        raise AssertionError("充分证据不应改写")

    async def generate(self, query, evidence, trace, **kwargs):
        return "学生票应按规则使用。[1] 具体规则可能变化，请以最新官方规定为准。"

    async def verify(self, query, answer, evidence, trace):
        return {"verdict": "pass", "revision_instruction": ""}


class InsufficientAgents(PassingAgents):
    async def grade(self, query, hits, trace):
        return {"sufficient": False, "accepted_indices": [1], "missing_aspects": ["适用时间"]}

    async def rewrite(self, query, category, missing_aspects, trace):
        return [query + " 适用时间"]

    async def generate(self, query, evidence, trace, **kwargs):
        raise AssertionError("证据不足时不得生成答案")


def test_agentic_service_returns_verified_answer():
    config = replace(settings, max_rounds=2)
    service = AgenticRagService(config, PassingAgents(), FakeVectors())
    live_events = []
    result = asyncio.run(
        service.answer(
            query="学生票怎么用？",
            category="student_ticket",
            session_id="s1",
            event_sink=live_events.append,
        )
    )
    assert result["found"] is True
    assert result["meta"]["verified"] is True
    assert result["meta"]["max_retrieval_rounds"] == 2
    assert result["sources"][0]["document_id"] == "d1"
    assert result["trace"][0]["label"] == "查询规划 Agent"
    assert result["trace"][0]["details"]["selected_tool"] == "BGE-M3 Embedding + Qdrant 向量检索"
    grading = next(event for event in result["trace"] if event["stage"] == "evidence_grading")
    assert grading["details"]["next_action"] == "使用已接受证据生成回答"
    completed = [event for event in live_events if event["status"] == "completed"]
    assert [event["stage"] for event in completed] == [
        "query_planning",
        "retrieval",
        "evidence_grading",
        "answer_generation",
        "answer_verification",
    ]
    assert next(event for event in completed if event["stage"] == "retrieval")["event_type"] == "tool"
    assert next(event for event in completed if event["stage"] == "answer_generation")["event_type"] == "llm"


def test_agentic_service_does_not_publish_insufficient_evidence():
    config = replace(settings, max_rounds=2)
    service = AgenticRagService(config, InsufficientAgents(), FakeVectors())
    result = asyncio.run(
        service.answer(
            query="学生票怎么用？",
            category="student_ticket",
            session_id="s2",
        )
    )
    assert result["found"] is False
    assert result["answer"] == NOT_FOUND_MESSAGE
    assert result["meta"]["retrieval_rounds"] == 2
    assert result["meta"]["max_retrieval_rounds"] == 2
