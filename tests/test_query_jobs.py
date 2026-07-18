import asyncio

from app import main as main_module
from app.local_trace import TraceRecorder
from app.query_jobs import QueryJobStore
from app.schemas import QueryRequest


def test_trace_recorder_emits_running_and_completed_events():
    async def scenario():
        emitted = []

        async def sink(event):
            emitted.append(event)

        trace = TraceRecorder("trace-1", event_sink=sink)
        async with trace.stage(
            "query_planning",
            label="查询规划 Agent",
            event_type="agent",
            progress=8,
        ) as details:
            details["decision_summary"] = "拆分为两个检索问题"
        return emitted, trace.events

    emitted, final_events = asyncio.run(scenario())
    assert [event["status"] for event in emitted] == ["running", "completed"]
    assert emitted[0]["event_id"] == emitted[1]["event_id"]
    assert emitted[1]["event_type"] == "agent"
    assert emitted[1]["progress"] == 8
    assert emitted[1]["details"]["decision_summary"] == "拆分为两个检索问题"
    assert final_events == [emitted[1]]


def test_query_job_store_upserts_live_event_and_keeps_result():
    async def scenario():
        store = QueryJobStore()
        job = await store.create()
        await store.start(job["job_id"])
        running = {
            "event_id": "event-1",
            "event_type": "tool",
            "stage": "retrieval",
            "label": "第1轮向量检索",
            "status": "running",
            "progress": 20,
            "started_at": "2026-07-16T00:00:00",
            "completed_at": None,
            "latency_ms": 0,
            "details": {"round": 1},
        }
        await store.publish(job["job_id"], running)
        completed = {
            **running,
            "status": "completed",
            "completed_at": "2026-07-16T00:00:01",
            "latency_ms": 1000,
            "details": {"round": 1, "retrieved_chunks": 4, "top_score": 0.72},
        }
        await store.publish(job["job_id"], completed)
        await store.complete(job["job_id"], {"found": True, "answer": "ok"})
        return await store.get(job["job_id"])

    result = asyncio.run(scenario())
    assert result["status"] == "completed"
    assert result["progress"] == 100
    assert len(result["events"]) == 1
    assert result["events"][0]["status"] == "completed"
    assert result["events"][0]["details"]["retrieved_chunks"] == 4
    assert result["result"]["answer"] == "ok"


def test_query_job_failure_closes_running_event_without_raw_exception():
    async def scenario():
        store = QueryJobStore()
        job = await store.create()
        await store.start(job["job_id"])
        await store.publish(job["job_id"], {
            "event_id": "event-2",
            "event_type": "llm",
            "stage": "answer_generation",
            "label": "回答生成 LLM",
            "status": "running",
            "progress": 82,
            "started_at": "2026-07-16T00:00:00",
            "completed_at": None,
            "latency_ms": 0,
            "details": {},
        })
        await store.fail(
            job["job_id"],
            error_type="query_timeout",
            message="Agentic RAG 查询超过 20 秒",
        )
        return await store.get(job["job_id"])

    result = asyncio.run(scenario())
    assert result["status"] == "failed"
    assert result["events"][0]["status"] == "failed"
    assert result["error"]["type"] == "query_timeout"
    assert "secret" not in result["error"]["message"]


def test_background_query_job_streams_trace_and_completes(monkeypatch):
    class FakeService:
        async def answer(self, *, request_id, event_sink, **kwargs):
            running = {
                "event_id": "event-live",
                "event_type": "agent",
                "stage": "query_planning",
                "label": "查询规划 Agent",
                "status": "running",
                "progress": 8,
                "started_at": "2026-07-16T00:00:00",
                "completed_at": None,
                "latency_ms": 0,
                "details": {},
            }
            await event_sink(running)
            await event_sink({
                **running,
                "status": "completed",
                "completed_at": "2026-07-16T00:00:01",
                "latency_ms": 1000,
                "details": {"subquery_count": 2},
            })
            return {
                "found": True,
                "answer": "回答",
                "sources": [],
                "trace_id": request_id,
                "meta": {},
                "trace": [],
            }

    async def scenario():
        store = QueryJobStore()
        monkeypatch.setattr(main_module, "query_jobs", store)
        monkeypatch.setattr(main_module, "rag_service", FakeService())
        job = await store.create()
        request = QueryRequest(
            session_id="session-1",
            query="学生票规则",
            category="student_ticket",
        )
        await main_module._run_query_job(job["job_id"], request)
        return await store.get(job["job_id"])

    result = asyncio.run(scenario())
    assert result["status"] == "completed"
    assert result["events"][0]["status"] == "completed"
    assert result["events"][0]["details"]["subquery_count"] == 2
    assert result["result"]["answer"] == "回答"
