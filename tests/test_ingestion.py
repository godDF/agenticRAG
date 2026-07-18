import asyncio

from app.ingestion import IngestionJobStore, IngestionReporter


def test_ingestion_job_records_progress_metrics_and_result():
    async def scenario():
        store = IngestionJobStore()
        job = await store.create("rules.md", 128)
        await store.start(job["job_id"])
        reporter = IngestionReporter(store, job["job_id"])
        event = await reporter.start(
            kind="tool",
            name="embedding",
            label="生成向量",
            progress=50,
        )
        await reporter.finish(
            event,
            progress=90,
            metrics={
                "embedding_tokens": 32,
                "estimated_cost_cny": 0.0,
                "cost_configured": True,
                "pricing_free": True,
                "pricing_source": "SiliconFlow 官方价格",
                "pricing_model": "BAAI/bge-m3",
            },
        )
        await store.complete(
            job["job_id"],
            {
                "document_id": "d1",
                "title": "规则",
                "category": "student_ticket",
                "source_name": "official",
                "source_url": "",
                "updated_at": "2026-07-16",
                "original_filename": "rules.md",
                "normalized_path": "kb/student_ticket/d1.md",
                "content_hash": "hash",
                "chunk_count": 2,
                "status": "indexed",
                "created_at": "2026-07-16T00:00:00",
            },
        )
        return await store.get(job["job_id"])

    result = asyncio.run(scenario())
    assert result["status"] == "completed"
    assert result["progress"] == 100
    assert result["metrics"]["embedding_tokens"] == 32
    assert result["metrics"]["pricing_free"] is True
    assert result["metrics"]["pricing_source"] == "SiliconFlow 官方价格"
    assert result["metrics"]["tool_calls"] == 1
    assert result["events"][0]["status"] == "completed"


def test_ingestion_failure_closes_all_running_events():
    async def scenario():
        store = IngestionJobStore()
        job = await store.create("broken.pdf", 32)
        await store.start(job["job_id"])
        reporter = IngestionReporter(store, job["job_id"])
        await reporter.start(kind="agent", name="coordinator", label="协调", progress=5)
        await reporter.start(kind="tool", name="parser", label="解析", progress=10)
        await store.fail(job["job_id"], ValueError("无法解析"))
        return await store.get(job["job_id"])

    result = asyncio.run(scenario())
    assert result["status"] == "failed"
    assert result["metrics"]["error_count"] == 1
    assert all(event["status"] == "failed" for event in result["events"])
