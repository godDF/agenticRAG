from __future__ import annotations

import asyncio
import copy
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Any


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


class IngestionJobStore:
    """In-memory progress store for short-lived upload/indexing jobs."""

    def __init__(self, max_items: int = 200):
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._tasks: set[asyncio.Task] = set()
        self._max_items = max_items
        self._lock = asyncio.Lock()

    async def create(self, filename: str, size_bytes: int) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        payload = {
            "job_id": job_id,
            "status": "queued",
            "progress": 5,
            "current_stage": "文件已到达服务端，等待处理",
            "filename": filename,
            "size_bytes": size_bytes,
            "created_at": _now(),
            "started_at": None,
            "completed_at": None,
            "events": [],
            "metrics": {
                "llm_input_tokens": 0,
                "llm_output_tokens": 0,
                "embedding_tokens": 0,
                "token_total": 0,
                "llm_calls": 0,
                "agent_calls": 0,
                "tool_calls": 0,
                "elapsed_ms": 0,
                "estimated_cost_cny": 0.0,
                "cost_configured": False,
                "pricing_free": False,
                "pricing_source": "",
                "pricing_model": "",
                "error_count": 0,
            },
            "result": None,
            "error": None,
        }
        async with self._lock:
            self._items[job_id] = payload
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)
        return payload

    async def get(self, job_id: str) -> dict[str, Any] | None:
        async with self._lock:
            item = self._items.get(job_id)
            if not item:
                return None
            result = copy.deepcopy(item)
            result.pop("_started_perf", None)
            self._clean_private(result)
            return result

    async def start(self, job_id: str) -> None:
        async with self._lock:
            item = self._items[job_id]
            item["status"] = "processing"
            item["started_at"] = _now()
            item["_started_perf"] = time.perf_counter()

    async def start_event(
        self,
        job_id: str,
        *,
        kind: str,
        name: str,
        label: str,
        progress: int,
        details: dict[str, Any] | None = None,
    ) -> str:
        event_id = uuid.uuid4().hex[:12]
        async with self._lock:
            item = self._items[job_id]
            event = {
                "event_id": event_id,
                "kind": kind,
                "name": name,
                "label": label,
                "status": "running",
                "progress": progress,
                "started_at": _now(),
                "completed_at": None,
                "latency_ms": None,
                "details": details or {},
                "error": None,
                "_started_perf": time.perf_counter(),
            }
            item["events"].append(event)
            item["progress"] = progress
            item["current_stage"] = label
            if kind == "agent":
                item["metrics"]["agent_calls"] += 1
            elif kind == "llm":
                item["metrics"]["llm_calls"] += 1
            elif kind == "tool":
                item["metrics"]["tool_calls"] += 1
        return event_id

    async def finish_event(
        self,
        job_id: str,
        event_id: str,
        *,
        progress: int,
        details: dict[str, Any] | None = None,
        metrics: dict[str, int | float | bool | str] | None = None,
    ) -> None:
        async with self._lock:
            item = self._items[job_id]
            event = next(event for event in item["events"] if event["event_id"] == event_id)
            event["status"] = "completed"
            event["completed_at"] = _now()
            event["latency_ms"] = int((time.perf_counter() - event.pop("_started_perf")) * 1000)
            if details:
                event["details"].update(details)
            if metrics:
                for key, value in metrics.items():
                    if key in {"cost_configured", "pricing_free"}:
                        item["metrics"][key] = bool(value)
                    elif isinstance(value, str):
                        item["metrics"][key] = value
                    else:
                        item["metrics"][key] = item["metrics"].get(key, 0) + value
                item["metrics"]["token_total"] = (
                    item["metrics"]["llm_input_tokens"]
                    + item["metrics"]["llm_output_tokens"]
                    + item["metrics"]["embedding_tokens"]
                )
            item["progress"] = progress
            item["current_stage"] = event["label"]

    async def complete(self, job_id: str, result: dict[str, Any]) -> None:
        async with self._lock:
            item = self._items[job_id]
            item["status"] = "completed"
            item["progress"] = 100
            item["current_stage"] = "知识库处理完成"
            item["completed_at"] = _now()
            item["result"] = result
            started = item.pop("_started_perf", None)
            if started is not None:
                item["metrics"]["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
            self._clean_private(item)

    async def fail(self, job_id: str, exc: Exception) -> None:
        async with self._lock:
            item = self._items[job_id]
            item["status"] = "failed"
            item["current_stage"] = "处理失败"
            item["completed_at"] = _now()
            item["error"] = {
                "type": type(exc).__name__,
                "message": str(exc)[:1000] or "未知错误",
            }
            item["metrics"]["error_count"] += 1
            started = item.pop("_started_perf", None)
            if started is not None:
                item["metrics"]["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
            for event in reversed(item["events"]):
                if event["status"] == "running":
                    event["status"] = "failed"
                    event["completed_at"] = _now()
                    event["latency_ms"] = int(
                        (time.perf_counter() - event.pop("_started_perf")) * 1000
                    )
                    event["error"] = item["error"]
            self._clean_private(item)

    def schedule(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    @staticmethod
    def _clean_private(item: dict[str, Any]) -> None:
        for event in item["events"]:
            event.pop("_started_perf", None)


class IngestionReporter:
    def __init__(self, store: IngestionJobStore, job_id: str):
        self.store = store
        self.job_id = job_id

    async def start(self, **kwargs) -> str:
        return await self.store.start_event(self.job_id, **kwargs)

    async def finish(self, event_id: str, **kwargs) -> None:
        await self.store.finish_event(self.job_id, event_id, **kwargs)


ingestion_jobs = IngestionJobStore()
