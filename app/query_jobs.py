from __future__ import annotations

import asyncio
import copy
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Any


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


class QueryJobStore:
    """In-memory store for short-lived Agentic RAG query jobs and live traces."""

    def __init__(self, max_items: int = 200):
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._tasks: set[asyncio.Task] = set()
        self._max_items = max_items
        self._lock = asyncio.Lock()

    async def create(self) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        payload = {
            "job_id": job_id,
            "status": "queued",
            "progress": 2,
            "current_stage": "等待 Agentic RAG 处理",
            "created_at": _now(),
            "started_at": None,
            "completed_at": None,
            "events": [],
            "result": None,
            "error": None,
        }
        async with self._lock:
            self._items[job_id] = payload
            self._items.move_to_end(job_id)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)
        return copy.deepcopy(payload)

    async def get(self, job_id: str) -> dict[str, Any] | None:
        async with self._lock:
            item = self._items.get(job_id)
            if item is None:
                return None
            self._items.move_to_end(job_id)
            return copy.deepcopy(item)

    async def start(self, job_id: str) -> None:
        async with self._lock:
            item = self._items[job_id]
            item["status"] = "running"
            item["started_at"] = _now()
            item["current_stage"] = "启动 Agentic RAG 协调器"
            item["progress"] = max(item["progress"], 5)

    async def publish(self, job_id: str, event: dict[str, Any]) -> None:
        """Upsert a running/completed event emitted by TraceRecorder."""
        async with self._lock:
            item = self._items[job_id]
            event_copy = copy.deepcopy(event)
            event_id = event_copy["event_id"]
            existing_index = next(
                (
                    index
                    for index, current in enumerate(item["events"])
                    if current.get("event_id") == event_id
                ),
                None,
            )
            if existing_index is None:
                item["events"].append(event_copy)
            else:
                item["events"][existing_index] = event_copy
            item["progress"] = max(item["progress"], int(event_copy.get("progress", 0)))
            item["current_stage"] = str(event_copy.get("label") or event_copy.get("stage"))

    async def complete(self, job_id: str, result: dict[str, Any]) -> None:
        async with self._lock:
            item = self._items[job_id]
            item["status"] = "completed"
            item["progress"] = 100
            item["current_stage"] = "Agentic RAG 查询完成"
            item["completed_at"] = _now()
            item["result"] = copy.deepcopy(result)

    async def fail(self, job_id: str, *, error_type: str, message: str) -> None:
        async with self._lock:
            item = self._items[job_id]
            item["status"] = "failed"
            item["current_stage"] = "Agentic RAG 查询失败"
            item["completed_at"] = _now()
            item["error"] = {
                "type": error_type[:100],
                "message": message[:500],
            }
            for event in item["events"]:
                if event.get("status") == "running":
                    event["status"] = "failed"
                    event["completed_at"] = _now()
                    event.setdefault("details", {})["error_type"] = error_type[:100]

    def schedule(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task


query_jobs = QueryJobStore()
