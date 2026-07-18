from __future__ import annotations

import asyncio
import copy
import inspect
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Awaitable, Callable


TraceEventSink = Callable[[dict[str, Any]], Awaitable[None] | None]


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


class TraceRecorder:
    def __init__(self, trace_id: str, event_sink: TraceEventSink | None = None):
        self.trace_id = trace_id
        self.event_sink = event_sink
        self.events: list[dict[str, Any]] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_hit_input_tokens = 0
        self.cache_miss_input_tokens = 0
        self.cache_usage_reported = True

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        try:
            result = self.event_sink(copy.deepcopy(event))
            if inspect.isawaitable(result):
                await result
        except Exception:
            # Observability must never make the actual RAG query fail.
            return

    @asynccontextmanager
    async def stage(
        self,
        name: str,
        label: str | None = None,
        *,
        event_type: str = "agent",
        progress: int = 0,
        **initial: Any,
    ):
        started = time.perf_counter()
        started_at = _now()
        event_id = uuid.uuid4().hex[:16]
        details = dict(initial)
        running_event = {
            "event_id": event_id,
            "event_type": event_type,
            "stage": name,
            "label": label or name,
            "status": "running",
            "progress": max(0, min(99, int(progress))),
            "started_at": started_at,
            "completed_at": None,
            "latency_ms": 0,
            "details": copy.deepcopy(details),
        }
        await self._emit(running_event)
        try:
            yield details
        except Exception as exc:
            event = {
                **running_event,
                "stage": name,
                "label": label or name,
                "status": "failed",
                "completed_at": _now(),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "details": {**details, "error_type": type(exc).__name__},
            }
            self.events.append(event)
            await self._emit(event)
            raise
        else:
            event = {
                **running_event,
                "stage": name,
                "label": label or name,
                "status": details.pop("status", "completed"),
                "completed_at": _now(),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "details": details,
            }
            self.events.append(event)
            await self._emit(event)

    def add_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_hit_input_tokens: int = 0,
        cache_miss_input_tokens: int | None = None,
        cache_usage_reported: bool = False,
    ) -> None:
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)
        cache_hit = min(max(0, cache_hit_input_tokens), max(0, input_tokens))
        cache_miss = (
            max(0, cache_miss_input_tokens)
            if cache_miss_input_tokens is not None
            else max(0, input_tokens - cache_hit)
        )
        if cache_hit + cache_miss != max(0, input_tokens):
            cache_miss = max(0, input_tokens - cache_hit)
        self.cache_hit_input_tokens += cache_hit
        self.cache_miss_input_tokens += cache_miss
        self.cache_usage_reported = self.cache_usage_reported and cache_usage_reported


class TraceStore:
    def __init__(self, max_items: int = 200):
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_items = max_items
        self._lock = asyncio.Lock()

    async def put(self, trace_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            self._items[trace_id] = payload
            self._items.move_to_end(trace_id)
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)

    async def get(self, trace_id: str) -> dict[str, Any] | None:
        async with self._lock:
            return self._items.get(trace_id)


trace_store = TraceStore()
